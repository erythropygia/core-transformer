import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import get_dist_info, print0
from .flash_attention import flash_attn


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config['n_head']
        self.n_kv_head = config.get('n_kv_head', config['n_head'])
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size=(-1, 0), kv_cache=None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            if kv_cache.kv_cache is None:
                kv_cache.init_cache(k.dtype, k.device)

            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.num_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=False)
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config, tokenizer=None, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.window_sizes = self._compute_window_sizes(config)

        padded_vocab_size = ((config['vocab_size'] + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config['vocab_size']:
            print0(f"Padding vocab_size from {config['vocab_size']} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        self.wte = nn.Embedding(padded_vocab_size, config['n_embd'])

        self.h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config['n_layer'])])


        self.lm_head = nn.Linear(config['n_embd'], padded_vocab_size, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config['n_layer']))
        self.x0_lambdas = nn.Parameter(torch.zeros(config['n_layer']))

        self.rotary_seq_len = config.get('block_size', 1024) * 10
        head_dim = config['n_embd'] // config['n_head']
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.0)

        if torch.cuda.is_available():
            self.cos = self.cos.to(dtype=torch.bfloat16)
            self.sin = self.sin.to(dtype=torch.bfloat16)
            self.wte = self.wte.to(dtype=torch.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.get('window_pattern', 'L').upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        block_size = config.get('block_size', 1024)
        long_window = block_size
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config['n_layer']):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            n_embd = self.config['n_embd']
            s = 3**0.5 * n_embd**-0.5
            torch.nn.init.uniform_(module.weight, -s, s)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def get_device(self):
        return self.wte.weight.device

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.9, 0.95), scalar_lr=0.5):
        model_dim = self.config['n_embd']
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.h.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)

        if ddp:
            from ..training.adamw import DistAdamW
            adamw_optimizer = DistAdamW(adam_groups, **adamw_kwargs)
        else:
            adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True, **adamw_kwargs)

        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        if ddp:
            from ..training.muon import DistMuon
            muon_optimizer = DistMuon(matrix_params, **muon_kwargs)
        else:
            from ..training.muon import Muon
            muon_optimizer = Muon(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary embeddings cache {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config['vocab_size']]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9, kv_cache=None):
        self.eval()

        stop_ids = set()
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "get_stop_token_ids"):
                stop_ids = set(self.tokenizer.get_stop_token_ids())
            elif hasattr(self.tokenizer, "get_eos_token_id"):
                stop_ids = {self.tokenizer.get_eos_token_id()}
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.get('block_size', 1024) else idx[:, -self.config.get('block_size', 1024):]

            logits = self(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stop_ids:
                for sid in stop_ids:
                    finished |= idx_next.squeeze(-1) == sid
                if bool(finished.all()):
                    break

        return idx

    def generate_from_prompt(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9, skip_special_tokens=True):
        self.eval()
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generate_from_prompt")

        tokens = self.tokenizer.encode(prompt, prepend=self.tokenizer.get_bos_token_id())
        if isinstance(tokens[0], list):
            tokens = tokens[0]

        device = self.get_device()
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = self.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=skip_special_tokens)
