import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .kv_cache import KVCache
from ..common import get_dist_info, print0
# Import Flash Attention 3 with SDPA fallback
from .flash_attention import flash_attn


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config['n_head']
        self.n_kv_head = config.get('n_kv_head', config['n_head'])  # GQA support
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # Separate Q, K, V projections (bias-free)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size=(-1, 0), kv_cache=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache (FA3-native format)
            # Initialize cache on first use
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
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.num_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
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
        x = F.relu(x).square()  # ReLU² activation
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
        
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        
        # Pad vocab for efficiency (DDP, tensor cores) - optimization from nanochat
        padded_vocab_size = ((config['vocab_size'] + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config['vocab_size']:
            print0(f"Padding vocab_size from {config['vocab_size']} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size
        
        # Embeddings (untied with lm_head)
        self.wte = nn.Embedding(padded_vocab_size, config['n_embd'])
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config['n_layer'])])
        
        # Final layer norm (functional, no learnable params)
        # We'll use norm() function directly in forward
        
        # Untied lm_head (separate from embeddings)
        self.lm_head = nn.Linear(config['n_embd'], padded_vocab_size, bias=False)
        
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config['n_layer']))
        self.x0_lambdas = nn.Parameter(torch.zeros(config['n_layer']))
        
        # Rotary embeddings
        # Over-compute them for efficiency (10X should be enough)
        self.rotary_seq_len = config.get('block_size', 1024) * 10
        head_dim = config['n_embd'] // config['n_head']
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to checkpoint
        self.register_buffer("sin", sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections (zeros like nanochat)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        for block in self.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        
        # Initialize per-layer scalars (done after apply to ensure correct values)
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
            self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init
        
        # Cast embeddings to bfloat16 if on CUDA (nanochat does this for memory efficiency)
        if torch.cuda.is_available():
            # Keep cos/sin in bfloat16 (they're precomputed, no gradients)
            self.cos = self.cos.to(dtype=torch.bfloat16)
            self.sin = self.sin.to(dtype=torch.bfloat16)
            # Cast embeddings to bfloat16 (optimizer can tolerate it, saves memory)
            self.wte = self.wte.to(dtype=torch.bfloat16)
    
    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.get('window_pattern', 'L').upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        block_size = config.get('block_size', 1024)
        long_window = block_size
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config['n_layer']):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Nanochat uses uniform initialization with bound = sqrt(3) * std
            # This prevents outliers better than normal distribution
            n_embd = self.config['n_embd']
            s = 3**0.5 * n_embd**-0.5  # sqrt(3) makes uniform have same std as normal
            torch.nn.init.uniform_(module.weight, -s, s)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Nanochat uses std=1.0 for embeddings (much larger than our 0.02)
            # This is important for proper training dynamics
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # Autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
        # Stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # Stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # Calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims for broadcasting
        return cos, sin
    
    def get_device(self):
        return self.wte.weight.device
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config['n_embd']
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate out all parameters into 5 groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas)
        matrix_params = list(self.h.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
        
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01),  # more sensitive because they accumulate in residual stream
            dict(params=x0_params, lr=scalar_lr),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)  # NOTE: weight decay is 0.0 for AdamW, only used in Muon
        
        # Use DistAdamW if DDP is enabled, otherwise use standard AdamW
        if ddp:
            from ..training.adamw import DistAdamW
            adamw_optimizer = DistAdamW(adam_groups, **adamw_kwargs)
        else:
            adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True, **adamw_kwargs)
        
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        if ddp:
            from ..training.muon import DistMuon
            muon_optimizer = DistMuon(matrix_params, **muon_kwargs)
        else:
            from ..training.muon import Muon
            muon_optimizer = Muon(matrix_params, **muon_kwargs)
        
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        return optimizers
    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        
        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary embeddings cache {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # If kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # truncate cache to current sequence length
        
        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)  # norm after token embedding
        x0 = x  # save initial normalized embedding for x0 residual (skip connection)
        for i, block in enumerate(self.h):
            # Apply per-layer scalars: x = lambda_resid * x + lambda_x0 * x0
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)  # final norm
        
        # Forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., :self.config['vocab_size']]  # crop padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss
        logits = softcap * torch.tanh(logits / softcap)  # logits softcap
        
        if targets is not None:
            # Training mode: compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # Inference mode: return the logits
            return logits
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9, kv_cache=None):
        self.eval()
        
        # Get EOS token ID if tokenizer is available
        eos_token_id = None
        if self.tokenizer is not None:
            try:
                eos_token_id = self.tokenizer.get_eos_token_id()
            except:
                pass
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.get('block_size', 1024) else idx[:, -self.config.get('block_size', 1024):]
            
            # Forward pass
            logits = self(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if EOS token is generated
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        
        return idx
    
    def generate_from_prompt(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9, skip_special_tokens=True):
        self.eval()
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generate_from_prompt")
        
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if isinstance(tokens[0], list):  # Handle batch encoding
            tokens = tokens[0]
        
        device = self.get_device()
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = self.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=skip_special_tokens)
