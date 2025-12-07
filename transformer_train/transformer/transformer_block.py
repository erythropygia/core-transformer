import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .kv_cache import KVCache
from .common import get_dist_info, print0


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

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(2)  # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively
        enable_gqa = self.n_head != self.n_kv_head  # Group Query Attention
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)  # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
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

    def forward(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Embeddings (untied with lm_head)
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config['n_layer'])])
        
        # Final layer norm (functional, no learnable params)
        # We'll use norm() function directly in forward
        
        # Untied lm_head (separate from embeddings)
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Rotary embeddings
        # Over-compute them for efficiency (10X should be enough)
        self.rotary_seq_len = config.get('block_size', 1024) * 10
        head_dim = config['n_embd'] // config['n_head']
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False means it's not saved to checkpoint
        self.register_buffer("sin", sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        
        # Cast embeddings to bfloat16 if on CUDA (saves memory)
        if torch.cuda.is_available():
            self.wte = self.wte.to(dtype=torch.bfloat16)
            self.cos = self.cos.to(dtype=torch.bfloat16)
            self.sin = self.sin.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
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
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config['n_embd']
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.h.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        
        # Use DistAdamW if DDP is enabled, otherwise use standard AdamW
        if ddp:
            from .adamw import DistAdamW
            adamw_optimizer = DistAdamW(adam_groups, **adamw_kwargs)
        else:
            adamw_optimizer = torch.optim.AdamW(adam_groups, fused=True, **adamw_kwargs)
        
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        if ddp:
            from .muon import DistMuon
            muon_optimizer = DistMuon(matrix_params, **muon_kwargs)
        else:
            from .muon import Muon
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
        # If kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # truncate cache to current sequence length
        
        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)  # norm after token embedding
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)  # final norm
        
        # Forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # logits softcap
        
        if targets is not None:
            # Training mode: compute and return the loss
            logits = logits.float()  # use fp32 for logits in training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return logits, loss
        else:
            # Inference mode: return the logits
            return logits
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9, kv_cache=None):
        self.eval()
        
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
        
        return idx
    
    def generate_from_prompt(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9):
        self.eval()
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generate_from_prompt")
        
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if isinstance(tokens[0], list):  # Handle batch encoding
            tokens = tokens[0]
        
        device = self.get_device()
        context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = self.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
