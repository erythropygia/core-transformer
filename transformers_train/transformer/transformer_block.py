import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
import math

_MESSAGES_PRINTED = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("Flash Attention available!")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("Flash Attention not available, using standard attention")

_MESSAGES_PRINTED = True


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)
        
        # Selective checkpointing
        self.use_selective_checkpointing = config.get('use_selective_checkpointing', False)
    
    def _attention_forward(self, x):
        return self.attn(self.ln1(x))
    
    def _mlp_forward(self, x):
        return self.mlp(self.ln2(x))
    
    def forward(self, x):
        # Selective checkpointing - sadece MLP'yi checkpoint yap
        if self.use_selective_checkpointing and self.training:
            # Attention normal, MLP checkpointed
            x = x + self._attention_forward(x)
            x = x + checkpoint.checkpoint(self._mlp_forward, x, use_reentrant=False)
        elif self.config.get('use_gradient_checkpointing', False) and self.training:
            # Full checkpointing
            def _forward_impl(x):
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
                return x
            x = checkpoint.checkpoint(_forward_impl, x, use_reentrant=False)
        else:
            # Normal forward
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        self.config = config
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = config['n_embd'] // config['n_head']
        self.use_flash = config.get('use_flash_attention', False) and FLASH_ATTENTION_AVAILABLE
        
        # Fused QKV projection
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        
        if not self.use_flash:
            self.attn_dropout = nn.Dropout(config['dropout'])
            # Causal mask
            self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                .view(1, 1, config['block_size'], config['block_size']))
        
        self.resid_dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        B, T, C = x.size()
        
        # QKV projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Flash Attention vs Standard Attention
        if self.use_flash and x.device.type == 'cuda':
            # Flash Attention - memory efficient
            q = q.transpose(1, 2).contiguous()  # (B, T, H, D)
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            y = flash_attn_func(
                q, k, v, 
                dropout_p=self.config['dropout'] if self.training else 0.0, 
                causal=True
            )
            y = y.view(B, T, C)
        else:
            # Standard attention with memory optimization
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            if hasattr(self, 'attn_dropout'):
                att = self.attn_dropout(att)
            
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=False)
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Embeddings
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.wpe = nn.Embedding(config['block_size'], config['n_embd'])
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config['n_embd'])

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['block_size'], f"Sequence length {t} exceeds block size {self.config['block_size']}"
        
        # Token ve position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.h:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        #Weight tying direkt embedding weight ile
        logits = F.linear(x, self.wte.weight)
        
        loss = None
        if targets is not None:
            # Label smoothing
            label_smoothing = self.config.get('label_smoothing', 0.0)
            if label_smoothing > 0.0:
                log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
                targets_flat = targets.view(-1)
                
                num_classes = logits.size(-1)
                smoothed_targets = torch.full_like(log_probs, label_smoothing / (num_classes - 1))
                smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), 1.0 - label_smoothing)
                
                valid_mask = (targets_flat != -1)
                if valid_mask.any():
                    loss = -(smoothed_targets[valid_mask] * log_probs[valid_mask]).sum(dim=-1).mean()
                else:
                    loss = torch.tensor(0.0, device=logits.device)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def get_num_params(self):
        # Weight tying nedeniyle lm_head sayılmaz, hesaplama hatasına sebep oluyor.
        n_params = sum(p.numel() for p in self.parameters())
        # Embedding weight'i iki kez sayıldığı için çıkar
        n_params -= self.wte.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9):
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            
            # Forward pass
            logits, _ = self(idx_cond)
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
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        context = torch.tensor(tokens, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)
        generated = self.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
