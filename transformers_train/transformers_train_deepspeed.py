import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as checkpoint
import json
import re
import random
import math
from tqdm import tqdm
from collections import Counter, defaultdict
from datasets import load_dataset
import unicodedata
from pathlib import Path
import wandb
from typing import Optional, Dict, Any, Tuple, Optional, List
import gc

import numpy as np

# Global flags to prevent spam messages
_MESSAGES_PRINTED = False

# DeepSpeed import for RTX 3050 optimization
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("DeepSpeed available - RTX 3050 memory optimization enabled")
except ImportError:
    DEEPSPEED_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("DeepSpeed not available, install with: pip install deepspeed")

# SafeTensors import for secure model saving
try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("SafeTensors available - using secure model format")
except ImportError:
    SAFETENSORS_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("SafeTensors not available, install with: pip install safetensors")

# SentencePiece import for Turkish tokenization
try:
    import sentencepiece as smp
    SENTENCEPIECE_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("SentencePiece available")
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("SentencePiece not available, install with: pip install sentencepiece")

# Flash Attention import (optional)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    if not _MESSAGES_PRINTED:
        print("Flash Attention available!")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("Flash Attention not available, using standard attention")

try:
    from nltk.translate.bleu_score import sentence_bleu
    import nltk
    nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("NLTK not available, BLEU scores will be skipped")

# Mark messages as printed
_MESSAGES_PRINTED = True

MODEL_CONFIG = {
    'n_embd': 768,          # 768 embedding dimension
    'n_layer': 12,          # 12 transformer layer
    'n_head': 12,           # 12 attention head
    'block_size': 2048,     # 2048 context window 
    'dropout': 0.1,         # Dropout
    'vocab_size': None,     # tokenizer'dan alınacak
    'use_flash_attention': True,   # Flash Attention aktif
    'use_gradient_checkpointing': True,  
    'use_selective_checkpointing': True, 
}

TRAINING_CONFIG = {
    'batch_size': 2,        # RTX 3050 için optimize (8->2)
    'learning_rate': 6e-4,  
    'weight_decay': 0.1,   
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'warmup_epochs': 3,     
    'max_epochs': 50,       
    'eval_interval': 2,     
    'save_interval': 5,     
    'accumulation_steps': 16,  # RTX 3050 için düşük (2 * 16 = 32 effective batch) 
    'use_wandb': True,
    'compile_model': False,
    'scheduler_type': 'cosine_with_warmup',
    'eval_generation_samples': 3,
    'max_eval_batches': 50,
    
    'use_cpu_offload': True,     
    'use_activation_checkpointing': True,
    'use_mixed_precision': True, 
    'dataloader_num_workers': 2, 
    'pin_memory': True,          # Memory transfer rate
    'prefetch_factor': 2,        # Prefetch optimization
    
    # Progress reporting
    'log_interval': 50,     
    'eval_steps': 1000,     
    'checkpoint_steps': 500,  
    
    # Regularization
    'early_stopping_patience': 8,  
    'early_stopping_min_delta': 0.005,
    'label_smoothing': 0.05,  
    'mixup_alpha': 0.1,      
    'use_cosine_restarts': False,
    
    'vocab_size': 32000,
    'max_data_samples': 150000, 
    
    # DeepSpeed RTX 3050 Optimization
    'use_deepspeed': False,          # DeepSpeed kullan (True/False)
    'deepspeed_config_path': 'transformers_train/config/simple_deepspeed.json',   # Config file path
    'zero_stage': 2,                 # ZeRO Stage 2 (RTX 3050 için optimal)
    'cpu_offload': True,             # CPU offload aktif
    'nvme_offload': False,           # NVMe offload (SSD gerekli)
    'allgather_bucket_size': 5e8,    # Memory optimization
    'reduce_bucket_size': 5e8,       # Memory optimization
}

TEST_PROMPTS = [
    "Türkiye'nin başkenti",
    "Yapay zeka teknolojisi",
    "İstanbul Boğazı"
]

def create_deepspeed_config(training_config, model_config):
    config = {
        "train_batch_size": training_config['batch_size'] * training_config['accumulation_steps'],
        "train_micro_batch_size_per_gpu": training_config['batch_size'],
        "gradient_accumulation_steps": training_config['accumulation_steps'],
        
        # Optimizer configuration
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config['learning_rate'],
                "betas": [training_config['beta1'], training_config['beta2']],
                "eps": 1e-8,
                "weight_decay": training_config['weight_decay']
            }
        },
        
        # Learning rate scheduler
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": training_config['learning_rate'],
                "warmup_num_steps": training_config['warmup_epochs'] * 1000  # Approximate
            }
        },
        
        # Mixed precision
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        
        # ZeRO Stage 2 (optimal memory vs speed balance)
        "zero_optimization": {
            "stage": training_config.get('zero_stage', 2),
            "allgather_partitions": True,
            "allgather_bucket_size": training_config.get('allgather_bucket_size', 5e8),
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": training_config.get('reduce_bucket_size', 5e8),
            "contiguous_gradients": True,
            "cpu_offload": training_config.get('cpu_offload', True)
        },
        
        # Activation checkpointing for memory savings
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        
        # Memory optimization for RTX 3050
        "memory_optimization": {
            "deepspeed_activation_checkpointing": True,
            "optimize_with_cpu_offload": True,
            "optimize_with_amp": True
        },
        
        # Gradient clipping
        "gradient_clipping": training_config['grad_clip'],
        
        # Checkpoint saving
        "steps_per_print": training_config['log_interval'],
        "wall_clock_breakdown": False,
        
        # Communication backend
        "comms_logger": {
            "enabled": False
        }
    }
    
    # ZeRO Stage 3 için ek ayarlar (çok aggressive memory saving)
    if training_config.get('zero_stage', 2) == 3:
        config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # NVMe offload (SSD gerekli)
    if training_config.get('nvme_offload', False):
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "nvme",
            "nvme_path": "/tmp/deepspeed_nvme",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": "/tmp/deepspeed_nvme",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        }
    
    return config

def create_tokenizer(model_path: str = None):
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("SentencePiece not available, install with: pip install sentencepiece")
    
    if not model_path:
        model_path = "turkish_tokenizer/turkish_tokenizer.model"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer file not found: {model_path}")
    
    # SentencePiece processor oluştur
    sp = smp.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"SentencePiece tokenizer loaded: {model_path}")
    print(f"Vocabulary size: {sp.vocab_size()}")
    
    # Wrapper class for compatibility
    class TokenizerWrapper:
        def __init__(self, sp_processor):
            self.sp = sp_processor
            
        def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
            if add_special_tokens:
                return [self.sp.bos_id()] + self.sp.encode_as_ids(text)
            else:
                return self.sp.encode_as_ids(text)
        
        def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
            if skip_special_tokens:
                filtered_ids = [
                    tid for tid in token_ids 
                    if tid not in [self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id()]
                ]
                return self.sp.decode_ids(filtered_ids)
            else:
                return self.sp.decode_ids(token_ids)
        
        def encode_as_pieces(self, text: str) -> List[str]:
            return self.sp.encode_as_pieces(text)
        
        @property
        def vocab_size(self) -> int:
            return self.sp.vocab_size()
        
        def get_vocab_size(self) -> int:
            return self.sp.vocab_size()
    
    return TokenizerWrapper(sp)

class Dataset(Dataset):
    def __init__(self, tokens, block_size, stride=None):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride or block_size // 2
        
        # Memory efficient: sadece start pozisyonlarını sakla
        self.sequences = []
        max_start = len(tokens) - block_size
        
        # Stride ile overlapping sequences oluştur
        for i in range(0, max_start, self.stride):
            if i + block_size < len(tokens):
                self.sequences.append(i)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        start_pos = self.sequences[idx]
        input_seq = self.tokens[start_pos:start_pos + self.block_size]
        target_seq = self.tokens[start_pos + 1:start_pos + self.block_size + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

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
        # Weight tying nedeniyle lm_head sayılmaz
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

# Memory Management Functions
def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # GPU işlemlerini bekle
    gc.collect()
    
def get_memory_usage_safe():
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU: {allocated:.2f}GB/{total:.1f}GB (Reserved: {reserved:.2f}GB)"
        except:
            return "GPU: Memory calculation error"
    return "CPU Mode"

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU: {allocated:.2f}GB/{reserved:.2f}GB"
    return "CPU Mode"

# Advanced LR Schedulers
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine learning rate scheduler with warmup"""
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_onecycle_scheduler(optimizer, max_lr, total_steps, div_factor=25, final_div_factor=10000):
    """OneCycle learning rate scheduler"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=0.3
    )

def get_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True):
    """Plateau-based learning rate scheduler"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose,
        min_lr=1e-6
    )

class EarlyStopping:    
    def __init__(self, patience=8, min_delta=0.005, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict({k: v.to(model.device) for k, v in self.best_weights.items()})
            return True
        return False

@torch.no_grad()
def calculate_perplexity(model, data_loader, device, device_type, max_batches=50):
    """Perplexity calculation - RTX 3050 için optimize"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            _, loss = model(inputs, targets)
        
        batch_tokens = targets.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1
        
        # Memory cleanup
        if batch_idx % 10 == 0:
            cleanup_memory()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    return perplexity

@torch.no_grad() 
def evaluate_generation_quality(model, tokenizer, test_prompts, device, max_new_tokens=30):
    """Generation quality evaluation - RTX 3050 için kısa"""
    model.eval()
    results = {
        'samples': [],
        'avg_length': 0,
        'unique_tokens_ratio': 0,
    }
    
    total_length = 0
    all_tokens = set()
    total_tokens = 0
    
    for prompt in test_prompts:
        try:
            generated = model.generate_from_prompt(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
            
            # Extract generated part
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            full_tokens = tokenizer.encode(generated, add_special_tokens=False)
            if len(full_tokens) > len(prompt_tokens):
                generated_tokens = full_tokens[len(prompt_tokens):]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                generated_text = generated
            
            results['samples'].append({
                'prompt': prompt,
                'generated': generated_text,
                'full_text': generated
            })
            
            # Statistics
            tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            total_length += len(tokens)
            all_tokens.update(tokens)
            total_tokens += len(tokens)
            
        except Exception as e:
            print(f"Generation error for '{prompt}': {e}")
            continue
    
    # Calculate metrics
    if results['samples']:
        results['avg_length'] = total_length / len(results['samples'])
        results['unique_tokens_ratio'] = len(all_tokens) / max(total_tokens, 1)
    
    return results

@torch.no_grad()
def evaluate_model_comprehensive(model, val_loader, tokenizer, device, device_type, config):
    """Comprehensive model evaluation - RTX 3050 optimized"""
    metrics = {}
    
    # 1. Standard loss evaluation
    total_loss = 0
    num_batches = 0
    max_batches = config.get('max_eval_batches', 50)
    
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            _, loss = model(inputs, targets)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Memory cleanup every 10 batches
        if batch_idx % 10 == 0:
            cleanup_memory()
    
    metrics['val_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # 2. Perplexity
    try:
        metrics['perplexity'] = calculate_perplexity(model, val_loader, device, device_type, max_batches=30)
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        metrics['perplexity'] = float('inf')
    
    # 3. Generation quality
    try:
        num_samples = config.get('eval_generation_samples', 3)
        gen_results = evaluate_generation_quality(
            model, tokenizer, TEST_PROMPTS[:num_samples], device
        )
        metrics['generation'] = gen_results
        metrics['avg_gen_length'] = gen_results['avg_length']
        metrics['unique_token_ratio'] = gen_results['unique_tokens_ratio']
    except Exception as e:
        print(f"Generation evaluation error: {e}")
        metrics['avg_gen_length'] = 0
        metrics['unique_token_ratio'] = 0
    
    return metrics

def train(
    tokenizer_path="turkish_tokenizer/turkish_tokenizer.model",
    resume_from_checkpoint=None,
    auto_resume=True,
    use_wandb=True,
    project_name="turkish-transformer-100m",
    pretrained_model_path=None,
    fresh_epochs=None
):
    
    print("Transformer train")
    print("="*80)
    
    # Initialize training state
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    # RTX 3050 CUDA Memory Optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = "cuda"
        print(f"Device: {device} ({torch.cuda.get_device_name()})")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        torch.cuda.empty_cache()
        cleanup_memory()
        print("Memory cleaned")
    else:
        device = torch.device('cpu')
        device_type = "cpu"
        print(f"CUDA not available, using CPU")
    
    # Memory info
    print(f"Memory usage at start: {get_memory_usage()}")
    
    # Features
    print(f"Flash Attention: {'OK' if FLASH_ATTENTION_AVAILABLE else 'NO'}")
    print(f"Gradient Checkpointing: OK")
    print(f"Mixed Precision: OK")
    print(f"CPU Offloading: OK")
    print(f"DeepSpeed: {'OK' if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed'] else 'NO'}")
    
    # Wandb
    if use_wandb and TRAINING_CONFIG['use_wandb']:
        wandb.init(
            project=project_name,
            config={**MODEL_CONFIG, **TRAINING_CONFIG},
            resume="allow" if resume_from_checkpoint else None
        )
    

    print(f"\nLoading data (max {TRAINING_CONFIG['max_data_samples']:,} samples)...")
    full_corpus = load_and_preprocess_data(max_samples=TRAINING_CONFIG['max_data_samples'])
    print(f"Loaded {len(full_corpus):,} text samples")
    

    print(f"\nLoading NEW tokenizer: {tokenizer_path}")
    tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Update config
    MODEL_CONFIG['vocab_size'] = tokenizer.vocab_size
    MODEL_CONFIG['label_smoothing'] = TRAINING_CONFIG.get('label_smoothing', 0.0)
    
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Test tokenization
    print("\nTesting new tokenizer:")
    test_text = "Türkiye'nin başkenti Ankara'dır. İstanbul Boğazı çok güzel."
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Original: {test_text}")
    print(f"Tokens: {len(tokens)} tokens")
    print(f"Decoded: {decoded}")
    print(f"Match: {'OK' if test_text.strip() == decoded.strip() else 'NO'}")
    
    # Tokenize data
    print("Tokenizing data...")
    all_tokens = []
    for text in tqdm(full_corpus, desc="Encoding texts"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        # Memory management
        if len(all_tokens) % 1000000 == 0:
            cleanup_memory()
    
    # Split data
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    print(f"Compression ratio: {len(' '.join(full_corpus)) / len(all_tokens):.2f} chars/token")
    
    # Cleanup
    del all_tokens, full_corpus
    cleanup_memory()
    
    # Create datasets
    train_dataset = Dataset(train_tokens, MODEL_CONFIG['block_size'])
    val_dataset = Dataset(val_tokens, MODEL_CONFIG['block_size'])
    
    # global_step already initialized at function start
    
    # Create basic DataLoader first (will be recreated after resume)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['dataloader_num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        prefetch_factor=TRAINING_CONFIG['prefetch_factor'],
        drop_last=True  # Consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['dataloader_num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        prefetch_factor=TRAINING_CONFIG['prefetch_factor'],
        drop_last=True
    )
    
    print(f"Train batches per epoch: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print(f"Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['accumulation_steps']}")
    
    # Initialize model
    print(f"\nInitializing Transformer...")
    model = Transformer(MODEL_CONFIG, tokenizer).to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.1f}M")
    print(f"Memory after model load: {get_memory_usage()}")
    
    # Load pretrained if specified
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\nLoading pretrained model: {pretrained_model_path}")
        model_state = load_file(pretrained_model_path)
        
        # Handle compiled model keys
        new_state_dict = {}
        for k, v in model_state.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded!")
        
        if fresh_epochs:
            TRAINING_CONFIG['max_epochs'] = fresh_epochs
            print(f"Training for {fresh_epochs} fresh epochs")
    
    # DeepSpeed vs Standard Training Setup
    if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
        print(f"\nSetting up DeepSpeed for RTX 3050...")
        
        # Create DeepSpeed config
        if TRAINING_CONFIG['deepspeed_config_path'] is None:
            deepspeed_config = create_deepspeed_config(TRAINING_CONFIG, MODEL_CONFIG)
            print(f"Using auto-generated DeepSpeed config:")
            print(f"  ZeRO Stage: {deepspeed_config['zero_optimization']['stage']}")
            print(f"  CPU Offload: {deepspeed_config['zero_optimization']['cpu_offload']}")
            print(f"  Mixed Precision: {deepspeed_config['fp16']['enabled']}")
            print(f"  Activation Checkpointing: {deepspeed_config['activation_checkpointing']['partition_activations']}")
        else:
            with open(TRAINING_CONFIG['deepspeed_config_path'], 'r') as f:
                deepspeed_config = json.load(f)
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config
        )
        
        # DeepSpeed handles everything
        model = model_engine
        scaler = None  # DeepSpeed handles mixed precision
        
        print(f"DeepSpeed initialized!")
        print(f"  Effective batch size: {deepspeed_config['train_batch_size']}")
        print(f"  Micro batch size: {deepspeed_config['train_micro_batch_size_per_gpu']}")
        print(f"  Memory usage after DeepSpeed: {get_memory_usage()}")
        
    else:
        print(f"\nUsing standard PyTorch training...")
        
        # Standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            betas=(TRAINING_CONFIG['beta1'], TRAINING_CONFIG['beta2']),
            eps=1e-8,
            fused=True if device_type == 'cuda' else False
        )
        
        # Scheduler
        total_steps = len(train_loader) * TRAINING_CONFIG['max_epochs'] // TRAINING_CONFIG['accumulation_steps']
        warmup_steps = len(train_loader) * TRAINING_CONFIG['warmup_epochs'] // TRAINING_CONFIG['accumulation_steps']
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print(f"Scheduler: Cosine with warmup ({warmup_steps:,} warmup steps)")
        
        # Mixed precision scaler
        scaler = GradScaler(device_type) if device_type == 'cuda' else None
        print(f"Mixed precision: {'OK' if scaler else 'NO'}")
    
        # Resume logic (skip for pretrained)
    if not pretrained_model_path:
        if auto_resume and not resume_from_checkpoint:
            resume_from_checkpoint = find_latest_checkpoint()
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            try:
                print(f"\nResuming from: {resume_from_checkpoint}")
                
                resume_info = load_checkpoint(
                    resume_from_checkpoint, model, optimizer, scheduler, scaler, device
                )
                loaded_epoch = resume_info['epoch']
                global_step = resume_info['global_step']
                best_val_loss = resume_info['best_val_loss']
                
                # CRITICAL: Global step'e göre doğru epoch'u hesapla
                estimated_steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
                calculated_epoch = global_step // estimated_steps_per_epoch
                
                print(f"RESUME ANALYSIS:")
                print(f"  Loaded epoch: {loaded_epoch}")
                print(f"  Global step: {global_step}")
                print(f"  Steps per epoch: {estimated_steps_per_epoch}")
                print(f"  Calculated epoch from global_step: {calculated_epoch}")
                
                start_epoch = calculated_epoch
                
                print(f"  >>> Using calculated epoch: {start_epoch}")
                print(f"  >>> This epoch has completed {global_step % estimated_steps_per_epoch} steps.")
                
                print(f"Resume successful: Epoch {start_epoch}, Step {global_step}, Best Val Loss: {best_val_loss:.4f}")
                print(f"Estimated steps per epoch: {estimated_steps_per_epoch}")
                
                # Recreate DataLoader with proper seed for resume
                print(f"Recreating DataLoader with seed based on global_step: {global_step}")
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=TRAINING_CONFIG['batch_size'],
                    shuffle=True,
                    num_workers=TRAINING_CONFIG['dataloader_num_workers'],
                    pin_memory=TRAINING_CONFIG['pin_memory'],
                    prefetch_factor=TRAINING_CONFIG['prefetch_factor'],
                    drop_last=True,
                    generator=torch.Generator().manual_seed(42 + global_step)
                )
                
            except Exception as e:
                print(f"Resume failed: {e}")
                print("Starting fresh training instead...")
                start_epoch = 0
                global_step = 0
                best_val_loss = float('inf')
    
    # Training state is already initialized at function start
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['early_stopping_patience'],
        min_delta=TRAINING_CONFIG['early_stopping_min_delta'],
        restore_best_weights=True
    )
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}")
    print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"Global step: {global_step}")
    print("="*80)
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory created: checkpoints/")
    
    for epoch in range(start_epoch, TRAINING_CONFIG['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['max_epochs']}")
        print(f"Memory before epoch: {get_memory_usage()}")
        print(f"Global step: {global_step}")
        
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        

        print(f"Resume info: Starting from saved epoch {epoch + 1}, global step {global_step}")
        estimated_steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
        expected_epoch = global_step // estimated_steps_per_epoch
        print(f"  Estimated steps per epoch: {estimated_steps_per_epoch}")
        print(f"  Expected epoch from global_step: {expected_epoch}")
        
        batch_counter = 0
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(device_type=device_type, enabled=(device_type == 'cuda' and not (DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']))):
                _, loss = model(inputs, targets)
                if not (DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']):
                    loss = loss / TRAINING_CONFIG['accumulation_steps']
            
            # Backward pass
            if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
                # DeepSpeed backward
                model.backward(loss)
            else:
                # Standard backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Update metrics
            if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
                batch_loss = loss.item()
            else:
                batch_loss = loss.item() * TRAINING_CONFIG['accumulation_steps']
            
            total_train_loss += batch_loss
            num_train_batches += 1
            batch_counter += 1
            
            # Update progress
            if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
                current_lr = model.get_lr()[0] if hasattr(model, 'get_lr') else TRAINING_CONFIG['learning_rate']
            else:
                current_lr = scheduler.get_last_lr()[0]
            
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'mem': get_memory_usage_safe(),
                'step': global_step,
                'processed': batch_counter
            })
            
            # Optimizer step
            if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
                    # DeepSpeed step
                    model.step()
                else:
                    # Standard step
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Increment global_step for actual training steps
                global_step += 1
                
                # Logging
                if global_step % TRAINING_CONFIG['log_interval'] == 0 and use_wandb:
                    wandb.log({
                        'train_loss_step': batch_loss,
                        'learning_rate': current_lr,
                        'global_step': global_step,
                        'epoch': epoch + 1
                    })
                
                # Quick evaluation
                if global_step % TRAINING_CONFIG['eval_steps'] == 0 and global_step > 0:
                    print(f"\nQuick eval at step {global_step}")
                    model.eval()
                    quick_val_loss = 0
                    quick_batches = 0
                    
                    with torch.no_grad():
                        for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                            if val_batch_idx >= 10:  # Quick eval
                                break
                            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                            
                            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                                _, val_loss = model(val_inputs, val_targets)
                            
                            quick_val_loss += val_loss.item()
                            quick_batches += 1
                    
                    avg_quick_val_loss = quick_val_loss / quick_batches if quick_batches > 0 else float('inf')
                    print(f"Step {global_step}: Train: {total_train_loss/num_train_batches:.4f}, Quick Val: {avg_quick_val_loss:.4f}")
                    
                    if use_wandb:
                        wandb.log({
                            'quick_val_loss': avg_quick_val_loss,
                            'train_loss_avg': total_train_loss/num_train_batches,
                            'global_step': global_step
                        })
                    
                    model.train()
                    cleanup_memory()
                
                # Checkpoint saving
                if global_step % TRAINING_CONFIG['checkpoint_steps'] == 0 and global_step > 0:
                    checkpoint_path = f"checkpoints/checkpoint_step_{global_step}.safetensors"
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step,
                        best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                        checkpoint_path=checkpoint_path
                    )
            
            if global_step % 10 == 0: 
                cleanup_memory()
                
            if batch_idx % TRAINING_CONFIG['accumulation_steps'] == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / num_train_batches
        print(f"\nEpoch {epoch + 1} completed: Train Loss: {avg_train_loss:.4f}")
        print(f"Memory after epoch: {get_memory_usage()}")
        
        # Comprehensive evaluation
        if (epoch + 1) % TRAINING_CONFIG['eval_interval'] == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION - Epoch {epoch + 1}")
            print(f"{'='*80}")
            
            metrics = evaluate_model_comprehensive(model, val_loader, tokenizer, device, device_type, TRAINING_CONFIG)
            
            val_loss = metrics['val_loss']
            perplexity = metrics['perplexity']
            
            print(f"METRICS:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Memory: {get_memory_usage()}")
            
            # Generation sample
            if 'generation' in metrics and metrics['generation']['samples']:
                sample = metrics['generation']['samples'][0]
                print(f"   GENERATION SAMPLE:")
                print(f"   Prompt: {sample['prompt']}")
                print(f"   Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
            
            print(f"{'='*80}")
            
            # Wandb logging
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'val_loss_epoch': val_loss,
                    'perplexity': perplexity,
                    'learning_rate_epoch': current_lr,
                    'global_step': global_step
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step,
                    best_val_loss, perplexity, MODEL_CONFIG, tokenizer_path,
                    checkpoint_path="checkpoints/best_model_130m_rtx3050.safetensors"
                )
                print(f"New best model saved! Val loss: {val_loss:.4f}")
            
            # Early stopping
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {early_stopping.best_loss:.4f}")
                break
            
            cleanup_memory()
        
        # Regular checkpoint
        if (epoch + 1) % TRAINING_CONFIG['save_interval'] == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.safetensors"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                checkpoint_path=checkpoint_path
            )
    
    if use_wandb:
        wandb.finish()
    
    print(f"\nTraining completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final memory usage: {get_memory_usage()}")
    
    return model

# Utility functions (unchanged but optimized)
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                   best_val_loss, best_perplexity, config, tokenizer_path, 
                   checkpoint_path="checkpoint.safetensors"):
    
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    
    # DeepSpeed checkpoint saving
    if DEEPSPEED_AVAILABLE and hasattr(model, 'save_checkpoint'):
        # DeepSpeed handles checkpoint saving
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        model.save_checkpoint(checkpoint_dir, checkpoint_name)
        
        # Save additional metadata
        metadata = {
            'epoch': str(epoch),
            'global_step': str(global_step),
            'best_val_loss': str(best_val_loss),
            'best_perplexity': str(best_perplexity),
            'config': json.dumps(config),
            'tokenizer_path': tokenizer_path,
            'training_config': json.dumps(TRAINING_CONFIG),
            'model_config': json.dumps(MODEL_CONFIG),
            'deepspeed': 'true',
            'model_type': 'TurkishTransformer'
        }
        
        metadata_path = checkpoint_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"DeepSpeed checkpoint saved: {checkpoint_name}")
        
    else:
        # Standard SafeTensors saving
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors not available. Install with: pip install safetensors")
        
        # Model state - weight tying için sadece embedding weight'i kaydet
        model_state = {}
        state_dict = model.state_dict() if hasattr(model, 'state_dict') else model.module.state_dict()
        
        for name, param in state_dict.items():
            if name != 'lm_head.weight':  # Weight tying nedeniyle lm_head skip
                model_state[name] = param
        
        # Metadata
        metadata = {
            'epoch': str(epoch),
            'global_step': str(global_step),
            'best_val_loss': str(best_val_loss),
            'best_perplexity': str(best_perplexity),
            'config': json.dumps(config),
            'tokenizer_path': tokenizer_path,
            'training_config': json.dumps(TRAINING_CONFIG),
            'model_config': json.dumps(MODEL_CONFIG),
            'weight_tying': 'true',
            'deepspeed': 'false',
            'model_type': 'TurkishTransformer'
        }
        
        # Save to SafeTensors
        save_file(model_state, checkpoint_path, metadata=metadata)
        
        # Save additional training state
        if optimizer is not None:
            additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
            additional_state = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            
            if scaler:
                additional_state['scaler'] = scaler.state_dict()
            
            torch.save(additional_state, additional_state_path)
        
        print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
    
    cleanup_memory()
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device=None):
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load metadata and model state
    from safetensors import safe_open
    metadata = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    
    model_state = load_file(checkpoint_path)
    
    # Remove compiled model prefixes
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    # Load model weights
    model.load_state_dict(new_state_dict, strict=False)
    print("Model weights loaded")
    
    # Load training state
    additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
    if os.path.exists(additional_state_path):
        additional_state = torch.load(additional_state_path, map_location=device, weights_only=True)
        
        if optimizer and 'optimizer' in additional_state:
            optimizer.load_state_dict(additional_state['optimizer'])
            print("Optimizer state loaded")
        
        if scheduler and 'scheduler' in additional_state and additional_state['scheduler']:
            scheduler.load_state_dict(additional_state['scheduler'])
            print("Scheduler state loaded")
        
        if scaler and 'scaler' in additional_state:
            scaler.load_state_dict(additional_state['scaler'])
            print("Scaler state loaded")
    
    # Parse metadata
    resume_info = {
        'epoch': int(metadata.get('epoch', '0')),
        'global_step': int(metadata.get('global_step', '0')),
        'best_val_loss': float(metadata.get('best_val_loss', 'inf')),
        'best_perplexity': float(metadata.get('best_perplexity', 'inf')),
        'config': json.loads(metadata.get('config', '{}')),
        'tokenizer_path': metadata.get('tokenizer_path', ''),
    }
    
    print(f"Resume from: Epoch {resume_info['epoch']}, Step {resume_info['global_step']}")
    cleanup_memory()
    
    return resume_info

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find latest checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_') and file.endswith('.safetensors'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        return None
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

def load_and_preprocess_data(max_samples=150000):
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    dataset = load_dataset("musabg/wikipedia-tr-summarization", split='train')
    processed_texts = []
    
    print(f"Dataset total size: {len(dataset):,}")
    
    max_samples = min(len(dataset), max_samples)
    
    for i in tqdm(range(max_samples), desc="Processing texts"):
        text = clean_text(dataset[i]["text"])
        if len(text) > 100:  # Longer texts for better training
            processed_texts.append(text)
        
        # Memory management
        if i % 10000 == 0:
            cleanup_memory()
    
    print(f"Processed {len(processed_texts):,} texts")
    return processed_texts

def generate(text, 
             model_path="checkpoints/best_model_100m.safetensors",
             tokenizer_path="turkish_tokenizer/turkish_tokenizer.model",
             max_new_tokens=100,
             temperature=0.8,
             top_p=0.9,
             top_k=50,
             device=None,
             use_half_precision=True):

    # Device setup
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    print(f"Generating text with RTX 3050 optimized model")
    print(f"Model: {model_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Device: {device}")
    print(f"Half precision: {use_half_precision}")
    
    # Load tokenizer
    tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Load model
    if not model_path.endswith('.safetensors'):
        model_path = model_path.replace('.pt', '.safetensors')
    
    from safetensors import safe_open
    metadata = {}
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    
    config = json.loads(metadata.get('config', '{}'))
    
    # Disable FlashAttention for generation to avoid dtype issues
    config['use_flash_attention'] = False
    print("FlashAttention disabled for generation (dtype compatibility)")
    
    # Create model
    model = Transformer(config, tokenizer).to(device)
    
    # Load weights
    model_state = load_file(model_path)
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Convert to half precision if requested and on CUDA
    if use_half_precision and device.type == 'cuda':
        model = model.half()
        print("Model converted to half precision (fp16)")
    
    model.eval()
    
    print(f"Model loaded: {model.get_num_params()/1e6:.1f}M parameters")
    print(f"Memory usage: {get_memory_usage()}")
    
    # Generate
    with torch.no_grad():
        try:
            generated_text = model.generate_from_prompt(
                text, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        except Exception as e:
            print(f"Generation failed: {e}")
            print("Falling back to CPU generation without half precision...")
            
            # Fallback to CPU with full precision
            model = model.float().cpu()
            generated_text = model.generate_from_prompt(
                text, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
    
    print(f"\nGenerated text:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    cleanup_memory()
    return generated_text

if __name__ == "__main__":
    try:
        #model = train()
        
        # Örnek kullanımlar:
        model = train(auto_resume=True)  # Resume from latest checkpoint
        # model = train(resume_from_checkpoint="checkpoints/checkpoint_step_1000.safetensors")
        # model = train(pretrained_model_path="checkpoints/best_model_130m_rtx3050.safetensors", fresh_epochs=10)
                
        # Generation example:
        #generate("Ekrem İmamoğlu", model_path="checkpoints/checkpoint_step_1500.safetensors", use_half_precision=False)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Proper cleanup for DeepSpeed/PyTorch
        try:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                print("Process group destroyed successfully")
        except:
            pass
        
        cleanup_memory()
        print("Training session ended")