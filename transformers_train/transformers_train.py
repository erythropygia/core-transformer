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

import numpy as np

# Global flags to prevent spam messages
_MESSAGES_PRINTED = False
_WEIGHT_TYING_MSG_PRINTED = False  # Add flag for weight tying message

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
    import sentencepiece as spm
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
    FLASH_ATTENTION_AVAILABLE = False
    if not _MESSAGES_PRINTED:
        print("Flash Attention available! But false by default due to precision issues")
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

# 100M Parametreli Model Konfigürasyonu
MODEL_CONFIG = {
    'n_embd': 768,          # 768 embedding dimension (GPT-2 Small benzeri)
    'n_layer': 12,          # 12 transformer layer
    'n_head': 12,           # 12 attention head
    'block_size': 1024,     # 1024 context window
    'dropout': 0.1,
    'vocab_size': None,     # Tokenizer'dan alınacak
    'use_flash_attention': False,  # Flash Attention kullan
    'use_gradient_checkpointing': True,  # Gradient checkpointing kullan
}

# Eğitim Konfigürasyonu
TRAINING_CONFIG = {
    'batch_size': 16,       # Büyük model için küçük batch
    'learning_rate': 6e-4,  # Biraz daha büyük LR
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'warmup_epochs': 2,
    'max_epochs': 20,
    'eval_interval': 1,     # Her epoch'ta evaluate et
    'save_interval': 2,     # Her 2 epoch'ta kaydet
    'accumulation_steps': 8, # Gradient accumulation
    'use_wandb': True,
    'compile_model': True,  # PyTorch 2.0 compile
    'scheduler_type': 'cosine_with_warmup',  # 'cosine_with_warmup', 'onecycle', 'plateau'
    'eval_generation_samples': 5,  # Generation quality için sample sayısı
    'max_eval_batches': 100,  # Evaluation batch limiti
    
    # Progress reporting
    'log_interval': 10,  # Her 100 batch'te progress logla
    'eval_steps': 500,   # Her 500 step'te hızlı evaluation yap
    'checkpoint_steps': 50,  # Her 100 step'te checkpoint kaydet
    
    'vocab_size': 32000,  # SentencePiece için vocab size
}

# Test prompts for generation quality evaluation
TEST_PROMPTS = [
    "Türkiye'nin başkenti Ankara",
    "Yapay zeka teknolojisi",
    "İstanbul Boğazı",
    "Osmanlı İmparatorluğu",
    "Matematik dersi"
]

class Tokenizer:
    """
    SentencePiece tabanlı Türkçe tokenizer - Türkçe'nin agglutinative yapısı için optimize edilmiş
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
            print(f"Loaded SentencePiece model: {model_path}")
            # Special token'ları modelden al
            self.special_tokens = []
            for i in range(min(10, self.sp.get_piece_size())):  # İlk 10 token genelde special token'lar
                piece = self.sp.id_to_piece(i)
                if piece.startswith('<') and piece.endswith('>'):
                    self.special_tokens.append(piece)
            print(f"Detected special tokens: {self.special_tokens}")
        else:
            # Eğitim için default special token'lar (artık kullanılmıyor)
            self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    
    def train(self, 
              corpus: List[str], 
              model_prefix: str = "turkish_spm",
              vocab_size: int = 32000,
              character_coverage: float = 0.995,
              verbose: bool = True):
        """
        SentencePiece modelini Türkçe corpus ile eğitir
        
        Args:
            corpus: Eğitim metinleri listesi
            model_prefix: Model dosya adı öneki
            vocab_size: Vocabulary boyutu
            character_coverage: Karakter kapsama oranı (Türkçe için 0.995 ideal)
        """
        
        # Corpus'u geçici dosyaya yaz
        corpus_file = f"{model_prefix}_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in corpus:
                f.write(text.strip() + '\n')
        
        # SentencePiece training parametreleri
        train_args = [
            f'--input={corpus_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={vocab_size}',
            f'--character_coverage={character_coverage}',
            '--model_type=bpe',  # BPE algoritması kullan
            '--split_by_unicode_script=true',  # Unicode script'lere göre ayır
            '--split_by_whitespace=true',  # Boşluklara göre ayır
            '--normalization_rule_name=nfkc',  # Unicode normalizasyonu
            '--remove_extra_whitespaces=true',
            '--input_sentence_size=2000000',  # Max sentence sayısı
            '--shuffle_input_sentence=true',  # Input'u karıştır
            
            # Türkçe özel ayarlar
            '--treat_whitespace_as_suffix=false',  # Türkçe için daha iyi
            '--allow_whitespace_only_pieces=true',
            '--max_sentence_length=16384',
            
            # Özel tokenlar
            '--pad_id=0',
            '--unk_id=1', 
            '--bos_id=2',
            '--eos_id=3',
            '--pad_piece=<pad>',
            '--unk_piece=<unk>',
            '--bos_piece=<s>',
            '--eos_piece=</s>',
        ]
        
        if verbose:
            print("Training SentencePiece model for Turkish...")
            print(f"Corpus size: {len(corpus)} texts")
            print(f"Vocab size: {vocab_size}")
        
        # Model eğit
        spm.SentencePieceTrainer.train(' '.join(train_args))
        
        # Modeli yükle
        self.model_path = f"{model_prefix}.model"
        self.sp.load(self.model_path)
        
        # Geçici dosyayı sil
        os.remove(corpus_file)
        
        if verbose:
            print(f"SentencePiece model trained and saved: {self.model_path}")
            print(f"Actual vocab size: {self.sp.get_piece_size()}")
            self._print_sample_tokens()
    
    def _print_sample_tokens(self):
        """Örnek tokenları göster"""
        sample_texts = [
            "Merhaba dünya",
            "İstanbul'da yaşıyorum", 
            "Türkiye Cumhuriyeti",
            "Çiçekçi dükkânı",
            "Öğretmenlik mesleği"
        ]
        
        print("\nSample tokenizations:")
        for text in sample_texts:
            tokens = self.sp.encode_as_pieces(text)
            print(f"'{text}' -> {tokens}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Metni token ID'lerine çevirir"""
        if add_special_tokens:
            # <s> ekle başına
            token_ids = [self.sp.bos_id()] + self.sp.encode_as_ids(text)
        else:
            token_ids = self.sp.encode_as_ids(text)
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token ID'lerini metne çevirir"""
        if skip_special_tokens:
            # Özel tokenları filtrele
            filtered_ids = [
                tid for tid in token_ids 
                if tid not in [self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()]
            ]
            return self.sp.decode_ids(filtered_ids)
        else:
            return self.sp.decode_ids(token_ids)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Metni token piece'lerine çevirir (debug için)"""
        return self.sp.encode_as_pieces(text)
    
    def get_vocab_size(self) -> int:
        """Vocabulary boyutunu döndürür"""
        return self.sp.get_piece_size()
    
    def save_model(self, save_path: str):
        """Modeli başka bir konuma kaydet"""
        if self.model_path and os.path.exists(self.model_path):
            import shutil
            shutil.copy2(self.model_path, save_path)
            print(f"Model saved to: {save_path}")
    
    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()
    
    @property 
    def token_to_id(self) -> Dict[str, int]:
        """Token string to ID mapping"""
        return {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}


def create_tokenizer(model_path: str = None) -> Tokenizer:
    """
    SentencePiece tokenizer oluşturur
    
    Args:
        model_path: Model dosya yolu
    """
    
    if not SENTENCEPIECE_AVAILABLE:
        raise ImportError("SentencePiece not available, install with: pip install sentencepiece")
    
    if not model_path:
        raise ValueError("model_path is required")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tokenizer file not found: {model_path}")
    
    tokenizer = Tokenizer(model_path)
    print(f"Using SentencePiece tokenizer (vocab_size: {tokenizer.vocab_size})")
    return tokenizer


class Dataset(Dataset):
    
    def __init__(self, tokens, block_size, stride=None):
        self.tokens = tokens
        self.block_size = block_size
        self.stride = stride or block_size // 2  # Overlapping sequences
        
        # Sequence'ları önceden hesapla
        self.sequences = []
        max_start = len(tokens) - block_size
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
    """Geliştirilmiş Transformer Block with Gradient Checkpointing"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.mlp = FeedForward(config)
    
    def _forward_impl(self, x):
        """Actual forward implementation"""
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    def forward(self, x):
        if self.config.get('use_gradient_checkpointing', False) and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        self.config = config
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = config['n_embd'] // config['n_head']
        self.use_flash = config.get('use_flash_attention', False) and FLASH_ATTENTION_AVAILABLE
        
        # Fused QKV projection (daha efficient)
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=False)
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        
        # Causal mask (Flash Attention kullanmıyorsak)
        if not self.use_flash:
            self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                .view(1, 1, config['block_size'], config['block_size']))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # QKV projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Flash Attention veya standard attention
        if self.use_flash and q.device.type in ['cuda', 'mps']:  # Flash attention supports both CUDA and MPS
            # Flash Attention kullan
            # Flash attention expects (B, S, H, D) format
            q = q.transpose(1, 2).contiguous()  # (B, T, H, D)
            k = k.transpose(1, 2).contiguous()  # (B, T, H, D) 
            v = v.transpose(1, 2).contiguous()  # (B, T, H, D)
            
            y = flash_attn_func(q, k, v, dropout_p=self.config['dropout'] if self.training else 0.0, causal=True)
            y = y.view(B, T, C)
        else:
            # Standard scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.c_proj(y))

class FeedForward(nn.Module):
    
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

class Transformers(nn.Module):
    
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Embeddings
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.wpe = nn.Embedding(config['block_size'], config['n_embd'])
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying (embeddings ve output projection aynı ağırlıkları paylaşır)
        # Instead of direct assignment, we'll use a property to handle weight tying
        self._weight_tying = True
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for output projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))
    
    @property
    def weight_tying(self):
        return self._weight_tying
    
    @weight_tying.setter
    def weight_tying(self, value):
        self._weight_tying = value
        if value:
            # When enabling weight tying, we don't actually share the weights
            # Instead, we'll handle it in forward pass
            pass
        else:
            # When disabling weight tying, we need to create a new weight tensor
            if hasattr(self, 'wte'):
                self.lm_head.weight = nn.Parameter(self.wte.weight.clone())
    
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
        
        # Final layer norm ve output projection
        x = self.ln_f(x)
        
        # Handle weight tying in forward pass
        if self._weight_tying:
            # Use embedding weights directly for output projection
            logits = F.linear(x, self.wte.weight)
        else:
            # Use separate weights
            logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=0.9):
        self.eval()
        
        for _ in range(max_new_tokens):
            # Context window'u kırp
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

@torch.no_grad()
def calculate_perplexity(model, data_loader, device, device_type, max_batches=100):
    """Perplexity"""
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
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    return perplexity

@torch.no_grad() 
def evaluate_generation_quality(model, tokenizer, test_prompts, device, max_new_tokens=50):
    """Generation quality evaluation"""
    model.eval()
    results = {
        'samples': [],
        'avg_length': 0,
        'unique_tokens_ratio': 0,
        'bleu_scores': []
    }
    
    total_length = 0
    all_tokens = set()
    total_tokens = 0
    
    for prompt in test_prompts:
        try:
            # Generate text
            generated = model.generate_from_prompt(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                top_k=50
            )
            
            # Extract only the generated part (remove prompt)
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
            
            # Length statistics
            tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            total_length += len(tokens)
            all_tokens.update(tokens)
            total_tokens += len(tokens)
            
        except Exception as e:
            print(f"Generation error for prompt '{prompt}': {e}")
            continue
    
    # Calculate metrics
    if results['samples']:
        results['avg_length'] = total_length / len(results['samples'])
        results['unique_tokens_ratio'] = len(all_tokens) / max(total_tokens, 1)
    
    return results

@torch.no_grad()
def evaluate_model_comprehensive(model, val_loader, tokenizer, device, device_type, config):
    """Comprehensive model evaluation"""
    metrics = {}
    
    # 1. Standard loss evaluation
    total_loss = 0
    num_batches = 0
    max_batches = config.get('max_eval_batches', 100)
    
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            _, loss = model(inputs, targets)
        
        total_loss += loss.item()
        num_batches += 1
    
    metrics['val_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # 2. Perplexity
    try:
        metrics['perplexity'] = calculate_perplexity(model, val_loader, device, device_type, max_batches)
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        metrics['perplexity'] = float('inf')
    
    # 3. Generation quality
    try:
        num_samples = config.get('eval_generation_samples', 5)
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
    auto_resume=True,  # Automatically find and resume from latest checkpoint
    use_wandb=True,
    project_name="transformers-100m-turkish"
):
    
    # Configure torch._dynamo to suppress errors and fall back to eager mode
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except ImportError:
        pass  # torch._dynamo might not be available in older PyTorch versions
    
    # Device setup - Try MPS (Metal GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = "mps"
        print(f"Using device: {device} (Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = "cuda"
        print(f"Using device: {device} (CUDA)")
    else:
        device = torch.device('cpu')
        device_type = "cpu"
        print(f"Using device: {device} (CPU)")
    
    print(f"Device type: {device_type}")
    
    # Feature availability info
    print(f"Flash Attention: {'OK' if FLASH_ATTENTION_AVAILABLE else 'NO'}")
    print(f"Gradient Checkpointing: OK")
    print(f"Advanced Metrics: OK")
    print(f"SentencePiece: OK")
    
    # Wandb initialization
    if use_wandb and TRAINING_CONFIG['use_wandb']:
        wandb.init(
            project=project_name,
            config={**MODEL_CONFIG, **TRAINING_CONFIG},
            resume="allow" if resume_from_checkpoint else None
        )
    
    # Load data
    print("Loading data...")
    full_corpus = load_and_preprocess_data(max_samples=100000)  # Daha büyük dataset
    
    # Ensure tokenizer path has .model extension for SentencePiece
    if not tokenizer_path.endswith('.model'):
        tokenizer_path += '.model'
    
    # Load existing tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}. Please provide an existing SentencePiece tokenizer.")
    
    print(f"Loading SentencePiece tokenizer from: {tokenizer_path}")
    tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Update config with vocab size
    MODEL_CONFIG['vocab_size'] = tokenizer.vocab_size
    
    # Disable Flash Attention completely due to precision compatibility issues
    MODEL_CONFIG['use_flash_attention'] = False
    print("Flash Attention disabled (precision compatibility issues)")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test tokenization
    print("\nTesting tokenization:")
    test_texts = [
        "Türkiye'nin başkenti Ankara'dır.",
        "İstanbul Boğazı çok güzel.",
        "Merhaba dünya! Nasılsınız?"
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Original: {text}")
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (length: {len(tokens)})")
        print(f"Decoded: {decoded}")
        print(f"Match: {'OK' if text.strip() == decoded.strip() else 'NO'}")
        print("-" * 50)
    
    # Tokenize data
    print("Tokenizing data...")
    all_tokens = []
    for text in tqdm(full_corpus, desc="Encoding texts"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # Split data
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    print(f"Compression ratio: {len(' '.join(full_corpus)) / len(all_tokens):.2f} chars/token")
    
    # Create datasets and dataloaders
    train_dataset = Dataset(train_tokens, MODEL_CONFIG['block_size'])
    val_dataset = Dataset(val_tokens, MODEL_CONFIG['block_size'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    model = Transformers(MODEL_CONFIG, tokenizer).to(device)
    print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Model features info
    print(f"Using Flash Attention: {MODEL_CONFIG.get('use_flash_attention', False) and FLASH_ATTENTION_AVAILABLE}")
    print(f"Using Gradient Checkpointing: {MODEL_CONFIG.get('use_gradient_checkpointing', False)}")
    
    # Compile model for better performance (PyTorch 2.0+)
    # Note: MPS backend has limited support for torch.compile, so we disable it for compatibility
    if TRAINING_CONFIG['compile_model'] and device_type != 'mps':
        try:
            model = torch.compile(model)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing with eager mode")
    elif device_type == 'mps':
        print("Model compilation disabled for MPS (Metal GPU) - using eager mode for better compatibility")
    else:
        print("Model compilation disabled - using eager mode")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        betas=(TRAINING_CONFIG['beta1'], TRAINING_CONFIG['beta2'])
    )
    
    # Advanced Learning Rate Scheduling
    total_steps = len(train_loader) * TRAINING_CONFIG['max_epochs']
    warmup_steps = len(train_loader) * TRAINING_CONFIG['warmup_epochs']
    
    scheduler_type = TRAINING_CONFIG.get('scheduler_type', 'cosine_with_warmup')
    
    if scheduler_type == 'cosine_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print("Using Cosine scheduler with warmup")
    elif scheduler_type == 'onecycle':
        scheduler = get_onecycle_scheduler(optimizer, TRAINING_CONFIG['learning_rate'], total_steps)
        print("Using OneCycle scheduler")
    elif scheduler_type == 'plateau':
        scheduler = get_plateau_scheduler(optimizer)
        print("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print("Using default Cosine scheduler with warmup")
    
    # Mixed precision scaler
    # Note: MPS doesn't support float64 operations needed by GradScaler, so we only use it for CUDA
    if device_type == 'cuda':
        scaler = GradScaler(device_type)
        print("Using mixed precision with GradScaler")
    else:
        scaler = None  # CPU and MPS don't use mixed precision scaler
        if device_type == 'mps':
            print("Mixed precision scaler disabled for MPS (float64 not supported)")
        else:
            print("Mixed precision scaler disabled for CPU")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_perplexity = float('inf')
    global_step = 0
    
    # Auto-resume: find latest checkpoint if no specific checkpoint given
    if auto_resume and not resume_from_checkpoint:
        resume_from_checkpoint = find_latest_checkpoint()
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        resume_info = load_checkpoint(
            resume_from_checkpoint, 
            model, 
            optimizer, 
            scheduler if scheduler_type != 'plateau' else None, 
            scaler, 
            device
        )
        
        start_epoch = resume_info['epoch']
        global_step = resume_info['global_step']
        best_val_loss = resume_info['best_val_loss']
        best_perplexity = resume_info['best_perplexity']
        
        print(f"Training will resume from Epoch {start_epoch + 1}, Step {global_step}")
    elif resume_from_checkpoint:
        print(f"Checkpoint file not found: {resume_from_checkpoint}")
        print("Starting training...")
    else:
        print("Starting training...")
    
    # Initialize step counter for logging (adjust if resuming)
    if global_step == 0:
        global_step = 0  
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, TRAINING_CONFIG['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['max_epochs']}")
        
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        # Skip batches if resuming mid-epoch
        steps_per_epoch = len(train_loader)
        current_epoch_steps = global_step % steps_per_epoch if steps_per_epoch > 0 else 0
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Skip already processed steps in current epoch
            if batch_idx < current_epoch_steps:
                continue
                
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Gradient accumulation
            # Note: Only enable autocast for CUDA; MPS has limited mixed precision support
            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                _, loss = model(inputs, targets)
                loss = loss / TRAINING_CONFIG['accumulation_steps']
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Calculate batch loss for display
            batch_loss = loss.item() * TRAINING_CONFIG['accumulation_steps']
            total_train_loss += batch_loss
            epoch_loss += batch_loss
            num_train_batches += 1
            
            # Update progress bar every batch
            current_lr = scheduler.get_last_lr()[0] if scheduler_type != 'plateau' else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'lr': f"{current_lr:.2e}",
                'step': global_step
            })
            
            # Check if we should do optimizer step (every accumulation_steps batches)
            if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                if scaler:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # CPU training without scaler
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                    optimizer.step()
                
                # Scheduler step (except plateau)
                if scheduler_type != 'plateau':
                    scheduler.step()
                optimizer.zero_grad()
                
                # INCREMENT GLOBAL STEP AFTER OPTIMIZER STEP
                global_step += 1
                
                # Update progress bar with new step count
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'step': global_step
                })
                
                # Regular logging to wandb
                if global_step % TRAINING_CONFIG['log_interval'] == 0 and use_wandb and TRAINING_CONFIG['use_wandb']:
                    wandb.log({
                        'train_loss_step': batch_loss,
                        'learning_rate': current_lr,
                        'global_step': global_step,
                        'epoch': epoch + 1
                    })
                
                # Quick evaluation at regular intervals
                if global_step % TRAINING_CONFIG['eval_steps'] == 0 and global_step > 0:
                    model.eval()
                    quick_val_loss = 0
                    quick_batches = 0
                    
                    with torch.no_grad():
                        for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                            if val_batch_idx >= 20:  # Quick evaluation with 20 batches
                                break
                            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                            
                            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                                _, val_loss = model(val_inputs, val_targets)
                            
                            quick_val_loss += val_loss.item()
                            quick_batches += 1
                    
                    avg_quick_val_loss = quick_val_loss / quick_batches if quick_batches > 0 else float('inf')
                    
                    print(f"\nStep {global_step}: Train Loss: {epoch_loss/num_train_batches:.4f}, Quick Val Loss: {avg_quick_val_loss:.4f}")
                    
                    if use_wandb and TRAINING_CONFIG['use_wandb']:
                        wandb.log({
                            'quick_val_loss': avg_quick_val_loss,
                            'train_loss_avg': epoch_loss/num_train_batches,
                            'global_step': global_step,
                            'epoch': epoch + 1
                        })
                    
                    model.train()  # Back to training mode
                
                # Step-based checkpoint saving
                if global_step % TRAINING_CONFIG['checkpoint_steps'] == 0 and global_step > 0:
                    step_checkpoint_path = f"checkpoints/checkpoint_step_{global_step}.safetensors"
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step,
                        best_val_loss, best_perplexity, MODEL_CONFIG, tokenizer_path,
                        checkpoint_path=step_checkpoint_path
                    )
        
        avg_train_loss = total_train_loss / num_train_batches
        print(f"\nEpoch {epoch + 1} completed: Avg Train Loss: {avg_train_loss:.4f}")
        
        # Comprehensive Evaluation
        if (epoch + 1) % TRAINING_CONFIG['eval_interval'] == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION - Epoch {epoch + 1}")
            print(f"{'='*60}")
            
            metrics = evaluate_model_comprehensive(model, val_loader, tokenizer, device, device_type, TRAINING_CONFIG)
            
            val_loss = metrics['val_loss']
            perplexity = metrics['perplexity']
            
            # Main metrics
            print(f"TRAINING METRICS:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Generation quality
            if metrics.get('avg_gen_length', 0) > 0:
                print(f"GENERATION QUALITY:")
                print(f"   Avg Length: {metrics.get('avg_gen_length', 0):.1f} tokens")
                print(f"   Unique Token Ratio: {metrics.get('unique_token_ratio', 0):.3f}")
            
            # Show one generation sample
            if 'generation' in metrics and metrics['generation']['samples']:
                sample = metrics['generation']['samples'][0]
                print(f"SAMPLE GENERATION:")
                print(f"   Prompt: {sample['prompt']}")
                print(f"   Generated: {sample['generated'][:150]}{'...' if len(sample['generated']) > 150 else ''}")
            
            print(f"{'='*60}")
            
            # Plateau scheduler step
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            
            # Comprehensive wandb logging
            if use_wandb and TRAINING_CONFIG['use_wandb']:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'val_loss_epoch': val_loss,
                    'perplexity': perplexity,
                    'learning_rate_epoch': current_lr,
                    'avg_gen_length': metrics.get('avg_gen_length', 0),
                    'unique_token_ratio': metrics.get('unique_token_ratio', 0),
                    'global_step': global_step
                }
                wandb.log(log_dict)
            
            # Save best model (based on validation loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_perplexity = perplexity
                
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step,
                    best_val_loss, best_perplexity, MODEL_CONFIG, tokenizer_path,
                    checkpoint_path="checkpoints/best_model_100m.safetensors"
                )
                print(f"New best model saved! Val loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % TRAINING_CONFIG['save_interval'] == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.safetensors"
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                best_val_loss, best_perplexity, MODEL_CONFIG, tokenizer_path,
                checkpoint_path=checkpoint_path
            )
    
    if use_wandb and TRAINING_CONFIG['use_wandb']:
        wandb.finish()
    
    print(f"\nTraining completed! Best Val Loss: {best_val_loss:.4f}, Best Perplexity: {best_perplexity:.2f}")
    return model

# Utility functions
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                   best_val_loss, best_perplexity, config, tokenizer_path, 
                   checkpoint_path="checkpoint.safetensors", use_safetensors=True):
    """
    Comprehensive checkpoint saving with SafeTensors format
    """
    # Ensure .safetensors extension
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors not available. Install with: pip install safetensors")
    
    # Prepare model state dict for SafeTensors
    model_state = model.state_dict()
    
    # Remove lm_head.weight since we handle weight tying in forward pass
    if 'lm_head.weight' in model_state:
        del model_state['lm_head.weight']
    
    # Prepare metadata for SafeTensors
    metadata = {
        'epoch': str(epoch),
        'global_step': str(global_step),
        'best_val_loss': str(best_val_loss),
        'best_perplexity': str(best_perplexity),
        'config': json.dumps(config),
        'tokenizer_path': tokenizer_path,
        'training_config': json.dumps(TRAINING_CONFIG),
        'model_config': json.dumps(MODEL_CONFIG),
        'weight_tying': 'true'  # Flag to indicate weight tying
    }
    
    # Save model weights with metadata to SafeTensors
    save_file(model_state, checkpoint_path, metadata=metadata)
    
    # Save optimizer and scheduler state separately (SafeTensors doesn't support these complex objects)
    additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
    additional_state = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    
    if scaler:
        additional_state['scaler'] = scaler.state_dict()
    
    torch.save(additional_state, additional_state_path)
    
    print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device=None):
    """
    Comprehensive checkpoint loading with SafeTensors format
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors not available. Install with: pip install safetensors")
    
    # Load metadata first
    from safetensors import safe_open
    metadata = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    
    # Load model weights from SafeTensors
    model_state = load_file(checkpoint_path)
    
    # Set weight tying based on metadata
    if metadata.get('weight_tying') == 'true':
        model.weight_tying = True
    else:
        model.weight_tying = False
    
    model.load_state_dict(model_state)
    print("Model weights loaded from SafeTensors")
    
    # Load additional training state
    additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
    additional_state = {}
    
    if os.path.exists(additional_state_path):
        additional_state = torch.load(additional_state_path, map_location=device, weights_only=True)
        
        # Load optimizer state
        if optimizer and 'optimizer' in additional_state:
            optimizer.load_state_dict(additional_state['optimizer'])
            print("Optimizer state loaded")
        
        # Load scheduler state
        if scheduler and 'scheduler' in additional_state and additional_state['scheduler']:
            scheduler.load_state_dict(additional_state['scheduler'])
            print("Scheduler state loaded")
        
        # Load scaler state
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
    
    print(f"Resume info: Epoch {resume_info['epoch']}, Step {resume_info['global_step']}")
    print(f"Best metrics: Val Loss {resume_info['best_val_loss']:.4f}, Perplexity {resume_info['best_perplexity']:.2f}")
    
    return resume_info

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find the latest SafeTensors checkpoint file in the directory
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_') and file.endswith('.safetensors'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time, get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

def load_and_preprocess_data(max_samples=100000):
    """Load and preprocess Turkish Wikipedia data"""
    def clean_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    dataset = load_dataset("musabg/wikipedia-tr-summarization", split='train')
    processed_texts = []
    
    for i in tqdm(range(min(len(dataset), max_samples)), desc="Preprocessing data"):
        text = clean_text(dataset[i]["text"])
        if len(text) > 50: 
            processed_texts.append(text)
    
    return processed_texts

def generate(text, 
             model_path="checkpoints/best_model_100m.safetensors",
             tokenizer_path="turkish_tokenizer/turkish_tokenizer.model",
             max_new_tokens=100,
             temperature=0.8,
             top_p=0.9,
             top_k=50,
             device=None):

    # Device setup
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors not available. Install with: pip install safetensors")
    
    tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Ensure .safetensors extension
    if not model_path.endswith('.safetensors'):
        model_path = model_path.replace('.pt', '.safetensors')
    
    # Load model metadata from SafeTensors
    from safetensors import safe_open
    metadata = {}
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    
    config = json.loads(metadata.get('config', '{}'))
    
    # Load model
    model = Transformers(config, tokenizer).to(device)
    
    # Load model weights from SafeTensors
    model_state = load_file(model_path)
    
    # Handle weight tying restoration
    if metadata.get('weight_tying') == 'true' and 'lm_head.weight' not in model_state and 'wte.weight' in model_state:
        # Restore weight tying by copying wte.weight to lm_head.weight
        print("🔗 Restoring weight tying for lm_head.weight")
        model_state['lm_head.weight'] = model_state['wte.weight']
    
    model.load_state_dict(model_state)
    print(f"Model weights loaded from SafeTensors: {model_path}")
    
    model.eval()
    
    print(f"Model parameters: {model.get_num_params()/1e6:.1f}M")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    
    with torch.no_grad():
        generated_text = model.generate_from_prompt(
            text, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
    
    print(f"Generated text:\n{generated_text}")
    print("="*60)
    
    return generated_text

if __name__ == "__main__":
    # Fresh training
    model = train()

    # Auto-resume (en son checkpoint'ten devam)
    #model = train(auto_resume=True)

    # Specific checkpoint'ten devam
    #model = train(resume_from_checkpoint="checkpoints/checkpoint_step_1.safetensors")

    # Model generate (SafeTensors)
    #generate("Türkiye'nin başkenti", model_path="checkpoints/checkpoint_step_1.safetensors")