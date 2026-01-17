MODEL_CONFIG = {
    'n_embd': 1280,         # embedding dimension (1280 for ~700M params)
    'n_layer': 32,          # transformer layers (32 deep)
    'n_head': 20,           # attention heads (1280 ÷ 20 = 64 head_dim)
    'n_kv_head': 10,        # GQA: 10 key/value heads (2x compression)
    'block_size': 1024,     # context window (optimal for Turkish)
    'vocab_size': None,     # will be set from tokenizer
    'window_pattern': 'L',  # sliding window: L=full context, S=half, "SSL"=pattern
}

TRAINING_CONFIG = {
    # Training stage: 'base' (pretraining), 'mid' (mid-training), 'sft' (chat fine-tuning)
    'training_stage': 'base',
    
    # Batch configuration (RTX 3090 24GB optimized)
    'batch_size': 8,                # 8 sequences per batch
    'accumulation_steps': 8,        # Effective batch = 64 (8 * 8)
    'grad_clip': 1.0,               # Gradient clipping
    
    # Muon + AdamW optimizer (nanochat-compatible)
    'use_muon_optimizer': True,
    'unembedding_lr': 0.004,        # lm_head learning rate
    'embedding_lr': 0.2,            # token embedding learning rate
    'matrix_lr': 0.02,              # transformer matrix (Muon) learning rate
    'scalar_lr': 0.5,               # resid_lambdas & x0_lambdas learning rate
    'weight_decay': 0.0,            # weight decay (only for Muon, not AdamW)
    'adam_betas': (0.8, 0.95),      # AdamW betas (nanochat default)
    
    # Training schedule
    'max_epochs': 20,               # 20 epochs through dataset
    'warmup_epochs': 0.2,           # Warmup for 0.2 epochs
    'tokens_per_epoch': 2_500_000_000,  # 2.5B tokens per epoch
    
    # Mixed precision
    'use_mixed_precision': True,    # bfloat16 training
    
    # Logging and evaluation
    'log_interval': 50,             # Log every 50 steps
    'eval_steps': 2000,             # Evaluate every 2000 steps
    'checkpoint_steps': 2000,       # Save checkpoint every 2000 steps
    'use_wandb': True,              # Weights & Biases logging
    
    # Data configuration
    'shuffle_parquet_files': True,
    'shuffle_seed': 42,
    'reshuffle_each_epoch': True,
    
    # Early stopping
    'early_stopping_patience': 8,
    'early_stopping_min_delta': 0.001,
}

TEST_PROMPTS = [
    "Türkiye'nin başkenti",
    "Yapay zeka teknolojisi",
    "İstanbul Boğazı"
]

# Show special tokens in generation outputs (for debugging)
SHOW_SPECIAL_TOKENS = True  # Set to False to hide <bos>, <eos> tokens

# -----------------------------------------------------------------------------
# Dataset Configurations
# -----------------------------------------------------------------------------

# Training Stage -> Dataset Mapping
# Her training stage için hangi dataset config'inin kullanılacağını belirleyin
TRAINING_STAGE_DATASET_MAP = {
    'base': 'BASE_DATASET_CONFIG',  # Pretraining: parquet streaming
    'mid': 'MID_DATASET_CONFIG',    # Mid-training: structured tasks (Wikipedia, news, QA, math)
    'sft': 'SFT_DATASET_CONFIG',    # SFT: conversation format (chat)
}

# Base Training (Pretraining) Dataset Config
# Kullanım: training_stage='base' olduğunda bu config kullanılır
BASE_DATASET_CONFIG = {
    'type': 'parquet',
    'data_dir': 'dataset/base_data',
    'text_column': 'text',
    # Note: Full dataset ~100B tokens, tokens_per_epoch from TRAINING_CONFIG
}

# Mid Training Dataset Config (Structured tasks)
# Kullanım: training_stage='mid' olduğunda bu config kullanılır
# Türkçe dataset'ler: Wikipedia, news, QA, math (all from parquet files)
MID_DATASET_CONFIG = {
    'datasets': [
        {
            'type': 'wikipedia', 
            'data_dir': 'dataset/mid_data/wikipedia',  # Parquet files directory
            'text_column': 'text',
            'max_samples': None  # None = use all
        },
        {
            'type': 'news', 
            'data_dir': 'dataset/mid_data/news',  # Parquet files directory
            'text_column': 'text',
            'max_samples': None  # None = use all
        },
        {
            'type': 'qa', 
            'data_dir': 'dataset/mid_data/qa',  # Parquet files directory
            'question_column': 'question',
            'answer_column': 'answer',
            'max_samples': None  # None = use all
        },
        {
            'type': 'math', 
            'data_dir': 'dataset/mid_data/math',  # Parquet files directory
            'question_column': 'soru',
            'answer_column': 'solution',
            'max_samples': None  # None = use all
        },
    ]
}

# SFT (Chat) Dataset Config
# Kullanım: training_stage='sft' olduğunda bu config kullanılır
# Türkçe conversation dataset'leri (parquet files)
SFT_DATASET_CONFIG = {
    'datasets': [
        {
            'type': 'chat', 
            'data_dir': 'sft_data',  # Parquet files directory
            'messages_column': 'messages',  # Column name in parquet files
            'max_samples': None  # None = use all
        },
    ]
}

# -----------------------------------------------------------------------------
# Config Validation
# -----------------------------------------------------------------------------

VALID_TRAINING_STAGES = ['base', 'mid', 'sft']

def validate_config():
    stage = TRAINING_CONFIG.get('training_stage', 'base')
    
    # Validate training stage
    if stage not in VALID_TRAINING_STAGES:
        raise ValueError(
            f"Invalid training_stage: '{stage}'. "
            f"Must be one of: {VALID_TRAINING_STAGES}"
        )
    
    # Validate dataset config exists
    if stage not in TRAINING_STAGE_DATASET_MAP:
        raise ValueError(
            f"Training stage '{stage}' not found in TRAINING_STAGE_DATASET_MAP"
        )
    
    dataset_config_name = TRAINING_STAGE_DATASET_MAP[stage]
    if dataset_config_name not in globals():
        raise ValueError(
            f"Dataset config not found: {dataset_config_name}"
        )
    
    print(f"Config validation passed: training_stage='{stage}', dataset_config='{dataset_config_name}'")