MODEL_CONFIG = {
    'n_embd': 768,          # 768 embedding dimension
    'n_layer': 14,          # 14 transformer layer (~120M parameters)
    'n_head': 12,           # 12 attention head (768 ÷ 12 = 64 head_dim)
    'n_kv_head': 12,        # number of key/value heads (GQA: can be less than n_head for efficiency)
    'block_size': 1024,     # 1024 context window
    'dropout': 0.1,         # Dropout (not used, kept for compatibility)
    'vocab_size': None,     # tokenizer'dan alınacak
    'use_flash_attention': True,  # Not used (uses PyTorch's scaled_dot_product_attention)
    'use_gradient_checkpointing': True,  
    'use_selective_checkpointing': True, 
}

TRAINING_CONFIG = {
    # Training stage: 'base' (pretraining), 'mid' (mid-training), 'sft' (chat fine-tuning)
    'training_stage': 'base',  # Config'den hangi stage'in hangi dataset ile olacağını belirleyin
    
    'batch_size': 4,       
    'learning_rate': 6e-4,  # Legacy: used if use_muon_optimizer is False
    'weight_decay': 0.1,   
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'warmup_epochs': 3,     
    'max_epochs': 50,       
    'eval_interval': 2,     
    'save_interval': 5,     
    'accumulation_steps': 8,  
    'use_wandb': True,
    'compile_model': False,
    'scheduler_type': 'cosine_with_warmup',
    'eval_generation_samples': 3, 
    'max_eval_batches': 50,
    
    # Muon optimizer settings
    'use_muon_optimizer': True,  # Use Muon + AdamW with separate learning rates
    'unembedding_lr': 0.004,  # Learning rate for lm_head
    'embedding_lr': 0.2,  # Learning rate for token embeddings
    'matrix_lr': 0.02,  # Learning rate for transformer matrix parameters (Muon) 
    
    'use_cpu_offload': False,     
    'use_activation_checkpointing': True,
    'use_mixed_precision': True, 
    'dataloader_num_workers': 2, 
    'pin_memory': True,         
    'prefetch_factor': 2,        
    
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
    
    # DeepSpeed 8GB VRAM Optimization
    'use_deepspeed': False,         
    'deepspeed_config_path': 'transformers_train/deepspeed_config/deepspeed_config.json',
    'zero_stage': 2,                 # ZeRO Stage 2 
    'cpu_offload': False,            
    'nvme_offload': False,           # NVMe offload (SSD gerekli)
    'allgather_bucket_size': 5e8,    # Memory optimization
    'reduce_bucket_size': 5e8,       # Memory optimization
}

TEST_PROMPTS = [
    "Türkiye'nin başkenti",
    "Yapay zeka teknolojisi",
    "İstanbul Boğazı"
]

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
    'type': 'parquet',  # Parquet files from lumees/turkish-corpus-100b
    'data_dir': 'base_data',  # Directory containing parquet files
    'text_column': 'text',  # Column name in parquet files
    'max_samples': None,  # None = use all
}

# Mid Training Dataset Config (Structured tasks)
# Kullanım: training_stage='mid' olduğunda bu config kullanılır
# Türkçe dataset'ler: Wikipedia, news, QA, math (all from parquet files)
MID_DATASET_CONFIG = {
    'datasets': [
        {
            'type': 'wikipedia', 
            'data_dir': 'mid_data/wikipedia',  # Parquet files directory
            'text_column': 'text',
            'max_samples': None  # None = use all
        },
        {
            'type': 'news', 
            'data_dir': 'mid_data/news',  # Parquet files directory
            'text_column': 'text',
            'max_samples': None  # None = use all
        },
        {
            'type': 'qa', 
            'data_dir': 'mid_data/qa',  # Parquet files directory
            'question_column': 'question',
            'answer_column': 'answer',
            'max_samples': None  # None = use all
        },
        {
            'type': 'math', 
            'data_dir': 'mid_data/math',  # Parquet files directory
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