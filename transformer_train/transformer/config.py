MODEL_CONFIG = {
    'n_embd': 896,          # 768 embedding dimension
    'n_layer': 18,          # 14 transformer layer (~120M parameters)
    'n_head': 14,           # 12 attention head (768 ÷ 12 = 64 head_dim)
    'n_kv_head': 7,        # number of key/value heads (GQA: can be less than n_head for efficiency)
    'block_size': 1024,     # 1024 context window
    'vocab_size': None,     # tokenizer'dan alınacak
}

TRAINING_CONFIG = {
    # Training stage: 'base' (pretraining), 'mid' (mid-training), 'sft' (chat fine-tuning)
    'training_stage': 'base',
    
    'batch_size': 2,       
    'learning_rate': 6e-4,  # Used if use_muon_optimizer is False
    'weight_decay': 0.08,   
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 0.6,
    'warmup_epochs': 0.2,  # Warmup for 0.05 epoch (~3,800 steps) - appropriate for 140M model
    'max_epochs': 20,       # 20 epochs = 100B tokens total (full dataset sweep)
    'eval_interval': 1,     # Full evaluation every epoch     
    'save_interval': 5,     
    'accumulation_steps': 32,  
    'use_wandb': True,
    'eval_generation_samples': 3, 
    'max_eval_batches': 25,
    
    # Muon optimizer settings
    'use_muon_optimizer': True,  # Use Muon + AdamW with separate learning rates
    'unembedding_lr': 0.002,  # Learning rate for lm_head (slightly increased for better learning)
    'embedding_lr': 0.03,  # Learning rate for token embeddings (balanced for 140M model)
    'matrix_lr': 0.010,  # Learning rate for transformer matrix parameters (Muon) (balanced for 140M model) 
    
    'use_mixed_precision': True, 
    'dataloader_num_workers': 2, 
    'pin_memory': True,         
    'prefetch_factor': 2,        

    # Token accounting (streaming datasets)
    # 1 epoch = 5B tokens (~5000 steps with batch_size=1, accumulation=16, block_size=1024)
    # This makes epochs more manageable for tracking progress
    'tokens_per_epoch': 2_500_000_000,  # 5B tokens per epoch
    
    # Progress reporting
    'log_interval': 50,     # Log every 50 steps
    'eval_steps': 2000,     # Evaluate + generate samples every 1000 steps
    'checkpoint_steps': 2000,  # Save checkpoint every 500 steps  
    
    # Data shuffling for better training
    'shuffle_parquet_files': True,  # Shuffle parquet file order each epoch
    'shuffle_seed': 42,  # Seed for reproducibility (None = random each run)
    'reshuffle_each_epoch': True,  # Re-shuffle parquet order every epoch
    
    # Early stopping
    'early_stopping_patience': 8,  
    'early_stopping_min_delta': 0.001,
    
    'vocab_size': 32000,
    'max_data_samples': 150000,  
    
    # DeepSpeed Optimization
    'use_deepspeed': False,         
    'deepspeed_config_path': 'transformer_train/deepspeed_config/deepspeed_config.json',
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
    'data_dir': 'dataset/base_data',  # Directory containing parquet files
    'text_column': 'text',  # Column name in parquet files
    'max_samples': None,  # None = use all
    'tokens_per_epoch': 5_000_000_000,  # 5B tokens per epoch (manageable size for progress tracking)
    # Note: Full dataset is ~100B tokens, so ~20 epochs to see all data
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