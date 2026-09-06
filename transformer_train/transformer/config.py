MODEL_CONFIG = {
    'n_embd': 1024,
    'n_layer': 20,
    'n_head': 16,
    'n_kv_head': 8,
    'block_size': 1024,
    'vocab_size': None,
    'window_pattern': 'L',
    'recurrent': {
        'enabled': True,
        'n_prelude': 6,
        'n_recurrent': 8,
        'n_coda': 6,
        'r_default': 4,
        'r_mean': 5.0,
        'r_sigma': 0.5,
        'r_min': 1,
        'r_max': 16,
        'backprop_depth': 8,
        'state_init': 'random',
        'state_init_std': 0.02,
        'adapter': True,
    },
}

TRAINING_CONFIG = {
    'training_stage': 'base',

    'batch_size': 8,
    'accumulation_steps': 8,
    'grad_clip': 1.0,

    'use_muon_optimizer': True,
    'unembedding_lr': 0.004,
    'embedding_lr': 0.2,
    'matrix_lr': 0.02,
    'scalar_lr': 0.5,
    'weight_decay': 0.0,
    'adam_betas': (0.8, 0.95),

    'max_epochs': 5,
    'warmup_epochs': 0.2,
    'tokens_per_epoch': 1_000_000_000,

    'use_mixed_precision': True,

    'log_interval': 50,
    'eval_steps': 2000,
    'checkpoint_steps': 2000,
    'use_wandb': True,

    'shuffle_parquet_files': True,
    'shuffle_seed': 42,
    'reshuffle_each_epoch': True,

    'early_stopping_patience': 8,
    'early_stopping_min_delta': 0.001,
}

TEST_PROMPTS = [
    "Türkiye'nin başkenti",
    "Yapay zeka teknolojisi",
    "İstanbul Boğazı"
]

SHOW_SPECIAL_TOKENS = True


TRAINING_STAGE_DATASET_MAP = {
    'base': 'BASE_DATASET_CONFIG',
    'mid': 'MID_DATASET_CONFIG',
    'sft': 'SFT_DATASET_CONFIG',
}

BASE_DATASET_CONFIG = {
    'type': 'parquet',
    'data_dir': 'dataset/base_data',
    'text_column': 'text',
}

MID_DATASET_CONFIG = {
    'datasets': [
        {
            'type': 'wikipedia', 
            'data_dir': 'dataset/mid_data/wikipedia',
            'text_column': 'text',
            'max_samples': None
        },
        {
            'type': 'news', 
            'data_dir': 'dataset/mid_data/news',
            'text_column': 'text',
            'max_samples': None
        },
        {
            'type': 'qa', 
            'data_dir': 'dataset/mid_data/qa',
            'question_column': 'question',
            'answer_column': 'answer',
            'max_samples': None
        },
        {
            'type': 'math', 
            'data_dir': 'dataset/mid_data/math',
            'question_column': 'soru',
            'answer_column': 'solution',
            'max_samples': None
        },
    ]
}

SFT_DATASET_CONFIG = {
    'datasets': [
        {
            'type': 'chat', 
            'data_dir': 'sft_data',
            'messages_column': 'messages',
            'max_samples': None
        },
    ]
}


VALID_TRAINING_STAGES = ['base', 'mid', 'sft']

def validate_config():
    stage = TRAINING_CONFIG.get('training_stage', 'base')

    if stage not in VALID_TRAINING_STAGES:
        raise ValueError(
            f"Invalid training_stage: '{stage}'. "
            f"Must be one of: {VALID_TRAINING_STAGES}"
        )

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
