MODEL_CONFIG = {
    'n_embd': 768,          # 768 embedding dimension
    'n_layer': 14,          # 14 transformer layer (~120M parameters)
    'n_head': 12,           # 12 attention head (768 ÷ 12 = 64 head_dim)
    'block_size': 1024,     # 1024 context window
    'dropout': 0.1,         # Dropout
    'vocab_size': None,     # tokenizer'dan alınacak
    'use_flash_attention': True, 
    'use_gradient_checkpointing': True,  
    'use_selective_checkpointing': True, 
}

TRAINING_CONFIG = {
    'batch_size': 4,       
    'learning_rate': 6e-4,  
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