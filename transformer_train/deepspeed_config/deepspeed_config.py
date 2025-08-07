
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
        
        # Memory optimization for low VRAM
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
    
    # ZeRO Stage 3 için ek ayarlar (çok aggressive memory saving yapar)
    if training_config.get('zero_stage', 2) == 3:
        config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        })
    
    # NVMe offload
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