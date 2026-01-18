import os
import math
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import wandb
from contextlib import redirect_stdout, redirect_stderr

from ..utils import EarlyStopping, cleanup_memory, evaluate_model_comprehensive, get_cosine_schedule_with_warmup, get_gpu_memory_info, get_memory_usage, get_memory_usage_safe
from ..config import (
    MODEL_CONFIG, TRAINING_CONFIG, TEST_PROMPTS, SHOW_SPECIAL_TOKENS,
    BASE_DATASET_CONFIG, MID_DATASET_CONFIG, SFT_DATASET_CONFIG,
    TRAINING_STAGE_DATASET_MAP, validate_config
)
from ..tokenizer import create_tokenizer, get_token_bytes
from ..data.dataset import TransformerDataset, load_and_preprocess_data
from ..data.sft_dataset import SFTDataset
from ..data.dataloader import tokenizing_distributed_data_loader_with_state, tokenizing_distributed_data_loader
from ..data.dataset_utils import create_mid_datasets, create_sft_datasets
from ..model.transformer_block import Transformer

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

def train(
    tokenizer_path="tokenizer",  # Default to tokenizer directory (RustBPE)
    resume_from_checkpoint=None,
    auto_resume=False,
    use_wandb=True,
    project_name="turkish-transformer-120m",
    pretrained_model_path=None,
    fresh_epochs=None
):
    
    print("Transformer train")
    print("="*80)
    
    # Validate configuration
    validate_config()
    
    # Initialize training state.
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    
    # CUDA Memory Optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = "cuda"
        
        gpu_info = get_gpu_memory_info()
        print(f"Device: {device} ({gpu_info.get('device_name', 'Unknown')})")
        print(f"VRAM: {gpu_info.get('total_gb', 0):.1f}GB total")
        
        torch.cuda.empty_cache()
        cleanup_memory()
        print("Initial memory cleaned")    
        print(f"Memory after cleanup: {get_memory_usage()}")
    else:
        device = torch.device('cpu')
        device_type = "cpu"
        print(f"CUDA not available, using CPU")
    
    print(f"Flash Attention: {'Available' if FLASH_ATTENTION_AVAILABLE else 'Not Available'}")
    
    # Mixed precision settings
    use_mixed_precision = device_type == 'cuda' and TRAINING_CONFIG.get('use_mixed_precision', True)
    use_bfloat16 = use_mixed_precision and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    
    if use_wandb and TRAINING_CONFIG['use_wandb']:
        wandb.init(
            project=project_name,
            config={**MODEL_CONFIG, **TRAINING_CONFIG},
            resume="allow" if resume_from_checkpoint else None
        )
    

    print(f"\nLoading tokenizer from: {tokenizer_path}")
    # Try to load as RustBPE tokenizer first
    if os.path.isdir(tokenizer_path):
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
    else:
        # Legacy support for SentencePiece
        tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Update config
    MODEL_CONFIG['vocab_size'] = tokenizer.vocab_size
    
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    
    print("\nTesting tokenizer:")
    test_text = "Türkiye'nin başkenti Ankara'dır. İstanbul Boğazı çok güzel."
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    if isinstance(tokens[0], list):
        tokens = tokens[0]
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Original: {test_text}")
    print(f"Tokens: {len(tokens)} tokens")
    print(f"Decoded: {decoded}")
    print(f"Match: {'OK' if test_text.strip() == decoded.strip() else 'NO'}")
    
    # Check training stage and select appropriate dataset config
    training_stage = TRAINING_CONFIG.get('training_stage', 'base')
    print(f"\nTraining stage: {training_stage}")
    
    # Get dataset config name from mapping
    if training_stage not in TRAINING_STAGE_DATASET_MAP:
        raise ValueError(
            f"Unknown training stage: {training_stage}. "
            f"Must be one of: {list(TRAINING_STAGE_DATASET_MAP.keys())}"
        )
    
    dataset_config_name = TRAINING_STAGE_DATASET_MAP[training_stage]
    print(f"Using dataset config: {dataset_config_name}")
    
    # Map config name to actual config
    dataset_config_map = {
        'BASE_DATASET_CONFIG': BASE_DATASET_CONFIG,
        'MID_DATASET_CONFIG': MID_DATASET_CONFIG,
        'SFT_DATASET_CONFIG': SFT_DATASET_CONFIG,
    }
    
    dataset_config = dataset_config_map[dataset_config_name].copy()
    print(f"Dataset config loaded: {dataset_config_name}")
    
    dataset_type = dataset_config.get('type', 'parquet')
    data_dir = dataset_config.get('data_dir', 'base_data')
    
    # Use parquet streaming for base training, otherwise use structured datasets
    use_parquet_streaming = (training_stage == 'base' and dataset_type == 'parquet')
    
    # Initialize dataloader_state_dict variable for later use
    dataloader_state_dict = None
    
    if use_parquet_streaming:
        print(f"Using parquet streaming dataloader from: {data_dir}")
        try:
            # Use streaming dataloader
            dataloader_resume_state_dict = None  # Will be set from checkpoint if resuming
            
            train_loader = tokenizing_distributed_data_loader_with_state(
                B=TRAINING_CONFIG['batch_size'],
                T=MODEL_CONFIG['block_size'],
                split="train",
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                device=str(device),
                resume_state_dict=dataloader_resume_state_dict,
                data_dir=data_dir,
                shuffle_parquet_files=TRAINING_CONFIG.get('shuffle_parquet_files', True),
                shuffle_seed=TRAINING_CONFIG.get('shuffle_seed', 42),
                reshuffle_each_epoch=TRAINING_CONFIG.get('reshuffle_each_epoch', True)
            )
            
            build_val_loader = lambda: tokenizing_distributed_data_loader(
                B=TRAINING_CONFIG['batch_size'],
                T=MODEL_CONFIG['block_size'],
                split="val",
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                device=str(device),
                data_dir=data_dir
            )
            
            # Kick off first batch
            x, y, dataloader_state_dict = next(train_loader)
            print(f"Parquet streaming dataloader initialized successfully!")
            print(f"Batch shape: {x.shape}, {y.shape}")
            
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Parquet streaming failed ({e}), falling back to HuggingFace dataset")
            use_parquet_streaming = False
    
    if not use_parquet_streaming:
        # Handle mid-training or SFT datasets
        if training_stage == 'mid':
            print(f"\nLoading mid-training datasets...")
            task_mixture = create_mid_datasets(dataset_config)
            
            # Convert to text format and tokenize
            full_corpus = []
            for text in task_mixture:
                full_corpus.append(text)
                if TRAINING_CONFIG['max_data_samples'] and len(full_corpus) >= TRAINING_CONFIG['max_data_samples']:
                    break
            
            print(f"Loaded {len(full_corpus):,} text samples from mid-training datasets")
            
        elif training_stage == 'sft':
            print(f"\nLoading SFT (chat) datasets...")
            sft_datasets = create_sft_datasets(dataset_config)
            
            # Collect conversations (not text, keep as conversation dicts)
            conversations = []
            for dataset in sft_datasets:
                for conv in dataset:
                    if isinstance(conv, dict) and 'messages' in conv:
                        conversations.append(conv)
                    
                    if TRAINING_CONFIG['max_data_samples'] and len(conversations) >= TRAINING_CONFIG['max_data_samples']:
                        break
                if TRAINING_CONFIG['max_data_samples'] and len(conversations) >= TRAINING_CONFIG['max_data_samples']:
                    break
            
            print(f"Loaded {len(conversations):,} conversation samples from SFT datasets")
            
        else:
            # Base training fallback (should not happen if parquet files exist)
            print(f"\nWarning: Parquet streaming not available, but base training requires parquet files.")
            print(f"Please ensure parquet files are in: {data_dir}")
            raise FileNotFoundError(f"Parquet files not found in {data_dir}. Base training requires parquet files.")
        
        # Different handling for SFT vs Mid training
        if training_stage == 'sft':
            # SFT: Use SFTDataset with loss masking
            print("Creating SFT datasets with loss masking...")
            split_idx = int(0.9 * len(conversations))
            train_conversations = conversations[:split_idx]
            val_conversations = conversations[split_idx:]
            
            print(f"Train conversations: {len(train_conversations):,}")
            print(f"Val conversations: {len(val_conversations):,}")
            
            train_dataset = SFTDataset(train_conversations, tokenizer, MODEL_CONFIG['block_size'])
            val_dataset = SFTDataset(val_conversations, tokenizer, MODEL_CONFIG['block_size'])
            
            del conversations
            cleanup_memory()
        else:
            # Mid training: Use standard TransformerDataset
            print("Tokenizing data...")
            all_tokens = []
            for text in tqdm(full_corpus, desc="Encoding texts"):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                    tokens = tokens[0]
                all_tokens.extend(tokens)
                
                if len(all_tokens) % 1000000 == 0:
                    cleanup_memory()
            
            # Split data
            split_idx = int(0.9 * len(all_tokens))
            train_tokens = all_tokens[:split_idx]
            val_tokens = all_tokens[split_idx:]
            
            print(f"Total tokens: {len(all_tokens):,}")
            print(f"Train tokens: {len(train_tokens):,}")
            print(f"Val tokens: {len(val_tokens):,}")
            
            del all_tokens, full_corpus
            cleanup_memory()
            
            train_dataset = TransformerDataset(train_tokens, MODEL_CONFIG['block_size'])
            val_dataset = TransformerDataset(val_tokens, MODEL_CONFIG['block_size'])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=TRAINING_CONFIG['batch_size'],
            shuffle=True,
            num_workers=TRAINING_CONFIG['dataloader_num_workers'],
            pin_memory=TRAINING_CONFIG['pin_memory'],
            prefetch_factor=TRAINING_CONFIG['prefetch_factor'],
            drop_last=True
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
        
        build_val_loader = lambda: val_loader
        dataloader_state_dict = None
    
    effective_batch_size = TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['accumulation_steps']
    tokens_per_step = effective_batch_size * MODEL_CONFIG['block_size']
    
    # Derive steps_per_epoch for both streaming and sized dataloaders
    tokens_per_epoch = TRAINING_CONFIG.get('tokens_per_epoch')
    if use_parquet_streaming:
        tokens_per_epoch = dataset_config.get('tokens_per_epoch', tokens_per_epoch)
        if tokens_per_epoch is None:
            raise ValueError("tokens_per_epoch must be set in TRAINING_CONFIG or dataset config for streaming dataloader.")
        steps_per_epoch = math.ceil(tokens_per_epoch / tokens_per_step)
    else:
        steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
    
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Planned steps per epoch: {steps_per_epoch:,}")
    
    # Initialize model
    print(f"\nInitializing Transformer...")
    model = Transformer(MODEL_CONFIG, tokenizer).to(device)
    
    print(f"Model parameters: {model.get_num_params()/1e6:.1f}M")
    print(f"Memory after model load: {get_memory_usage()}")
    
    # Store original model for checkpointing and evaluation
    orig_model = model
    
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
    
    # Compile model for better performance (before optimizer setup)
    if device_type == 'cuda':
        print("\nCompiling model with torch.compile...")
        model = torch.compile(model, dynamic=False)
        print("Model compiled successfully!")
    
    # Precompute token bytes for BPB (bits per byte) calculation
    print("\nPrecomputing token bytes for BPB calculation...")
    token_bytes = get_token_bytes(tokenizer, device=device)
    avg_bytes_per_token = token_bytes[token_bytes > 0].float().mean().item()
    print(f"Average bytes per token: {avg_bytes_per_token:.2f}")
    
    print(f"\nUsing standard PyTorch training...")
    
    # Use Muon optimizer if enabled, otherwise use standard AdamW
    use_muon = TRAINING_CONFIG.get('use_muon_optimizer', False)
    
    if use_muon:
        print("Using Muon optimizer with separate learning rates")
        
        # Apply LR scaling based on model dimension
        dmodel_lr_scale = (MODEL_CONFIG['n_embd'] / 768) ** -0.5
        
        optimizers = orig_model.setup_optimizers(
            unembedding_lr=TRAINING_CONFIG.get('unembedding_lr', 0.004) * dmodel_lr_scale,
            embedding_lr=TRAINING_CONFIG.get('embedding_lr', 0.2) * dmodel_lr_scale,
            matrix_lr=TRAINING_CONFIG.get('matrix_lr', 0.02) * dmodel_lr_scale,
            weight_decay=TRAINING_CONFIG.get('weight_decay', 0.0),
            adam_betas=(TRAINING_CONFIG.get('beta1', 0.9), TRAINING_CONFIG.get('beta2', 0.95))
        )
        print(f"  LR scale factor (based on d_model={MODEL_CONFIG['n_embd']}): {dmodel_lr_scale:.4f}")
        # For compatibility, we'll use the first optimizer (AdamW) for scheduler
        # In practice, you might want separate schedulers for each optimizer
        optimizer = optimizers[0]  # AdamW optimizer
        muon_optimizer = optimizers[1]  # Muon optimizer
        print(f"  AdamW optimizer: {len(optimizer.param_groups)} param groups")
        print(f"  Muon optimizer: {len(muon_optimizer.param_groups)} param groups")
    else:
        optimizer = torch.optim.AdamW(
            orig_model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            betas=(TRAINING_CONFIG['beta1'], TRAINING_CONFIG['beta2']),
            eps=1e-8,
            fused=True if device_type == 'cuda' else False
        )
        muon_optimizer = None
    
    # Scheduler
    total_steps = steps_per_epoch * TRAINING_CONFIG['max_epochs']
    warmup_steps = min(
        total_steps,
        max(1, int(steps_per_epoch * TRAINING_CONFIG['warmup_epochs']))
    )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Scheduler: Cosine with warmup ({warmup_steps:,} warmup steps)")
    
    # Create scheduler for Muon optimizer if using Muon
    muon_scheduler = None
    if use_muon and muon_optimizer is not None:
        # Create a separate scheduler for Muon optimizer with same schedule
        muon_scheduler = get_cosine_schedule_with_warmup(muon_optimizer, warmup_steps, total_steps)
        print(f"Muon scheduler: Cosine with warmup ({warmup_steps:,} warmup steps)")
    
    # Muon momentum scheduler function
    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum
    
    # Mixed precision scaler
    use_grad_scaler = use_mixed_precision and not use_bfloat16
    scaler = GradScaler(device_type) if use_grad_scaler else None
    print(f"Mixed precision: {'bf16' if use_bfloat16 else 'fp16' if use_mixed_precision else 'disabled'}"
          f" ({'GradScaler' if scaler else 'no scaler'})")
    
        # Resume logic (skip for pretrained)
    if not pretrained_model_path:
        if auto_resume and not resume_from_checkpoint:
            resume_from_checkpoint = find_latest_checkpoint()
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            try:
                print(f"\nResuming from: {resume_from_checkpoint}")
                
                resume_info = load_checkpoint(
                    resume_from_checkpoint, model, optimizer, scheduler, scaler, device,
                    muon_optimizer=muon_optimizer if 'muon_optimizer' in locals() else None,
                    muon_scheduler=muon_scheduler if 'muon_scheduler' in locals() else None
                )
                loaded_epoch = resume_info['epoch']
                global_step = resume_info['global_step']
                best_val_loss = resume_info['best_val_loss']
                
                calculated_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
                
                print(f"RESUME ANALYSIS:")
                print(f"  Loaded epoch: {loaded_epoch}")
                print(f"  Global step: {global_step}")
                print(f"  Steps per epoch: {steps_per_epoch}")
                print(f"  Calculated epoch from global_step: {calculated_epoch}")
                
                # Use the maximum of loaded_epoch and calculated_epoch to be safe
                start_epoch = max(loaded_epoch, calculated_epoch)
                
                print(f"Using epoch: {start_epoch}")
                print(f"This epoch has completed {global_step % steps_per_epoch if steps_per_epoch > 0 else 0} steps.")
                
                print(f"Resume successful: Epoch {start_epoch}, Step {global_step}, Best Val Loss: {best_val_loss:.4f}")
                print(f"Estimated steps per epoch: {steps_per_epoch}")
                
                # If using parquet streaming, recreate dataloader with resume state
                if use_parquet_streaming and resume_info.get('dataloader_state_dict'):
                    dl_state = resume_info['dataloader_state_dict']
                    pq_idx = dl_state.get('pq_idx', 0)
                    rg_idx = dl_state.get('rg_idx', 0)
                    print(f"Resuming dataloader from checkpoint state")
                    print(f"  Parquet Index: {pq_idx}, Row Group: {rg_idx}")
                    train_loader = tokenizing_distributed_data_loader_with_state(
                        B=TRAINING_CONFIG['batch_size'],
                        T=MODEL_CONFIG['block_size'],
                        split="train",
                        tokenizer=tokenizer,
                        tokenizer_path=tokenizer_path,
                        device=str(device),
                        resume_state_dict=resume_info['dataloader_state_dict'],
                        data_dir=data_dir
                    )
                    # Kick off first batch
                    x, y, dataloader_state_dict = next(train_loader)
                    print(f"Dataloader resumed from checkpoint!")
                elif not use_parquet_streaming:
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
    
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['early_stopping_patience'],
        min_delta=TRAINING_CONFIG['early_stopping_min_delta'],
        restore_best_weights=True
    )
    
    print(f"\nStarting training from epoch {start_epoch + 1}")
    print(f"Max epochs: {TRAINING_CONFIG['max_epochs']}")
    print(f"Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"Global step: {global_step}")
    print("="*80)
    
    os.makedirs("checkpoints", exist_ok=True)
    print("Checkpoint directory created: checkpoints/")
    
    for epoch in range(start_epoch, TRAINING_CONFIG['max_epochs']):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{TRAINING_CONFIG['max_epochs']}")
        print(f"{'='*80}")
        print(f"Memory before epoch: {get_memory_usage()}")
        print(f"Global step: {global_step}")
        
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        

        if(global_step > 0):
            print(f"Resume info: Starting from saved epoch {epoch + 1}, global step {global_step}")
        
        if use_parquet_streaming:
            # Streaming dataloader: infinite iterator, iteration-based
            max_iterations = TRAINING_CONFIG.get('max_iterations', None)
            if max_iterations is None:
                # Exact iteration budget derived from tokens_per_epoch
                max_iterations = steps_per_epoch * TRAINING_CONFIG['accumulation_steps']
            
            print(f"  Using iteration-based training (streaming)")
            print(f"  Max iterations (microbatches): {max_iterations:,}")
            print(f"  Optimizer steps per epoch: {steps_per_epoch:,}")
            print(f"  Accumulation steps: {TRAINING_CONFIG['accumulation_steps']}")
            
            # Get parquet file list for progress tracking
            from ..data.dataset_utils import list_parquet_files
            parquet_files = list_parquet_files(data_dir)
            total_parquets = len(parquet_files[:-1])  # Exclude validation file
            print(f"  Total training parquet files: {total_parquets}")
            
            batch_counter = 0
            iteration = 0
            current_parquet_info = ""
            epoch_start_step = global_step  # Track where this epoch started
            for batch_idx in range(max_iterations):
                try:
                    batch_data = next(train_loader)
                    if isinstance(batch_data, tuple) and len(batch_data) == 3:
                        inputs, targets, dataloader_state_dict = batch_data
                    else:
                        inputs, targets = batch_data
                        dataloader_state_dict = None
                except StopIteration:
                    break
                
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                iteration += 1
                
                # Forward pass with mixed precision
                with autocast(
                    device_type=device_type,
                    dtype=autocast_dtype,
                    enabled=use_mixed_precision
                ):
                    loss = model(inputs, targets)
                    loss = loss / TRAINING_CONFIG['accumulation_steps']
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                batch_loss = loss.item() * TRAINING_CONFIG['accumulation_steps']
                batch_bpb = batch_loss / math.log(2)  # Convert nats to bits
                
                total_train_loss += batch_loss
                num_train_batches += 1
                batch_counter += 1
                
                current_lr = scheduler.get_last_lr()[0]
                
                # Update parquet info for progress tracking
                if dataloader_state_dict is not None:
                    pq_idx = dataloader_state_dict.get('pq_idx', 0)
                    rg_idx = dataloader_state_dict.get('rg_idx', 0)
                    current_parquet_info = f"pq:{pq_idx+1}/{total_parquets} rg:{rg_idx}"
                
                if batch_idx % 10 == 0:  # Update progress bar less frequently for streaming
                    postfix_dict = {
                        'loss': f"{batch_loss:.4f}",
                        'bpb': f"{batch_bpb:.4f}",
                        'lr': f"{current_lr:.2e}",
                        'mem': get_memory_usage_safe(),
                        'epoch': epoch + 1,  # Show current epoch
                        'step': global_step,
                        'iter': iteration
                    }
                    if current_parquet_info:
                        postfix_dict['parquet'] = current_parquet_info
                    progress_bar.set_postfix(postfix_dict)
                
                # Optimizer step
                if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                    # Update Muon momentum before optimizer step
                    if muon_optimizer is not None:
                        muon_momentum = get_muon_momentum(global_step)
                        for group in muon_optimizer.param_groups:
                            group["momentum"] = muon_momentum
                    
                    if scaler:
                        scaler.unscale_(optimizer)
                        if muon_optimizer is not None:
                            all_params = list(optimizer.param_groups[0]['params']) + list(optimizer.param_groups[1]['params'])
                            if muon_optimizer.param_groups:
                                all_params.extend([p for group in muon_optimizer.param_groups for p in group['params']])
                            torch.nn.utils.clip_grad_norm_(all_params, TRAINING_CONFIG['grad_clip'])
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        scaler.step(optimizer)
                        if muon_optimizer is not None:
                            muon_optimizer.step()
                        scaler.update()
                    else:
                        if muon_optimizer is not None:
                            all_params = list(optimizer.param_groups[0]['params']) + list(optimizer.param_groups[1]['params'])
                            if muon_optimizer.param_groups:
                                all_params.extend([p for group in muon_optimizer.param_groups for p in group['params']])
                            torch.nn.utils.clip_grad_norm_(all_params, TRAINING_CONFIG['grad_clip'])
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        optimizer.step()
                        if muon_optimizer is not None:
                            muon_optimizer.step()
                    
                    scheduler.step()
                    if muon_scheduler is not None:
                        muon_scheduler.step()
                    optimizer.zero_grad()
                    if muon_optimizer is not None:
                        muon_optimizer.zero_grad()
                    
                    # Increment global_step for actual training steps
                    global_step += 1
                    
                    # Update epoch based on global_step for streaming mode
                    current_epoch = global_step // steps_per_epoch if steps_per_epoch > 0 else epoch
                    if current_epoch != epoch:
                        # Epoch boundary crossed
                        print(f"\n{'='*80}")
                        print(f"EPOCH COMPLETED: Moving from Epoch {epoch + 1} to Epoch {current_epoch + 1}")
                        print(f"{'='*80}")
                        epoch = current_epoch
                        progress_bar.set_description(f"Training Epoch {epoch + 1}")
                    
                    if global_step % TRAINING_CONFIG['log_interval'] == 0 and use_wandb:
                        log_dict = {
                            'train_loss_step': batch_loss,
                            'train_bpb_step': batch_bpb,
                            'learning_rate': current_lr,
                            'global_step': global_step,
                            'iteration': iteration,
                            'epoch': epoch + 1  # Add epoch to streaming logs
                        }
                        if dataloader_state_dict is not None:
                            log_dict['parquet_index'] = dataloader_state_dict.get('pq_idx', 0)
                            log_dict['row_group_index'] = dataloader_state_dict.get('rg_idx', 0)
                            log_dict['data_progress'] = (dataloader_state_dict.get('pq_idx', 0) / total_parquets) * 100
                        wandb.log(log_dict)
                    
                    if global_step % TRAINING_CONFIG['eval_steps'] == 0 and global_step > 0:
                        eval_info = f"\nQuick eval at step {global_step}"
                        if dataloader_state_dict is not None:
                            pq_idx = dataloader_state_dict.get('pq_idx', 0)
                            eval_info += f" | Parquet {pq_idx+1}/{total_parquets}"
                        print(eval_info)
                        model.eval()
                        quick_val_loss = 0
                        quick_batches = 0
                        
                        val_loader_eval = build_val_loader()
                        
                        with torch.no_grad():
                            for val_batch_idx, val_batch_data in enumerate(val_loader_eval):
                                if val_batch_idx >= 10:  # Quick eval
                                    break
                                
                                # Handle both regular datasets and SFT datasets (with loss masking)
                                if isinstance(val_batch_data, tuple) and len(val_batch_data) == 3:
                                    val_inputs, val_targets, val_loss_mask = val_batch_data
                                    val_inputs = val_inputs.to(device)
                                    val_targets = val_targets.to(device)
                                    val_loss_mask = val_loss_mask.to(device)
                                else:
                                    val_inputs, val_targets = val_batch_data
                                    val_inputs = val_inputs.to(device)
                                    val_targets = val_targets.to(device)
                                    val_loss_mask = None
                                
                                with autocast(
                                    device_type=device_type,
                                    dtype=autocast_dtype,
                                    enabled=use_mixed_precision
                                ):
                                    if val_loss_mask is not None:
                                        # SFT: Compute loss with masking
                                        logits = model(val_inputs, targets=None)
                                        logits_flat = logits.view(-1, logits.size(-1))
                                        targets_flat = val_targets.view(-1)
                                        mask_flat = val_loss_mask.view(-1)
                                        loss_unreduced = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                                        val_loss = (loss_unreduced * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                                    else:
                                        # Regular training
                                        val_loss = model(val_inputs, val_targets)
                                    
                                    quick_val_loss += val_loss.item()
                                    quick_batches += 1
                        
                        avg_quick_val_loss = quick_val_loss / quick_batches if quick_batches > 0 else float('inf')
                        quick_perplexity = math.exp(avg_quick_val_loss) if avg_quick_val_loss < 10 else float('inf')
                        
                        print(f"{'='*60}")
                        print(f"STEP {global_step} EVALUATION")
                        print(f"{'='*60}")
                        print(f"  Train Loss: {total_train_loss/num_train_batches:.4f}")
                        print(f"  Val Loss: {avg_quick_val_loss:.4f}")
                        print(f"  Perplexity: {quick_perplexity:.2f}")
                        
                        # Generate test samples
                        print(f"\n  Test Generations:")
                        for test_idx, prompt in enumerate(TEST_PROMPTS[:3]):  # Test first 3 prompts
                            try:
                                with torch.no_grad():
                                    with autocast(
                                        device_type=device_type,
                                        dtype=autocast_dtype,
                                        enabled=use_mixed_precision
                                    ):
                                        generated = model.generate_from_prompt(
                                            prompt,
                                            max_new_tokens=50,
                                            temperature=0.8,
                                            top_k=40,
                                            skip_special_tokens=(not SHOW_SPECIAL_TOKENS)
                                        )
                                    print(f"    [{test_idx+1}] Prompt: {prompt}")
                                    print(f"        Output: {generated[:200]}{'...' if len(generated) > 200 else ''}")
                            except Exception as e:
                                print(f"    [{test_idx+1}] Generation failed: {e}")
                        
                        print(f"{'='*60}\n")
                        
                        if use_wandb:
                            wandb.log({
                                'quick_val_loss': avg_quick_val_loss,
                                'quick_perplexity': quick_perplexity,
                                'train_loss_avg': total_train_loss/num_train_batches,
                                'global_step': global_step,
                                'epoch': epoch + 1  # Add epoch to quick eval logs
                            })
                        
                        model.train()
                        cleanup_memory()
                    
                    if global_step % TRAINING_CONFIG['checkpoint_steps'] == 0 and global_step > 0:
                        checkpoint_path = f"checkpoints/checkpoint_step_{global_step}.safetensors"
                        if dataloader_state_dict is not None:
                            pq_idx = dataloader_state_dict.get('pq_idx', 0)
                            print(f"\nCheckpoint: Step {global_step} | Epoch {epoch + 1} | Parquet {pq_idx+1}/{total_parquets} | Progress: {(pq_idx/total_parquets)*100:.1f}%")
                            # Update dataloader state dict with current epoch
                            dataloader_state_dict_with_epoch = dataloader_state_dict.copy()
                            dataloader_state_dict_with_epoch['epoch'] = epoch
                        else:
                            dataloader_state_dict_with_epoch = None
                        save_checkpoint(
                            model, optimizer, scheduler, scaler, epoch, global_step,
                            best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                            checkpoint_path=checkpoint_path, muon_optimizer=muon_optimizer if 'muon_optimizer' in locals() else None,
                            muon_scheduler=muon_scheduler if 'muon_scheduler' in locals() else None,
                            dataloader_state_dict=dataloader_state_dict_with_epoch if use_parquet_streaming else None
                        )
                    
                    if global_step % 50 == 0:
                        cleanup_memory()
                    
                    if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                        torch.cuda.empty_cache()
        else:
            # Regular dataloader: epoch-based
            steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
            expected_epoch_from_step = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Expected epoch from global_step: {expected_epoch_from_step}")
            print(f"  Total batches this epoch: {len(train_loader)}")
            print(f"  Accumulation steps: {TRAINING_CONFIG['accumulation_steps']}")
            
            batch_counter = 0
            for batch_idx, batch_data in enumerate(progress_bar):
                # Handle both regular datasets and SFT datasets (with loss masking)
                if training_stage == 'sft' and len(batch_data) == 3:
                    inputs, targets, loss_mask = batch_data
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    loss_mask = loss_mask.to(device, non_blocking=True)
                else:
                    inputs, targets = batch_data
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    loss_mask = None
                
                # Forward pass with mixed precision
                with autocast(
                    device_type=device_type,
                    dtype=autocast_dtype,
                    enabled=use_mixed_precision
                ):
                    if loss_mask is not None:
                        # SFT: Compute loss with masking (only supervised tokens)
                        logits = model(inputs, targets=None)  # Get logits without loss
                        logits_flat = logits.view(-1, logits.size(-1))
                        targets_flat = targets.view(-1)
                        loss_mask_flat = loss_mask.view(-1)
                        
                        # Compute loss only on masked positions
                        loss_unreduced = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                        loss = (loss_unreduced * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-8)
                    else:
                        # Regular training: standard cross-entropy
                        loss = model(inputs, targets)
                    
                    loss = loss / TRAINING_CONFIG['accumulation_steps']
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                batch_loss = loss.item() * TRAINING_CONFIG['accumulation_steps']
                batch_bpb = batch_loss / math.log(2)  # Convert nats to bits
                
                total_train_loss += batch_loss
                num_train_batches += 1
                batch_counter += 1
                
                current_lr = scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'mem': get_memory_usage_safe(),
                    'epoch': epoch + 1,  # Show current epoch
                    'step': global_step,
                    'processed': batch_counter
                })
                
                # Optimizer step
                if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                    # Update Muon momentum before optimizer step
                    if muon_optimizer is not None:
                        muon_momentum = get_muon_momentum(global_step)
                        for group in muon_optimizer.param_groups:
                            group["momentum"] = muon_momentum
                    
                    if scaler:
                        scaler.unscale_(optimizer)
                        if muon_optimizer is not None:
                            # Clip gradients for both optimizers
                            all_params = list(optimizer.param_groups[0]['params']) + list(optimizer.param_groups[1]['params'])
                            if muon_optimizer.param_groups:
                                all_params.extend([p for group in muon_optimizer.param_groups for p in group['params']])
                            torch.nn.utils.clip_grad_norm_(all_params, TRAINING_CONFIG['grad_clip'])
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        scaler.step(optimizer)
                        if muon_optimizer is not None:
                            muon_optimizer.step()
                        scaler.update()
                    else:
                        if muon_optimizer is not None:
                            # Clip gradients for both optimizers
                            all_params = list(optimizer.param_groups[0]['params']) + list(optimizer.param_groups[1]['params'])
                            if muon_optimizer.param_groups:
                                all_params.extend([p for group in muon_optimizer.param_groups for p in group['params']])
                            torch.nn.utils.clip_grad_norm_(all_params, TRAINING_CONFIG['grad_clip'])
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        optimizer.step()
                        if muon_optimizer is not None:
                            muon_optimizer.step()
                    
                    scheduler.step()
                    if muon_scheduler is not None:
                        muon_scheduler.step()
                    optimizer.zero_grad()
                    if muon_optimizer is not None:
                        muon_optimizer.zero_grad()
                
                    # Increment global_step for actual training steps
                    global_step += 1
                    
                    if global_step % TRAINING_CONFIG['log_interval'] == 0 and use_wandb:
                        wandb.log({
                            'train_loss_step': batch_loss,
                            'learning_rate': current_lr,
                            'global_step': global_step,
                            'epoch': epoch + 1
                        })
                
                if global_step % TRAINING_CONFIG['eval_steps'] == 0 and global_step > 0:
                    print(f"\nQuick eval at step {global_step}")
                    model.eval()
                    quick_val_loss = 0
                    quick_batches = 0
                    
                    val_loader_eval = build_val_loader() if use_parquet_streaming else val_loader
                    
                    with torch.no_grad():
                        for val_batch_idx, val_batch_data in enumerate(val_loader_eval):
                            if val_batch_idx >= 10:  # Quick eval
                                break
                            
                            # Handle both regular datasets and SFT datasets (with loss masking)
                            if isinstance(val_batch_data, tuple) and len(val_batch_data) == 3:
                                val_inputs, val_targets, val_loss_mask = val_batch_data
                                val_inputs = val_inputs.to(device)
                                val_targets = val_targets.to(device)
                                val_loss_mask = val_loss_mask.to(device)
                            else:
                                val_inputs, val_targets = val_batch_data
                                val_inputs = val_inputs.to(device)
                                val_targets = val_targets.to(device)
                                val_loss_mask = None
                            
                            with torch.autocast(
                                    device_type=device_type,
                                    dtype=autocast_dtype,
                                    enabled=use_mixed_precision
                                ):
                                if val_loss_mask is not None:
                                    # SFT: Compute loss with masking
                                    logits = model(val_inputs, targets=None)
                                    logits_flat = logits.view(-1, logits.size(-1))
                                    targets_flat = val_targets.view(-1)
                                    mask_flat = val_loss_mask.view(-1)
                                    loss_unreduced = F.cross_entropy(logits_flat, targets_flat, reduction='none')
                                    val_loss = (loss_unreduced * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                                else:
                                    # Regular training
                                    val_loss = model(val_inputs, val_targets)
                            
                            quick_val_loss += val_loss.item()
                            quick_batches += 1
                    
                    avg_quick_val_loss = quick_val_loss / quick_batches if quick_batches > 0 else float('inf')
                    quick_perplexity = math.exp(avg_quick_val_loss) if avg_quick_val_loss < 10 else float('inf')
                    
                    print(f"{'='*60}")
                    print(f"STEP {global_step} EVALUATION")
                    print(f"{'='*60}")
                    print(f"  Train Loss: {total_train_loss/num_train_batches:.4f}")
                    print(f"  Val Loss: {avg_quick_val_loss:.4f}")
                    print(f"  Perplexity: {quick_perplexity:.2f}")
                    
                    # Generate test samples
                    print(f"\n  Test Generations:")
                    for test_idx, prompt in enumerate(TEST_PROMPTS[:3]):  # Test first 3 prompts
                        try:
                            with torch.no_grad():
                                with autocast(
                                    device_type=device_type,
                                    dtype=autocast_dtype,
                                    enabled=use_mixed_precision
                                ):
                                    generated = model.generate_from_prompt(
                                        prompt,
                                        max_new_tokens=50,
                                        temperature=0.8,
                                        top_k=40,
                                        skip_special_tokens=(not SHOW_SPECIAL_TOKENS)
                                    )
                                print(f"    [{test_idx+1}] Prompt: {prompt}")
                                print(f"        Output: {generated[:200]}{'...' if len(generated) > 200 else ''}")
                        except Exception as e:
                            print(f"    [{test_idx+1}] Generation failed: {e}")
                    
                    print(f"{'='*60}\n")
                    
                    if use_wandb:
                        wandb.log({
                            'quick_val_loss': avg_quick_val_loss,
                            'quick_perplexity': quick_perplexity,
                            'train_loss_avg': total_train_loss/num_train_batches,
                            'global_step': global_step,
                            'epoch': epoch + 1  # Add epoch to quick eval logs
                        })
                    
                    model.train()
                    cleanup_memory()
                
                if global_step % TRAINING_CONFIG['checkpoint_steps'] == 0 and global_step > 0:
                    checkpoint_path = f"checkpoints/checkpoint_step_{global_step}.safetensors"
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step,
                        best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                        checkpoint_path=checkpoint_path, muon_optimizer=muon_optimizer if 'muon_optimizer' in locals() else None,
                        muon_scheduler=muon_scheduler if 'muon_scheduler' in locals() else None,
                        dataloader_state_dict=None  # Regular dataloader doesn't need state
                    )
            
            if global_step % 50 == 0: 
                cleanup_memory()
                
            # Only clear cache after actual optimizer steps, not every batch
            if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_bpb = avg_train_loss * 0.69314718056 / avg_bytes_per_token
        print(f"\nEpoch {epoch + 1} completed: Train Loss: {avg_train_loss:.4f}, BPB: {avg_train_bpb:.4f}")
        
        # Show streaming parquet progress
        if use_parquet_streaming and dataloader_state_dict is not None:
            pq_idx = dataloader_state_dict.get('pq_idx', 0)
            rg_idx = dataloader_state_dict.get('rg_idx', 0)
            progress_pct = (pq_idx / total_parquets) * 100
            print(f"Data Progress: Parquet {pq_idx+1}/{total_parquets} ({progress_pct:.1f}%) | Row Group: {rg_idx}")
            # Estimate epochs based on data coverage
            estimated_epochs = ((pq_idx / total_parquets) + epoch) if total_parquets > 0 else epoch
            print(f"Estimated total data epochs covered: {estimated_epochs:.2f}")
        
        print(f"Memory after epoch: {get_memory_usage()}")
        
        if (epoch + 1) % TRAINING_CONFIG['eval_interval'] == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION - Epoch {epoch + 1}")
            print(f"{'='*80}")
            
            val_loader_eval = build_val_loader() if use_parquet_streaming else val_loader
            metrics = evaluate_model_comprehensive(model, val_loader_eval, tokenizer, device, device_type, TRAINING_CONFIG)
            
            val_loss = metrics['val_loss']
            perplexity = metrics['perplexity']
            val_bpb = val_loss * 0.69314718056 / avg_bytes_per_token
            
            print(f"METRICS:")
            print(f"   Train Loss: {avg_train_loss:.4f}, BPB: {avg_train_bpb:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, BPB: {val_bpb:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            print(f"   Memory: {get_memory_usage()}")
            
            if 'generation' in metrics and metrics['generation']['samples']:
                sample = metrics['generation']['samples'][0]
                print(f"   GENERATION SAMPLE:")
                print(f"   Prompt: {sample['prompt']}")
                print(f"   Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
            
            print(f"{'='*80}")
            
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'train_bpb_epoch': avg_train_bpb,
                    'val_loss_epoch': val_loss,
                    'val_bpb_epoch': val_bpb,
                    'perplexity': perplexity,
                    'learning_rate_epoch': current_lr,
                    'global_step': global_step
                })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Update dataloader state dict with current epoch for best model checkpoint
                best_model_dl_state = None
                if use_parquet_streaming and 'dataloader_state_dict' in locals() and dataloader_state_dict is not None:
                    best_model_dl_state = dataloader_state_dict.copy()
                    best_model_dl_state['epoch'] = epoch
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step,
                    best_val_loss, perplexity, MODEL_CONFIG, tokenizer_path,
                    checkpoint_path="checkpoints/best_model_120m_8gb.safetensors",
                    muon_optimizer=muon_optimizer if 'muon_optimizer' in locals() else None,
                    muon_scheduler=muon_scheduler if 'muon_scheduler' in locals() else None,
                    dataloader_state_dict=best_model_dl_state
                )
                print(f"New best model saved! Val loss: {val_loss:.4f}")
            
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {early_stopping.best_loss:.4f}")
                break
            
            cleanup_memory()
        
        if (epoch + 1) % TRAINING_CONFIG['save_interval'] == 0:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.safetensors"
            # Update dataloader state dict with current epoch for periodic checkpoint
            periodic_dl_state = None
            if use_parquet_streaming and 'dataloader_state_dict' in locals() and dataloader_state_dict is not None:
                periodic_dl_state = dataloader_state_dict.copy()
                periodic_dl_state['epoch'] = epoch
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                checkpoint_path=checkpoint_path, muon_optimizer=muon_optimizer if 'muon_optimizer' in locals() else None,
                muon_scheduler=muon_scheduler if 'muon_scheduler' in locals() else None,
                dataloader_state_dict=periodic_dl_state
            )
    
    if use_wandb:
        wandb.finish()
    
    print(f"\nTraining completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Final memory usage: {get_memory_usage()}")
    
    # Log to report
    try:
        from ..report import get_report
        training_stage = TRAINING_CONFIG.get('training_stage', 'base')
        get_report().log(section=f"{training_stage.capitalize()} model training", data=[
            {
                "Training stage": training_stage,
                "Number of epochs": epoch + 1,
                "Best validation loss": best_val_loss,
                "Best perplexity": early_stopping.best_loss if hasattr(early_stopping, 'best_loss') else float('inf'),
                "Global step": global_step,
            },
            MODEL_CONFIG,
            TRAINING_CONFIG,
        ])
    except Exception as e:
        print(f"Warning: Could not log to report: {e}")
    
    return model

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                   best_val_loss, best_perplexity, config, tokenizer_path, 
                   checkpoint_path="checkpoint.safetensors", muon_optimizer=None, muon_scheduler=None, dataloader_state_dict=None):
    
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("SafeTensors not available. Install with: pip install safetensors")
    
    # Model state - untied weights, so save everything
    model_state = {}
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else model.module.state_dict()
    
    for name, param in state_dict.items():
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
        'weight_tying': 'false',  # Untied weights
        'model_type': 'Transformer'
    }
    
    # Add dataloader state if available
    # Use single source of truth for epoch (from dataloader state if available)
    if dataloader_state_dict is not None:
        metadata['parquet_index'] = str(dataloader_state_dict.get('pq_idx', 0))
        metadata['row_group_index'] = str(dataloader_state_dict.get('rg_idx', 0))
        # Epoch from dataloader is source of truth
        current_epoch = dataloader_state_dict.get('epoch', epoch)
        metadata['epoch'] = str(current_epoch)
    
    # Save to SafeTensors
    save_file(model_state, checkpoint_path, metadata=metadata)
    
    if optimizer is not None:
        additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
        additional_state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }
        
        if scaler:
            additional_state['scaler'] = scaler.state_dict()
        
        if muon_optimizer is not None:
            additional_state['muon_optimizer'] = muon_optimizer.state_dict()
        
        if muon_scheduler is not None:
            additional_state['muon_scheduler'] = muon_scheduler.state_dict()
        
        if dataloader_state_dict is not None:
            additional_state['dataloader_state_dict'] = dataloader_state_dict
        
        torch.save(additional_state, additional_state_path)
    
    print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
    
    cleanup_memory()
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, device=None, muon_optimizer=None, muon_scheduler=None):
    
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
    
    # Load model weights with better error reporting
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            print(f"  First 10: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            print(f"  First 10: {unexpected_keys[:10]}")
    print("Model weights loaded")
    
    # Load training state
    additional_state_path = checkpoint_path.replace('.safetensors', '_state.pt')
    additional_state = {}
    if os.path.exists(additional_state_path):
        additional_state = torch.load(additional_state_path, map_location=device, weights_only=True)
        
        if optimizer and 'optimizer' in additional_state:
            try:
                optimizer.load_state_dict(additional_state['optimizer'])
                print("Optimizer state loaded")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
                print("Continuing with fresh optimizer state...")
        
        if scheduler and 'scheduler' in additional_state and additional_state['scheduler']:
            try:
                scheduler.load_state_dict(additional_state['scheduler'])
                print("Scheduler state loaded")
                # Verify and fix scheduler step to match global_step
                expected_step = int(metadata.get('global_step', '0'))
                # LambdaLR uses 'last_epoch' in its state_dict
                if 'last_epoch' in additional_state['scheduler']:
                    saved_step = additional_state['scheduler']['last_epoch']
                    if saved_step != expected_step:
                        print(f"Warning: Scheduler step ({saved_step}) != global_step ({expected_step})")
                        print(f"Manually advancing scheduler from {saved_step} to {expected_step}")
                        # Manually step scheduler to correct position
                        for _ in range(saved_step, expected_step):
                            scheduler.step()
                        print(f"Scheduler now at step {expected_step}")
                else:
                    # For LambdaLR, we need to step manually
                    print(f"Manually setting scheduler to step {expected_step}")
                    for _ in range(expected_step):
                        scheduler.step()
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
                print("Continuing with fresh scheduler state...")
                # Manually set scheduler to correct step even if loading failed
                expected_step = int(metadata.get('global_step', '0'))
                print(f"Manually advancing scheduler to step {expected_step}")
                for _ in range(expected_step):
                    scheduler.step()
        
        if scaler and 'scaler' in additional_state:
            try:
                scaler.load_state_dict(additional_state['scaler'])
                print("Scaler state loaded")
            except Exception as e:
                print(f"Warning: Failed to load scaler state: {e}")
                print("Continuing with fresh scaler state...")
        
        if muon_optimizer and 'muon_optimizer' in additional_state:
            try:
                muon_optimizer.load_state_dict(additional_state['muon_optimizer'])
                print("Muon optimizer state loaded")
            except Exception as e:
                print(f"Warning: Failed to load muon optimizer state: {e}")
                print("Continuing with fresh muon optimizer state...")
        
        if muon_scheduler and 'muon_scheduler' in additional_state and additional_state['muon_scheduler']:
            try:
                muon_scheduler.load_state_dict(additional_state['muon_scheduler'])
                print("Muon scheduler state loaded")
                # Verify and fix muon scheduler step to match global_step
                expected_step = int(metadata.get('global_step', '0'))
                if 'last_epoch' in additional_state['muon_scheduler']:
                    saved_step = additional_state['muon_scheduler']['last_epoch']
                    if saved_step != expected_step:
                        print(f"Warning: Muon scheduler step ({saved_step}) != global_step ({expected_step})")
                        print(f"Manually advancing muon scheduler from {saved_step} to {expected_step}")
                        for _ in range(saved_step, expected_step):
                            muon_scheduler.step()
                        print(f"Muon scheduler now at step {expected_step}")
                else:
                    print(f"Manually setting muon scheduler to step {expected_step}")
                    for _ in range(expected_step):
                        muon_scheduler.step()
            except Exception as e:
                print(f"Warning: Failed to load muon scheduler state: {e}")
                print("Continuing with fresh muon scheduler state...")
                # Manually set muon scheduler to correct step even if loading failed
                expected_step = int(metadata.get('global_step', '0'))
                print(f"Manually advancing muon scheduler to step {expected_step}")
                for _ in range(expected_step):
                    muon_scheduler.step()
    
    # Parse metadata
    dataloader_state = additional_state.get('dataloader_state_dict', None) if os.path.exists(additional_state_path) else None
    
    # Prioritize epoch from dataloader_state_dict (most accurate for streaming mode)
    saved_epoch = int(metadata.get('epoch', '0'))
    if dataloader_state is not None and 'epoch' in dataloader_state:
        saved_epoch = dataloader_state['epoch']
        print(f"Using epoch from dataloader state: {saved_epoch}")
    
    resume_info = {
        'epoch': saved_epoch,
        'global_step': int(metadata.get('global_step', '0')),
        'best_val_loss': float(metadata.get('best_val_loss', 'inf')),
        'best_perplexity': float(metadata.get('best_perplexity', 'inf')),
        'config': json.loads(metadata.get('config', '{}')),
        'tokenizer_path': metadata.get('tokenizer_path', ''),
        'dataloader_state_dict': dataloader_state,
    }
    
    print(f"Resume from: Epoch {resume_info['epoch']}, Step {resume_info['global_step']}")
    
    # Show parquet info if available in metadata
    if metadata.get('parquet_index') or metadata.get('row_group_index'):
        pq_idx = int(metadata.get('parquet_index', '0'))
        rg_idx = int(metadata.get('row_group_index', '0'))
        print(f"  Saved Parquet State: Parquet {pq_idx+1}, Row Group {rg_idx}")
    cleanup_memory()
    
    return resume_info

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
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

def generate(text, 
             model_path="checkpoints/best_model_120m.safetensors",
             tokenizer_path="turkish_tokenizer/turkish_tokenizer.model",
             max_new_tokens=100,
             temperature=0.8,
             top_p=0.9,
             top_k=10,
             device=None,
             use_half_precision=True,
             silent=True):

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    if silent:
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                if os.path.isdir(tokenizer_path):
                    tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
                else:
                    tokenizer = create_tokenizer(model_path=tokenizer_path)
    else:
        if os.path.isdir(tokenizer_path):
            tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
        else:
            tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    if not model_path.endswith('.safetensors'):
        model_path = model_path.replace('.pt', '.safetensors')
    
    from safetensors import safe_open
    metadata = {}
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    
    config = json.loads(metadata.get('config', '{}'))
    
    # Disable FlashAttention for generation to avoid dtype issues
    config['use_flash_attention'] = False
    
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
    
    model.eval()
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_bfloat16 = use_half_precision and torch.cuda.is_bf16_supported() if device_type == 'cuda' else False
    autocast_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    
    with torch.no_grad():
        try:
            with autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=(use_half_precision and device_type == 'cuda')
            ):
                generated_text = model.generate_from_prompt(
                    text, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
        except Exception as e:
            # Fallback to CPU with full precision
            model = model.float().cpu()
            with autocast(device_type='cpu', enabled=False):
                generated_text = model.generate_from_prompt(
                    text, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    
    cleanup_memory()
    
    if not silent:
        print(generated_text)
    
    return generated_text