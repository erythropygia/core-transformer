import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import json
import re
from tqdm import tqdm
from datasets import load_dataset
import unicodedata
import wandb
from contextlib import redirect_stdout, redirect_stderr

from deepspeed_config.deepspeed_config import create_deepspeed_config

from .utils import EarlyStopping, cleanup_memory, evaluate_model_comprehensive, get_cosine_schedule_with_warmup, get_gpu_memory_info, get_memory_usage, get_memory_usage_safe
from .config import MODEL_CONFIG, TRAINING_CONFIG, TEST_PROMPTS
from .tokenizer import create_tokenizer
from .dataset import TransformerDataset, load_and_preprocess_data
from .transformer_block import Transformer

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

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

try:
    from nltk.translate.bleu_score import sentence_bleu
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

def train(
    tokenizer_path="turkish_tokenizer/turkish_tokenizer.model",
    resume_from_checkpoint=None,
    auto_resume=False,
    use_wandb=True,
    project_name="turkish-transformer-120m",
    pretrained_model_path=None,
    fresh_epochs=None
):
    
    print("Transformer train")
    print("="*80)
    
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
    
    print(f"Flash Attention: {'OK' if FLASH_ATTENTION_AVAILABLE else 'NO'}")
    print(f"Gradient Checkpointing: OK")
    print(f"Mixed Precision: OK")
    print(f"CPU Offloading: OK")
    print(f"DeepSpeed: {'OK' if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed'] else 'NO'}")
    
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
    
    print("\nTesting new tokenizer:")
    test_text = "Türkiye'nin başkenti Ankara'dır. İstanbul Boğazı çok güzel."
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Original: {test_text}")
    print(f"Tokens: {len(tokens)} tokens")
    print(f"Decoded: {decoded}")
    print(f"Match: {'OK' if test_text.strip() == decoded.strip() else 'NO'}")
    
    print("Tokenizing data...")
    all_tokens = []
    for text in tqdm(full_corpus, desc="Encoding texts"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
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
    print(f"Compression ratio: {len(' '.join(full_corpus)) / len(all_tokens):.2f} chars/token")
    
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
    
    if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
        print(f"\nSetting up DeepSpeed for GPU")
        
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
        
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_config
        )
        
        model = model_engine
        scaler = None  # DeepSpeed handles mixed precision
        
        print(f"DeepSpeed initialized!")
        print(f"  Effective batch size: {deepspeed_config['train_batch_size']}")
        print(f"  Micro batch size: {deepspeed_config['train_micro_batch_size_per_gpu']}")
        print(f"  Memory usage after DeepSpeed: {get_memory_usage()}")
        
    else:
        print(f"\nUsing standard PyTorch training...")
        
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
                
                steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
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
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['max_epochs']}")
        print(f"Memory before epoch: {get_memory_usage()}")
        print(f"Global step: {global_step}")
        
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        

        if(global_step > 0):
            print(f"Resume info: Starting from saved epoch {epoch + 1}, global step {global_step}")
        steps_per_epoch = len(train_loader) // TRAINING_CONFIG['accumulation_steps']
        expected_epoch_from_step = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Expected epoch from global_step: {expected_epoch_from_step}")
        print(f"  Total batches this epoch: {len(train_loader)}")
        print(f"  Accumulation steps: {TRAINING_CONFIG['accumulation_steps']}")
        
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
                model.backward(loss)
            else:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            if DEEPSPEED_AVAILABLE and TRAINING_CONFIG['use_deepspeed']:
                batch_loss = loss.item()
            else:
                batch_loss = loss.item() * TRAINING_CONFIG['accumulation_steps']
            
            total_train_loss += batch_loss
            num_train_batches += 1
            batch_counter += 1
            
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
                    model.step()
                else:
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
                
                if global_step % TRAINING_CONFIG['checkpoint_steps'] == 0 and global_step > 0:
                    checkpoint_path = f"checkpoints/checkpoint_step_{global_step}.safetensors"
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step,
                        best_val_loss, float('inf'), MODEL_CONFIG, tokenizer_path,
                        checkpoint_path=checkpoint_path
                    )
            
            if global_step % 50 == 0: 
                cleanup_memory()
                
            # Only clear cache after actual optimizer steps, not every batch
            if (batch_idx + 1) % TRAINING_CONFIG['accumulation_steps'] == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / num_train_batches
        print(f"\nEpoch {epoch + 1} completed: Train Loss: {avg_train_loss:.4f}")
        print(f"Memory after epoch: {get_memory_usage()}")
        
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
                    'val_loss_epoch': val_loss,
                    'perplexity': perplexity,
                    'learning_rate_epoch': current_lr,
                    'global_step': global_step
                })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, global_step,
                    best_val_loss, perplexity, MODEL_CONFIG, tokenizer_path,
                    checkpoint_path="checkpoints/best_model_120m_8gb.safetensors"
                )
                print(f"New best model saved! Val loss: {val_loss:.4f}")
            
            if early_stopping(val_loss, model):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                print(f"Best validation loss: {early_stopping.best_loss:.4f}")
                break
            
            cleanup_memory()
        
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

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                   best_val_loss, best_perplexity, config, tokenizer_path, 
                   checkpoint_path="checkpoint.safetensors"):
    
    if not checkpoint_path.endswith('.safetensors'):
        checkpoint_path = checkpoint_path.replace('.pt', '.safetensors')
    
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    
    if DEEPSPEED_AVAILABLE and hasattr(model, 'save_checkpoint'):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        model.save_checkpoint(checkpoint_dir, checkpoint_name)
        
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
                tokenizer = create_tokenizer(model_path=tokenizer_path)
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
            # Fallback to CPU with full precision
            model = model.float().cpu()
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
