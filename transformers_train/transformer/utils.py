import torch
import math
import gc
from torch.amp import autocast
from .config import TEST_PROMPTS

# Memory Management Functions
def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  
    gc.collect()
    
def get_memory_usage_safe():
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU: {allocated:.2f}GB/{total:.1f}GB (Reserved: {reserved:.2f}GB)"
        except Exception as e:
            return f"GPU: Memory error - {str(e)[:50]}"
    return "CPU Mode"

def get_memory_usage():
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU: {allocated:.2f}GB/{total:.1f}GB (Reserved: {reserved:.2f}GB)"
        except Exception as e:
            return f"GPU: Error - {str(e)[:30]}"
    return "CPU Mode"

def get_gpu_memory_info():
    if torch.cuda.is_available():
        try:
            device_props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = device_props.total_memory / 1024**3
            free = total - reserved
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': free,
                'utilization_pct': (reserved / total) * 100,
                'device_name': device_props.name
            }
        except Exception as e:
            return {'error': str(e)}
    return {'error': 'CUDA not available'}

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
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        pct_start=0.3
    )

def get_plateau_scheduler(optimizer, mode='min', factor=0.5, patience=3, verbose=True):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose,
        min_lr=1e-6
    )

@torch.no_grad()
def calculate_perplexity(model, data_loader, device, device_type, max_batches=50):
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
        
        if batch_idx % 10 == 0:
            cleanup_memory()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    return perplexity

@torch.no_grad() 
def evaluate_generation_quality(model, tokenizer, test_prompts, device, max_new_tokens=30):
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
            
            tokens = tokenizer.encode(generated_text, add_special_tokens=False)
            total_length += len(tokens)
            all_tokens.update(tokens)
            total_tokens += len(tokens)
            
        except Exception as e:
            print(f"Generation error for '{prompt}': {e}")
            continue
    
    if results['samples']:
        results['avg_length'] = total_length / len(results['samples'])
        results['unique_tokens_ratio'] = len(all_tokens) / max(total_tokens, 1)
    
    return results

@torch.no_grad()
def evaluate_model_comprehensive(model, val_loader, tokenizer, device, device_type, config):
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
        
        if batch_idx % 10 == 0:
            cleanup_memory()
    
    metrics['val_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    try:
        metrics['perplexity'] = calculate_perplexity(model, val_loader, device, device_type, max_batches=30)
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        metrics['perplexity'] = float('inf')
    
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
