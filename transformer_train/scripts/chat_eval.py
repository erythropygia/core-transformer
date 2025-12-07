import argparse
import sys
import os
from contextlib import nullcontext

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from transformer.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from transformer.engine import Engine
from transformer.transformer_block import Transformer
from transformer.tokenizer import create_tokenizer
from transformer.config import MODEL_CONFIG
from safetensors import safe_open
from safetensors.torch import load_file
import json

from tasks.arc_tr import ARC_TR
from tasks.gsm8k_tr import GSM8K_TR
from tasks.mmlu_tr import MMLU_TR

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = next(model.parameters()).device

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = next(model.parameters()).device
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    num_passed, total = 0, 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_problems)
        batch_indices = list(range(batch_start, batch_end))

        # Stride the batch across ranks
        rank_batch_indices = [idx for idx in batch_indices if idx % ddp_world_size == ddp_rank]
        if len(rank_batch_indices) == 0:
            continue

        # Tokenize all prompts in the batch
        conversations = [task_object[idx] for idx in rank_batch_indices]
        encoded_prompts = [tokenizer.render_for_completion(conv) for conv in conversations]

        # Pad to same length
        max_len = max(len(p) for p in encoded_prompts)
        input_ids = torch.full((len(encoded_prompts), max_len), bos, dtype=torch.long, device=device)
        for i, prompt in enumerate(encoded_prompts):
            input_ids[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

        # Forward pass
        with torch.no_grad():
            logits, _ = model.forward(input_ids, targets=None)
            # Get logits at the last position of each prompt
            last_positions = torch.tensor([len(p) - 1 for p in encoded_prompts], device=device)
            last_logits = logits[torch.arange(len(encoded_prompts), device=device), last_positions]  # (batch_size, vocab_size)

        # For each problem, check which letter has the highest logit
        for i, (conv, prompt) in enumerate(zip(conversations, encoded_prompts)):
            letters = conv.get('letters', ['A', 'B', 'C', 'D'])
            # Get logits for each letter token
            letter_logits = []
            for letter in letters:
                letter_id = tokenizer.encode_special(letter)
                letter_logits.append(last_logits[i, letter_id].item())
            
            # Find the letter with highest logit
            pred_idx = max(range(len(letters)), key=lambda j: letter_logits[j])
            predicted_letter = letters[pred_idx]
            
            # Evaluate
            passed = task_object.evaluate(conv, predicted_letter)
            total += 1
            num_passed += int(passed)

        # Logging
        print(f"\r\033[KRank {ddp_rank} | Batch {batch_idx+1}/{num_batches} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    return num_passed/total

# -----------------------------------------------------------------------------
# Main evaluation function

def load_model_from_checkpoint(checkpoint_path, device):
    print0(f"Loading model from: {checkpoint_path}")
    
    # Load metadata
    metadata = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata()
    
    # Load config
    config = json.loads(metadata.get('config', '{}'))
    if not config:
        config = MODEL_CONFIG.copy()
    
    # Load tokenizer
    tokenizer_path = metadata.get('tokenizer_path', 'tokenizer')
    if os.path.isdir(tokenizer_path):
        tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
    else:
        tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    config['vocab_size'] = tokenizer.vocab_size
    
    # Create model
    model = Transformer(config, tokenizer).to(device)
    
    # Load weights
    model_state = load_file(checkpoint_path)
    new_state_dict = {}
    for k, v in model_state.items():
        if k.startswith('_orig_mod.'):
            new_key = k[len('_orig_mod.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--task', type=str, required=True, help='Task name (e.g., ARC-Easy, GSM8K, MMLU)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples per problem (for generative tasks)')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for categorical tasks')
    parser.add_argument('--max-problems', type=int, default=None, help='Max number of problems to evaluate')
    args = parser.parse_args()

    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model
    if args.checkpoint is not None:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
    else:
        # Try to find latest checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                model, tokenizer = load_model_from_checkpoint(checkpoint_path, device)
            else:
                raise FileNotFoundError("No checkpoint found. Please specify --checkpoint")
        else:
            raise FileNotFoundError("No checkpoint directory found. Please specify --checkpoint")

    # Create engine
    engine = Engine(model, tokenizer)

    # Load task
    task_name = args.task
    if task_name.startswith("ARC-"):
        subset = task_name.split("-", 1)[1]  # "Easy" or "Challenge"
        task = ARC_TR(subset=f"ARC-{subset}", split="test")
        eval_type = "categorical"
    elif task_name == "GSM8K":
        task = GSM8K_TR(subset="main", split="test")
        eval_type = "generative"
    elif task_name == "MMLU":
        task = MMLU_TR(subset="all", split="test")
        eval_type = "categorical"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    print0(f"Evaluating on {task_name} ({eval_type} task)")

    # Run evaluation
    with autocast_ctx:
        if eval_type == "generative":
            accuracy = run_generative_eval(
                task, tokenizer, model, engine,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems
            )
        else:
            accuracy = run_categorical_eval(
                task, tokenizer, model,
                batch_size=args.batch_size,
                max_problems=args.max_problems
            )

    print0(f"Task: {task_name}, Accuracy: {accuracy:.4f}")

    # Log to report
    if ddp_rank == 0:
        from transformer.report import get_report
        # Determine stage from checkpoint path or default to 'sft'
        stage = "sft"  # Default, could be improved to detect from checkpoint
        get_report().log(section=f"Chat evaluation {stage}", data=[
            vars(args), # CLI args
            {task_name: accuracy},
        ])

    compute_cleanup()

if __name__ == "__main__":
    main()

