import argparse
import sys
import os
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from transformer.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from transformer.model.engine import Engine
from transformer.checkpoint import load_model_from_checkpoint

from tasks.arc_tr import ARC_TR
from tasks.gsm8k_tr import GSM8K_TR
from tasks.mmlu_tr import MMLU_TR


def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = next(model.parameters()).device

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)

        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

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


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = next(model.parameters()).device
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    num_passed, total = 0, 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_problems)
        batch_indices = list(range(batch_start, batch_end))

        rank_batch_indices = [idx for idx in batch_indices if idx % ddp_world_size == ddp_rank]
        if len(rank_batch_indices) == 0:
            continue

        conversations = [task_object[idx] for idx in rank_batch_indices]
        encoded_prompts = [tokenizer.render_for_completion(conv) for conv in conversations]

        max_len = max(len(p) for p in encoded_prompts)
        input_ids = torch.full((len(encoded_prompts), max_len), bos, dtype=torch.long, device=device)
        for i, prompt in enumerate(encoded_prompts):
            input_ids[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model.forward(input_ids, targets=None)
            last_positions = torch.tensor([len(p) - 1 for p in encoded_prompts], device=device)
            last_logits = logits[torch.arange(len(encoded_prompts), device=device), last_positions]

        for i, (conv, prompt) in enumerate(zip(conversations, encoded_prompts)):
            letters = conv.get('letters', ['A', 'B', 'C', 'D'])
            letter_logits = []
            for letter in letters:
                letter_id = tokenizer.encode_special(letter)
                letter_logits.append(last_logits[i, letter_id].item())

            pred_idx = max(range(len(letters)), key=lambda j: letter_logits[j])
            predicted_letter = letters[pred_idx]

            passed = task_object.evaluate(conv, predicted_letter)
            total += 1
            num_passed += int(passed)

        print(f"\r\033[KRank {ddp_rank} | Batch {batch_idx+1}/{num_batches} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

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

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    if args.checkpoint is not None:
        model, tokenizer, _meta = load_model_from_checkpoint(args.checkpoint, device)
    else:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                model, tokenizer, _meta = load_model_from_checkpoint(checkpoint_path, device)
            else:
                raise FileNotFoundError("No checkpoint found. Please specify --checkpoint")
        else:
            raise FileNotFoundError("No checkpoint directory found. Please specify --checkpoint")

    engine = Engine(model, tokenizer)

    task_name = args.task
    if task_name.startswith("ARC-"):
        subset = task_name.split("-", 1)[1]
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

    if ddp_rank == 0:
        from transformer.report import get_report
        stage = "sft"
        get_report().log(section=f"Chat evaluation {stage}", data=[
            vars(args),
            {task_name: accuracy},
        ])

    compute_cleanup()

if __name__ == "__main__":
    main()
