import os
import sys
import itertools
import argparse
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from transformer.common import compute_init, compute_cleanup, print0, get_base_dir, get_dist_info, autodetect_device_type
from transformer.model.engine import Engine
from transformer.training.muon import Muon
from transformer.checkpoint import load_model_from_checkpoint
from safetensors.torch import save_file
import json

from tasks.gsm8k_tr import GSM8K_TR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='dummy', help='Run name for logging')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to load')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--device-batch-size', type=int, default=8, help='Batch size for forward pass')
    parser.add_argument('--examples-per-step', type=int, default=16, help='Total examples per step across all ranks')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples per example')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--unembedding-lr', type=float, default=0.004, help='Learning rate for unembedding layer')
    parser.add_argument('--embedding-lr', type=float, default=0.2, help='Learning rate for embedding layer')
    parser.add_argument('--matrix-lr', type=float, default=0.02, help='Learning rate for matrix layers')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--init-lr-frac', type=float, default=0.05, help='Initial learning rate fraction')
    parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--save-every', type=int, default=60, help='Save checkpoint every N steps')
    parser.add_argument('--eval-every', type=int, default=60, help='Evaluate every N steps')
    parser.add_argument('--eval-examples', type=int, default=400, help='Number of examples for evaluation')
    return parser.parse_args()


def get_batch_generator(train_task, tokenizer, model, engine, device, device_batch_size, num_samples, max_new_tokens, temperature, top_k, autocast_ctx, step):
    set(tokenizer.get_stop_token_ids())
    pad_id = tokenizer.get_pad_token_id()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)

    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]

        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = (num_samples + device_batch_size - 1) // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            batch_size_this_step = min(device_batch_size, num_samples - len(generated_token_sequences))
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=batch_size_this_step,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = 1.0 if train_task.evaluate(conversation, generated_text) else 0.0
            rewards.append(reward)

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [seq + [pad_id] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]

        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_token_sequences, inputs, targets, rewards, advantages


@torch.no_grad()
def run_gsm8k_eval(task, tokenizer, engine, device_batch_size, max_examples, num_samples, max_completion_tokens, temperature, top_k, autocast_ctx):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

    records = []
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        assert num_samples <= device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )

        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})

        records.append({
            "idx": idx,
            "outcomes": outcomes,
        })

    return records


def main():
    args = parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    if args.checkpoint and os.path.exists(args.checkpoint):
        model, tokenizer, meta = load_model_from_checkpoint(args.checkpoint, device, eval_mode=False)
        config = meta["model_config"]
    else:
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                model, tokenizer, meta = load_model_from_checkpoint(checkpoint_path, device, eval_mode=False)
                config = meta["model_config"]
            else:
                raise FileNotFoundError("No checkpoint found. Please specify --checkpoint")
        else:
            raise FileNotFoundError("No checkpoint directory found. Please specify --checkpoint")

    engine = Engine(model, tokenizer)

    train_task = GSM8K_TR(subset="main", split="train")
    val_task = GSM8K_TR(subset="main", split="test")
    num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
    print0(f"Calculated number of steps: {num_steps}")

    embedding_params = []
    unembedding_params = []
    matrix_params = []

    for name, param in model.named_parameters():
        if 'embed' in name.lower():
            embedding_params.append(param)
        elif 'lm_head' in name.lower() or 'unembed' in name.lower():
            unembedding_params.append(param)
        else:
            matrix_params.append(param)

    optimizers = []
    if embedding_params:
        optimizers.append(Muon(embedding_params, lr=args.embedding_lr, weight_decay=args.weight_decay))
    if unembedding_params:
        optimizers.append(Muon(unembedding_params, lr=args.unembedding_lr, weight_decay=args.weight_decay))
    if matrix_params:
        optimizers.append(Muon(matrix_params, lr=args.matrix_lr, weight_decay=args.weight_decay))

    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * args.init_lr_frac
            group["initial_lr"] = group["lr"]

    def get_lr_multiplier(it):
        return 1.0 - it / num_steps

    print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
    assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world_size"
    examples_per_rank = args.examples_per_step // ddp_world_size
    print0(f"Calculated examples per rank: {examples_per_rank}")

    batch_iterator = get_batch_generator(
        train_task, tokenizer, model, engine, device,
        args.device_batch_size, args.num_samples, args.max_new_tokens,
        args.temperature, args.top_k, autocast_ctx, 0
    )

    for step in range(num_steps):
        if step % args.eval_every == 0:
            model.eval()
            with autocast_ctx:
                records = run_gsm8k_eval(
                    val_task, tokenizer, engine, args.device_batch_size,
                    args.eval_examples, args.device_batch_size, args.max_new_tokens,
                    temperature=1.0, top_k=args.top_k, autocast_ctx=autocast_ctx
                )

            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)

            passk = []
            for k in range(1, args.device_batch_size + 1):
                correct = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
                passk.append(correct / max(num_records.item(), 1))

            print_passk = [f"Pass@{k}: {passk[k-1]:.4f}" for k in range(1, args.device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")

        rewards_list = []
        sequence_lengths = []

        for example_step in range(examples_per_rank):
            sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

            model.train()
            assert inputs_all.size(0) % args.device_batch_size == 0
            num_passes = inputs_all.size(0) // args.device_batch_size

            for pass_idx in range(num_passes):
                b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                advantages = advantages_all[b0:b1]

                with autocast_ctx:
                    _, loss2d = model.forward(inputs, targets=targets)
                    logp = -loss2d

                pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                num_valid = (targets >= 0).sum().clamp(min=1)
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
                loss = -pg_obj

                loss.backward()

            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)

        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
        print0(f"Step {step}/{num_steps} | Average reward: {mean_reward:.4f} | Average sequence length: {mean_sequence_length:.2f}")

        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
            opt.step()
        model.zero_grad(set_to_none=True)

        if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
            checkpoint_dir = os.path.join(get_base_dir(), "chatrl_checkpoints", f"step_{step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step:06d}.safetensors")

            save_file(model.state_dict(), checkpoint_path, metadata={
                "config": json.dumps(config),
                "tokenizer_path": "tokenizer",
                "global_step": str(step),
            })
            print0(f"Saved checkpoint to {checkpoint_path}")

    if master_process:
        from transformer.report import get_report
        get_report().log(section="Chat RL", data=[
            vars(args),
        ])

    compute_cleanup()

if __name__ == "__main__":
    main()
