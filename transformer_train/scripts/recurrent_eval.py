import argparse
import json
import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformer.common import compute_init, compute_cleanup, print0, autodetect_device_type
from transformer.checkpoint import load_model_from_checkpoint, find_latest_checkpoint
from transformer.data.dataloader import tokenizing_distributed_data_loader
from transformer.tokenizer import get_token_bytes


@torch.no_grad()
def eval_at_depth(model, batches, r, token_bytes, autocast_ctx):
    total_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    for inputs, targets in batches:
        with autocast_ctx:
            loss = model(inputs, targets, r=r, loss_reduction='sum')
        valid = targets != -1
        total_nats += float(loss)
        total_tokens += int(valid.sum())
        total_bytes += int(token_bytes[targets[valid]].sum())
    nats_per_token = total_nats / max(total_tokens, 1)
    bits_per_byte = total_nats / max(total_bytes, 1) / 0.6931471805599453
    return {
        "r": r,
        "nats_per_token": nats_per_token,
        "perplexity": float(torch.exp(torch.tensor(nats_per_token))),
        "bits_per_byte": bits_per_byte,
        "tokens": total_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="results/recurrent_depth_scaling.json")
    args = ap.parse_args()

    device_type = autodetect_device_type()
    ddp, rank, local_rank, world_size, device = compute_init(device_type)
    autocast_ctx = (torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
                    if device_type == "cuda" else nullcontext())

    ckpt = args.checkpoint or find_latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found; pass --checkpoint")
    model, tokenizer, meta = load_model_from_checkpoint(ckpt, device)

    if not hasattr(model, "r_default"):
        raise SystemExit(
            f"{ckpt} is not a recurrent checkpoint. Train with "
            f"MODEL_CONFIG['recurrent']['enabled'] = True."
        )

    token_bytes = get_token_bytes(tokenizer, device=device)

    loader = tokenizing_distributed_data_loader(
        B=args.batch_size, T=args.seq_len, split=args.split, tokenizer=tokenizer
    )
    batches = [next(loader) for _ in range(args.batches)]
    print0(f"Held out {len(batches)} batches of {args.batch_size} x {args.seq_len}")

    train_r = meta.get("model_config", {}).get("recurrent", {}).get("r_mean")
    rows = []
    for r in args.depths:
        row = eval_at_depth(model, batches, r, token_bytes, autocast_ctx)
        passes = model.n_prelude + model.n_recurrent * r + model.n_coda
        row["layer_passes"] = passes
        row["flop_multiplier"] = passes / model.config["n_layer"]
        rows.append(row)
        print0(f"  r={r:>3} | ppl {row['perplexity']:9.3f} | bpb {row['bits_per_byte']:.4f} "
               f"| {passes:>4} passes ({row['flop_multiplier']:.2f}x)")

    best = min(rows, key=lambda x: x["perplexity"])
    baseline = next((x for x in rows if x["r"] == 1), rows[0])
    monotone = all(rows[i]["perplexity"] >= rows[i + 1]["perplexity"] for i in range(len(rows) - 1))

    payload = {
        "checkpoint": ckpt,
        "train_r_mean": train_r,
        "depths": rows,
        "best_r": best["r"],
        "best_perplexity": best["perplexity"],
        "baseline_r1_perplexity": baseline["perplexity"],
        "monotone_improvement": monotone,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print0("")
    print0(f"best r = {best['r']} at ppl {best['perplexity']:.3f} "
           f"(r=1 baseline {baseline['perplexity']:.3f})")
    print0(f"monotone in depth: {monotone}")
    if train_r is not None and best["r"] > train_r:
        print0(f"extrapolates beyond training depth (train mean {train_r})")
    print0(f"saved {args.out}")
    compute_cleanup()


if __name__ == "__main__":
    main()
