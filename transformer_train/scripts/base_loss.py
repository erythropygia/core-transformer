import os
import sys
from contextlib import nullcontext
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.common import compute_init, print0, compute_cleanup, autodetect_device_type
from transformer.data.dataloader import tokenizing_distributed_data_loader
from transformer.tokenizer import create_tokenizer, get_token_bytes
from transformer.training.loss_eval import evaluate_bpb
from transformer.model.engine import Engine
from transformer.model.transformer_block import Transformer
from transformer.config import MODEL_CONFIG
from safetensors import safe_open
from safetensors.torch import load_file
import json

# Configuration
device_batch_size = 32
split_tokens = 20*524288  # number of tokens to evaluate per split
checkpoint_path = None  # optional checkpoint path
device_type = "" # cuda|cpu|mps (empty => autodetect)

# Load the base model and the tokenizer
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Load model
if checkpoint_path and os.path.exists(checkpoint_path):
    from transformer.training.train import load_checkpoint
    model, tokenizer, meta = load_model_from_checkpoint(checkpoint_path, device)
    sequence_len = meta.get("model_config", {}).get("block_size", 1024)
else:
    # Try to find latest checkpoint
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
        if checkpoint_files:
            latest = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            model, tokenizer, meta = load_model_from_checkpoint(checkpoint_path, device)
            sequence_len = meta.get("model_config", {}).get("block_size", 1024)
        else:
            raise FileNotFoundError("No checkpoint found")
    else:
        raise FileNotFoundError("No checkpoint directory found")

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Evaluate the loss on each split
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0, "split_tokens must be divisible by tokens_per_step"
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(tokenizer, device=device)
bpb_results = {}
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(
        device_batch_size, sequence_len, split_name, 
        tokenizer=tokenizer, device=str(device)
    )
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# Master process also samples from the model
samples = []
if ddp_rank == 0:
    prompts = [
        "Türkiye'nin başkenti",
        "Altının kimyasal sembolü",
        "Dün Cuma ise, yarın",
        "Sıcak kelimesinin zıttı",
        "Güneş sistemindeki gezegenler:",
        "En sevdiğim renk",
        "5*x + 3 = 13 ise, x",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

    # Log to report
    from transformer.report import get_report
    get_report().log(section="Base model loss", data=[
        {
            "train bpb": bpb_results.get("train", 0.0),
            "val bpb": bpb_results.get("val", 0.0),
        },
    ])

# Cleanup
compute_cleanup()

