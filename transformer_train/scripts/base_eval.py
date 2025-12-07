import os
import sys
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.common import get_base_dir, print0, autodetect_device_type, download_file_with_lock
from transformer.tokenizer import create_tokenizer
from transformer.model.transformer_block import Transformer
from transformer.eval.core_eval import evaluate_task
from transformer.config import MODEL_CONFIG
from safetensors import safe_open
from safetensors.torch import load_file

# -----------------------------------------------------------------------------
# Specific function dealing with I/O etc.

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    
    if not os.path.exists(config_path):
        print0(f"Warning: CORE eval bundle not found at {eval_bundle_dir}")
        print0("CORE evaluation requires downloading eval_bundle.zip (~162MB)")
        print0("This is optional - you can skip CORE evaluation if needed")
        return {
            "results": {},
            "centered_results": {},
            "core_metric": 0.0
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    if os.path.exists(eval_meta_data):
        with open(eval_meta_data, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_name = row['Eval Task']
                random_baseline = row['Random baseline']
                random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        if not os.path.exists(data_path):
            print0(f"SKIP (data not found)")
            continue
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines.get(label, 0.25)  # Default 25% for MC
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results) if centered_results else 0.0
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    args = parser.parse_args()

    # distributed / precision setup
    device_type = autodetect_device_type()
    from transformer.common import compute_init, compute_cleanup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer
    if args.checkpoint is not None:
        model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)
        model_name = os.path.basename(args.checkpoint)
        model_slug = os.path.splitext(model_name)[0]
    else:
        # Try to find latest checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                checkpoint_path = os.path.join(checkpoint_dir, latest)
                model, tokenizer = load_model_from_checkpoint(checkpoint_path, device)
                model_name = latest
                model_slug = os.path.splitext(latest)[0]
            else:
                raise FileNotFoundError("No checkpoint found. Please specify --checkpoint")
        else:
            raise FileNotFoundError("No checkpoint directory found. Please specify --checkpoint")

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

        # Log to report
        from transformer.report import get_report
        get_report().log(section="Base model evaluation", data=[
            {
                "Model": model_name,
                "CORE metric": core_metric,
            },
            centered_results, # the full table
        ])

    compute_cleanup()

if __name__ == "__main__":
    main()

