import os
from collections import deque
import torch
import pyarrow.parquet as pq
from typing import Optional

from .common import get_dist_info
from .dataset_utils import list_parquet_files
from .tokenizer import create_tokenizer


def tokenizing_distributed_data_loader_with_state(
    B: int,
    T: int,
    split: str = "train",
    tokenizer=None,
    tokenizer_path: str = "tokenizer",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[dict] = None,
    data_dir: Optional[str] = None
):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # Get tokenizer
    if tokenizer is None:
        if os.path.isdir(tokenizer_path):
            tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
        else:
            tokenizer = create_tokenizer(model_path=tokenizer_path)
    
    # Get BOS token
    bos_token = tokenizer.get_bos_token_id()
    
    # Get distributed info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # Get parquet files
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found. Please provide parquet files in {data_dir or 'base_data'} directory."
        )
    
    # Split train/val (last file is val)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    
    # Infinite iterator over document batches
    def document_batches():
        resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict is not None else None
        pq_idx = resume_pq_idx
        
        while True:  # iterate infinitely (multi-epoch)
            while pq_idx < len(parquet_paths):  # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                
                # Start from resume point if resuming on same file, otherwise from DDP rank
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1  # advance by 1 so we don't repeat data
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None  # only do this once
                else:
                    rg_idx = ddp_rank
                
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()  # each batch is a parquet group
                    
                    # Tokenizer encode might want smaller batches
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    
                    rg_idx += ddp_world_size  # advance to next row group (DDP)
                
                pq_idx += 1  # advance to next parquet file
    
    batches = document_batches()
    
    # Now emit batches of tokens
    needed_tokens = B * T + 1  # +1 for target at last token
    token_buffer = deque()  # stream tokens on the right, pop from the left
    
    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            
            # Tokenize batch
            try:
                token_lists = tokenizer.encode(
                    doc_batch,
                    prepend=bos_token,
                    num_threads=tokenizer_threads if isinstance(doc_batch, list) else 1
                )
            except TypeError:
                # Fallback to single encoding
                token_lists = [tokenizer.encode(text, prepend=bos_token) for text in doc_batch]
            
            # Add tokens to buffer
            for tokens in token_lists:
                if isinstance(tokens, list):
                    token_buffer.extend(tokens)
                else:
                    token_buffer.append(tokens)
        
        # Extract needed tokens
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        # Create tensors
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        
        # Create inputs/targets
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        
        # Reshape and move to device
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        
        # State dict for resuming
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}
        
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets

