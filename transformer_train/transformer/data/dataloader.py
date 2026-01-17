import os
from collections import deque
import torch
import pyarrow.parquet as pq
from typing import Optional
import random

from ..common import get_dist_info
from .dataset_utils import list_parquet_files
from ..tokenizer import create_tokenizer


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
    data_dir: Optional[str] = None,
    shuffle_parquet_files: bool = True,
    shuffle_seed: Optional[int] = 42,
    reshuffle_each_epoch: bool = True
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
    
    # Split train/val: Use last 10% of parquet files for validation
    split_idx = int(len(parquet_paths) * 0.9)
    parquet_paths = parquet_paths[:split_idx] if split == "train" else parquet_paths[split_idx:]
    
    if not parquet_paths:
        raise ValueError(f"No parquet files found for split '{split}'. Total files: {len(parquet_paths)}")
    
    # Initial shuffle of parquet files (for training only)
    if split == "train" and shuffle_parquet_files:
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(parquet_paths)
        print(f"Shuffled {len(parquet_paths)} parquet files (seed: {shuffle_seed})")
        print(f"   First 3 files: {[os.path.basename(p) for p in parquet_paths[:3]]}")
    
    # Infinite iterator over document batches
    def document_batches():
        resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict is not None else None
        resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict is not None else 0
        pq_idx = resume_pq_idx
        current_epoch = resume_epoch
        epoch_parquet_order = parquet_paths.copy()  # Initial order
        
        while True:  # iterate infinitely (multi-epoch)
            while pq_idx < len(epoch_parquet_order):  # iterate over all parquet files
                filepath = epoch_parquet_order[pq_idx]
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
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, current_epoch)
                    
                    rg_idx += ddp_world_size  # advance to next row group (DDP)
                
                pq_idx += 1  # advance to next parquet file
                
                # End of epoch - reshuffle if enabled
                if pq_idx >= len(epoch_parquet_order):
                    current_epoch += 1
                    pq_idx = 0
                    
                    # Re-shuffle parquet files for next epoch
                    if split == "train" and shuffle_parquet_files and reshuffle_each_epoch:
                        if shuffle_seed is not None:
                            random.seed(shuffle_seed + current_epoch)  # Different seed per epoch
                        epoch_parquet_order = parquet_paths.copy()
                        random.shuffle(epoch_parquet_order)
                        if ddp_rank == 0:  # Only print from main process
                            print(f"\nEpoch {current_epoch}: Reshuffled parquet files")
                            print(f"   First 3 files: {[os.path.basename(p) for p in epoch_parquet_order[:3]]}")
    
    batches = document_batches()
    
    # Now emit batches of tokens
    needed_tokens = B * T + 1  # +1 for target at last token
    token_buffer = deque()  # stream tokens on the right, pop from the left
    
    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
            
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
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


def tokenizing_distributed_data_loader_bos_bestfit(
    B: int,
    T: int,
    split: str = "train",
    tokenizer=None,
    tokenizer_path: str = "tokenizer",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[dict] = None,
    data_dir: Optional[str] = None,
    shuffle_parquet_files: bool = True,
    shuffle_seed: Optional[int] = 42,
    reshuffle_each_epoch: bool = True,
    buffer_size: int = 1000
):
    """
    BOS-aligned bestfit packing dataloader.
    
    Key features:
    - Every row starts with BOS token
    - Document packing to maximize token utilization
    - ~100% utilization with acceptable cropping loss
    
    This is the advanced version inspired by nanochat's bos_bestfit implementation.
    """
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
    
    # Split train/val
    split_idx = int(len(parquet_paths) * 0.9)
    parquet_paths = parquet_paths[:split_idx] if split == "train" else parquet_paths[split_idx:]
    
    if not parquet_paths:
        raise ValueError(f"No parquet files found for split '{split}'.")
    
    # Initial shuffle
    if split == "train" and shuffle_parquet_files:
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(parquet_paths)
        print(f"BOS Bestfit: Shuffled {len(parquet_paths)} parquet files")
    
    # Document iterator
    def document_iterator():
        resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict else 0
        resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict else None
        resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict else 0
        pq_idx = resume_pq_idx
        current_epoch = resume_epoch
        epoch_parquet_order = parquet_paths.copy()
        
        while True:
            while pq_idx < len(epoch_parquet_order):
                filepath = epoch_parquet_order[pq_idx]
                pf = pq.ParquetFile(filepath)
                
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    
                    for i in range(0, len(batch), tokenizer_batch_size):
                        doc_batch = batch[i:i+tokenizer_batch_size]
                        
                        # Tokenize with BOS
                        try:
                            token_lists = tokenizer.encode(
                                doc_batch,
                                prepend=bos_token,
                                num_threads=tokenizer_threads if isinstance(doc_batch, list) else 1
                            )
                        except TypeError:
                            token_lists = [tokenizer.encode(text, prepend=bos_token) for text in doc_batch]
                        
                        # Yield individual documents
                        for tokens in token_lists:
                            if isinstance(tokens, list) and len(tokens) > 0:
                                yield tokens, (pq_idx, rg_idx, current_epoch)
                    
                    rg_idx += ddp_world_size
                
                pq_idx += 1
                
                if pq_idx >= len(epoch_parquet_order):
                    current_epoch += 1
                    pq_idx = 0
                    
                    if split == "train" and shuffle_parquet_files and reshuffle_each_epoch:
                        if shuffle_seed is not None:
                            random.seed(shuffle_seed + current_epoch)
                        epoch_parquet_order = parquet_paths.copy()
                        random.shuffle(epoch_parquet_order)
                        if ddp_rank == 0:
                            print(f"BOS Bestfit Epoch {current_epoch}: Reshuffled")
    
    docs = document_iterator()
    
    # BOS-aligned bestfit packing
    row_size = T + 1  # +1 for target
    document_buffer = []
    
    while True:
        # Fill buffer with documents
        while len(document_buffer) < buffer_size:
            doc_tokens, state = next(docs)
            document_buffer.append((doc_tokens, state))
        
        # Pack documents into rows (bestfit)
        rows = []
        current_row = []
        current_row_len = 0
        last_state = None
        
        i = 0
        while i < len(document_buffer) and len(rows) < B:
            doc_tokens, state = document_buffer[i]
            doc_len = len(doc_tokens)
            last_state = state
            
            # Can this document fit in current row?
            if current_row_len + doc_len <= row_size:
                current_row.extend(doc_tokens)
                current_row_len += doc_len
                i += 1
            else:
                # Current row is full, finalize it
                if current_row_len > 0:
                    # Pad if needed
                    if current_row_len < row_size:
                        # Pad with zeros (or use a pad token)
                        current_row.extend([0] * (row_size - current_row_len))
                    rows.append(current_row[:row_size])
                    current_row = []
                    current_row_len = 0
                
                # If document is too large, crop it and start new row
                if doc_len > row_size:
                    rows.append(doc_tokens[:row_size])
                    # Discard the rest (acceptable cropping loss)
                    i += 1
                else:
                    # Start new row with this document
                    current_row = doc_tokens.copy()
                    current_row_len = doc_len
                    i += 1
        
        # Finalize last row if needed
        if current_row_len > 0 and len(rows) < B:
            if current_row_len < row_size:
                current_row.extend([0] * (row_size - current_row_len))
            rows.append(current_row[:row_size])
        
        # Remove used documents from buffer
        document_buffer = document_buffer[i:]
        
        # Convert rows to tensors
        if len(rows) == B:
            use_cuda_optimizations = device == "cuda"
            scratch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda_optimizations)
            
            # Create inputs/targets
            inputs = scratch[:, :-1].to(device=device, non_blocking=use_cuda_optimizations)
            targets = scratch[:, 1:].to(device=device, non_blocking=use_cuda_optimizations)
            
            # State dict
            state_dict = last_state[1] if last_state else {"pq_idx": 0, "rg_idx": 0, "epoch": 0}
            
            yield inputs, targets, state_dict

