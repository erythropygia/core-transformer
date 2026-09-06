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

    if tokenizer is None:
        if os.path.isdir(tokenizer_path):
            tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
        else:
            tokenizer = create_tokenizer(model_path=tokenizer_path)

    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found. Please provide parquet files in {data_dir or 'base_data'} directory."
        )

    split_idx = int(len(parquet_paths) * 0.9)
    parquet_paths = parquet_paths[:split_idx] if split == "train" else parquet_paths[split_idx:]

    if not parquet_paths:
        raise ValueError(f"No parquet files found for split '{split}'. Total files: {len(parquet_paths)}")

    if split == "train" and shuffle_parquet_files:
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(parquet_paths)
        print(f"Shuffled {len(parquet_paths)} parquet files (seed: {shuffle_seed})")
        print(f"   First 3 files: {[os.path.basename(p) for p in parquet_paths[:3]]}")

    def document_batches():
        resume_pq_idx = resume_state_dict.get("pq_idx", 0) if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict.get("rg_idx", None) if resume_state_dict is not None else None
        resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict is not None else 0
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
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, current_epoch)

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
                            print(f"\nEpoch {current_epoch}: Reshuffled parquet files")
                            print(f"   First 3 files: {[os.path.basename(p) for p in epoch_parquet_order[:3]]}")

    batches = document_batches()

    needed_tokens = B * T + 1
    token_buffer = deque()

    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)

            try:
                token_lists = tokenizer.encode(
                    doc_batch,
                    prepend=bos_token,
                    append=eos_token,
                    num_threads=tokenizer_threads if isinstance(doc_batch, list) else 1
                )
            except TypeError:
                token_lists = [tokenizer.encode(text, prepend=bos_token, append=eos_token) for text in doc_batch]

            for tokens in token_lists:
                if isinstance(tokens, list):
                    token_buffer.extend(tokens)
                else:
                    token_buffer.append(tokens)

        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)

        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

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
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    if tokenizer is None:
        if os.path.isdir(tokenizer_path):
            tokenizer = create_tokenizer(tokenizer_dir=tokenizer_path)
        else:
            tokenizer = create_tokenizer(model_path=tokenizer_path)

    bos_token = tokenizer.get_bos_token_id()
    eos_token = tokenizer.get_eos_token_id()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No parquet files found. Please provide parquet files in {data_dir or 'base_data'} directory."
        )

    split_idx = int(len(parquet_paths) * 0.9)
    parquet_paths = parquet_paths[:split_idx] if split == "train" else parquet_paths[split_idx:]

    if not parquet_paths:
        raise ValueError(f"No parquet files found for split '{split}'.")

    if split == "train" and shuffle_parquet_files:
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(parquet_paths)
        if ddp_rank == 0:
            print(f"BOS Bestfit: Shuffled {len(parquet_paths)} parquet files")

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

                        try:
                            token_lists = tokenizer.encode(
                                doc_batch,
                                prepend=bos_token,
                                append=eos_token,
                                num_threads=tokenizer_threads if isinstance(doc_batch, list) else 1
                            )
                        except TypeError:
                            token_lists = [tokenizer.encode(text, prepend=bos_token, append=eos_token) for text in doc_batch]

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

    row_capacity = T + 1
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 0

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_tokens, (pq_idx, rg_idx, epoch) = next(docs)
        doc_buffer.append(doc_tokens)

    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])

            rows.append(row[:row_capacity])

        use_cuda_optimizations = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda_optimizations)

        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda_optimizations)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda_optimizations)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        yield inputs, targets, state_dict
