from collections import deque
import torch
from typing import Optional, Iterator, Tuple
from .common import get_dist_info


def tokenizing_streaming_data_loader(
    tokenizer,
    data_source,
    B: int,
    T: int,
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[dict] = None,
    bos_token_id: Optional[int] = None
):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # Get BOS token
    if bos_token_id is None:
        if hasattr(tokenizer, 'get_bos_token_id'):
            bos_token_id = tokenizer.get_bos_token_id()
        else:
            bos_token_id = 0  # fallback
    
    # Get distributed info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # Token buffer
    needed_tokens = B * T + 1  # +1 for target at last token
    token_buffer = deque()  # stream tokens on the right, pop from the left
    
    # Text iterator
    if isinstance(data_source, str):
        # Assume it's a parquet file path or directory
        try:
            import pyarrow.parquet as pq
            if data_source.endswith('.parquet'):
                parquet_files = [data_source]
            else:
                import os
                parquet_files = sorted([
                    os.path.join(data_source, f) 
                    for f in os.listdir(data_source) 
                    if f.endswith('.parquet')
                ])
            
            def document_batches():
                for filepath in parquet_files:
                    pf = pq.ParquetFile(filepath)
                    for rg_idx in range(pf.num_row_groups):
                        rg = pf.read_row_group(rg_idx)
                        batch = rg.column('text').to_pylist()
                        for i in range(0, len(batch), tokenizer_batch_size):
                            yield batch[i:i+tokenizer_batch_size], (filepath, rg_idx)
            
            text_iter = document_batches()
        except ImportError:
            raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow")
    else:
        # Assume it's an iterator of text strings
        def document_batches():
            for text in data_source:
                yield [text], None
        
        text_iter = document_batches()
    
    # Main loop
    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            try:
                doc_batch, state_info = next(text_iter)
            except StopIteration:
                # Restart from beginning for infinite iteration
                if isinstance(data_source, str):
                    text_iter = document_batches()
                    doc_batch, state_info = next(text_iter)
                else:
                    return  # End of data source
            
            # Tokenize batch
            if hasattr(tokenizer, 'encode') and hasattr(tokenizer.encode, '__call__'):
                # Check if tokenizer supports batch encoding
                try:
                    token_lists = tokenizer.encode(
                        doc_batch, 
                        prepend=bos_token_id, 
                        num_threads=tokenizer_threads if isinstance(doc_batch, list) else 1
                    )
                except TypeError:
                    # Fallback to single encoding
                    token_lists = [tokenizer.encode(text, prepend=bos_token_id) for text in doc_batch]
            else:
                token_lists = [tokenizer.encode(text, prepend=bos_token_id) for text in doc_batch]
            
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
        
        # State dict for resuming (if available)
        state_dict = state_info if state_info else {}
        
        yield inputs, targets, state_dict

