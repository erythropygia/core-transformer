import os
import torch
import torch.distributed as dist


def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def print0(s="", **kwargs):
    if is_ddp():
        ddp_rank = int(os.environ.get('RANK', 0))
        if ddp_rank == 0:
            print(s, **kwargs)
    else:
        # Single GPU: always print
        print(s, **kwargs)


def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def compute_init(device_type="cuda"):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    if ddp:
        # Initialize distributed training
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if device_type == 'cuda' else 'gloo')
        
        # Set device for this rank
        device = torch.device(f"{device_type}:{ddp_local_rank}")
        if device_type == 'cuda':
            torch.cuda.set_device(device)
    else:
        device = torch.device(device_type)
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_base_dir():
    base_dir = os.environ.get('CORE_TRANSFORMER_BASE_DIR', os.getcwd())
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    import urllib.request
    import filelock
    
    base_dir = get_base_dir()
    filepath = os.path.join(base_dir, filename)
    lockpath = filepath + ".lock"
    
    # If file already exists, skip download
    if os.path.exists(filepath):
        print0(f"File already exists: {filepath}")
        if postprocess_fn:
            postprocess_fn(filepath)
        return filepath
    
    # Acquire lock and download
    with filelock.FileLock(lockpath, timeout=3600):
        # Check again after acquiring lock (another process might have downloaded it)
        if os.path.exists(filepath):
            print0(f"File already exists (downloaded by another process): {filepath}")
            if postprocess_fn:
                postprocess_fn(filepath)
            return filepath
        
        print0(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print0(f"Downloaded {filename}")
        
        if postprocess_fn:
            postprocess_fn(filepath)
    
    return filepath

