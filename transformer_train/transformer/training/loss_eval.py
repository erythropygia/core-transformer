import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    device = next(model.parameters()).device
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model.forward(x, targets=y, loss_reduction='none').view(-1)
        y = y.view(-1)
        if (y.int() < 0).any():
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
