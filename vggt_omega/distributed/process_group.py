"""Context-parallel process-group setup and small collective helpers.

For inference the CP group spans the whole world (one sequence at a time, no DP).
"""
import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Init NCCL from torchrun env vars. Returns (rank, world_size, local_rank)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cp_group():
    """The context-parallel group (== world group for inference)."""
    return dist.group.WORLD


def all_gather_ints(value: int, group, device) -> list[int]:
    """All-gather one int per rank into an ordered Python list."""
    world = dist.get_world_size(group)
    t = torch.tensor([value], dtype=torch.long, device=device)
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t, group=group)
    return [int(x.item()) for x in out]
