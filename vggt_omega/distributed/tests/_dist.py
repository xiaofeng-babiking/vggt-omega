"""Spawn a small NCCL process group to exercise collectives in tests.

gloo does not implement all-to-all, so distributed tests require real GPUs.
``run_distributed`` spawns ``world_size`` processes, each pinned to ``cuda:rank``;
any assertion failure in a child re-raises in the parent (mp.spawn join=True).
"""
from __future__ import annotations

from collections.abc import Callable
import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    # Slight TOCTOU race between close and re-bind; acceptable in test-only contexts.
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world_size: int, port: int, fn: Callable[[int, int], None]) -> None:
    import torch

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Callable[[int, int], None], world_size: int) -> None:
    """Run ``fn(rank, world_size)`` in ``world_size`` spawned NCCL processes."""
    mp.spawn(_worker, args=(world_size, _free_port(), fn), nprocs=world_size, join=True)
