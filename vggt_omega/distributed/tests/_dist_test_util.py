"""Helpers to run a function across a spawned gloo process group and collect results."""
import os
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank, world_size, port, out_dir, fn, args):
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        result = fn(rank, world_size, *args)
        torch.save(result, os.path.join(out_dir, f"r{rank}.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed(fn, world_size, *args):
    """Spawn `world_size` gloo procs, run fn(rank, world_size, *args), return list-by-rank."""
    out_dir = tempfile.mkdtemp()
    port = free_port()
    mp.spawn(_entry, args=(world_size, port, out_dir, fn, args), nprocs=world_size, join=True)
    return [torch.load(os.path.join(out_dir, f"r{r}.pt")) for r in range(world_size)]
