"""Helpers to run a function across a spawned gloo process group and collect results.

These multi-rank parity tests run on the gloo/CPU backend so they need no GPU.
A from-scratch model must be initialized with :func:`init_finite` before use:
several params (e.g. ``LayerScale.gamma``) are allocated with ``torch.empty`` and
left uninitialized until ``reset_parameters()``/the checkpoint fills them, which
under pytest memory churn yields garbage that overflows to NaN. See that helper.
"""
import os
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_finite(module):
    """Initialize a from-scratch module so every parameter is finite & well-conditioned.

    Several submodules allocate parameters with ``torch.empty`` and rely on
    ``reset_parameters()`` (or the released checkpoint) to fill them — notably
    ``LayerScale.gamma`` and ``LinearKMaskedBias.bias_mask`` (NaN by design). On a
    fresh process ``torch.empty`` happens to return zeros, but under pytest memory
    churn it returns reused garbage (occasionally huge), which overflows to ``inf``
    and then ``NaN`` downstream. Tests MUST initialize these before use.
    """
    for m in module.modules():
        reset = getattr(m, "reset_parameters", None)
        if callable(reset):
            reset()
    for name, buf in module.named_buffers():
        if name.endswith("bias_mask"):
            torch.nn.init.ones_(buf)
    return module


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank, world_size, init_method, out_dir, fn, args):
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        init_method=init_method,
    )
    try:
        result = fn(rank, world_size, *args)
        torch.save(result, os.path.join(out_dir, f"r{rank}.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed(fn, world_size, *args):
    """Spawn `world_size` gloo procs, run fn(rank, world_size, *args), return list-by-rank."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # File-based gloo rendezvous avoids the TOCTOU race of binding a free TCP port.
        # gloo creates the rendezvous file itself, so it must NOT be pre-created.
        init_method = f"file://{os.path.join(tmpdir, 'rendezvous')}"
        mp.spawn(_entry, args=(world_size, init_method, tmpdir, fn, args), nprocs=world_size, join=True)
        # weights_only=False: trusted test-only data; allows arbitrary result
        # structures (tuples/dicts/tensors) returned by fn to load correctly.
        return [
            torch.load(os.path.join(tmpdir, f"r{r}.pt"), weights_only=False)
            for r in range(world_size)
        ]
