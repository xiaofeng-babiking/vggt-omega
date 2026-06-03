"""Ulysses all-to-all: round-trip identity and exact attention parity."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytest

from vggt_omega.distributed.ulysses import scatter_heads, gather_heads
from vggt_omega.distributed.tests._dist import run_distributed

skip_no_multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 CUDA devices + NCCL"
)


def _roundtrip(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    B, H, N_local, d = 1, 4, 3, 8
    x = torch.randn(B, H, N_local, d, device=f"cuda:{rank}")
    g = dist.group.WORLD
    y = scatter_heads(x, g, world_size)
    assert y.shape == (B, H // world_size, world_size * N_local, d)
    z = gather_heads(y, g, world_size)
    assert torch.allclose(z, x, atol=1e-5)


def _attn_parity(rank: int, world_size: int) -> None:
    torch.manual_seed(0)  # identical q/k/v on every rank
    B, H, N, d = 1, 4, 6, 8
    dev = f"cuda:{rank}"
    q = torch.randn(B, H, N, d, device=dev)
    k = torch.randn(B, H, N, d, device=dev)
    v = torch.randn(B, H, N, d, device=dev)
    ref = F.scaled_dot_product_attention(q, k, v)
    n_local = N // world_size
    sl = slice(rank * n_local, (rank + 1) * n_local)
    g = dist.group.WORLD
    ql = scatter_heads(q[:, :, sl].contiguous(), g, world_size)
    kl = scatter_heads(k[:, :, sl].contiguous(), g, world_size)
    vl = scatter_heads(v[:, :, sl].contiguous(), g, world_size)
    out = gather_heads(F.scaled_dot_product_attention(ql, kl, vl), g, world_size)
    assert torch.allclose(out, ref[:, :, sl], atol=1e-4)


@skip_no_multigpu
def test_scatter_gather_roundtrip():
    run_distributed(_roundtrip, world_size=2)


@skip_no_multigpu
def test_attention_parity():
    run_distributed(_attn_parity, world_size=2)
