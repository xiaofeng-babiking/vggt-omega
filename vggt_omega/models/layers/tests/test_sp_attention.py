"""A full SelfAttentionBlock with the SP hook matches single-GPU, and the
gather-block-slice pattern (used by register blocks / camera trunk) matches too.
"""
from __future__ import annotations

import torch
import pytest

from vggt_omega.models.layers.block import SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import init_weights_vit
from vggt_omega.distributed.tests._dist import run_distributed

skip_no_multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 CUDA devices + NCCL"
)


def _make_block(dim: int, heads: int, dev: str) -> SelfAttentionBlock:
    torch.manual_seed(0)  # identical weights on every rank
    block = SelfAttentionBlock(
        dim=dim, num_heads=heads, ffn_ratio=4.0, qkv_bias=True, proj_bias=True,
        ffn_bias=True, init_values=1e-5, use_qk_norm=True, mask_k_bias=True,
    ).to(dev).eval()
    # Initialize bias_mask (filled with nan by default; needs 0/1 values to avoid nan outputs).
    block.apply(init_weights_vit)
    return block


def _global_block_parity(rank: int, world_size: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from vggt_omega.distributed import ParallelContext

    dev = f"cuda:{rank}"
    dim, heads, N = 32, 4, 6  # N divisible by world_size
    block = _make_block(dim, heads, dev)
    torch.manual_seed(1)  # identical full input on every rank
    x_full = torch.randn(1, N, dim, device=dev)
    with torch.inference_mode():
        ref = block(x_full, None)
    n_local = N // world_size
    sl = slice(rank * n_local, (rank + 1) * n_local)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("sp",))
    block.attn.seq_parallel_ctx = ParallelContext(mesh)
    with torch.inference_mode():
        out_local = block(x_full[:, sl].contiguous(), None)
    assert torch.allclose(out_local, ref[:, sl], atol=1e-3)


def _gather_block_slice_parity(rank: int, world_size: int) -> None:
    """The register-branch / camera-trunk pattern: all-gather frames, run a
    normal (ctx=None) block on the full set, slice this shard back."""
    from torch.distributed.device_mesh import init_device_mesh
    from vggt_omega.distributed import ParallelContext

    dev = f"cuda:{rank}"
    dim, heads, S, T = 32, 4, 4, 3  # S frames, T special tokens/frame
    block = _make_block(dim, heads, dev)
    torch.manual_seed(1)
    x_full = torch.randn(1, S, T, dim, device=dev)  # identical on every rank
    with torch.inference_mode():
        ref = block(x_full.reshape(1, S * T, dim), None).view(1, S, T, dim)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("sp",))
    ctx = ParallelContext(mesh)
    s_local = S // world_size
    local = x_full[:, rank * s_local : (rank + 1) * s_local].contiguous()
    gathered = ctx.all_gather_frames(local, frame_dim=1)
    with torch.inference_mode():
        out = block(gathered.reshape(1, S * T, dim), None).view(1, S, T, dim)
    out_local = ctx.slice_local_frames(out, frame_dim=1)
    assert torch.allclose(out_local, ref[:, rank * s_local : (rank + 1) * s_local], atol=1e-3)


@skip_no_multigpu
def test_global_block_parity():
    run_distributed(_global_block_parity, world_size=2)


@skip_no_multigpu
def test_gather_block_slice_parity():
    run_distributed(_gather_block_slice_parity, world_size=2)
