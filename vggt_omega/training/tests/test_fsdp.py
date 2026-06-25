# vggt_omega/training/tests/test_fsdp.py
"""FSDP2 correctness tests on the gloo/CPU backend (no GPU needed).

These mirror vggt_omega/distributed/tests/_dist_test_util.run_distributed: spawn
N gloo procs, run fn(rank, world_size, ...), collect per-rank results. FSDP runs
on a CPU DeviceMesh ("cpu") in fp32 here — the bf16 path is validated by the GPU
smoke (see plan Task 8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from vggt_omega.distributed.tests._dist_test_util import run_distributed
from vggt_omega.models.layers import SelfAttentionBlock
from vggt_omega.training.parallel import apply_fsdp, grad_norm_to_float


def _tiny_stack(dim=32, num_heads=4, depth=2):
    """A minimal model built from the real SelfAttentionBlock (no NaN buffers,
    no LayerScale): mask_k_bias=False, use_qk_norm=False, init_values=None."""
    class TinyStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                SelfAttentionBlock(
                    dim=dim, num_heads=num_heads, init_values=None,
                    mask_k_bias=False, use_qk_norm=False,
                )
                for _ in range(depth)
            )
            self.head = nn.Linear(dim, dim)

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return self.head(x)

    return TinyStack()


def _shard_and_introspect(rank, world_size):
    torch.manual_seed(0)
    model = _tiny_stack()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
    apply_fsdp(model, mesh, param_dtype="float32", reduce_dtype="float32")
    # every block param is now a DTensor
    block_params_are_dtensor = all(
        isinstance(p, DTensor) for p in model.blocks.parameters()
    )
    x = torch.randn(4, 6, 32)
    out = model(x)
    F.mse_loss(out, torch.zeros_like(out)).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    return {
        "block_params_are_dtensor": block_params_are_dtensor,
        "out_shape": tuple(out.shape),
        "grad_finite": bool(torch.isfinite(torch.tensor(grad_norm_to_float(gn))).item()),
    }


def test_apply_fsdp_shards_blocks_and_runs():
    results = run_distributed(_shard_and_introspect, 2)
    for r in results:
        assert r["block_params_are_dtensor"] is True
        assert r["out_shape"] == (4, 6, 32)
        assert r["grad_finite"] is True
