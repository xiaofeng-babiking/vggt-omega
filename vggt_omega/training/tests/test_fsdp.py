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
    block_params = list(model.blocks.parameters())
    assert block_params, "no block params found post-FSDP"
    block_params_are_dtensor = all(isinstance(p, DTensor) for p in block_params)
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


def _reference_grads():
    """Single-process, full-batch mean-loss gradients (the parity target)."""
    torch.manual_seed(0)
    model = _tiny_stack()
    torch.manual_seed(1)
    x = torch.randn(4, 6, 32)
    torch.manual_seed(2)
    target = torch.randn(4, 6, 32)
    F.mse_loss(model(x), target).backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


def _fsdp_grads(rank, world_size):
    torch.manual_seed(0)
    model = _tiny_stack()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
    apply_fsdp(model, mesh, param_dtype="float32", reduce_dtype="float32")
    torch.manual_seed(1)
    x = torch.randn(4, 6, 32)
    torch.manual_seed(2)
    target = torch.randn(4, 6, 32)
    lo, hi = rank * 2, rank * 2 + 2  # equal local batch of 2 per rank
    F.mse_loss(model(x[lo:hi]), target[lo:hi]).backward()
    # full_tensor() reconstructs the unsharded grad (all ranks agree)
    return {
        name: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).detach().clone()
        for name, p in model.named_parameters()
    }


def test_fsdp_grads_match_single_process():
    ref = _reference_grads()
    results = run_distributed(_fsdp_grads, 2)
    got = results[0]
    assert set(got) == set(ref)
    for name in ref:
        assert torch.allclose(got[name], ref[name], atol=2e-4, rtol=1e-3), name


import tempfile, os as _os
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions


def _save_full_and_return_keys(rank, world_size, tmpdir):
    torch.manual_seed(0)
    model = _tiny_stack()
    ref_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}  # pre-shard weights
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
    apply_fsdp(model, mesh, param_dtype="float32", reduce_dtype="float32")
    # collective on ALL ranks; only rank 0 gets the populated dict
    full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    info = {"is_rank0": rank == 0, "num_keys": len(full)}
    if rank == 0:
        assert not any(isinstance(v, DTensor) for v in full.values()), "consolidated dict has DTensors"
        # the consolidated dict reconstructs the original unsharded weights
        assert set(full) == set(ref_sd)
        for k in ref_sd:
            assert torch.allclose(full[k], ref_sd[k], atol=1e-6), k
        # and a fresh, non-FSDP model loads it (inference.py compatibility)
        fresh = _tiny_stack()
        fresh.load_state_dict(full)
        info["loaded_into_plain_model"] = True
    return info


def test_full_state_dict_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        results = run_distributed(_save_full_and_return_keys, 2, tmp)
    rank0 = next(r for r in results if r["is_rank0"])
    assert rank0["loaded_into_plain_model"] is True
    assert rank0["num_keys"] > 0
