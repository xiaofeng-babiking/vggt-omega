# vggt_omega/training/parallel.py
"""PyTorch-native FSDP2 (`fully_shard`) helpers for the VGGT-Omega trainer.

DDP replicates the whole model on every rank; FSDP shards parameters, gradients,
and optimizer state across the data-parallel group (~1/N memory). These helpers
build the DeviceMesh + mixed-precision policy and apply `fully_shard` bottom-up
over every `SelfAttentionBlock` (the single repeating transformer block used by
the aggregator, the patch-embed ViT encoder, and the head trunks) and then the
root model. They compose with the model's existing `use_reentrant=False`
activation checkpointing (FSDP outside, checkpoint inside).
"""
import os

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from vggt_omega.models.layers import SelfAttentionBlock

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _dtype(name):
    if name in (None, "none"):
        return None
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r} (expected one of {list(_DTYPES)} or 'none')")
    return _DTYPES[name]


def build_dp_mesh(world_size, *, hybrid_shard=False, shard_size=None, device_type="cuda") -> DeviceMesh:
    """Data-parallel DeviceMesh.

    Full-shard (default): 1-D ("dp_shard",) over the whole world.
    HSDP (hybrid_shard=True): 2-D ("dp_replicate","dp_shard"); params are sharded
    within each `shard_size`-rank group (default = torchrun's LOCAL_WORLD_SIZE, i.e.
    one node) and replicated across groups, so the per-block all-gather stays
    intra-node and only the gradient reduce crosses nodes. Dim order is load-bearing:
    dim0=replicate, dim1=shard.
    """
    if not hybrid_shard:
        return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp_shard",))
    if shard_size in (None, 0):
        shard_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    shard_size = int(shard_size)
    if shard_size <= 0 or world_size % shard_size != 0:
        raise ValueError(
            f"world_size {world_size} not divisible by fsdp.shard_size {shard_size}"
        )
    replicate = world_size // shard_size
    return init_device_mesh(
        device_type, (replicate, shard_size), mesh_dim_names=("dp_replicate", "dp_shard")
    )


def build_mp_policy(param_dtype="bfloat16", reduce_dtype="bfloat16") -> MixedPrecisionPolicy:
    """Sharded master weights stay fp32; compute casts to param_dtype, reduce in reduce_dtype."""
    return MixedPrecisionPolicy(param_dtype=_dtype(param_dtype), reduce_dtype=_dtype(reduce_dtype))


def apply_fsdp(
    model, mesh, *, reshard_after_forward=True, param_dtype="bfloat16",
    reduce_dtype="bfloat16", cpu_offload=False,
) -> nn.Module:
    """Shard every SelfAttentionBlock, then the root model, in place. Returns the model."""
    kw = dict(
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=build_mp_policy(param_dtype, reduce_dtype),
        offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
    )
    for module in model.modules():
        if isinstance(module, SelfAttentionBlock):
            fully_shard(module, **kw)
    fully_shard(model, **kw)
    return model


def grad_norm_to_float(total_norm) -> float:
    """clip_grad_norm_ returns a Replicate() DTensor under FSDP2; reduce to a python float.

    MUST be called on every rank (not inside a rank-0-only branch): converting a
    scalar DTensor from rank 0 alone triggers an unmatched reduction collective and
    deadlocks.
    """
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    return float(total_norm)
