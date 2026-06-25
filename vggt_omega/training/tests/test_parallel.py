# vggt_omega/training/tests/test_parallel.py
"""Unit tests for the FSDP2 helpers that need no process group."""
import pytest
import torch

from vggt_omega.training.parallel import (
    build_mp_policy,
    build_dp_mesh,
    grad_norm_to_float,
)


def test_build_mp_policy_maps_dtypes():
    pol = build_mp_policy("bfloat16", "float32")
    assert pol.param_dtype == torch.bfloat16
    assert pol.reduce_dtype == torch.float32


def test_build_mp_policy_none_passthrough():
    pol = build_mp_policy("float32", "none")
    assert pol.param_dtype == torch.float32
    assert pol.reduce_dtype is None


def test_build_mp_policy_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unknown dtype"):
        build_mp_policy("float8", "bfloat16")


def test_build_dp_mesh_hybrid_requires_divisible_world():
    # 7 ranks cannot be split into shard groups of 4 -> error before any PG init.
    with pytest.raises(ValueError, match="not divisible"):
        build_dp_mesh(7, hybrid_shard=True, shard_size=4)


def test_grad_norm_to_float_plain_tensor():
    assert grad_norm_to_float(torch.tensor(3.5)) == pytest.approx(3.5)
