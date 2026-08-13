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


# --- resolve_hybrid_shard ---------------------------------------------------
from vggt_omega.training.parallel import resolve_hybrid_shard  # noqa: E402


def test_resolve_hybrid_shard_bool_passthrough():
    assert resolve_hybrid_shard(True, world_size=8, local_world_size=8) is True
    assert resolve_hybrid_shard(False, world_size=16, local_world_size=8) is False


def test_resolve_hybrid_shard_none_is_false():
    assert resolve_hybrid_shard(None, world_size=16, local_world_size=8) is False


def test_resolve_hybrid_shard_strings():
    assert resolve_hybrid_shard("true", world_size=8, local_world_size=8) is True
    assert resolve_hybrid_shard("false", world_size=16, local_world_size=8) is False


def test_resolve_hybrid_shard_auto_enables_on_multi_node():
    assert resolve_hybrid_shard("auto", world_size=16, local_world_size=8) is True


def test_resolve_hybrid_shard_auto_stays_off_on_single_node():
    assert resolve_hybrid_shard("auto", world_size=8, local_world_size=8) is False


def test_resolve_hybrid_shard_auto_without_local_world_stays_off():
    assert resolve_hybrid_shard("auto", world_size=8, local_world_size=None) is False


def test_resolve_hybrid_shard_rejects_unknown_value():
    with pytest.raises(ValueError, match="hybrid_shard"):
        resolve_hybrid_shard("maybe", world_size=8, local_world_size=8)
