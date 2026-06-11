import pytest
import torch

from vggt_omega.training.trainer import resolve_comm_hook


def test_resolve_comm_hook_bf16():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    assert resolve_comm_hook("bf16") is default_hooks.bf16_compress_hook


def test_resolve_comm_hook_none_and_unknown():
    assert resolve_comm_hook("none") is None
    assert resolve_comm_hook(None) is None
    with pytest.raises(ValueError, match="grad_compression"):
        resolve_comm_hook("fp8")
