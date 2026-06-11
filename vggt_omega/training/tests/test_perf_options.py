import pytest
import torch
from omegaconf import OmegaConf

from vggt_omega.training.tests.test_trainer import SyntheticTrainData
from vggt_omega.training.trainer import Trainer, resolve_comm_hook


def test_resolve_comm_hook_bf16():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    assert resolve_comm_hook("bf16") is default_hooks.bf16_compress_hook


def test_resolve_comm_hook_none_and_unknown():
    assert resolve_comm_hook("none") is None
    assert resolve_comm_hook(None) is None
    with pytest.raises(ValueError, match="grad_compression"):
        resolve_comm_hook("fp8")


def _tiny_cfg(tmp_path):
    cfg = OmegaConf.load("vggt_omega/training/config/train_smoke.yaml")
    cfg.run.output_dir = str(tmp_path)
    cfg.run.max_steps = 1
    cfg.model.checkpoint = None
    cfg.model.embed_dim = 64
    return cfg


def test_fused_adamw_falls_back_on_cpu(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    cfg.optim.fused = True
    t = Trainer(cfg, data_override=SyntheticTrainData())
    assert t.optimizer.defaults["fused"] is not None   # wiring exists: we always pass a bool
    fused = t.optimizer.param_groups[0].get("fused", t.optimizer.defaults.get("fused"))
    if t.device.type == "cuda":
        assert fused
    else:
        assert not fused
