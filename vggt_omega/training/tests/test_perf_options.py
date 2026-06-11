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


def test_apply_grad_compression_registers_configured_hook():
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    from vggt_omega.training.trainer import apply_grad_compression

    class StubDDP:
        def __init__(self):
            self.hook = None

        def register_comm_hook(self, state, hook):
            self.hook = hook

    ddp = StubDDP()
    apply_grad_compression(ddp, "bf16")
    assert ddp.hook is default_hooks.bf16_compress_hook
    ddp = StubDDP()
    apply_grad_compression(ddp, "none")
    assert ddp.hook is None


def test_load_encoder_weights_restores_only_patch_embed(tmp_path):
    import torch

    from vggt_omega.models import VGGTOmega
    from vggt_omega.training.trainer import init_model_from_scratch, load_encoder_weights

    torch.manual_seed(0)
    source = VGGTOmega(embed_dim=64)
    init_model_from_scratch(source)
    path = tmp_path / "source.pt"
    torch.save(source.state_dict(), path)

    torch.manual_seed(1)
    target = VGGTOmega(embed_dim=64)
    init_model_from_scratch(target)
    src_qkv = source.aggregator.frame_blocks[0].attn.qkv.weight
    assert not torch.equal(target.aggregator.frame_blocks[0].attn.qkv.weight, src_qkv)

    n = load_encoder_weights(target, str(path))
    assert n > 0
    for (kt, vt), (ks, vs) in zip(
        target.aggregator.patch_embed.state_dict().items(),
        source.aggregator.patch_embed.state_dict().items(),
    ):
        assert kt == ks and torch.equal(vt, vs), kt
    # everything outside the encoder stays at the target's own random init
    assert not torch.equal(target.aggregator.frame_blocks[0].attn.qkv.weight, src_qkv)


def test_load_encoder_weights_rejects_checkpoint_without_encoder(tmp_path):
    import pytest
    import torch

    from vggt_omega.models import VGGTOmega
    from vggt_omega.training.trainer import load_encoder_weights

    path = tmp_path / "junk.pt"
    torch.save({"foo": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="aggregator.patch_embed"):
        load_encoder_weights(VGGTOmega(embed_dim=64), str(path))
