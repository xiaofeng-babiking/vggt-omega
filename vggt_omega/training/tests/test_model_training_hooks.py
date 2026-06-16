import torch

from vggt_omega.distributed.tests.test_model import _init_random_model
from vggt_omega.models import VGGTOmega


def _tiny_model():
    torch.manual_seed(0)
    model = VGGTOmega(embed_dim=64)
    _init_random_model(model)
    return model


def test_forward_returns_patch_tokens_opt_in():
    m = _tiny_model().eval()
    imgs = torch.rand(1, 2, 3, 64, 64)
    with torch.no_grad():
        out_default = m(imgs)
        out_tokens = m(imgs, return_last_patch_tokens=True)
    assert "patch_tokens" not in out_default
    assert out_tokens["patch_tokens"].shape == (1, 2, 16, 128)


def test_gradient_checkpointing_matches_baseline():
    m = _tiny_model().train()
    assert m.aggregator.gradient_checkpointing is False
    imgs = torch.rand(1, 2, 3, 64, 64)
    out_a = m(imgs)["pose_enc"]
    m.aggregator.gradient_checkpointing = True
    out_b = m(imgs)["pose_enc"]
    assert torch.allclose(out_a, out_b, atol=1e-6)
    out_b.sum().backward()
    g = [p.grad for p in m.aggregator.parameters() if p.grad is not None]
    assert len(g) > 0 and all(torch.isfinite(t).all() for t in g)
