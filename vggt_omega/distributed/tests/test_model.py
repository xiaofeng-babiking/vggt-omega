import pytest
import torch
import torch.distributed as dist

from vggt_omega.distributed.attention import AllGatherKVAttention
from vggt_omega.distributed.model import ContextParallelVGGTOmega


def _init_random_model(model):
    """Properly initialize a from-scratch (checkpoint-less) VGGT-Omega for testing.

    Several parameters are created with ``torch.empty`` (LayerScale.gamma, the ViT
    cls/storage/mask tokens) and only populated by ``reset_parameters`` / the
    released checkpoint -- a bare ``VGGTOmega(...)`` leaves them as uninitialized
    CUDA memory, which is non-deterministic garbage (often NaN) and poisons the
    forward. ``LinearKMaskedBias.bias_mask`` is likewise NaN-by-design until the
    checkpoint sets it. Initialize all of them so the forward is finite and the
    CP-vs-base comparison is meaningful (a CP bug cannot hide behind a NaN==NaN).
    """
    for m in model.modules():
        reset = getattr(m, "reset_parameters", None)
        if callable(reset):
            reset()
        if getattr(m, "bias_mask", None) is not None:
            torch.nn.init.ones_(m.bias_mask)
    # Catch any parameter still left as raw torch.empty garbage (e.g. ViT tokens).
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                p.normal_(std=1e-3)


def test_cp_model_swaps_submodules_and_loads_base_state_dict():
    from vggt_omega.models import VGGTOmega
    from vggt_omega.distributed.aggregator import ContextParallelAggregator
    from vggt_omega.distributed.camera_head import ContextParallelCameraHead

    # embed_dim=64 keeps it light for CI; must be a multiple of 64 (patch_embed
    # hardcodes num_heads=16, RoPE needs head_dim%4==0). embed_dim=32 would crash.
    base = VGGTOmega(embed_dim=64, enable_depth=True, enable_camera=True)
    cp = ContextParallelVGGTOmega(cp_group=None, strategy=AllGatherKVAttention(), embed_dim=64)
    assert isinstance(cp.aggregator, ContextParallelAggregator)
    assert isinstance(cp.camera_head, ContextParallelCameraHead)
    # Identical parameter tree -> released checkpoint loads cleanly.
    missing, unexpected = cp.load_state_dict(base.state_dict(), strict=False)
    assert not missing and not unexpected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full-model forward needs CUDA autocast + NCCL")
def test_cp_model_g1_matches_base_forward():
    # world_size=1 CP group => every all_gather is identity => exact match.
    # Uses the production backend (nccl) on GPU; the gloo/CPU path is disabled.
    from vggt_omega.models import VGGTOmega

    dist.init_process_group(backend="nccl", rank=0, world_size=1, init_method="tcp://127.0.0.1:29512")
    try:
        torch.manual_seed(0)
        base = VGGTOmega(embed_dim=64).cuda().eval()
        _init_random_model(base)
        cp = ContextParallelVGGTOmega(
            cp_group=dist.group.WORLD, strategy=AllGatherKVAttention(), embed_dim=64
        ).cuda().eval()
        # Every initialized param/buffer is in state_dict, so cp becomes identical.
        cp.load_state_dict(base.state_dict())
        images = torch.rand(1, 4, 3, 32, 32).cuda()
        with torch.inference_mode():
            ref = base(images)
            got = cp(images)
        # Guard against a vacuous pass: equal_nan is left at its default (False), so
        # NaN != NaN would fail; additionally require the reference to be finite.
        assert torch.isfinite(ref["depth"]).all(), "test setup produced non-finite ref depth"
        assert torch.isfinite(ref["pose_enc"]).all(), "test setup produced non-finite ref pose_enc"
        torch.testing.assert_close(got["depth"], ref["depth"], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(got["pose_enc"], ref["pose_enc"], atol=1e-2, rtol=1e-2)
    finally:
        dist.destroy_process_group()
