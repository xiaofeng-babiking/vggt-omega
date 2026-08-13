"""CPU unit tests for GSDecoder's fusion pipeline.

The fusion strategy primitives themselves are covered in
``gaussian_splat/fuser/tests/test_fuser.py``; here we test the decoder's
integrated pipeline (2D selection -> unproject to reference frame -> 3D
aggregation -> decode) and the strategy plug-in surface.
"""

import pytest
import torch

# No importorskip on torch_scatter: the plain-torch scatter fallback is this
# repo's production path, and this integration suite must cover it.

from gaussian_splat.fuser import (  # noqa: E402
    ConfidenceFuse2D,
    Fuse2D,
    NoFuse2D,
    NoFuse3D,
    VoxelFuse3D,
)

from vggt_omega.models.heads.gaussian_head import GSDecoder  # noqa: E402
from vggt_omega.utils.geometry import invert_transform_points, unproject_depth  # noqa: E402

_SH_DEGREE = 1  # raw_gs_dim = 1 + 7 + 3 * 4 = 20
_VOXEL = 0.1

_B, _S, _H, _W = 2, 1, 4, 4
_N = _S * _H * _W


def _pipeline_inputs(seed: int = 0, **decoder_overrides):
    torch.manual_seed(seed)
    kwargs = dict(sh_degree=_SH_DEGREE, fuse_2d=NoFuse2D(), fuse_3d=VoxelFuse3D(voxel_size=_VOXEL))
    kwargs.update(decoder_overrides)
    decoder = GSDecoder(**kwargs)
    gs_map = torch.randn(_B, _S, _H, _W, decoder.raw_gs_dim)
    gs_conf = torch.randn(_B, _S, _H, _W)
    depth = torch.rand(_B, _S, _H, _W) + 0.5
    depth_conf = torch.rand(_B, _S, _H, _W) + 1.0
    extrinsics = torch.eye(4)[:3].expand(_B, _S, 3, 4).contiguous()
    intrinsics = torch.tensor(
        [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
    ).expand(_B, _S, 3, 3).contiguous()
    return decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics


def test_forward_no_merge_matches_direct_decode():
    """Tiny voxels + no filter: fused forward == decode() on unprojected pixels."""
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_3d=VoxelFuse3D(voxel_size=1e-6)
    )
    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)

    assert out.valid.all()
    assert out.pixel_mask.all()
    assert out.conf_threshold_value is None
    assert out.gaussians.means.shape == (_B, _N, 3)

    points = invert_transform_points(extrinsics, unproject_depth(depth, intrinsics))
    direct = decoder.decode(gs_map.reshape(_B, _N, -1), points.reshape(_B, _N, 3))

    # Voxel order is unique-sorted, not pixel order: compare per-batch as sets
    # by matching each fused mean to its source pixel.
    for b in range(_B):
        distances = torch.cdist(out.gaussians.means[b], direct.means[b])
        match = distances.argmin(dim=1)
        assert distances.min(dim=1).values.max() < 1e-5
        assert match.sort().values.tolist() == list(range(_N))  # a permutation
        assert torch.allclose(out.gaussians.opacities[b], direct.opacities[b][match], atol=1e-5)
        assert torch.allclose(out.gaussians.harmonics[b], direct.harmonics[b][match], atol=1e-5)
        assert torch.allclose(out.gaussians.scales[b], direct.scales[b][match], atol=1e-5)


def test_forward_means_land_in_reference_frame():
    """A camera translated by t: fused means come out shifted by -R^T t."""
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_3d=VoxelFuse3D(voxel_size=1e-6)
    )
    translated = extrinsics.clone()
    translated[..., :, 3] = torch.tensor([0.5, -0.25, 1.0])

    base = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)
    moved = decoder(gs_map, gs_conf, depth, depth_conf, translated, intrinsics)

    shift = torch.tensor([-0.5, 0.25, -1.0])  # R = I -> reference point = cam point - t
    base_sorted = base.gaussians.means.reshape(-1, 3)
    moved_sorted = moved.gaussians.means.reshape(-1, 3) - shift
    for tensor in (base_sorted, moved_sorted):
        assert torch.isfinite(tensor).all()
    base_sorted = base_sorted[base_sorted[:, 0].argsort()]
    moved_sorted = moved_sorted[moved_sorted[:, 0].argsort()]
    assert torch.allclose(base_sorted, moved_sorted, atol=1e-4)


def test_forward_confidence_filter_drops_pixels():
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_2d=ConfidenceFuse2D(conf_threshold=0.5), fuse_3d=VoxelFuse3D(voxel_size=1e-6)
    )
    depth_conf = torch.full((_B, _S, _H, _W), 1.0)
    depth_conf[0].view(-1)[:12] = 3.0
    depth_conf[1].view(-1)[:4] = 3.0

    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)

    assert torch.equal(out.pixel_mask, depth_conf > 2.0)
    assert out.conf_threshold_value is not None
    assert out.valid.sum(dim=1).tolist() == [12, 4]  # ragged, right-padded
    assert out.gaussians.means.shape == (_B, 12, 3)
    # Padded slots are invisible and parked far away.
    assert (out.gaussians.opacities[1, 4:] == 0).all()
    assert (out.gaussians.means[1, 4:] == -1e4).all()


def test_forward_rejects_all_dropped_scene():
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_2d=ConfidenceFuse2D(conf_threshold=0.5)
    )
    with pytest.raises(ValueError, match="too aggressive"):
        # Uniform confidence: strict > quantile is False everywhere.
        decoder(gs_map, gs_conf, depth, torch.ones(_B, _S, _H, _W), extrinsics, intrinsics)


def test_forward_auto_voxel_size_per_scene():
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_3d=VoxelFuse3D()  # voxel_size=None -> suggest_voxel_size per scene
    )
    depth = torch.full((_B, _S, _H, _W), 2.0)
    depth[1] = 4.0

    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)
    assert out.voxel_sizes[0] == pytest.approx(0.5 * 2.0 / 4.0)  # k * median depth / focal
    assert out.voxel_sizes[1] == pytest.approx(2 * out.voxel_sizes[0])


def test_forward_gradients_reach_head_outputs():
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_3d=VoxelFuse3D(voxel_size=0.5)
    )
    gs_map = gs_map.requires_grad_(True)
    gs_conf = gs_conf.requires_grad_(True)

    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)
    (out.gaussians.opacities.sum() + out.gaussians.means.sum()).backward()

    assert gs_map.grad is not None and gs_map.grad.abs().sum() > 0
    assert gs_conf.grad is not None and gs_conf.grad.abs().sum() > 0


def test_forward_rejects_channel_mismatch():
    decoder, _, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs()
    bad_map = torch.randn(_B, _S, _H, _W, decoder.raw_gs_dim + 1)
    with pytest.raises(ValueError, match="channels"):
        decoder(bad_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)


def test_no_fuse_3d_keeps_per_pixel_set_in_pixel_order():
    """NoFuse2D + NoFuse3D: forward degenerates to the unfused per-pixel decode."""
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_3d=NoFuse3D()
    )
    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)

    assert out.gaussians.means.shape == (_B, _N, 3)
    assert out.valid.all()
    assert out.voxel_sizes == [None, None]

    points = invert_transform_points(extrinsics, unproject_depth(depth, intrinsics))
    direct = decoder.decode(gs_map.reshape(_B, _N, -1), points.reshape(_B, _N, 3))
    # NoFuse3D preserves input (pixel) order, so fields compare elementwise.
    assert torch.allclose(out.gaussians.means, direct.means, atol=1e-6)
    assert torch.allclose(out.gaussians.opacities, direct.opacities, atol=1e-6)
    assert torch.allclose(out.gaussians.harmonics, direct.harmonics, atol=1e-6)


def test_custom_fuse_2d_strategy_plugs_in():
    class KeepFirstHalf(Fuse2D):
        def forward(self, depth_conf):
            mask = torch.zeros_like(depth_conf, dtype=torch.bool)
            mask.reshape(depth_conf.shape[0], -1)[:, : depth_conf[0].numel() // 2] = True
            return mask, None

    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_2d=KeepFirstHalf(), fuse_3d=NoFuse3D()
    )
    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)

    assert out.conf_threshold_value is None
    assert out.valid.sum(dim=1).tolist() == [_N // 2, _N // 2]
    assert out.pixel_mask.reshape(_B, -1)[:, : _N // 2].all()
    assert not out.pixel_mask.reshape(_B, -1)[:, _N // 2 :].any()


def test_fusion_strategies_are_registered_submodules():
    decoder = GSDecoder(sh_degree=_SH_DEGREE)
    assert isinstance(decoder.fuse_2d, ConfidenceFuse2D)  # AnySplat defaults
    assert isinstance(decoder.fuse_3d, VoxelFuse3D)
    names = dict(decoder.named_modules())
    assert "fuse_2d" in names and "fuse_3d" in names
    # Default strategies are parameter-free: checkpoints stay unaffected.
    assert len(decoder.state_dict()) == 0


def test_forward_survives_non_finite_confidence():
    """A nan in depth_conf must NOT kill the process.

    The quantile of a tensor containing nan is nan, and `conf > nan` is False for
    every pixel, so every scene looks empty however mild the quantile is. That is
    a transient numerical event, not a bad configuration: the trainer's
    cross-rank guard skips the step on the resulting non-finite loss, but only if
    the forward gets far enough to produce one. Raising here instead took down a
    healthy 16-rank job (step ~1600, 2 nodes).
    """
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_2d=ConfidenceFuse2D(conf_threshold=0.1)
    )
    depth_conf = depth_conf.clone()
    depth_conf[0, 0, 0, 0] = float("nan")  # one bad pixel poisons the whole quantile

    out = decoder(gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics)

    # The forward completed and produced the right shapes -- that is the whole
    # point; the values are expected to be non-finite and are the loss guard's job.
    assert out.gaussians.means.shape[0] == _B
    assert out.valid.shape[0] == _B
    assert bool(out.valid.any()), "every scene was dropped; the loss guard gets nothing to skip on"


def test_forward_survives_all_confidence_non_finite():
    """Even with NOTHING finite, keep the shapes valid rather than raising."""
    decoder, gs_map, gs_conf, depth, depth_conf, extrinsics, intrinsics = _pipeline_inputs(
        fuse_2d=ConfidenceFuse2D(conf_threshold=0.1)
    )
    out = decoder(
        gs_map, gs_conf, depth, torch.full_like(depth_conf, float("nan")),
        extrinsics, intrinsics,
    )
    assert out.gaussians.means.shape[0] == _B
    assert bool(out.valid.any())


def test_suggest_voxel_size_survives_non_finite_depth():
    """Non-finite depth must not raise: `nan > 0` is False everywhere, which looks
    identical to a degenerate scene but is a transient training event the
    trainer's cross-rank guard already handles -- if the forward survives to
    produce a loss."""
    from gaussian_splat.fuser.fuse_3d import suggest_voxel_size

    depth = torch.full((1, 1, 4, 4), float("nan"))
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3) * 100.0
    intrinsics[..., 2, 2] = 1.0
    out = suggest_voxel_size(depth, intrinsics)
    assert out > 0 and torch.isfinite(torch.tensor(out)), "placeholder scale must be usable"


def test_suggest_voxel_size_still_rejects_degenerate_depth():
    """A FINITE but all-non-positive depth is a real degenerate scene and must
    still raise -- skipping it would silently train on nothing."""
    from gaussian_splat.fuser.fuse_3d import suggest_voxel_size

    depth = torch.zeros(1, 1, 4, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3) * 100.0
    intrinsics[..., 2, 2] = 1.0
    with pytest.raises(ValueError, match="no positive values"):
        suggest_voxel_size(depth, intrinsics)
