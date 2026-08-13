"""CPU unit tests for the fusion strategy primitives (plus a CUDA parity check)."""

import pytest
import torch

# No importorskip on torch_scatter: fuse_by_voxel falls back to torch when the
# compiled extension is absent, and that fallback is what runs in this repo's
# environment -- so skipping here would leave the only path in use uncovered.
from gaussian_splat.fuser import confidence_mask, fuse_by_voxel, suggest_voxel_size

_RAW_DIM = 20
_VOXEL = 0.1

_HAS_CUDA = torch.cuda.is_available()


def _scene(points: list[list[float]], confs: list[float], feat_values: list[float] | None = None):
    """One view of len(points) pixels: (V=1, H=1, W=N, ...) feats/points/conf."""
    num = len(points)
    pts = torch.tensor(points, dtype=torch.float32).view(1, 1, num, 3)
    conf = torch.tensor(confs, dtype=torch.float32).view(1, 1, num)
    if feat_values is None:
        feat_values = [float(i + 1) for i in range(num)]
    feats = torch.stack([torch.full((_RAW_DIM,), v) for v in feat_values]).view(1, 1, num, _RAW_DIM)
    return feats[0], pts[0], conf[0]


def test_same_voxel_equal_confidence_averages():
    # Two points 1cm apart share a 10cm voxel; equal confidence -> midpoint.
    feats, points, conf = _scene([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [2.0, 2.0], [1.0, 3.0])
    fused_points, fused_feats = fuse_by_voxel(feats, points, conf, _VOXEL)

    assert fused_points.shape == (1, 3)
    assert fused_feats.shape == (1, _RAW_DIM)
    assert torch.allclose(fused_points[0], torch.tensor([0.005, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(fused_feats[0], torch.full((_RAW_DIM,), 2.0), atol=1e-5)


def test_same_voxel_confidence_dominates():
    # A 20-logit confidence gap makes the softmax pick the confident candidate.
    feats, points, conf = _scene([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [0.0, 20.0], [1.0, 3.0])
    fused_points, fused_feats = fuse_by_voxel(feats, points, conf, _VOXEL)

    assert torch.allclose(fused_points[0], torch.tensor([0.01, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(fused_feats[0], torch.full((_RAW_DIM,), 3.0), atol=1e-5)


def test_distinct_voxels_pass_through():
    # Points in separate voxels keep their own values (weight 1 each).
    feats, points, conf = _scene(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 5.0, -3.0], [1.0, 2.0, 3.0]
    )
    fused_points, fused_feats = fuse_by_voxel(feats, points, conf, _VOXEL)

    assert fused_points.shape == (3, 3)
    # torch.unique sorts lexicographically, and these are already x-ascending.
    assert torch.allclose(fused_points, points.reshape(3, 3), atol=1e-6)
    assert torch.allclose(fused_feats[:, 0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


def test_larger_voxels_compress_more():
    torch.manual_seed(0)
    points = torch.rand(1, 8, 8, 3)
    feats = torch.randn(1, 8, 8, _RAW_DIM)
    conf = torch.randn(1, 8, 8)

    counts = [fuse_by_voxel(feats, points, conf, size)[0].shape[0] for size in (0.05, 0.2, 1.0)]
    assert counts[0] > counts[1] > counts[2]
    assert counts[0] <= 64  # never more anchors than input pixels
    assert counts[2] >= 1


def test_fusion_is_differentiable():
    feats, points, conf = _scene([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [0.5, 1.5], [1.0, 3.0])
    feats = feats.clone().requires_grad_(True)
    conf = conf.clone().requires_grad_(True)
    points = points.clone().requires_grad_(True)

    fused_points, fused_feats = fuse_by_voxel(feats, points, conf, _VOXEL)
    (fused_points.sum() + fused_feats.sum()).backward()

    # Gradient reaches the confidence head through the softmax weights, and
    # the features/points through the weighted sums.
    assert conf.grad is not None and torch.isfinite(conf.grad).all()
    assert conf.grad.abs().sum() > 0
    assert feats.grad is not None and feats.grad.abs().sum() > 0
    assert points.grad is not None and points.grad.abs().sum() > 0


def test_confidence_mask_quantile_and_strictness():
    conf = torch.full((2, 1, 4, 4), 1.0)
    conf[0].view(-1)[:12] = 3.0
    conf[1].view(-1)[:4] = 3.0

    mask, threshold = confidence_mask(conf, conf_threshold=0.5)
    assert torch.allclose(threshold, torch.tensor(2.0))
    assert torch.equal(mask, conf > 2.0)  # ties at the threshold are dropped
    assert mask.sum() == 16


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 0.999])
def test_oversized_quantile_matches_torch_quantile(monkeypatch, q):
    """Above 2**24 elements torch.quantile refuses ("input tensor is too
    large"), which 16 views at 832x1296 (17.25M pixels) already exceeds -- so
    the fallback must be the SAME number, not an approximation, or the
    confidence cut would silently depend on resolution. The size limit is
    monkeypatched rather than allocating 17M floats in a unit test."""
    from gaussian_splat.fuser import fuse_2d

    torch.manual_seed(0)
    conf = torch.rand(2, 3, 16, 16) * 5.0

    expected = torch.quantile(conf.flatten(), q)
    monkeypatch.setattr(fuse_2d, "_QUANTILE_MAX_ELEMENTS", 4)  # force the sort path
    actual = fuse_2d._global_quantile(conf, q)

    assert torch.allclose(actual, expected, atol=1e-6), f"{actual} != {expected}"


def test_confidence_mask_works_past_the_quantile_size_limit(monkeypatch):
    """The end-to-end guard: confidence_mask must not raise on a tensor bigger
    than torch.quantile accepts. This crashed a full quarter-resolution garden
    run after the forward had already completed."""
    from gaussian_splat.fuser import fuse_2d

    monkeypatch.setattr(fuse_2d, "_QUANTILE_MAX_ELEMENTS", 4)
    conf = torch.full((2, 1, 4, 4), 1.0)
    conf[0].view(-1)[:12] = 3.0
    conf[1].view(-1)[:4] = 3.0

    mask, threshold = confidence_mask(conf, conf_threshold=0.5)

    assert torch.allclose(threshold, torch.tensor(2.0))
    assert mask.sum() == 16


def _intrinsics(focal: float, size: int = 32) -> torch.Tensor:
    return torch.tensor(
        [[focal, 0.0, size / 2], [0.0, focal, size / 2], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)


def test_suggest_voxel_size_is_k_times_pixel_footprint():
    depth = torch.full((1, 1, 8, 8), 2.0)
    intrinsics = _intrinsics(200.0)

    # footprint = median_depth / focal = 2 / 200 = 0.01
    assert suggest_voxel_size(depth, intrinsics) == pytest.approx(0.005)  # k=0.5 default
    assert suggest_voxel_size(depth, intrinsics, k=1.0) == pytest.approx(0.01)
    assert suggest_voxel_size(depth, intrinsics, k=0.25) == pytest.approx(0.0025)


def test_suggest_voxel_size_scales_with_scene():
    intrinsics = _intrinsics(200.0)
    near = suggest_voxel_size(torch.full((1, 1, 8, 8), 1.0), intrinsics)
    far = suggest_voxel_size(torch.full((1, 1, 8, 8), 10.0), intrinsics)
    assert far == pytest.approx(10 * near)  # deeper scene -> proportionally coarser

    depth = torch.full((1, 1, 8, 8), 2.0)
    wide = suggest_voxel_size(depth, _intrinsics(100.0))
    tele = suggest_voxel_size(depth, _intrinsics(400.0))
    assert wide == pytest.approx(4 * tele)  # longer focal -> finer voxels


def test_suggest_voxel_size_is_robust_and_validates():
    # Median, not mean: a few wild depths must not move the estimate.
    depth = torch.full((1, 1, 4, 4), 2.0)
    depth[0, 0, 0, 0] = 1e4
    assert suggest_voxel_size(depth, _intrinsics(200.0)) == pytest.approx(0.005)

    # Non-positive depths are ignored; an all-invalid map is an error.
    mixed = torch.full((1, 1, 4, 4), 2.0)
    mixed[0, 0, 0, :] = -1.0
    assert suggest_voxel_size(mixed, _intrinsics(200.0)) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="no positive"):
        suggest_voxel_size(torch.zeros(1, 1, 4, 4), _intrinsics(200.0))
    with pytest.raises(ValueError, match="k must be positive"):
        suggest_voxel_size(depth, _intrinsics(200.0), k=0.0)

    # Trailing singleton dim (DenseHead's raw layout) is accepted.
    assert suggest_voxel_size(
        torch.full((1, 1, 4, 4, 1), 2.0), _intrinsics(200.0)
    ) == pytest.approx(0.005)


def test_suggested_size_barely_merges_within_one_view():
    """k=0.5 is sub-pixel by construction: a single view survives intact.

    Neighbouring pixels sit one full footprint apart, so a half-footprint
    voxel cannot merge them — real compression comes from cross-view overlap.
    """
    size, focal = 16, 200.0
    depth = torch.full((size, size), 2.0)
    intrinsics = _intrinsics(focal, size)
    fx = intrinsics[0, 0, 0, 0]
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    points = torch.stack(
        [(x - size / 2) / fx * depth, (y - size / 2) / fx * depth, depth], dim=-1
    ).unsqueeze(0)
    feats = torch.randn(1, size, size, _RAW_DIM)
    conf = torch.randn(1, size, size)

    voxel_size = suggest_voxel_size(depth.view(1, 1, size, size), intrinsics)
    kept = fuse_by_voxel(feats, points, conf, voxel_size)[0].shape[0]
    assert kept == size * size  # nothing merged

    # Well past the cliff, the same view collapses hard.
    coarse = suggest_voxel_size(depth.view(1, 1, size, size), intrinsics, k=8.0)
    assert fuse_by_voxel(feats, points, conf, coarse)[0].shape[0] < 0.1 * size * size


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device")
def test_cuda_matches_cpu():
    torch.manual_seed(0)
    points = torch.rand(1, 8, 8, 3)
    feats = torch.randn(1, 8, 8, _RAW_DIM)
    conf = torch.randn(1, 8, 8)

    cpu_points, cpu_feats = fuse_by_voxel(feats, points, conf, 0.2)
    cuda_points, cuda_feats = fuse_by_voxel(
        feats.cuda(), points.cuda(), conf.cuda(), 0.2
    )

    assert cuda_points.shape == cpu_points.shape
    assert torch.allclose(cuda_points.cpu(), cpu_points, atol=1e-5)
    assert torch.allclose(cuda_feats.cpu(), cpu_feats, atol=1e-5)


def test_fuse_by_voxel_survives_non_finite_points():
    """A non-finite coordinate must not size the scatter output.

    Regression for the crash that killed the 32-view run at step 10030:
    `(points / voxel_size).round().int()` on a non-finite value is undefined, and
    torch_scatter -- called without dim_size -- then sized its allocation from the
    resulting garbage index, asking for 4.96e18 elements. Both halves are fixed
    here (coordinates bounded, dim_size passed), so this asserts the whole path
    returns a correctly-shaped finite result instead of raising.
    """
    import torch

    from gaussian_splat.fuser.fuse_3d import fuse_by_voxel

    n, c = 64, 8
    torch.manual_seed(0)
    points = torch.randn(n, 3)
    feats = torch.randn(n, c)
    conf = torch.randn(n)
    # one of each pathology, which is what a bad render actually produces
    points[3] = float("nan")
    points[7] = float("inf")
    points[11] = float("-inf")
    points[15] = 1e30

    fused_points, fused_feats = fuse_by_voxel(feats, points, conf, voxel_size=0.05)

    assert torch.isfinite(fused_points).all(), "non-finite escaped into fused points"
    assert torch.isfinite(fused_feats).all(), "non-finite escaped into fused feats"
    assert fused_points.shape[0] == fused_feats.shape[0]
    assert 0 < fused_points.shape[0] <= n, fused_points.shape
    assert fused_feats.shape[1] == c
