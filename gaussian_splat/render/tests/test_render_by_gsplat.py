"""GPU (musa) tests for the AnySplat splatting-decoder port.

gsplat rasterization has no CPU path, so everything here is skipped unless a
musa device and the gsplat package are available (both true in the
vggt-omega container).
"""

import pytest
import torch

try:  # torch_musa registers the musa device on import
    import torch_musa  # noqa: F401

    _HAS_MUSA = torch.musa.is_available()
except ImportError:
    _HAS_MUSA = False

pytest.importorskip("gsplat")
pytestmark = pytest.mark.skipif(not _HAS_MUSA, reason="gsplat rasterization needs a musa device")

from vggt_omega.models.heads import Gaussians  # noqa: E402
from vggt_omega.models.heads.gaussian_head import build_covariance  # noqa: E402

from gaussian_splat.render import render_by_gsplat  # noqa: E402

_H, _W = 64, 64
_SH0 = 0.28209479177387814  # DC spherical harmonic basis constant


def _single_gaussian_scene(num_views: int = 2, device: str = "musa"):
    """One opaque red gaussian 2m in front of the first camera, white bg."""
    device = torch.device(device)
    means = torch.tensor([[[0.0, 0.0, 2.0]]], device=device)
    scales = torch.full((1, 1, 3), 0.2, device=device)
    rotations = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device)  # identity, XYZW
    # sh_degree=0: color = 0.5 + SH0 * dc -> dc for pure red.
    dc = torch.tensor([1.0, 0.0, 0.0], device=device).sub(0.5).div(_SH0)
    gaussians = Gaussians(
        means=means,
        covariances=build_covariance(scales, rotations),
        harmonics=dc.view(1, 1, 3, 1),
        opacities=torch.ones(1, 1, device=device),
        scales=scales,
        rotations=rotations,
    )

    extrinsics = torch.eye(4, device=device)[:3].repeat(1, num_views, 1, 1)
    if num_views > 1:
        extrinsics[0, 1, 0, 3] = 0.05  # second camera shifted 5cm along +x
    intrinsics = torch.tensor(
        [[float(_W), 0.0, _W / 2], [0.0, float(_H), _H / 2], [0.0, 0.0, 1.0]], device=device
    ).repeat(1, num_views, 1, 1)
    return gaussians, extrinsics, intrinsics


def test_render_shapes_and_ranges():
    gaussians, extrinsics, intrinsics = _single_gaussian_scene()
    out = render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))

    assert out.color.shape == (1, 2, 3, _H, _W)
    assert out.depth.shape == (1, 2, _H, _W)
    assert out.alpha.shape == (1, 2, _H, _W)
    assert torch.isfinite(out.color).all()
    assert out.color.min() >= 0 and out.color.max() <= 1
    assert out.alpha.min() >= 0 and out.alpha.max() <= 1 + 1e-6
    assert out.nonfinite_pixels == 0


def test_healthy_render_reports_no_neutralised_pixels():
    """The counter must stay 0 on a good render, or it is useless as a signal."""
    gaussians, extrinsics, intrinsics = _single_gaussian_scene(num_views=1)
    assert render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W)).nonfinite_pixels == 0


def test_render_neutralises_a_non_finite_rasterizer_output(monkeypatch):
    """A nan from the rasterizer must not reach the caller.

    The rasterizer manufactures nan from entirely finite splats -- the tile
    accumulator sums an unbounded colour with no finite check, and a splat at a
    camera centre drives 0 * inf inside the SH kernel. Measured at 8-10% of training
    steps. Because the non-finite loss guard is COLLECTIVE, one such pixel on one
    rank discarded the step for every rank, so this was costing ~9% of a 7-GPU run.

    The clamp that used to be here did not catch it: clamp maps +-inf but propagates
    nan. Patch the rasterization call to return nan/inf directly rather than trying
    to provoke the real thing, which is sample-dependent and rare.
    """
    # importlib, not `import ... as mod`: the package re-exports the FUNCTION under
    # the module's own name, so the plain import binds the function and shadows it.
    import importlib

    mod = importlib.import_module("gaussian_splat.render.render_by_gsplat")

    gaussians, extrinsics, intrinsics = _single_gaussian_scene(num_views=2)
    real = mod.rasterization

    def poisoned(*args, **kwargs):
        rendering, alpha, meta = real(*args, **kwargs)
        rendering = rendering.clone()
        alpha = alpha.clone()
        rendering[0, 0, 0, :] = float("nan")  # colour+depth nan, view 0 pixel (0,0)
        rendering[0, 1, 1, 0] = float("inf")  # colour inf, view 0 pixel (1,1)
        rendering[1, 2, 2, 3] = float("nan")  # depth-only nan, view 1 pixel (2,2)
        return rendering, alpha, meta

    monkeypatch.setattr(mod, "rasterization", poisoned)
    out = mod.render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))

    assert torch.isfinite(out.color).all(), "nan escaped into color"
    assert torch.isfinite(out.depth).all(), "nan escaped into depth"
    assert torch.isfinite(out.alpha).all(), "nan escaped into alpha"
    assert out.color.min() >= 0 and out.color.max() <= 1

    # alpha is zeroed on every affected pixel -- colour-nan, colour-inf and
    # depth-nan alike -- so render_depth_loss excludes them instead of regressing a
    # neutralised depth of 0.0 against the teacher's real depth. Masking the
    # inf-colour pixel too costs one valid depth sample and buys one rule instead
    # of three; at ~1e-5 of the pixels in a frame that is the right trade.
    assert out.alpha[0, 0, 0, 0] == 0.0  # colour + depth nan
    assert out.alpha[0, 0, 1, 1] == 0.0  # colour inf
    assert out.alpha[0, 1, 2, 2] == 0.0  # depth nan only
    assert out.nonfinite_pixels == 3


def test_render_single_red_gaussian():
    gaussians, extrinsics, intrinsics = _single_gaussian_scene()
    out = render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))

    center = out.color[0, 0, :, _H // 2, _W // 2]
    corner = out.color[0, 0, :, 0, 0]
    assert center[0] > 0.8 and center[1] < 0.2 and center[2] < 0.2  # red splat
    assert torch.allclose(corner, torch.ones_like(corner), atol=1e-4)  # white background
    assert out.alpha[0, 0, _H // 2, _W // 2] > 0.9
    assert out.alpha[0, 0, 0, 0] < 1e-3
    # RGB+D depth at the splat center is the gaussian's z.
    assert abs(out.depth[0, 0, _H // 2, _W // 2].item() - 2.0) < 0.1


def test_render_accepts_3x4_and_4x4_extrinsics():
    gaussians, extrinsics_3x4, intrinsics = _single_gaussian_scene()
    bottom = torch.zeros_like(extrinsics_3x4[..., :1, :])
    bottom[..., 0, 3] = 1.0
    extrinsics_4x4 = torch.cat([extrinsics_3x4, bottom], dim=-2)

    out_3x4 = render_by_gsplat(gaussians, extrinsics_3x4, intrinsics, (_H, _W))
    out_4x4 = render_by_gsplat(gaussians, extrinsics_4x4, intrinsics, (_H, _W))
    assert torch.allclose(out_3x4.color, out_4x4.color, atol=1e-6)


def test_render_keeps_view_dim_for_single_view():
    gaussians, extrinsics, intrinsics = _single_gaussian_scene(num_views=1)
    out = render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))
    assert out.color.shape == (1, 1, 3, _H, _W)
    assert out.depth.shape == (1, 1, _H, _W)
    assert out.alpha.shape == (1, 1, _H, _W)


def _random_scene(num_gaussians=64, num_views=2, seed=0, device="musa"):
    """A scene with NON-identity rotations, so a wrong quaternion order shows."""
    device = torch.device(device)
    torch.manual_seed(seed)
    means = torch.randn(1, num_gaussians, 3, device=device) * 0.3
    means[..., 2] += 2.5  # in front of the cameras
    scales = torch.rand(1, num_gaussians, 3, device=device) * 0.08 + 0.02
    # Deliberately anisotropic + randomly rotated: an isotropic gaussian or an
    # identity quaternion would render the same under ANY quaternion convention.
    scales[..., 0] *= 4.0
    rotations = torch.nn.functional.normalize(
        torch.randn(1, num_gaussians, 4, device=device), dim=-1
    )
    gaussians = Gaussians(
        means=means,
        covariances=build_covariance(scales, rotations),
        harmonics=torch.rand(1, num_gaussians, 3, 1, device=device),
        opacities=torch.rand(1, num_gaussians, device=device) * 0.5 + 0.5,
        scales=scales,
        rotations=rotations,
    )
    extrinsics = torch.eye(4, device=device)[:3].repeat(1, num_views, 1, 1)
    if num_views > 1:
        extrinsics[0, 1, 0, 3] = 0.05
    intrinsics = torch.tensor(
        [[float(_W), 0.0, _W / 2], [0.0, float(_H), _H / 2], [0.0, 0.0, 1.0]], device=device
    ).repeat(1, num_views, 1, 1)
    return gaussians, extrinsics, intrinsics


def test_scales_quats_path_matches_explicit_covariance():
    """Passing scales+quats must render what covars= used to render.

    The renderer switched from handing gsplat a precomputed covariance to
    handing it scales and quaternions, so that gradients can reach those two.
    That is only safe if the quaternion is reordered XYZW -> WXYZ, and this is
    the test that would catch it if it were not: the scene is anisotropic and
    randomly rotated, so a wrong order changes every splat's footprint.
    """
    from gsplat import rasterization

    gaussians, extrinsics, intrinsics = _random_scene()
    ours = render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))

    bottom = torch.zeros_like(extrinsics[..., :1, :])
    bottom[..., 0, 3] = 1.0
    viewmats = torch.cat([extrinsics, bottom], dim=-2)[0]
    rendering, _, _ = rasterization(  # the OLD call: covariance, no quats/scales
        gaussians.means[0],
        gaussians.rotations[0],
        gaussians.scales[0],
        gaussians.opacities[0],
        gaussians.harmonics.permute(0, 1, 3, 2).contiguous()[0],
        viewmats,
        intrinsics[0],
        _W,
        _H,
        sh_degree=0,
        render_mode="RGB+D",
        packed=False,
        near_plane=1e-10,
        backgrounds=torch.ones(extrinsics.shape[1], 3, device=gaussians.means.device),
        radius_clip=0.1,
        covars=gaussians.covariances[0],
        rasterize_mode="classic",
    )
    reference = rendering[..., :3].clamp(0, 1).permute(0, 3, 1, 2)
    assert torch.allclose(ours.color[0], reference, atol=1e-5), (
        f"max |diff| = {(ours.color[0] - reference).abs().max().item():.3e}"
    )


def test_gradients_reach_scales_and_rotations():
    """The reason for the switch: covars= made these two unreachable.

    gsplat sets quats/scales to None as soon as covars is given, so under the
    old call a per-scene optimiser could step them forever with no effect.
    """
    gaussians, extrinsics, intrinsics = _random_scene()
    gaussians.scales = gaussians.scales.detach().requires_grad_(True)
    gaussians.rotations = gaussians.rotations.detach().requires_grad_(True)
    gaussians.means = gaussians.means.detach().requires_grad_(True)
    # Stale on purpose: if the renderer still read covariances, the loss would
    # depend on this constant and NOT on the tensors above.
    gaussians.covariances = gaussians.covariances.detach()

    out = render_by_gsplat(gaussians, extrinsics, intrinsics, (_H, _W))
    out.color.square().mean().backward()

    for name in ("scales", "rotations", "means"):
        grad = getattr(gaussians, name).grad
        assert grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient in {name}"
        assert grad.abs().max() > 0, f"gradient to {name} is identically zero"
