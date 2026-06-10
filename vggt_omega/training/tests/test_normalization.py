import torch

from vggt_omega.training.losses import normalize_gt_into_first_camera, unproject_depth
from vggt_omega.training.tests.conftest import (
    _intrinsics_for_scene,
    _random_consistent_scene,
)


def test_scene_fixture_reprojects_onto_pixel_grid():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=3, seed=3)
    R, t = ext[..., :3], ext[..., 3]
    cam = torch.einsum("bsij,bshwj->bshwi", R, wp) + t[:, :, None, None]
    assert torch.allclose(cam[..., 2][mask], dep[mask], atol=1e-5)
    K = _intrinsics_for_scene(B=1, S=3)
    uvw = torch.einsum("bsij,bshwj->bshwi", K, cam)
    px = uvw[..., :2] / uvw[..., 2:].clamp(min=1e-8)
    H, W = dep.shape[-2:]
    vs, us = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([us, vs], dim=-1).expand(1, 3, H, W, 2)
    assert torch.allclose(px[mask], grid[mask], atol=1e-4)


def test_normalize_anchors_first_frame_to_identity():
    ext, dep, wp, mask = _random_consistent_scene(B=2, S=4)
    n_ext, n_dep, n_wp, scale = normalize_gt_into_first_camera(ext, dep, wp, mask)
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(n_ext[:, 0, :3, :3], eye, atol=1e-5)
    assert torch.allclose(n_ext[:, 0, :3, 3], torch.zeros(2, 3), atol=1e-5)


def test_normalize_unit_average_distance():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=3)
    n_ext, n_dep, n_wp, scale = normalize_gt_into_first_camera(ext, dep, wp, mask)
    d = n_wp[mask].norm(dim=-1).mean()
    assert torch.allclose(d, torch.tensor(1.0), atol=1e-4)


def test_normalize_scale_invariance():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=3)
    ext2 = ext.clone()
    ext2[..., 3] *= 7.0
    out_a = normalize_gt_into_first_camera(ext, dep, wp, mask)
    out_b = normalize_gt_into_first_camera(ext2, dep * 7, wp * 7, mask)
    for a, b in zip(out_a[:3], out_b[:3]):
        assert torch.allclose(a, b, atol=1e-4)


def test_unproject_inverts_projection():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=3)
    K = _intrinsics_for_scene(B=1, S=3)
    pts = unproject_depth(dep, ext, K)
    assert pts.shape == wp.shape
    assert torch.allclose(pts[mask], wp[mask], atol=1e-4)


def test_unproject_normalized_scene_matches_normalized_points():
    ext, dep, wp, mask = _random_consistent_scene(B=2, S=3, seed=1)
    n_ext, n_dep, n_wp, scale = normalize_gt_into_first_camera(ext, dep, wp, mask)
    K = _intrinsics_for_scene(B=2, S=3)
    pts = unproject_depth(n_dep, n_ext, K)
    assert torch.allclose(pts[mask], n_wp[mask], atol=1e-4)


def test_unproject_is_differentiable():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=2)
    dep = dep.clone().requires_grad_(True)
    pts = unproject_depth(dep, ext, _intrinsics_for_scene(B=1, S=2))
    pts[mask].sum().backward()
    assert dep.grad is not None and torch.isfinite(dep.grad).all()
