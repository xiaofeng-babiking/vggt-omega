"""Tests for GT unit-space normalization on hand-crafted scenes.

Scenes are built so the expected scale is exact:

* a single pixel at the principal point with depth ``d`` unprojects to
  ``(0, 0, d)``, so the average point distance — the scale — is exactly ``d``;
* with camera 0 at the identity, the world frame *is* the first-camera frame,
  so normalized extrinsics/translations follow by dividing by ``d``;
* invalid pixels (depth 0) must not move the scale.
"""

from __future__ import annotations

import pytest
import torch

from vggt_omega.losses import normalize_scene_to_first_camera


def _intrinsics(f: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def _identity_extrinsic() -> torch.Tensor:
    return torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=-1)


def test_single_pixel_scene_scale_is_depth():
    depth = torch.full((1, 1, 1, 1), 2.0)
    extrinsics = _identity_extrinsic()[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    assert scene.scale.item() == pytest.approx(2.0)
    assert scene.depths.item() == pytest.approx(1.0)
    assert torch.allclose(scene.points[0, 0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert scene.valid_mask.all()


def test_first_camera_becomes_identity_and_translations_scale():
    # Camera 1 is the world shifted by +1 in z; its pixel sees the same world
    # point (0, 0, 2) at depth 3. Points: both (0, 0, 2) in cam0 -> scale 2.
    depth = torch.tensor([2.0, 3.0]).view(1, 2, 1, 1)
    extr0 = _identity_extrinsic()
    extr1 = torch.cat([torch.eye(3), torch.tensor([[0.0], [0.0], [1.0]])], dim=-1)
    extrinsics = torch.stack([extr0, extr1])[None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None].expand(1, 2, 3, 3)

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    assert scene.scale.item() == pytest.approx(2.0)
    assert torch.allclose(scene.extrinsics[0, 0], _identity_extrinsic(), atol=1e-6)
    assert torch.allclose(scene.extrinsics[0, 1, :, 3], torch.tensor([0.0, 0.0, 0.5]), atol=1e-6)
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(scene.points[0, 0, 0, 0], expected, atol=1e-6)
    assert torch.allclose(scene.points[0, 1, 0, 0], expected, atol=1e-6)


def test_nonidentity_first_camera_is_normalized_away():
    # A world rotation/translation applied to BOTH cameras must cancel: the
    # normalized scene only depends on relative geometry.
    rot = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = torch.tensor([[0.5], [-0.25], [1.0]])
    extr0 = torch.cat([rot, t], dim=-1)
    extrinsics = extr0[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]
    depth = torch.full((1, 1, 1, 1), 2.0)

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    assert torch.allclose(scene.extrinsics[0, 0], _identity_extrinsic(), atol=1e-6)
    assert scene.scale.item() == pytest.approx(2.0)


def test_invalid_pixels_do_not_affect_scale():
    depth = torch.tensor([[2.0, 0.0]]).view(1, 1, 1, 2)
    extrinsics = _identity_extrinsic()[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    assert scene.scale.item() == pytest.approx(2.0)
    assert scene.valid_mask[0, 0, 0, 0].item() is True
    assert scene.valid_mask[0, 0, 0, 1].item() is False


def test_sky_and_point_mask_excluded():
    depth = torch.tensor([[2.0, -1.0]]).view(1, 1, 1, 2)
    masks = torch.tensor([[True, True]]).view(1, 1, 1, 2)
    extrinsics = _identity_extrinsic()[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth, point_masks=masks)
    assert scene.scale.item() == pytest.approx(2.0)

    masks[..., 0] = False
    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth, point_masks=masks)
    assert not scene.valid_mask.any()


def test_non_finite_depths_do_not_poison_scale_or_outputs():
    # One inf and one NaN pixel: both must be masked out AND must not leak
    # NaN into the scale (inf * 0 = NaN) or the returned depths/points.
    depth = torch.tensor([[2.0, float("inf"), float("nan"), 2.0]]).view(1, 1, 1, 4)
    extrinsics = _identity_extrinsic()[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    clean = normalize_scene_to_first_camera(
        extrinsics, intrinsics, torch.tensor([[2.0, 0.0, 0.0, 2.0]]).view(1, 1, 1, 4)
    )
    assert torch.isfinite(scene.scale).all()
    assert torch.isfinite(scene.depths).all()
    assert torch.isfinite(scene.points).all()
    assert scene.scale.item() == pytest.approx(clean.scale.item())
    assert scene.valid_mask.tolist() == [[[[True, False, False, True]]]]


def test_empty_scene_keeps_scale_one_and_finite():
    depth = torch.zeros(1, 1, 2, 2)
    extrinsics = _identity_extrinsic()[None, None]
    intrinsics = _intrinsics(1.0, 0.0, 0.0)[None, None]

    scene = normalize_scene_to_first_camera(extrinsics, intrinsics, depth)

    assert scene.scale.item() == pytest.approx(1.0)
    assert torch.isfinite(scene.points).all()
    assert not scene.valid_mask.any()
