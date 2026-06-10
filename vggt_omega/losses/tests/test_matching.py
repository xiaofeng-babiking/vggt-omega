"""Tests for matching-pair construction and the matching loss.

Pair construction is checked on scenes where the correct correspondences are
known by construction:

* two frames sharing an identical camera see every pixel at the same location,
  so positives must pair each patch with itself; the relative pose is pure
  identity (no translation), the epipolar geometry is degenerate, and no
  negative can be *proven* — so none may be returned;
* a stereo pair translated along x has horizontal epipolar lines, so the
  epipolar distance between patch centers is exactly their vertical offset
  ``|Δcy|`` — negatives must respect the configured minimum;
* depth-inconsistent frames (occlusion stand-in) must yield no positives.

The loss itself is checked on hand-built tokens with known cosine
similarities.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vggt_omega.losses import MatchingPairConfig, MatchingPairs, build_matching_pairs, matching_loss

H = W = 16
PATCH = 4
PATCH_W = W // PATCH


def _identity_extrinsic() -> torch.Tensor:
    return torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=-1)


def _intrinsics(f: float = 8.0) -> torch.Tensor:
    return torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])


def _row_colored_images(num_frames: int) -> torch.Tensor:
    """Images whose mean RGB differs per patch row (passes the RGB check)."""
    images = torch.zeros(1, num_frames, 3, H, W)
    for row in range(H // PATCH):
        images[:, :, 0, row * PATCH : (row + 1) * PATCH, :] = row / (H // PATCH)
        images[:, :, 1] = 0.5
    return images


def _two_frame_scene(translation_x: float = 0.0, depth_values=(1.0, 1.0)):
    extr0 = _identity_extrinsic()
    extr1 = _identity_extrinsic()
    extr1[0, 3] = translation_x
    extrinsics = torch.stack([extr0, extr1])[None]
    intrinsics = _intrinsics()[None, None].expand(1, 2, 3, 3).contiguous()
    depths = torch.stack(
        [torch.full((H, W), depth_values[0]), torch.full((H, W), depth_values[1])]
    )[None]
    valid = torch.ones_like(depths, dtype=torch.bool)
    return extrinsics, intrinsics, depths, valid


def test_identical_cameras_match_each_patch_to_itself():
    extrinsics, intrinsics, depths, valid = _two_frame_scene()
    images = _row_colored_images(2)
    gen = torch.Generator().manual_seed(0)

    pairs = build_matching_pairs(images, depths, extrinsics, intrinsics, valid, PATCH, generator=gen)

    assert pairs.positive.shape[0] > 0
    assert (pairs.positive[:, 2] == pairs.positive[:, 4]).all()  # query patch == target patch
    assert (pairs.positive[:, 1] != pairs.positive[:, 3]).all()  # across frames only
    # The 4px boundary margin leaves only the central 2x2 patch block.
    for patch in pairs.positive[:, 2].tolist():
        assert patch // PATCH_W in (1, 2) and patch % PATCH_W in (1, 2)
    # Pure-identity relative pose proves nothing: no negatives allowed.
    assert pairs.negative.shape[0] == 0


def test_stereo_negatives_respect_epipolar_and_rgb_thresholds():
    extrinsics, intrinsics, depths, valid = _two_frame_scene(translation_x=0.5)
    images = _row_colored_images(2)
    config = MatchingPairConfig(min_epipolar_distance=6.0, min_rgb_distance=0.05)
    gen = torch.Generator().manual_seed(0)

    pairs = build_matching_pairs(images, depths, extrinsics, intrinsics, valid, PATCH, config, generator=gen)

    assert pairs.negative.shape[0] > 0
    assert pairs.negative.shape[0] <= pairs.positive.shape[0]  # balanced
    # For x-translation the epipolar distance is the vertical offset of the
    # patch centers; recompute it independently of the implementation.
    cy_q = (pairs.negative[:, 2] // PATCH_W).float() * PATCH + PATCH / 2
    cy_t = (pairs.negative[:, 4] // PATCH_W).float() * PATCH + PATCH / 2
    assert ((cy_q - cy_t).abs() >= 6.0).all()
    # Row colors differ by row index / 4 in the red channel.
    rgb_gap = (cy_q - cy_t).abs() / PATCH / (H // PATCH)
    assert (rgb_gap >= 0.05).all()


def test_uniform_appearance_blocks_negatives():
    extrinsics, intrinsics, depths, valid = _two_frame_scene(translation_x=0.5)
    images = torch.full((1, 2, 3, H, W), 0.5)
    config = MatchingPairConfig(min_epipolar_distance=6.0, min_rgb_distance=0.05)
    gen = torch.Generator().manual_seed(0)

    pairs = build_matching_pairs(images, depths, extrinsics, intrinsics, valid, PATCH, config, generator=gen)
    assert pairs.negative.shape[0] == 0


def test_depth_inconsistency_blocks_positives():
    # Frame 1 reports depth 2 where reprojection predicts 1 (> 1% off).
    extrinsics, intrinsics, depths, valid = _two_frame_scene(depth_values=(1.0, 2.0))
    images = _row_colored_images(2)
    gen = torch.Generator().manual_seed(0)

    pairs = build_matching_pairs(images, depths, extrinsics, intrinsics, valid, PATCH, generator=gen)
    # Neither direction is consistent: 1 vs 2 and 2 vs 1 both fail the 1% tol.
    assert pairs.positive.shape[0] == 0


def test_single_frame_has_no_pairs():
    extrinsics, intrinsics, depths, valid = _two_frame_scene()
    pairs = build_matching_pairs(
        _row_colored_images(1),
        depths[:, :1],
        extrinsics[:, :1],
        intrinsics[:, :1],
        valid[:, :1],
        PATCH,
        generator=torch.Generator().manual_seed(0),
    )
    assert pairs.positive.shape[0] == 0
    assert pairs.negative.shape[0] == 0


def _pairs(positive_rows, negative_rows) -> MatchingPairs:
    def as_tensor(rows):
        return torch.tensor(rows, dtype=torch.long).reshape(-1, 5)

    return MatchingPairs(positive=as_tensor(positive_rows), negative=as_tensor(negative_rows))


def test_matching_loss_analytic_values():
    # Tokens: positives identical (cos sim 1), negatives orthogonal (cos sim 0).
    tokens = torch.zeros(1, 2, 2, 4)
    tokens[0, 0, 0, 0] = tokens[0, 1, 0, 0] = 1.0  # positive pair, s = 1
    tokens[0, 0, 1, 1] = 1.0  # negative pair, s = 0
    tokens[0, 1, 1, 2] = 1.0

    loss = matching_loss(tokens, _pairs([[0, 0, 0, 1, 0]], [[0, 0, 1, 1, 1]]))
    expected = math.log(1 + math.exp(-1.0)) + math.log(2.0)
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_matching_loss_decreases_with_alignment():
    tokens = torch.zeros(1, 2, 1, 2)
    tokens[0, 0, 0] = torch.tensor([1.0, 0.0])
    pairs = _pairs([[0, 0, 0, 1, 0]], [])

    tokens[0, 1, 0] = torch.tensor([0.0, 1.0])  # orthogonal
    far = matching_loss(tokens, pairs).item()
    tokens[0, 1, 0] = torch.tensor([1.0, 0.0])  # aligned
    near = matching_loss(tokens, pairs).item()
    assert near < far
    assert near == pytest.approx(-F.logsigmoid(torch.tensor(1.0)).item(), rel=1e-6)


def test_empty_pairs_give_zero_loss_with_gradient():
    tokens = torch.randn(1, 2, 4, 8, requires_grad=True)
    loss = matching_loss(tokens, _pairs([], []))
    assert loss.item() == pytest.approx(0.0)
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
