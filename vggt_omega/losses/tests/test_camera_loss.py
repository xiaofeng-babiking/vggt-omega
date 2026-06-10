"""Tests for the camera loss against analytically-known L1 values.

The loss is the batch/frame mean of the L1 norm over the 9D encoding, so a
perturbation of a single component by ``δ`` in every frame moves the loss by
exactly ``δ``.
"""

from __future__ import annotations

import pytest
import torch

from vggt_omega.losses import camera_loss
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

IMAGE_SIZE = (8, 8)


def _scene(num_frames: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=-1).repeat(1, num_frames, 1, 1)
    extrinsics[0, :, 2, 3] = torch.linspace(0.0, 1.0, num_frames)
    intrinsics = torch.eye(3).repeat(1, num_frames, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 4.0
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 4.0
    return extrinsics, intrinsics


def test_perfect_prediction_is_zero():
    extrinsics, intrinsics = _scene()
    pred = extri_intri_to_pose_encoding(extrinsics, intrinsics, IMAGE_SIZE)
    assert camera_loss(pred, extrinsics, intrinsics, IMAGE_SIZE).item() == pytest.approx(0.0)


@pytest.mark.parametrize("component", [0, 5, 8])  # tx, a quaternion entry, fov_w
def test_single_component_offset_moves_loss_by_delta(component):
    extrinsics, intrinsics = _scene()
    pred = extri_intri_to_pose_encoding(extrinsics, intrinsics, IMAGE_SIZE)
    pred[..., component] += 0.25
    assert camera_loss(pred, extrinsics, intrinsics, IMAGE_SIZE).item() == pytest.approx(0.25)


def test_loss_is_frame_count_invariant():
    # The same per-frame error must give the same loss for 2 and 8 frames.
    losses = []
    for num_frames in (2, 8):
        extrinsics, intrinsics = _scene(num_frames)
        pred = extri_intri_to_pose_encoding(extrinsics, intrinsics, IMAGE_SIZE)
        pred[..., 1] += 0.5
        losses.append(camera_loss(pred, extrinsics, intrinsics, IMAGE_SIZE).item())
    assert losses[0] == pytest.approx(losses[1])
