"""Tests for the combined VGGT-Omega loss (paper Eq. (1)).

Predictions equal to the *normalized* ground truth must drive every geometric
loss to (numerically) zero, the total must be the λ-weighted sum of the parts,
and gradients must flow back to every prediction tensor.
"""

from __future__ import annotations

import pytest
import torch

from vggt_omega.losses import VGGTOmegaLoss, normalize_scene_to_first_camera
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

B, S, H, W, PATCH = 1, 2, 16, 16, 4
NUM_PATCHES = (H // PATCH) * (W // PATCH)


def _batch() -> dict[str, torch.Tensor]:
    extr0 = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=-1)
    extr1 = extr0.clone()
    extr1[2, 3] = 1.0  # one metre forward
    intrinsics = torch.tensor([[8.0, 0.0, W / 2], [0.0, 8.0, H / 2], [0.0, 0.0, 1.0]])
    depths = torch.stack([torch.full((H, W), 2.0), torch.full((H, W), 3.0)])

    images = torch.zeros(B, S, 3, H, W)
    for row in range(H // PATCH):
        images[:, :, 0, row * PATCH : (row + 1) * PATCH, :] = row / (H // PATCH)

    return {
        "images": images,
        "depths": depths[None],
        "extrinsics": torch.stack([extr0, extr1])[None],
        "intrinsics": intrinsics[None, None].expand(B, S, 3, 3).contiguous(),
    }


def _perfect_predictions(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scene = normalize_scene_to_first_camera(batch["extrinsics"], batch["intrinsics"], batch["depths"])
    pose_enc = extri_intri_to_pose_encoding(scene.extrinsics, batch["intrinsics"], (H, W))
    return {
        "pose_enc": pose_enc,
        "depth": scene.depths[..., None],
        "depth_conf": torch.ones(B, S, H, W),
        "patch_tokens": torch.randn(B, S, NUM_PATCHES, 8),
    }


def test_perfect_prediction_geometric_losses_are_zero():
    batch = _batch()
    predictions = _perfect_predictions(batch)
    loss_fn = VGGTOmegaLoss(lambda_match=0.0, patch_size=PATCH)

    losses = loss_fn(predictions, batch)

    assert losses["loss_camera"].item() == pytest.approx(0.0, abs=1e-5)
    assert losses["loss_depth"].item() == pytest.approx(0.0, abs=1e-5)
    assert losses["loss_point"].item() == pytest.approx(0.0, abs=1e-4)
    assert losses["loss"].item() == pytest.approx(0.0, abs=1e-4)


def test_total_is_weighted_sum_of_parts():
    batch = _batch()
    predictions = _perfect_predictions(batch)
    predictions["pose_enc"] = predictions["pose_enc"] + 0.1
    predictions["depth"] = predictions["depth"] * 1.2
    loss_fn = VGGTOmegaLoss(patch_size=PATCH)

    losses = loss_fn(predictions, batch, generator=torch.Generator().manual_seed(0))

    expected = (
        5.0 * losses["loss_camera"] + 1.0 * losses["loss_depth"] + 0.5 * losses["loss_point"] + 0.1 * losses["loss_match"]
    )
    assert losses["loss"].item() == pytest.approx(expected.item(), rel=1e-6)
    assert losses["loss_camera"].item() > 0
    assert losses["loss_depth"].item() > 0
    assert losses["loss_point"].item() > 0
    assert losses["loss_match"].item() > 0


def test_zero_weight_skips_loss():
    batch = _batch()
    predictions = _perfect_predictions(batch)
    loss_fn = VGGTOmegaLoss(lambda_cam=0.0, lambda_point=0.0, lambda_match=0.0, patch_size=PATCH)

    losses = loss_fn(predictions, batch)
    assert losses["loss_camera"].item() == 0.0
    assert losses["loss_point"].item() == 0.0
    assert losses["loss_match"].item() == 0.0


def test_missing_patch_tokens_raises_unless_disabled():
    # Silently dropping a paper loss (e.g. model left in eval mode) would give
    # wrong totals with no signal; the loss must demand an explicit opt-out.
    batch = _batch()
    predictions = _perfect_predictions(batch)
    del predictions["patch_tokens"]

    with pytest.raises(KeyError, match="patch_tokens"):
        VGGTOmegaLoss(patch_size=PATCH)(predictions, batch)

    losses = VGGTOmegaLoss(patch_size=PATCH, lambda_match=0.0)(predictions, batch)
    assert losses["loss_match"].item() == 0.0
    assert torch.isfinite(losses["loss"])


def test_gradients_flow_to_all_predictions():
    batch = _batch()
    predictions = _perfect_predictions(batch)
    predictions = {k: v.clone().requires_grad_(True) for k, v in predictions.items()}
    # Perturb so no loss sits exactly at a stationary point.
    perturbed = {
        "pose_enc": predictions["pose_enc"] + 0.05,
        "depth": predictions["depth"] * 1.1,
        "depth_conf": predictions["depth_conf"] + 0.5,
        "patch_tokens": predictions["patch_tokens"],
    }

    losses = VGGTOmegaLoss(patch_size=PATCH)(perturbed, batch, generator=torch.Generator().manual_seed(0))
    losses["loss"].backward()

    for name, tensor in predictions.items():
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_non_finite_gt_depth_keeps_loss_and_gradients_finite():
    # A single inf/NaN GT pixel must stay confined to the validity mask: the
    # loss and the gradients reaching the predictions must remain finite.
    batch = _batch()
    batch["depths"][0, 0, 3, 3] = float("inf")
    batch["depths"][0, 1, 5, 5] = float("nan")
    predictions = _perfect_predictions(batch)
    predictions = {k: v.clone().requires_grad_(True) for k, v in predictions.items()}

    losses = VGGTOmegaLoss(patch_size=PATCH)(predictions, batch, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for name, tensor in predictions.items():
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name


def test_all_invalid_gt_yields_finite_zero_geometry_losses():
    batch = _batch()
    batch["depths"] = torch.zeros_like(batch["depths"])
    predictions = _perfect_predictions(batch)

    losses = VGGTOmegaLoss(patch_size=PATCH)(predictions, batch)
    assert torch.isfinite(losses["loss"])
    assert losses["loss_depth"].item() == pytest.approx(0.0)
    assert losses["loss_point"].item() == pytest.approx(0.0)
