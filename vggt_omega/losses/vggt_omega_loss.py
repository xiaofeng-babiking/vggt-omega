# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Combined VGGT-Omega training loss (paper Eq. (1)).

``L = λ_cam·L_cam + λ_depth·L_depth + λ_point·L_point + λ_match·L_match``
with the paper's weights (Sec. A.1) as defaults. Ground truth is normalized to
the first camera's unit coordinate space before any loss is computed;
predictions are never normalized.
"""

import torch
import torch.nn as nn

from .camera_loss import camera_loss
from .depth_loss import depth_loss
from .matching import MatchingPairConfig, build_matching_pairs, matching_loss
from .normalization import normalize_scene_to_first_camera
from .point_loss import point_loss


class VGGTOmegaLoss(nn.Module):
    """Multi-task supervision for VGGT-Omega.

    Args:
        lambda_cam / lambda_depth / lambda_point / lambda_match: loss weights;
            defaults follow the paper (5.0, 1.0, 0.5, 0.1). Set a weight to 0
            to skip that loss entirely.
        alpha: weight of the −log c aleatoric-uncertainty term in the depth and
            point losses. Unspecified in the paper; 0.2 follows VGGT. Set 0 to
            disable (recommended when fine-tuning on very small datasets).
        patch_size: model patch size, for matching-pair construction.
        matching_config: thresholds/sampling sizes for pair construction.
    """

    def __init__(
        self,
        lambda_cam: float = 5.0,
        lambda_depth: float = 1.0,
        lambda_point: float = 0.5,
        lambda_match: float = 0.1,
        alpha: float = 0.2,
        patch_size: int = 16,
        matching_config: MatchingPairConfig | None = None,
    ) -> None:
        super().__init__()
        self.lambda_cam = lambda_cam
        self.lambda_depth = lambda_depth
        self.lambda_point = lambda_point
        self.lambda_match = lambda_match
        self.alpha = alpha
        self.patch_size = patch_size
        self.matching_config = matching_config or MatchingPairConfig()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all enabled losses.

        Args:
            predictions: model outputs with ``pose_enc`` (B, S, 9), ``depth``
                (B, S, H, W, 1), ``depth_conf`` (B, S, H, W), and — for the
                matching loss — ``patch_tokens`` (B, S, P, C).
            batch: GT with ``images`` (B, S, 3, H, W) in [0, 1], ``depths``
                (B, S, H, W), ``extrinsics`` (B, S, 3, 4) camera-from-world,
                ``intrinsics`` (B, S, 3, 3), optional ``point_masks``
                (B, S, H, W) bool.
            generator: optional RNG for deterministic matching-pair sampling.

        Returns:
            dict with ``loss`` (weighted total) and the unweighted
            ``loss_camera`` / ``loss_depth`` / ``loss_point`` / ``loss_match``.
        """
        images = batch["images"]
        height, width = images.shape[-2:]
        scene = normalize_scene_to_first_camera(
            batch["extrinsics"],
            batch["intrinsics"],
            batch["depths"],
            batch.get("point_masks"),
        )

        zero = torch.zeros((), device=images.device, dtype=torch.float32)
        losses = {"loss_camera": zero, "loss_depth": zero, "loss_point": zero, "loss_match": zero}

        if self.lambda_cam != 0:
            losses["loss_camera"] = camera_loss(
                predictions["pose_enc"], scene.extrinsics, batch["intrinsics"], (height, width)
            )
        if self.lambda_depth != 0:
            losses["loss_depth"] = depth_loss(
                predictions["depth"], predictions["depth_conf"], scene.depths, scene.valid_mask, alpha=self.alpha
            )
        if self.lambda_point != 0:
            losses["loss_point"] = point_loss(
                predictions["depth"],
                predictions["depth_conf"],
                predictions["pose_enc"],
                scene.points,
                scene.depths,
                scene.valid_mask,
                alpha=self.alpha,
            )
        if self.lambda_match != 0:
            if "patch_tokens" not in predictions:
                raise KeyError(
                    "lambda_match != 0 but predictions lack 'patch_tokens'. The model exposes "
                    "them in training mode only; set lambda_match=0 to skip the matching loss."
                )
            pairs = build_matching_pairs(
                images,
                batch["depths"],
                batch["extrinsics"],
                batch["intrinsics"],
                scene.valid_mask,
                self.patch_size,
                self.matching_config,
                generator=generator,
            )
            losses["loss_match"] = matching_loss(predictions["patch_tokens"], pairs)

        losses["loss"] = (
            self.lambda_cam * losses["loss_camera"]
            + self.lambda_depth * losses["loss_depth"]
            + self.lambda_point * losses["loss_point"]
            + self.lambda_match * losses["loss_match"]
        )
        return losses
