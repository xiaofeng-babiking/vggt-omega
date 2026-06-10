# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Camera loss (VGGT-Omega paper, Sec. 3.2).

``L_cam = Σ_i |ĝ_i − g_i|``: an L1 objective between the predicted and
ground-truth 9D pose encodings (translation, quaternion, FoV). The paper sums
over frames; we average over batch and frames so the magnitude is invariant to
the per-batch frame count (folds into λ_cam).
"""

import torch

from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


def camera_loss(
    pred_pose_enc: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    """L1 between predicted and GT pose encodings.

    Args:
        pred_pose_enc: (B, S, 9) predicted encoding [t, quat_xyzw, fov_h, fov_w].
        gt_extrinsics: (B, S, 3, 4) GT camera-from-first-camera [R|t], already
            normalized to unit space (see ``normalize_scene_to_first_camera``).
        gt_intrinsics: (B, S, 3, 3) GT pinhole matrices.
        image_size_hw: (H, W) of the input images, for the FoV terms.
    """
    gt_pose_enc = extri_intri_to_pose_encoding(gt_extrinsics, gt_intrinsics, image_size_hw)
    return (pred_pose_enc.float() - gt_pose_enc).abs().sum(dim=-1).mean()
