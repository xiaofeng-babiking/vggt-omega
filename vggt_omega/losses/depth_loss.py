# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Depth loss (VGGT-Omega paper, Sec. 3.2).

Aleatoric-uncertainty-weighted L1 with a gradient-consistency term and the
relative-scale factor ``1 + 1/D``; residual ``e = D̂ − D`` against GT depth
normalized to unit space.
"""

import torch

from .regression import confidence_weighted_loss


def depth_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Confidence-weighted depth loss.

    Args:
        pred_depth: (B, S, H, W, 1) or (B, S, H, W) predicted depth.
        pred_conf: (B, S, H, W) predicted confidence, ≥ 1.
        gt_depth: (B, S, H, W) GT depth normalized to unit space.
        valid: (B, S, H, W) bool mask of pixels with usable GT.
        alpha: weight of the −log c uncertainty term (see ``regression``).
    """
    if pred_depth.dim() == gt_depth.dim():
        pred_depth = pred_depth[..., None]
    residual = pred_depth.float() - gt_depth.float()[..., None]
    return confidence_weighted_loss(residual, pred_conf, gt_depth, valid, alpha=alpha)
