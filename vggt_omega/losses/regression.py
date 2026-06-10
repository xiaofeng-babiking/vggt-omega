# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared confidence-weighted regression core for the depth and point losses.

Both losses in the VGGT-Omega paper (Sec. 3.2) have the form

    Σ_i ‖c_i ⊙ (1 + D_i⁻¹) ⊙ e_i‖ + ‖c_i ⊙ ∇e_i‖ − α Σ_i log c_i

with ``c`` the predicted aleatoric confidence (≥ 1), ``D`` the normalized GT
depth (so ``1 + 1/D`` up-weights near geometry, accounting for relative
scale), ``e`` the residual (scalar for depth, 3D for points), and ``∇`` the
spatial finite-difference gradient. They differ only in the residual.

Sums are implemented as masked means so magnitudes are invariant to image
resolution and frame count (folds into the λ weights).
"""

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over ``mask``; 0 when the mask is empty."""
    mask = mask.float()
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def spatial_gradient_magnitude(
    residual: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel magnitudes of the x/y finite differences of a residual field.

    Args:
        residual: (B, S, H, W, C) signed residuals.
        valid: (B, S, H, W) bool; a difference is valid when both pixels are.

    Returns:
        (grad_x, valid_x, grad_y, valid_y): magnitudes (..., H, W-1) / (..., H-1, W)
        as the per-pixel L2 norm over channels, with their validity masks.
    """
    grad_x = (residual[..., :, 1:, :] - residual[..., :, :-1, :]).norm(dim=-1)
    grad_y = (residual[..., 1:, :, :] - residual[..., :-1, :, :]).norm(dim=-1)
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    return grad_x, valid_x, grad_y, valid_y


def confidence_weighted_loss(
    residual: torch.Tensor,
    confidence: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.2,
    depth_eps: float = 1e-6,
) -> torch.Tensor:
    """Confidence-weighted residual loss with gradient and uncertainty terms.

    Args:
        residual: (B, S, H, W, C) signed residuals (C=1 for depth, 3 for points).
        confidence: (B, S, H, W) predicted confidence, ≥ 1.
        gt_depth: (B, S, H, W) normalized GT depth for the (1 + 1/D) factor.
        valid: (B, S, H, W) bool mask of pixels with usable GT.
        alpha: weight of the −log c uncertainty term. The paper does not give a
            value; the default follows VGGT. 0 disables the term (recommended
            by the paper when fine-tuning on very small datasets).
        depth_eps: clamp for D in 1 + 1/D, guarding degenerate GT depths.
    """
    residual = residual.float()
    confidence = confidence.float()

    scale_weight = 1.0 + 1.0 / gt_depth.clamp(min=depth_eps)
    main = confidence * scale_weight * residual.norm(dim=-1)
    loss = masked_mean(main, valid)

    grad_x, valid_x, grad_y, valid_y = spatial_gradient_magnitude(residual, valid)
    grad_terms = torch.cat(
        [
            (confidence[..., :, 1:] * grad_x).flatten(),
            (confidence[..., 1:, :] * grad_y).flatten(),
        ]
    )
    grad_mask = torch.cat([valid_x.flatten(), valid_y.flatten()])
    loss = loss + masked_mean(grad_terms, grad_mask)

    if alpha != 0.0:
        loss = loss - alpha * masked_mean(torch.log(confidence), valid)
    return loss
