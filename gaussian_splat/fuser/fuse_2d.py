# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""2D selection strategies (AnySplat@6dced92 ``render_conf`` port).

Source: the ``render_conf`` / ``conf_threshold`` -> ``conf_valid_mask``
branch of ``src/model/encoder/anysplat.py`` — a single global
``torch.quantile`` over every view's depth confidence with strict ``>``
semantics (ties at the threshold are dropped).
"""

import torch

from .base import Fuse2D

#: ``torch.quantile`` refuses inputs above this many elements ("quantile() input
#: tensor is too large" -- ATen asserts numel() <= 1 << 24). 16 views of a
#: 832x1296 frame is 17.25M pixels, so mip-NeRF-360 at its native quarter
#: resolution trips it while the same call at 512 does not.
_QUANTILE_MAX_ELEMENTS = 1 << 24


def _global_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """``torch.quantile(values.flatten(), q)`` without its 2**24 size ceiling.

    Above the limit the quantile is read off a full sort, reproducing
    ``torch.quantile``'s default "linear" interpolation exactly: with ``pos =
    q*(n-1)``, the result interpolates between the two order statistics
    bracketing ``pos``. Sorting has no size cap and is what quantile does
    internally anyway, so this is the same number by a slower route -- not a
    subsample or an approximation, which would make the confidence cut (and
    therefore the splat count) depend on resolution in a way nobody could see.
    """
    flat = values.flatten()
    if flat.numel() <= _QUANTILE_MAX_ELEMENTS:
        return torch.quantile(flat, q)

    ordered = flat.sort().values
    position = q * (ordered.numel() - 1)
    lower = int(position)  # floor: position is non-negative
    upper = min(lower + 1, ordered.numel() - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def confidence_mask(depth_conf: torch.Tensor, conf_threshold: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Global-quantile keep mask over per-pixel depth confidence.

    Args:
        depth_conf: (B, S, H, W) depth confidence (``DenseHead`` scale, > 1).
        conf_threshold: quantile in [0, 1); the lowest ``conf_threshold``
            fraction of pixels (across ALL scenes and views, matching
            AnySplat) falls below the cut.

    Returns:
        (mask, threshold): (B, S, H, W) bool with strict ``conf > threshold``
        semantics and the scalar cut.
    """
    threshold = _global_quantile(depth_conf, conf_threshold)
    return depth_conf > threshold, threshold


class ConfidenceFuse2D(Fuse2D):
    """AnySplat's ``render_conf``: keep pixels above a global confidence quantile."""

    def __init__(self, conf_threshold: float = 0.1) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold

    def forward(self, depth_conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return confidence_mask(depth_conf, self.conf_threshold)


class NoFuse2D(Fuse2D):
    """Keep every pixel (disable the 2D filter)."""

    def forward(self, depth_conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.ones_like(depth_conf, dtype=torch.bool), None
