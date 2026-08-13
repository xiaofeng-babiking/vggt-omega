# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stage contracts for ``GSDecoder``'s pluggable fusion pipeline.

The decoder (``vggt_omega/models/heads/gaussian_head.py``) runs a 2D
selection stage over pixel-aligned splat candidates, then a per-scene 3D
aggregation stage over the survivors. Both are swappable: subclass one of
these bases and pass the instance to ``GSDecoder(fuse_2d=..., fuse_3d=...)``.

Bases are ``nn.Module`` so learned strategies can carry parameters and
move/serialize with the decoder that owns them; this package stays free of
``vggt_omega`` imports so the model package can depend on it without cycles.
"""

import torch
import torch.nn as nn


class Fuse2D(nn.Module):
    """Pluggable 2D selection stage: which pixel-aligned candidates enter fusion."""

    def forward(self, depth_conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, S, H, W) depth confidence -> ((B, S, H, W) bool keep-mask, aux
        threshold tensor or ``None``)."""
        raise NotImplementedError


class Fuse3D(nn.Module):
    """Pluggable 3D aggregation stage: collapse one scene's candidates.

    Called once per scene on the 2D-filter survivors, flattened. ``depth`` /
    ``intrinsics`` are that scene's slices ((1, S, H, W) / (1, S, 3, 3)) for
    strategies that need the scene scale.
    """

    def forward(
        self,
        feats: torch.Tensor,
        points: torch.Tensor,
        conf: torch.Tensor,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """(K, C) feats, (K, 3) points, (K,) conf -> ((M, 3) points, (M, C)
        feats, info dict — e.g. ``{"voxel_size": ...}``)."""
        raise NotImplementedError
