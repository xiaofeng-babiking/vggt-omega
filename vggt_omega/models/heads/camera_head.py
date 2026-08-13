# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.heads.gaussian_head import stable_softplus
from vggt_omega.models.layers import SelfAttentionBlock


class CameraHead(nn.Module):
    """Camera head used by the released VGGT-Omega checkpoints."""

    def __init__(self, dim_in: int = 2048, fov_activation: str = "softplus") -> None:
        super().__init__()

        if fov_activation not in FOV_ACTIVATIONS:
            raise ValueError(
                f"fov_activation must be one of {FOV_ACTIVATIONS}, got {fov_activation!r}"
            )
        #: Not a parameter or buffer -- a plain str, so checkpoints stay compatible
        #: in both directions and this never appears in a state dict.
        self.fov_activation = fov_activation
        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        # Head-local transformer blocks that mix camera and register tokens across frames.
        self.trunk = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim_in,
                    num_heads=16,
                    ffn_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1e-5,
                    use_qk_norm=False,
                    mask_k_bias=True,
                )
                for _ in range(4)
            ]
        )
        self.trunk_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.camera_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.Linear(dim_in // 2, 9, bias=True),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
    ) -> torch.Tensor:
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which CameraHead needs.")
        batch_size, num_frames, num_tokens, _ = tokens.shape

        if patch_token_start is None:
            raise ValueError("patch_token_start is required for CameraHead")
        if patch_token_start > num_tokens:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({num_tokens})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        camera_and_register_tokens = tokens[:, :, :patch_token_start]
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)

        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_token_start, -1)
        rope_sincos = None
        for block in self.trunk:
            camera_and_register_tokens = block(camera_and_register_tokens, rope_sincos)

        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames, patch_token_start, -1)
        camera_tokens = self.trunk_norm(camera_and_register_tokens[:, :, 0])
        return _apply_camera_activation(self.camera_branch(camera_tokens), self.fov_activation)


#: Smallest fov the head can emit, in radians. Kept off zero because the focal
#: length is ``(H/2)/tan(fov/2)`` -- at fov 0 that is a division by zero.
FOV_MIN = 0.01

#: Sharpness of the softplus. ``softplus(x, beta) -> relu(x)`` as beta grows; at
#: 50 the two agree to ~1e-15 for x >= 0.6, which is where real cameras live
#: (verify/fov_range_probe.py: fov 0.61-1.21 rad on the pretrained checkpoint),
#: while the softplus keeps a non-zero derivative for x < 0 where relu has none.
FOV_SOFTPLUS_BETA = 50.0

FOV_ACTIVATIONS = ("softplus", "relu")


def _apply_camera_activation(raw_camera: torch.Tensor, fov_activation: str = "softplus") -> torch.Tensor:
    """Map the head's raw 9-vector to ``[translation(3), quaternion(4), fov(2)]``.

    The fov activation is the one part that is not a free choice. The original
    ``relu(raw) + FOV_MIN`` is an ABSORBING state: once ``raw`` goes negative the
    gradient through the relu is exactly zero, so no loss term -- not the
    photometric one, not the teacher anchor -- can ever raise that fov again. The
    camera is stuck at FOV_MIN, which is a pencil camera (fy ~= 51200 at H=512)
    and the documented route to "memorised dust" geometry.

    ``softplus`` (the default) removes the trap while being a no-op where it
    matters: real fovs sit at 0.61-1.21 rad, ~61x the floor, and at beta=50 the
    softplus is identical to the relu there to well below fp32 resolution. That
    equivalence is load-bearing rather than cosmetic -- the frozen teacher runs
    this same code, so an activation that shifted predictions on the operating
    range would move the anchor's own target and corrupt the oracle the student
    is regressed onto.

    ``relu`` reproduces the original behaviour bit-exactly, for comparing against
    checkpoints trained before this change.
    """
    if fov_activation not in FOV_ACTIVATIONS:
        raise ValueError(
            f"fov_activation must be one of {FOV_ACTIVATIONS}, got {fov_activation!r}"
        )
    translation = raw_camera[..., :3]
    quaternion = raw_camera[..., 3:7]
    raw_fov = raw_camera[..., 7:]
    if fov_activation == "relu":
        fov = F.relu(raw_fov) + FOV_MIN
    else:
        # stable_softplus rather than F.softplus: the exact relu/exp/log1p
        # identity (see gaussian_head.stable_softplus) is bit-stable across
        # backends, so student/teacher numerics stay comparable with the
        # sibling MUSA repo's checkpoints -- whose fused softplus kernel had a
        # genuine dead zone (exact 0.0 below raw <= -0.333 at beta=50) that
        # would have reinstated the very trap this activation removes.
        #
        # softplus_beta(x) = softplus(beta*x) / beta; beta sharpens it toward
        # relu, which is what keeps it a no-op on the real operating range.
        fov = stable_softplus(raw_fov * FOV_SOFTPLUS_BETA) / FOV_SOFTPLUS_BETA + FOV_MIN
    return torch.cat([translation, quaternion, fov], dim=-1)
