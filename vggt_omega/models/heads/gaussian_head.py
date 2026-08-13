# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

"""Port of AnySplat's gaussian head + decoding path (AnySplat@6dced92).

Two components, split exactly along the raw/activated boundary:
  - ``GSDPTHead`` predicts a raw (unactivated) per-frame, pixel-aligned
    gaussian parameter map from cached aggregator tokens.
  - ``GSDecoder`` turns those per-frame maps into ONE fused gaussian set in
    the reference coordinate system (VGGT's world frame — the first camera):
      a. 2D confidence filter: drop the pixels whose depth confidence falls
         below a global quantile (AnySplat's ``render_conf`` path);
      b. unproject the surviving pixel-aligned splats to the reference frame
         via predicted depth + cameras, and fuse them by voxelization — a
         differentiable softmax-over-``gs_conf`` weighted average of raw
         features and positions per voxel (AnySplat's anchor fusion);
      c. decode the fused anchors with the gaussian activations (density
         sigmoid + opacity warm-up, softplus scales, quaternion
         normalization, per-degree SH masking).
    Fusion averages RAW features before decoding — the AnySplat order, and
    not interchangeable with decoding first: averaging density logits is not
    averaging opacities, and averaging quaternions before normalizing is not
    averaging rotations. The activation core is exposed as
    ``GSDecoder.decode`` for callers that need an unfused per-pixel set.

Sources:
  - ``src/model/encoder/heads/vggt_dpt_gs_head.py`` -> ``VGGT_DPT_GS_Head``
    (the ``gaussian_param_head`` of the AnySplat encoder).
  - ``src/model/encoder/vggt/heads/dpt_head.py``    -> parent ``DPTHead``
    scaffolding (``FeatureFusionBlock`` / ``ResidualConvUnit`` /
    ``_make_scratch`` / ``_make_fusion_block`` / ``custom_interpolate``).
    Kept verbatim — NOT swapped for ``dense_head.py``'s variants — because the
    two DPT decoders genuinely differ: this head upsamples refinenet1 by
    scale-factor 2 and bilinearly interpolates to full resolution before a
    direct-RGB skip connection, while ``DenseHead`` stays at 1/4 scale and
    pixel-shuffles. Module attribute names match AnySplat exactly, so an
    AnySplat ``gaussian_param_head.*`` state dict loads into ``GSDPTHead``
    unchanged (verified key-for-key against the source class).
  - ``src/model/encoder/common/gaussian_adapter.py`` -> ``UnifiedGaussianAdapter``
    (scale/rotation/SH decoding, per-degree SH mask, ``stable_softplus``)
  - ``src/model/encoder/common/gaussians.py``        -> ``build_covariance`` /
    ``quaternion_to_matrix`` (XYZW/scipy quaternion order)
  - ``src/model/types.py``                           -> ``Gaussians``
  - ``src/model/encoder/anysplat.py``                -> ``map_pdf_to_opacity``
    (density -> opacity warm-up schedule), the density/means assembly in
    ``EncoderAnySplat.forward``, and ``pad_tensor_list`` (the batched
    fusion padding in ``GSDecoder.forward``). The ``render_conf`` quantile
    mask and ``voxelizaton_with_fusion`` (sic) ports live in
    ``gaussian_splat/fuser/`` as the ``Fuse2D`` / ``Fuse3D`` strategies
    this decoder plugs in.

Head architecture on top of the plain DPT decoder:
  - ``input_merger``: a 7x7 conv + ReLU on the raw input RGB, added to the
    fused DPT features after they are interpolated to full image resolution.
    This full-resolution skip is what lets per-pixel gaussians pick up
    high-frequency texture the 1/4-scale DPT pyramid has already smoothed out.
  - ``scratch.output_conv2``: re-declared with a wider hidden layer
    (128 vs the parent's 32) to decode the larger gaussian parameter vector.

Output layout per pixel, with ``d_sh = (sh_degree + 1) ** 2`` spherical
harmonic coefficients (AnySplat's shipped config: ``sh_degree=4`` -> d_sh = 25,
``raw_gs_dim`` = 1 + 7 + 3 * 25 = 83):

    gs_map (B, S, H, W, raw_gs_dim):   raw (unactivated) parameters
        [0:1]     density     -> sigmoid + opacity mapping in ``GSDecoder``
        [1:4]     scale       -> 1e-3 * softplus, clamped, in ``GSDecoder``
        [4:8]     quaternion  -> normalized in ``GSDecoder``; XYZW order
                                 (scipy convention, matching AnySplat — NOT
                                 the WXYZ order gsplat expects; downstream
                                 rendering must consume the covariances built
                                 by ``GSDecoder`` or reorder)
        [8:]      SH          -> laid out as 3 color groups of d_sh, masked
                                 per-degree in ``GSDecoder``
    gs_conf (B, S, H, W):              raw confidence channel; the voxel
        fusion's per-voxel softmax weights are formed over it, so gradients
        reach this head output through the fused anchors.

Deliberate deviations from the AnySplat source (interface only; numerics
unchanged for the shipped ``sh_degree=4`` config):
  - ``sh_degree`` replaces the raw ``output_dim`` argument; the channel count
    is derived (AnySplat computed 84 = raw_gs_dim + 1 at its call site).
  - ``output_conv2``'s hidden width and ``input_merger``'s output width are
    both ``features // 2`` (= 128) unconditionally. AnySplat switched them to
    32 when ``output_dim <= 50``, but that branch is broken at runtime (the
    128-channel DPT features cannot be added to a 32-channel RGB skip) and is
    unreachable in the shipped config.
  - tokens are indexed via ``aggregated_tokens_list[layer_idx]`` with a
    ``None`` check (this repo's sparse cache convention) instead of AnySplat's
    ``len(list) > 10`` heuristic, and are cast to fp32 on entry (our
    aggregator emits bf16 under autocast; ``DenseHead`` does the same).
  - forward args follow this repo's head convention
    (``patch_token_start``, keyword ``images``), and the conf channel is
    returned as a separate tensor instead of packed as channel 83.
  - ``patch_size`` defaults to 16 (this repo's ViT); AnySplat/VGGT used 14.
    No parameters depend on it, so checkpoint compatibility is unaffected.
  - the parent's unused ``activation`` / ``conf_activation`` / ``down_ratio``
    / ``feature_only`` arguments are dropped: AnySplat's subclass forward
    never calls ``activate_head`` and always renders at full resolution.
  - decoder centers come from this repo's ``unproject_depth`` +
    ``invert_transform_points`` (camera-from-world [R|t] extrinsics) instead
    of ``batchify_unproject_depth_map_to_point_map``; both are the OpenCV
    pinhole unprojection on the integer pixel grid.
  - the 2D confidence filter and the voxel fusion — mutually exclusive
    branches upstream (``cfg.voxelize`` picks one) — are composed in
    sequence here: the quantile cut prunes low-confidence pixels first and
    only survivors enter fusion. Both stages are pluggable strategies from
    ``gaussian_splat.fuser`` (``Fuse2D`` / ``Fuse3D`` subclasses passed to
    ``GSDecoder``): ``ConfidenceFuse2D`` + ``VoxelFuse3D`` are the
    defaults, ``NoFuse2D`` / ``NoFuse3D`` disable a stage, and
    ``GSDecoder(fuse_2d=NoFuse2D())`` is exactly AnySplat's voxelize
    branch.
  - AnySplat's ``opacity_conf`` modulation (post-decode, non-voxelized path
    only) and its test-time ``gs_prune`` are not carried into the fused
    pipeline.
  - the voxel size is scene-adaptive by default (``voxel_k`` x the median
    pixel footprint, see ``suggest_voxel_size``); AnySplat shipped an
    absolute ``voxel_size`` in scene-normalized units. Pass ``voxel_size``
    to pin it.
  - ``torch_scatter`` (optional) is imported
    lazily inside ``gaussian_splat.fuser``'s voxel fusion, so importing
    this module — and every non-3DGS head — works without it.
  - fusion carries one upstream quirk bit-for-bit: the per-voxel softmax
    max-stabilizer does not perfectly cancel against AnySplat's ``+ 1e-6``
    denominator epsilon; both are kept in place.
  - the pose-conditioned ``GaussianAdapter`` variant (pixel-space scale
    multipliers) is not ported; AnySplat's pose-free path never uses it.
  - ``einops``/``jaxtyping`` idioms are rewritten in plain torch (this repo
    does not depend on either); ``sh.broadcast_to`` is dropped because the
    reshape already yields the target shape.
  - opacity-mapping defaults (initial=0, final=0, warm_up=1) come from
    AnySplat's shipped ``config/model/encoder/anysplat.yaml`` and make the
    mapping the identity; pass other values to enable the warm-up schedule.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian_splat.fuser import ConfidenceFuse2D, Fuse2D, Fuse3D, VoxelFuse3D
from vggt_omega.utils.geometry import invert_transform_points, unproject_depth

from .utils import create_uv_grid, position_grid_to_embed

logger = logging.getLogger(__name__)


class GSDPTHead(nn.Module):
    """DPT head predicting raw per-pixel 3D gaussian parameters."""

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16,
        sh_degree: int = 4,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.sh_degree = sh_degree
        self.intermediate_layer_idx = intermediate_layer_idx

        # density(1) + scale(3) + quaternion(4) + SH(3 * d_sh)
        self.raw_gs_dim = 1 + 7 + 3 * (sh_degree + 1) ** 2
        output_dim = self.raw_gs_dim + 1  # + confidence

        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = features // 2

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, head_features_2, 7, 1, 3),
            nn.ReLU(),
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for GSDPTHead")

        _, num_frames, _, _, _ = images.shape

        if frames_chunk_size is None or frames_chunk_size >= num_frames:
            return self._forward_impl(aggregated_tokens_list, images, patch_token_start)

        assert frames_chunk_size > 0

        gs_map_chunks = []
        gs_conf_chunks = []
        for frames_start_idx in range(0, num_frames, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, num_frames)
            gs_map_chunk, gs_conf_chunk = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_token_start,
                frames_start_idx,
                frames_end_idx,
            )
            gs_map_chunks.append(gs_map_chunk)
            gs_conf_chunks.append(gs_conf_chunk)

        return torch.cat(gs_map_chunks, dim=1), torch.cat(gs_conf_chunks, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, num_frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which GSDPTHead needs.")
            x = x[:, :, patch_token_start:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            if x.dtype != torch.float32:
                x = x.float()

            x = x.reshape(batch_size * num_frames, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)

        fused = self.scratch_forward(multi_scale_features)
        fused = custom_interpolate(fused, size=(height, width), mode="bilinear", align_corners=True)

        direct_img_feat = self.input_merger(images.reshape(batch_size * num_frames, *images.shape[2:]).float())
        fused = fused + direct_img_feat

        fused = self._apply_pos_embed(fused, width, height)
        out = self.scratch.output_conv2(fused)

        gs_map = out[:, : self.raw_gs_dim].permute(0, 2, 3, 1)
        gs_conf = out[:, self.raw_gs_dim]

        gs_map = gs_map.view(batch_size, num_frames, *gs_map.shape[1:])
        gs_conf = gs_conf.view(batch_size, num_frames, *gs_conf.shape[1:])

        if gs_map.dtype != torch.float32 or gs_conf.dtype != torch.float32:
            raise TypeError(f"GSDPTHead outputs must be fp32, got gs_map={gs_map.dtype}, conf={gs_conf.dtype}")

        return gs_map, gs_conf

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        # No target size: AnySplat's refinenet1 upsamples by scale-factor 2 and
        # the caller interpolates to full resolution afterwards.
        out = self.scratch.refinenet1(out, layer_1_rn)
        return self.scratch.output_conv1(out)


@dataclass
class Gaussians:
    """A flat batch of world-space 3D gaussians (N = frames * height * width)."""

    means: torch.Tensor  # (B, N, 3)
    covariances: torch.Tensor  # (B, N, 3, 3)
    harmonics: torch.Tensor  # (B, N, 3, d_sh)
    opacities: torch.Tensor  # (B, N)
    scales: torch.Tensor  # (B, N, 3)
    rotations: torch.Tensor  # (B, N, 4), XYZW


def stable_softplus(x: torch.Tensor) -> torch.Tensor:
    """softplus(x) = log(1 + exp(x)) via an identity exact for all x.

    Inherited from the MUSA port, where the fused softplus kernel silently
    returned exactly 0.0 for every x <= -16.636 (in AnySplat that zeroed
    99,774 of 4,732,827 scales, which ply export then wrote as -inf). CUDA's
    F.softplus has no such defect, but this identity is kept as the shared
    implementation: it is bit-stable across backends, so checkpoints and
    numerics stay comparable with runs from the sibling MUSA repo.

    The identity cannot overflow, because exp(-|x|) <= 1 -- unlike a naive
    log1p(exp(x)), which goes inf above x ~ 88. For large x the log1p term
    falls under fp32 resolution (2.06e-9 at x = 20), so this also matches
    PyTorch's own threshold branch.
    """
    return F.relu(x) + torch.log1p(torch.exp(-x.abs()))


def quaternion_to_matrix(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation matrices from XYZW (scipy-order) quaternions -> (..., 3, 3)."""
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(*quaternions.shape[:-1], 3, 3)


def build_covariance(scale: torch.Tensor, rotation_xyzw: torch.Tensor) -> torch.Tensor:
    """World covariances R S S^T R^T from (..., 3) scales and (..., 4) XYZW quats."""
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return rotation @ scale @ scale.transpose(-1, -2) @ rotation.transpose(-1, -2)


_FEAT_PAD_VALUE = -1e10  # sigmoid(-1e10) = 0 -> padded anchors decode to opacity 0
_POINT_PAD_VALUE = -1e4  # AnySplat's "invalid voxel" position marker


@dataclass
class GSDecoderOutput:
    """Fused gaussians in the reference frame, plus the pipeline's masks."""

    gaussians: Gaussians  # (B, M, ...) fused anchors, right-padded per scene
    valid: torch.Tensor  # (B, M) bool, False on padded slots
    pixel_mask: torch.Tensor  # (B, S, H, W) bool, pixels that survived the 2D stage
    conf_threshold_value: torch.Tensor | None  # the 2D stage's cut (None if it has none)
    voxel_sizes: list[float | None]  # per scene; None when the 3D stage reports none


def _pad_tensor_list(tensor_list: list[torch.Tensor], length: int, value: float) -> torch.Tensor:
    padded = []
    for t in tensor_list:
        pad_len = length - t.shape[0]
        if pad_len > 0:
            padding = torch.full((pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype)
            t = torch.cat([t, padding], dim=0)
        padded.append(t)
    return torch.stack(padded)


class GSDecoder(nn.Module):
    """Fuse raw per-frame ``GSDPTHead`` maps into one reference-frame gaussian set."""

    def __init__(
        self,
        sh_degree: int = 4,
        opacity_initial: float = 0.0,
        opacity_final: float = 0.0,
        opacity_warm_up: int = 1,
        scale_multiplier: float = 1e-3,
        scale_max: float = 0.3,
        eps: float = 1e-8,
        fuse_2d: Fuse2D | None = None,
        fuse_3d: Fuse3D | None = None,
    ) -> None:
        """``fuse_2d`` / ``fuse_3d`` select the fusion strategies; ``None``
        picks the AnySplat defaults (``ConfidenceFuse2D(0.1)`` and
        scene-adaptive ``VoxelFuse3D()``). Pass ``NoFuse2D()`` /
        ``NoFuse3D()`` to disable a stage explicitly, or any other
        ``Fuse2D`` / ``Fuse3D`` implementation to swap it.
        """
        super().__init__()

        self.sh_degree = sh_degree
        self.opacity_initial = opacity_initial
        self.opacity_final = opacity_final
        self.opacity_warm_up = opacity_warm_up
        self.scale_multiplier = scale_multiplier
        self.scale_max = scale_max
        self.eps = eps
        self.fuse_2d = fuse_2d if fuse_2d is not None else ConfidenceFuse2D()
        self.fuse_3d = fuse_3d if fuse_3d is not None else VoxelFuse3D()

        # Bias SH towards a large DC component and small view-dependent terms
        # at initialization: degree d coefficients are damped by 0.1 * 0.25^d.
        sh_mask = torch.ones((self.d_sh,), dtype=torch.float32)
        for degree in range(1, sh_degree + 1):
            sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        self.register_buffer("sh_mask", sh_mask, persistent=False)

    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def raw_gs_dim(self) -> int:
        return 1 + 7 + 3 * self.d_sh

    def forward(
        self,
        gs_map: torch.Tensor,
        gs_conf: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        global_step: int | None = None,
    ) -> GSDecoderOutput:
        """``fuse_2d`` filter -> unproject to the reference frame -> ``fuse_3d``.

        Args:
            gs_map: (B, S, H, W, raw_gs_dim) raw ``GSDPTHead`` output.
            gs_conf: (B, S, H, W) raw head confidence; fusion softmax weights.
            depth: (B, S, H, W) or (B, S, H, W, 1) predicted depth.
            depth_conf: (B, S, H, W) or (B, S, H, W, 1) depth confidence
                (``DenseHead``); drives the 2D pixel filter.
            extrinsics: (B, S, 3, 4) camera-from-world [R|t], OpenCV. VGGT's
                world IS the reference (first-camera) frame, so unprojected
                points land in reference coordinates directly.
            intrinsics: (B, S, 3, 3) pinhole matrices in pixels.
            global_step: opacity warm-up step; ``None`` = fully warmed up.

        Returns:
            ``GSDecoderOutput`` with the fused, right-padded ``Gaussians``
            (B, M, ...) in the reference frame and the pipeline's masks.
        """
        if gs_map.shape[-1] != self.raw_gs_dim:
            raise ValueError(
                f"gs_map has {gs_map.shape[-1]} channels but GSDecoder("
                f"sh_degree={self.sh_degree}) expects {self.raw_gs_dim} channels"
            )
        if depth.dim() == gs_map.dim():
            depth = depth.squeeze(-1)
        if depth_conf.dim() == gs_map.dim():
            depth_conf = depth_conf.squeeze(-1)

        batch_size = gs_map.shape[0]

        # (b) part 1 — unproject the pixel-aligned splats to the reference frame.
        camera_points = unproject_depth(depth.float(), intrinsics.float())
        points = invert_transform_points(extrinsics.float(), camera_points)

        # (a) 2D selection.
        pixel_mask, threshold = self.fuse_2d(depth_conf)

        # A NON-FINITE threshold is a different failure from a too-aggressive one,
        # and must not be treated the same way. It means ``depth_conf`` itself went
        # non-finite: the quantile is then nan, ``conf > nan`` is False for every
        # pixel, and every scene looks "empty" no matter how mild the quantile is.
        #
        # That is a transient numerical event, and the trainer already knows how to
        # survive one -- Trainer._finish_micro_step votes across ranks on a
        # non-finite loss, skips the backward for the whole group, and aborts only
        # after 20 in a row. But that guard sits AFTER the loss, and raising here
        # kills the process before any loss exists, so the guard never gets a turn.
        # Measured cost of that gap: a single bad sample at step ~1600 raised on two
        # ranks of one node and took down a 16-rank, 2-node job that was otherwise
        # healthy (gate 100%, photo_context falling).
        #
        # So degrade instead of raising: keep whatever confidence is still finite so
        # the forward can complete, and let the nan flow into the loss where the
        # existing cross-rank guard handles it. Deliberately NOT a separate skip
        # mechanism -- routing into the loss guard inherits its rank-unanimous vote
        # (every rank must skip together or the next collective deadlocks) and its
        # consecutive-failure abort, neither of which is safe to reimplement here.
        #
        # A genuinely too-aggressive threshold still raises below, unchanged: that
        # one is a configuration error, it is reproducible, and skipping it would
        # silently train on nothing.
        if threshold is not None and not bool(torch.isfinite(torch.as_tensor(threshold))):
            fallback = torch.isfinite(depth_conf)
            # Count BEFORE the fixup below, or the message reports the fixup's own
            # all-True scenes as "finite" and hides the very thing worth seeing --
            # an all-nan confidence map then reads as 100% finite.
            n_finite = int(fallback.sum())
            # Per SCENE, not globally: one scene with no finite confidence would
            # still hit the empty-keep raise below even when others are fine.
            empty = ~fallback.reshape(batch_size, -1).any(dim=1)
            n_rescued = int(empty.sum())
            if n_rescued:
                fallback[empty] = True
            logger.warning(
                f"{type(self.fuse_2d).__name__} produced a non-finite threshold "
                f"({float(threshold)}): {fallback.numel() - n_finite} of {fallback.numel()} "
                f"depth_conf values are non-finite"
                + (f", and {n_rescued} scene(s) had none finite at all (kept whole to "
                   "preserve shapes)" if n_rescued else "")
                + ". Letting the forward complete so the trainer's cross-rank guard can skip "
                "the step. Persisting means the model is diverging, not that this fallback "
                "is wrong."
            )
            pixel_mask = fallback

        # (b) part 2 — 3D aggregation of the survivors, one scene at a time.
        fused_points: list[torch.Tensor] = []
        fused_feats: list[torch.Tensor] = []
        voxel_sizes: list[float | None] = []
        for b in range(batch_size):
            keep = pixel_mask[b].reshape(-1)
            if not bool(keep.any()):
                cut = f" (threshold {float(threshold):.4f})" if threshold is not None else ""
                raise ValueError(
                    f"{type(self.fuse_2d).__name__} kept 0 of {keep.numel()} pixels in scene {b}{cut}; "
                    "the 2D selection is too aggressive"
                )
            scene_points, scene_feats, info = self.fuse_3d(
                gs_map[b].reshape(-1, self.raw_gs_dim)[keep],
                points[b].reshape(-1, 3)[keep],
                gs_conf[b].reshape(-1)[keep],
                depth=depth[b : b + 1],
                intrinsics=intrinsics[b : b + 1],
            )
            fused_points.append(scene_points)
            fused_feats.append(scene_feats)
            voxel_sizes.append(info.get("voxel_size"))

        counts = torch.tensor([p.shape[0] for p in fused_points], device=gs_map.device)
        max_voxels = int(counts.max())
        valid = torch.arange(max_voxels, device=gs_map.device)[None, :] < counts[:, None]

        # (c) decode the fused anchors; padded slots decode to opacity 0
        # (sigmoid(-1e10) == 0) and are masked explicitly on top.
        gaussians = self.decode(
            _pad_tensor_list(fused_feats, max_voxels, _FEAT_PAD_VALUE),
            _pad_tensor_list(fused_points, max_voxels, _POINT_PAD_VALUE),
            global_step=global_step,
        )
        gaussians.opacities = gaussians.opacities * valid

        return GSDecoderOutput(
            gaussians=gaussians,
            valid=valid,
            pixel_mask=pixel_mask,
            conf_threshold_value=threshold,
            voxel_sizes=voxel_sizes,
        )

    def decode(
        self,
        raw_features: torch.Tensor,
        means: torch.Tensor,
        global_step: int | None = None,
    ) -> Gaussians:
        """Apply the gaussian activations to raw features on given centers.

        The activation core of ``forward``: fusion averages raw features and
        derives its own centers, then decodes here. Also the public escape
        hatch for an UNFUSED per-pixel set — decode the flattened ``gs_map``
        on unprojected points directly. Averaging raw features and decoding
        afterwards is the AnySplat order and is NOT interchangeable with
        decoding first: sigmoid/normalize do not commute with averaging.

        Args:
            raw_features: (..., raw_gs_dim) unactivated head output.
            means: (..., 3) gaussian centers, batch dims matching.
            global_step: opacity warm-up step; ``None`` = fully warmed up.
        """
        if raw_features.shape[-1] != self.raw_gs_dim:
            raise ValueError(
                f"raw features have {raw_features.shape[-1]} channels but GSDecoder("
                f"sh_degree={self.sh_degree}) expects {self.raw_gs_dim} channels"
            )
        raw_features = raw_features.float()

        density = raw_features[..., 0].sigmoid()
        opacities = self.map_pdf_to_opacity(density, global_step)

        scales = self.scale_multiplier * stable_softplus(raw_features[..., 1:4])
        scales = scales.clamp_max(self.scale_max)

        rotations = raw_features[..., 4:8]
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + self.eps)

        sh = raw_features[..., 8:].reshape(*raw_features.shape[:-1], 3, self.d_sh)
        sh = sh * self.sh_mask

        return Gaussians(
            means=means.float(),
            covariances=build_covariance(scales, rotations),
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

    def map_pdf_to_opacity(self, pdf: torch.Tensor, global_step: int | None = None) -> torch.Tensor:
        """AnySplat's opacity warm-up: https://www.desmos.com/calculator/opvwti3ba9"""
        if global_step is None:
            warmth = 1.0
        else:
            warmth = min(global_step / self.opacity_warm_up, 1.0)
        x = self.opacity_initial + warmth * (self.opacity_final - self.opacity_initial)
        exponent = 2.0**x

        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        has_residual=has_residual,
    )


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, activation: nn.Module, has_residual: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None, size: Tuple[int, int] | None = None
    ) -> torch.Tensor:
        output = x
        if self.has_residual:
            if residual is None:
                raise ValueError("FeatureFusionBlock requires a residual tensor when has_residual=True")
            output = output + self.resConfUnit1(residual)

        output = self.resConfUnit2(output)
        if size is None:
            output = custom_interpolate(output, scale_factor=2.0, mode="bilinear", align_corners=True)
        else:
            output = custom_interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        if scale_factor is None:
            raise ValueError("custom_interpolate requires either size or scale_factor")
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )

    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
    interpolated_chunks = [
        F.interpolate(
            chunk,
            size=size,
            mode=mode,
            align_corners=align_corners,
        )
        for chunk in chunks
    ]
    return torch.cat(interpolated_chunks, dim=0).contiguous()
