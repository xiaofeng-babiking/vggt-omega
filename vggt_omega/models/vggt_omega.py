# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from gaussian_splat.fuser import Fuse2D, Fuse3D

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, GSDecoder, GSDPTHead, TextAlignmentHead
from vggt_omega.utils.pose_enc import encoding_to_camera


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction.

    With ``enable_3dgs=True`` the AnySplat-ported gaussian head
    (``heads/gaussian_head.py``) runs as an extra head and predictions gain
    raw ``gs_map`` / ``gs_conf`` — plus decoded world-space ``gaussians``
    when the forward is called with ``decode_gaussians=True``.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        enable_3dgs: bool = False,
        gs_sh_degree: int = 4,
        gs_fuse_2d: Fuse2D | None = None,
        gs_fuse_3d: Fuse3D | None = None,
        gs_opacity_initial: float = 0.0,
        gs_opacity_final: float = 0.0,
        gs_opacity_warm_up: int = 1,
        fov_activation: str = "softplus",
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        # fov_activation: "softplus" (default) keeps a gradient below zero so a
        # shrinking fov can recover; "relu" reproduces the original absorbing
        # floor bit-exactly. See heads/camera_head._apply_camera_activation.
        self.camera_head = (
            CameraHead(dim_in=2 * embed_dim, fov_activation=fov_activation) if enable_camera else None
        )
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None
        # AnySplat gaussian head port: raw per-pixel gaussian parameters from
        # the same cached aggregator layers DenseHead reads. GSDecoder holds no
        # learnable state today (a non-persistent SH mask buffer; the default
        # fusion strategies are parameter-free), so released checkpoints still
        # load with strict=False, missing only gs_dpt_head.*. gs_fuse_2d /
        # gs_fuse_3d swap the decoder's fusion strategies (heads.Fuse2D /
        # heads.Fuse3D); None picks the AnySplat defaults.
        self.gs_dpt_head = (
            GSDPTHead(dim_in=2 * embed_dim, patch_size=patch_size, sh_degree=gs_sh_degree) if enable_3dgs else None
        )
        # gs_opacity_*: AnySplat's opacity warm-up, off by default (initial ==
        # final == 0 makes map_pdf_to_opacity the identity). A positive x pushes
        # opacity UP -- at pdf=0.5, x=2 gives 0.89 against the identity's 0.5 --
        # so warming from a positive initial down to 0 forces the model to
        # commit to opaque surfaces early, before it can learn that fading
        # everything out is a cheap way to lower a photometric loss.
        self.gs_decoder = (
            GSDecoder(
                sh_degree=gs_sh_degree,
                fuse_2d=gs_fuse_2d,
                fuse_3d=gs_fuse_3d,
                opacity_initial=gs_opacity_initial,
                opacity_final=gs_opacity_final,
                opacity_warm_up=gs_opacity_warm_up,
            )
            if enable_3dgs
            else None
        )

    def forward(
        self,
        images: torch.Tensor,
        return_last_patch_tokens: bool = False,
        return_features: bool = False,
        decode_gaussians: bool = False,
        global_step: int | None = None,
    ) -> dict:
        """``decode_gaussians`` also runs ``GSDecoder`` on the predicted depth
        and cameras (requires ``enable_3dgs`` plus the camera and depth
        heads); ``global_step`` drives its opacity warm-up schedule (``None``
        = fully warmed up).
        """
        if decode_gaussians and self.gs_dpt_head is None:
            raise ValueError("decode_gaussians=True requires enable_3dgs=True")
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        if return_last_patch_tokens:
            # Last-layer patch tokens feed the matching loss; opt-in so eval/
            # inference forwards skip the extra tensor (the trainer passes the flag).
            predictions["patch_tokens"] = final_tokens[:, :, patch_token_start:]
        if return_features:
            # Cached aggregator layers (indices 4/11/17/23 by default), each
            # (B,S,N,2*embed_dim). The self-supervised distillation loss matches
            # these token features between student and EMA teacher across layers.
            predictions["features"] = [t for t in aggregated_tokens_list if t is not None]
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.gs_dpt_head is not None:
                gs_map, gs_conf = self.gs_dpt_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["gs_map"] = gs_map  # (B, S, H, W, raw_gs_dim), raw
                predictions["gs_conf"] = gs_conf

                if decode_gaussians:
                    if "pose_enc" not in predictions or "depth" not in predictions:
                        raise ValueError(
                            "decode_gaussians=True needs the camera and depth heads enabled "
                            "(gaussian centers come from predicted depth and cameras, and the "
                            "2D filter reads the depth confidence)"
                        )
                    height, width = images.shape[-2:]
                    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], (height, width))
                    # Full GSDecoder pipeline: 2D confidence filter -> unproject
                    # to the reference frame -> voxel fusion -> decode.
                    decoded = self.gs_decoder(
                        gs_map,
                        gs_conf,
                        predictions["depth"],
                        predictions["depth_conf"],
                        extrinsics,
                        intrinsics,
                        global_step=global_step,
                    )
                    predictions["gaussians"] = decoded.gaussians
                    predictions["gaussians_valid"] = decoded.valid
                    predictions["gaussians_pixel_mask"] = decoded.pixel_mask

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
