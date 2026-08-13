# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Port of AnySplat's splatting decoder (AnySplat@6dced92).

Renders the ``Gaussians`` produced by ``vggt_omega.models.heads.GSDecoder``
back to images with gsplat's ``rasterization``.

Sources:
  - ``src/model/decoder/decoder_splatting_cuda.py`` -> ``DecoderSplattingCUDA``
    (the gsplat call and its arguments)
  - ``src/model/decoder/decoder.py``                -> ``DecoderOutput``

Scales and quaternions are passed to gsplat DIRECTLY; ``covars=`` is not used.
That matters for gradients: gsplat sets ``quats, scales = None, None`` the
moment ``covars`` is given (``rendering.py``), so a precomputed covariance makes
``Gaussians.scales`` and ``Gaussians.rotations`` unreachable from the loss —
fine for a pure forward, fatal for per-scene refinement, which optimises exactly
those two.

The rotations are REORDERED on the way in: ``Gaussians.rotations`` is XYZW
(scipy order, inherited from AnySplat) and gsplat documents wxyz, so this call
does ``[..., [3, 0, 1, 2]]``. AnySplat sidestepped the mismatch by passing
covariances; here it is converted instead. gsplat normalizes the quaternion
internally, so an unnormalized one is safe — which is what lets an optimizer
step the raw components without renormalizing between steps.

``Gaussians.covariances`` is consequently NOT read by this renderer. It stays on
the dataclass because ``GSDecoder`` fills it and other callers may want it, but
it can now drift from scales/rotations without changing what is rendered.

Deliberate deviations from the AnySplat source (interface only; per-camera
numerics unchanged):
  - camera conventions follow this repo: ``extrinsics`` are camera-from-world
    [R|t] (3x4 or 4x4, OpenCV) used directly as gsplat viewmats, and
    ``intrinsics`` are in pixels. AnySplat took camera-to-world matrices plus
    resolution-normalized intrinsics and converted both inside.
  - one ``rasterization`` call renders all V views of a scene (viewmats
    batched); AnySplat looped view-by-view with single-camera calls.
  - depth/alpha are squeezed with an explicit ``squeeze(-1)``; AnySplat's bare
    ``.squeeze()`` also collapses the view dimension when V = 1.
  - ``rasterize_mode`` / ``near_plane`` / ``radius_clip`` are arguments with
    AnySplat's hardcoded values as defaults. NOTE: rasterize_mode must match
    how the checkpoint was trained — models fine-tuned with antialiasing
    render garbage under "classic" and vice versa.
  - the unused pieces are not ported: ``near``/``far`` per-view planes
    (superseded by the hardcoded ``near_plane=1e-10``), ``depth_mode``,
    ``cam_rot_delta``/``cam_trans_delta``, ``make_scale_invariant``, the
    ``Decoder`` ABC/cfg plumbing, and ``DecoderOutput.lod_rendering``
    (always ``None``).
  - ``background_color`` defaults to white, AnySplat's shipped config value.
"""

from dataclasses import dataclass
from math import sqrt

import torch
from gsplat import rasterization

from vggt_omega.models.heads import Gaussians


def xyzw_to_wxyz(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) scipy-order XYZW -> gsplat/PLY-order WXYZ.

    Plain indexing, so it is differentiable and allocation-cheap. The same
    reorder appears in ``gaussian_splat.utils.dump`` for the .ply writer; both
    exist because ``Gaussians.rotations`` follows AnySplat's scipy order while
    gsplat and the INRIA .ply format both use WXYZ.
    """
    return quaternions[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) gsplat/PLY-order WXYZ -> scipy-order XYZW. Inverse of the above."""
    return quaternions[..., [1, 2, 3, 0]]


@dataclass
class RenderOutput:
    color: torch.Tensor  # (B, V, 3, H, W), finite and clamped to [0, 1]
    depth: torch.Tensor  # (B, V, H, W), finite
    alpha: torch.Tensor  # (B, V, H, W), finite; 0 wherever the raw render was not
    #: gsplat's per-call auxiliary dict (one per batch element), only when
    #: ``return_info``. Densification strategies read ``means2d``/``radii`` from
    #: it, and it is dropped otherwise so the ordinary forward keeps nothing alive.
    info: list | None = None
    #: Pixels whose raw colour or depth came back non-finite and were neutralised.
    #: Zero on a healthy render. Worth logging rather than ignoring: it is the only
    #: remaining signal that the rasterizer is producing garbage, now that doing so
    #: no longer poisons the loss.
    nonfinite_pixels: int = 0
    #: Split of the above by channel. Kept separate because it discriminates causes
    #: that the combined count cannot: a colour-only nan implicates the SH evaluation
    #: and the tile colour accumulator, a depth-only nan implicates the depth
    #: accumulation, and both together implicate something upstream of either --
    #: the conic, the projection, or the tile intersection itself.
    nonfinite_color_px: int = 0
    nonfinite_depth_px: int = 0


def render_by_gsplat(
    gaussians: Gaussians,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rasterize_mode: str = "classic",
    near_plane: float = 1e-10,
    radius_clip: float = 0.1,
    return_info: bool = False,
) -> RenderOutput:
    """Rasterize world-space gaussians into RGB + depth + alpha per view.

    Args:
        gaussians: flat gaussian set (``heads.Gaussians``), fields (B, N, ...).
        extrinsics: (B, V, 3, 4) or (B, V, 4, 4) camera-from-world [R|t],
            OpenCV coordinates (``encoding_to_camera`` output).
        intrinsics: (B, V, 3, 3) pinhole matrices in pixels.
        image_shape: (H, W) of the rendered views.
        background_color: RGB fill for empty pixels.
        rasterize_mode: "classic" (AnySplat) or "antialiased"; must match the
            mode the weights were trained with.
        near_plane: gsplat near plane (AnySplat hardcoded 1e-10).
        radius_clip: skip gaussians below this projected radius, in pixels.
        return_info: also return gsplat's auxiliary dict per batch element, which
            a densification strategy needs (``means2d`` gradients, ``radii``).
    """
    batch_size, num_views = intrinsics.shape[:2]
    height, width = image_shape

    if extrinsics.shape[-2:] == (3, 4):
        bottom_row = torch.zeros_like(extrinsics[..., :1, :])
        bottom_row[..., 0, 3] = 1.0
        extrinsics = torch.cat([extrinsics, bottom_row], dim=-2)

    # gsplat wants SH as (N, K, 3); Gaussians carries (B, N, 3, K).
    features = gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
    sh_degree = int(sqrt(features.shape[-2])) - 1
    backgrounds = torch.tensor(background_color, dtype=torch.float32, device=intrinsics.device)

    colors, depths, alphas, infos = [], [], [], []
    nonfinite = nonfinite_color = nonfinite_depth = 0
    for i in range(batch_size):
        rendering, alpha, info = rasterization(
            gaussians.means[i].float(),
            xyzw_to_wxyz(gaussians.rotations[i]).float(),  # gsplat documents wxyz
            gaussians.scales[i].float(),
            gaussians.opacities[i].float(),
            features[i].float(),
            extrinsics[i].float(),  # (V, 4, 4) camera-from-world == viewmats
            intrinsics[i].float(),
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB+D",
            packed=False,
            near_plane=near_plane,
            backgrounds=backgrounds.unsqueeze(0).expand(num_views, -1),
            radius_clip=radius_clip,
            rasterize_mode=rasterize_mode,
        )  # rendering (V, H, W, 4), alpha (V, H, W, 1)
        color, depth = torch.split(rendering, [3, 1], dim=-1)

        # The rasterizer can return non-finite pixels from entirely finite splats:
        # the tile accumulator sums an unbounded `color * alpha * T` with no finite
        # check, so two large opposite-sign colours give inf + (-inf) = nan, and a
        # splat at a camera centre makes gsplat's SH kernel compute rsqrtf(0) = inf
        # then 0 * inf = nan. Measured on 9-scene mip-NeRF-360: 8-10% of steps, flat
        # over thousands of steps, with every gaussian input field finite.
        #
        # `clamp` alone does NOT bound that. It maps +-inf to 1.0/0.0 but PROPAGATES
        # nan, so the clamp that was here before this comment let a nan render reach
        # the loss while the docstring claimed the output was bounded. nan_to_num has
        # to come first -- the same ordering trap documented for the LPIPS clamp in
        # gaussian_splat/utils/losses.py.
        bad_color = ~torch.isfinite(color).all(dim=-1)
        bad_depth = ~torch.isfinite(depth).squeeze(-1)
        bad = bad_color | bad_depth
        nonfinite += int(bad.sum())
        nonfinite_color += int(bad_color.sum())
        nonfinite_depth += int(bad_depth.sum())

        color = torch.nan_to_num(color, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        depth = torch.nan_to_num(depth.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        # Zero alpha on the affected pixels rather than leaving them "covered".
        # render_depth_loss keeps a pixel when alpha > render_depth_min_alpha, so a
        # neutralised depth of 0.0 left at high alpha would be regressed against the
        # teacher's real depth and inject a large fabricated error. Zeroed, it is
        # simply excluded -- the honest outcome for a pixel that was never rendered.
        alpha = torch.nan_to_num(alpha.squeeze(-1), nan=0.0, posinf=1.0, neginf=0.0)
        alpha = alpha.masked_fill(bad, 0.0)

        infos.append(info)
        colors.append(color.permute(0, 3, 1, 2))
        depths.append(depth)
        alphas.append(alpha)

    return RenderOutput(
        color=torch.stack(colors),
        depth=torch.stack(depths),
        alpha=torch.stack(alphas),
        info=infos if return_info else None,
        nonfinite_pixels=nonfinite,
        nonfinite_color_px=nonfinite_color,
        nonfinite_depth_px=nonfinite_depth,
    )
