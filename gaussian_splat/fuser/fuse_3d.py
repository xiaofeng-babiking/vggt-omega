# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""3D aggregation strategies (AnySplat@6dced92 anchor-fusion port).

Source: ``voxelizaton_with_fusion`` (sic) in ``src/model/encoder/anysplat.py``
— every candidate is assigned to the voxel of its 3D position, and all
candidates sharing a voxel collapse into one anchor, a softmax-over-confidence
weighted average of positions and RAW features. Fusing raw features before
the gaussian activations is load-bearing (averaging density logits is not
averaging opacities); ``GSDecoder`` decodes only after this stage.

Carried over faithfully, including one quirk: the per-voxel max subtracted
before ``exp`` is the standard softmax stabilizer, which normally cancels
exactly — but AnySplat's ``+ 1e-6`` in the denominator makes the result
depend on it very slightly. It is kept (not detached, epsilon in place) so
this matches upstream bit-for-bit.

Uses ``torch_scatter`` when installed (imported lazily); otherwise exact
plain-torch fallbacks run, so this package stays importable without it.
"""

import logging

import torch

from .base import Fuse3D

logger = logging.getLogger(__name__)


def suggest_voxel_size(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    k: float = 0.5,
) -> float:
    """Scene-adaptive voxel size: ``k`` x one pixel's world footprint.

    A pixel at depth ``d`` covers roughly ``d / focal`` of world space, so
    that footprint is the natural unit for "are these two candidates the same
    surface point?". Below 1x, fusion merges only near-duplicate observations;
    above it, voxels start swallowing genuinely distinct detail.

    This matters because AnySplat's shipped ``voxel_size`` (0.001-0.002) is
    in *normalized* scene units — their pipeline divides the point cloud by
    its mean point norm, and VGGT-Omega's predictions already come out at
    mean ||p|| ~ 1 — and because it silently assumes a particular focal
    length. Two scenes reconstructed at focals of 208 px and 399 px give
    footprints differing by ~2x, so the same absolute voxel size means
    different things in each.

    Measured retention (fraction of per-pixel gaussians surviving fusion) on
    mipnerf360/garden and on a second scene, indexed by this ratio:

        k       garden    other     garden PSNR vs unfused
        0.25    99.9%     99.9%     67 dB
        0.50    98.6%     98.9%     57 dB   <- default, visually lossless
        1.00    81.4%     85.8%     34 dB   <- best compression/quality point
        2.00    50.6%     25.6%     18 dB   <- past the cliff, avoid

    Retention tracks ``k`` closely across unrelated scenes up to 1x, which an
    absolute voxel size does not. ``k=0.5`` defaults to the conservative end
    (AnySplat's shipped setting sits here, which is why their reported
    primitive reduction is only ~19%); raise it toward 1.0 to trade a few dB
    for roughly a fifth fewer primitives.

    Args:
        depth: (..., H, W) or (..., H, W, 1) predicted depth, same units as
            the points that will be fused.
        intrinsics: (..., 3, 3) pinhole matrices in pixels.
        k: multiple of the pixel footprint; see the table above.

    Returns:
        The suggested ``voxel_size``, as a plain float.

    Note:
        One scale is derived for everything passed in. For a batch mixing
        scenes of different scales, call per scene —
        ``suggest_voxel_size(depth[b : b + 1], intrinsics[b : b + 1])``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if depth.shape[-1] == 1 and depth.dim() == intrinsics.dim() + 2:
        depth = depth.squeeze(-1)

    positive = depth[depth > 0]
    if positive.numel() == 0:
        # Distinguish "no depth is positive" from "no depth is a NUMBER". A depth
        # map that has gone non-finite fails this test too -- `nan > 0` is False
        # for every element -- but it is a transient training event, not a
        # degenerate scene, and the trainer already survives it: the non-finite
        # value reaches the loss and Trainer._finish_micro_step skips the step for
        # the whole rank group. Raising here kills the process first, so the guard
        # never runs. Same defect as the one fixed in GSDecoder.decode, one stage
        # later in the same call chain: that fix let a non-finite confidence
        # through, and this raise then caught the non-finite DEPTH behind it and
        # took the job down anyway (observed on a 2-node run, ranks 8-9).
        #
        # So fall through with a placeholder scale and let the nan propagate. The
        # voxel size only bins points that are already nan, so its value cannot
        # change any result -- it just has to be finite and positive so the
        # binning arithmetic does not raise on its own.
        if not torch.isfinite(depth).all():
            logger.warning(
                "suggest_voxel_size: depth is non-finite, so no value is positive. "
                "Using a placeholder scale and letting the non-finite value reach the "
                "loss, where the trainer's cross-rank guard skips the step. Persisting "
                "means the model is diverging."
            )
            return float(k)
        raise ValueError("depth contains no positive values; cannot infer a scene scale")
    median_depth = positive.float().median()

    # Mean of fx and fy: pixels need not be square, and the footprint differs
    # per axis, so the isotropic voxel uses the average of the two.
    focal = torch.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]]).float().mean()
    if not torch.isfinite(focal) or focal <= 0:
        raise ValueError(f"intrinsics give a non-positive focal length ({float(focal)})")

    return float(k * median_depth / focal)


def _scatter_ops():
    """``(scatter_add, scatter_amax)`` over dim 0, from torch_scatter or torch.

    torch_scatter is optional (not installed in this repo's env today), so the
    plain-torch fallback is the path that runs; the compiled extension is used
    when present. Both are exact, not approximate:

    * ``index_add_`` is the same sum, and is differentiable in ``src``.
    * ``scatter_reduce_(reduce="amax")`` returns no gradient, but the only
      consumer here is the per-voxel softmax's stability shift, and a softmax is
      invariant to it — subtracting a detached max leaves both the value and the
      gradient unchanged.

    Signature is ``(src, index, num_out)`` rather than torch_scatter's
    ``(src, index, dim)``: the output length is derivable there and not here.
    """
    try:
        from torch_scatter import scatter_add, scatter_max
    except ModuleNotFoundError:
        def scatter_add_fallback(src, index, num_out):
            return src.new_zeros((num_out, *src.shape[1:])).index_add_(0, index, src)

        def scatter_amax_fallback(src, index, num_out):
            # include_self=False: every voxel index appears at least once (it
            # comes from torch.unique's inverse), so no entry keeps the fill.
            return src.new_zeros(num_out).scatter_reduce_(
                0, index, src, reduce="amax", include_self=False
            )

        return scatter_add_fallback, scatter_amax_fallback

    # dim_size=num_out is LOAD-BEARING, not a micro-optimisation. Without it
    # torch_scatter sizes its output from index.max()+1 -- from the data -- while
    # the fallback above sizes it from num_out, which the caller already knows. A
    # single garbage index therefore allocated 4_955_965_102_604_288_001 elements
    # and killed a 50k run at step 10030 with "Storage size calculation
    # overflowed". Passing num_out makes the two paths agree and makes the output
    # length independent of the index values entirely.
    return (
        lambda src, index, num_out: scatter_add(src, index, dim=0, dim_size=num_out),
        lambda src, index, num_out: scatter_max(src, index, dim=0, dim_size=num_out)[0],
    )


def fuse_by_voxel(
    feats: torch.Tensor,
    points: torch.Tensor,
    conf: torch.Tensor,
    voxel_size: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse one scene's candidates into per-voxel anchors.

    Differentiable end to end: gradients flow into the confidence through
    the softmax weights and into features/points through the weighted sums.

    Args:
        feats: (..., C) raw per-pixel features (``gs_map`` layout, any
            leading shape — flattened internally).
        points: (..., 3) positions in a shared frame, same units as
            ``voxel_size``, leading shape matching ``feats``.
        conf: (...) raw confidence logits (``gs_conf``).
        voxel_size: cube edge length.
        eps: denominator epsilon, AnySplat's 1e-6.

    Returns:
        (num_voxels, 3) fused positions and (num_voxels, C) fused features.
        Voxel order follows ``torch.unique``'s sorted index order.
    """
    scatter_add, scatter_amax = _scatter_ops()  # noqa: PLC0415 - keep the package importable without torch_scatter

    flat_points = points.reshape(-1, 3).float()
    flat_feats = feats.reshape(-1, feats.shape[-1]).float()
    flat_conf = conf.reshape(-1).float()

    # Casting a non-finite or very large coordinate to int32 is undefined, and the
    # garbage index that falls out is what fed the overflow above. 24 context views
    # makes this reachable in practice: that run logged 16 collective non-finite
    # steps in 10030, where the 16-view run logged 0 in 50000. Bound the
    # coordinates so the cast is always defined. The limit keeps the quotient
    # inside int32 with a full bit of headroom, and it sits orders of magnitude
    # outside any real scene, so finite geometry is untouched.
    limit = float(2**30) * voxel_size
    flat_points = torch.nan_to_num(
        flat_points, nan=0.0, posinf=limit, neginf=-limit
    ).clamp(-limit, limit)

    voxel_indices = (flat_points / voxel_size).round().int()
    voxels, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
    num_voxels = voxels.shape[0]

    # Per-voxel softmax over confidence.
    conf_voxel_max = scatter_amax(flat_conf, inverse, num_voxels)
    conf_exp = torch.exp(flat_conf - conf_voxel_max[inverse])
    voxel_weights = scatter_add(conf_exp, inverse, num_voxels)
    weights = (conf_exp / (voxel_weights[inverse] + eps)).unsqueeze(-1)

    voxel_points = scatter_add(flat_points * weights, inverse, num_voxels)
    voxel_feats = scatter_add(flat_feats * weights, inverse, num_voxels)
    return voxel_points, voxel_feats


class VoxelFuse3D(Fuse3D):
    """AnySplat's anchor fusion: softmax-over-confidence average per voxel.

    ``voxel_size=None`` derives the edge per scene as ``voxel_k`` x the median
    pixel footprint (``suggest_voxel_size``).
    """

    def __init__(self, voxel_size: float | None = None, voxel_k: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.voxel_k = voxel_k
        self.eps = eps

    def forward(
        self,
        feats: torch.Tensor,
        points: torch.Tensor,
        conf: torch.Tensor,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        voxel_size = self.voxel_size
        if voxel_size is None:
            if depth is None or intrinsics is None:
                raise ValueError("VoxelFuse3D(voxel_size=None) needs depth and intrinsics to derive one")
            voxel_size = suggest_voxel_size(depth, intrinsics, k=self.voxel_k)
        fused_points, fused_feats = fuse_by_voxel(feats, points, conf, voxel_size, eps=self.eps)
        return fused_points, fused_feats, {"voxel_size": voxel_size}


class NoFuse3D(Fuse3D):
    """Keep the per-pixel candidates as-is (disable 3D fusion)."""

    def forward(
        self,
        feats: torch.Tensor,
        points: torch.Tensor,
        conf: torch.Tensor,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return points.reshape(-1, 3), feats.reshape(-1, feats.shape[-1]), {}
