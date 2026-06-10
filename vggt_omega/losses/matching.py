# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Matching loss on last-layer patch tokens (VGGT-Omega paper, Sec. 3.2, A.2).

``L_match = E_pos[−log σ(s)] + E_neg[−log(1 − σ(s))]`` where ``s`` is the
cosine similarity between L2-normalized patch tokens: a class-balanced binary
cross-entropy that pulls together tokens observing the same 3D location and
pushes apart tokens that provably observe different ones.

Pairs are constructed from GT geometry (Sec. A.2), never from predictions:

* positives — valid pixels sampled in each query frame are unprojected through
  the GT depth/camera and reprojected into every other frame; hits that stay
  inside the image (minus a boundary margin) and agree with the target depth
  (small relative tolerance) vote for (query patch, target patch) pairs, and
  target patches collecting more than ``positive_overlap_threshold`` of a
  query patch's samples become positives;
* negatives — random cross-frame patch pairs kept only when the target patch
  center is far from the query patch's epipolar line (dynamic content can
  break depth checks, so geometry must *prove* the mismatch) AND the patches
  differ in mean RGB; subsampled to balance the positives.

The paper leaves the sampling sizes and thresholds unspecified; they are
exposed in :class:`MatchingPairConfig` with conservative defaults.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vggt_omega.utils.geometry import compose_with_inverse


@dataclass
class MatchingPairConfig:
    """Knobs for GT-geometry pair construction (defaults are not from the paper)."""

    samples_per_patch: int = 16
    max_query_patches: int = 64
    min_query_correspondences: int = 4
    positive_overlap_threshold: float = 0.1
    depth_consistency_rel_tol: float = 0.01
    boundary_margin: int = 4
    min_epipolar_distance: float = 16.0
    min_rgb_distance: float = 0.1
    negative_candidate_factor: int = 4


@dataclass
class MatchingPairs:
    """Index tensors of shape (N, 5): [batch, query_frame, query_patch, target_frame, target_patch]."""

    positive: torch.Tensor
    negative: torch.Tensor


@torch.no_grad()
def build_matching_pairs(
    images: torch.Tensor,
    depths: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: int,
    config: MatchingPairConfig | None = None,
    generator: torch.Generator | None = None,
) -> MatchingPairs:
    """Construct positive/negative patch-token pairs from GT geometry.

    Args:
        images: (B, S, 3, H, W) in [0, 1], for the negative appearance check.
        depths: (B, S, H, W) GT depth (raw scale is fine; geometry-only use).
        extrinsics: (B, S, 3, 4) GT camera-from-world [R|t].
        intrinsics: (B, S, 3, 3) GT pinhole matrices.
        valid_mask: (B, S, H, W) bool, pixels with usable GT depth.
        patch_size: token patch size (model patch size).
        config: thresholds and sampling sizes.
        generator: optional RNG for deterministic sampling.
    """
    cfg = config or MatchingPairConfig()
    device = depths.device
    if generator is not None and generator.device != device:
        # torch sampling ops require the generator on the tensor device; derive
        # a device-matched generator deterministically from the provided one.
        seed = int(torch.randint(2**31 - 1, (1,), generator=generator).item())
        generator = torch.Generator(device=device).manual_seed(seed)
    batch_size, num_frames, height, width = depths.shape
    patch_h, patch_w = height // patch_size, width // patch_size
    num_patches = patch_h * patch_w

    depths = depths.float()
    extrinsics = extrinsics.float()
    intrinsics = intrinsics.float()

    positives: list[torch.Tensor] = []
    pos_count = 0
    for b in range(batch_size):
        for q in range(num_frames):
            pos = _positive_pairs_for_query(
                b, q, depths[b], extrinsics[b], intrinsics[b], valid_mask[b], patch_size, patch_w, num_patches, cfg, generator
            )
            if pos is not None:
                positives.append(pos)
                pos_count += pos.shape[0]

    empty = torch.zeros((0, 5), dtype=torch.long, device=device)
    positive = torch.cat(positives) if positives else empty
    negative = _negative_pairs(images, extrinsics, intrinsics, patch_size, patch_w, num_patches, pos_count, cfg, generator)
    return MatchingPairs(positive=positive, negative=negative)


def _positive_pairs_for_query(
    batch_idx: int,
    query: int,
    depths: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    patch_size: int,
    patch_w: int,
    num_patches: int,
    cfg: MatchingPairConfig,
    generator: torch.Generator | None,
) -> torch.Tensor | None:
    """Positive pairs for one (batch, query frame); tensors are per-batch-element."""
    device = depths.device
    num_frames, height, width = depths.shape

    valid_idx = valid_mask[query].reshape(-1).nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None
    num_samples = min(cfg.samples_per_patch * num_patches, valid_idx.numel())
    perm = torch.randperm(valid_idx.numel(), device=device, generator=generator)[:num_samples]
    pix = valid_idx[perm]
    py, px = pix // width, pix % width

    # Unproject the sampled query pixels to world coordinates.
    z_q = depths[query].reshape(-1)[pix]
    K_q = intrinsics[query]
    x_cam = torch.stack(
        [
            (px.float() - K_q[0, 2]) / K_q[0, 0] * z_q,
            (py.float() - K_q[1, 2]) / K_q[1, 1] * z_q,
            z_q,
        ],
        dim=-1,
    )
    R_q, t_q = extrinsics[query, :3, :3], extrinsics[query, :3, 3]
    x_world = (x_cam - t_q) @ R_q  # R_q^T @ (x - t) for row vectors

    query_patch = (py // patch_size) * patch_w + px // patch_size
    samples_per_query_patch = torch.bincount(query_patch, minlength=num_patches).float()

    # Project into all frames at once: (S, n, 3).
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    x_target = torch.einsum("sij,nj->sni", R, x_world) + t[:, None, :]
    z_t = x_target[..., 2]
    safe_z = z_t.clamp(min=1e-9)
    u = intrinsics[:, 0, 0, None] * x_target[..., 0] / safe_z + intrinsics[:, 0, 2, None]
    v = intrinsics[:, 1, 1, None] * x_target[..., 1] / safe_z + intrinsics[:, 1, 2, None]
    ui, vi = u.round().long(), v.round().long()

    margin = cfg.boundary_margin
    in_bounds = (
        (z_t > 1e-9)
        & (ui >= margin)
        & (ui < width - margin)
        & (vi >= margin)
        & (vi < height - margin)
    )
    ui_safe = ui.clamp(0, width - 1)
    vi_safe = vi.clamp(0, height - 1)
    flat = vi_safe * width + ui_safe
    target_depth = torch.gather(depths.reshape(num_frames, -1), 1, flat)
    target_valid = torch.gather(valid_mask.reshape(num_frames, -1), 1, flat)
    consistent = (
        in_bounds
        & target_valid
        & ((z_t - target_depth).abs() <= cfg.depth_consistency_rel_tol * target_depth)
    )
    consistent[query] = False

    target_patch = (vi_safe // patch_size) * patch_w + ui_safe // patch_size

    pair_rows: list[torch.Tensor] = []
    total_correspondences = torch.zeros(num_patches, device=device)
    for t_frame in range(num_frames):
        hits = consistent[t_frame]
        if not hits.any():
            continue
        combined = query_patch[hits] * num_patches + target_patch[t_frame][hits]
        counts = torch.bincount(combined, minlength=num_patches * num_patches).float()
        counts = counts.reshape(num_patches, num_patches)
        total_correspondences += counts.sum(dim=1)
        ratio = counts / samples_per_query_patch.clamp(min=1.0)[:, None]
        qp, tp = (ratio > cfg.positive_overlap_threshold).nonzero(as_tuple=True)
        if qp.numel():
            rows = torch.stack(
                [
                    torch.full_like(qp, batch_idx),
                    torch.full_like(qp, query),
                    qp,
                    torch.full_like(qp, t_frame),
                    tp,
                ],
                dim=-1,
            )
            pair_rows.append(rows)
    if not pair_rows:
        return None
    pairs = torch.cat(pair_rows)

    # Sample query patches with probability proportional to their total
    # correspondences, among those with sufficiently many.
    eligible = total_correspondences >= cfg.min_query_correspondences
    weights = total_correspondences * eligible.float()
    num_select = min(cfg.max_query_patches, int((weights > 0).sum()))
    if num_select == 0:
        return None
    selected = torch.multinomial(weights, num_select, replacement=False, generator=generator)
    keep = torch.isin(pairs[:, 2], selected)
    pairs = pairs[keep]
    return pairs if pairs.numel() else None


def _negative_pairs(
    images: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    patch_size: int,
    patch_w: int,
    num_patches: int,
    num_positives: int,
    cfg: MatchingPairConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Random cross-frame patch pairs passing the epipolar + appearance checks."""
    device = images.device
    batch_size, num_frames = images.shape[:2]
    empty = torch.zeros((0, 5), dtype=torch.long, device=device)
    if num_positives == 0 or num_frames < 2:
        return empty

    num_candidates = cfg.negative_candidate_factor * num_positives
    rand = lambda high, n: torch.randint(high, (n,), device=device, generator=generator)
    b_idx = rand(batch_size, num_candidates)
    q_frame = rand(num_frames, num_candidates)
    t_offset = 1 + rand(num_frames - 1, num_candidates)
    t_frame = (q_frame + t_offset) % num_frames
    q_patch = rand(num_patches, num_candidates)
    t_patch = rand(num_patches, num_candidates)

    centers = _patch_centers(num_patches, patch_w, patch_size, device)
    x_q = centers[q_patch]  # (N, 3) homogeneous pixel coords
    x_t = centers[t_patch]

    fundamental = _fundamental_matrix(
        extrinsics[b_idx, q_frame], extrinsics[b_idx, t_frame], intrinsics[b_idx, q_frame], intrinsics[b_idx, t_frame]
    )
    line = torch.einsum("nij,nj->ni", fundamental, x_q)
    line_norm = line[:, :2].norm(dim=-1)
    epipolar_distance = (line * x_t).sum(dim=-1).abs() / line_norm.clamp(min=1e-9)
    # Degenerate epipolar geometry (≈ pure rotation) proves nothing: reject.
    geometric = (line_norm > 1e-9) & (epipolar_distance >= cfg.min_epipolar_distance)

    patch_rgb = F.avg_pool2d(images.flatten(0, 1).float(), patch_size).flatten(-2)  # (B*S, 3, P)
    rgb_q = patch_rgb[b_idx * num_frames + q_frame, :, q_patch]
    rgb_t = patch_rgb[b_idx * num_frames + t_frame, :, t_patch]
    appearance = (rgb_q - rgb_t).norm(dim=-1) >= cfg.min_rgb_distance

    keep = (geometric & appearance).nonzero(as_tuple=True)[0]
    if keep.numel() > num_positives:
        sub = torch.randperm(keep.numel(), device=device, generator=generator)[:num_positives]
        keep = keep[sub]
    return torch.stack([b_idx[keep], q_frame[keep], q_patch[keep], t_frame[keep], t_patch[keep]], dim=-1)


def _patch_centers(num_patches: int, patch_w: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """Homogeneous pixel coordinates of patch centers -> (P, 3)."""
    idx = torch.arange(num_patches, device=device)
    cx = (idx % patch_w).float() * patch_size + patch_size / 2.0
    cy = (idx // patch_w).float() * patch_size + patch_size / 2.0
    return torch.stack([cx, cy, torch.ones_like(cx)], dim=-1)


def _fundamental_matrix(
    extr_q: torch.Tensor, extr_t: torch.Tensor, K_q: torch.Tensor, K_t: torch.Tensor
) -> torch.Tensor:
    """Fundamental matrices F with x_t^T F x_q = 0, for (N, 3, 4) / (N, 3, 3) batches."""
    rel = compose_with_inverse(extr_t, extr_q)  # target-from-query
    R_rel, t_rel = rel[..., :3, :3], rel[..., :3, 3]
    essential = _cross_matrix(t_rel) @ R_rel
    return torch.linalg.inv(K_t).transpose(-1, -2) @ essential @ torch.linalg.inv(K_q)


def _cross_matrix(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric cross-product matrices [v]_x for (N, 3) vectors."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def matching_loss(patch_tokens: torch.Tensor, pairs: MatchingPairs) -> torch.Tensor:
    """Class-balanced BCE on cosine similarity of L2-normalized patch tokens.

    Args:
        patch_tokens: (B, S, P, C) last-attention-layer patch tokens.
        pairs: indices from :func:`build_matching_pairs`.
    """
    tokens = F.normalize(patch_tokens.float(), dim=-1)
    loss = tokens.sum() * 0.0  # keeps the graph alive when there are no pairs
    if pairs.positive.numel():
        pos = pairs.positive
        s = (tokens[pos[:, 0], pos[:, 1], pos[:, 2]] * tokens[pos[:, 0], pos[:, 3], pos[:, 4]]).sum(dim=-1)
        loss = loss - F.logsigmoid(s).mean()
    if pairs.negative.numel():
        neg = pairs.negative
        s = (tokens[neg[:, 0], neg[:, 1], neg[:, 2]] * tokens[neg[:, 0], neg[:, 3], neg[:, 4]]).sum(dim=-1)
        loss = loss - F.logsigmoid(-s).mean()  # 1 − σ(s) = σ(−s)
    return loss
