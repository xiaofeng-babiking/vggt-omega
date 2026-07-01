"""Shared scene-level evaluation driver for VGGT-Omega inference.

Single source of truth for HOW predictions are scored against a dataset's
advertised ground truth, used by BOTH the single-GPU (inference.py) and the
frame-sharded distributed (distributed_inference.py) entrypoints so their
metrics.json is identical by construction.

Two scorers + an aggregator, split so the distributed path can reduce across
ranks between per-frame scoring (depth_sums on each rank) and final aggregation
(aggregate_depth_from_sums on rank 0). The dataset contract: a metric is scored
only when its GT modality is advertised (NYU ships placeholder poses, DL3DV
placeholder depth -> never scored); depth frames with no valid GT pixel are
skipped.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric

# Per-frame depth scalars aggregated (frame-weighted) into the mono_depth schema.
DEPTH_KEYS = ("abs_rel_mean", "abs_rel_rmse", "delta1", "delta2", "delta3")


def score_camera_pose(
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    *,
    modalities: Iterable[str],
    num_frames: int,
    vis_path: Optional[str] = None,
) -> Optional[dict]:
    """ATE / RPE on camera-to-world trajectories (Sim3-aligned). None when
    EXTRINSICS GT is not advertised or there are < 2 frames (RPE needs relative
    motion), so placeholder poses are never scored."""
    if "extrinsics" not in set(modalities) or num_frames < 2:
        return None
    return CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run(vis_path=vis_path)


def score_depth_frames(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    *,
    modalities: Iterable[str],
) -> list[dict]:
    """Per-frame mono-depth scoring (per-image median alignment). One dict per
    SCORED frame with the DEPTH_KEYS scalars; [] when DEPTH GT is not advertised.
    Frames with no valid GT pixel are skipped. gt_depth/pred_depth: (S, H, W)."""
    if "depths" not in set(modalities):
        return []
    out: list[dict] = []
    for i in range(gt_depth.shape[0]):
        if not (gt_depth[i] > 0).any():
            continue
        res = MonoDepthMetric(gt_depth[i], pred_depth[i], align="median").run()
        out.append(
            {
                "abs_rel_mean": float(res["abs_rel"]["mean"]),
                "abs_rel_rmse": float(res["abs_rel"]["rmse"]),
                "delta1": float(res["delta"]["delta1"]),
                "delta2": float(res["delta"]["delta2"]),
                "delta3": float(res["delta"]["delta3"]),
            }
        )
    return out


def depth_sums(per_frame: list[dict]) -> tuple[dict, int]:
    """Local partial sums of each DEPTH_KEY + the scored-frame count, for
    frame-weighted aggregation (single-process over all frames; distributed
    summed again across ranks before aggregation)."""
    sums = {k: float(sum(d[k] for d in per_frame)) for k in DEPTH_KEYS}
    return sums, len(per_frame)


def aggregate_depth_from_sums(sums: dict, count: int) -> Optional[dict]:
    """Frame-weighted means from summed depth scalars + scored-frame count.
    None when no frame was scored. The canonical mono_depth block."""
    if count <= 0:
        return None
    out = {k: sums[k] / count for k in DEPTH_KEYS}
    out["num_frames"] = int(count)
    return out


def assemble_metrics(
    scene: str,
    num_frames: int,
    camera_pose: Optional[dict],
    mono_depth: Optional[dict],
    **extra,
) -> dict:
    """The metrics.json dict. `extra` carries run-specific metadata (resolution
    for single-GPU; world_size / cp_strategy for distributed)."""
    return {
        "scene": scene,
        "num_frames": int(num_frames),
        "camera_pose": camera_pose,
        "mono_depth": mono_depth,
        **extra,
    }
