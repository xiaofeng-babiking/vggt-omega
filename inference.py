"""Run VGGT-Omega on a dataset's sequences and evaluate camera-pose + mono-depth.

Drives inference from the *training* :class:`~vggt_omega.datasets.composed_dataset.ComposedDataset`,
instantiated from a per-dataset configure (``--configure``) loaded with OmegaConf,
so each frame is tensorized through the exact same contract the model is trained
on. The configure file carries the ``dataset`` + ``common_config`` (with the eval
overrides baked in) plus an ``inference`` block (``num_frames``, ``image_scale``)
that shapes the frames/resolution the dataset yields; the checkpoint and the
output/fusion knobs stay as command-line flags. Per sequence we:

  1. predict **camera poses** and **monocular depth** (+ a fused point cloud);
  2. dump them (depth/conf PNGs, ``cameras.json``, ``pointcloud.ply``);
  3. evaluate against the TUM ground truth -- camera pose (ATE / RPE) and mono
     depth (Abs Rel / delta).

TUM has no independent point-cloud ground truth: its "world points" are only the
GT depth re-projected through the GT poses, so the fused cloud is exported for
visualization but NOT scored (scoring it against re-projected depth is circular).

Conventions (see ``vggt_omega/datasets``): extrinsics are world-to-camera OpenCV
``[R|t]``; depth is metres with ``0`` = invalid. VGGT predicts geometry only up to
a global scale, so depth uses a per-image median scale and poses a Umeyama
``Sim3`` alignment before scoring.

Usage (the dataset, sequences, frame count and resolution come from the
``--configure`` file; the checkpoint and output/fusion knobs are flags), single GPU::

    python inference.py
    python inference.py --configure vggt_omega/datasets/config/tum.yaml
    python inference.py --checkpoint /path/to/model.pt --output_root /tmp/out

``--help`` lists every flag.
"""

import json
import os
import sys
import time

import gflags
import numpy as np
import torch
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.inference_common import (
    FLAGS,
    unproject_depth_map_to_point_map,
    world_to_camera_to_camera_to_world,
    save_uint16_image,
    write_ply,
    load_config,
    build_dataset,
    resolve_frame_ids,
    load_sample,
    gt_from_sample,
)
from vggt_omega.evaluates.scene import (
    score_camera_pose,
    score_depth_frames,
    depth_sums,
    aggregate_depth_from_sums,
    assemble_metrics,
)

logger = get_logger("vggt_omega.inference")

# --- command-line flags (inference.py-specific) ------------------------------
gflags.DEFINE_boolean(
    "profile_mfu",
    False,
    "After the timed forward, count forward FLOPs (FlopCounterMode) and log "
    "model-FLOPs utilization vs the A100 bf16 peak. Adds one extra forward pass.",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_status(tag: str) -> None:
    """Log current / peak GPU memory and free/total (no-op on CPU)."""
    if device != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    logger.info(
        f"[GPU {tag}] alloc={torch.cuda.memory_allocated() / 1e9:.1f}G "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}G "
        f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}G "
        f"free={free / 1e9:.1f}G / {total / 1e9:.1f}G"
    )


A100_BF16_PEAK = 312e12  # dense bf16 tensor-core peak FLOP/s (A100 80GB)


def _log_mfu(model, images: torch.Tensor, infer_secs: float) -> None:
    """Count forward FLOPs once (FlopCounterMode, FMA=2) and log throughput +
    model-FLOPs utilization against the A100 bf16 peak.

    The rate uses the CLEAN forward time ``infer_secs``; the counted pass here is
    for FLOPs only (its own wall time carries dispatch overhead, so it is not
    timed). FlopCounterMode tallies real mm / conv / SDPA ops, so attention's
    S^2 term is included -- which the 2*N*D analytic shortcut would miss for
    VGGT's alternating frame/global attention. Adds one extra forward (opt-in)."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        model(images)
    flops = fc.get_total_flops()
    logger.info(
        f"[MFU] {flops / 1e12:.1f} TFLOP fwd / {infer_secs:.1f}s = "
        f"{flops / infer_secs / 1e12:.1f} TFLOP/s -> "
        f"{flops / infer_secs / A100_BF16_PEAK * 100:.1f}% of A100 bf16 "
        f"({A100_BF16_PEAK / 1e12:.0f} TF/s peak)"
    )


# --- 2. Model + inference ----------------------------------------------------
def build_model() -> VGGTOmega:
    """Build VGGT-Omega once and load the checkpoint."""
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(FLAGS.checkpoint, map_location="cpu"))
    gpu_status("model loaded")
    return model


def run_inference(model: VGGTOmega, images: torch.Tensor) -> dict:
    """Run the forward pass on one sequence's frames and extract prediction arrays.

    ``images`` is the training-identical ``(S,3,H,W)`` tensor in ``[0,1]`` produced
    by the dataset loader -- no hand-rolled normalization here.
    """
    images = images.contiguous().to(device)
    logger.info(
        f"running inference on {images.shape[0]} frames @ {images.shape[-2]}x{images.shape[-1]} (HxW) ..."
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_infer = time.time()
    try:
        with torch.inference_mode():
            predictions = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        gpu_status("OOM")
        total = torch.cuda.mem_get_info()[1] / 1e9
        # native-VGA cost ~= 5.9 + 0.087*N GB (README curve scaled to ~1200 tok/frame).
        fits = int(max(0, (total - 5.9) / 0.087))
        raise SystemExit(
            f"\nCUDA OOM on {images.shape[0]} frames @ {images.shape[-2]}x{images.shape[-1]}. "
            f"A single {total:.0f}G GPU fits ~{fits} native-VGA frames in one pass. "
            f"Lower --num_frames (<=~{fits}) or drop --image_scale."
        )
    infer_secs = time.time() - t_infer
    logger.info(f"inference done in {infer_secs:.1f}s")
    gpu_status("after forward")
    if FLAGS.profile_mfu:
        _log_mfu(model, images, infer_secs)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )

    # Pull to CPU/numpy (float() guards against bf16, which numpy can't hold).
    return {
        "pred_depth": predictions["depth"].float().cpu().numpy()[0],  # (S, H, W, 1)
        "pred_conf": predictions["depth_conf"].float().cpu().numpy()[0],  # (S, H, W)
        "images_pred": predictions["images"]
        .float()
        .cpu()
        .numpy()[0],  # (S, 3, H, W) [0,1]
        "pred_extrinsics": extrinsics.float().cpu().numpy()[0],  # (S, 3, 4) world->cam
        "pred_intrinsics": intrinsics.float().cpu().numpy()[0],  # (S, 3, 3) pixels
    }


def dump_and_eval(
    seq_name: str,
    output_dir: str,
    frame_ids,
    loaded: dict,
    pred: dict,
    conf_percentile: float,
    max_points: int,
) -> None:
    """Dump predictions, fuse the PLY, evaluate metrics, and report for one sequence."""
    pred_depth = pred["pred_depth"]
    pred_conf = pred["pred_conf"]
    images_np = pred["images_pred"]
    pred_extrinsics = pred["pred_extrinsics"]
    pred_intrinsics = pred["pred_intrinsics"]
    gt_depth = loaded["gt_depth"]
    gt_extrinsics = loaded["gt_extrinsics"]

    num_f, height, width = pred_depth.shape[:3]
    pred_depth_2d = pred_depth[..., 0]  # (S, H, W)
    images_hwc_pred = np.transpose(images_np, (0, 2, 3, 1))  # (S, H, W, 3)

    # --- 3. Dump predicted depth/conf PNGs + cameras.json ------------------------
    depth_dir = os.path.join(output_dir, "depth")
    conf_dir = os.path.join(output_dir, "conf")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    valid_depth = np.isfinite(pred_depth_2d) & (pred_depth_2d > 0)
    depth_max = float(pred_depth_2d[valid_depth].max()) if valid_depth.any() else 1.0
    depth_scale = 65535.0 / depth_max if depth_max > 0 else 1.0
    finite_conf = np.isfinite(pred_conf)
    conf_max = float(pred_conf[finite_conf].max()) if finite_conf.any() else 1.0
    conf_scale = 65535.0 / conf_max if conf_max > 0 else 1.0

    frames_meta = []
    for i in tqdm(range(num_f), desc="dump depth/conf", unit="frame"):
        name = f"frame_{i:04d}.png"
        save_uint16_image(pred_depth_2d[i], depth_scale, os.path.join(depth_dir, name))
        save_uint16_image(pred_conf[i], conf_scale, os.path.join(conf_dir, name))
        frames_meta.append(
            {
                "index": int(i),
                "frame_id": int(frame_ids[i]),
                "depth": os.path.join("depth", name),
                "conf": os.path.join("conf", name),
                "intrinsics": pred_intrinsics[i].tolist(),
                "extrinsics": pred_extrinsics[i].tolist(),
            }
        )

    camera_meta = {
        "scene": seq_name,
        "image_width": int(width),
        "image_height": int(height),
        "num_frames": int(num_f),
        "depth_scale": depth_scale,
        "depth_max": depth_max,
        "conf_scale": conf_scale,
        "conf_max": conf_max,
        "depth_unit": "uint16_value / depth_scale",
        "extrinsics_convention": "world_to_camera (OpenCV), 3x4 [R|t]",
        "intrinsics_convention": "pixels, 3x3 K",
        "frames": frames_meta,
    }
    with open(os.path.join(output_dir, "cameras.json"), "w") as f:
        json.dump(camera_meta, f, indent=2)

    # --- 4. Fuse predicted depth + RGB into a world-frame PLY --------------------
    pred_world_points = unproject_depth_map_to_point_map(
        pred_depth, pred_extrinsics, pred_intrinsics
    )  # (S, H, W, 3)
    points = pred_world_points.reshape(-1, 3)
    colors = (images_hwc_pred.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_flat = pred_conf.reshape(-1)
    depth_flat = pred_depth_2d.reshape(-1)

    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf_flat) & (depth_flat > 0)
    if conf_percentile > 0 and mask.any():
        threshold = np.percentile(conf_flat[mask], conf_percentile)
        mask &= conf_flat >= threshold
    points, colors = points[mask], colors[mask]

    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[keep], colors[keep]

    ply_path = os.path.join(output_dir, "pointcloud.ply")
    write_ply(ply_path, points, colors)

    # --- 5. Evaluate against the dataset's advertised GT modalities (shared path) -
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    modalities = set(loaded.get("modalities") or [])

    camera_pose_metrics = score_camera_pose(
        world_to_camera_to_camera_to_world(gt_extrinsics),
        world_to_camera_to_camera_to_world(pred_extrinsics),
        modalities=modalities,
        num_frames=num_f,
        vis_path=os.path.join(metrics_dir, "camera_pose"),
    )
    mono_depth_metrics = aggregate_depth_from_sums(
        *depth_sums(score_depth_frames(gt_depth, pred_depth_2d, modalities=modalities))
    )

    all_metrics = assemble_metrics(
        seq_name, num_f, camera_pose_metrics, mono_depth_metrics,
        resolution=[int(height), int(width)],
    )
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- 6. Report ---------------------------------------------------------------
    report = [
        f"[{seq_name}] {num_f} frames @ {height}x{width} -> {output_dir}",
        f"  point cloud: {points.shape[0]} points -> {ply_path}",
    ]
    if camera_pose_metrics is not None:
        report += [
            "  camera pose (Sim3-aligned):",
            f"    ATE  rmse = {camera_pose_metrics['ate']['rmse']:.4f} m",
            f"    RPE  trans rmse = {camera_pose_metrics['rpe_trans']['rmse']:.4f} m",
            f"    RPE  rot   rmse = {camera_pose_metrics['rpe_rot']['rmse']:.4f} deg",
        ]
    else:
        report.append("  camera pose: skipped (no EXTRINSICS ground truth)")
    if mono_depth_metrics is not None:
        report += [
            "  mono depth (median-aligned, mean over frames):",
            f"    Abs Rel = {mono_depth_metrics['abs_rel_mean']:.4f}",
            f"    delta1  = {mono_depth_metrics['delta1']:.4f}",
        ]
    else:
        report.append("  mono depth: skipped (no DEPTH ground truth)")
    report.append(f"  full metrics -> {os.path.join(metrics_dir, 'metrics.json')}")
    logger.info("\n".join(report))


def main():
    cfg = load_config()
    inf = cfg.inference
    dataset = build_dataset(cfg)
    model = build_model()

    num_seqs = dataset.num_sequences()
    logger.info(
        f"{num_seqs} sequence(s) @ {dataset.img_size} long side (scale {inf.image_scale}), "
        f"num_frames={'all' if inf.num_frames <= 0 else inf.num_frames}"
    )

    for seq_index in range(num_seqs):
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = resolve_frame_ids(dataset, seq_index, inf.num_frames)
        logger.info(
            f"[{seq_name}] ({seq_index + 1}/{num_seqs}) {len(frame_ids)} frames"
        )

        sample = load_sample(dataset, seq_index, frame_ids)
        pred = run_inference(model, sample["images"])

        output_dir = os.path.join(FLAGS.output_root, seq_name)
        dump_and_eval(
            seq_name,
            output_dir,
            frame_ids,
            gt_from_sample(sample),
            pred,
            FLAGS.conf_percentile,
            FLAGS.max_points,
        )


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)  # parse command-line flags
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUse --help for the full flag list.")
    main()
