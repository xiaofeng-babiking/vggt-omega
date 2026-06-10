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

import cv2
import gflags
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from vggt_omega import datasets as vggt_datasets
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.datasets.composed_dataset import ComposedDataset
from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric

logger = get_logger("vggt_omega.inference")

# Per-dataset configure dir (ships tum.yaml, the default --configure target).
DATASET_CONFIG_DIR = os.path.join(os.path.dirname(vggt_datasets.__file__), "config")

# --- command-line flags ------------------------------------------------------
# Inputs to locate (checkpoint, configure) plus the output/fusion knobs. The
# dataset-loading knobs (num_frames, image_scale) live in the configure YAML's
# `inference` block instead, since they shape what frames/resolution the dataset
# yields and so belong with the dataset definition.
FLAGS = gflags.FLAGS
gflags.DEFINE_string(
    "checkpoint",
    "/jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt",
    "Path to the VGGT-Omega checkpoint (.pt).",
)
gflags.DEFINE_string(
    "configure",
    os.path.join(DATASET_CONFIG_DIR, "tum.yaml"),
    "Path to the per-dataset configure (.yaml): `dataset` + `common_config` + an "
    "`inference` block (num_frames, image_scale), loaded with OmegaConf and "
    "instantiated like training.",
)
gflags.DEFINE_string(
    "output_root",
    "outputs",
    "Output root directory; a per-sequence subdirectory is created under it.",
)
gflags.DEFINE_float(
    "conf_percentile",
    20.0,
    "Drop the lowest this-percent of points (by confidence) from the fused cloud.",
)
gflags.DEFINE_integer(
    "max_points",
    5_000_000,
    "Cap on exported point-cloud size (the fused cloud is a visualization, not scored).",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def effective_long_side(native_long: int, image_scale: float) -> int:
    """Native long side scaled by `image_scale`, snapped to a /16 multiple (ViT-friendly)."""
    return max(16, int(round(native_long * image_scale / 16)) * 16)


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


# --- geometry / IO helpers ---------------------------------------------------
def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Unproject per-frame depth into a common world frame -> (S, H, W, 3).

    `extrinsic` is world-to-camera (OpenCV) [R|t], so world = R^T @ (cam - t).
    """
    depth = depth_map[..., 0]
    num, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num, height, width))
    y = np.broadcast_to(y[None], (num, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def world_to_camera_to_camera_to_world(w2c: np.ndarray) -> np.ndarray:
    """Invert world-to-camera ``(S, 3, 4)`` -> camera-to-world ``(S, 4, 4)``.

    The translation column of the result is the camera centre in world coords --
    the trajectory positions :class:`CameraPoseMetric` consumes.
    """
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    rot_c2w = np.transpose(rotation, (0, 2, 1))
    trans_c2w = -np.einsum("sij,sj->si", rot_c2w, translation)
    c2w = np.tile(np.eye(4), (w2c.shape[0], 1, 1))
    c2w[:, :3, :3] = rot_c2w
    c2w[:, :3, 3] = trans_c2w
    return c2w


def save_uint16_image(array: np.ndarray, scale: float, path: str) -> None:
    """Scale a float map and write it as a single-channel 16-bit PNG."""
    scaled = np.rint(array.astype(np.float64) * scale)
    scaled = np.clip(scaled, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, scaled)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a colored point cloud as a binary little-endian PLY."""
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    vertex = np.empty(
        n,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = (
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


# --- 1. Load the dataset (RGB + GT depth/poses/intrinsics) --------------------
def load_config():
    """Load the per-dataset configure (.yaml) with OmegaConf. It must define
    ``dataset`` + ``common_config`` (instantiated like training) and an
    ``inference`` block with ``num_frames`` and ``image_scale`` (the dataset
    sampling / resolution knobs). Output/fusion knobs are command-line flags."""
    return OmegaConf.load(FLAGS.configure)


def build_dataset(cfg) -> ComposedDataset:
    """Instantiate the *training* ComposedDataset from the per-dataset configure.

    The configure file (loaded by :func:`load_config`) supplies ``dataset`` and
    ``common_config`` verbatim, with the eval overrides (training off, ordered
    ids, deterministic resize, no augmentation) already baked into the YAML and
    instantiated exactly as training does -- so inference cannot silently drift.
    The target long side is the data's NATIVE long side x ``inference.image_scale``,
    read from the dataset itself rather than hardcoded.
    """
    dataset = instantiate(cfg.dataset, common_config=cfg.common_config, _recursive_=False)
    native_h, native_w = dataset.native_image_size()
    dataset.set_img_size(
        effective_long_side(max(native_h, native_w), cfg.inference.image_scale)
    )
    return dataset


def resolve_frame_ids(dataset: ComposedDataset, seq_index: int, num_frames: int) -> np.ndarray:
    """Ordered frame ids for one sequence: all frames (``num_frames<=0``) or evenly spaced."""
    num_available = dataset.sequence_num_frames(seq_index)
    if num_frames <= 0 or num_frames >= num_available:
        return np.arange(num_available)  # ALL frames, ordered
    return np.linspace(0, num_available - 1, num_frames).round().astype(int)


def load_sample(dataset: ComposedDataset, seq_index: int, frame_ids) -> dict:
    """Training-identical tensorized sample for ``frame_ids`` of one sequence
    (images ``(S,3,H,W)`` in ``[0,1]`` + the full GT modality set)."""
    t_load = time.time()
    native_h, native_w = dataset.native_image_size(seq_index)
    aspect_ratio = min(native_h, native_w) / max(native_h, native_w)
    sample = dataset.get_sample(seq_index, ids=frame_ids, aspect_ratio=aspect_ratio)
    logger.info(f"loaded {len(frame_ids)} frames in {time.time() - t_load:.1f}s")
    return sample


def gt_from_sample(sample: dict) -> dict:
    """Pull the ground-truth arrays inference scores against (eval semantics
    unchanged: GT depth + extrinsics; predicted intrinsics drive unprojection).

    ``modalities`` (the vendor's advertised GT set) rides along so the metrics
    stage can skip scores whose GT does not exist for this dataset — e.g. NYU
    ships no poses and DL3DV no depth; their placeholder arrays must not be
    scored as ground truth."""
    return {
        "gt_depth": sample["depths"]
        .numpy()
        .astype(np.float32),  # (S, H, W) m, 0=invalid
        "gt_extrinsics": sample["extrinsics"]
        .numpy()
        .astype(np.float32),  # (S, 3, 4) world->cam
        "modalities": list(sample.get("modalities", [])),
    }


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
    logger.info(f"inference done in {time.time() - t_infer:.1f}s")
    gpu_status("after forward")

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

    # --- 5. Evaluate against the dataset's advertised GT modalities --------------
    # Vendors declare which arrays are real ground truth (sample["modalities"]);
    # placeholder arrays (NYU's identity poses, DL3DV's zero depth) are never
    # scored. A metric whose GT is absent is reported as null in metrics.json.
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    modalities = set(loaded.get("modalities") or [])

    # 5a. Camera pose: ATE / RPE on camera-to-world trajectories (Sim3-aligned,
    # since VGGT poses are metric only up to a global scale). Needs EXTRINSICS
    # GT and >= 2 poses (RPE is over relative motions).
    camera_pose_metrics = None
    if "extrinsics" in modalities and num_f >= 2:
        gt_c2w = world_to_camera_to_camera_to_world(gt_extrinsics)
        pred_c2w = world_to_camera_to_camera_to_world(pred_extrinsics)
        camera_pose_metrics = CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run(
            vis_path=os.path.join(metrics_dir, "camera_pose")
        )

    # 5b. Mono depth: per-frame Abs Rel / delta (per-image median alignment),
    # aggregated to the headline means across frames. Needs DEPTH GT; frames
    # without a single valid GT pixel (possible with sparse LiDAR) are skipped
    # rather than crashing the metric.
    mono_depth_metrics = None
    if "depths" in modalities:
        per_frame_depth = []
        for i in tqdm(range(num_f), desc="mono-depth eval", unit="frame"):
            if not (gt_depth[i] > 0).any():
                continue
            res = MonoDepthMetric(gt_depth[i], pred_depth_2d[i], align="median").run()
            per_frame_depth.append(res)
        if per_frame_depth:
            mono_depth_metrics = {
                "abs_rel_mean": float(np.mean([d["abs_rel"]["mean"] for d in per_frame_depth])),
                "abs_rel_rmse": float(np.mean([d["abs_rel"]["rmse"] for d in per_frame_depth])),
                "delta1": float(np.mean([d["delta"]["delta1"] for d in per_frame_depth])),
                "delta2": float(np.mean([d["delta"]["delta2"] for d in per_frame_depth])),
                "delta3": float(np.mean([d["delta"]["delta3"] for d in per_frame_depth])),
                "num_frames": len(per_frame_depth),
            }

    all_metrics = {
        "scene": seq_name,
        "num_frames": int(num_f),
        "resolution": [int(height), int(width)],
        "camera_pose": camera_pose_metrics,
        "mono_depth": mono_depth_metrics,
    }
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
