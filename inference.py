"""Run VGGT-Omega on TUM RGB-D sequences and evaluate camera-pose + mono-depth.

Drives inference from the *training* :class:`~vggt_omega.datasets.composed_dataset.ComposedDataset`
(instantiated from the dataset Hydra config, eval knobs overridden), so each frame
is tensorized through the exact same contract the model is trained on. Per sequence we:

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

Usage (all arguments are gflags; defaults run ALL sequences, ALL frames, native
resolution), single GPU::

    python inference.py
    python inference.py --image_scale=0.5 --sequences=rgbd_dataset_freiburg3_sitting_halfsphere
    python inference.py --num_frames=200 --tum_dir=/path/to/tum

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

# Training dataset Hydra config (anchors common_conf so eval geometry/tensorization
# stay identical to training; only the eval knobs below are overridden).
DATASET_CONFIG_DIR = os.path.join(os.path.dirname(vggt_datasets.__file__), "config")

# --- command-line flags ------------------------------------------------------
FLAGS = gflags.FLAGS
gflags.DEFINE_string(
    "checkpoint", "/jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt",
    "Path to the VGGT-Omega checkpoint (.pt).")
gflags.DEFINE_string(
    "tum_dir", "/jfs/guibiao/streamVGGT/data/eval/tum",
    "Root directory holding the TUM RGB-D sequence folders.")
gflags.DEFINE_list(
    "sequences", [],
    "TUM sequences to run, as names or glob patterns (e.g. 'rgbd_dataset_freiburg3_*'). "
    "Empty (default) = ALL sequences under --tum_dir.")
gflags.DEFINE_integer(
    "num_frames", 720,
    "Frames per sequence. 0 (default) = ALL frames; otherwise this many, evenly spaced and ordered.")
gflags.DEFINE_float(
    "image_scale", 1.0,
    "Resolution scale factor. 1.0 (default) = native long side (--img_size); the effective long "
    "side is round(img_size * image_scale) snapped to a multiple of 16.")
gflags.DEFINE_integer(
    "img_size", 640,
    "Base long-side resolution (pixels) before --image_scale. TUM native VGA = 640.")
gflags.DEFINE_float(
    "aspect_ratio", 0.75,
    "Image aspect ratio H/W used to derive the target shape (TUM 640x480 -> 0.75).")
gflags.DEFINE_string(
    "output_root", "outputs",
    "Output root directory; a per-sequence subdirectory is created under it.")
gflags.DEFINE_float(
    "conf_percentile", 20.0,
    "Drop the lowest this-percent of points (by confidence) from the fused cloud.")
gflags.DEFINE_integer(
    "max_points", 5_000_000,
    "Cap on exported point-cloud size (the fused cloud is a visualization, not scored).")

device = "cuda" if torch.cuda.is_available() else "cpu"


def effective_img_size() -> int:
    """Long-side resolution after applying --image_scale, snapped to a /16 multiple."""
    raw = FLAGS.img_size * FLAGS.image_scale
    return max(16, int(round(raw / 16)) * 16)


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

    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
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
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


# --- 1. Load TUM sequences (RGB + GT depth/poses/intrinsics) ------------------
def build_dataset() -> ComposedDataset:
    """Instantiate the *training* ComposedDataset from the dataset Hydra config,
    overriding only the eval knobs.

    Everything else (patch_size, rescale, landscape_check, load_track, the TUM
    vendor settings) is inherited from the training config, so inference cannot
    silently drift from training. The overrides reproduce deterministic eval
    geometry: native resolution, no scale/colour augmentation, explicit ordered
    ids honored verbatim (no random remap / nearby-window sampling).
    """
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(DATASET_CONFIG_DIR, "default_dataset.yaml")),
        OmegaConf.load(os.path.join(DATASET_CONFIG_DIR, "default.yaml")),
    )
    common = cfg.data.train.common_config
    common.img_size = effective_img_size()  # native eval resolution (was 512 in training)
    common.training = False                  # no scale/colour augmentation
    common.inside_random = False             # honor explicit seq_index / ids
    common.rescale_aug = False               # deterministic resize
    common.get_nearby = False                # use our ordered ids verbatim
    common.allow_duplicate_img = False
    common.augs.scales = None

    dataset_cfg = cfg.data.train.dataset
    vendor_cfg = dataset_cfg.dataset_configs[0]
    vendor_cfg.TUM_DIR = FLAGS.tum_dir
    vendor_cfg.sequences = list(FLAGS.sequences) if FLAGS.sequences else ["*"]

    return instantiate(dataset_cfg, common_config=common, _recursive_=False)


def resolve_frame_ids(dataset: ComposedDataset, seq_index: int) -> np.ndarray:
    """Ordered frame ids for one sequence: all frames (--num_frames<=0) or evenly spaced."""
    num_available = dataset.sequence_num_frames(seq_index)
    nf = FLAGS.num_frames
    if nf <= 0 or nf >= num_available:
        return np.arange(num_available)                                  # ALL frames, ordered
    return np.linspace(0, num_available - 1, nf).round().astype(int)


def load_sample(dataset: ComposedDataset, seq_index: int, frame_ids) -> dict:
    """Training-identical tensorized sample for ``frame_ids`` of one sequence
    (images ``(S,3,H,W)`` in ``[0,1]`` + the full GT modality set)."""
    t_load = time.time()
    sample = dataset.get_sample(seq_index, ids=frame_ids, aspect_ratio=FLAGS.aspect_ratio)
    logger.info(f"loaded {len(frame_ids)} frames in {time.time() - t_load:.1f}s")
    return sample


def gt_from_sample(sample: dict) -> dict:
    """Pull the ground-truth arrays inference scores against (eval semantics
    unchanged: GT depth + extrinsics; predicted intrinsics drive unprojection)."""
    return {
        "gt_depth": sample["depths"].numpy().astype(np.float32),            # (S, H, W) m, 0=invalid
        "gt_extrinsics": sample["extrinsics"].numpy().astype(np.float32),   # (S, 3, 4) world->cam
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
    logger.info(f"running inference on {images.shape[0]} frames @ {images.shape[-2]}x{images.shape[-1]} (HxW) ...")
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
        "pred_depth": predictions["depth"].float().cpu().numpy()[0],        # (S, H, W, 1)
        "pred_conf": predictions["depth_conf"].float().cpu().numpy()[0],    # (S, H, W)
        "images_pred": predictions["images"].float().cpu().numpy()[0],      # (S, 3, H, W) [0,1]
        "pred_extrinsics": extrinsics.float().cpu().numpy()[0],             # (S, 3, 4) world->cam
        "pred_intrinsics": intrinsics.float().cpu().numpy()[0],             # (S, 3, 3) pixels
    }


def dump_and_eval(seq_name: str, output_dir: str, frame_ids, loaded: dict, pred: dict) -> None:
    """Dump predictions, fuse the PLY, evaluate metrics, and report for one sequence."""
    pred_depth = pred["pred_depth"]
    pred_conf = pred["pred_conf"]
    images_np = pred["images_pred"]
    pred_extrinsics = pred["pred_extrinsics"]
    pred_intrinsics = pred["pred_intrinsics"]
    gt_depth = loaded["gt_depth"]
    gt_extrinsics = loaded["gt_extrinsics"]

    num_f, height, width = pred_depth.shape[:3]
    pred_depth_2d = pred_depth[..., 0]                                  # (S, H, W)
    images_hwc_pred = np.transpose(images_np, (0, 2, 3, 1))            # (S, H, W, 3)

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
    if FLAGS.conf_percentile > 0 and mask.any():
        threshold = np.percentile(conf_flat[mask], FLAGS.conf_percentile)
        mask &= conf_flat >= threshold
    points, colors = points[mask], colors[mask]

    if FLAGS.max_points and points.shape[0] > FLAGS.max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=FLAGS.max_points, replace=False)
        points, colors = points[keep], colors[keep]

    ply_path = os.path.join(output_dir, "pointcloud.ply")
    write_ply(ply_path, points, colors)

    # --- 5. Evaluate against TUM ground truth ------------------------------------
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # 5a. Camera pose: ATE / RPE on camera-to-world trajectories (Sim3-aligned,
    # since VGGT poses are metric only up to a global scale).
    gt_c2w = world_to_camera_to_camera_to_world(gt_extrinsics)
    pred_c2w = world_to_camera_to_camera_to_world(pred_extrinsics)
    camera_pose_metrics = CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run(
        vis_path=os.path.join(metrics_dir, "camera_pose")
    )

    # 5b. Mono depth: per-frame Abs Rel / delta (per-image median alignment),
    # aggregated to the headline means across frames.
    per_frame_depth = []
    for i in tqdm(range(num_f), desc="mono-depth eval", unit="frame"):
        res = MonoDepthMetric(gt_depth[i], pred_depth_2d[i], align="median").run()
        per_frame_depth.append(res)
    mono_depth_metrics = {
        "abs_rel_mean": float(np.mean([d["abs_rel"]["mean"] for d in per_frame_depth])),
        "abs_rel_rmse": float(np.mean([d["abs_rel"]["rmse"] for d in per_frame_depth])),
        "delta1": float(np.mean([d["delta"]["delta1"] for d in per_frame_depth])),
        "delta2": float(np.mean([d["delta"]["delta2"] for d in per_frame_depth])),
        "delta3": float(np.mean([d["delta"]["delta3"] for d in per_frame_depth])),
        "num_frames": int(num_f),
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
    logger.info(
        f"[{seq_name}] {num_f} frames @ {height}x{width} -> {output_dir}\n"
        f"  point cloud: {points.shape[0]} points -> {ply_path}\n"
        f"  camera pose (Sim3-aligned):\n"
        f"    ATE  rmse = {camera_pose_metrics['ate']['rmse']:.4f} m\n"
        f"    RPE  trans rmse = {camera_pose_metrics['rpe_trans']['rmse']:.4f} m\n"
        f"    RPE  rot   rmse = {camera_pose_metrics['rpe_rot']['rmse']:.4f} deg\n"
        f"  mono depth (median-aligned, mean over frames):\n"
        f"    Abs Rel = {mono_depth_metrics['abs_rel_mean']:.4f}\n"
        f"    delta1  = {mono_depth_metrics['delta1']:.4f}\n"
        f"  full metrics -> {os.path.join(metrics_dir, 'metrics.json')}"
    )


def main():
    dataset = build_dataset()
    model = build_model()

    num_seqs = dataset.num_sequences()
    logger.info(
        f"{num_seqs} sequence(s) @ {effective_img_size()} long side "
        f"(img_size {FLAGS.img_size} x scale {FLAGS.image_scale}), "
        f"num_frames={'all' if FLAGS.num_frames <= 0 else FLAGS.num_frames}"
    )

    for seq_index in range(num_seqs):
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = resolve_frame_ids(dataset, seq_index)
        logger.info(f"[{seq_name}] ({seq_index + 1}/{num_seqs}) {len(frame_ids)} frames")

        sample = load_sample(dataset, seq_index, frame_ids)
        pred = run_inference(model, sample["images"])

        output_dir = os.path.join(FLAGS.output_root, seq_name)
        dump_and_eval(seq_name, output_dir, frame_ids, gt_from_sample(sample), pred)


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)  # parse command-line flags
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUse --help for the full flag list.")
    main()
