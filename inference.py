"""Run VGGT-Omega on a TUM RGB-D sequence and evaluate all metrics.

Drives inference from :class:`~vggt_omega.datasets.vendors.tum.TumDataset`, which
yields RGB frames together with ground-truth depth / camera poses / intrinsics,
all processed to the same resolution the model sees. We:

  1. predict **camera poses**, **monocular depth**, and a fused **point cloud**;
  2. dump them (depth/conf PNGs, ``cameras.json``, ``pointcloud.ply``);
  3. evaluate every metric family in :mod:`vggt_omega.evaluates` against the TUM
     ground truth -- camera pose (ATE / RPE), mono depth (Abs Rel / delta), and
     point cloud (accuracy / completeness / chamfer / normal-consistency / F-score).

Conventions (see ``vggt_omega/datasets``): extrinsics are world-to-camera OpenCV
``[R|t]``; depth is metres with ``0`` = invalid. VGGT predicts geometry only up to
a global scale/pose, so depth uses a per-image median scale, poses a Umeyama
``Sim3`` alignment, and the cloud an ICP+scale registration before scoring.
"""

import json
import os

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.datasets.vendors.tum import TumDataset
from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric, PointcloudMetric

# --- configuration -----------------------------------------------------------
checkpoint_path = "/jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt"
TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
sequence = "rgbd_dataset_freiburg3_sitting_halfsphere"
num_frames = 24
# TUM is 640x480 -> aspect (H/W) 0.75; this maps to a 512x384 (WxH) model input.
aspect_ratio = 0.75

output_dir = os.path.join("outputs", sequence)
# Drop the lowest `conf_percentile`% of points (by confidence) from the fused cloud.
conf_percentile = 20.0
# Cap cloud size for export and for the (KD-tree / ICP) point-cloud metric.
max_points = 3_000_000
metric_max_points = 200_000
# Inlier distance (metres, in GT scale after ICP) for the point-cloud F-score.
fscore_threshold = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"


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


# --- 1. Load a TUM sequence (RGB + GT depth/poses/intrinsics) -----------------
common_conf = OmegaConf.create(
    {
        "img_size": 512,
        "patch_size": 16,
        "training": False,       # no scale augmentation at eval
        "inside_random": False,  # honor the explicit seq_index / ids below
        "allow_duplicate_img": False,
        "get_nearby": False,     # use our ordered ids verbatim (not a random window)
        "rescale": True,
        "rescale_aug": False,
        "landscape_check": False,
        "augs": {"scales": None},
    }
)
dataset = TumDataset(
    common_conf=common_conf,
    split="train",
    TUM_DIR=TUM_DIR,
    sequences=[sequence],
    len_train=num_frames,
)
# Evenly-spaced, ordered frames across the sequence -> a real trajectory with baseline.
num_available = len(dataset.data_store[dataset.sequence_list[0]])
frame_ids = np.linspace(0, num_available - 1, num_frames).round().astype(int)
batch = dataset.get_data(seq_index=0, ids=frame_ids, aspect_ratio=aspect_ratio)

images_hwc = np.stack(batch["images"])                      # (S, H, W, 3) uint8, RGB
gt_depth = np.stack(batch["depths"]).astype(np.float32)     # (S, H, W) metres, 0=invalid
gt_extrinsics = np.stack(batch["extrinsics"]).astype(np.float32)  # (S, 3, 4) world->cam
gt_world_points = np.stack(batch["world_points"]).astype(np.float32)  # (S, H, W, 3)
gt_point_masks = np.stack(batch["point_masks"]).astype(bool)      # (S, H, W)

# --- 2. Inference on those exact frames --------------------------------------
model = VGGTOmega().to(device).eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = (
    torch.from_numpy(images_hwc.astype(np.float32))
    .permute(0, 3, 1, 2)
    .div(255.0)
    .contiguous()
    .to(device)
)
with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"], predictions["images"].shape[-2:]
)

# Pull to CPU/numpy (float() guards against bf16, which numpy can't hold).
pred_depth = predictions["depth"].float().cpu().numpy()[0]          # (S, H, W, 1)
pred_conf = predictions["depth_conf"].float().cpu().numpy()[0]      # (S, H, W)
images_np = predictions["images"].float().cpu().numpy()[0]          # (S, 3, H, W) [0,1]
pred_extrinsics = extrinsics.float().cpu().numpy()[0]               # (S, 3, 4) world->cam
pred_intrinsics = intrinsics.float().cpu().numpy()[0]               # (S, 3, 3) pixels

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
for i in range(num_f):
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
    "scene": sequence,
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
for i in range(num_f):
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

# 5c. Point cloud: GT cloud from TUM world points; predicted cloud from predicted
# depth+poses. ICP+scale registers the prediction before scoring.
gt_cloud = gt_world_points[gt_point_masks].reshape(-1, 3)
pred_cloud = pred_world_points.reshape(-1, 3)
pred_cloud = pred_cloud[np.isfinite(pred_cloud).all(axis=1) & (depth_flat > 0)]
pointcloud_metrics = PointcloudMetric(
    gt_cloud,
    pred_cloud,
    align="icp",
    align_scale=True,
    threshold=fscore_threshold,
    max_points=metric_max_points,
    seed=0,
).run(vis_path=os.path.join(metrics_dir, "pointcloud"))

all_metrics = {
    "scene": sequence,
    "num_frames": int(num_f),
    "resolution": [int(height), int(width)],
    "camera_pose": camera_pose_metrics,
    "mono_depth": mono_depth_metrics,
    "pointcloud": pointcloud_metrics,
}
with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

# --- 6. Report ---------------------------------------------------------------
print(f"[{sequence}] {num_f} frames @ {height}x{width} -> {output_dir}")
print(f"  point cloud: {points.shape[0]} points -> {ply_path}")
print("  camera pose (Sim3-aligned):")
print(f"    ATE  rmse = {camera_pose_metrics['ate']['rmse']:.4f} m")
print(f"    RPE  trans rmse = {camera_pose_metrics['rpe_trans']['rmse']:.4f} m")
print(f"    RPE  rot   rmse = {camera_pose_metrics['rpe_rot']['rmse']:.4f} deg")
print("  mono depth (median-aligned, mean over frames):")
print(f"    Abs Rel = {mono_depth_metrics['abs_rel_mean']:.4f}")
print(f"    delta1  = {mono_depth_metrics['delta1']:.4f}")
print("  point cloud (ICP+scale aligned):")
print(f"    chamfer mean = {pointcloud_metrics['chamfer']['mean']:.4f} m")
print(f"    accuracy mean = {pointcloud_metrics['accuracy']['mean']:.4f} m")
print(f"    completeness mean = {pointcloud_metrics['completeness']['mean']:.4f} m")
print(f"    F-score@{fscore_threshold} = {pointcloud_metrics['fscore']['fscore']:.4f}")
print(f"  full metrics -> {os.path.join(metrics_dir, 'metrics.json')}")
