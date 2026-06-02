import os
import glob
import json
import random

import cv2
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "/jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt"
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
image_dir = "/jfs/guibiao/streamVGGT/data/eval/tum/rgbd_dataset_freiburg3_sitting_halfsphere/rgb"
image_names = sorted(glob.glob(os.path.join(image_dir, "*.png")))
image_names = random.sample(image_names, k=10)

# Where to write depth/conf images, camera json, and the fused point cloud.
output_dir = os.path.join("outputs", os.path.basename(image_dir.rstrip("/")))
# Drop the lowest `conf_percentile`% of points (by confidence) from the fused cloud.
conf_percentile = 20.0
# Cap the fused cloud size; points beyond this are randomly subsampled (0 = no cap).
max_points = 3_000_000


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Unproject per-frame depth into a common world frame -> (S, H, W, 3).

    `extrinsic` is world-to-camera (OpenCV) [R|t], so world = R^T @ (cam - t).
    """
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


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
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
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
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

# Pull everything to CPU/numpy (float() guards against bf16, which numpy can't hold).
depth = predictions["depth"].float().cpu().numpy()[0]  # (S, H, W, 1)
depth_conf = predictions["depth_conf"].float().cpu().numpy()[0]  # (S, H, W)
images_np = predictions["images"].float().cpu().numpy()[0]  # (S, 3, H, W), RGB in [0, 1]
extrinsics = extrinsics.float().cpu().numpy()[0]  # (S, 3, 4) world-to-camera (OpenCV)
intrinsics = intrinsics.float().cpu().numpy()[0]  # (S, 3, 3) pixels

num_frames, height, width = depth.shape[:3]
images_hwc = np.transpose(images_np, (0, 2, 3, 1))  # (S, H, W, 3)

depth_dir = os.path.join(output_dir, "depth")
conf_dir = os.path.join(output_dir, "conf")
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(conf_dir, exist_ok=True)

# --- Adaptive 16-bit scale factors (no clipping, full precision) -------------
depth_2d = depth[..., 0]
valid_depth = np.isfinite(depth_2d) & (depth_2d > 0)
depth_max = float(depth_2d[valid_depth].max()) if valid_depth.any() else 1.0
depth_scale = 65535.0 / depth_max if depth_max > 0 else 1.0

finite_conf = np.isfinite(depth_conf)
conf_max = float(depth_conf[finite_conf].max()) if finite_conf.any() else 1.0
conf_scale = 65535.0 / conf_max if conf_max > 0 else 1.0

# --- 1. Dump per-frame depth + confidence as 16-bit PNGs ---------------------
frames_meta = []
for i in range(num_frames):
    name = f"frame_{i:04d}.png"
    save_uint16_image(depth_2d[i], depth_scale, os.path.join(depth_dir, name))
    save_uint16_image(depth_conf[i], conf_scale, os.path.join(conf_dir, name))
    frames_meta.append(
        {
            "index": i,
            "image": os.path.basename(image_names[i]) if i < len(image_names) else None,
            "depth": os.path.join("depth", name),
            "conf": os.path.join("conf", name),
            "fx": float(intrinsics[i, 0, 0]),
            "fy": float(intrinsics[i, 1, 1]),
            "cx": float(intrinsics[i, 0, 2]),
            "cy": float(intrinsics[i, 1, 2]),
            "intrinsics": intrinsics[i].tolist(),
            "extrinsics": extrinsics[i].tolist(),
        }
    )

# --- 2. Dump camera intrinsics/extrinsics (+ scale factors) as JSON ----------
camera_meta = {
    "scene": os.path.basename(output_dir),
    "image_width": int(width),
    "image_height": int(height),
    "num_frames": int(num_frames),
    "depth_scale": depth_scale,  # depth = uint16_value / depth_scale
    "depth_max": depth_max,
    "conf_scale": conf_scale,  # conf  = uint16_value / conf_scale
    "conf_max": conf_max,
    "depth_unit": "uint16_value / depth_scale",
    "extrinsics_convention": "world_to_camera (OpenCV), 3x4 [R|t]",
    "intrinsics_convention": "pixels, 3x3 K",
    "frames": frames_meta,
}
with open(os.path.join(output_dir, "cameras.json"), "w") as f:
    json.dump(camera_meta, f, indent=2)

# --- 3. Fuse depth + RGB into a single world-frame PLY point cloud ------------
world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)  # (S,H,W,3)
points = world_points.reshape(-1, 3)
colors = (images_hwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
conf_flat = depth_conf.reshape(-1)
depth_flat = depth_2d.reshape(-1)

mask = np.isfinite(points).all(axis=1) & np.isfinite(conf_flat) & (depth_flat > 0)
if conf_percentile > 0 and mask.any():
    threshold = np.percentile(conf_flat[mask], conf_percentile)
    mask &= conf_flat >= threshold

points = points[mask]
colors = colors[mask]

if max_points and points.shape[0] > max_points:
    rng = np.random.default_rng(0)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    points = points[keep]
    colors = colors[keep]

ply_path = os.path.join(output_dir, "pointcloud.ply")
write_ply(ply_path, points, colors)

print(f"Wrote {num_frames} depth/conf frames + cameras.json to {output_dir}")
print(
    f"  depth_scale={depth_scale:.4f} (depth_max={depth_max:.4f}), "
    f"conf_scale={conf_scale:.4f} (conf_max={conf_max:.4f})"
)
print(f"  point cloud: {points.shape[0]} points -> {ply_path}")
