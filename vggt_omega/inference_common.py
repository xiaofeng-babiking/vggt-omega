"""Shared inference helpers for the single-GPU and distributed entrypoints.

Holds the command-line flags common to both, the dataset/config pipeline, and
the geometry / point-cloud IO -- so inference.py and distributed_inference.py
import from here instead of from each other.
"""
import os
import time

import cv2
import gflags
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vggt_omega import datasets as vggt_datasets
from vggt_omega.datasets.dataloaders.composed_dataset import ComposedDataset
from vggt_omega.utils.logger import get_logger

# Per-dataset configure dir (ships tum.yaml, the default --configure target).
DATASET_CONFIG_DIR = os.path.join(os.path.dirname(vggt_datasets.__file__), "configures")

logger = get_logger("vggt_omega.inference_common")

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
gflags.DEFINE_integer(
    "loader_workers",
    -1,
    "Thread workers for parallel frame loading in get_sample: -1 = auto "
    "(min(32, cores) / local ranks), 0 or 1 = serial.",
)


def effective_long_side(native_long: int, image_scale: float) -> int:
    """Native long side scaled by `image_scale`, snapped to a /16 multiple (ViT-friendly)."""
    return max(16, int(round(native_long * image_scale / 16)) * 16)


# Canonical implementations moved to vggt_omega.utils.geometry (the COLMAP
# dataset path and the self-supervised trainer need them without dragging in
# this module's gflags/cv2 surface); re-exported here for existing callers.
from vggt_omega.utils.geometry import (  # noqa: E402,F401
    unproject_depth_map_to_point_map,
    world_to_camera_to_camera_to_world,
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
    (images ``(S,3,H,W)`` in ``[0,1]`` + the full GT modality set). Frames load
    on ``--loader_workers`` threads (-1 = auto, <=1 = serial)."""
    t_load = time.time()
    native_h, native_w = dataset.native_image_size(seq_index)
    aspect_ratio = min(native_h, native_w) / max(native_h, native_w)
    num_workers = None if FLAGS.loader_workers < 0 else FLAGS.loader_workers
    sample = dataset.get_sample(
        seq_index, ids=frame_ids, aspect_ratio=aspect_ratio, num_workers=num_workers
    )
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
