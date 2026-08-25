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
gflags.DEFINE_string(
    "sequences",
    None,
    "Comma-separated scene names or globs that REPLACE every dataset_config's "
    "`sequences` in the configure (e.g. --sequences bicycle,garden). Unset = the "
    "configure's own list.",
)

# --- 3D gaussian splatting (ported from the sibling MUSA repo's inference.py) --
gflags.DEFINE_boolean(
    "enable_3dgs",
    False,
    "Build the gaussian head, decode a 3DGS scene from the predicted depth + cameras, "
    "rasterize the input views back out of it and score that self-render "
    "(PSNR/SSIM/LPIPS). Dumps gaussians.ply (INRIA/SuperSplat layout), "
    "render_compare/ (per-view ground truth | rendered | error sheet) and "
    "metrics/gaussians.json. REQUIRES a checkpoint trained with a gaussian head; "
    "--gs_sh_degree must match the one it was trained at.",
)
gflags.DEFINE_integer(
    "gs_sh_degree",
    4,
    "Spherical-harmonic degree of the gaussian head. MUST match the training config's "
    "model.gs_sh_degree or the checkpoint will not load (the head's channel count is "
    "(deg+1)^2 * 3).",
)
gflags.DEFINE_string(
    "gs_rasterize_mode",
    "classic",
    "gsplat rasterization mode for the self-render: 'classic' or 'antialiased'. Must "
    "match the mode the weights were trained with (selfsup.rasterize_mode).",
)
gflags.DEFINE_integer(
    "gs_render_chunk",
    0,
    "[--enable_3dgs] Views rasterized per gsplat call. 0 = auto, sized so the "
    "spherical-harmonic expansion (views x gaussians x coeffs x 3 floats) stays "
    "under a fixed byte budget. Output is bit-identical to a single call.",
)
gflags.DEFINE_boolean(
    "gs_dump_ply",
    True,
    "[--enable_3dgs] Write the fused scene to gaussians.ply. Off (--nogs_dump_ply) "
    "when only the scores are wanted: a 3.5M-splat sh3 scene is ~860 MB and took "
    "~9 min to write on a busy disk, several times the render + scoring itself.",
)
gflags.DEFINE_string(
    "gs_lpips_net",
    "vgg",
    "LPIPS trunk for the self-render score: 'vgg' (the loss trunk this repo trains "
    "against) or 'alex' (the trunk papers report).",
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
    The target long side comes from :func:`resolve_long_side`: an absolute
    ``inference.long_side`` when the configure states one, else the data's
    NATIVE long side x ``inference.image_scale`` -- read from the dataset itself
    rather than hardcoded.
    """
    apply_sequence_override(cfg)
    dataset = instantiate(cfg.dataset, common_config=cfg.common_config, _recursive_=False)
    native_h, native_w = dataset.native_image_size()
    dataset.set_img_size(resolve_long_side(cfg, max(native_h, native_w)))
    return dataset


def apply_sequence_override(cfg) -> None:
    """Rewrite every ``dataset_config``'s ``sequences`` from ``--sequences``.

    Mutates ``cfg`` in place before instantiation, because the scene list is
    consumed by ``SequenceDataset.__init__`` and cannot be changed afterwards.
    A no-op when the flag is unset. Raises if the configure declares no
    ``sequences`` anywhere, rather than silently running the configure's own
    scenes under a flag the caller believed had taken effect.
    """
    if FLAGS.sequences is None:
        return
    names = [s.strip() for s in FLAGS.sequences.split(",") if s.strip()]
    if not names:
        raise SystemExit("--sequences was empty; pass names or globs, e.g. --sequences bicycle")
    applied = 0
    for entry in OmegaConf.select(cfg, "dataset.dataset_configs", default=[]) or []:
        if "sequences" in entry:
            entry.sequences = names
            applied += 1
    if not applied:
        raise SystemExit(
            f"--sequences {FLAGS.sequences!r} was given but {FLAGS.configure} declares no "
            "dataset_configs[*].sequences to override."
        )
    logger.info(f"--sequences override: {names} (applied to {applied} dataset_config(s))")


def resolve_long_side(cfg, native_long: int) -> int:
    """Target long side: ``inference.long_side`` if stated, else ``image_scale``.

    ``image_scale`` is RELATIVE to whatever the image directory happens to be,
    which is the wrong knob once a model has a trained resolution: mip-NeRF-360's
    images_4 is 1297 px wide for garden but 1237 for bicycle, so the 0.4 that
    lands exactly on 512 for garden gives 496 for bicycle -- silently off
    distribution, and not comparable BETWEEN scenes. ``inference.long_side``
    states the target directly and holds across scenes.
    """
    long_side = OmegaConf.select(cfg, "inference.long_side", default=None)
    if long_side:
        return max(16, int(round(int(long_side) / 16)) * 16)
    return effective_long_side(native_long, cfg.inference.image_scale)


def resolve_frame_ids(dataset: ComposedDataset, seq_index: int, inference_cfg) -> np.ndarray:
    """Ordered frame ids for one sequence, per the configure's ``inference`` block.

    Delegates to :meth:`ComposedDataset.resolve_frame_ids` -- ``num_frames``
    (<=0 = all), ``frame_stride``, ``frame_sampling`` (linspace / random /
    window) and ``seed`` -- so the single-GPU and distributed entrypoints
    cannot drift apart on which frames they score.
    """
    return dataset.resolve_frame_ids(seq_index, inference_cfg)


def resolve_checkpoint(checkpoint: str) -> str:
    """Accept a trainer sidecar in place of the weights file.

    ``trainer_step*.pt`` carries optimizer + scheduler + RNG state and no weights
    at all, so loading it as a state_dict fails with an opaque key error. The
    trainer writes the pair together and ``Trainer.resume`` recovers the weights
    by exactly this substitution.
    """
    directory, name = os.path.split(checkpoint)
    if not name.startswith("trainer_step"):
        return checkpoint
    weights = os.path.join(directory, name.replace("trainer_step", "model_step", 1))
    if not os.path.exists(weights):
        raise SystemExit(
            f"{checkpoint} is the trainer sidecar (optimizer + scheduler + RNG state, "
            f"no model weights); the weights belong in {weights}, which is missing."
        )
    logger.info(f"{name} is the trainer sidecar; loading weights from {os.path.basename(weights)}")
    return weights


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
