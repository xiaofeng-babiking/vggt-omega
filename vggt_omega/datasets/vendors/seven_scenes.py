"""Microsoft 7-Scenes vendor for the VGGT-Omega dataset API.

7-Scenes is an indoor RGB-D relocalization benchmark captured with a Kinect v1.
Each scene (chess, fire, heads, office, pumpkin, redkitchen, stairs) holds
several ``seq-NN`` directories of ~1000 sequential frames at 30 Hz, with four
files per frame::

    frame-XXXXXX.color.png       RGB, 640x480
    frame-XXXXXX.depth.png       raw Kinect depth in the *depth-sensor* frame
    frame-XXXXXX.depth.proj.png  depth registered/projected into the *color* frame
    frame-XXXXXX.pose.txt        4x4 camera-to-world matrix (OpenCV camera frame)

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is 16-bit PNG in **millimeters** (meters = value / 1000); 0 and 65535
  are the invalid sentinels and map to 0.
* We load ``.depth.proj.png`` (not the raw ``.depth.png``) because it is
  registered to the color camera, so depth aligns with the RGB pixels and with
  the color-camera intrinsics. (Raw depth differs from proj by a horizontal
  registration shift -- the RGB/IR baseline -- and is misaligned with color.)
* ``pose.txt`` is camera-to-world in the OpenCV optical frame (x-right, y-down,
  z-forward), so world->camera is its inverse with no axis flip. Cross-frame
  depth reprojection closes to ~2 mm, confirming the convention.
* Intrinsics are the de-facto-standard 7-Scenes pinhole: focal 585 px,
  principal point (320, 240) for the native 640x480 color image. Override via
  ``intrinsics=[fx, fy, cx, cy]`` if you have a per-rig calibration.

7-Scenes ships no per-frame timestamps; the frames are a 30 Hz video, so we
synthesize ``timestamp = frame_index / 30`` (faithful relative capture time, not
fabricated precision) and advertise TIMESTAMP for video ordering.

Pose files are read **lazily** (only for the frames actually sampled): the full
benchmark is ~43k frames and eagerly parsing every ``pose.txt`` at construction
would dominate startup. Construction only globs the color frames per sequence.
"""
from __future__ import annotations

import glob
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class SevenScenesDataset(BaseDataset):
    """Microsoft 7-Scenes as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # De-facto-standard 7-Scenes color-camera pinhole intrinsics (fx, fy, cx, cy)
    # for the native 640x480 frame, as used across the relocalization literature.
    _FOCAL = 585.0
    _PRINCIPAL_POINT = (320.0, 240.0)

    # 7-Scenes provides RGB + registered metric depth + GT camera poses. As with
    # the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth re-projected
    # through the GT poses (not an independent point-cloud GT), so they are NOT
    # advertised as evaluable GT modalities -- process_one_image still computes
    # them (e.g. for depth-supervised point heads), they just must not be scored
    # as a point cloud.
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.TIMESTAMP,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
        }
    )

    # Frame rate used to synthesize timestamps (Kinect v1 streams at ~30 Hz).
    _FPS = 30.0

    @staticmethod
    def read_seven_scenes_pose(path: str) -> np.ndarray:
        """Read a 7-Scenes ``pose.txt`` -> (4,4) float64 camera-to-world matrix.

        Uses a plain float split rather than ``np.loadtxt`` (which is ~100x slower
        per file and would dominate the per-frame sampling cost).
        """
        with open(path) as f:
            vals = f.read().split()
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size != 16:
            raise ValueError(f"7-Scenes pose {path!r}: expected 16 values, got {arr.size}")
        return arr.reshape(4, 4)

    @classmethod
    def seven_scenes_pose_to_w2c(cls, path: str) -> np.ndarray:
        """Read ``pose.txt`` (camera-to-world, OpenCV) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; no matrix inverse needed). Raises ValueError if the
        pose is non-finite (7-Scenes occasionally marks failed GT frames as inf).
        """
        c2w = cls.read_seven_scenes_pose(path)
        if not np.isfinite(c2w).all():
            raise ValueError(f"7-Scenes pose {path!r} is non-finite (failed GT frame)")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_seven_scenes_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
        """Read a 7-Scenes 16-bit depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 millimeter counts (meters = value /
        depth_scale). 0 and 65535 are the invalid sentinels and map to 0.
        """
        arr = np.asarray(Image.open(path)).astype(np.float32)
        invalid = (arr == 0) | (arr == 65535) | ~np.isfinite(arr)
        depth = arr / float(depth_scale)
        depth[invalid] = 0.0
        return depth

    @classmethod
    def seven_scenes_intrinsics(cls, override=None) -> np.ndarray:
        """(3,3) pinhole K for 7-Scenes' native 640x480 color frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the standard 7-Scenes
        focal=585, principal point=(320,240) is used.
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            fx = fy = cls._FOCAL
            cx, cy = cls._PRINCIPAL_POINT
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _split_seq_dirs(scene_dir: str, split: str) -> list[str]:
        """Sequence dir names for ``scene_dir`` under ``split``.

        ``split`` in {"train", "test"} reads the scene's ``TrainSplit.txt`` /
        ``TestSplit.txt`` (lines like ``sequence3`` -> ``seq-03``); any other
        value (e.g. "all") returns every ``seq-*`` directory.
        """
        if split in ("train", "test"):
            fname = "TrainSplit.txt" if split == "train" else "TestSplit.txt"
            split_path = os.path.join(scene_dir, fname)
            if os.path.isfile(split_path):
                names = []
                with open(split_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        num = int(line.lower().replace("sequence", ""))
                        names.append(f"seq-{num:02d}")
                return names
            logging.warning(
                "7-Scenes %s: no %s; falling back to all seq-* dirs",
                os.path.basename(scene_dir), fname,
            )
        return sorted(
            os.path.basename(d)
            for d in glob.glob(os.path.join(scene_dir, "seq-*"))
            if os.path.isdir(d)
        )

    def _list_frames(self, seq_dir: str) -> list:
        """List a sequence dir -> [(color_path, depth_path, pose_path, frame_num)],
        ordered by frame number and subsampled by ``self.frame_step``.

        Poses are NOT read here (lazy): only color frames are enumerated.
        """
        frames = []
        for color_path in glob.glob(os.path.join(seq_dir, "frame-*.color.png")):
            stem = os.path.basename(color_path).split(".")[0]  # "frame-000123"
            frame_num = int(stem.split("-")[1])
            depth_path = color_path.replace(".color.png", self._depth_suffix)
            pose_path = color_path.replace(".color.png", ".pose.txt")
            frames.append((color_path, depth_path, pose_path, frame_num))
        frames.sort(key=lambda fr: fr[3])
        if self.frame_step > 1:
            frames = frames[:: self.frame_step]
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "test",
        SEVEN_SCENES_DIR: str = None,
        scenes=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 1000.0,
        depth_variant: str = "proj",
        frame_step: int = 1,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if SEVEN_SCENES_DIR is None:
            raise ValueError("SEVEN_SCENES_DIR must be specified")
        if depth_variant not in ("proj", "raw"):
            raise ValueError(f"depth_variant must be 'proj' or 'raw', got {depth_variant!r}")
        if frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")

        self.SEVEN_SCENES_DIR = SEVEN_SCENES_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self._depth_suffix = ".depth.proj.png" if depth_variant == "proj" else ".depth.png"
        self.frame_step = frame_step
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Resolve scene directories (each must contain seq-* subdirs).
        patterns = scenes or ["*"]
        scene_dirs = sorted(
            {
                d
                for pat in patterns
                for d in glob.glob(os.path.join(SEVEN_SCENES_DIR, pat))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "seq-*"))
            }
        )

        self.data_store = {}
        for scene_dir in scene_dirs:
            scene = os.path.basename(scene_dir.rstrip("/"))
            for seq in self._split_seq_dirs(scene_dir, split):
                seq_dir = os.path.join(scene_dir, seq)
                if not os.path.isdir(seq_dir):
                    logging.warning("7-Scenes %s/%s: listed in split but missing; skipping", scene, seq)
                    continue
                frames = self._list_frames(seq_dir)
                if len(frames) < min_num_images:
                    logging.warning(
                        "7-Scenes %s/%s: only %d frames (< %d); skipping",
                        scene, seq, len(frames), min_num_images,
                    )
                    continue
                # key doubles as the inference output-dir name (scene-grouped)
                self.data_store[f"{scene}/{seq}"] = frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable 7-Scenes sequences under {SEVEN_SCENES_DIR} "
                f"(split={split!r}, scenes={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            color_path = self.data_store[name][0][0]
            with Image.open(color_path) as im:
                w, h = im.size  # PIL reports (W, H) without decoding pixels
            self._native_size_cache[name] = (h, w)
        return self._native_size_cache[name]

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            # len_train is a virtual epoch length usually >> #sequences; the sampler
            # emits indices in [0, len_train). That only works with inside_random=True
            # (which remapped seq_index above). Give a clear error, not a raw IndexError.
            if seq_index is None or not 0 <= seq_index < self.sequence_list_len:
                raise ValueError(
                    f"seq_index={seq_index} out of range [0, {self.sequence_list_len}); "
                    "when len_train > #sequences, set inside_random=True so the sampler "
                    "index is remapped into range."
                )
            seq_name = self.sequence_list[seq_index]
        frames = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        K = self.seven_scenes_intrinsics(self.intrinsics_override)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, original_sizes = [], [], []

        for i in ids:
            color_path, depth_path, pose_path, frame_num = frames[int(i)]
            image = read_image_cv2(color_path)
            if image is None:
                # Globbed from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"7-Scenes: could not read image {color_path}"
                )
            depth_map = self.read_seven_scenes_depth(depth_path, self.depth_scale)
            pose_w2c = self.seven_scenes_pose_to_w2c(pose_path)
            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri,
                intri,
                world_p,
                cam_p,
                pmask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                pose_w2c.copy(),
                K.copy(),
                original_size,
                target_image_shape,
                filepath=color_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            sky_masks.append(depth_map < 0)  # 7-Scenes is indoor: always all-False (sky convention = depth<0)
            timestamps.append(frame_num / self._FPS)
            original_sizes.append(original_size)

        return {
            "seq_name": "7scenes_" + seq_name,
            "ids": np.array(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "sky_masks": sky_masks,
            "timestamps": np.array(timestamps, dtype=np.float64),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
