"""TUM RGB-D vendor for the VGGT-Omega dataset API.

TUM sequences provide RGB + depth (16-bit PNG, meters = pixel/5000) + a
groundtruth trajectory (timestamp tx ty tz qx qy qz qw, camera-to-world in the
color optical / OpenCV frame). RGB, depth and groundtruth are on different
clocks, so frames are associated by nearest timestamp. Intrinsics are per-camera
constants (freiburg1/2/3).
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
from vggt_omega.datasets.vendors.common import (
    associate,
    quat_to_rotation,
    read_file_list,
)


class TumDataset(BaseDataset):
    """TUM RGB-D as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Official TUM per-camera pinhole intrinsics (fx, fy, cx, cy).
    _TUM_INTRINSICS = {
        "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
        "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764),
        "freiburg3": (535.4, 539.2, 320.1, 247.6),
    }

    @staticmethod
    def tum_pose_to_w2c(t, q) -> np.ndarray:
        """TUM (translation t, quaternion q) camera-to-world -> world-to-camera (3,4) OpenCV."""
        t = np.asarray(t, dtype=np.float64).reshape(3)
        rot_c2w = quat_to_rotation(q)              # camera-to-world rotation
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ t
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_tum_depth(path: str, depth_scale: float = 5000.0) -> np.ndarray:
        """Read a TUM 16-bit depth PNG -> float32 (H,W) meters. 0 stays 0 (invalid).

        NOTE: TUM stores plain uint16 integer counts (meters = value / depth_scale).
        This is NOT the float16-bit encoding the vendored dataset_util.read_depth
        assumes, so TUM uses this dedicated reader.
        """
        arr = np.asarray(Image.open(path)).astype(np.float32)
        depth = arr / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @classmethod
    def tum_intrinsics(cls, seq_name: str, override=None) -> np.ndarray:
        """(3,3) pinhole K for a TUM sequence. `override`=[fx,fy,cx,cy] wins; else the
        freiburg1/2/3 table is matched by substring. Raises ValueError if unknown."""
        if override is not None:
            fx, fy, cx, cy = override
        else:
            cam = next((k for k in cls._TUM_INTRINSICS if k in seq_name), None)
            if cam is None:
                raise ValueError(
                    f"no TUM intrinsics for sequence {seq_name!r}; pass intrinsics=[fx,fy,cx,cy]"
                )
            fx, fy, cx, cy = cls._TUM_INTRINSICS[cam]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @classmethod
    def associate_tum_sequence(cls, seq_dir: str, max_diff: float = 0.02):
        """Associate rgb/depth/groundtruth in a TUM sequence dir.

        Returns a list of (rgb_path, depth_path, w2c (3,4) float32, timestamp float).
        """
        rgb = read_file_list(os.path.join(seq_dir, "rgb.txt"))
        depth = read_file_list(os.path.join(seq_dir, "depth.txt"))
        gt = read_file_list(os.path.join(seq_dir, "groundtruth.txt"))
        gt_ts = sorted(gt)
        frames = []
        for t_rgb, t_dep in associate(list(rgb), list(depth), max_diff):
            t_gt = min(gt_ts, key=lambda g: abs(g - t_rgb))
            if abs(t_gt - t_rgb) > max_diff:
                continue
            tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt[t_gt])
            w2c = cls.tum_pose_to_w2c(np.array([tx, ty, tz]), (qx, qy, qz, qw))
            frames.append(
                (
                    os.path.join(seq_dir, rgb[t_rgb][0]),
                    os.path.join(seq_dir, depth[t_dep][0]),
                    w2c,
                    t_rgb,
                )
            )
        return frames

    # TUM provides RGB + raw metric depth + GT poses/intrinsics. WORLD_POINTS /
    # CAM_POINTS are only the depth re-projected through the (GT) poses, not an
    # independent point-cloud GT (e.g. a laser scan), so they are NOT advertised
    # as evaluable GT modalities. (process_one_image still computes them, e.g. for
    # depth-supervised point heads, but they must not be scored as a point cloud.)
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

    def __init__(
        self,
        common_conf,
        split: str = "train",
        TUM_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        assoc_max_diff: float = 0.02,
        depth_scale: float = 5000.0,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if TUM_DIR is None:
            raise ValueError("TUM_DIR must be specified")
        self.TUM_DIR = TUM_DIR
        self.expand_ratio = expand_ratio
        self.assoc_max_diff = assoc_max_diff
        self.depth_scale = depth_scale
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        patterns = sequences or ["*"]
        seq_dirs = sorted(
            {
                d
                for pat in patterns
                for d in glob.glob(os.path.join(TUM_DIR, pat))
                if os.path.isdir(d)
            }
        )
        self.data_store = {}
        for sd in seq_dirs:
            name = os.path.basename(sd.rstrip("/"))
            frames = self.associate_tum_sequence(sd, assoc_max_diff)
            if len(frames) < min_num_images:
                logging.warning(
                    "TUM seq %s: only %d aligned frames (< %d); skipping",
                    name, len(frames), min_num_images,
                )
                continue
            self.data_store[name] = frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(f"No usable TUM sequences under {TUM_DIR}")
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of associated frames in the sequence at ``local_idx`` of this
        vendor's ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self.data_store[name][0][0]
            with Image.open(rgb_path) as im:
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
        K = self.tum_intrinsics(seq_name, self.intrinsics_override)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, original_sizes = [], [], []

        for i in ids:
            rgb_path, depth_path, pose_w2c, ts = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Associated from rgb.txt, so the file should exist; fail loudly
                # (a silent skip would yield fewer than img_per_seq frames and
                # break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"TUM: could not read image {rgb_path} (listed in rgb.txt but unreadable)"
                )
            depth_map = self.read_tum_depth(depth_path, self.depth_scale)
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
                filepath=rgb_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            sky_masks.append(depth_map < 0)  # TUM is indoor: always all-False (sky convention = depth<0)
            timestamps.append(ts)
            original_sizes.append(original_size)

        return {
            "seq_name": "tum_" + seq_name,
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
