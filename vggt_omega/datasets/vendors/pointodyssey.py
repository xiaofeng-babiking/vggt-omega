"""PointOdyssey vendor for the VGGT-Omega dataset API.

PointOdyssey is a large synthetic video dataset of heavily dynamic scenes
(animated characters/animals) rendered at 30 fps. The copy this vendor reads is
a *preprocessed (CUT3R-style)* export -- NOT the official release layout::

    POINTODYSSEY_DIR/{train,val,test}/{seq}/
        rgb/%05d.jpg       RGB, 960x540 uint8 JPEG
        depth/%05d.npy     float32 (540,960) depth, ALREADY METERS, 0 = invalid
        cam/%05d.npz       keys 'pose' (4,4) c2w float32, 'intrinsics' (3,3) float32

Conventions used here (validated empirically against this dataset, not assumed):

* ``cam/%05d.npz['pose']`` is **camera-to-world in the OpenCV optical frame**
  (x-right, y-down, z-forward); world->camera is its rigid inverse with no axis
  flip. Cross-frame depth reprojection closes to 0.1-0.6% median relative error
  with this convention (vs ~2-19% if the pose were treated as w2c), confirming
  it on train/val/test sequences with real camera motion.
* Depth is float32 npy **already in meters** (indoor medians ~3.4 m, sub-1%
  cross-frame reprojection error confirms metric scale). This copy is NOT the
  official 16-bit PNG /1000 encoding -- no scaling is applied.
* ``depth == 0`` encodes BOTH invalid pixels and sky/background (zeros cluster
  at the image top in outdoor-ish synthetic scenes, up to ~18% of pixels). The
  two cannot be separated, so sky is NOT remapped to negative depth and
  SKY_MASK is neither advertised nor is a ``sky_masks`` key emitted (scenes DO
  contain sky, so an all-False mask would be wrong GT -- same rule as MegaDepth).
* Intrinsics live in each frame's npz, are constant within a sequence but vary
  across sequences (fx=fy in {576, 666.5, 800}; principal point always
  (480, 270) for the native 960x540 frame). Override via
  ``intrinsics=[fx, fy, cx, cy]`` only if you must force a single calibration.
* The original anno.npz with point trajectories/masks is absent from this copy,
  so TRACK is not offered. WORLD_POINTS / CAM_POINTS are only the depth
  re-projected through the GT poses (not an independent point-cloud GT), so as
  with the TUM/7-Scenes vendors they are NOT advertised as evaluable GT.

No per-frame timestamps are stored; the frames are a 30 fps video keyed by a
contiguous 5-digit frame index, so we synthesize ``timestamp = frame_index /
30`` (faithful relative capture time, not fabricated precision) and advertise
TIMESTAMP for video ordering -- the same policy as the 7-Scenes vendor.

Sequences are long (~850 to ~4300 frames) and live on a network filesystem, so
construction only lists the split directory for sequence NAMES; per-sequence
frame enumeration (one ``rgb/`` listing) happens lazily on first access and is
cached. The ``min_num_images`` floor is therefore also enforced lazily: a
too-short sequence raises a clear ValueError when first touched (every sequence
in this copy has >= ~850 frames, so this never fires on intact data).
"""
from __future__ import annotations

import fnmatch
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class PointOdysseyDataset(BaseDataset):
    """PointOdyssey as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # PointOdyssey provides RGB + metric depth + GT per-frame poses/intrinsics.
    # As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth
    # re-projected through the GT poses (not an independent point-cloud GT), so
    # they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just
    # must not be scored as a point cloud. SKY_MASK is not advertised because
    # this copy conflates sky and invalid pixels at depth==0 (see module doc);
    # no "sky_masks" key is emitted either -- an all-False mask for scenes
    # that contain sky would be wrong GT.
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.TIMESTAMP,
            Modality.POINT_MASK,
        }
    )

    # Frame rate used to synthesize timestamps (PointOdyssey is rendered at 30 fps).
    _FPS = 30.0

    _SPLITS = ("train", "val", "test")

    @staticmethod
    def pointodyssey_pose_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV frame) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; no matrix inverse needed). Raises ValueError on a
        wrong shape or a non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"PointOdyssey pose: expected (4,4) c2w, got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("PointOdyssey pose is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_pointodyssey_depth(path: str) -> np.ndarray:
        """Read a PointOdyssey ``depth/%05d.npy`` -> float32 (H,W) meters.

        The npy is already float32 meters (this copy is preconverted; NOT the
        official 16-bit PNG /1000 format). 0 stays 0 (invalid-or-sky); any
        non-finite or negative value (none observed in this copy) is mapped to
        0 so the DEPTH convention (0=invalid, <0=sky) is never violated by junk.
        """
        depth = np.load(path)
        if depth.ndim != 2:
            raise ValueError(
                f"PointOdyssey depth {path!r}: expected (H,W) array, got shape {depth.shape}"
            )
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    @staticmethod
    def read_pointodyssey_cam(path: str):
        """Read a PointOdyssey ``cam/%05d.npz`` -> (c2w (4,4) float64, K (3,3) float64).

        Raises ValueError if the archive is missing the 'pose'/'intrinsics' keys
        or their shapes are wrong.
        """
        with np.load(path) as cam:
            missing = {k for k in ("pose", "intrinsics") if k not in cam}
            if missing:
                raise ValueError(
                    f"PointOdyssey cam {path!r}: missing keys {sorted(missing)}"
                )
            c2w = np.asarray(cam["pose"], dtype=np.float64)
            K = np.asarray(cam["intrinsics"], dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"PointOdyssey cam {path!r}: pose shape {c2w.shape} != (4,4)")
        if K.shape != (3, 3):
            raise ValueError(f"PointOdyssey cam {path!r}: intrinsics shape {K.shape} != (3,3)")
        return c2w, K

    @classmethod
    def pointodyssey_intrinsics(cls, K_raw=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K in pixels of the native 960x540 frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame ``K_raw``
        from the cam npz is validated and used. Raises ValueError when neither
        is provided or ``K_raw`` is malformed.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K_raw is None:
            raise ValueError(
                "PointOdyssey intrinsics: need the per-frame K from the cam npz "
                "or an override=[fx, fy, cx, cy]"
            )
        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(
                f"PointOdyssey intrinsics: expected finite (3,3), got shape {K.shape}"
            )
        return K

    def __init__(
        self,
        common_conf,
        split: str = "train",
        POINTODYSSEY_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if POINTODYSSEY_DIR is None:
            raise ValueError("POINTODYSSEY_DIR must be specified")
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {self._SPLITS}, got {split!r}")

        self.POINTODYSSEY_DIR = POINTODYSSEY_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        split_dir = os.path.join(POINTODYSSEY_DIR, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"PointOdyssey split dir not found: {split_dir}")

        # Cheap construction: ONE directory listing for sequence names; frame
        # enumeration is deferred to _load_sequence (lazy, cached per sequence).
        patterns = sequences or ["*"]
        with os.scandir(split_dir) as it:
            names = sorted(
                e.name
                for e in it
                if e.is_dir()
                and any(fnmatch.fnmatchcase(e.name, pat) for pat in patterns)
            )
        # data_store holds lightweight handles (sequence dirs), not frame lists.
        self.data_store = {name: os.path.join(split_dir, name) for name in names}

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable PointOdyssey sequences under {split_dir} "
                f"(split={split!r}, sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}

    def _load_sequence(self, seq_name: str) -> list:
        """Lazily enumerate one sequence's frames -> sorted frame indices (ints).

        One ``rgb/`` directory listing on first access, then cached. Raises
        ValueError if the sequence has fewer than ``min_num_images`` frames
        (enforced here rather than at construction so startup stays cheap).
        """
        cached = self._frames_cache.get(seq_name)
        if cached is not None:
            return cached
        rgb_dir = os.path.join(self.data_store[seq_name], "rgb")
        frame_ids = sorted(
            int(f[: -len(".jpg")]) for f in os.listdir(rgb_dir) if f.endswith(".jpg")
        )
        if len(frame_ids) < self.min_num_images:
            raise ValueError(
                f"PointOdyssey {self.split}/{seq_name}: only {len(frame_ids)} frames "
                f"(< {self.min_num_images})"
            )
        self._frames_cache[seq_name] = frame_ids
        return frame_ids

    def _frame_paths(self, seq_name: str, frame_idx: int):
        """(rgb_path, depth_path, cam_path) for one frame of one sequence."""
        seq_dir = self.data_store[seq_name]
        stem = f"{frame_idx:05d}"
        return (
            os.path.join(seq_dir, "rgb", stem + ".jpg"),
            os.path.join(seq_dir, "depth", stem + ".npy"),
            os.path.join(seq_dir, "cam", stem + ".npz"),
        )

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Loads that one
        sequence's frame list lazily."""
        return len(self._load_sequence(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            frame_ids = self._load_sequence(name)
            rgb_path, _, _ = self._frame_paths(name, frame_ids[0])
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
        frames = self._load_sequence(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        timestamps, original_sizes = [], []

        for i in ids:
            frame_idx = frames[int(i)]
            rgb_path, depth_path, cam_path = self._frame_paths(seq_name, frame_idx)
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the rgb dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"PointOdyssey: could not read image {rgb_path}"
                )
            depth_map = self.read_pointodyssey_depth(depth_path)
            c2w, K_raw = self.read_pointodyssey_cam(cam_path)
            pose_w2c = self.pointodyssey_pose_to_w2c(c2w)
            K = self.pointodyssey_intrinsics(K_raw, self.intrinsics_override)
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
            timestamps.append(frame_idx / self._FPS)
            original_sizes.append(original_size)

        return {
            "seq_name": "pointodyssey_" + seq_name,
            "ids": np.array(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            # NO "sky_masks": SKY_MASK is not advertised and an all-False mask
            # would be wrong GT (sky is conflated with invalid at depth==0).
            "timestamps": np.array(timestamps, dtype=np.float64),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
