"""Virtual KITTI 2 (preprocessed/flattened copy) vendor for the VGGT-Omega dataset API.

This copy is NOT the official VKITTI2 release layout (no ``frames/rgb/...``
subtree, no textgt ``extrinsic.txt``/``intrinsic.txt``); the cameras were
flattened into per-frame npz files::

    VKITTI_DIR/train/Scene{01,02,06,18,20}/<variation>/Camera_{0,1}/
        {idx:05d}_rgb.jpg            RGB, 1242x375, JPEG uint8
        {idx:05d}_depth.png          single-channel uint16, depth in CENTIMETERS
        {idx:05d}_cam.npz            'camera_pose' (4,4) + 'camera_intrinsics' (3,3)
        {idx:05d}_forwardFlow.png    (frames 0..N-2; not loaded here)
        {idx:05d}_backwardFlow.png   (frames 1..N-1; not loaded here)

with ``<variation>`` one of 15-deg-left, 15-deg-right, 30-deg-left,
30-deg-right, clone, fog, morning, overcast, rain, sunset. A *sequence* here is
one camera stream ``Scene/variation/Camera_N`` (5 scenes x 10 variations x 2
cameras = 100 streams, 233..837 frames each). Only a single on-disk ``train/``
split exists; the constructor's ``split`` argument selects the virtual epoch
length only.

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is 16-bit PNG in **centimeters** (meters = value / 100). The raw value
  65535 (= 655.35 m) is the sky/far clamp and maps to **-1.0** (the repo-wide
  "<0 = sky" convention); no zero/NaN invalid pixels were observed, but any
  non-finite value maps to 0 (invalid).
* ``cam.npz['camera_pose']`` is **camera-to-world** in OpenCV camera axes
  (x-right, y-down, z-forward) -- the OPPOSITE of the official VKITTI2 textgt
  ``extrinsic.txt`` (which stores world-to-camera); this copy pre-inverted it.
  Cross-frame depth reprojection closes to ~0.1% median relative error for the
  c2w reading (vs ~34% for the w2c reading), and the recovered stereo baseline
  is 0.533 m (KITTI's published 0.532 m), confirming both convention and
  metric scale. world->camera is therefore ``inv(camera_pose)[:3, :4]``.
* ``cam.npz['camera_intrinsics']`` is the (3,3) pinhole K in pixels of the
  native 1242x375 frame; it is globally constant (fx=fy=725.0087, cx=620.5,
  cy=187.0) but is read from the per-frame npz to follow the disk. Override
  via ``intrinsics=[fx, fy, cx, cy]``.
* Scenes are Unity-rendered clones of real KITTI drives at ~10 Hz: depth and
  poses are exact synthetic GT, fully metric, video-ordered. The dataset ships
  no timestamps, so ``timestamp = frame_index / 10`` is synthesized (faithful
  relative capture time at the nominal 10 fps, not fabricated precision).
* The two stereo cameras (baseline 0.533 m) are exposed via CAMERA_ID (0/1),
  parsed from the ``Camera_N`` directory name; the pose npz is per-camera so
  no extra stereo handling is needed.

Known caveats (documented, handled gracefully):

* The 10 weather/condition variations of a scene share the same trajectory and
  geometry (clone/fog/morning/overcast/rain/sunset are identical GT; the
  15/30-deg variants only rotate the cameras), so sampling several variations
  of one scene duplicates 3D content.
* Scenes are dynamic (moving cars) and this copy has no dynamic-object masks:
  depth+pose are exact per frame, but multi-view consistency breaks on moving
  objects (~92% of pixels pass 5% reprojection at a 5-frame gap; the residual
  is dynamic objects/occlusion).

Per-frame files are enumerated **lazily** (one directory listing per sequence,
on first access, cached): the copy holds ~100 streams x up to ~840 frames on a
network FS and eager enumeration would dominate startup. Construction only
globs the ``Scene*/<variation>/Camera_*`` directory names.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class VkittiDataset(BaseDataset):
    """Virtual KITTI 2 as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # VKITTI2 is rendered at the nominal KITTI rate (~10 Hz); used to synthesize
    # timestamps (the copy ships none).
    _FPS = 10.0

    # Raw uint16 depth value used as the sky/far clamp (= 655.35 m at cm scale).
    _SKY_RAW = 65535

    # VKITTI provides RGB + exact synthetic metric depth + GT poses/intrinsics.
    # As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth
    # re-projected through the GT poses (not an independent point-cloud GT), so
    # they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just
    # must not be scored as a point cloud.
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.TIMESTAMP,
            Modality.CAMERA_ID,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
        }
    )

    @staticmethod
    def vkitti_pose_to_w2c(camera_pose) -> np.ndarray:
        """VKITTI ``cam.npz['camera_pose']`` (4,4) camera-to-world (OpenCV axes)
        -> world-to-camera (3,4) float32 OpenCV.

        This copy stores camera-to-world (the official VKITTI2 textgt stores
        world-to-camera -- verified empirically, see module docstring), so the
        pose is inverted. Raises ValueError on a non-(4,4) or non-finite pose.
        """
        c2w = np.asarray(camera_pose, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"VKITTI camera_pose: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("VKITTI camera_pose is non-finite")
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @staticmethod
    def read_vkitti_depth(
        path: str, depth_scale: float = 100.0, sky_raw: int = 65535
    ) -> np.ndarray:
        """Read a VKITTI 16-bit depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 centimeter counts (meters = value /
        depth_scale). The ``sky_raw`` clamp value (65535 = 655.35 m) encodes
        sky/far and maps to -1.0 (repo convention: depth<0 = sky); non-finite
        values map to 0 (invalid). No zero-invalid pixels were observed in this
        copy, and a raw 0 stays 0 m (= invalid by convention) anyway.
        """
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"VKITTI: could not read depth {path}")
        if raw.ndim != 2:
            raise ValueError(
                f"VKITTI depth {path!r}: expected single-channel, got shape {raw.shape}"
            )
        raw = raw.astype(np.float32)
        depth = raw / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        depth[raw == float(sky_raw)] = -1.0
        return depth

    @classmethod
    def vkitti_intrinsics(cls, K_raw=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K for a VKITTI frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise ``K_raw`` (the per-frame
        ``cam.npz['camera_intrinsics']``, globally constant fx=fy=725.0087,
        cx=620.5, cy=187.0 @ 1242x375) is validated and returned. Raises
        ValueError when neither is given or ``K_raw`` is not (3,3)/finite.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K_raw is None:
            raise ValueError(
                "no VKITTI intrinsics: pass K_raw from cam.npz or intrinsics=[fx,fy,cx,cy]"
            )
        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"VKITTI camera_intrinsics: expected (3,3), got {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("VKITTI camera_intrinsics is non-finite")
        return K

    @classmethod
    def read_vkitti_cam(cls, path: str):
        """Read a VKITTI ``cam.npz`` -> (w2c (3,4) float32, K_raw (3,3) float32).

        The npz holds 'camera_pose' (camera-to-world, inverted here) and
        'camera_intrinsics' (raw K; pass through :meth:`vkitti_intrinsics`).
        """
        with np.load(path) as cam:
            w2c = cls.vkitti_pose_to_w2c(cam["camera_pose"])
            K_raw = np.asarray(cam["camera_intrinsics"], dtype=np.float32)
        return w2c, K_raw

    @staticmethod
    def parse_camera_id(seq_name: str) -> int:
        """Stereo camera id (0/1) from a ``Scene/variation/Camera_N`` sequence
        name. Raises ValueError if the name does not end in ``Camera_<int>``."""
        leaf = seq_name.rstrip("/").rsplit("/", 1)[-1]
        prefix = "Camera_"
        if leaf.startswith(prefix):
            try:
                return int(leaf[len(prefix):])
            except ValueError:
                pass
        raise ValueError(
            f"VKITTI sequence {seq_name!r}: expected a 'Camera_<int>' leaf directory"
        )

    @staticmethod
    def _list_frames(seq_dir: str) -> list:
        """List a camera-stream dir -> [(rgb_path, depth_path, cam_path,
        frame_idx)], ordered by frame index. One directory listing; no stats on
        the sibling depth/cam files (missing ones fail loudly at read time)."""
        frames = []
        for fname in os.listdir(seq_dir):
            if not fname.endswith("_rgb.jpg"):
                continue
            stem = fname[: -len("_rgb.jpg")]  # "00000"
            try:
                frame_idx = int(stem)
            except ValueError:
                continue
            frames.append(
                (
                    os.path.join(seq_dir, fname),
                    os.path.join(seq_dir, stem + "_depth.png"),
                    os.path.join(seq_dir, stem + "_cam.npz"),
                    frame_idx,
                )
            )
        frames.sort(key=lambda fr: fr[3])
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKITTI_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 100.0,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if VKITTI_DIR is None:
            raise ValueError("VKITTI_DIR must be specified")

        self.VKITTI_DIR = VKITTI_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # This copy has a single on-disk "train" split; `split` only picks the
        # virtual epoch length above. Tolerate roots that already point inside it.
        split_root = os.path.join(VKITTI_DIR, "train")
        self._root = split_root if os.path.isdir(split_root) else VKITTI_DIR

        # Enumerate camera-stream NAMES only (Scene*/<variation>/Camera_*):
        # ~56 directory listings, no per-frame work (frames are listed lazily).
        stream_names = sorted(
            os.path.relpath(d, self._root)
            for d in glob.glob(os.path.join(self._root, "*", "*", "Camera_*"))
            if os.path.isdir(d)
        )

        # `sequences` filters by exact name or glob pattern; a pattern matching
        # a prefix (e.g. "Scene02/clone") selects all its camera streams.
        patterns = sequences or ["*"]
        self.data_store = {
            name: os.path.join(self._root, name)
            for name in stream_names
            if any(
                fnmatch.fnmatch(name, pat)
                or fnmatch.fnmatch(name, pat.rstrip("/") + "/*")
                for pat in patterns
            )
        }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable VKITTI sequences under {self._root} (sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}

    def _sequence_frames(self, seq_name: str) -> list:
        """Frame list for ``seq_name``, enumerated lazily on first access and
        cached. Raises ValueError if the stream has fewer than
        ``min_num_images`` frames (never the case in this copy: 233..837)."""
        frames = self._frames_cache.get(seq_name)
        if frames is None:
            frames = self._list_frames(self.data_store[seq_name])
            if len(frames) < self.min_num_images:
                raise ValueError(
                    f"VKITTI {seq_name}: only {len(frames)} frames "
                    f"(< {self.min_num_images})"
                )
            self._frames_cache[seq_name] = frames
        return frames

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self._sequence_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._sequence_frames(name)[0][0]
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
        frames = self._sequence_frames(seq_name)
        cam_id = self.parse_camera_id(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, camera_ids, original_sizes = [], [], [], []

        for i in ids:
            rgb_path, depth_path, cam_path, frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the stream dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(f"VKITTI: could not read image {rgb_path}")
            depth_map = self.read_vkitti_depth(depth_path, self.depth_scale, self._SKY_RAW)
            pose_w2c, K_raw = self.read_vkitti_cam(cam_path)
            K = self.vkitti_intrinsics(K_raw, self.intrinsics_override)
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
            sky_masks.append(depth_map < 0)  # sky/far clamp mapped to -1.0 in read_vkitti_depth
            timestamps.append(frame_idx / self._FPS)
            camera_ids.append(cam_id)
            original_sizes.append(original_size)

        return {
            "seq_name": "vkitti_" + seq_name,
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
            "camera_ids": np.array(camera_ids, dtype=np.int32),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
