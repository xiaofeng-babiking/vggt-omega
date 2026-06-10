"""ScanNet (preprocessed ``scans_train`` copy) vendor for the VGGT-Omega dataset API.

This copy is NOT the raw ScanNet release (no ``.sens``, no ``pose/``/``intrinsic/``
txt files): it is a DUSt3R/CUT3R-style per-frame extraction. Layout::

    {SCANNET_DIR}/scans_train/sceneXXXX_XX/      1510 scene dirs (multiple rescans
        color/%05d.jpg                           per physical scene share a prefix)
        depth/%05d.png                           16-bit PNG, millimeters
        cam/%05d.npz                             keys 'intrinsics' (3,3), 'pose' (4,4)
        new_scene_metadata.npz                   frame-id list + sliding-window pair
                                                 helper (stale abs paths; unused here)

Conventions used here (verified empirically against this copy, not assumed):

* Depth is 16-bit PNG in **millimeters** (meters = value / 1000); 0 is the
  invalid sentinel (~1-10% of pixels per frame) and stays 0. Indoor data with
  no sky: SKY_MASK is advertised like the other sky-free indoor vendors
  (TUM/7-Scenes), and the emitted ``sky_masks`` are always all-False (sky
  convention = depth<0).
* ``cam/%05d.npz`` key ``'pose'`` is **camera-to-world in the OpenCV optical
  frame** (x-right, y-down, z-forward); world->camera is its rigid inverse with
  no axis flip. Cross-frame depth reprojection closes to ~0.5% median relative
  error, simultaneously confirming the pose convention and the /1000 depth scale.
* ``'intrinsics'`` is a per-frame (3,3) K, constant within a scene but slightly
  different across scenes (e.g. fx=577.59 fy=578.73 cx=318.91 cy=242.68). Color
  was already resized to the 640x480 depth resolution, so the stored K is valid
  for both RGB and depth as stored. Override via ``intrinsics=[fx, fy, cx, cy]``.
* Frames are a ~30 fps RGB-D video (305..~5600 frames per scene), every raw
  frame present, metric scale. No per-frame clock ships in this copy, but the
  frames are a 30 Hz video, so -- like the 7-Scenes vendor -- we synthesize
  ``timestamp = frame_number / 30`` (faithful relative capture time, not
  fabricated precision; the numeric filename stem is used, so any frame-id gap
  stays a gap) and advertise TIMESTAMP for video ordering.

Laziness / scalability: 1510 scenes x up to ~5600 frames each. Construction
performs a single directory listing of ``scans_train`` to enumerate scene
NAMES; per-scene frame lists are built on first access (one listing of that
scene's ``color/`` dir) and cached, like the 7-Scenes vendor's lazy poses.
Consequently the ``min_num_images`` check is also lazy: a scene that turns out
too short raises a clear ValueError at first access (every surveyed scene has
>= ~300 frames, so this only fires on a corrupt/partial scene -- exclude it via
``sequences``).

Non-finite poses: raw ScanNet marks tracking failures with ``-inf`` pose rows.
None were found in this preprocessed copy (160 sampled poses across 12 random
scenes + scene0000_00 came back finite -- the failures appear filtered during
preprocessing), but the guard is kept: a non-finite pose raises ValueError with
the offending file path, mirroring the 7-Scenes vendor.
"""
from __future__ import annotations

import fnmatch
import os
import random

import cv2
import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class ScannetDataset(BaseDataset):
    """ScanNet (preprocessed) as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # ScanNet provides RGB + registered metric depth + GT camera poses and
    # per-frame intrinsics. As with the TUM vendor, WORLD_POINTS / CAM_POINTS
    # are only the depth re-projected through the GT poses (not an independent
    # point-cloud GT), so they are NOT advertised as evaluable GT modalities --
    # process_one_image still computes them (e.g. for depth-supervised point
    # heads), they just must not be scored as a point cloud. Indoor data:
    # SKY_MASK is advertised like TUM/7-Scenes (the emitted masks are always
    # all-False; sky convention = depth<0). No per-frame clock ships, but the
    # frames are a ~30 fps video, so TIMESTAMP is synthesized as
    # frame_number / 30 and advertised (see module docstring).
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

    # Frame rate used to synthesize timestamps (ScanNet RGB-D is a ~30 Hz video).
    _FPS = 30.0

    @staticmethod
    def scannet_pose_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV frame) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; no matrix inverse needed). Raises ValueError on a
        wrong shape or a non-finite pose (raw ScanNet marks tracking failures
        with -inf entries).
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(
                f"ScanNet pose: expected a (4,4) camera-to-world matrix, got shape {c2w.shape}"
            )
        if not np.isfinite(c2w).all():
            raise ValueError("ScanNet pose is non-finite (tracking-failure frame)")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_scannet_cam(path: str):
        """Read a ScanNet ``cam/%05d.npz`` -> (K (3,3) float32, c2w (4,4) float64).

        ``'intrinsics'`` is the pinhole K in pixels of the stored 640x480 frame;
        ``'pose'`` is camera-to-world in the OpenCV optical frame. Raises
        ValueError if either key is missing.
        """
        with np.load(path) as cam:
            if "intrinsics" not in cam or "pose" not in cam:
                raise ValueError(
                    f"ScanNet cam file {path!r}: expected keys 'intrinsics' and 'pose', "
                    f"got {sorted(cam.files)}"
                )
            K = np.asarray(cam["intrinsics"], dtype=np.float32)
            c2w = np.asarray(cam["pose"], dtype=np.float64)
        return K, c2w

    @staticmethod
    def read_scannet_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
        """Read a ScanNet 16-bit depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 millimeter counts (meters = value /
        depth_scale, verified by reprojection). 0 stays 0 (invalid); any
        non-finite value also maps to 0.
        """
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"ScanNet: could not read depth {path}")
        depth = arr.astype(np.float32) / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @classmethod
    def scannet_intrinsics(cls, K=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K for a ScanNet frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame ``K`` from
        ``cam/*.npz`` is validated (shape, finiteness, positive focals) and
        passed through. Raises ValueError when neither is usable.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K is None:
            raise ValueError(
                "ScanNet intrinsics: need the per-frame K from cam/*.npz or "
                "intrinsics=[fx, fy, cx, cy]"
            )
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"ScanNet intrinsics: invalid K\n{K!r}")
        return K

    def _get_frames(self, seq_name: str) -> list:
        """Frame list for ``seq_name`` -> [(color_path, depth_path, cam_path,
        frame_idx)], ordered by frame index, subsampled by ``self.frame_step``.

        Built lazily on first access (one listing of the scene's ``color/`` dir;
        nothing is decoded) and cached in ``self.data_store``. The
        ``min_num_images`` check happens here, lazily (see module docstring).
        """
        frames = self.data_store.get(seq_name)
        if frames is None:
            seq_dir = os.path.join(self._scans_dir, seq_name)
            color_dir = os.path.join(seq_dir, "color")
            if not os.path.isdir(color_dir):
                raise ValueError(f"ScanNet {seq_name}: missing color/ dir under {seq_dir}")
            frames = []
            for fname in os.listdir(color_dir):
                stem, ext = os.path.splitext(fname)
                if ext.lower() != ".jpg" or not stem.isdigit():
                    continue
                frames.append(
                    (
                        os.path.join(color_dir, fname),
                        os.path.join(seq_dir, "depth", stem + ".png"),
                        os.path.join(seq_dir, "cam", stem + ".npz"),
                        int(stem),
                    )
                )
            frames.sort(key=lambda fr: fr[3])
            if self.frame_step > 1:
                frames = frames[:: self.frame_step]
            if len(frames) < self.min_num_images:
                raise ValueError(
                    f"ScanNet {seq_name}: only {len(frames)} frames "
                    f"(< min_num_images={self.min_num_images}); surveyed scenes have "
                    ">= ~300 frames, so this indicates a corrupt/partial scene -- "
                    "exclude it via `sequences` (frame counts are checked lazily)."
                )
            self.data_store[seq_name] = frames
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SCANNET_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 1000.0,
        frame_step: int = 1,
        intrinsics=None,
        scans_subdir: str = "scans_train",
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if SCANNET_DIR is None:
            raise ValueError("SCANNET_DIR must be specified")
        if frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")

        self.SCANNET_DIR = SCANNET_DIR
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self.frame_step = frame_step
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # This copy ships a single split dir (scans_train); enumerate scene
        # NAMES with one directory listing -- per-frame enumeration is deferred
        # to _get_frames (first access per scene), so construction stays fast
        # for all 1510 scenes on a network FS.
        self._scans_dir = os.path.join(SCANNET_DIR, scans_subdir)
        if not os.path.isdir(self._scans_dir):
            raise ValueError(f"ScanNet: {self._scans_dir} is not a directory")
        patterns = sequences or ["*"]
        with os.scandir(self._scans_dir) as it:
            all_names = [e.name for e in it if e.is_dir()]
        self.sequence_list = sorted(
            {n for n in all_names if any(fnmatch.fnmatchcase(n, p) for p in patterns)}
        )

        # Lazy cache: seq_name -> frame list, filled by _get_frames on demand.
        self.data_store = {}
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable ScanNet sequences under {self._scans_dir} (sequences={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Triggers the
        lazy per-scene frame listing on first call."""
        return len(self._get_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            color_path = self._get_frames(name)[0][0]
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
        frames = self._get_frames(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, original_sizes = [], [], []

        for i in ids:
            color_path, depth_path, cam_path, frame_num = frames[int(i)]
            image = read_image_cv2(color_path)
            if image is None:
                # Enumerated from the color/ dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"ScanNet: could not read image {color_path} (listed in color/ but unreadable)"
                )
            depth_map = self.read_scannet_depth(depth_path, self.depth_scale)
            K_frame, c2w = self.read_scannet_cam(cam_path)
            try:
                pose_w2c = self.scannet_pose_to_w2c(c2w)
            except ValueError as e:
                raise ValueError(f"ScanNet {cam_path}: {e}") from e
            K = self.scannet_intrinsics(K_frame, self.intrinsics_override)
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
            sky_masks.append(depth_map < 0)  # ScanNet is indoor: always all-False (sky convention = depth<0)
            # NOMINAL ~30 fps clock from the ACTUAL numeric frame stem (not the
            # positional index), so any frame-id gap stays a gap in time.
            timestamps.append(frame_num / self._FPS)
            original_sizes.append(original_size)

        return {
            "seq_name": "scannet_" + seq_name,
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
