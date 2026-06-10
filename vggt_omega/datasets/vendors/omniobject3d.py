"""OmniObject3D vendor for the VGGT-Omega dataset API.

OmniObject3D is a large synthetic multi-view dataset of object-centric Blender
renders: 216 categories, 5914 object sequences, each with exactly 100 views
(``r_0`` .. ``r_99``). Only a ``train/`` split exists at the data root::

    <OMNIOBJECT3D_DIR>/train/<category>/<category>_<NNN>/
        rgb/r_<i>.png     RGB 800x800, white background (no alpha)
        depth/r_<i>.npy   float32 (800,800) z-depth, background = exactly 0.0
        cam/r_<i>.npz     keys "intrinsics" (3,3) and "pose" (4,4)

Conventions used here (validated empirically against this dataset, not assumed):

* ``cam/r_<i>.npz["pose"]`` is **camera-to-world in the OpenCV optical frame**
  (x-right, y-down, z-forward). Despite the Blender origin, the usual
  OpenGL->OpenCV axis flip has ALREADY been applied upstream (negative-zero
  artifacts in the last row are the fingerprint of the ``diag(1,-1,-1)`` flip)
  -- do NOT flip again. world->camera is simply ``np.linalg.inv(pose)[:3]``.
  Cross-frame depth reprojection closes to ~0.09% median relative error on
  adjacent-on-sphere views, confirming the convention; the OpenGL hypotheses
  land zero valid pixels.
* Depth is **z-depth along the optical axis** (not ray distance), already in
  the same normalized scene units as the poses (no scaling). Background /
  invalid pixels are exactly 0.0; there is no sky concept (synthetic object
  renders), so SKY_MASK is NOT advertised and no ``sky_masks`` key is emitted
  (unadvertised keys must not leak into samples via carry_extra_modalities).
* Intrinsics are stored per-frame in the npz but are globally constant:
  fx = fy = 1111.1111, cx = cy = 400.0 for the native 800x800 frame.
* Scale is normalized, NOT metric: every camera sits on a sphere of radius
  ~4.031 around the object at the origin regardless of real object size, so
  ``is_metric=False``.
* Consecutive frame indices are NOT a camera trajectory: adjacent views are
  57-107 degrees apart on the viewing sphere (random sphere sampling), so
  ``is_video=False`` and no timestamps are synthesized. ``get_nearby`` /
  ``expand_ratio`` still narrow the *index* window for contract parity, but
  index-nearby does not imply pose-nearby here.
* Object id numbering has gaps (e.g. ``apple_004`` missing), so objects are
  enumerated from the directory listing, never formatted from indices.

Scalability: with 5914 sequences on a network FS, construction only lists the
category directories that can match ``sequences`` (one listing per matching
category; the full default enumeration is a few seconds). Per-sequence frame
lists, the ``min_num_images`` check and native sizes are deferred to first
access and cached, in the same spirit as the 7-Scenes vendor's lazy poses.

``sequences`` patterns are paths relative to ``train/``: a bare pattern (no
``/``) globs whole categories (like the 7-Scenes ``scenes`` knob), while a
``category/object`` pattern globs individual objects, e.g.
``["anise"]`` (29 objects) or ``["toy_plane/toy_plane_001"]`` (one object).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class OmniObject3dDataset(BaseDataset):
    """OmniObject3D as a VGGT-Omega BaseDataset (unordered multi-view, normalized scale)."""

    # OmniObject3D provides RGB + rendered z-depth + GT camera poses/intrinsics.
    # As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth
    # re-projected through the GT poses (not an independent point-cloud GT), so
    # they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just
    # must not be scored as a point cloud. No timestamps (unordered views) and
    # no sky (object renders), so neither TIMESTAMP nor SKY_MASK is advertised
    # (and no "sky_masks" key is emitted: unadvertised keys must not leak into
    # samples via carry_extra_modalities).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
        }
    )

    @staticmethod
    def omniobject3d_pose_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV axes, as stored) -> world-to-camera (3,4) float32.

        The stored pose is already in the OpenCV optical frame (the Blender->
        OpenCV flip was applied upstream), so world->camera is just the matrix
        inverse with no axis flip. Raises ValueError on a malformed or
        non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"OmniObject3D pose: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("OmniObject3D pose is non-finite")
        return np.linalg.inv(c2w)[:3].astype(np.float32)

    @staticmethod
    def read_omniobject3d_depth(path: str) -> np.ndarray:
        """Read a ``depth/r_<i>.npy`` -> float32 (H,W) z-depth in scene units.

        Background/invalid pixels are stored as exactly 0.0 and stay 0
        (invalid). Non-finite or negative values (never observed, but negative
        is reserved for the sky convention this dataset does not have) are
        defensively mapped to 0. Raises ValueError if the array is not 2-D.
        """
        depth = np.asarray(np.load(path), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"OmniObject3D depth {path!r}: expected 2-D array, got shape {depth.shape}"
            )
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    @staticmethod
    def read_omniobject3d_cam(path: str):
        """Read a ``cam/r_<i>.npz`` -> (K (3,3) float32, c2w (4,4) float64).

        The npz holds exactly two keys: ``intrinsics`` (pinhole K in pixels of
        the native frame) and ``pose`` (camera-to-world, OpenCV axes). Raises
        ValueError on missing keys or malformed shapes.
        """
        with np.load(path) as cam:
            if "intrinsics" not in cam or "pose" not in cam:
                raise ValueError(
                    f"OmniObject3D cam {path!r}: expected keys 'intrinsics' and 'pose', "
                    f"got {sorted(cam.keys())}"
                )
            K = np.asarray(cam["intrinsics"], dtype=np.float32)
            c2w = np.asarray(cam["pose"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"OmniObject3D cam {path!r}: intrinsics shape {K.shape} != (3,3)")
        if c2w.shape != (4, 4):
            raise ValueError(f"OmniObject3D cam {path!r}: pose shape {c2w.shape} != (4,4)")
        return K, c2w

    @classmethod
    def omniobject3d_intrinsics(cls, K=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K in pixels of the native frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame ``K`` from
        the cam npz is validated and used (globally constant in practice:
        fx=fy=1111.1111, cx=cy=400 for 800x800). Raises ValueError when neither
        is given or K is malformed.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K is None:
            raise ValueError(
                "no OmniObject3D intrinsics given; pass the cam-npz K or intrinsics=[fx,fy,cx,cy]"
            )
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"OmniObject3D intrinsics malformed: {K!r}")
        return K

    @staticmethod
    def _pattern_matches(pattern: str, category: str, obj: str) -> bool:
        """True if a ``sequences`` pattern selects ``category/obj``. A bare
        pattern (no '/') globs the category; otherwise the full name is globbed."""
        if "/" in pattern:
            return fnmatch.fnmatch(f"{category}/{obj}", pattern)
        return fnmatch.fnmatch(category, pattern)

    @staticmethod
    def _list_frames(seq_dir: str) -> list:
        """List one object dir -> [(rgb_path, depth_path, cam_path, frame_idx)],
        ordered by frame index (``r_<i>`` is not zero-padded, so lexical order
        would interleave r_1/r_10). Called lazily, once per accessed sequence."""
        frames = []
        rgb_dir = os.path.join(seq_dir, "rgb")
        for entry in os.scandir(rgb_dir):
            stem, ext = os.path.splitext(entry.name)  # "r_42", ".png"
            if ext != ".png" or not stem.startswith("r_"):
                continue
            frame_idx = int(stem.split("_", 1)[1])
            frames.append(
                (
                    entry.path,
                    os.path.join(seq_dir, "depth", f"{stem}.npy"),
                    os.path.join(seq_dir, "cam", f"{stem}.npz"),
                    frame_idx,
                )
            )
        frames.sort(key=lambda fr: fr[3])
        return frames

    def _get_frames(self, seq_name: str) -> list:
        """Frame list for ``seq_name``, listed lazily on first access and cached.

        ``min_num_images`` is enforced here (it cannot be checked for all 5914
        sequences at construction without defeating lazy enumeration); a too-
        short sequence fails loudly rather than silently under-filling a batch.
        """
        if seq_name not in self._frames_cache:
            frames = self._list_frames(self.data_store[seq_name])
            if len(frames) < self.min_num_images:
                raise ValueError(
                    f"OmniObject3D seq {seq_name}: only {len(frames)} frames "
                    f"(< min_num_images={self.min_num_images})"
                )
            self._frames_cache[seq_name] = frames
        return self._frames_cache[seq_name]

    def __init__(
        self,
        common_conf,
        split: str = "train",
        OMNIOBJECT3D_DIR: str = None,
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

        if OMNIOBJECT3D_DIR is None:
            raise ValueError("OMNIOBJECT3D_DIR must be specified")
        self.OMNIOBJECT3D_DIR = OMNIOBJECT3D_DIR
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        # OmniObject3D ships only a train/ split; `split` only picks the
        # virtual epoch length (the data itself is identical).
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Accept either the dataset root (containing train/) or train/ itself.
        train_dir = os.path.join(OMNIOBJECT3D_DIR, "train")
        if not os.path.isdir(train_dir):
            train_dir = OMNIOBJECT3D_DIR

        # Enumerate sequence NAMES only (no per-frame I/O): one listing of the
        # category level, then one listing per category that can match a
        # pattern. Bare patterns glob categories; "cat/obj" patterns glob
        # objects within their (pruned) category.
        patterns = list(sequences) if sequences else ["*"]
        cat_heads = [p.split("/", 1)[0] if "/" in p else p for p in patterns]
        try:
            categories = sorted(
                e.name
                for e in os.scandir(train_dir)
                if e.is_dir() and any(fnmatch.fnmatch(e.name, h) for h in cat_heads)
            )
        except FileNotFoundError:
            raise ValueError(f"OmniObject3D root not found: {OMNIOBJECT3D_DIR}")

        self.data_store = {}  # "category/object" -> object dir path (lazy handle)
        for cat in categories:
            cat_dir = os.path.join(train_dir, cat)
            for entry in sorted(os.scandir(cat_dir), key=lambda e: e.name):
                if not entry.is_dir():
                    continue
                if any(self._pattern_matches(p, cat, entry.name) for p in patterns):
                    self.data_store[f"{cat}/{entry.name}"] = entry.path

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable OmniObject3D sequences under {OMNIOBJECT3D_DIR} "
                f"(sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}
        logging.info(
            "OmniObject3D: %d sequences (split=%s)", self.sequence_list_len, split
        )

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Lists that one
        sequence lazily on first call."""
        return len(self._get_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._get_frames(name)[0][0]
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
        frames = self._get_frames(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            # Index-window narrowing only: OmniObject3D views are unordered on
            # the sphere, so index-nearby does not imply pose-nearby.
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            rgb_path, depth_path, cam_path, _frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Listed from the rgb dir, so the file should exist; fail loudly
                # (a silent skip would yield fewer than img_per_seq frames and
                # break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"OmniObject3D: could not read image {rgb_path}"
                )
            depth_map = self.read_omniobject3d_depth(depth_path)
            K_npz, c2w = self.read_omniobject3d_cam(cam_path)
            pose_w2c = self.omniobject3d_pose_to_w2c(c2w)
            K = self.omniobject3d_intrinsics(K_npz, self.intrinsics_override)
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
            original_sizes.append(original_size)

        return {
            "seq_name": "omniobject3d_" + seq_name,
            "ids": np.array(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            # NO "sky_masks": SKY_MASK is not advertised, so the key must not
            # leak into samples via carry_extra_modalities.
            "original_sizes": original_sizes,
            "is_metric": False,   # normalized scale: cameras at radius ~4.031 regardless of object size
            "is_video": False,    # unordered views on the sphere, not a trajectory
            "modalities": set(self.available_modalities),
        }
