"""Waymo Open Dataset (preprocessed extraction) vendor for the VGGT-Omega dataset API.

This copy is NOT raw Waymo tfrecords: each ``train/segment-*.tfrecord`` entry is
a DIRECTORY of preprocessed per-frame files (someone already ran the extraction
step), fully decodable with cv2 + numpy -- no tensorflow / waymo_open_dataset
needed::

    train/
    |-- invalid_files.h5                              optional frame blacklist (see below)
    `-- segment-<id>_with_camera_labels.tfrecord/     798 segment DIRECTORIES
        |-- {frame:05d}_{cam}.jpg    RGB              cam in {1..5}, frame 0..~198 contiguous
        |-- {frame:05d}_{cam}.exr    sparse LiDAR depth (float32 meters)
        `-- {frame:05d}_{cam}.npz    'intrinsics' (3,3), 'cam2world' (4,4), 'distortion' (5,)

Conventions used here (validated empirically against this copy, not assumed):

* One **sequence = one segment x one camera** (a single 10 Hz video stream);
  CAMERA_ID is advertised. Camera ids follow the Waymo convention (1=FRONT,
  2=FRONT_LEFT, 3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT). Resolutions are
  per-camera: cams 1-3 are 341x512 (HxW), cams 4-5 are 236x512.
* Depth ``.exr`` is single-channel float32 **already in meters** (sparse
  projected LiDAR, only ~11-17% of pixels valid); exactly 0.0 = no return /
  invalid. Sky has NO special encoding (it is 0 like any other no-return
  pixel), so SKY_MASK is not derivable: it is neither advertised nor is a
  ``sky_masks`` key emitted -- this is outdoor driving data that DOES contain
  sky, so an all-False mask would be wrong GT (same rule as MegaDepth).
* Reading EXR with OpenCV requires the env var ``OPENCV_IO_ENABLE_OPENEXR=1``;
  this module sets it (``os.environ.setdefault``) before importing cv2, and the
  installed OpenCV build was verified to honor it even when set post-import.
* ``npz['cam2world']`` is **camera-to-world with OpenCV camera axes** (x right,
  y down, z forward), verified by cross-frame depth reprojection (median rel
  err 0.007); world->camera is its rigid inverse with no axis flip. The world
  frame is a per-run global frame with huge offsets (~1e4 m), so poses are
  **recentered per segment** (subtracting the first frame's camera position)
  before the float32 cast -- all 5 cameras of a segment share one recentered
  world frame.
* ``npz['intrinsics']`` is a (3,3) pinhole K in pixels of the stored (already
  downscaled) image; it is read per frame (same npz as the pose). The stored
  ``distortion`` (5,) coefficients were verified negligible at this resolution
  (median reproj err 0.0070 pinhole vs 0.0071 with distortion) and are ignored.
* No timestamps exist on disk; temporal order / 10 Hz capture is implied by the
  contiguous frame index only, so TIMESTAMP is not advertised.

Sequences are enumerated **lazily**: construction does a single directory
listing of segment NAMES (798 segments x 5 cams = 3990 sequences); the
per-segment frame listing (one ``os.listdir`` of ~3000 entries, shared by all 5
camera-sequences of that segment) is deferred to first access and cached.
Consequently ``min_num_images`` is enforced at first access (a too-short
on-disk listing raises ValueError then, not at construction) -- a full-disk
sweep verified 171-200 frames per camera for every segment, so this should not
trigger; a blacklist-caused shortfall falls back to unfiltered (see below).

``invalid_files.h5`` is a per-segment frame blacklist (group per segment dir
name, dataset ``invalid_pairs`` of ``'{cam}_{frame:05d}'`` tokens). Reading it
requires h5py, which is NOT installed in this environment, so the blacklist is
OPTIONAL and UN-APPLIED by default (``use_blacklist=False``); when enabled, h5py
is imported softly and a missing h5py only logs a warning (no filtering). The
real blacklist contains whole-camera runs (e.g. ~195 of 199 frames of one
camera flagged), and 'invalid' is a quality flag, not corruption -- the frames
decode fine. So when filtering would leave a sequence below ``min_num_images``,
the blacklist is IGNORED for that sequence (warn + fall back to the unfiltered
frames) instead of raising: sequence_list is already published at construction,
so a lazy raise would kill a DataLoader worker mid-training and crash
ComposedDataset sequence enumeration.
"""
from __future__ import annotations

import logging
import os
import random
import re
from fnmatch import fnmatch

# Must be set before OpenCV's EXR codec is first used (see module docstring).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class WaymoDataset(BaseDataset):
    """Preprocessed Waymo Open Dataset as a VGGT-Omega BaseDataset.

    Per-camera 10 Hz driving video with sparse metric LiDAR depth and GT poses.
    """

    # Suffix shared by every segment directory; stripped for sequence names.
    _SEG_SUFFIX = "_with_camera_labels.tfrecord"

    # "{frame:05d}_{cam}.jpg" -- the jpg is the anchor file of a frame; the
    # paired .exr / .npz are derived from the same stem.
    _FRAME_RE = re.compile(r"^(\d{5})_(\d)\.jpg$")

    # Waymo camera ids present in this extraction.
    _ALL_CAMERAS = (1, 2, 3, 4, 5)

    # Waymo provides RGB + sparse metric LiDAR depth + GT poses/intrinsics, with
    # 5 cameras per segment (CAMERA_ID). As with the TUM vendor, WORLD_POINTS /
    # CAM_POINTS are only the depth re-projected through the GT poses (not an
    # independent point-cloud GT), so they are NOT advertised as evaluable GT
    # modalities -- process_one_image still computes them, they just must not be
    # scored as a point cloud. No SKY_MASK (sky is indistinguishable from other
    # no-return pixels; no "sky_masks" key is emitted either -- an all-False
    # mask for outdoor driving data would be wrong GT) and no TIMESTAMP
    # (nothing on disk; order is frame index).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
            Modality.CAMERA_ID,
        }
    )

    @classmethod
    def segment_short_name(cls, dir_basename: str) -> str:
        """Segment directory basename -> short sequence-name stem.

        Strips the ``_with_camera_labels.tfrecord`` suffix (or a bare
        ``.tfrecord``) so sequence names stay readable; the leading
        ``segment-<id>`` part is kept verbatim.
        """
        if dir_basename.endswith(cls._SEG_SUFFIX):
            return dir_basename[: -len(cls._SEG_SUFFIX)]
        if dir_basename.endswith(".tfrecord"):
            return dir_basename[: -len(".tfrecord")]
        return dir_basename

    @classmethod
    def parse_segment_listing(cls, filenames) -> dict:
        """Parse a segment directory listing -> ``{cam: [frame_num, ...]}`` with
        each camera's frame numbers sorted ascending.

        Only ``{frame:05d}_{cam}.jpg`` entries are counted (the jpg anchors a
        frame; ``.exr`` / ``.npz`` share its stem); stray files are ignored.
        """
        per_cam: dict = {}
        for fn in filenames:
            m = cls._FRAME_RE.match(fn)
            if not m:
                continue
            frame_num, cam = int(m.group(1)), int(m.group(2))
            per_cam.setdefault(cam, []).append(frame_num)
        for cam in per_cam:
            per_cam[cam].sort()
        return per_cam

    @staticmethod
    def waymo_pose_to_w2c(c2w, anchor=None) -> np.ndarray:
        """Waymo ``cam2world`` (4,4) -> world-to-camera (3,4) float32 OpenCV.

        ``cam2world`` is camera-to-world in the OpenCV optical frame (verified
        by cross-frame depth reprojection), so world->camera is the rigid
        inverse [R^T | -R^T t] -- no axis flip. ``anchor`` (3,) is subtracted
        from the camera position first to recenter the huge (~1e4 m) global
        world coordinates before the float32 cast. Raises ValueError on a
        non-(4,4) or non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Waymo cam2world: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("Waymo cam2world is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        if anchor is not None:
            trans_c2w = trans_c2w - np.asarray(anchor, dtype=np.float64).reshape(3)
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_waymo_depth(path: str) -> np.ndarray:
        """Read a Waymo ``.exr`` depth -> float32 (H,W) meters.

        The EXR stores single-channel float32 sparse projected LiDAR already in
        meters; exactly 0.0 = no return / invalid. Non-finite or negative
        values (none observed on disk) are defensively mapped to 0. Requires
        ``OPENCV_IO_ENABLE_OPENEXR=1`` (set at module import).
        """
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Waymo: could not read depth EXR {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    @staticmethod
    def waymo_intrinsics(K, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K from a Waymo npz ``intrinsics`` entry.

        ``override``=[fx, fy, cx, cy] wins (e.g. for a re-calibrated rig);
        otherwise the stored per-frame K is validated and cast. Raises
        ValueError if K is not (3,3) finite.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"Waymo intrinsics: expected finite (3,3), got shape {K.shape}")
        return K.astype(np.float32)

    @staticmethod
    def filter_blacklisted_frames(frames, cam: int, invalid_tokens) -> list:
        """Drop frame tuples whose ``'{cam}_{frame:05d}'`` token is blacklisted.

        ``frames`` entries are ``(jpg, exr, npz, frame_num)``; an empty/None
        token set is a no-op (the default when h5py is unavailable).
        """
        if not invalid_tokens:
            return list(frames)
        return [fr for fr in frames if f"{cam}_{fr[3]:05d}" not in invalid_tokens]

    def __init__(
        self,
        common_conf,
        split: str = "train",
        WAYMO_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        cameras=(1, 2, 3, 4, 5),
        intrinsics=None,
        use_blacklist: bool = False,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if WAYMO_DIR is None:
            raise ValueError("WAYMO_DIR must be specified")
        cameras = tuple(int(c) for c in cameras)
        if not cameras or any(c not in self._ALL_CAMERAS for c in cameras):
            raise ValueError(f"cameras must be a non-empty subset of {self._ALL_CAMERAS}, got {cameras}")

        self.WAYMO_DIR = WAYMO_DIR
        # This copy ships a single 'train' subdir (no val/test on disk); `split`
        # only selects the virtual epoch length.
        seg_root = os.path.join(WAYMO_DIR, "train")
        if not os.path.isdir(seg_root):
            seg_root = WAYMO_DIR  # WAYMO_DIR may point directly at the segment dirs
        self.seg_root = seg_root
        self.expand_ratio = expand_ratio
        self.cameras = cameras
        self.intrinsics_override = intrinsics
        self.use_blacklist = use_blacklist
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Cheap construction: ONE directory listing of segment names; per-frame
        # enumeration (one listdir per segment) is deferred to first access.
        patterns = list(sequences) if sequences else None
        self._segment_dirs = {}  # short name -> absolute segment dir
        self.data_store = {}     # seq_name -> (segment short name, cam) handle
        try:
            entries = sorted(os.scandir(seg_root), key=lambda e: e.name)
        except FileNotFoundError:
            raise ValueError(f"Waymo segment root {seg_root} does not exist")
        for entry in entries:
            if not entry.name.startswith("segment-") or not entry.is_dir():
                continue
            short = self.segment_short_name(entry.name)
            for cam in cameras:
                seq_name = f"{short}/cam{cam}"
                if patterns is not None and not any(
                    fnmatch(seq_name, p) or fnmatch(short, p) or fnmatch(entry.name, p)
                    for p in patterns
                ):
                    continue
                self._segment_dirs[short] = entry.path
                self.data_store[seq_name] = (short, cam)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Waymo sequences under {seg_root} (sequences={patterns})"
            )

        self._blacklist_path = os.path.join(seg_root, "invalid_files.h5")
        self._blacklist_cache = {}     # segment short name -> frozenset of tokens
        self._warned_no_h5py = False
        self._segment_listing_cache = {}  # segment short name -> {cam: [frame_num]}
        self._frames_cache = {}        # seq_name -> [(jpg, exr, npz, frame_num)]
        self._anchor_cache = {}        # segment short name -> (3,) float64
        self._native_size_cache = {}

    # --- lazy per-segment metadata ---------------------------------------

    def _segment_listing(self, short: str) -> dict:
        """``{cam: [frame_num, ...]}`` for one segment, from a single cached
        ``os.listdir`` shared by all 5 camera-sequences of that segment."""
        if short not in self._segment_listing_cache:
            self._segment_listing_cache[short] = self.parse_segment_listing(
                os.listdir(self._segment_dirs[short])
            )
        return self._segment_listing_cache[short]

    def _segment_blacklist(self, short: str):
        """Blacklist tokens ``'{cam}_{frame:05d}'`` for one segment, or an empty
        set when disabled / h5py missing / file unreadable (soft failure)."""
        if not self.use_blacklist:
            return frozenset()
        if short in self._blacklist_cache:
            return self._blacklist_cache[short]
        tokens = frozenset()
        try:
            import h5py  # OPTIONAL: not installed in this environment
        except ImportError:
            if not self._warned_no_h5py:
                logging.warning(
                    "Waymo: use_blacklist=True but h5py is not installed; "
                    "proceeding without the invalid_files.h5 frame blacklist"
                )
                self._warned_no_h5py = True
            self._blacklist_cache[short] = tokens
            return tokens
        try:
            with h5py.File(self._blacklist_path, "r") as f:
                seg_dir_base = os.path.basename(self._segment_dirs[short])
                key = next((k for k in (seg_dir_base, short) if k in f), None)
                if key is not None:
                    grp = f[key]
                    ds = grp["invalid_pairs"] if "invalid_pairs" in grp else grp
                    tokens = frozenset(
                        t.decode() if isinstance(t, bytes) else str(t)
                        for t in np.asarray(ds).ravel()
                    )
        except Exception as exc:  # blacklist is best-effort; never block loading
            logging.warning("Waymo: failed to read blacklist for %s: %s", short, exc)
        self._blacklist_cache[short] = tokens
        return tokens

    def _frames(self, seq_name: str) -> list:
        """Frame tuples ``(jpg_path, exr_path, npz_path, frame_num)`` for one
        sequence, lazily enumerated and cached. Raises ValueError only if the
        on-disk listing itself has fewer than ``min_num_images`` frames (lazy
        counterpart of the eager vendors' construction-time skip; never seen on
        this copy -- every camera has >=171 frames). If instead the BLACKLIST
        filter would cause the shortfall (real blacklists contain whole-camera
        runs), the blacklist is ignored for this sequence with a warning rather
        than raising: a lazy raise here would kill a DataLoader worker
        mid-training and crash ComposedDataset sequence enumeration."""
        if seq_name in self._frames_cache:
            return self._frames_cache[seq_name]
        short, cam = self.data_store[seq_name]
        seg_dir = self._segment_dirs[short]
        frame_nums = self._segment_listing(short).get(cam, [])
        frames = [
            (
                os.path.join(seg_dir, f"{n:05d}_{cam}.jpg"),
                os.path.join(seg_dir, f"{n:05d}_{cam}.exr"),
                os.path.join(seg_dir, f"{n:05d}_{cam}.npz"),
                n,
            )
            for n in frame_nums
        ]
        n_on_disk = len(frames)
        filtered = self.filter_blacklisted_frames(frames, cam, self._segment_blacklist(short))
        if len(filtered) < self.min_num_images <= len(frames):
            # The blacklist (a quality flag; blacklisted frames decode fine)
            # would drop this sequence below min_num_images -- e.g. real
            # blacklists flag ~195/199 frames of a whole camera. Fall back to
            # the unfiltered frames with a warning instead of raising, because
            # sequence_list is already published (raising lazily would kill a
            # DataLoader worker mid-training and break ComposedDataset
            # sequence enumeration).
            logging.warning(
                "Waymo sequence %s: blacklist would leave only %d of %d frames "
                "(< min_num_images=%d); ignoring the blacklist for this sequence",
                seq_name, len(filtered), len(frames), self.min_num_images,
            )
        else:
            frames = filtered
        if len(frames) < self.min_num_images:
            # Only reachable when the ON-DISK listing itself is short (a
            # blacklist-caused shortfall fell back above), so report n_on_disk.
            raise ValueError(
                f"Waymo sequence {seq_name}: only {n_on_disk} frames on disk "
                f"(< min_num_images={self.min_num_images}); segments are expected "
                "to have ~171-200 frames per camera -- exclude this sequence or "
                "lower min_num_images"
            )
        self._frames_cache[seq_name] = frames
        return frames

    def _segment_anchor(self, short: str) -> np.ndarray:
        """Per-segment recentering anchor: the camera position (``cam2world``
        translation) of the first frame of the lowest camera id present in the
        segment listing. Shared by all camera-sequences of the segment so they
        stay in one consistent recentered world frame."""
        if short not in self._anchor_cache:
            listing = self._segment_listing(short)
            if not listing:
                raise ValueError(f"Waymo segment {short}: no frames found on disk")
            cam = min(listing)
            frame_num = listing[cam][0]
            npz_path = os.path.join(self._segment_dirs[short], f"{frame_num:05d}_{cam}.npz")
            with np.load(npz_path) as d:
                c2w = np.asarray(d["cam2world"], dtype=np.float64)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                raise ValueError(f"Waymo: bad anchor cam2world in {npz_path}")
            self._anchor_cache[short] = c2w[:3, 3].copy()
        return self._anchor_cache[short]

    # --- BaseDataset contract ---------------------------------------------

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration; triggers the
        lazy per-segment listing on first access)."""
        return len(self._frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's jpg header and cached
        (per-camera: cams 1-3 are 341x512, cams 4-5 are 236x512)."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            jpg_path = self._frames(name)[0][0]
            with Image.open(jpg_path) as im:
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
        frames = self._frames(seq_name)
        short, cam = self.data_store[seq_name]
        anchor = self._segment_anchor(short)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        camera_ids, original_sizes = [], []

        for i in ids:
            jpg_path, exr_path, npz_path, _frame_num = frames[int(i)]
            image = read_image_cv2(jpg_path)
            if image is None:
                # Enumerated from the segment listing, so the file should exist;
                # fail loudly (a silent skip would yield fewer than img_per_seq
                # frames and break fixed-V batch stacking).
                raise FileNotFoundError(f"Waymo: could not read image {jpg_path}")
            depth_map = self.read_waymo_depth(exr_path)
            if depth_map.shape != image.shape[:2]:
                raise ValueError(
                    f"Waymo: depth {depth_map.shape} does not match image "
                    f"{image.shape[:2]} for {exr_path}"
                )
            with np.load(npz_path) as d:
                K = self.waymo_intrinsics(d["intrinsics"], self.intrinsics_override)
                pose_w2c = self.waymo_pose_to_w2c(d["cam2world"], anchor=anchor)
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
                filepath=jpg_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            camera_ids.append(cam)
            original_sizes.append(original_size)

        return {
            "seq_name": "waymo_" + seq_name,
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
            # would be wrong GT for outdoor driving data (sky lives in depth==0).
            "camera_ids": np.array(camera_ids, dtype=np.int32),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
