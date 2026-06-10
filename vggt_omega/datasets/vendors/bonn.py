"""Bonn RGB-D Dynamic vendor for the VGGT-Omega dataset API.

Bonn RGB-D Dynamic (Palazzolo et al., IROS 2019) is a TUM-format indoor RGB-D
benchmark of 26 mostly-dynamic sequences (people, balloons, boxes), nested one
level below the dataset root::

    BONN_DIR/rgbd_bonn_dataset/rgbd_bonn_<scene>/
        rgb/<unix_ts>.png       RGB, 640x480 uint8
        depth/<unix_ts>.png     16-bit depth registered to color, meters = value/5000
        rgb.txt, depth.txt      TUM index files ("timestamp path", 2 comment lines)
        groundtruth.txt         TUM trajectory "ts tx ty tz qx qy qz qw" (100 Hz mocap)
        rgb_110/, depth_110/,   pre-associated 110-frame eval subset (full-stream
        groundtruth_110.txt     frames 30..139; groundtruth_110.txt has NO header
                                and is numpy-savetxt scientific notation)

Conventions verified empirically on this data (see ``.dataset_surveys/bonn.json``):

* Depth: uint16 PNG, meters = value / 5000 (TUM scale, verified metrically against
  the mocap translations); 0 = invalid (~15% of pixels). Indoor only, so SKY_MASK
  is advertised and always all-False (tum/7scenes repo convention).
* Intrinsics: no calibration file ships on disk; the official published Bonn
  calibration fx=542.822841 fy=542.576870 cx=315.593520 cy=237.756098 (640x480)
  is the default, overridable via ``intrinsics=[fx, fy, cx, cy]`` (sequence names
  contain no freiburg key, so the TUM table never applies).
* POSE FRAME TRADEOFF (important): ``groundtruth.txt`` is camera-to-world of the
  **OptiTrack marker body**, NOT the color optical frame.

  - ``pose_frame="camera"`` (default) right-multiplies each marker c2w by the
    constant hand-eye transform ``MARKER_TO_CAM`` before inverting to w2c. This
    is the geometrically correct OpenCV camera pose: cross-frame depth
    reprojection closes to 0.6-3.6% median relative error (vs 3.5-81% with the
    raw marker poses). Required for any pixel-level use of the extrinsics
    (reprojection, world points, depth-consistency eval).
  - ``pose_frame="marker"`` returns the raw GT poses exactly as published.
    Published ATE protocols (MonST3R, streamVGGT) evaluate against these raw
    marker trajectories; use this mode for parity with published pose-ATE
    numbers. The two trajectories differ by a constant ~1.9 cm rig offset
    rotated through each frame's orientation, so on rotating sequences they
    genuinely diverge (and reprojection through marker poses fails).

* ``subset="110"`` (default) uses the rgb_110/depth_110/groundtruth_110.txt eval
  split: 110 consecutive frames per sequence, pre-associated index-aligned
  triplets, loaded exactly as published (the MonST3R/streamVGGT protocol). On 25
  of the 26 sequences the rgb-depth/rgb-gt timestamp skew is <= ~55 ms. The one
  outlier is rgbd_bonn_static: its groundtruth_110.txt rows sit a constant
  ~230-250 ms off the rgb_110 timestamps (depth_110 up to ~184 ms), so that
  sequence's _110 extrinsics close cross-frame depth reprojection only to ~8%
  median relative error (vs 0.6-3.6% elsewhere). Index-pairing the published
  files is the right parity behavior, but for pixel-level use of
  rgbd_bonn_static's extrinsics prefer ``subset="full"``, whose 20 ms
  nearest-timestamp gate re-associates that sequence accurately.
  ``subset="full"`` uses the full streams (332-10916 frames) with TUM-style
  nearest-timestamp association, performed lazily per sequence on first access
  and cached per process: each DataLoader worker pays it once per sequence,
  <1 s even for the ~11k-frame rgbd_bonn_static (windowed matching plus one
  listdir per depth dir; see :meth:`BonnDataset._associate_windowed`). Several
  sequences list PNGs in rgb.txt/depth.txt that are missing on disk
  (rgbd_bonn_static: 10916 listed vs 10000 depth PNGs) -- those frames are
  skipped with a warning.
* Timestamps are real unix capture times (PNG filenames / index files), emitted
  per frame as TIMESTAMP.

Frame tuples have the exact TUM layout ``(rgb_path, depth_path, w2c, ts)``, so
this vendor subclasses :class:`TumDataset` and reuses its depth reader,
``get_data`` loop, and geometry plumbing wholesale.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.modality import Modality
from vggt_omega.datasets.vendors.common import (
    quat_to_rotation,
    read_file_list,
)
from vggt_omega.datasets.vendors.tum import TumDataset


class _LazySequenceStore(dict):
    """Mapping ``seq_name -> frames`` that builds (and caches) each sequence's
    frame list on first access. Used for ``subset="full"``, where eagerly
    associating every sequence at construction would be needless work (~6.5 s
    for all 26 sequences). The cache is per process: every DataLoader worker
    re-pays the association on its first access to a sequence, <1 s per
    sequence even at ~11k frames (windowed matching + one listdir per depth
    dir -- see :meth:`BonnDataset._associate_windowed`)."""

    def __init__(self, loader):
        super().__init__()
        self._loader = loader

    def __missing__(self, key):
        frames = self._loader(key)
        self[key] = frames
        return frames


class BonnDataset(TumDataset):
    """Bonn RGB-D Dynamic as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Official published Bonn pinhole calibration (fx, fy, cx, cy) for the native
    # 640x480 frame (no calibration file ships with the dataset).
    _BONN_INTRINSICS = (542.822841, 542.576870, 315.593520, 237.756098)

    # Constant OptiTrack-marker-body -> color-camera (OpenCV) transform X, so that
    # c2w_camera = c2w_marker @ X. Recovered by Tsai-Lenz AX=XB hand-eye
    # calibration on this on-disk data (60 motion pairs from rgbd_bonn_static +
    # rgbd_bonn_static_close_far; residuals: rotation median ~0.63 deg,
    # translation median ~1.6 cm). With X applied, cross-frame depth reprojection
    # closes to 0.6-3.6% median relative error on held-out pairs (incl. dynamic
    # balloon/crowd scenes) vs 3.5-81% with the raw marker poses. The transform is
    # a ~172 deg rotation with a ~1.9 cm offset. See .dataset_surveys/bonn.json.
    MARKER_TO_CAM = np.array(
        [
            [-0.96509344, -0.17127006, 0.19814445, 0.00130562],
            [-0.26129314, 0.57791660, -0.77313537, -0.00304165],
            [0.01790398, -0.79792166, -0.60249521, -0.01835103],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Bonn provides RGB + registered metric depth + GT trajectories/intrinsics --
    # the same modality set as TUM. As there, WORLD_POINTS / CAM_POINTS are only
    # the depth re-projected through the GT poses (not an independent point-cloud
    # GT), so they are NOT advertised as evaluable GT modalities. Indoor data:
    # SKY_MASK is advertised and always all-False (repo convention).
    AVAILABLE = TumDataset.AVAILABLE

    @staticmethod
    def _frame_timestamp(path: str) -> float:
        """Unix timestamp encoded in a Bonn frame filename (``<ts>.png``)."""
        return float(os.path.splitext(os.path.basename(path))[0])

    @classmethod
    def bonn_intrinsics(cls, override=None) -> np.ndarray:
        """(3,3) pinhole K for Bonn's single 640x480 camera.

        ``override``=[fx, fy, cx, cy] wins; otherwise the official published Bonn
        calibration is used (the dataset ships no calibration file).
        """
        if override is None:
            override = cls._BONN_INTRINSICS
        return cls.tum_intrinsics("bonn", override=override)

    @classmethod
    def bonn_pose_to_w2c(cls, t, q, pose_frame: str = "camera") -> np.ndarray:
        """Bonn GT pose (translation ``t``, quaternion ``q``=(qx,qy,qz,qw);
        camera-to-world of the OptiTrack MARKER BODY) -> world-to-camera (3,4)
        float32 OpenCV.

        ``pose_frame="camera"`` right-multiplies the marker c2w by
        :attr:`MARKER_TO_CAM` before inverting, yielding the reproject-consistent
        color-camera pose. ``pose_frame="marker"`` inverts the raw GT pose
        (parity with published ATE protocols).
        """
        if pose_frame not in ("camera", "marker"):
            raise ValueError(f"pose_frame must be 'camera' or 'marker', got {pose_frame!r}")
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_to_rotation(q)
        c2w[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        if pose_frame == "camera":
            c2w = c2w @ cls.MARKER_TO_CAM
        rot_w2c = c2w[:3, :3].T
        trans_w2c = -rot_w2c @ c2w[:3, 3]
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @classmethod
    def load_110_sequence(cls, seq_dir: str, pose_frame: str = "camera") -> list:
        """Load the pre-associated 110-frame eval subset of one sequence dir.

        ``rgb_110/``, ``depth_110/`` and ``groundtruth_110.txt`` are index-aligned
        triplets (verified across all 26 sequences), so no timestamp association
        is needed -- files are sorted numerically by their timestamp filenames and
        zipped with the GT rows. ``groundtruth_110.txt`` has no header and is
        numpy scientific notation. Returns ``[(rgb_path, depth_path, w2c, ts)]``
        (the TUM frame-tuple layout). Raises ValueError if the triplet counts
        disagree.
        """
        rgb_files = sorted(
            glob.glob(os.path.join(seq_dir, "rgb_110", "*.png")), key=cls._frame_timestamp
        )
        depth_files = sorted(
            glob.glob(os.path.join(seq_dir, "depth_110", "*.png")), key=cls._frame_timestamp
        )
        gt = np.atleast_2d(np.loadtxt(os.path.join(seq_dir, "groundtruth_110.txt")))
        if not (len(rgb_files) == len(depth_files) == len(gt)) or gt.shape[1] != 8:
            raise ValueError(
                f"Bonn {seq_dir}: _110 subset not index-aligned "
                f"({len(rgb_files)} rgb, {len(depth_files)} depth, gt shape {gt.shape})"
            )
        frames = []
        for rgb_path, depth_path, row in zip(rgb_files, depth_files, gt):
            w2c = cls.bonn_pose_to_w2c(row[1:4], row[4:8], pose_frame=pose_frame)
            frames.append((rgb_path, depth_path, w2c, cls._frame_timestamp(rgb_path)))
        return frames

    @staticmethod
    def _count_index_entries(path: str) -> int:
        """Number of non-comment, non-empty lines in a TUM index file -- a cheap
        upper bound on a sequence's frame count, used to pre-filter sequences in
        ``subset="full"`` mode without paying for association at construction."""
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    n += 1
        return n

    @staticmethod
    def _associate_windowed(first, second, max_diff: float) -> list:
        """Greedy nearest-timestamp matching, output-identical to TUM
        ``common.associate`` (same candidate set, same best-pair-first greedy
        resolution, same sorted ``[(t1, t2)]`` result) but with the candidate
        pairs found by a sorted-window scan instead of the all-pairs sweep.

        The shared helper is O(n*m): on rgbd_bonn_static's full streams
        (10916 x 10906 timestamps) it takes ~100 s, which a single lazy access
        in ``subset="full"`` would pay per DataLoader worker. This windowed
        version is O((n+m) log m + k log k) and matches the same streams in
        well under a second.
        """
        second_arr = np.array(sorted(second), dtype=np.float64)
        first_arr = np.asarray(list(first), dtype=np.float64)
        lo = np.searchsorted(second_arr, first_arr - max_diff, side="left")
        hi = np.searchsorted(second_arr, first_arr + max_diff, side="right")
        potential = []
        for a, l, h in zip(first_arr.tolist(), lo.tolist(), hi.tolist()):
            for b in second_arr[l:h].tolist():
                if abs(a - b) < max_diff:  # strict <, exactly as the reference
                    potential.append((abs(a - b), a, b))
        potential.sort()
        remaining_first, remaining_second = set(first), set(second)
        matches = []
        for _, a, b in potential:
            if a in remaining_first and b in remaining_second:
                remaining_first.remove(a)
                remaining_second.remove(b)
                matches.append((a, b))
        matches.sort()
        return matches

    def _associate_full_sequence(self, seq_name: str) -> list:
        """Associate one sequence's full rgb/depth/groundtruth streams
        (TUM-style nearest-timestamp matching, via the windowed equivalent of
        the shared helper), skipping frames whose PNGs are missing on disk
        (several sequences list more files than exist, e.g. rgbd_bonn_static:
        10916 in depth.txt vs 10000 PNGs).

        Called lazily by the data_store on first access per sequence; the
        corrected/raw pose (per ``self.pose_frame``) is baked into the frames.
        Returns ``[(rgb_path, depth_path, w2c, ts)]``.
        """
        if seq_name not in self._seq_dirs:
            raise KeyError(seq_name)
        seq_dir = self._seq_dirs[seq_name]
        rgb = read_file_list(os.path.join(seq_dir, "rgb.txt"))
        depth = read_file_list(os.path.join(seq_dir, "depth.txt"))
        gt = read_file_list(os.path.join(seq_dir, "groundtruth.txt"))
        gt_ts = np.array(sorted(gt))

        # Missing-file check via ONE directory listing per depth dir instead of
        # a stat call per frame: on network storage (/jfs) ~11k individual
        # os.path.exists calls cost ~90 s; a single listdir is near-instant and
        # gives the same answer for regular files.
        listing_cache: dict[str, set] = {}

        def _on_disk(path: str) -> bool:
            d, base = os.path.split(path)
            if d not in listing_cache:
                listing_cache[d] = set(os.listdir(d)) if os.path.isdir(d) else set()
            return base in listing_cache[d]

        frames, n_missing = [], 0
        for t_rgb, t_dep in self._associate_windowed(
            list(rgb), list(depth), self.assoc_max_diff
        ):
            # Nearest GT timestamp, vectorized: the mocap runs at 100 Hz (up to
            # ~36k entries), so a per-frame linear min() scan would dominate.
            j = np.searchsorted(gt_ts, t_rgb)
            cand = gt_ts[max(j - 1, 0) : j + 1]
            t_gt = float(cand[np.argmin(np.abs(cand - t_rgb))])
            if abs(t_gt - t_rgb) > self.assoc_max_diff:
                continue
            rgb_path = os.path.join(seq_dir, rgb[t_rgb][0])
            depth_path = os.path.join(seq_dir, depth[t_dep][0])
            # Only depth is checked: depth/ is the truncated stream on disk (rgb
            # counts match rgb.txt exactly). An unreadable rgb file still raises
            # loudly in get_data.
            if not _on_disk(depth_path):
                n_missing += 1
                continue
            vals = [float(v) for v in gt[t_gt]]
            w2c = self.bonn_pose_to_w2c(vals[0:3], vals[3:7], pose_frame=self.pose_frame)
            frames.append((rgb_path, depth_path, w2c, t_rgb))
        if n_missing:
            logging.warning(
                "Bonn %s: skipped %d associated frames whose PNGs are missing on disk",
                seq_name, n_missing,
            )
        if len(frames) < self.min_num_images:
            # The construction-time filter only sees the rgb.txt upper bound, so
            # enforce the real minimum here (a short batch would break stacking).
            raise ValueError(
                f"Bonn {seq_name}: only {len(frames)} associated frames "
                f"(< min_num_images={self.min_num_images})"
            )
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        BONN_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        subset: str = "110",
        pose_frame: str = "camera",
        assoc_max_diff: float = 0.02,
        depth_scale: float = 5000.0,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        # Intentionally bypass TumDataset.__init__ (it requires TUM_DIR and
        # eagerly associates TUM-layout sequences); initialize BaseDataset
        # directly and do the Bonn-specific discovery here.
        BaseDataset.__init__(self, common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if BONN_DIR is None:
            raise ValueError("BONN_DIR must be specified")
        if subset not in ("110", "full"):
            raise ValueError(f"subset must be '110' or 'full', got {subset!r}")
        if pose_frame not in ("camera", "marker"):
            raise ValueError(f"pose_frame must be 'camera' or 'marker', got {pose_frame!r}")

        self.BONN_DIR = BONN_DIR
        self.subset = subset
        self.pose_frame = pose_frame
        self.expand_ratio = expand_ratio
        self.assoc_max_diff = assoc_max_diff
        self.depth_scale = depth_scale
        # Always hand the inherited get_data an explicit override: Bonn sequence
        # names contain no freiburg key, so the TUM intrinsics table never applies.
        self.intrinsics_override = (
            list(intrinsics) if intrinsics is not None else list(self._BONN_INTRINSICS)
        )
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # The official archive nests sequences one level down; also accept
        # BONN_DIR pointing directly at the sequence parent.
        nested = os.path.join(BONN_DIR, "rgbd_bonn_dataset")
        seq_root = nested if os.path.isdir(nested) else BONN_DIR
        patterns = sequences or ["*"]
        seq_dirs = sorted(
            {
                d
                for pat in patterns
                for d in glob.glob(os.path.join(seq_root, pat))
                if os.path.isdir(d)
            }
        )

        self._seq_dirs = {}
        if subset == "110":
            self.data_store = {}
            for sd in seq_dirs:
                name = os.path.basename(sd.rstrip("/"))
                if not os.path.isfile(os.path.join(sd, "groundtruth_110.txt")):
                    logging.warning(
                        "Bonn %s: no _110 eval subset; skipping (use subset='full')", name
                    )
                    continue
                frames = self.load_110_sequence(sd, pose_frame=pose_frame)
                if len(frames) < min_num_images:
                    logging.warning(
                        "Bonn %s: only %d _110 frames (< %d); skipping",
                        name, len(frames), min_num_images,
                    )
                    continue
                self._seq_dirs[name] = sd
                self.data_store[name] = frames
        else:
            # Lazy full mode: association is deferred to first access per
            # sequence; construction only counts rgb.txt entries (an upper bound
            # on the associated frame count) to filter degenerate sequences.
            self.data_store = _LazySequenceStore(self._associate_full_sequence)
            for sd in seq_dirs:
                name = os.path.basename(sd.rstrip("/"))
                index = os.path.join(sd, "rgb.txt")
                if not os.path.isfile(index):
                    logging.warning("Bonn %s: no rgb.txt; skipping", name)
                    continue
                n_listed = self._count_index_entries(index)
                if n_listed < min_num_images:
                    logging.warning(
                        "Bonn %s: only %d frames listed (< %d); skipping",
                        name, n_listed, min_num_images,
                    )
                    continue
                self._seq_dirs[name] = sd

        self.sequence_list = sorted(self._seq_dirs)
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Bonn sequences under {BONN_DIR} "
                f"(subset={subset!r}, sequences={patterns})"
            )
        self._native_size_cache = {}

    # sequence_num_frames / native_image_size are inherited from TumDataset and
    # work unchanged on this vendor's data_store (lazy stores build on access).

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """Identical to :meth:`TumDataset.get_data` (the frame tuples share the
        TUM layout and the intrinsics override is always set); only the vendor
        prefix on ``seq_name`` differs."""
        batch = super().get_data(
            seq_index=seq_index,
            img_per_seq=img_per_seq,
            seq_name=seq_name,
            ids=ids,
            aspect_ratio=aspect_ratio,
        )
        batch["seq_name"] = "bonn_" + batch["seq_name"].removeprefix("tum_")
        return batch
