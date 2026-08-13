"""Conformance + correctness tests for :class:`BaseSequence` vendors.

The shared interface laws live in :class:`BaseSequenceContract`. A vendor test is
just a subclass that implements :meth:`make_sequence` (returning a ready
``BaseSequence`` over a tiny synthetic on-disk sequence, or a real eval dataset
that's skipped when absent); every contract method then runs against it::

    class TestMyVendor(BaseSequenceContract):
        def make_sequence(self):
            ...
            return MyVendorSequence(root, seq_id)

``BaseSequenceContract`` deliberately does **not** start with ``Test`` so pytest
never collects it on its own — only concrete vendor subclasses run. The contract
tests *invariants* (shapes, dtypes, the c2w pose math, intrinsic scaling), not the
exact Python wrapper a vendor returns, so a vendor may expose poses as raw ``(4, 4)``
matrices or as (single-frame) ``NumpySE3Trajectory`` slices and still conform.

Vendor-specific decode correctness (depth scale, intrinsic values, pose convention)
belongs in the subclass as extra ``test_*`` methods — see :class:`TestTumSequence`.
"""
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from vggt_omega.datasets.sequences.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.sequences.tum import TumSequence
from vggt_omega.datasets.utils.se3_trajectory import NumpySE3Trajectory

# --------------------------------------------------------------------------- #
# shared constants / helpers
# --------------------------------------------------------------------------- #
H, W = 12, 16  # synthetic frame size
_EVAL = "/jfs/guibiao/streamVGGT/data/eval"  # optional real eval root (skipped if absent)


def _pose_to_matrix(pose) -> np.ndarray:
    """Coerce a single-frame ``get_pose`` result to a ``(4, 4)`` c2w matrix.

    Accepts a single-frame ``SE3Trajectory`` (``transform_matrix`` is a method
    returning ``(1, 4, 4)``), a value object exposing a ``transform_matrix``
    property, or a raw ``(4, 4)`` ndarray — so the contract is agnostic to the
    representation a vendor returns."""
    tm = getattr(pose, "transform_matrix", None)
    M = np.asarray(tm() if callable(tm) else tm if tm is not None else pose, dtype=np.float64)
    if M.ndim == 3 and M.shape[0] == 1:  # single-frame trajectory -> drop batch axis
        M = M[0]
    return M


# --------------------------------------------------------------------------- #
# the shared contract
# --------------------------------------------------------------------------- #
class BaseSequenceContract:
    """Interface laws every ``BaseSequence`` vendor must satisfy.

    Subclass and implement :meth:`make_sequence`. Each ``test_*`` below is skipped
    automatically for sensors that don't advertise the relevant modality, so the
    same suite fits an RGB-only, an RGB-D, or a posed multi-sensor vendor.
    """

    def make_sequence(self) -> BaseSequence:
        raise NotImplementedError

    # -- small helpers (not collected; no test_ prefix) --------------------- #
    def _primary_sensor(self, s):
        return s.get_sensors()[0]

    def _frame_names(self, s, sid):
        """Public-ish list of frame names for a sensor (manifest-backed)."""
        return list(s._manifest[sid][Modality.NAME])

    # -- identity / discovery ----------------------------------------------- #
    def test_is_base_sequence(self):
        assert isinstance(self.make_sequence(), BaseSequence)

    def test_sensors_nonempty(self):
        assert len(self.make_sequence().get_sensors()) >= 1

    def test_length_positive(self):
        s = self.make_sequence()
        for sid in s.get_sensors():
            assert s.get_length(sid) > 0

    def test_modalities_subset_of_known(self):
        s = self.make_sequence()
        for sid in s.get_sensors():
            assert s.get_modalities(sid) <= set(Modality)

    def test_frame_names_match_length(self):
        s = self.make_sequence()
        for sid in s.get_sensors():
            assert len(self._frame_names(s, sid)) == s.get_length(sid)

    def test_get_frame_indices_agrees_with_singular(self):
        # The bulk name->index find/map must match the singular lookup elementwise
        # and follow the *query* order (here reversed to exercise ordering).
        s = self.make_sequence()
        for sid in s.get_sensors():
            names = self._frame_names(s, sid)
            subset = list(reversed(names))[: max(1, len(names) // 2)]
            assert s.get_frame_indices(sid, subset) == [
                s.get_frame_index(sid, n) for n in subset
            ]

    # -- per-frame getters honour advertised modalities --------------------- #
    def test_rgb_shape(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.RGB not in s.get_modalities(sid):
            pytest.skip("sensor has no RGB")
        rgb = s.get_rgb(sid, 0)
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert rgb.dtype == np.uint8

    def test_depth_shape_aligns_to_rgb(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.DEPTH not in s.get_modalities(sid):
            pytest.skip("sensor has no DEPTH")
        depth = s.get_depth(sid, 0)
        assert depth.ndim == 2
        if Modality.RGB in s.get_modalities(sid):
            assert depth.shape == s.get_rgb(sid, 0).shape[:2]

    def test_pose_is_valid_c2w(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.POSE not in s.get_modalities(sid):
            pytest.skip("sensor has no POSE")
        name0 = self._frame_names(s, sid)[0]
        M = _pose_to_matrix(s.get_pose(sid, name0))
        assert M.shape == (4, 4)
        np.testing.assert_allclose(M[3], [0, 0, 0, 1], atol=1e-6)
        R = M[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_get_poses_matches_get_pose(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.POSE not in s.get_modalities(sid):
            pytest.skip("sensor has no POSE")
        names = self._frame_names(s, sid)
        poses = s.get_poses(sid)
        assert len(poses) == len(names)
        for i in (0, len(names) // 2, len(names) - 1):
            np.testing.assert_allclose(
                _pose_to_matrix(poses[i]),
                _pose_to_matrix(s.get_pose(sid, names[i])),
                atol=1e-9,
            )

    def test_intrinsic_is_fxfycxcy(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.INTRINSIC not in s.get_modalities(sid):
            pytest.skip("sensor has no INTRINSIC")
        K = np.asarray(s.get_intrinsic(sid))
        assert K.shape == (4,)  # [fx, fy, cx, cy]
        assert K[0] > 0 and K[1] > 0  # positive focal lengths

    def test_scaled_intrinsic_rescales_per_axis(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.INTRINSIC not in s.get_modalities(sid):
            pytest.skip("sensor has no INTRINSIC")
        K = np.asarray(s.get_intrinsic(sid), dtype=np.float64)
        # native (no image_size) returns the calibration unchanged
        np.testing.assert_allclose(s.scaled_intrinsic(sid), K, atol=1e-4)
        h0, w0 = s.get_rgb_size(sid)
        th, tw = h0 // 2, w0 // 3  # non-uniform so the per-axis mapping is exercised
        sx, sy = tw / w0, th / h0
        Kr = np.asarray(s.scaled_intrinsic(sid, image_size=(th, tw)), dtype=np.float64)
        # fx, cx scale with width; fy, cy scale with height
        np.testing.assert_allclose(
            Kr, [K[0] * sx, K[1] * sy, K[2] * sx, K[3] * sy], rtol=1e-5
        )

    def test_extrinsic_self_is_identity(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.EXTRINSIC not in s.get_modalities(sid):
            pytest.skip("sensor has no EXTRINSIC")
        E = np.asarray(s.get_extrinsic(sid, sid), dtype=np.float64)
        assert E.shape == (4, 4)
        np.testing.assert_allclose(E, np.eye(4), atol=1e-9)

    def test_timestamp_type_and_monotonic(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.TIMESTAMP not in s.get_modalities(sid):
            pytest.skip("sensor has no TIMESTAMP")
        names = self._frame_names(s, sid)[:50]
        ts = [s.get_timestamp(sid, n) for n in names]
        assert all(isinstance(t, float) for t in ts)
        assert ts == sorted(ts)

    def test_valid_mask_if_advertised(self):
        s = self.make_sequence()
        sid = self._primary_sensor(s)
        if Modality.RGB_VALID_MASK not in s.get_modalities(sid):
            pytest.skip("sensor advertises no valid mask")
        m = s.get_rgb_valid_mask(sid, 0)
        assert m.dtype == bool and m.ndim == 2
        if Modality.RGB in s.get_modalities(sid):
            assert m.shape == s.get_rgb(sid, 0).shape[:2]


# =========================================================================== #
# TUM RGB-D
# =========================================================================== #
def _write_tum_sequence(
    root,
    seq_id="rgbd_dataset_freiburg1_xyz",
    n=4,
    depth_offset=0.001,
    gt_lo_pad=0.01,
    gt_hi_pad=0.01,
    reverse_gt=False,
):
    """Write a tiny TUM-layout sequence under ``root`` and return its directory.

    RGB/depth share a frame index (depth filename offset ``depth_offset`` s, below
    the default ``rgbd_sync_diff``); groundtruth is sampled densely from
    ``t0 - gt_lo_pad`` to ``t_last + gt_hi_pad`` (so by default it spans the color
    clock and interpolation never extrapolates). Camera is at world ``(k, 0, 0)``
    with identity rotation, so frame ``k``'s c2w translation is exactly ``[k, 0, 0]``.

    Edge-path knobs: ``depth_offset`` > rgbd_sync_diff trips the sync assertion;
    a negative ``gt_lo_pad`` leaves early frames outside the groundtruth span;
    ``reverse_gt`` writes the groundtruth rows in descending time (tests the sort).
    """
    d = os.path.join(root, seq_id)
    os.makedirs(os.path.join(d, "rgb"))
    os.makedirs(os.path.join(d, "depth"))
    t0, dt_frame = 1305031910.0, 0.05
    rgb_lines, depth_lines = [], []
    for k in range(n):
        t = t0 + k * dt_frame
        Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).save(
            os.path.join(d, "rgb", f"{t:.6f}.png")
        )
        td = t + depth_offset
        Image.fromarray(np.full((H, W), (k + 1) * 1000, np.uint16)).save(
            os.path.join(d, "depth", f"{td:.6f}.png")
        )
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_lines.append(f"{td:.6f} depth/{td:.6f}.png")
    gt_lines, g, end = [], t0 - gt_lo_pad, t0 + dt_frame * (n - 1) + gt_hi_pad
    while g <= end:
        x = (g - t0) / dt_frame
        gt_lines.append(f"{g:.6f} {x:.6f} 0 0 0 0 0 1")  # ts tx ty tz qx qy qz qw
        g += 0.01
    if reverse_gt:
        gt_lines = gt_lines[::-1]
    open(os.path.join(d, "rgb.txt"), "w").write("# color\n" + "\n".join(rgb_lines) + "\n")
    open(os.path.join(d, "depth.txt"), "w").write("# depth\n" + "\n".join(depth_lines) + "\n")
    open(os.path.join(d, "groundtruth.txt"), "w").write("# gt\n" + "\n".join(gt_lines) + "\n")
    return d


class TestTumSequence(BaseSequenceContract):
    """TUM RGB-D conformance + vendor-specific decode checks."""

    SEQ_ID = "rgbd_dataset_freiburg1_xyz"

    def make_sequence(self) -> TumSequence:
        root = tempfile.mkdtemp()
        _write_tum_sequence(root, self.SEQ_ID, n=4)
        return TumSequence(root, self.SEQ_ID)

    # ----- TUM-specific correctness ---------------------------------------- #
    def test_depth_scale_metres(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        # frame 0 depth PNG holds counts == 1000; metres = 1000 / 5000 = 0.2.
        assert np.allclose(s.get_depth(sid, 0), 0.2)

    def test_intrinsic_matches_freiburg1(self):
        s = self.make_sequence()
        K = np.asarray(s.get_intrinsic(s.get_sensors()[0]))
        np.testing.assert_allclose(
            K, [517.306408, 516.469215, 318.643040, 255.313989], rtol=1e-5
        )

    def test_pose_translation_follows_groundtruth(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        # camera at world (k, 0, 0): frame k's c2w translation is [k, 0, 0].
        for k in (0, 2, 3):
            M = _pose_to_matrix(s.get_pose(sid, k))
            np.testing.assert_allclose(M[:3, 3], [k, 0, 0], atol=1e-4)

    def test_discover_finds_sequence(self):
        s = self.make_sequence()
        assert self.SEQ_ID in TumSequence.discover(s._data_root)

    # ----- pose representation (trajectory, not a pose value object) ------- #
    def test_get_pose_is_single_frame_trajectory(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        p = s.get_pose(sid, 1)
        assert isinstance(p, NumpySE3Trajectory) and len(p) == 1
        # the sampler surface must work directly on the returned slice
        twist = p.boxminus(s.get_pose(sid, 0))  # frame 1 - frame 0, translation +1
        assert np.isclose(np.linalg.norm(np.asarray(twist)), 1.0, atol=1e-4)
        assert isinstance(p.inverse(), NumpySE3Trajectory)

    def test_get_poses_is_full_trajectory(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        traj = s.get_poses(sid)
        assert isinstance(traj, NumpySE3Trajectory)
        assert len(traj) == s.get_length(sid)

    # ----- "TUM provides no X" getters all raise --------------------------- #
    def test_unsupported_getters_raise(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        with pytest.raises(NotImplementedError):
            s.get_rgb_semantic_mask(sid, 0)
        with pytest.raises(NotImplementedError):
            s.get_rgb_dynamic_mask(sid, 0)
        with pytest.raises(NotImplementedError):
            s.get_rgb_valid_mask(sid, 0)
        with pytest.raises(NotImplementedError):
            s.get_depth_confidence(sid, 0)
        with pytest.raises(NotImplementedError):
            s.get_tracks(sid)
        with pytest.raises(NotImplementedError):
            s.get_pointcloud(sid)

    # ----- calibration: per-camera intrinsics + unknown camera ------------- #
    def test_intrinsic_selected_per_freiburg_camera(self):
        for cam, expected in TumSequence._TUM_INTRINSICS.items():
            root = tempfile.mkdtemp()
            seq_id = f"rgbd_dataset_{cam}_demo"
            _write_tum_sequence(root, seq_id, n=3)
            s = TumSequence(root, seq_id)
            np.testing.assert_allclose(
                np.asarray(s.get_intrinsic(s.get_sensors()[0])), expected, rtol=1e-5
            )

    def test_unknown_camera_raises(self):
        root = tempfile.mkdtemp()
        seq_id = "rgbd_dataset_unknown_demo"  # no freiburg1/2/3 token
        _write_tum_sequence(root, seq_id, n=3)
        with pytest.raises(ValueError):
            TumSequence(root, seq_id)

    # ----- construction-time guards / robustness --------------------------- #
    def test_rgb_depth_sync_drops_unsynced_frames(self):
        # Real TUM rgb/depth run on different clocks at different frame counts, so
        # load_manifest pairs each color frame with its NEAREST depth and *drops*
        # any color frame with no depth within rgbd_sync_diff (it does not raise --
        # the zip+assert variant crashed on real data; see commit 25e2c54). Here the
        # 0.05 s depth offset (>> 5 ms) leaves the first color frame unmatched.
        root = tempfile.mkdtemp()
        _write_tum_sequence(root, self.SEQ_ID, n=3, depth_offset=0.05)
        seq = TumSequence(root, self.SEQ_ID)
        sensor = seq.get_sensors()[0]
        # unsynced frame(s) dropped (< n), synced ones survive (> 0): no crash.
        assert 0 < seq.get_length(sensor) < 3

    def test_groundtruth_out_of_span_color_frames_dropped(self):
        # groundtruth starts 0.10 s after the first color frame, so the early color
        # frames fall outside the GT time span. load_manifest drops them (rather than
        # raising) so the pose interpolation in load_sensor_poses never extrapolates;
        # the in-span frames are kept and each gets an interpolated pose.
        root = tempfile.mkdtemp()
        _write_tum_sequence(root, self.SEQ_ID, n=4, gt_lo_pad=-0.10)
        seq = TumSequence(root, self.SEQ_ID)
        sensor = seq.get_sensors()[0]
        kept = seq.get_length(sensor)
        assert 0 < kept < 4  # pre-span frames dropped, in-span frames kept
        # every kept frame has an interpolated c2w pose (poses align 1:1 with frames).
        assert len(seq.get_poses(sensor)) == kept

    def test_unsorted_groundtruth_is_handled(self):
        # groundtruth written in descending time must still interpolate correctly.
        root = tempfile.mkdtemp()
        _write_tum_sequence(root, self.SEQ_ID, n=4, reverse_gt=True)
        s = TumSequence(root, self.SEQ_ID)
        sid = s.get_sensors()[0]
        for k in (0, 3):
            np.testing.assert_allclose(
                _pose_to_matrix(s.get_pose(sid, k))[:3, 3], [k, 0, 0], atol=1e-4
            )


# =========================================================================== #
# frame-name -> frame-index find/map (get_frame_index / get_frame_indices)
# =========================================================================== #
class _NameMapSequence(BaseSequence):
    """Minimal vendor exposing only a NAME manifest.

    Frame names are deliberately *not* equal to their positions, so the
    name->index find/map is exercised in the general case (the singular
    ``.index()`` lookup conflates the two when names happen to be ``0..n-1``).
    Only ``_manifest[sid][Modality.NAME]`` is touched by the index lookups, so
    the on-disk ``BaseSequence.__init__`` and the decode getters are skipped.
    """

    def __init__(self, names, sensor_id=0):
        self._manifest = {sensor_id: {Modality.NAME: list(names)}}

    def load_manifest(self):
        raise NotImplementedError

    def load_calibration_tree(self):
        raise NotImplementedError

    def load_sensor_poses(self):
        raise NotImplementedError

    def get_modalities(self, sensor_id):
        raise NotImplementedError

    def get_length(self, sensor_id):
        raise NotImplementedError

    def get_rgb(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_rgb_semantic_mask(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_rgb_dynamic_mask(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_rgb_valid_mask(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_depth(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_depth_confidence(self, sensor_id, frame_name):
        raise NotImplementedError

    def get_tracks(self, sensor_id):
        raise NotImplementedError

    def get_pointcloud(self, sensor_id):
        raise NotImplementedError


class TestGetFrameIndices:
    """``get_frame_indices`` maps a subset of frame names to their positions in
    the sensor's frame-sorted NAME list, preserving the *subset* order."""

    NAMES = [10, 20, 30, 40, 50]  # non-contiguous: name != position

    def test_maps_subset_to_positions_in_subset_order(self):
        s = _NameMapSequence(self.NAMES)
        assert s.get_frame_indices(0, [30, 10, 50]) == [2, 0, 4]

    def test_preserves_repeated_names(self):
        s = _NameMapSequence(self.NAMES)
        assert s.get_frame_indices(0, [20, 20, 40]) == [1, 1, 3]

    def test_empty_subset_returns_empty(self):
        s = _NameMapSequence(self.NAMES)
        assert s.get_frame_indices(0, []) == []

    def test_full_list_maps_to_identity_range(self):
        s = _NameMapSequence(self.NAMES)
        assert s.get_frame_indices(0, self.NAMES) == list(range(len(self.NAMES)))

    def test_agrees_with_singular_get_frame_index(self):
        s = _NameMapSequence(self.NAMES)
        subset = [50, 30, 10, 40]
        assert s.get_frame_indices(0, subset) == [
            s.get_frame_index(0, n) for n in subset
        ]

    def test_unknown_name_raises(self):
        s = _NameMapSequence(self.NAMES)
        with pytest.raises(KeyError):
            s.get_frame_indices(0, [10, 999])


# =========================================================================== #
# index addressing: the decode surface addresses frames by POSITION, not name
# =========================================================================== #
class _FakeTraj:
    """Minimal pose-trajectory slice: wraps an ``(k, 4, 4)`` stack and returns
    it from ``transform_matrix()`` (the only method the base getters/parse use)."""

    def __init__(self, mats):
        self._mats = mats

    def transform_matrix(self):
        return self._mats


class _FakePoses:
    """Position-indexable pose store mirroring the BaseSE3Trajectory indexing the
    decode path relies on. ``[i]`` (int) -> single-frame slice; ``[idxs]``
    (list/array) -> multi-frame slice -- without the real SE(3) machinery."""

    def __init__(self, mats):
        self._mats = list(mats)  # list of (4, 4) arrays

    def __getitem__(self, key):
        idxs = [key] if isinstance(key, (int, np.integer)) else list(key)
        return _FakeTraj(np.stack([self._mats[i] for i in idxs], axis=0))


class _IndexAddrSequence(BaseSequence):
    """In-memory vendor whose frame NAMES are arbitrary strings (``"a"``, ``"b"``,
    ``"c"``) and therefore deliberately != their positions. Pins that the decode
    surface (get_pose / get_timestamp / parse) addresses frames by POSITION
    (frame_index): any name resolution on these paths would raise on a
    string-named, non-positional vendor.

    Bypasses ``BaseSequence.__init__`` (no disk): sets the manifest + poses
    directly. Only POSE + TIMESTAMP are advertised, so parse() never reaches the
    image getters."""

    _SENSOR = 0

    def __init__(self):
        mats = []
        for k in range(3):
            m = np.eye(4)
            m[0, 3] = float(k)  # translation x encodes the position
            mats.append(m)
        self._manifest = {
            self._SENSOR: {
                Modality.NAME: ["a", "b", "c"],            # names != positions, not int
                Modality.TIMESTAMP: [100.0, 200.0, 300.0],
            }
        }
        self._sensor_poses = {self._SENSOR: _FakePoses(mats)}

    def load_manifest(self):
        raise NotImplementedError

    def load_calibration_tree(self):
        raise NotImplementedError

    def load_sensor_poses(self):
        raise NotImplementedError

    def get_modalities(self, sensor_id):
        return {Modality.POSE, Modality.TIMESTAMP}

    def get_length(self, sensor_id):
        return len(self._manifest[sensor_id][Modality.NAME])

    def get_rgb(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_rgb_semantic_mask(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_rgb_dynamic_mask(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_rgb_valid_mask(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_depth(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_depth_confidence(self, sensor_id, frame_index):
        raise NotImplementedError

    def get_tracks(self, sensor_id):
        raise NotImplementedError

    def get_pointcloud(self, sensor_id):
        raise NotImplementedError


class TestIndexAddressing:
    """The decode surface addresses frames by position, never by name."""

    def test_get_pose_addresses_by_position(self):
        seq = _IndexAddrSequence()
        np.testing.assert_allclose(seq.get_pose(0, 0)[:3, 3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(seq.get_pose(0, 2)[:3, 3], [2.0, 0.0, 0.0])

    def test_get_timestamp_addresses_by_position(self):
        seq = _IndexAddrSequence()
        assert seq.get_timestamp(0, 1) == 200.0


# =========================================================================== #
# rescale_intrinsic — pure per-axis scaling (usable during construction)
# =========================================================================== #
class TestRescaleIntrinsic:
    """rescale_intrinsic scales fx,cx by width ratio and fy,cy by height ratio."""

    K = np.array([3844.898776958602, 3852.356179237937, 2593.5, 1680.5])

    def test_identity_when_sizes_match(self):
        out = BaseSequence.rescale_intrinsic(self.K, (5187, 3361), (5187, 3361))
        np.testing.assert_allclose(out, self.K, rtol=1e-12)

    def test_quarter_scale_for_images_4(self):
        out = BaseSequence.rescale_intrinsic(self.K, (5187, 3361), (1297, 840))
        sx, sy = 1297 / 5187, 840 / 3361
        np.testing.assert_allclose(
            out, [self.K[0] * sx, self.K[1] * sy, self.K[2] * sx, self.K[3] * sy], rtol=1e-12
        )

    def test_per_axis_scales_independently(self):
        # non-uniform target exercises fx/cx vs fy/cy separately
        out = BaseSequence.rescale_intrinsic(self.K, (100, 100), (50, 25))
        np.testing.assert_allclose(
            out, [self.K[0] * 0.5, self.K[1] * 0.25, self.K[2] * 0.5, self.K[3] * 0.25],
            rtol=1e-12,
        )


# =========================================================================== #
# COLMAP — full vendor conformance against the real garden reconstruction
# =========================================================================== #
_COLMAP_ROOT = "/jfs/jing.feng/datasets/mipnerf360"  # real dataset; skipped if absent
_COLMAP_SEQ = "garden"


def _pycolmap_available() -> bool:
    try:
        import pycolmap  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_COLMAP_ROOT, _COLMAP_SEQ)),
    reason="garden COLMAP dataset not present",
)
@pytest.mark.skipif(not _pycolmap_available(), reason="pycolmap not installed")
class TestColmapSequence(BaseSequenceContract):
    """COLMAP conformance (via BaseSequenceContract) + vendor-specific checks.

    Runs against the real garden reconstruction; skipped when the dataset or
    pycolmap is unavailable (mirrors the _EVAL-absent pattern).
    """

    def make_sequence(self):
        from vggt_omega.datasets.sequences.colmap import ColmapSequence

        return ColmapSequence(_COLMAP_ROOT, _COLMAP_SEQ)

    def test_single_camera_pinhole_intrinsic(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        np.testing.assert_allclose(
            np.asarray(s.get_intrinsic(sid)),
            [3844.898776958602, 3852.356179237937, 2593.5, 1680.5],
            rtol=1e-4,
        )

    def test_frame_count(self):
        s = self.make_sequence()
        assert s.get_length(s.get_sensors()[0]) == 185

    def test_frame_names_are_positions(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        assert list(s._manifest[sid][Modality.NAME]) == list(range(185))

    def test_rgb_paths_are_sorted_colmap_filenames(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        base = [os.path.basename(p) for p in s._manifest[sid][Modality.RGB]]
        assert base[0] == "DSC07956.JPG" and base[-1] == "DSC08140.JPG"
        assert base == sorted(base)

    def test_pointcloud_shape(self):
        s = self.make_sequence()
        pc = s.get_pointcloud(s.get_sensors()[0])
        assert pc.shape == (138766, 3) and pc.dtype == np.float32

    def test_discover_finds_garden(self):
        from vggt_omega.datasets.sequences.colmap import ColmapSequence

        assert _COLMAP_SEQ in ColmapSequence.discover(_COLMAP_ROOT)

    def test_images_4_rescales_intrinsic(self):
        from vggt_omega.datasets.sequences.colmap import ColmapSequence

        sid = self.make_sequence().get_sensors()[0]
        Kf = np.asarray(self.make_sequence().get_intrinsic(sid), dtype=np.float64)
        Kq = np.asarray(
            ColmapSequence(_COLMAP_ROOT, _COLMAP_SEQ, images_dir="images_4").get_intrinsic(sid),
            dtype=np.float64,
        )
        np.testing.assert_allclose(Kq / Kf, [0.25, 0.25, 0.25, 0.25], atol=3e-3)

    def test_unsupported_getters_raise(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        for fn in (
            s.get_depth,
            s.get_depth_confidence,
            s.get_rgb_semantic_mask,
            s.get_rgb_dynamic_mask,
            s.get_rgb_valid_mask,
        ):
            with pytest.raises(NotImplementedError):
                fn(sid, 0)
        with pytest.raises(NotImplementedError):
            s.get_tracks(sid)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_COLMAP_ROOT, _COLMAP_SEQ)),
    reason="garden COLMAP dataset not present",
)
@pytest.mark.skipif(not _pycolmap_available(), reason="pycolmap not installed")
class TestMipNerf360Sequence:
    """The pyramid wrapper: scale_factor picks images_N and rescales intrinsics."""

    def test_scale_factor_selects_pyramid_level_and_rescales(self):
        from vggt_omega.datasets.sequences.mipnerf360 import MipNerf360Sequence

        seq = MipNerf360Sequence(_COLMAP_ROOT, _COLMAP_SEQ, scale_factor=4)
        sid = seq.get_sensors()[0]
        assert seq.get_length(sid) == 185
        # images_4 = 1297x840 on garden; intrinsics land at quarter scale.
        h, w = seq.get_rgb_size(sid)
        assert (w, h) == (1297, 840)
        K = np.asarray(seq.get_intrinsic(sid), dtype=np.float64)
        np.testing.assert_allclose(
            K / np.array([3844.898776958602, 3852.356179237937, 2593.5, 1680.5]),
            [0.25, 0.25, 0.25, 0.25],
            atol=3e-3,
        )

    def test_rejects_unknown_scale_factor(self):
        from vggt_omega.datasets.sequences.mipnerf360 import MipNerf360Sequence

        with pytest.raises(ValueError, match="scale_factor"):
            MipNerf360Sequence(_COLMAP_ROOT, _COLMAP_SEQ, scale_factor=3)

    def test_no_depth_modality(self):
        from vggt_omega.datasets.sequences.mipnerf360 import MipNerf360Sequence

        seq = MipNerf360Sequence(_COLMAP_ROOT, _COLMAP_SEQ, scale_factor=8)
        assert Modality.DEPTH not in seq.get_modalities(seq.get_sensors()[0])
