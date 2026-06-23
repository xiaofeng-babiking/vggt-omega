"""Unified BaseSequence vendor conformance + correctness tests.

This single module merges the former per-family sequence test files
(test_family_a_sequences.py, test_family_bc_sequences.py, test_rgbd_sequences.py,
test_tum_sequence.py). Every vendor either:

  * builds a tiny synthetic on-disk sequence in its native layout via a fixture and
    runs the shared :class:`BaseSequenceContract` interface laws against it, or
  * points :meth:`make_sequence` at a real eval dataset and is skipped when absent.

Vendor-specific decode correctness (depth scale, pose convention, intrinsics) lives
in the per-vendor extra checks. The shared contract mixin is defined inline below;
its name does not start with ``Test`` so pytest never collects it on its own.
"""
import os

import numpy as np
import pytest
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose

from vggt_omega.datasets.vendors.tartanair import TartanAirSequence
from vggt_omega.datasets.vendors.mvs_synth import MvsSynthSequence
from vggt_omega.datasets.vendors.hypersim import HypersimSequence
from vggt_omega.datasets.vendors.omniobject3d import OmniObject3DSequence
from vggt_omega.datasets.vendors.wildrgbd import WildRgbdSequence
from vggt_omega.datasets.vendors.co3d import Co3dSequence
from vggt_omega.datasets.vendors.arkitscenes import ArkitScenesSequence
from vggt_omega.datasets.vendors.dl3dv import Dl3dvSequence
from vggt_omega.datasets.vendors.vkitti import VkittiSequence
from vggt_omega.datasets.vendors.pointodyssey import PointOdysseySequence
from vggt_omega.datasets.vendors.spring import SpringSequence
from vggt_omega.datasets.vendors.dynamic_replica import DynamicReplicaSequence
from vggt_omega.datasets.vendors.neu3d import Neu3dSequence
from vggt_omega.datasets.vendors.sintel import SintelSequence
from vggt_omega.datasets.vendors.kitti import KittiSequence
from vggt_omega.datasets.vendors.bonn import BonnSequence
from vggt_omega.datasets.vendors.neural_rgbd import NeuralRgbdSequence
from vggt_omega.datasets.vendors.nyu import NyuSequence
from vggt_omega.datasets.vendors.seven_scenes import SevenScenesSequence
from vggt_omega.datasets.vendors.scannet import ScannetSequence
from vggt_omega.datasets.vendors.tum import TumSequence

# --------------------------------------------------------------------------- #
# shared fixtures / helpers
# --------------------------------------------------------------------------- #
H, W, N = 12, 16, 4
_K = np.array([[100.0, 0, 8.0], [0, 110.0, 6.0], [0, 0, 1.0]], np.float32)
_EVAL = "/jfs/guibiao/streamVGGT/data/eval"


def _c2w(k):
    """camera-to-world: identity rotation, translation (k, 0, 0)."""
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [k, 0.0, 0.0]
    return c2w


def _rgb(p, k, mode="RGB"):
    Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).convert(mode).save(p)


def _first_dir(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))[0]


_first_subdir = _first_dir  # alias kept from the merged files


# --------------------------------------------------------------------------- #
# shared conformance contract (formerly sequence_contract.BaseSequenceContract)
# --------------------------------------------------------------------------- #
def _self_per_frame():
    return BaseSequence._PER_FRAME_MODALITIES


class BaseSequenceContract:
    """Interface laws for any BaseSequence vendor. Subclass + set make_sequence()."""

    def make_sequence(self) -> BaseSequence:
        raise NotImplementedError

    # ----- identity / discovery -------------------------------------------- #
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
        known = set(Modality)
        for sid in s.get_sensors():
            assert s.get_modalities(sid) <= known

    # ----- per-frame getters honour advertised modalities ------------------ #
    def test_advertised_modalities_decode(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        if Modality.RGB in mods:
            rgb = s.get_rgb(sid, 0)
            assert rgb.ndim == 3 and rgb.shape[2] == 3
        if Modality.DEPTH in mods:
            depth = s.get_depth(sid, 0)
            assert depth.ndim == 2
            if Modality.RGB in mods:
                assert depth.shape == s.get_rgb(sid, 0).shape[:2]
        if Modality.POSE in mods:
            assert isinstance(s.get_pose(sid, 0), BaseSE3Pose)
        if Modality.INTRINSIC in mods:
            K = s.get_intrinsic(sid)
            assert K.shape == (3, 3)
        if Modality.TIMESTAMP in mods:
            assert isinstance(s.get_timestamp(sid, 0), float)

    def test_timestamps_monotonic_if_present(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.TIMESTAMP not in s.get_modalities(sid):
            return
        n = min(s.get_length(sid), 50)
        ts = [s.get_timestamp(sid, i) for i in range(n)]
        assert ts == sorted(ts)

    def test_get_poses_matches_get_pose(self):
        # get_poses() returns all frame-sorted c2w poses; get_pose(i) == get_poses()[i].
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.POSE not in s.get_modalities(sid):
            return
        poses = s.get_poses(sid)
        n = s.get_length(sid)
        assert len(poses) == n
        for i in (0, n // 2, n - 1):
            np.testing.assert_allclose(
                poses[i].transform_matrix, s.get_pose(sid, i).transform_matrix, atol=1e-9
            )

    # ----- parse (the base template) --------------------------------------- #
    def test_parse_stacks_in_order(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        n = s.get_length(sid)
        ids = [0, min(1, n - 1), min(2, n - 1)]
        out = s.parse(sid, ids, image_size=(60, 80))
        for m, arr in out.items():
            assert arr.shape[0] == len(ids)
        if Modality.RGB in out:
            assert out[Modality.RGB].shape[1:] == (60, 80, 3)
        if Modality.DEPTH in out:
            assert out[Modality.DEPTH].shape[1:] == (60, 80)

    def test_parse_only_returns_requested(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        if Modality.RGB not in mods:
            return
        out = s.parse(sid, [0, 1], modalities={Modality.RGB})
        assert set(out.keys()) == {Modality.RGB}

    def test_extrinsic_is_per_sequence_calibration(self):
        # EXTRINSIC is the static inter-sensor transform (calibration), fetched via
        # get_extrinsic(src, dst) — NOT a per-frame parse output.
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.EXTRINSIC not in s.get_modalities(sid):
            return
        assert isinstance(s.get_extrinsic(sid, sid), BaseSE3Pose)

    def test_parse_rejects_unavailable_modality(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        missing = (set(_self_per_frame()) - s.get_modalities(sid))
        if not missing:
            return
        with pytest.raises(ValueError):
            s.parse(sid, [0], modalities={next(iter(missing))})

    def test_parse_rejects_per_sequence_modality(self):
        # INTRINSIC / EXTRINSIC are per-sequence calibration; parse must reject them
        # even when the sensor advertises them.
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        for cal in (Modality.INTRINSIC, Modality.EXTRINSIC):
            if cal in mods:
                with pytest.raises(ValueError):
                    s.parse(sid, [0], modalities={cal})

    def test_scaled_intrinsic_rescales(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.INTRINSIC not in s.get_modalities(sid):
            return
        K = s.get_intrinsic(sid)
        h0, w0 = s.read_image_size(s._frame_image_path(sid, 0))
        # native (no image_size) returns K unchanged.
        np.testing.assert_allclose(s.scaled_intrinsic(sid), K, atol=1e-4)
        Kr = s.scaled_intrinsic(sid, image_size=(h0 // 2, w0 // 2))
        np.testing.assert_allclose(Kr[0, 0], K[0, 0] * (w0 // 2) / w0, rtol=1e-3)
        np.testing.assert_allclose(Kr[1, 1], K[1, 1] * (h0 // 2) / h0, rtol=1e-3)


# =========================================================================== #
# Family A: synthetic RGB-D + pose vendors
# =========================================================================== #

# --------------------------------------------------------------------------- #
# TartanAir
# --------------------------------------------------------------------------- #
@pytest.fixture
def tartanair(tmp_path):
    d = tmp_path / "train" / "env" / "Easy" / "P000"
    d.mkdir(parents=True)
    for k in range(N):
        _rgb(d / f"{k:06d}_rgb.png", k)
        np.save(d / f"{k:06d}_depth.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / f"{k:06d}_cam.npz", camera_pose=_c2w(k), camera_intrinsics=_K)
    return str(tmp_path), "env/Easy/P000"


class TestTartanAir(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, tartanair):
        self._root, self._seq = tartanair

    def make_sequence(self):
        return TartanAirSequence(self._root, self._seq)

    def test_depth_and_pose(self):
        s = self.make_sequence()
        np.testing.assert_allclose(s.get_depth(0, 2), 3.0, atol=1e-4)
        np.testing.assert_allclose(s.get_pose(0, 2).translation, [2, 0, 0], atol=1e-6)

    def test_sky_encoding(self):
        # depth > sky_threshold -> -1.0
        d = os.path.join(self._root, "train", self._seq)
        np.save(os.path.join(d, "000000_depth.npy"),
                np.full((H, W), 20000.0, np.float32))  # sky
        s = TartanAirSequence(self._root, self._seq)
        assert (s.get_depth(0, 0) == -1.0).all()


# --------------------------------------------------------------------------- #
# MVS-Synth
# --------------------------------------------------------------------------- #
@pytest.fixture
def mvs_synth(tmp_path):
    d = tmp_path / "train" / "0000"
    for sub in ("rgb", "depth", "cam"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"{k:04d}.jpg", k)
        np.save(d / "depth" / f"{k:04d}.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / "cam" / f"{k:04d}.npz", intrinsics=_K, pose=_c2w(k))
    return str(tmp_path), "0000"


class TestMvsSynth(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, mvs_synth):
        self._root, self._seq = mvs_synth

    def make_sequence(self):
        return MvsSynthSequence(self._root, self._seq)

    def test_depth_pose(self):
        s = self.make_sequence()
        np.testing.assert_allclose(s.get_depth(0, 1), 2.0, atol=1e-4)
        np.testing.assert_allclose(s.get_pose(0, 3).translation, [3, 0, 0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Hypersim
# --------------------------------------------------------------------------- #
@pytest.fixture
def hypersim(tmp_path):
    d = tmp_path / "ai_001_001" / "cam_00"
    d.mkdir(parents=True)
    for k in range(N):
        _rgb(d / f"{k:06d}_rgb.png", k)
        np.save(d / f"{k:06d}_depth.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / f"{k:06d}_cam.npz", pose=_c2w(k), intrinsics=_K)
    return str(tmp_path), "ai_001_001/cam_00"


class TestHypersim(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, hypersim):
        self._root, self._seq = hypersim

    def make_sequence(self):
        return HypersimSequence(self._root, self._seq)

    def test_nan_depth_to_zero(self):
        d = os.path.join(self._root, self._seq)
        arr = np.full((H, W), 2.0, np.float32)
        arr[0, 0] = np.nan
        np.save(os.path.join(d, "000000_depth.npy"), arr)
        s = self.make_sequence()
        assert s.get_depth(0, 0)[0, 0] == 0.0


# --------------------------------------------------------------------------- #
# OmniObject3D
# --------------------------------------------------------------------------- #
@pytest.fixture
def omniobject3d(tmp_path):
    d = tmp_path / "train" / "apple" / "apple_001"
    for sub in ("rgb", "depth", "cam"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"r_{k}.png", k)
        np.save(d / "depth" / f"r_{k}.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / "cam" / f"r_{k}.npz", intrinsics=_K, pose=_c2w(k))
    return str(tmp_path), "apple/apple_001"


class TestOmniObject3D(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, omniobject3d):
        self._root, self._seq = omniobject3d

    def make_sequence(self):
        return OmniObject3DSequence(self._root, self._seq)

    def test_no_timestamp(self):
        assert Modality.TIMESTAMP not in self.make_sequence().get_modalities(0)


# --------------------------------------------------------------------------- #
# WildRGB-D
# --------------------------------------------------------------------------- #
@pytest.fixture
def wildrgbd(tmp_path):
    d = tmp_path / "apple" / "scenes" / "scene_001"
    for sub in ("rgb", "depth", "metadata"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"{k:05d}.jpg", k)
        Image.fromarray(np.full((H, W), (k + 1) * 1000, np.uint16)).save(d / "depth" / f"{k:05d}.png")
        np.savez(d / "metadata" / f"{k:05d}.npz", camera_intrinsics=_K.astype(np.float64), camera_pose=_c2w(k))
    return str(tmp_path), "apple/scenes/scene_001"


class TestWildRgbd(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, wildrgbd):
        self._root, self._seq = wildrgbd

    def make_sequence(self):
        return WildRgbdSequence(self._root, self._seq)

    def test_depth_mm(self):
        np.testing.assert_allclose(self.make_sequence().get_depth(0, 1), 2.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# CO3D
# --------------------------------------------------------------------------- #
@pytest.fixture
def co3d(tmp_path):
    d = tmp_path / "apple" / "seq1"
    (d / "images").mkdir(parents=True)
    (d / "depths").mkdir(parents=True)
    for k in range(N):
        name = f"frame{k:06d}"
        _rgb(d / "images" / f"{name}.jpg", k)
        # depth png normalized: value v -> v/65535 * max_depth. Use max_depth=10,
        # store constant value so decoded depth = 10 * v/65535.
        Image.fromarray(np.full((H, W), 6553, np.uint16)).save(d / "depths" / f"{name}.jpg.geometric.png")
        np.savez(d / "images" / f"{name}.npz",
                 camera_pose=_c2w(k), camera_intrinsics=_K.astype(np.float64),
                 maximum_depth=np.float32(10.0))
    return str(tmp_path), "apple/seq1"


class TestCo3d(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, co3d):
        self._root, self._seq = co3d

    def make_sequence(self):
        return Co3dSequence(self._root, self._seq)

    def test_depth_rescaled_by_max_depth(self):
        s = self.make_sequence()
        # 6553/65535 * 10 ~= 1.0
        np.testing.assert_allclose(s.get_depth(0, 0).mean(), 6553 / 65535 * 10.0, rtol=1e-3)


# --------------------------------------------------------------------------- #
# ARKitScenes
# --------------------------------------------------------------------------- #
@pytest.fixture
def arkitscenes(tmp_path):
    scene = "40000000"
    d = tmp_path / "Training" / scene
    (d / "vga_wide").mkdir(parents=True)
    (d / "lowres_depth").mkdir(parents=True)
    images, trajs, intrs = [], [], []
    for k in range(N):
        ts = 100.0 + k  # increasing
        stem = f"{scene}_{ts:.3f}"
        _rgb(d / "vga_wide" / f"{stem}.jpg", k)
        Image.fromarray(np.full((H, W), (k + 1) * 1000, np.uint16)).save(d / "lowres_depth" / f"{stem}.png")
        images.append(f"{stem}.png")
        trajs.append(_c2w(k))
        intrs.append([W, H, 100.0, 110.0, 8.0, 6.0])
    # store UNSORTED (reversed) to exercise the timestamp sort
    np.savez(
        d / "scene_metadata.npz",
        images=np.array(images[::-1]),
        trajectories=np.array(trajs[::-1]),
        intrinsics=np.array(intrs[::-1]),
    )
    return str(tmp_path), scene


class TestArkitScenes(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _setup(self, arkitscenes):
        self._root, self._seq = arkitscenes

    def make_sequence(self):
        return ArkitScenesSequence(self._root, self._seq)

    def test_sorted_by_timestamp_and_depth(self):
        s = self.make_sequence()
        ts = [s.get_timestamp(0, i) for i in range(s.get_length(0))]
        assert ts == sorted(ts)
        np.testing.assert_allclose(s.get_depth(0, 1), 2.0, atol=1e-4)
        # frame 0 (earliest ts) corresponds to k=0 -> pose translation (0,0,0)
        np.testing.assert_allclose(s.get_pose(0, 0).translation, [0, 0, 0], atol=1e-6)


# =========================================================================== #
# Family B/C: COLMAP-posed, flow/track, and data-verifiable vendors
# =========================================================================== #

# --------------------------------------------------------------------------- #
# DL3DV (RGB + pose only, no depth)
# --------------------------------------------------------------------------- #
@pytest.fixture
def dl3dv(tmp_path):
    d = tmp_path / "abc123" / "dense"
    (d / "rgb").mkdir(parents=True)
    (d / "cam").mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"frame_{k:05d}.png", k)
        np.savez(d / "cam" / f"frame_{k:05d}.npz", pose=_c2w(k), intrinsic=_K)
    return str(tmp_path), "abc123"


class TestDl3dv(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _s(self, dl3dv):
        self._root, self._seq = dl3dv

    def make_sequence(self):
        return Dl3dvSequence(self._root, self._seq)

    def test_no_depth_modality(self):
        assert Modality.DEPTH not in self.make_sequence().get_modalities(0)


# --------------------------------------------------------------------------- #
# VKITTI (cm depth + sky 65535)
# --------------------------------------------------------------------------- #
@pytest.fixture
def vkitti(tmp_path):
    d = tmp_path / "train" / "Scene01" / "clone" / "Camera_0"
    d.mkdir(parents=True)
    for k in range(N):
        _rgb(d / f"{k:05d}_rgb.jpg", k)
        Image.fromarray(np.full((H, W), (k + 1) * 100, np.uint16)).save(d / f"{k:05d}_depth.png")
        np.savez(d / f"{k:05d}_cam.npz", camera_pose=_c2w(k), camera_intrinsics=_K)
    return str(tmp_path), "Scene01/clone/Camera_0"


class TestVkitti(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _s(self, vkitti):
        self._root, self._seq = vkitti

    def make_sequence(self):
        return VkittiSequence(self._root, self._seq)

    def test_depth_cm_and_sky(self):
        s = self.make_sequence()
        np.testing.assert_allclose(s.get_depth(0, 1), 2.0, atol=1e-4)  # 200 cm -> 2 m
        Image.fromarray(np.full((H, W), 65535, np.uint16)).save(
            os.path.join(self._root, "train", self._seq, "00000_depth.png"))
        assert (VkittiSequence(self._root, self._seq).get_depth(0, 0) == -1.0).all()


# --------------------------------------------------------------------------- #
# PointOdyssey (npy metric depth, synth ts)
# --------------------------------------------------------------------------- #
@pytest.fixture
def pointodyssey(tmp_path):
    d = tmp_path / "train" / "seq01"
    for sub in ("rgb", "depth", "cam"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"{k:05d}.jpg", k)
        np.save(d / "depth" / f"{k:05d}.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / "cam" / f"{k:05d}.npz", pose=_c2w(k), intrinsics=_K)
    return str(tmp_path), "seq01"


class TestPointOdyssey(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _s(self, pointodyssey):
        self._root, self._seq = pointodyssey

    def make_sequence(self):
        return PointOdysseySequence(self._root, self._seq)

    def test_timestamp_30fps(self):
        s = self.make_sequence()
        np.testing.assert_allclose(s.get_timestamp(0, 3), 3 / 30.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Spring (npy metric depth)
# --------------------------------------------------------------------------- #
@pytest.fixture
def spring(tmp_path):
    d = tmp_path / "train" / "0001"
    for sub in ("rgb", "depth", "cam"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        _rgb(d / "rgb" / f"{k:04d}.png", k)
        np.save(d / "depth" / f"{k:04d}.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / "cam" / f"{k:04d}.npz", pose=_c2w(k), intrinsics=_K)
    return str(tmp_path), "0001"


class TestSpring(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _s(self, spring):
        self._root, self._seq = spring

    def make_sequence(self):
        return SpringSequence(self._root, self._seq)


# --------------------------------------------------------------------------- #
# Dynamic Replica (RGBA rgb, float-second timestamps)
# --------------------------------------------------------------------------- #
@pytest.fixture
def dynamic_replica(tmp_path):
    d = tmp_path / "train" / "abc-1_obj" / "left"
    for sub in ("rgb", "depth", "cam"):
        (d / sub).mkdir(parents=True)
    for k in range(N):
        t = k / 30.0
        _rgb(d / "rgb" / f"{t}.png", k, mode="RGBA")
        np.save(d / "depth" / f"{t}.npy", np.full((H, W), k + 1.0, np.float32))
        np.savez(d / "cam" / f"{t}.npz", pose=_c2w(k), intrinsics=_K)
    return str(tmp_path), "abc-1_obj/left"


class TestDynamicReplica(BaseSequenceContract):
    @pytest.fixture(autouse=True)
    def _s(self, dynamic_replica):
        self._root, self._seq = dynamic_replica

    def make_sequence(self):
        return DynamicReplicaSequence(self._root, self._seq)

    def test_rgba_dropped(self):
        assert self.make_sequence().get_rgb(0, 0).shape == (H, W, 3)


# --------------------------------------------------------------------------- #
# Real-data trio: Sintel / Neu3D / KITTI
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(os.path.join(_EVAL, "sintel")), reason="Sintel not available")
class TestSintel(BaseSequenceContract):
    def make_sequence(self):
        root = os.path.join(_EVAL, "sintel")
        seq = _first_dir(os.path.join(root, "training", "final"))
        return SintelSequence(root, seq)

    def test_pose_is_inverse_of_stored_w2c(self):
        s = self.make_sequence()
        _, w2c = s._read_cam(s._frames[0][2])
        T = np.eye(4)
        T[:3] = w2c
        # get_pose returns c2w == inv(stored w2c)
        np.testing.assert_allclose(s.get_pose(0, 0).transform_matrix, np.linalg.inv(T), atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not os.path.isdir(os.path.join(_EVAL, "neu3d")), reason="Neu3D not available")
class TestNeu3d(BaseSequenceContract):
    def make_sequence(self):
        root = os.path.join(_EVAL, "neu3d")
        for sc in sorted(os.listdir(root)):
            scd = os.path.join(root, sc)
            if not os.path.isdir(scd):
                continue
            for cam in sorted(os.listdir(scd)):
                if os.path.isdir(os.path.join(scd, cam, "images")):
                    return Neu3dSequence(root, f"{sc}/{cam}")
        pytest.skip("no neu3d scene/cam with images/")

    def test_image_only(self):
        assert self.make_sequence().get_modalities(0) == {Modality.RGB}


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_EVAL, "kitti", "val")), reason="KITTI not available"
)
class TestKitti(BaseSequenceContract):
    def make_sequence(self):
        val = os.path.join(_EVAL, "kitti", "val")
        drive = sorted(d for d in os.listdir(val) if "drive" in d)[0]
        return KittiSequence(os.path.join(_EVAL, "kitti"), drive, camera=2)

    def test_depth_metric_and_motion(self):
        s = self.make_sequence()
        d = s.get_depth(0, 0)
        assert d[d > 0].min() > 0
        # driving trajectory accrues large displacement over the drive.
        last = s.get_length(0) - 1
        assert np.linalg.norm(s.get_pose(0, last).translation) > 1.0


# =========================================================================== #
# RGB-D vendors: real-data conformance (skipped when dataset absent)
# =========================================================================== #
BONN_DIR = os.path.join(_EVAL, "bonn")
SEVEN_DIR = os.path.join(_EVAL, "7scenes")
NEURAL_DIR = os.path.join(_EVAL, "neural_rgbd")
NYU_DIR = os.path.join(_EVAL, "nyu")


@pytest.mark.skipif(not os.path.isdir(BONN_DIR), reason="Bonn dataset not available")
class TestBonnSequence(BaseSequenceContract):
    SEQ = "rgbd_bonn_balloon"

    def make_sequence(self):
        return BonnSequence(BONN_DIR, self.SEQ)

    def test_depth_scale_5000(self):
        s = self.make_sequence()
        raw = np.asarray(Image.open(s._frames[0][1])).astype(np.float32) / 5000.0
        np.testing.assert_allclose(s.get_depth(0, 0), raw, atol=1e-6)

    def test_camera_vs_marker_pose_differ(self):
        cam = BonnSequence(BONN_DIR, self.SEQ, pose_frame="camera")
        mark = BonnSequence(BONN_DIR, self.SEQ, pose_frame="marker")
        assert not np.allclose(cam.get_pose(0, 0).translation, mark.get_pose(0, 0).translation)


@pytest.mark.skipif(not os.path.isdir(SEVEN_DIR), reason="7-Scenes dataset not available")
class TestSevenScenesSequence(BaseSequenceContract):
    def make_sequence(self):
        return SevenScenesSequence(SEVEN_DIR, "chess/seq-01")

    def test_intrinsic_default_focal_585(self):
        K = self.make_sequence().get_intrinsic(0)
        np.testing.assert_allclose([K[0, 0], K[1, 1]], [585.0, 585.0])
        np.testing.assert_allclose([K[0, 2], K[1, 2]], [320.0, 240.0])

    def test_pose_matches_raw_pose_txt(self):
        s = self.make_sequence()
        raw = np.loadtxt(s._frames[0][2]).reshape(4, 4)
        # NumpySE3Pose round-trips rotation through a quaternion (~1e-6); compare loosely.
        np.testing.assert_allclose(s.get_pose(0, 0).transform_matrix, raw, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not os.path.isdir(NEURAL_DIR), reason="Neural RGB-D dataset not available")
class TestNeuralRgbdSequence(BaseSequenceContract):
    def make_sequence(self):
        return NeuralRgbdSequence(NEURAL_DIR, _first_subdir(NEURAL_DIR))

    def test_opengl_to_opencv_flip_applied(self):
        # The OpenCV c2w rotation must equal the raw OpenGL c2w rotation with the
        # Y/Z columns negated (independent re-read of poses.txt).
        s = self.make_sequence()
        with open(os.path.join(s.seq_dir, "poses.txt")) as f:
            gl = np.asarray(f.read().split(), dtype=np.float64).reshape(-1, 4, 4)[0]
        cv = s.get_pose(0, 0).transform_matrix
        expect = gl[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(cv[:3, :3], expect, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not os.path.isdir(NYU_DIR), reason="NYU dataset not available")
class TestNyuSequence(BaseSequenceContract):
    def make_sequence(self):
        return NyuSequence(NYU_DIR)

    def test_only_rgb_depth(self):
        assert self.make_sequence().get_modalities(0) == {Modality.RGB, Modality.DEPTH}

    def test_depth_is_metric_npy(self):
        s = self.make_sequence()
        raw = np.load(s._frames[0][2]).astype(np.float32)
        raw[~np.isfinite(raw) | (raw < 0)] = 0.0
        np.testing.assert_allclose(s.get_depth(0, 0), raw, atol=1e-6)

    def test_pose_is_identity(self):
        p = self.make_sequence().get_pose(0, 0)
        np.testing.assert_allclose(p.transform_matrix, np.eye(4), atol=1e-9)


# --------------------------------------------------------------------------- #
# ScanNet (no eval copy on this machine; synthetic conformance instead)
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_scannet(tmp_path):
    """Build a tiny ScanNet-layout scene: color/depth/cam npz for 3 frames."""
    scene = "scene9999_00"
    seq = tmp_path / "scans_train" / scene
    (seq / "color").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)
    (seq / "cam").mkdir(parents=True)
    for k in range(3):
        Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).save(seq / "color" / f"{k:05d}.jpg")
        Image.fromarray(np.full((H, W), (k + 1) * 1000, np.uint16)).save(seq / "depth" / f"{k:05d}.png")
        np.savez(seq / "cam" / f"{k:05d}.npz", intrinsics=_K, pose=_c2w(k))
    return str(tmp_path), scene


def test_scannet_synthetic(synthetic_scannet):
    root, scene = synthetic_scannet
    s = ScannetSequence(root, scene)
    assert s.get_length(0) == 3
    np.testing.assert_allclose(s.get_depth(0, 1), 2.0, atol=1e-4)  # frame 1 -> 2 m
    np.testing.assert_allclose(s.get_pose(0, 2).translation, [2, 0, 0], atol=1e-6)
    K = s.get_intrinsic(0)
    np.testing.assert_allclose([K[0, 0], K[1, 1]], [100.0, 110.0])
    out = s.parse(0, [0, 2], image_size=(6, 8))
    assert out[Modality.RGB].shape == (2, 6, 8, 3)
    # parse is per-frame only: it must not emit per-sequence calibration.
    assert Modality.INTRINSIC not in out and Modality.EXTRINSIC not in out
    # w2c view comes from the per-frame pose getter.
    w2c = s.get_pose(0, 0).inverse().transform_matrix[:3]
    T = np.eye(4)
    T[:3] = w2c
    np.testing.assert_allclose((T @ s.get_pose(0, 0).transform_matrix), np.eye(4), atol=1e-5)


# =========================================================================== #
# TUM RGB-D vendor (synthetic + real-data smoke)
# =========================================================================== #
TUM_DIR = os.path.join(_EVAL, "tum")
HAVE_TUM = os.path.isdir(TUM_DIR)
_REAL_SEQ = "rgbd_dataset_freiburg3_sitting_static"


@pytest.fixture
def synthetic_tum(tmp_path):
    """Build a minimal freiburg1 TUM sequence on disk: 4 rgb/depth/gt frames."""
    seq_id = "rgbd_dataset_freiburg1_synth"
    seq = tmp_path / seq_id
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)

    n = 4
    rgb_lines, depth_lines, gt_lines = [], [], []
    for k in range(n):
        ts = 100.0 + k * 0.1  # shared clock so association is exact
        rgb_rel = f"rgb/{k}.png"
        dep_rel = f"depth/{k}.png"
        Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).save(seq / rgb_rel)
        # depth = (k+1) metres -> uint16 counts = metres * 5000
        Image.fromarray(np.full((H, W), (k + 1) * 5000, np.uint16)).save(seq / dep_rel)
        rgb_lines.append(f"{ts:.6f} {rgb_rel}")
        depth_lines.append(f"{ts:.6f} {dep_rel}")
        # camera-to-world: identity rotation, translation = (k, 0, 0)
        gt_lines.append(f"{ts:.6f} {float(k)} 0 0 0 0 0 1")

    (seq / "rgb.txt").write_text("# color\n" + "\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("# depth\n" + "\n".join(depth_lines) + "\n")
    (seq / "groundtruth.txt").write_text("# gt\n" + "\n".join(gt_lines) + "\n")
    return str(tmp_path), seq_id, n


def test_tum_is_base_sequence(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    assert isinstance(s, BaseSequence)


def test_tum_manifest_and_discovery(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    assert s.get_sensors() == [0]
    assert s.get_length(0) == n
    mods = s.get_modalities(0)
    assert {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP,
            Modality.INTRINSIC, Modality.EXTRINSIC} == mods


def test_tum_timestamps_sorted(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    ts = [s.get_timestamp(0, i) for i in range(n)]
    assert ts == sorted(ts)
    np.testing.assert_allclose(ts[0], 100.0, atol=1e-6)


def test_tum_rgb_and_depth_decode(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    rgb = s.get_rgb(0, 0)
    assert rgb.shape == (12, 16, 3) and rgb.dtype == np.uint8
    depth = s.get_depth(0, 2)  # frame 2 -> 3 metres
    assert depth.shape == (12, 16) and depth.dtype == np.float32
    np.testing.assert_allclose(depth, 3.0, atol=1e-4)


def test_tum_frame_image_path_and_read_image_size(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    path = s._frame_image_path(0, 0)
    assert path.endswith("rgb/0.png")
    # base-class header read returns native (H, W) without decoding.
    assert s.read_image_size(path) == (12, 16)


def test_tum_scaled_intrinsic_native_and_resized(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    K = s.get_intrinsic(0)
    # image_size=None -> native intrinsic unchanged.
    np.testing.assert_allclose(s.scaled_intrinsic(0), K, atol=1e-5)
    # resized to half (native is 12x16) -> fx, fy, cx, cy all halved.
    Ks = s.scaled_intrinsic(0, image_size=(6, 8))
    np.testing.assert_allclose(Ks[0, 0], K[0, 0] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[1, 1], K[1, 1] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[0, 2], K[0, 2] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[1, 2], K[1, 2] * 0.5, atol=1e-3)


def test_tum_intrinsic_from_freiburg1(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    K = s.get_intrinsic(0)
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K[0, 0], 517.306408, atol=1e-3)  # fx (freiburg1)
    np.testing.assert_allclose(K[1, 2], 255.313989, atol=1e-3)  # cy


def test_tum_intrinsic_override(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id, intrinsics=(100.0, 110.0, 50.0, 60.0))
    K = s.get_intrinsic(0)
    np.testing.assert_allclose(np.diag(K), [100.0, 110.0, 1.0])
    np.testing.assert_allclose(K[:2, 2], [50.0, 60.0])


def test_tum_pose_is_camera_to_world(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    p = s.get_pose(0, 3)
    assert isinstance(p, BaseSE3Pose)
    # gt translation for frame k is (k, 0, 0), identity rotation.
    np.testing.assert_allclose(p.translation, [3.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(p.rotation_matrix, np.eye(3), atol=1e-6)


def test_tum_extrinsic_is_identity(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    e = s.get_extrinsic(0, 0)
    np.testing.assert_allclose(e.translation, [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(e.rotation_matrix, np.eye(3), atol=1e-9)


def test_tum_parse_stacks_per_modality(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(
        0,
        [0, 2, 3],
        modalities={Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP},
    )
    assert out[Modality.RGB].shape == (3, 12, 16, 3)
    assert out[Modality.DEPTH].shape == (3, 12, 16)
    assert out[Modality.POSE].shape == (3, 4, 4)
    assert out[Modality.TIMESTAMP].shape == (3,)
    assert out[Modality.TIMESTAMP].dtype == np.float64


def test_tum_parse_rejects_per_sequence_calibration(synthetic_tum):
    # INTRINSIC / EXTRINSIC are per-sequence calibration, not per-frame -> rejected.
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(ValueError):
        s.parse(0, [0], modalities={Modality.INTRINSIC})
    with pytest.raises(ValueError):
        s.parse(0, [0], modalities={Modality.EXTRINSIC})


def test_tum_parse_resizes(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(0, [0, 1], modalities={Modality.RGB, Modality.DEPTH},
                  image_size=(6, 8))  # half of (12, 16)
    assert out[Modality.RGB].shape == (2, 6, 8, 3)
    assert out[Modality.DEPTH].shape == (2, 6, 8)


def test_tum_parse_float32_image_dtype(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(0, [0, 1], modalities={Modality.RGB}, image_dtype="float32")
    assert out[Modality.RGB].dtype == np.float32
    assert out[Modality.RGB].max() <= 1.0


def test_tum_extrinsic_is_inverse_of_pose(synthetic_tum):
    # get_extrinsic is the static inter-sensor transform (identity for TUM); the
    # w2c view of a frame is get_pose(...).inverse().
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    w2c = s.get_pose(0, 3).inverse().transform_matrix[:3]
    c2w = s.get_pose(0, 3).transform_matrix
    T = np.eye(4)
    T[:3] = w2c
    np.testing.assert_allclose((T @ c2w)[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose((T @ c2w)[:3, 3], 0.0, atol=1e-5)


def test_tum_frame_out_of_range_raises(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(IndexError):
        s.get_rgb(0, n)


def test_tum_unknown_intrinsics_raises(tmp_path):
    # a sequence name with no freiburg{1,2,3} substring and no override.
    seq_id = "rgbd_dataset_unknown_cam"
    seq = tmp_path / seq_id
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(seq / "rgb/0.png")
    Image.fromarray(np.zeros((4, 4), np.uint16)).save(seq / "depth/0.png")
    (seq / "rgb.txt").write_text("1.0 rgb/0.png\n")
    (seq / "depth.txt").write_text("1.0 depth/0.png\n")
    (seq / "groundtruth.txt").write_text("1.0 0 0 0 0 0 0 1\n")
    with pytest.raises(ValueError):
        TumSequence(str(tmp_path), seq_id)


def test_tum_parse_rejects_unavailable_modality(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(ValueError):
        s.parse(0, [0], modalities={Modality.DEPTH_CONFIDENCE})


def test_tum_unsupported_getters_raise(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(NotImplementedError):
        s.get_semantic_mask(0, 0)
    with pytest.raises(NotImplementedError):
        s.get_dynamic_mask(0, 0)
    with pytest.raises(NotImplementedError):
        s.get_depth_confidence(0, 0)
    with pytest.raises(NotImplementedError):
        s.get_tracks(0)
    with pytest.raises(NotImplementedError):
        s.get_pointcloud()


@pytest.mark.skipif(not HAVE_TUM, reason="TUM dataset not available")
def test_tum_real_sequence_smoke():
    s = TumSequence(TUM_DIR, _REAL_SEQ)
    assert s.get_length(0) > 100
    rgb = s.get_rgb(0, 0)
    depth = s.get_depth(0, 0)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert depth.shape == rgb.shape[:2]
    assert depth[depth > 0].min() > 0  # metric depth populated
    out = s.parse(0, [0, 10, 20], image_size=(120, 160))
    assert out[Modality.RGB].shape == (3, 120, 160, 3)
    assert out[Modality.POSE].shape == (3, 4, 4)
