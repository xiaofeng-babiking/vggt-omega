"""Synthetic-data conformance tests for the family-a RGB-D + pose vendors.

Each vendor builds a tiny on-disk sequence in its native layout via a fixture,
then runs the shared :class:`BaseSequenceContract` interface laws against it (no
real datasets needed). Vendor-specific decode correctness (depth scale, pose
convention) is asserted in the per-vendor extra checks.
"""
import numpy as np
import pytest
from PIL import Image

from vggt_omega.datasets.base_sequence import Modality
from vggt_omega.datasets.tests.sequence_contract import BaseSequenceContract
from vggt_omega.datasets.vendors.tartanair import TartanAirSequence
from vggt_omega.datasets.vendors.mvs_synth import MvsSynthSequence
from vggt_omega.datasets.vendors.hypersim import HypersimSequence
from vggt_omega.datasets.vendors.omniobject3d import OmniObject3DSequence
from vggt_omega.datasets.vendors.wildrgbd import WildRgbdSequence
from vggt_omega.datasets.vendors.co3d import Co3dSequence
from vggt_omega.datasets.vendors.arkitscenes import ArkitScenesSequence

H, W, N = 12, 16, 4
_K = np.array([[100.0, 0, 8.0], [0, 110.0, 6.0], [0, 0, 1.0]], np.float32)


def _c2w(k):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [k, 0.0, 0.0]  # translation (k,0,0), identity rotation
    return c2w


def _rgb(p, k):
    Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).save(p)


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
        root, seq = self._root, self._seq
        import os
        d = os.path.join(root, "train", seq)
        np.save(os.path.join(d, "000000_depth.npy"),
                np.full((H, W), 20000.0, np.float32))  # sky
        s = TartanAirSequence(root, seq)
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
        import os
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
