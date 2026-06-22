"""Conformance tests for the family-b (COLMAP-posed), family-c (flow/track) and
the data-verifiable (sintel/neu3d/kitti) BaseSequence vendors.

Synthetic on-disk fixtures exercise the npz/npy/EXR vendors through the shared
:class:`BaseSequenceContract`; sintel/neu3d/kitti additionally run against the
real eval datasets when present (skipif otherwise).
"""
import os

import numpy as np
import pytest
from PIL import Image

from vggt_omega.datasets.base_sequence import Modality
from vggt_omega.datasets.tests.sequence_contract import BaseSequenceContract

from vggt_omega.datasets.vendors.dl3dv import Dl3dvSequence
from vggt_omega.datasets.vendors.vkitti import VkittiSequence
from vggt_omega.datasets.vendors.pointodyssey import PointOdysseySequence
from vggt_omega.datasets.vendors.spring import SpringSequence
from vggt_omega.datasets.vendors.dynamic_replica import DynamicReplicaSequence
from vggt_omega.datasets.vendors.neu3d import Neu3dSequence
from vggt_omega.datasets.vendors.sintel import SintelSequence
from vggt_omega.datasets.vendors.kitti import KittiSequence

H, W, N = 12, 16, 4
_K = np.array([[100.0, 0, 8.0], [0, 110.0, 6.0], [0, 0, 1.0]], np.float32)
_EVAL = "/jfs/guibiao/streamVGGT/data/eval"


def _c2w(k):
    c = np.eye(4, dtype=np.float64)
    c[:3, 3] = [k, 0.0, 0.0]
    return c


def _rgb(p, k, mode="RGB"):
    arr = np.full((H, W, 3), k * 10, np.uint8)
    Image.fromarray(arr).convert(mode).save(p)


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
        import os as _os
        Image.fromarray(np.full((H, W), 65535, np.uint16)).save(
            _os.path.join(self._root, "train", self._seq, "00000_depth.png"))
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
# Real-data trio
# --------------------------------------------------------------------------- #
def _first_dir(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))[0]


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
