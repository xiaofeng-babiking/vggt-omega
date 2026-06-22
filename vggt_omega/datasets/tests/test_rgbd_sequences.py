"""Vendor conformance + real-data smoke tests for the BaseSequence RGB-D vendors.

Each vendor subclasses :class:`BaseSequenceContract` (shared interface laws) and
points :meth:`make_sequence` at a real sequence on disk; the whole class is
skipped when that dataset is absent. Vendor-specific correctness (depth scale,
pose convention, intrinsics) is asserted against an independent re-read of the
raw files in the dedicated checks at the bottom.
"""
import os

import numpy as np
import pytest

from vggt_omega.datasets.base_sequence import Modality
from vggt_omega.datasets.tests.sequence_contract import BaseSequenceContract
from vggt_omega.datasets.vendors.bonn import BonnSequence
from vggt_omega.datasets.vendors.neural_rgbd import NeuralRgbdSequence
from vggt_omega.datasets.vendors.nyu import NyuSequence
from vggt_omega.datasets.vendors.seven_scenes import SevenScenesSequence
from vggt_omega.datasets.vendors.scannet import ScannetSequence

_EVAL = "/jfs/guibiao/streamVGGT/data/eval"
BONN_DIR = os.path.join(_EVAL, "bonn")
SEVEN_DIR = os.path.join(_EVAL, "7scenes")
NEURAL_DIR = os.path.join(_EVAL, "neural_rgbd")
NYU_DIR = os.path.join(_EVAL, "nyu")


def _first_subdir(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))[0]


# --------------------------------------------------------------------------- #
# Bonn
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(BONN_DIR), reason="Bonn dataset not available")
class TestBonnSequence(BaseSequenceContract):
    SEQ = "rgbd_bonn_balloon"

    def make_sequence(self):
        return BonnSequence(BONN_DIR, self.SEQ)

    def test_depth_scale_5000(self):
        s = self.make_sequence()
        from PIL import Image
        raw = np.asarray(Image.open(s._frames[0][1])).astype(np.float32) / 5000.0
        np.testing.assert_allclose(s.get_depth(0, 0), raw, atol=1e-6)

    def test_camera_vs_marker_pose_differ(self):
        cam = BonnSequence(BONN_DIR, self.SEQ, pose_frame="camera")
        mark = BonnSequence(BONN_DIR, self.SEQ, pose_frame="marker")
        assert not np.allclose(cam.get_pose(0, 0).translation, mark.get_pose(0, 0).translation)


# --------------------------------------------------------------------------- #
# 7-Scenes
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Neural RGB-D
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# NYU
# --------------------------------------------------------------------------- #
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
    from PIL import Image

    scene = "scene9999_00"
    seq = tmp_path / "scans_train" / scene
    (seq / "color").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)
    (seq / "cam").mkdir(parents=True)
    H, W = 12, 16
    K = np.array([[100.0, 0, 8.0], [0, 110.0, 6.0], [0, 0, 1.0]], np.float32)
    for k in range(3):
        Image.fromarray(np.full((H, W, 3), k * 10, np.uint8)).save(seq / "color" / f"{k:05d}.jpg")
        Image.fromarray(np.full((H, W), (k + 1) * 1000, np.uint16)).save(seq / "depth" / f"{k:05d}.png")
        c2w = np.eye(4)
        c2w[:3, 3] = [k, 0, 0]  # translation (k,0,0), identity rotation
        np.savez(seq / "cam" / f"{k:05d}.npz", intrinsics=K, pose=c2w)
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
