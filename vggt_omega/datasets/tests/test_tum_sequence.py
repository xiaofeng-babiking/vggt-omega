"""Tests for the BaseSequence-based TUM RGB-D vendor (TumSequence).

These run against a real TUM sequence directory if present; otherwise they are
skipped. A tiny synthetic on-disk sequence covers the manifest/association and
getter logic without needing the full dataset.
"""
import os

import numpy as np
import pytest
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose
from vggt_omega.datasets.vendors.tum import TumSequence

TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
HAVE_TUM = os.path.isdir(TUM_DIR)
_REAL_SEQ = "rgbd_dataset_freiburg3_sitting_static"


# --------------------------------------------------------------------------- #
# synthetic sequence fixture (no external data required)
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_tum(tmp_path):
    """Build a minimal freiburg1 TUM sequence on disk: 4 rgb/depth/gt frames."""
    seq_id = "rgbd_dataset_freiburg1_synth"
    seq = tmp_path / seq_id
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir(parents=True)

    n = 4
    H, W = 12, 16
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


# --------------------------------------------------------------------------- #
# synthetic-data tests
# --------------------------------------------------------------------------- #

def test_is_base_sequence(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    assert isinstance(s, BaseSequence)


def test_manifest_and_discovery(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    assert s.get_sensors() == [0]
    assert s.get_length(0) == n
    mods = s.get_modalities(0)
    assert {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP,
            Modality.INTRINSIC, Modality.EXTRINSIC} == mods


def test_timestamps_sorted(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    ts = [s.get_timestamp(0, i) for i in range(n)]
    assert ts == sorted(ts)
    np.testing.assert_allclose(ts[0], 100.0, atol=1e-6)


def test_rgb_and_depth_decode(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    rgb = s.get_rgb(0, 0)
    assert rgb.shape == (12, 16, 3) and rgb.dtype == np.uint8
    depth = s.get_depth(0, 2)  # frame 2 -> 3 metres
    assert depth.shape == (12, 16) and depth.dtype == np.float32
    np.testing.assert_allclose(depth, 3.0, atol=1e-4)


def test_frame_image_path_and_read_image_size(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    path = s._frame_image_path(0, 0)
    assert path.endswith("rgb/0.png")
    # base-class header read returns native (H, W) without decoding.
    assert s.read_image_size(path) == (12, 16)


def test_scaled_intrinsic_native_and_resized(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    K = s.get_intrinsic(0)
    # image_size=None -> native intrinsic unchanged.
    np.testing.assert_allclose(s.scaled_intrinsic(0, 0), K, atol=1e-5)
    # resized to half (native is 12x16) -> fx, fy, cx, cy all halved.
    Ks = s.scaled_intrinsic(0, 0, image_size=(6, 8))
    np.testing.assert_allclose(Ks[0, 0], K[0, 0] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[1, 1], K[1, 1] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[0, 2], K[0, 2] * 0.5, atol=1e-3)
    np.testing.assert_allclose(Ks[1, 2], K[1, 2] * 0.5, atol=1e-3)


def test_intrinsic_from_freiburg1(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    K = s.get_intrinsic(0)
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K[0, 0], 517.306408, atol=1e-3)  # fx (freiburg1)
    np.testing.assert_allclose(K[1, 2], 255.313989, atol=1e-3)  # cy


def test_intrinsic_override(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id, intrinsics=(100.0, 110.0, 50.0, 60.0))
    K = s.get_intrinsic(0)
    np.testing.assert_allclose(np.diag(K), [100.0, 110.0, 1.0])
    np.testing.assert_allclose(K[:2, 2], [50.0, 60.0])


def test_pose_is_camera_to_world(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    p = s.get_pose(0, 3)
    assert isinstance(p, BaseSE3Pose)
    # gt translation for frame k is (k, 0, 0), identity rotation.
    np.testing.assert_allclose(p.translation, [3.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(p.rotation_matrix, np.eye(3), atol=1e-6)


def test_extrinsic_is_identity(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    e = s.get_extrinsic(0, 0)
    np.testing.assert_allclose(e.translation, [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(e.rotation_matrix, np.eye(3), atol=1e-9)


def test_parse_stacks_per_modality(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(
        0,
        [0, 2, 3],
        modalities={Modality.RGB, Modality.DEPTH, Modality.INTRINSIC,
                    Modality.EXTRINSIC, Modality.POSE, Modality.TIMESTAMP},
    )
    assert out[Modality.RGB].shape == (3, 12, 16, 3)
    assert out[Modality.DEPTH].shape == (3, 12, 16)
    assert out[Modality.INTRINSIC].shape == (3, 3, 3)
    assert out[Modality.EXTRINSIC].shape == (3, 3, 4)
    assert out[Modality.POSE].shape == (3, 4, 4)
    assert out[Modality.TIMESTAMP].shape == (3,)
    assert out[Modality.TIMESTAMP].dtype == np.float64


def test_parse_resizes_and_rescales_intrinsic(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(0, [0, 1], modalities={Modality.RGB, Modality.DEPTH, Modality.INTRINSIC},
                  image_size=(6, 8))  # half of (12, 16)
    assert out[Modality.RGB].shape == (2, 6, 8, 3)
    assert out[Modality.DEPTH].shape == (2, 6, 8)
    K = s.get_intrinsic(0)
    np.testing.assert_allclose(out[Modality.INTRINSIC][0, 0, 0], K[0, 0] * 0.5, atol=1e-3)
    np.testing.assert_allclose(out[Modality.INTRINSIC][0, 1, 1], K[1, 1] * 0.5, atol=1e-3)


def test_parse_float32_image_dtype(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(0, [0, 1], modalities={Modality.RGB}, image_dtype="float32")
    assert out[Modality.RGB].dtype == np.float32
    assert out[Modality.RGB].max() <= 1.0


def test_parse_extrinsic_is_w2c_inverse_of_pose(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    out = s.parse(0, [3], modalities={Modality.EXTRINSIC})
    w2c = out[Modality.EXTRINSIC][0]  # (3,4)
    c2w = s.get_pose(0, 3).transform_matrix
    T = np.eye(4)
    T[:3] = w2c
    np.testing.assert_allclose((T @ c2w)[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose((T @ c2w)[:3, 3], 0.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# validation / error paths
# --------------------------------------------------------------------------- #

def test_frame_out_of_range_raises(synthetic_tum):
    root, seq_id, n = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(IndexError):
        s.get_rgb(0, n)


def test_unknown_intrinsics_raises(tmp_path):
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


def test_parse_rejects_unavailable_modality(synthetic_tum):
    root, seq_id, _ = synthetic_tum
    s = TumSequence(root, seq_id)
    with pytest.raises(ValueError):
        s.parse(0, [0], modalities={Modality.DEPTH_CONFIDENCE})


def test_unsupported_getters_raise(synthetic_tum):
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


# --------------------------------------------------------------------------- #
# real-data smoke test (skipped if the dataset is absent)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not HAVE_TUM, reason="TUM dataset not available")
def test_real_sequence_smoke():
    s = TumSequence(TUM_DIR, _REAL_SEQ)
    assert s.get_length(0) > 100
    rgb = s.get_rgb(0, 0)
    depth = s.get_depth(0, 0)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert depth.shape == rgb.shape[:2]
    assert depth[depth > 0].min() > 0  # metric depth populated
    out = s.parse(0, [0, 10, 20], image_size=(120, 160))
    assert out[Modality.RGB].shape == (3, 120, 160, 3)
    assert out[Modality.EXTRINSIC].shape == (3, 3, 4)
