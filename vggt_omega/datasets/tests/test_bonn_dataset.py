import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.bonn import BonnDataset

BONN_DIR = "/jfs/guibiao/streamVGGT/data/eval/bonn"
HAVE_BONN = os.path.isdir(BONN_DIR)
# A dynamic sequence with rotation-rich camera motion: the hand-eye-corrected
# camera poses close cross-frame reprojection at ~2% here while the raw marker
# poses are off by ~25%, so it cleanly discriminates the pose conventions.
SEQ = "rgbd_bonn_balloon"


def _common_conf():
    return OmegaConf.create(
        {
            "img_size": 512,
            "patch_size": 16,
            "training": True,
            "inside_random": False,
            "allow_duplicate_img": True,
            "get_nearby": True,
            "rescale": True,
            "rescale_aug": True,
            "landscape_check": False,
            "augs": {"scales": [0.8, 1.2]},
        }
    )


def _integration_common():
    return OmegaConf.merge(
        _common_conf(),
        OmegaConf.create(
            {
                "fix_img_num": -1,
                "fix_aspect_ratio": 1.0,
                "load_track": False,
                "track_num": 1024,
                "load_depth": True,
                "debug": False,
                "repeat_batch": False,
                "img_nums": [2, 6],
                "max_img_per_gpu": 12,
                "augs": {
                    "scales": [0.8, 1.2],
                    "aspects": [1.0, 1.0],
                    "cojitter": False,
                    "cojitter_ratio": 0.3,
                    "color_jitter": None,
                    "gray_scale": False,
                    "gau_blur": False,
                },
            }
        ),
    )


def _eval_common():
    """Deterministic eval-mode common_config: no aug, no random remap, explicit ids
    honored verbatim (matches how inference.py drives the loader)."""
    return OmegaConf.merge(
        _integration_common(),
        OmegaConf.create(
            {
                "training": False,
                "inside_random": False,
                "rescale_aug": False,
                "get_nearby": False,
                "allow_duplicate_img": False,
                "augs": {"scales": None},
            }
        ),
    )


def _bonn_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.bonn.BonnDataset",
                "split": "test",
                "BONN_DIR": BONN_DIR,
                "sequences": list(seqs),
                "len_test": n,
            }
        ],
    }


# --- Bonn-specific helper unit tests (no data required) ---


def test_marker_to_cam_is_rigid_transform():
    """The embedded hand-eye constant must be a valid SE(3): orthonormal R with
    det +1 and the documented ~1.9 cm lever arm."""
    X = BonnDataset.MARKER_TO_CAM
    R, t = X[:3, :3], X[:3, 3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0.999
    assert 0.005 < np.linalg.norm(t) < 0.05
    np.testing.assert_allclose(X[3], [0.0, 0.0, 0.0, 1.0])


def test_bonn_pose_to_w2c_marker_inverts_c2w():
    """pose_frame='marker' is a plain TUM-style c2w inversion of the raw pose."""
    w2c = BonnDataset.bonn_pose_to_w2c(
        np.array([1.0, 2.0, 3.0]), (0.0, 0.0, 0.0, 1.0), pose_frame="marker"
    )
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_bonn_pose_to_w2c_camera_applies_hand_eye():
    """pose_frame='camera' must equal inv(T_w_marker @ X)[:3] exactly (the
    survey-verified recipe), for a non-trivial marker pose."""
    t = np.array([0.5, -1.2, 2.0])
    q = np.array([0.1, -0.2, 0.3, 0.9])
    q = tuple(q / np.linalg.norm(q))
    w2c = BonnDataset.bonn_pose_to_w2c(t, q, pose_frame="camera")

    from vggt_omega.datasets.vendors.common import quat_to_rotation

    T = np.eye(4)
    T[:3, :3] = quat_to_rotation(q)
    T[:3, 3] = t
    expected = np.linalg.inv(T @ BonnDataset.MARKER_TO_CAM)[:3, :]
    np.testing.assert_allclose(w2c, expected, atol=1e-6)
    # And it must differ from the raw marker inversion (the transforms are
    # genuinely different frames).
    raw = BonnDataset.bonn_pose_to_w2c(t, q, pose_frame="marker")
    assert np.abs(w2c - raw).max() > 0.1


def test_bonn_pose_to_w2c_rejects_unknown_pose_frame():
    with pytest.raises(ValueError, match="pose_frame"):
        BonnDataset.bonn_pose_to_w2c(np.zeros(3), (0.0, 0.0, 0.0, 1.0), pose_frame="optical")


def test_bonn_intrinsics_default_and_override():
    K = BonnDataset.bonn_intrinsics()
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        [542.822841, 542.576870, 315.593520, 237.756098],
        rtol=1e-6,
    )
    K2 = BonnDataset.bonn_intrinsics(override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0


def test_frame_timestamp_parses_filename():
    assert BonnDataset._frame_timestamp("/x/rgb_110/1548266470.85761.png") == pytest.approx(
        1548266470.85761
    )


def test_associate_windowed_matches_quadratic_reference():
    """_associate_windowed replaces the O(n*m) common.associate in the
    subset='full' path (which took ~100 s on rgbd_bonn_static's 10916x10906
    streams) and must reproduce its greedy matching EXACTLY -- including
    contended candidates, ties, and the strict < max_diff gate."""
    from vggt_omega.datasets.vendors.common import associate

    rng = np.random.default_rng(0)
    first = sorted({round(t, 4) for t in rng.uniform(0.0, 30.0, 400)})
    second = sorted({round(t, 4) for t in rng.uniform(0.0, 30.0, 380)})
    ref = associate(first, second, 0.05)
    assert len(ref) > 100  # the random streams genuinely overlap
    assert BonnDataset._associate_windowed(first, second, 0.05) == ref

    # Contention: 0.01 is the best candidate for both 0.0 and 0.011; greedy
    # best-pair-first must give it to 0.011 and push 0.0 out of the gate.
    first2, second2 = [0.0, 0.011], [0.01, 0.05]
    ref2 = associate(first2, second2, 0.02)
    assert ref2 == [(0.011, 0.01)]
    assert BonnDataset._associate_windowed(first2, second2, 0.02) == ref2

    # Strict gate: a pair exactly max_diff apart is NOT a match (reference
    # uses <, not <=).
    assert BonnDataset._associate_windowed([0.0], [0.02], 0.02) == []


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_associate_windowed_matches_reference_on_real_streams():
    """On a real full rgb/depth stream pair (balloon, ~440 frames -- small
    enough that the quadratic reference is still fast), the windowed matcher
    returns the identical association the shared helper would."""
    from vggt_omega.datasets.vendors.common import associate, read_file_list

    seq_dir = os.path.join(BONN_DIR, "rgbd_bonn_dataset", SEQ)
    rgb = read_file_list(os.path.join(seq_dir, "rgb.txt"))
    depth = read_file_list(os.path.join(seq_dir, "depth.txt"))
    ref = associate(list(rgb), list(depth), 0.02)
    assert len(ref) > 400
    assert BonnDataset._associate_windowed(list(rgb), list(depth), 0.02) == ref


def test_count_index_entries(tmp_path):
    p = tmp_path / "rgb.txt"
    p.write_text("# color images\n# timestamp filename\n\n1.0 rgb/1.0.png\n2.0 rgb/2.0.png\n")
    assert BonnDataset._count_index_entries(str(p)) == 2


def _write_fake_110_sequence(seq_dir, n=3):
    """Tiny fake _110 subset: timestamps chosen so lexicographic filename order
    differs from numeric order (9.5 < 10.5 numerically but not as strings),
    groundtruth_110.txt headerless in scientific notation (the on-disk format)."""
    ts = [9.5, 10.5, 11.5][:n]
    (seq_dir / "rgb_110").mkdir(parents=True)
    (seq_dir / "depth_110").mkdir(parents=True)
    for t in ts:
        (seq_dir / "rgb_110" / f"{t}.png").touch()
        (seq_dir / "depth_110" / f"{t}.png").touch()
    rows = np.array([[t, 0.1 * t, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for t in ts])
    np.savetxt(seq_dir / "groundtruth_110.txt", rows)  # scientific notation, no header
    return ts


def test_load_110_sequence_sorts_numerically_and_parses_headerless_gt(tmp_path):
    seq_dir = tmp_path / "rgbd_bonn_fake"
    ts = _write_fake_110_sequence(seq_dir)
    frames = BonnDataset.load_110_sequence(str(seq_dir), pose_frame="marker")
    assert len(frames) == len(ts)
    # numeric order honored (lexicographic would put 10.5/11.5 before 9.5)
    assert [f[3] for f in frames] == ts
    for (rgb_path, depth_path, w2c, t) in frames:
        assert os.path.basename(rgb_path) == f"{t}.png"
        assert "depth_110" in depth_path
        assert w2c.shape == (3, 4)
        # identity rotation, translation (0.1*t, 0, 1) -> w2c t = -(0.1*t, 0, 1)
        np.testing.assert_allclose(w2c[:, 3], [-0.1 * t, 0.0, -1.0], atol=1e-6)


def test_load_110_sequence_misaligned_raises(tmp_path):
    seq_dir = tmp_path / "rgbd_bonn_fake"
    _write_fake_110_sequence(seq_dir)
    os.remove(seq_dir / "depth_110" / "9.5.png")  # break rgb/depth count parity
    with pytest.raises(ValueError, match="not index-aligned"):
        BonnDataset.load_110_sequence(str(seq_dir))


def test_constructor_validates_args(tmp_path):
    with pytest.raises(ValueError, match="BONN_DIR"):
        BonnDataset(common_conf=_common_conf())
    with pytest.raises(ValueError, match="subset"):
        BonnDataset(common_conf=_common_conf(), BONN_DIR=str(tmp_path), subset="55")
    with pytest.raises(ValueError, match="pose_frame"):
        BonnDataset(common_conf=_common_conf(), BONN_DIR=str(tmp_path), pose_frame="optical")
    with pytest.raises(ValueError, match="No usable Bonn sequences"):
        BonnDataset(common_conf=_common_conf(), BONN_DIR=str(tmp_path))


def _write_fake_full_sequence(seq_dir, n=4, missing_depth_idx=1):
    """Tiny fake full-stream sequence (rgb.txt/depth.txt/groundtruth.txt with TUM
    headers) where one depth PNG listed in depth.txt is missing on disk."""
    (seq_dir / "rgb").mkdir(parents=True)
    (seq_dir / "depth").mkdir(parents=True)
    rgb_lines, depth_lines, gt_lines = [], [], []
    for i in range(n):
        t_rgb, t_dep = 1000.0 + 0.05 * i, 1000.005 + 0.05 * i
        rgb_lines.append(f"{t_rgb} rgb/{t_rgb}.png")
        depth_lines.append(f"{t_dep} depth/{t_dep}.png")
        gt_lines.append(f"{t_rgb} {0.1 * i} 0.0 1.0 0.0 0.0 0.0 1.0")
        (seq_dir / "rgb" / f"{t_rgb}.png").touch()
        if i != missing_depth_idx:
            (seq_dir / "depth" / f"{t_dep}.png").touch()
    (seq_dir / "rgb.txt").write_text("# color images\n# ts file\n" + "\n".join(rgb_lines) + "\n")
    (seq_dir / "depth.txt").write_text("# depth images\n# ts file\n" + "\n".join(depth_lines) + "\n")
    (seq_dir / "groundtruth.txt").write_text("# gt\n# ts tx ty tz qx qy qz qw\n" + "\n".join(gt_lines) + "\n")


def test_full_subset_is_lazy_and_skips_missing_depth(tmp_path):
    seq_dir = tmp_path / "rgbd_bonn_fake"
    _write_fake_full_sequence(seq_dir, n=4, missing_depth_idx=1)
    ds = BonnDataset(
        common_conf=_common_conf(), BONN_DIR=str(tmp_path), subset="full", min_num_images=2
    )
    assert ds.sequence_list == ["rgbd_bonn_fake"]
    assert len(ds.data_store) == 0                  # nothing associated yet (lazy)
    assert ds.sequence_num_frames(0) == 3           # 4 listed, 1 depth PNG missing
    assert len(ds.data_store) == 1                  # built once, cached
    for rgb_path, depth_path, w2c, ts in ds.data_store["rgbd_bonn_fake"]:
        assert os.path.exists(depth_path)
        assert w2c.shape == (3, 4)
    # frames keep stream order (rgb timestamps strictly increasing)
    ts = [f[3] for f in ds.data_store["rgbd_bonn_fake"]]
    assert ts == sorted(ts)


def test_full_subset_lazy_raises_below_min_num_images(tmp_path):
    seq_dir = tmp_path / "rgbd_bonn_fake"
    _write_fake_full_sequence(seq_dir, n=4, missing_depth_idx=1)  # 3 usable frames
    ds = BonnDataset(
        common_conf=_common_conf(), BONN_DIR=str(tmp_path), subset="full", min_num_images=4
    )
    # rgb.txt lists 4 (passes the cheap construction filter) but only 3 survive
    with pytest.raises(ValueError, match="min_num_images"):
        ds.sequence_num_frames(0)


# --- Bonn integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_sample_schema_and_conventions():
    ds = BonnDataset(
        common_conf=_common_conf(),
        split="test",
        BONN_DIR=BONN_DIR,
        sequences=[SEQ],
        len_test=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert ds.sequence_num_frames(0) == 110         # the pre-associated eval subset

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "bonn_" + SEQ
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    pmask = np.stack(batch["point_masks"])
    sky = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert (depth >= 0).all() and not sky.any()           # indoor: no sky, all-False masks
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True
    assert batch["timestamps"].dtype == np.float64
    assert (batch["timestamps"] > 1.5e9).all()            # real unix capture clocks

    validate_sample(batch, ds.available_modalities)


def _reprojection_closure(pose_frame):
    """Median cross-frame relative depth-reprojection error on a rotation-rich
    pair, end-to-end through process_one_image."""
    ds = BonnDataset(
        common_conf=_eval_common(),
        split="test",
        BONN_DIR=BONN_DIR,
        sequences=[SEQ],
        len_test=10,
        pose_frame=pose_frame,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([0, 100]), aspect_ratio=0.75)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    wA = world[0][pmask[0]]
    E, K = extr[1], intr[1]
    camB = wA @ E[:3, :3].T + E[:3, 3]
    z = camB[:, 2]
    u = camB[:, 0] / z * K[0, 0] + K[0, 2]
    v = camB[:, 1] / z * K[1, 1] + K[1, 2]
    H, W = depth[1].shape
    ok = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.round(u[ok]).astype(int)
    vi = np.round(v[ok]).astype(int)
    measured = depth[1][vi, ui]
    valid = measured > 0
    assert valid.sum() > 500
    rel = np.abs(z[ok][valid] - measured[valid]) / measured[valid]
    return float(np.median(rel))


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_reprojection_closure_with_camera_pose_frame():
    """With the hand-eye-corrected camera poses (default), frame A's world points
    reprojected into frame B match B's depth: validates depth scale + intrinsics +
    pose convention end-to-end. Measured ~2.2% on this pair; the raw marker poses
    give ~25%, so the 5% gate FAILS under the wrong (uncorrected) convention."""
    assert _reprojection_closure("camera") < 0.05


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_reprojection_closure_degrades_with_marker_pose_frame():
    """The asymmetry IS the convention check: the raw OptiTrack marker-body poses
    (published ATE protocol) must NOT reproject-close on a rotating sequence --
    if they did, the hand-eye correction would be a no-op and the survey's
    marker-frame finding wrong."""
    err_marker = _reprojection_closure("marker")
    err_camera = _reprojection_closure("camera")
    assert err_marker > 0.10                # measured ~25%: clearly fails the gate
    assert err_marker > 4 * err_camera     # and is far worse than the corrected poses


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_getitem_tuple_index():
    ds = BonnDataset(
        common_conf=_common_conf(),
        split="test",
        BONN_DIR=BONN_DIR,
        sequences=[SEQ],
        len_test=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_full_subset_on_disk_contains_110_frames():
    """subset='full' lazily associates the raw streams; the pre-associated _110
    clip (full-stream frames 30..139) must come back as a subset of it."""
    ds110 = BonnDataset(
        common_conf=_eval_common(), split="test", BONN_DIR=BONN_DIR,
        sequences=["rgbd_bonn_synchronous"], len_test=10,
    )
    dsf = BonnDataset(
        common_conf=_eval_common(), split="test", BONN_DIR=BONN_DIR,
        sequences=["rgbd_bonn_synchronous"], len_test=10, subset="full",
    )
    assert len(dsf.data_store) == 0                       # lazy: nothing built yet
    n = dsf.sequence_num_frames(0)                        # triggers association
    assert n >= 110
    full_names = {os.path.basename(f[0]) for f in dsf.data_store["rgbd_bonn_synchronous"]}
    names_110 = {os.path.basename(f[0]) for f in ds110.data_store["rgbd_bonn_synchronous"]}
    assert names_110 <= full_names


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _bonn_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _bonn_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Drift guard: the same vendor.get_data + manual tensorize must match byte-for-byte.
    vendor = composed.base_dataset.datasets[0]
    batch = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(ids), aspect_ratio=0.75
    )
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)
    # Order honored: per-frame timestamps follow the requested id order.
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_test), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, "rgbd_bonn_crowd"]
    composed = instantiate(
        _bonn_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name]) == 110


@pytest.mark.skipif(not HAVE_BONN, reason=f"Bonn data not found at {BONN_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _bonn_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # Bonn native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
