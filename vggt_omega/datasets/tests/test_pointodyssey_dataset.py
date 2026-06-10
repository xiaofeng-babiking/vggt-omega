import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.pointodyssey import PointOdysseyDataset

POINTODYSSEY_DIR = "/jfs/Data_4DFF/train_data/pointodyssey"
HAVE_PO = os.path.isdir(POINTODYSSEY_DIR)
# "ani" is a train sequence with real camera motion (survey-verified reprojection
# closure); restricting `sequences` keeps integration tests fast.
SEQ = "ani"


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


def _po_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.pointodyssey.PointOdysseyDataset",
                "split": "train",
                "POINTODYSSEY_DIR": POINTODYSSEY_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- PointOdyssey-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = PointOdysseyDataset.pointodyssey_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_inverts_c2w():
    # 90-degree rotation about z + translation: w2c must be [R^T | -R^T t]
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.5, -1.5, 2.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = PointOdysseyDataset.pointodyssey_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], R.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -R.T @ t, atol=1e-6)
    # composing w2c after c2w must give identity (round-trips a camera point)
    pt_cam = np.array([0.3, 0.7, 2.0, 1.0])
    pt_world = c2w @ pt_cam
    np.testing.assert_allclose(w2c @ pt_world, pt_cam[:3], atol=1e-5)


def test_pose_to_w2c_rejects_bad_pose():
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        PointOdysseyDataset.pointodyssey_pose_to_w2c(bad)
    with pytest.raises(ValueError, match="4,4"):
        PointOdysseyDataset.pointodyssey_pose_to_w2c(np.eye(3))


def test_depth_reader_meters_and_invalid(tmp_path):
    arr = np.array([[0.0, 1.5], [np.inf, -2.0]], dtype=np.float32)
    p = tmp_path / "00000.npy"
    np.save(p, arr)
    depth = PointOdysseyDataset.read_pointodyssey_depth(str(p))
    assert depth.dtype == np.float32
    # already meters (no scaling); non-finite and negative junk -> 0 (invalid)
    np.testing.assert_allclose(depth, [[0.0, 1.5], [0.0, 0.0]])
    np.save(p, np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="expected \\(H,W\\)"):
        PointOdysseyDataset.read_pointodyssey_depth(str(p))


def test_cam_reader_pose_and_intrinsics(tmp_path):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    K = np.array([[576.0, 0.0, 480.0], [0.0, 576.0, 270.0], [0.0, 0.0, 1.0]], np.float32)
    p = tmp_path / "00000.npz"
    np.savez(p, pose=c2w, intrinsics=K)
    c2w_r, K_r = PointOdysseyDataset.read_pointodyssey_cam(str(p))
    np.testing.assert_allclose(c2w_r, c2w)
    np.testing.assert_allclose(K_r, K)
    # missing keys -> clear ValueError
    p2 = tmp_path / "00001.npz"
    np.savez(p2, pose=c2w)
    with pytest.raises(ValueError, match="intrinsics"):
        PointOdysseyDataset.read_pointodyssey_cam(str(p2))


def test_intrinsics_assembly_override_and_error():
    K_raw = np.array([[576.0, 0.0, 480.0], [0.0, 576.0, 270.0], [0.0, 0.0, 1.0]])
    K = PointOdysseyDataset.pointodyssey_intrinsics(K_raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_raw)
    K2 = PointOdysseyDataset.pointodyssey_intrinsics(K_raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0  # override wins over K_raw
    with pytest.raises(ValueError, match="intrinsics"):
        PointOdysseyDataset.pointodyssey_intrinsics(None)
    with pytest.raises(ValueError, match="3,3"):
        PointOdysseyDataset.pointodyssey_intrinsics(np.eye(4))


def test_constructor_rejects_missing_dir_and_bad_split():
    with pytest.raises(ValueError, match="POINTODYSSEY_DIR"):
        PointOdysseyDataset(common_conf=_common_conf(), POINTODYSSEY_DIR=None)
    with pytest.raises(ValueError, match="split"):
        PointOdysseyDataset(
            common_conf=_common_conf(), split="bogus", POINTODYSSEY_DIR="/nonexistent"
        )


# --- PointOdyssey integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_pointodyssey_sample_schema_and_conventions():
    ds = PointOdysseyDataset(
        common_conf=_common_conf(),
        split="train",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities  # sky/invalid conflated at 0

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    pmask = np.stack(batch["point_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth >= 0).all()                             # 0=invalid; no sky encoding
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "pointodyssey_" + SEQ
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_reprojection_closure_across_frames():
    """World points from two nearby frames must be mutually consistent: frame A's
    valid world points reprojected into frame B land at depths matching B's depth
    map (locks depth-scale x pose-convention x intrinsics end-to-end through
    process_one_image). PointOdyssey scenes are dynamic, so nearby frames + the
    median keep moving content from dominating the statistic."""
    ds = PointOdysseyDataset(
        common_conf=_eval_common(),
        split="train",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([100, 120]), aspect_ratio=0.75)
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
    rel_err = np.abs(z[ok][valid] - measured[valid]) / measured[valid]
    assert valid.sum() >= 500
    assert np.median(rel_err) < 0.05


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_getitem_tuple_index():
    ds = PointOdysseyDataset(
        common_conf=_common_conf(),
        split="train",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_split_dirs_and_sequence_filtering():
    """split maps to the train/val/test dirs; `sequences` glob patterns filter the
    cheap name enumeration; min_num_images is enforced lazily on first access."""
    val_ds = PointOdysseyDataset(
        common_conf=_common_conf(), split="val",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR, len_test=10,
    )
    test_ds = PointOdysseyDataset(
        common_conf=_common_conf(), split="test",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR, len_test=10,
    )
    assert val_ds.sequence_list_len > 0 and test_ds.sequence_list_len > 0
    assert val_ds.len_train == 10 and test_ds.len_train == 10  # len_test for non-train

    glob_ds = PointOdysseyDataset(
        common_conf=_common_conf(), split="train",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR, sequences=["ani1*"], len_train=10,
    )
    assert glob_ds.sequence_list_len > 0
    assert all(name.startswith("ani1") for name in glob_ds.sequence_list)

    with pytest.raises(ValueError, match="No usable PointOdyssey"):
        PointOdysseyDataset(
            common_conf=_common_conf(), split="train",
            POINTODYSSEY_DIR=POINTODYSSEY_DIR, sequences=["no_such_seq_*"],
        )

    short = PointOdysseyDataset(
        common_conf=_common_conf(), split="train",
        POINTODYSSEY_DIR=POINTODYSSEY_DIR, sequences=[SEQ],
        len_train=10, min_num_images=10**9,
    )
    with pytest.raises(ValueError, match="only .* frames"):
        short.sequence_num_frames(0)   # lazy min_num_images enforcement


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _po_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _po_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]                                     # deliberately unordered
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Order honored verbatim: synthesized timestamps are frame_index / 30, a
    # per-frame quantity, so they must follow the requested (unordered) ids.
    np.testing.assert_allclose(
        sample["timestamps"].numpy(), np.array(ids, dtype=np.float64) / 30.0
    )

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
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, "ani2_s"]
    composed = instantiate(
        _po_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        local_idx = vendor.sequence_list.index(name)
        n = composed.sequence_num_frames(gi)
        assert n == vendor.sequence_num_frames(local_idx)
        assert n >= vendor.min_num_images


@pytest.mark.skipif(not HAVE_PO, reason=f"PointOdyssey data not found at {POINTODYSSEY_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _po_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (540, 960)                       # PointOdyssey native (H, W)

    composed.set_img_size(960)                        # native long side
    assert composed.img_size == 960
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # 540 is not /16-divisible, so the short side snaps to 528
    assert tuple(s["images"].shape[-2:]) == (528, 960)

    composed.set_img_size(480)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (256, 480)
