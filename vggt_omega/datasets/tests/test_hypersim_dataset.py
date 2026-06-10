import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.hypersim import HypersimDataset

HYPERSIM_DIR = "/jfs/Data_4DFF/train_data/hypersim"
HAVE_HYPERSIM = os.path.isdir(HYPERSIM_DIR)
# ai_001_001 has a single cam dir (cam_00, 98 frames); keep construction fast.
SCENE = "ai_001_001"
SEQ = "ai_001_001/cam_00"


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


def _hypersim_dataset_cfg(seqs=(SCENE,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.hypersim.HypersimDataset",
                "split": "train",
                "HYPERSIM_DIR": HYPERSIM_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- Hypersim-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = HypersimDataset.hypersim_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_roundtrip():
    # 90-degree rotation about z plus a translation: w2c must invert c2w exactly.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, -2.0, 0.5])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = t
    w2c = HypersimDataset.hypersim_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], rot.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -rot.T @ t, atol=1e-6)
    # a world point on the camera center maps to the camera origin
    np.testing.assert_allclose(w2c[:3, :3] @ t + w2c[:, 3], np.zeros(3), atol=1e-6)


def test_pose_to_w2c_rejects_bad():
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        HypersimDataset.hypersim_pose_to_w2c(bad)
    with pytest.raises(ValueError, match="4,4"):
        HypersimDataset.hypersim_pose_to_w2c(np.eye(3))


def test_depth_reader_units_invalid_and_shape(tmp_path):
    arr = np.array([[1.5, np.nan], [np.inf, 65.9]], dtype=np.float32)
    p = tmp_path / "000000_depth.npy"
    np.save(p, arr)
    depth = HypersimDataset.read_hypersim_depth(str(p))
    assert depth.dtype == np.float32
    # values already meters (no scaling); NaN/inf -> 0 invalid; never negative
    np.testing.assert_allclose(depth, [[1.5, 0.0], [0.0, 65.9]])

    np.save(tmp_path / "bad_depth.npy", np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        HypersimDataset.read_hypersim_depth(str(tmp_path / "bad_depth.npy"))


def test_intrinsics_passthrough_override_and_error():
    raw = np.array([[886.81, -0.0, 512.0], [0.0, 886.81, 384.0], [0.0, 0.0, 1.0]])
    K = HypersimDataset.hypersim_intrinsics(raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, raw.astype(np.float32))
    K2 = HypersimDataset.hypersim_intrinsics(raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        HypersimDataset.hypersim_intrinsics(np.zeros(4))
    with pytest.raises(ValueError, match="intrinsics"):
        HypersimDataset.hypersim_intrinsics(None)


def test_read_hypersim_cam_and_missing_keys(tmp_path):
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    K_raw = np.array([[886.81, 0.0, 512.0], [0.0, 886.81, 384.0], [0.0, 0.0, 1.0]], np.float32)
    p = tmp_path / "000000_cam.npz"
    np.savez(p, pose=c2w, intrinsics=K_raw)

    K, w2c = HypersimDataset.read_hypersim_cam(str(p))
    assert K.shape == (3, 3) and w2c.shape == (3, 4)
    np.testing.assert_allclose(K, K_raw)
    np.testing.assert_allclose(w2c[:3, :3], rot.T, atol=1e-6)
    # override wins over the stored K
    K_o, _ = HypersimDataset.read_hypersim_cam(str(p), intrinsics_override=[10.0, 20.0, 1.0, 2.0])
    assert K_o[0, 0] == 10.0 and K_o[1, 1] == 20.0

    p_bad = tmp_path / "missing_cam.npz"
    np.savez(p_bad, pose=c2w)
    with pytest.raises(ValueError, match="keys"):
        HypersimDataset.read_hypersim_cam(str(p_bad))


# --- Hypersim integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_hypersim_sample_schema_and_conventions():
    ds = HypersimDataset(
        common_conf=_common_conf(),
        split="train",
        HYPERSIM_DIR=HYPERSIM_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    assert ds.sequence_list == [SEQ]
    assert Modality.EXTRINSICS in ds.available_modalities
    # reprojected depth must NOT be advertised as point-cloud GT; no sky labels
    assert Modality.WORLD_POINTS not in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities

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
    sky = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert (depth >= 0).all()                             # NaN mapped to 0, never negative
    assert not sky.any()                                  # no sky labels in this copy
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "hypersim_" + SEQ
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_hypersim_reprojection_closure():
    """World points from two nearby frames must be mutually consistent: frame A's
    valid world points reprojected into frame B land at depths matching B's depth
    map (locks depth-scale x pose-convention x intrinsics end-to-end through
    process_one_image)."""
    ds = HypersimDataset(
        common_conf=_eval_common(),
        split="train",
        HYPERSIM_DIR=HYPERSIM_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([0, 2]), aspect_ratio=0.75)
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


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_hypersim_getitem_tuple_index():
    ds = HypersimDataset(
        common_conf=_common_conf(),
        split="train",
        HYPERSIM_DIR=HYPERSIM_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_hypersim_skips_empty_and_short_cam_dirs():
    """36 of the 793 cam dirs are completely empty (e.g. ai_017_009/cam_01) and a
    few hold fewer than min_num_images frames (ai_011_010/cam_00 has 9); both must
    be dropped at construction, and an all-unusable selection must raise."""
    ds = HypersimDataset(
        common_conf=_common_conf(),
        split="train",
        HYPERSIM_DIR=HYPERSIM_DIR,
        sequences=["ai_017_009"],
        len_train=10,
    )
    assert "ai_017_009/cam_01" not in ds.sequence_list      # empty dir skipped
    assert ds.sequence_list == [
        "ai_017_009/cam_00", "ai_017_009/cam_02", "ai_017_009/cam_03",
    ]
    with pytest.raises(ValueError, match="No usable"):
        HypersimDataset(
            common_conf=_common_conf(),
            split="train",
            HYPERSIM_DIR=HYPERSIM_DIR,
            sequences=["ai_011_010"],   # single cam dir with only 9 frames
            len_train=10,
        )


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_hypersim_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _hypersim_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extra batch modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _hypersim_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    seq_name = composed.sequence_name(0)
    batch = vendor.get_data(seq_name=seq_name, ids=np.array(ids), aspect_ratio=0.75)
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)

    # Order honored, proven via the per-frame extrinsics (the camera moves, so
    # every frame's pose is distinct): each position must equal an independent
    # single-frame fetch of exactly that id (deterministic under _eval_common).
    extr = sample["extrinsics"].numpy()
    assert not np.allclose(extr[1], extr[2])                # frames truly differ
    for k in (1, 2):                                        # ids 7 and 3, out of order
        single = vendor.get_data(
            seq_name=seq_name, ids=np.array([ids[k]]), aspect_ratio=0.75
        )
        np.testing.assert_allclose(extr[k], single["extrinsics"][0], atol=1e-6)


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate
    sequences; frame counts are listed lazily per sequence and cached."""
    from hydra.utils import instantiate

    composed = instantiate(
        _hypersim_dataset_cfg(seqs=[SCENE, "ai_001_002"]),
        common_config=_eval_common(),
        _recursive_=False,
    )
    vendor = composed.base_dataset.datasets[0]
    # ai_001_001 has 1 cam dir, ai_001_002 has 4
    assert composed.num_sequences() == len(vendor.sequence_list) == 5
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in vendor.sequence_list
        n = composed.sequence_num_frames(gi)
        assert n >= 24
        assert n == len(vendor.data_store[name])    # lazy frame list now cached
    # spot-check a known count (frame indices are non-contiguous: 98 over 000000..000099)
    assert vendor.sequence_num_frames(vendor.sequence_list.index(SEQ)) == 98


@pytest.mark.skipif(not HAVE_HYPERSIM, reason=f"Hypersim data not found at {HYPERSIM_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _hypersim_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (768, 1024)                      # Hypersim native (H, W)

    composed.set_img_size(1024)                       # native long side
    assert composed.img_size == 1024
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (768, 1024)

    composed.set_img_size(512)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (384, 512)
