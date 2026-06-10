import os
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.spring import SpringDataset

SPRING_DIR = "/jfs/Data_4DFF/train_data/spring"
HAVE_SPRING = os.path.isdir(SPRING_DIR)
# "0001" (248 frames) for the main integration tests; "0027" is the shortest (13 frames).
SEQ = "0001"


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


def _spring_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.spring.SpringDataset",
                "split": "train",
                "SPRING_DIR": SPRING_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- Spring-specific helper unit tests (no data required) ---


def test_spring_pose_to_w2c_identity():
    w2c = SpringDataset.spring_pose_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_spring_pose_to_w2c_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = SpringDataset.spring_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_spring_pose_to_w2c_rotation_inverts_c2w():
    # 90 deg rotation about z plus a translation: w2c must be the exact inverse.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = t
    w2c = SpringDataset.spring_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], rot.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -rot.T @ t, atol=1e-6)
    # The camera center must map to the camera-frame origin.
    np.testing.assert_allclose(w2c[:3, :3] @ t + w2c[:, 3], np.zeros(3), atol=1e-6)


def test_spring_pose_to_w2c_rejects_bad():
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        SpringDataset.spring_pose_to_w2c(bad)
    with pytest.raises(ValueError, match="expected"):
        SpringDataset.spring_pose_to_w2c(np.eye(3))


def test_spring_depth_reader_units_and_invalid(tmp_path):
    # Meters with scale 1.0; nan/inf/negatives map to 0 (invalid), 0 stays 0.
    arr = np.array([[0.0, 1.5], [np.nan, -2.0]], dtype=np.float32)
    p = tmp_path / "0000.npy"
    np.save(p, arr)
    depth = SpringDataset.read_spring_depth(str(p))
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.5], [0.0, 0.0]])


def test_spring_depth_reader_rejects_non_2d(tmp_path):
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        SpringDataset.read_spring_depth(str(p))


def test_spring_intrinsics_passthrough_override_and_error():
    K_raw = np.array(
        [[1090.909, 0.0, 479.75], [0.0, 1090.909, 269.75], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K = SpringDataset.spring_intrinsics(K_raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_raw, rtol=1e-6)
    K2 = SpringDataset.spring_intrinsics(K_raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="expected"):
        SpringDataset.spring_intrinsics(np.eye(4))
    bad = K_raw.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SpringDataset.spring_intrinsics(bad)


def test_spring_cam_reader_roundtrip_and_missing_key(tmp_path):
    K = np.array(
        [[646.46, 0.0, 479.75], [0.0, 646.46, 269.75], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    p = tmp_path / "0000.npz"
    np.savez(p, intrinsics=K, pose=c2w)
    K_out, w2c = SpringDataset.read_spring_cam(str(p))
    np.testing.assert_allclose(K_out, K, rtol=1e-6)
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-5)
    p2 = tmp_path / "bad.npz"
    np.savez(p2, pose=c2w)  # missing 'intrinsics'
    with pytest.raises(ValueError, match="keys"):
        SpringDataset.read_spring_cam(str(p2))


# --- Spring integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_spring_sample_schema_and_conventions():
    ds = SpringDataset(
        common_conf=_common_conf(),
        split="train",
        SPRING_DIR=SPRING_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    # Spring quirks: no sky sentinel and no timestamps -> not advertised.
    assert Modality.SKY_MASK not in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "spring_" + SEQ
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
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert (depth >= 0).all()                             # no sky sentinel in Spring
    assert "sky_masks" not in batch                       # unadvertised -> key not emitted
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_spring_cross_frame_world_points_agree():
    """World points from two frames must be mutually consistent: frame A's valid
    world points reprojected into frame B land at depths matching B's depth map
    (locks depth-scale x pose-convention x intrinsics consistency end-to-end
    through process_one_image). Pair and threshold are calibrated adversarially
    so a c2w/w2c flip CANNOT pass: on ids [0, 30] of seq 0001 the correct recipe
    (w2c = inv(pose)) measures median rel err 0.00079 while skipping the
    inversion measures 0.312 -- the 0.005 bound sits >6x above the former and
    >60x below the latter. (A narrow pair like [0, 5] is useless here: small
    inter-frame motion lets the flipped recipe close to 0.016.)"""
    ds = SpringDataset(
        common_conf=_eval_common(),
        split="train",
        SPRING_DIR=SPRING_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([0, 30]), aspect_ratio=0.75)
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
    assert np.median(rel_err) < 0.005


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_spring_getitem_tuple_index():
    ds = SpringDataset(
        common_conf=_common_conf(),
        split="train",
        SPRING_DIR=SPRING_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_spring_lazy_enumeration_and_short_sequences():
    """Construction lists only sequence NAMES (one directory listing); per-sequence
    frame lists stay unloaded until first access. Spring's sub-min_num_images
    sequences (e.g. 0027 with 13 frames) are kept and usable with explicit ids."""
    t0 = time.monotonic()
    ds = SpringDataset(
        common_conf=_eval_common(),
        split="train",
        SPRING_DIR=SPRING_DIR,
        len_train=10,
    )
    elapsed = time.monotonic() - t0
    assert ds.sequence_list_len == 37                       # all train sequences, gaps respected
    assert "0003" not in ds.sequence_list                   # numbering gap: enumerate, not range
    assert elapsed < 10.0                                   # names only; no frame enumeration
    assert all(v is None for v in ds.data_store.values())   # nothing eagerly enumerated

    i = ds.sequence_list.index("0027")
    assert ds.sequence_num_frames(i) == 13                  # shortest seq, kept (< min_num_images)
    assert ds.data_store["0027"] is not None                # cached after first access
    assert ds.data_store[SEQ] is None                       # other sequences still lazy

    b = ds.get_data(seq_name="0027", ids=np.array([0, 12]), aspect_ratio=1.0)
    assert b["frame_num"] == 2

    # Random sampling without replacement beyond the frame count must fail with
    # an error NAMING the sequence (not numpy's opaque population error).
    # _eval_common sets allow_duplicate_img=False.
    with pytest.raises(ValueError, match=r"0027.*20 distinct.*13"):
        ds.get_data(seq_name="0027", img_per_seq=20, aspect_ratio=1.0)


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_spring_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _spring_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" not in sample               # unadvertised -> key not emitted
    assert "timestamps" not in sample              # Spring ships no timestamps
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _spring_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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

    # Order honored verbatim: per-frame extrinsics (the camera moves every frame)
    # of the reversed-id request are exactly the reversed per-frame extrinsics.
    sample_rev = composed.get_sample(0, ids=ids[::-1], aspect_ratio=0.75)
    np.testing.assert_allclose(
        sample["extrinsics"].numpy(), sample_rev["extrinsics"].numpy()[::-1]
    )
    # And position k corresponds to frame ids[k]: a singleton fetch of id 7
    # reproduces position 1 of the unordered request.
    single = composed.get_sample(0, ids=[7], aspect_ratio=0.75)
    np.testing.assert_allclose(
        sample["extrinsics"].numpy()[1], single["extrinsics"].numpy()[0]
    )
    # The poses genuinely differ across frames (the order proof is non-trivial).
    assert not np.allclose(sample["extrinsics"].numpy()[0], sample["extrinsics"].numpy()[1])


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = ["0001", "0012"]
    composed = instantiate(
        _spring_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        local = vendor.sequence_list.index(name)
        assert composed.sequence_num_frames(gi) == vendor.sequence_num_frames(local)
        assert composed.sequence_num_frames(gi) >= 13


@pytest.mark.skipif(not HAVE_SPRING, reason=f"Spring data not found at {SPRING_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _spring_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (540, 960)                       # Spring export native (H, W)

    composed.set_img_size(960)                        # native long side
    assert composed.img_size == 960
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # 540 is not divisible by patch_size=16, so the short side snaps to 528.
    assert tuple(s["images"].shape[-2:]) == (528, 960)

    composed.set_img_size(480)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (256, 480)   # int(480*540/960)=270 -> 256
