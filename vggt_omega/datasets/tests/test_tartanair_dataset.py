import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.tartanair import TartanAirDataset

TARTANAIR_DIR = "/jfs/Data_4DFF/train_data/tartanair"
HAVE_TARTANAIR = os.path.isdir(TARTANAIR_DIR)
# A single 2176-frame outdoor sequence keeps integration tests fast; frames
# >= ~400 contain sky (depth > 10000 m) for the sky-encoding tests.
SEQ = "abandonedfactory/Easy/P000"
SEQ2 = "abandonedfactory/Easy/P001"


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


def _tartanair_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.tartanair.TartanAirDataset",
                "split": "train",
                "TARTANAIR_DIR": TARTANAIR_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- TartanAir-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity():
    w2c = TartanAirDataset.tartanair_pose_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_pose_to_w2c_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = TartanAirDataset.tartanair_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_no_axis_remap():
    """The dump's camera_pose is already OpenCV camera-to-world, so w2c must be
    the plain rigid inverse [R^T | -R^T t] with NO NED axis remap."""
    # 90-degree rotation about z plus a translation
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = TartanAirDataset.tartanair_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], R.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -R.T @ t, atol=1e-6)
    # round trip: w2c applied to the camera center lands at the origin
    np.testing.assert_allclose(w2c[:3, :3] @ t + w2c[:, 3], np.zeros(3), atol=1e-6)


def test_pose_to_w2c_rejects_bad_inputs():
    with pytest.raises(ValueError, match="expected"):
        TartanAirDataset.tartanair_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        TartanAirDataset.tartanair_pose_to_w2c(bad)


def test_decode_depth_valid_sky_invalid():
    arr = np.array(
        [[2.5, 0.5, 16288.0], [5000.0, np.nan, np.inf]], dtype=np.float32
    )
    depth = TartanAirDataset.decode_tartanair_depth(arr)
    # valid stays metric; sky (>10000 or +inf) -> -1; ambiguous band & nan -> 0
    np.testing.assert_allclose(depth, [[2.5, 0.5, -1.0], [0.0, 0.0, -1.0]])
    assert depth.dtype == np.float32


def test_decode_depth_custom_thresholds_and_negatives():
    arr = np.array([[10.0, 50.0, -3.0]], dtype=np.float32)
    depth = TartanAirDataset.decode_tartanair_depth(arr, valid_max=20.0, sky_threshold=40.0)
    np.testing.assert_allclose(depth, [[10.0, -1.0, 0.0]])


def test_read_depth_npy(tmp_path):
    arr = np.array([[1.5, 12000.0], [3.0, 7.25]], dtype=np.float32)
    p = tmp_path / "000000_depth.npy"
    np.save(p, arr)
    depth = TartanAirDataset.read_tartanair_depth(str(p))
    np.testing.assert_allclose(depth, [[1.5, -1.0], [3.0, 7.25]])


def test_read_cam_npz_and_missing_key(tmp_path):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    p = tmp_path / "000000_cam.npz"
    np.savez(p, camera_pose=c2w, camera_intrinsics=np.eye(3, dtype=np.float32))
    w2c = TartanAirDataset.read_tartanair_cam(str(p))
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)

    p2 = tmp_path / "000001_cam.npz"
    np.savez(p2, not_a_pose=np.eye(4))
    with pytest.raises(ValueError, match="camera_pose"):
        TartanAirDataset.read_tartanair_cam(str(p2))


def test_intrinsics_default_override_and_error():
    K = TartanAirDataset.tartanair_intrinsics()
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    assert K[0, 0] == 320.0 and K[1, 1] == 320.0
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0
    K2 = TartanAirDataset.tartanair_intrinsics(override=[100.0, 110.0, 50.0, 40.0])
    assert K2[0, 0] == 100.0 and K2[1, 1] == 110.0 and K2[0, 2] == 50.0 and K2[1, 2] == 40.0
    with pytest.raises(ValueError, match="fx, fy, cx, cy"):
        TartanAirDataset.tartanair_intrinsics(override=[100.0, 110.0])


def test_match_sequences_prefix_and_glob():
    names = [
        "abandonedfactory/Easy/P000",
        "abandonedfactory/Easy/P001",
        "abandonedfactory/Hard/P000",
        "ocean/Easy/P002",
    ]
    m = TartanAirDataset._match_sequences
    assert m(names, None) == names                              # default: all
    assert m(names, ["abandonedfactory/Easy/P000"]) == names[:1]  # exact
    assert m(names, ["abandonedfactory/Easy"]) == names[:2]      # difficulty prefix
    assert m(names, ["abandonedfactory"]) == names[:3]           # env prefix
    assert m(names, ["*/Hard"]) == [names[2]]                    # glob on prefix
    assert m(names, ["ocean", "abandonedfactory/Hard/*"]) == [names[2], names[3]]


def test_lazy_construction_and_min_num_images(tmp_path):
    """Construction discovers sequence NAMES only (no per-frame listing); a
    too-thin sequence raises a clear ValueError on first frame access."""
    seq_dir = tmp_path / "train" / "envA" / "Easy" / "P000"
    seq_dir.mkdir(parents=True)
    for i in range(3):
        (seq_dir / f"{i:06d}_rgb.png").write_bytes(b"")
        (seq_dir / f"{i:06d}_depth.npy").write_bytes(b"")
        (seq_dir / f"{i:06d}_cam.npz").write_bytes(b"")
    ds = TartanAirDataset(
        common_conf=_common_conf(), split="train", TARTANAIR_DIR=str(tmp_path)
    )
    assert ds.sequence_list == ["envA/Easy/P000"]
    assert ds._frames_cache == {}          # nothing enumerated at construction
    with pytest.raises(ValueError, match="min_num_images"):
        ds.sequence_num_frames(0)
    # with a lower threshold the lazy listing succeeds and is cached
    ds2 = TartanAirDataset(
        common_conf=_common_conf(), split="train", TARTANAIR_DIR=str(tmp_path),
        min_num_images=2,
    )
    assert ds2.sequence_num_frames(0) == 3
    assert "envA/Easy/P000" in ds2._frames_cache


def test_construction_requires_dir_and_sequences():
    with pytest.raises(ValueError, match="TARTANAIR_DIR"):
        TartanAirDataset(common_conf=_common_conf(), split="train")


# --- TartanAir integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_tartanair_sample_schema_and_conventions():
    ds = TartanAirDataset(
        common_conf=_common_conf(),
        split="train",
        TARTANAIR_DIR=TARTANAIR_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities  # no clock in this dump

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
    smask = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert depth[depth > 0].max() < 1000                  # sky was remapped, not left huge
    assert smask.dtype == bool and (smask == (depth < 0)).all()
    assert not (pmask & smask).any()                      # sky never counts as valid
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_tartanair_sky_frames_have_negative_depth():
    """Frame 800 of abandonedfactory/Easy/P000 is ~40% sky (raw depth ~16288 m):
    the vendor must encode it as depth == -1 (sky_masks True), never as huge
    positive depth, and exclude it from point_masks."""
    ds = TartanAirDataset(
        common_conf=_eval_common(),
        split="train",
        TARTANAIR_DIR=TARTANAIR_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([800]), aspect_ratio=0.75)
    depth = b["depths"][0]
    sky = b["sky_masks"][0]
    assert sky.mean() > 0.1                               # plenty of sky in this frame
    np.testing.assert_allclose(depth[sky], -1.0)
    assert not b["point_masks"][0][sky].any()
    assert depth[depth > 0].max() < 1000                  # no leftover sky-scale values


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_tartanair_reprojection_closure():
    """World points from frame A reprojected into nearby frame B must match B's
    depth map (locks depth-scale x pose-convention x intrinsics consistency
    end-to-end through process_one_image)."""
    ds = TartanAirDataset(
        common_conf=_eval_common(),
        split="train",
        TARTANAIR_DIR=TARTANAIR_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([10, 13]), aspect_ratio=0.75)
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


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_tartanair_getitem_tuple_index():
    ds = TartanAirDataset(
        common_conf=_common_conf(),
        split="train",
        TARTANAIR_DIR=TARTANAIR_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_tartanair_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _tartanair_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert "timestamps" not in sample              # not advertised: no clock in dump
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _tartanair_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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

    # Order honored verbatim: position k carries frame ids[k]'s pose. Extrinsics
    # pass through process_one_image untouched (no rotate with landscape_check
    # off), so compare against the raw w2c read straight from that frame's npz.
    # Camera translation moves 6-14 cm/frame, so frames are distinguishable.
    seq_dir = vendor.data_store[composed.sequence_name(0)]
    for k, fid in enumerate(ids):
        w2c = TartanAirDataset.read_tartanair_cam(
            os.path.join(seq_dir, f"{fid:06d}_cam.npz")
        )
        np.testing.assert_allclose(sample["extrinsics"][k].numpy(), w2c, atol=1e-6)
    np.testing.assert_array_equal(sample["ids"].numpy(), np.array(ids))


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, SEQ2]
    composed = instantiate(
        _tartanair_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor._get_frames(name))
        assert composed.sequence_num_frames(gi) >= 300   # survey: thinnest seq ~300


@pytest.mark.skipif(not HAVE_TARTANAIR, reason=f"TartanAir data not found at {TARTANAIR_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _tartanair_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # TartanAir native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
