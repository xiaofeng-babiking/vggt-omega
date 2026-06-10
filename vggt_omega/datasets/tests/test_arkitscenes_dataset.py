import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.arkitscenes import ArkitScenesDataset

ARKITSCENES_DIR = "/jfs/Data_4DFF/train_data/arkitscenes"
HAVE_ARKIT = os.path.isdir(ARKITSCENES_DIR)
# Small Training scenes for fast integration tests: 40753679 is landscape
# (640x480, 405 frames), 47115118 is portrait (480x640, 497 frames).
SCENE = "40753679"
SCENE_B = "40753686"
SCENE_PORTRAIT = "47115118"
TEST_SCENE = "41126307"


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


def _arkit_dataset_cfg(seqs=(SCENE,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.arkitscenes.ArkitScenesDataset",
                "split": "train",
                "ARKITSCENES_DIR": ARKITSCENES_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- ARKitScenes-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = ArkitScenesDataset.arkit_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_inverts_rotation():
    # camera-to-world = 90 deg about z + translation; w2c must be the exact inverse
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    c2w = np.array(
        [[c, -s, 0, 0.5], [s, c, 0, -1.5], [0, 0, 1, 2.0], [0, 0, 0, 1]], dtype=np.float64
    )
    w2c = ArkitScenesDataset.arkit_pose_to_w2c(c2w)
    full = np.vstack([w2c, [0, 0, 0, 1]])
    np.testing.assert_allclose(full @ c2w, np.eye(4), atol=1e-6)


def test_pose_to_w2c_rejects_bad_shape_and_non_finite():
    with pytest.raises(ValueError, match="4,4"):
        ArkitScenesDataset.arkit_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        ArkitScenesDataset.arkit_pose_to_w2c(bad)


def test_depth_reader_units_and_invalid(tmp_path):
    import cv2

    arr = np.array([[0, 1000], [2000, 7000]], dtype=np.uint16)
    p = str(tmp_path / "40753679_6790.148.png")
    assert cv2.imwrite(p, arr)
    depth = ArkitScenesDataset.read_arkit_depth(p)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.0], [2.0, 7.0]])  # mm -> m, 0 invalid
    with pytest.raises(FileNotFoundError):
        ArkitScenesDataset.read_arkit_depth(str(tmp_path / "missing.png"))


def test_intrinsics_assembly_override_and_error():
    # row = [w, h, fx, fy, cx, cy] (per-frame ARKit calibration)
    K = ArkitScenesDataset.arkit_intrinsics([640.0, 480.0, 532.931, 532.9, 316.4, 241.7])
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K[0, 0], 532.931)
    np.testing.assert_allclose(K[1, 1], 532.9)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [316.4, 241.7])
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0
    K2 = ArkitScenesDataset.arkit_intrinsics(None, override=[100.0, 100.0, 50.0, 40.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0 and K2[1, 2] == 40.0
    with pytest.raises(ValueError, match="got 4 values"):
        ArkitScenesDataset.arkit_intrinsics([532.0, 532.0, 316.0, 241.0])


def test_parse_timestamp():
    assert ArkitScenesDataset.parse_arkit_timestamp("40753679_6790.148.png") == 6790.148
    assert ArkitScenesDataset.parse_arkit_timestamp("40753679_6790.148.jpg") == 6790.148
    assert ArkitScenesDataset.parse_arkit_timestamp("/a/b/40753679_0.067.png") == 0.067
    with pytest.raises(ValueError, match="timestamp"):
        ArkitScenesDataset.parse_arkit_timestamp("noseparator.png")
    with pytest.raises(ValueError, match="timestamp"):
        ArkitScenesDataset.parse_arkit_timestamp("scene_notanumber.png")


# --- ARKitScenes integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_sample_schema_and_conventions():
    ds = ArkitScenesDataset(
        common_conf=_common_conf(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    # Indoor data: SKY_MASK is advertised with all-False masks (tum/7scenes
    # convention); reprojected depth must never be advertised as point-cloud GT.
    assert Modality.SKY_MASK in ds.available_modalities
    assert Modality.WORLD_POINTS not in ds.available_modalities
    assert Modality.CAM_POINTS not in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "arkitscenes_" + SCENE
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
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True
    assert not np.stack(batch["sky_masks"]).any()         # indoor: no sky

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_timestamps_sorted_and_per_frame_intrinsics():
    """Frames must come out time-sorted (the on-disk npz arrays are NOT sorted),
    and intrinsics must be the per-frame ARKit calibration (not one per-scene K,
    which drifts up to ~17 px within a scene)."""
    ds = ArkitScenesDataset(
        common_conf=_eval_common(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    n = ds.sequence_num_frames(0)
    ids = np.linspace(0, n - 1, 12).astype(int)
    batch = ds.get_data(seq_name=SCENE, ids=ids, aspect_ratio=0.75)
    ts = batch["timestamps"]
    assert ts.dtype == np.float64
    assert np.all(np.diff(ts) > 0)                       # strictly increasing
    # ~10 Hz capture, but scenes contain gaps, so only bound the average rate loosely.
    assert 0.05 < np.median(np.diff(ts) / np.diff(ids)) < 0.5
    # Per-frame calibration: across the scene the focals are not all identical.
    intr = np.stack(batch["intrinsics"])
    assert np.unique(intr[:, 0, 0]).size > 1


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
@pytest.mark.parametrize(
    "scene,ids",
    [(SCENE, [0, 5]), (SCENE_PORTRAIT, [10, 12])],
    ids=["landscape", "portrait"],
)
def test_arkit_reprojection_closure(scene, ids):
    """World points from two nearby frames must be mutually consistent: frame A's
    valid world points reprojected into frame B land at depths matching B's depth
    map. Locks depth-scale x pose-convention x per-frame-intrinsics consistency
    end-to-end through process_one_image.

    The 0.02 threshold (NOT looser) and the portrait scene are both load-bearing:
    with the FLIPPED pose convention (stored c2w passed through as w2c) the median
    relative error measured through this exact pipeline is 0.040 on the landscape
    pair -- which would PASS a 0.05 threshold -- and 0.217 on the portrait pair;
    the correct convention gives 0.0034 / 0.0089. Only <0.02 on both scenes
    separates the conventions with margin in both directions."""
    ds = ArkitScenesDataset(
        common_conf=_eval_common(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[scene],
        len_train=10,
    )
    b = ds.get_data(seq_name=scene, ids=np.array(ids), aspect_ratio=0.75)
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
    assert np.median(rel_err) < 0.02


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_getitem_tuple_index():
    ds = ArkitScenesDataset(
        common_conf=_common_conf(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_full_training_enumeration_is_cheap():
    """Construction over ALL Training scenes must be index-driven (scene_list.json
    + all_metadata counts), not per-scene npz loads: ~988 of the 4332 Training
    dirs are EMPTY, and only scenes with >= min_num_images frames survive."""
    import time

    t0 = time.time()
    ds = ArkitScenesDataset(
        common_conf=_common_conf(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        len_train=10,
    )
    elapsed = time.time() - t0
    assert elapsed < 20.0, f"full Training construction took {elapsed:.1f}s"
    # 3344 valid scenes, 68 of which have < 24 frames at the current snapshot.
    assert 3000 < ds.sequence_list_len <= 3344
    assert "44796475" not in ds.data_store          # a known EMPTY scene dir
    assert not ds._scene_cache                       # nothing eagerly loaded


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_default_test_split_skips_empty_scene():
    """Default FULL Test construction (sequences=None) must not crash on the
    completely EMPTY scene dir Test/41159368 (the empty-dir quirk exists in Test
    too, with no scene_list.json to hide it): it counts 0 frames, is dropped by
    the min_num_images filter, and the 23 valid scenes remain."""
    ds = ArkitScenesDataset(
        common_conf=_eval_common(),
        split="test",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        len_test=10,
    )
    assert "41159368" not in ds.data_store           # the known EMPTY Test dir
    assert ds.sequence_list_len == 23                # 24 dirs on disk, 23 valid
    assert TEST_SCENE in ds.data_store
    assert not ds._scene_cache                       # nothing eagerly loaded


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_test_split_and_portrait_scene():
    """Test split enumerates by listdir (no index files ship); portrait scenes
    expose their native 480x640 geometry and still load fine."""
    test_ds = ArkitScenesDataset(
        common_conf=_eval_common(),
        split="test",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[TEST_SCENE],
        len_test=10,
    )
    assert test_ds.sequence_list == [TEST_SCENE]
    assert test_ds.sequence_num_frames(0) == 395
    assert test_ds.native_image_size(0) == (480, 640)   # landscape Test scene

    portrait = ArkitScenesDataset(
        common_conf=_eval_common(),
        split="train",
        ARKITSCENES_DIR=ARKITSCENES_DIR,
        sequences=[SCENE_PORTRAIT],
        len_train=10,
    )
    assert portrait.native_image_size(0) == (640, 480)  # portrait (H, W)
    b = portrait.get_data(seq_name=SCENE_PORTRAIT, ids=np.array([0, 1]), aspect_ratio=1.0)
    assert b["frame_num"] == 2
    assert np.isfinite(np.stack(b["extrinsics"])).all()


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_arkit_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _arkit_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _arkit_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    # Order honored verbatim: per-frame timestamps follow the requested (unordered)
    # id order -- frames are time-sorted, so timestamp order mirrors id order.
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])
    ts = sample["timestamps"].numpy()
    assert np.array_equal(np.argsort(ts), np.argsort(np.array(ids, dtype=np.int64)))


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SCENE, SCENE_B]
    composed = instantiate(
        _arkit_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    expected = {SCENE: 405, SCENE_B: 315}           # from all_metadata counts
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == expected[name]
        local = vendor.sequence_list.index(name)
        assert vendor.sequence_num_frames(local) == expected[name]


@pytest.mark.skipif(not HAVE_ARKIT, reason=f"ARKitScenes data not found at {ARKITSCENES_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _arkit_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # native vga_wide (H, W), landscape

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
