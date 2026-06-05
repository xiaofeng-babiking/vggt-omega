import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.seven_scenes import SevenScenesDataset

SEVEN_SCENES_DIR = "/jfs/guibiao/streamVGGT/data/eval/7scenes"
HAVE_7S = os.path.isdir(SEVEN_SCENES_DIR)
# "heads" is the smallest scene (2 sequences); use it for fast integration tests.
SCENE = "heads"


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


def _7s_dataset_cfg(scenes=(SCENE,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.seven_scenes.SevenScenesDataset",
                "split": "test",
                "SEVEN_SCENES_DIR": SEVEN_SCENES_DIR,
                "scenes": list(scenes),
                "len_test": n,
            }
        ],
    }


# --- 7-Scenes-specific helper unit tests (no data required) ---


def test_pose_to_w2c_inverts_c2w(tmp_path):
    # camera-to-world = pure translation by (1,2,3), identity rotation
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    p = tmp_path / "frame-000000.pose.txt"
    p.write_text("\n".join(" ".join(str(v) for v in row) for row in c2w))
    w2c = SevenScenesDataset.seven_scenes_pose_to_w2c(str(p))
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rejects_non_finite(tmp_path):
    p = tmp_path / "frame-000000.pose.txt"
    p.write_text("inf " * 16)
    with pytest.raises(ValueError, match="non-finite"):
        SevenScenesDataset.seven_scenes_pose_to_w2c(str(p))


def test_intrinsics_default_and_override():
    K = SevenScenesDataset.seven_scenes_intrinsics()
    assert K.shape == (3, 3) and K[2, 2] == 1.0
    assert K[0, 0] == 585.0 and K[1, 1] == 585.0
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0
    K2 = SevenScenesDataset.seven_scenes_intrinsics(override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0


def test_depth_reader_units_and_invalid(tmp_path):
    from PIL import Image

    arr = np.array([[0, 1000], [2000, 65535]], dtype=np.uint16)
    p = tmp_path / "frame-000000.depth.proj.png"
    Image.fromarray(arr).save(p)
    depth = SevenScenesDataset.read_seven_scenes_depth(str(p))
    np.testing.assert_allclose(depth, [[0.0, 1.0], [2.0, 0.0]])  # mm->m, 0 & 65535 invalid


# --- 7-Scenes integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_sample_schema_and_conventions():
    ds = SevenScenesDataset(
        common_conf=_common_conf(),
        split="test",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR,
        scenes=[SCENE],
        len_test=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities

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
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_cross_frame_world_points_agree():
    """World points from two frames of the same sequence must be mutually
    consistent: frame A's valid world points reprojected into frame B land at
    depths matching B's depth map (validates pose+depth+intrinsic agreement
    end-to-end through process_one_image)."""
    ds = SevenScenesDataset(
        common_conf=_eval_common(),
        split="test",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR,
        scenes=[SCENE],
        len_test=10,
    )
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 15]), aspect_ratio=0.75)
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
    err = np.abs(z[ok][valid] - measured[valid])
    assert valid.sum() > 1000
    assert np.median(err) < 0.05  # agree to < 5 cm


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_getitem_tuple_index():
    ds = SevenScenesDataset(
        common_conf=_common_conf(),
        split="test",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR,
        scenes=[SCENE],
        len_test=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_split_files_select_sequences():
    """split='test' / 'train' pick the scene's TestSplit/TrainSplit sequences;
    'all' takes every seq-* dir. heads has 2 sequences split 1 train / 1 test."""
    test_ds = SevenScenesDataset(
        common_conf=_common_conf(), split="test",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR, scenes=[SCENE], len_test=10,
    )
    train_ds = SevenScenesDataset(
        common_conf=_common_conf(), split="train",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR, scenes=[SCENE], len_train=10,
    )
    all_ds = SevenScenesDataset(
        common_conf=_common_conf(), split="all",
        SEVEN_SCENES_DIR=SEVEN_SCENES_DIR, scenes=[SCENE], len_test=10,
    )
    assert set(test_ds.sequence_list).isdisjoint(train_ds.sequence_list)
    assert len(all_ds.sequence_list) == len(test_ds.sequence_list) + len(train_ds.sequence_list)


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _7s_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _7s_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_test), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    composed = instantiate(
        _7s_dataset_cfg(scenes=[SCENE], n=20), common_config=_eval_common(), _recursive_=False
    )
    vendor = composed.base_dataset.datasets[0]
    assert composed.num_sequences() == len(vendor.sequence_list)
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in vendor.sequence_list
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_7S, reason=f"7-Scenes data not found at {SEVEN_SCENES_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _7s_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # 7-Scenes native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
