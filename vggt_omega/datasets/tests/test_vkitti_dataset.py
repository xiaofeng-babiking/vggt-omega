import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.vkitti import VkittiDataset

VKITTI_DIR = "/jfs/Data_4DFF/train_data/vkitti"
HAVE_VKITTI = os.path.isdir(VKITTI_DIR)
# Scene02 is the smallest scene (233 frames per stream); use its clone variation
# (two stereo camera streams) for fast integration tests.
SEQ_CAM0 = "Scene02/clone/Camera_0"
SEQ_CAM1 = "Scene02/clone/Camera_1"


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


def _vkitti_dataset_cfg(seqs=(SEQ_CAM0,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.vkitti.VkittiDataset",
                "split": "train",
                "VKITTI_DIR": VKITTI_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- VKITTI-specific helper unit tests (no data required) ---


def test_vkitti_pose_to_w2c_identity():
    w2c = VkittiDataset.vkitti_pose_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_vkitti_pose_to_w2c_translation():
    # camera-to-world = pure translation by (1,2,3) -> w2c translation = -(1,2,3)
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = VkittiDataset.vkitti_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_vkitti_pose_to_w2c_rotation_axis_remap():
    # c2w = 90 deg about z + translation; w2c must be the exact rigid inverse
    # [R^T | -R^T t] (locks the inversion AND that no axis flip is applied).
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3] = Rz
    c2w[:3, 3] = t
    w2c = VkittiDataset.vkitti_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], Rz.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -Rz.T @ t, atol=1e-6)


def test_vkitti_pose_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match=r"\(4,4\)"):
        VkittiDataset.vkitti_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        VkittiDataset.vkitti_pose_to_w2c(bad)


def test_vkitti_depth_decode_units_sky_and_invalid(tmp_path):
    import cv2

    # cm -> m; 65535 is the sky clamp -> -1.0; raw 0 stays 0 (= invalid).
    arr = np.array([[0, 100], [65535, 12345]], dtype=np.uint16)
    p = tmp_path / "00000_depth.png"
    assert cv2.imwrite(str(p), arr)
    depth = VkittiDataset.read_vkitti_depth(str(p))
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.0], [-1.0, 123.45]])
    with pytest.raises(FileNotFoundError):
        VkittiDataset.read_vkitti_depth(str(tmp_path / "missing.png"))


def test_vkitti_intrinsics_assembly_override_and_error():
    K_raw = np.array([[725.0087, 0, 620.5], [0, 725.0087, 187.0], [0, 0, 1]])
    K = VkittiDataset.vkitti_intrinsics(K_raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_raw)
    K2 = VkittiDataset.vkitti_intrinsics(K_raw, override=[100.0, 100.0, 50.0, 40.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0 and K2[1, 2] == 40.0
    with pytest.raises(ValueError, match="intrinsics"):
        VkittiDataset.vkitti_intrinsics(None)
    with pytest.raises(ValueError, match=r"\(3,3\)"):
        VkittiDataset.vkitti_intrinsics(np.eye(4))


def test_vkitti_read_cam_npz(tmp_path):
    # synthetic cam.npz with the on-disk keys: pose inverted, K passed through
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [4.0, 5.0, 6.0]
    K = np.array([[725.0, 0, 620.5], [0, 725.0, 187.0], [0, 0, 1]], dtype=np.float32)
    p = tmp_path / "00000_cam.npz"
    np.savez(p, camera_pose=c2w, camera_intrinsics=K)
    w2c, K_raw = VkittiDataset.read_vkitti_cam(str(p))
    np.testing.assert_allclose(w2c[:, 3], [-4.0, -5.0, -6.0], atol=1e-6)
    np.testing.assert_allclose(K_raw, K)


def test_vkitti_parse_camera_id():
    assert VkittiDataset.parse_camera_id("Scene01/clone/Camera_0") == 0
    assert VkittiDataset.parse_camera_id("Scene20/30-deg-left/Camera_1") == 1
    with pytest.raises(ValueError, match="Camera_"):
        VkittiDataset.parse_camera_id("Scene01/clone")
    with pytest.raises(ValueError, match="Camera_"):
        VkittiDataset.parse_camera_id("Scene01/clone/Camera_x")


# --- VKITTI integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_vkitti_sample_schema_and_conventions():
    ds = VkittiDataset(
        common_conf=_common_conf(),
        split="train",
        VKITTI_DIR=VKITTI_DIR,
        sequences=[SEQ_CAM0, SEQ_CAM1],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.CAMERA_ID in ds.available_modalities

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
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    np.testing.assert_array_equal(sky, depth < 0)         # sky convention: depth<0
    assert not (pmask & sky).any()                        # sky never counts as valid
    assert batch["camera_ids"].dtype == np.int32
    assert (batch["camera_ids"] == 0).all()               # seq_index 0 = Camera_0
    assert batch["is_metric"] is True and batch["is_video"] is True
    assert batch["seq_name"] == "vkitti_" + SEQ_CAM0

    validate_sample(batch, ds.available_modalities)

    # second stream is Camera_1: camera_ids must follow the stream
    b1 = ds.get_data(seq_index=1, img_per_seq=2, aspect_ratio=1.0)
    assert (b1["camera_ids"] == 1).all()


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_vkitti_reprojection_closure():
    """Frame A's valid world points reprojected into nearby frame B land at
    depths matching B's depth map (locks depth-scale x pose-convention x
    intrinsics consistency end-to-end through process_one_image). VKITTI is
    dynamic (moving cars), so the median over mostly-static road pixels is the
    robust statistic."""
    ds = VkittiDataset(
        common_conf=_eval_common(),
        split="train",
        VKITTI_DIR=VKITTI_DIR,
        sequences=[SEQ_CAM0],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ_CAM0, ids=np.array([0, 3]), aspect_ratio=0.5)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    assert (depth < 0).any()  # a road scene: the sky clamp must survive processing

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


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_vkitti_getitem_tuple_index():
    ds = VkittiDataset(
        common_conf=_common_conf(),
        split="train",
        VKITTI_DIR=VKITTI_DIR,
        sequences=[SEQ_CAM0],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_vkitti_sequence_filter_patterns():
    """`sequences` accepts exact names and glob/prefix patterns over the
    Scene/variation/Camera_N stream names."""
    exact = VkittiDataset(
        common_conf=_common_conf(), VKITTI_DIR=VKITTI_DIR,
        sequences=[SEQ_CAM0], len_train=10,
    )
    assert exact.sequence_list == [SEQ_CAM0]
    prefix = VkittiDataset(
        common_conf=_common_conf(), VKITTI_DIR=VKITTI_DIR,
        sequences=["Scene02/clone"], len_train=10,   # prefix -> both cameras
    )
    assert prefix.sequence_list == [SEQ_CAM0, SEQ_CAM1]
    with pytest.raises(ValueError, match="No usable VKITTI sequences"):
        VkittiDataset(
            common_conf=_common_conf(), VKITTI_DIR=VKITTI_DIR,
            sequences=["SceneXX/*"], len_train=10,
        )


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_vkitti_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _vkitti_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modalities carried through
    assert "camera_ids" in sample
    assert "sky_masks" in sample
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _vkitti_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Order honored verbatim: VKITTI synthesizes timestamp = frame_index / 10 Hz,
    # so the per-frame timestamps must equal the requested ids in their order.
    np.testing.assert_allclose(
        sample["timestamps"].numpy(), np.asarray(ids, dtype=np.float64) / 10.0
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


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ_CAM0, SEQ_CAM1]
    composed = instantiate(
        _vkitti_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor._sequence_frames(name))
        assert composed.sequence_num_frames(gi) == 233  # Scene02 stream length


@pytest.mark.skipif(not HAVE_VKITTI, reason=f"VKITTI data not found at {VKITTI_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _vkitti_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (375, 1242)                      # VKITTI native (H, W)

    composed.set_img_size(1242)                       # native long side
    assert composed.img_size == 1242
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # short side snaps down to the patch grid: int(1242*375/1242)=375 -> 368
    assert tuple(s["images"].shape[-2:]) == (368, 1242)

    composed.set_img_size(496)                        # half-res-ish long side (16*31)
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # int(496*375/1242)=149 -> snapped to 144
    assert tuple(s2["images"].shape[-2:]) == (144, 496)
