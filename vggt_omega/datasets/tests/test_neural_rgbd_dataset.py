import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.neural_rgbd import NeuralRgbdDataset

NEURAL_RGBD_DIR = "/jfs/guibiao/streamVGGT/data/eval/neural_rgbd"
HAVE_NRGBD = os.path.isdir(NEURAL_RGBD_DIR)
# thin_geometry is the smallest scene (395 frames); use it for fast integration tests.
SCENE = "thin_geometry"


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


def _nrgbd_dataset_cfg(seqs=(SCENE,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.neural_rgbd.NeuralRgbdDataset",
                "split": "test",
                "NEURAL_RGBD_DIR": NEURAL_RGBD_DIR,
                "sequences": list(seqs),
                "len_test": n,
            }
        ],
    }


# --- Neural-RGB-D-specific helper unit tests (no data required) ---


def test_opengl_c2w_to_w2c_identity_flips_camera_axes():
    # Identity c2w in OpenGL axes -> w2c rotation = diag(1,-1,-1) (Y/Z flipped).
    w2c = NeuralRgbdDataset.opengl_c2w_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], 0.0, atol=1e-6)


def test_opengl_c2w_to_w2c_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = NeuralRgbdDataset.opengl_c2w_to_w2c(c2w)
    # t_w2c = -(R_gl @ flip)^T t = -diag(1,-1,-1) @ (1,2,3) = (-1, 2, 3)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, 2.0, 3.0], atol=1e-6)


def test_opengl_c2w_to_w2c_inverts_flipped_pose():
    """[w2c; 0 0 0 1] must be the exact inverse of the Y/Z-flipped c2w for a
    general rigid pose (rotation about an arbitrary axis + translation)."""
    from vggt_omega.datasets.vendors.common import quat_to_rotation

    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rotation((0.1, -0.2, 0.3, 0.9))
    c2w[:3, 3] = [0.5, -1.5, 2.0]
    w2c = NeuralRgbdDataset.opengl_c2w_to_w2c(c2w)

    c2w_cv = c2w.copy()
    c2w_cv[:3, 1:3] *= -1  # the survey-verified OpenGL->OpenCV column flip
    w2c44 = np.vstack([w2c, [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(w2c44 @ c2w_cv, np.eye(4), atol=1e-6)


def test_opengl_c2w_to_w2c_rejects_non_finite_and_bad_shape():
    bad = np.eye(4)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        NeuralRgbdDataset.opengl_c2w_to_w2c(bad)
    with pytest.raises(ValueError, match="4,4"):
        NeuralRgbdDataset.opengl_c2w_to_w2c(np.eye(3))


def test_read_poses_parses_stacked_matrices_and_nan_literals(tmp_path):
    rows = ["1 0 0 0.5", "0 1 0 -0.5", "0 0 1 2", "0 0 0 1"]
    nan_rows = ["-nan(ind) 0 0 0", "0 nan(ind) 0 0", "0 0 1 0", "0 0 0 1"]
    p = tmp_path / "poses.txt"
    p.write_text("\n".join(rows + nan_rows) + "\n")
    poses = NeuralRgbdDataset.read_neural_rgbd_poses(str(p))
    assert poses.shape == (2, 4, 4)
    np.testing.assert_allclose(poses[0, :3, 3], [0.5, -0.5, 2.0])
    assert np.isnan(poses[1, 0, 0]) and np.isnan(poses[1, 1, 1])  # Windows literals -> NaN


def test_read_poses_rejects_bad_value_count(tmp_path):
    p = tmp_path / "poses.txt"
    p.write_text("1 2 3 4 5\n")
    with pytest.raises(ValueError, match="multiple of 16"):
        NeuralRgbdDataset.read_neural_rgbd_poses(str(p))


def test_depth_reader_units_and_invalid(tmp_path):
    from PIL import Image

    arr = np.array([[0, 1000], [2000, 3500]], dtype=np.uint16)
    p = tmp_path / "depth0.png"
    Image.fromarray(arr).save(p)
    depth = NeuralRgbdDataset.read_neural_rgbd_depth(str(p))
    np.testing.assert_allclose(depth, [[0.0, 1.0], [2.0, 3.5]])  # mm->m, 0 stays invalid


def test_intrinsics_default_and_override():
    K = NeuralRgbdDataset.neural_rgbd_intrinsics(554.2562584220408)
    assert K.shape == (3, 3) and K[2, 2] == 1.0
    assert K[0, 0] == np.float32(554.2562584220408) and K[1, 1] == K[0, 0]
    assert K[0, 2] == 319.5 and K[1, 2] == 239.5  # verified (W-1)/2, (H-1)/2 at 640x480
    K2 = NeuralRgbdDataset.neural_rgbd_intrinsics(554.0, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0


def test_depth_variant_validated():
    with pytest.raises(ValueError, match="depth_variant"):
        NeuralRgbdDataset(
            common_conf=_common_conf(),
            NEURAL_RGBD_DIR="/nonexistent",
            depth_variant="depth_gt",  # only thin_geometry ships it; not uniform
        )


# --- Neural RGB-D integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_sample_schema_and_conventions():
    ds = NeuralRgbdDataset(
        common_conf=_common_conf(),
        split="test",
        NEURAL_RGBD_DIR=NEURAL_RGBD_DIR,
        sequences=[SCENE],
        len_test=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities  # no clocks on disk

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
    assert not sky.any()                                  # indoor: all-False sky masks
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "neural_rgbd_" + SCENE
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_cross_frame_reprojection_closure():
    """World points from frame A reprojected into frame B must land at depths
    matching B's depth map (validates the OpenGL->OpenCV pose flip + mm depth
    scale + focal.txt intrinsics end-to-end through process_one_image).

    Survey: the correct c2w+flip convention closes at ~6e-4 median relative
    error; the wrong (no-flip) convention is >= 0.01 on this scene -- and at
    this (0, 100) baseline projects ZERO overlapping points -- so the 0.005
    threshold (plus the count assert) fails under a flipped convention."""
    ds = NeuralRgbdDataset(
        common_conf=_eval_common(),
        split="test",
        NEURAL_RGBD_DIR=NEURAL_RGBD_DIR,
        sequences=[SCENE],
        len_test=10,
    )
    b = ds.get_data(seq_name=SCENE, ids=np.array([0, 100]), aspect_ratio=0.75)
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
    assert valid.sum() > 500
    assert np.median(rel_err) < 0.005  # correct convention ~6e-4; no-flip fails


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_getitem_tuple_index():
    ds = NeuralRgbdDataset(
        common_conf=_common_conf(),
        split="test",
        NEURAL_RGBD_DIR=NEURAL_RGBD_DIR,
        sequences=[SCENE],
        len_test=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _nrgbd_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _nrgbd_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    # Order honored: per-frame extrinsics follow the requested (unordered) ids.
    np.testing.assert_allclose(
        sample["extrinsics"].numpy(), np.stack(batch["extrinsics"]), rtol=1e-6
    )
    np.testing.assert_array_equal(batch["ids"], ids)


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_test), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = ["staircase", "thin_geometry"]
    composed = instantiate(
        _nrgbd_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])
    # thin_geometry is the smallest scene with 395 frames (survey-verified count)
    assert vendor.data_store["thin_geometry"] is not None
    assert len(vendor.data_store["thin_geometry"]) == 395


@pytest.mark.skipif(not HAVE_NRGBD, reason=f"Neural RGB-D data not found at {NEURAL_RGBD_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _nrgbd_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # Neural RGB-D native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
