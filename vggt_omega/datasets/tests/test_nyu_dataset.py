import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.nyu import NyuDataset

NYU_DIR = "/jfs/guibiao/streamVGGT/data/eval/nyu"
HAVE_NYU = os.path.isdir(NYU_DIR)
# A few known frame ids (non-contiguous original h5 ids) for fast integration tests.
FRAMES = ["00001", "00002", "00009"]


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


def _nyu_dataset_cfg(seqs=tuple(FRAMES), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.nyu.NyuDataset",
                "split": "test",
                "NYU_DIR": NYU_DIR,
                "sequences": list(seqs),
                "len_test": n,
            }
        ],
    }


# --- NYU-specific helper unit tests (no data required) ---


def test_nyu_intrinsics_default_and_override():
    K = NyuDataset.nyu_intrinsics()
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K[0, 0], 518.857901, atol=1e-4)
    np.testing.assert_allclose(K[1, 1], 519.469611, atol=1e-4)
    np.testing.assert_allclose(K[0, 2], 325.582245, atol=1e-4)
    np.testing.assert_allclose(K[1, 2], 253.736166, atol=1e-4)
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0
    K2 = NyuDataset.nyu_intrinsics(override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0


def test_nyu_identity_extrinsic():
    w2c = NyuDataset.identity_extrinsic()
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]))


def test_nyu_read_depth_passthrough_and_invalid(tmp_path):
    arr = np.array([[0.713, 9.987], [np.nan, -1.0], [np.inf, 2.5]], dtype=np.float32)
    p = tmp_path / "00001.npy"
    np.save(p, arr)
    depth = NyuDataset.read_nyu_depth(str(p))
    assert depth.dtype == np.float32
    # meters pass through at scale 1.0; non-finite / negative map to 0 (invalid)
    np.testing.assert_allclose(depth, [[0.713, 9.987], [0.0, 0.0], [0.0, 2.5]])


def test_nyu_list_frames_pairs_by_basename(tmp_path):
    from PIL import Image

    images_dir = tmp_path / "nyu_images"
    depths_dir = tmp_path / "nyu_depths"
    images_dir.mkdir()
    depths_dir.mkdir()
    # Non-contiguous ids, written out of order to check sorting.
    for name in ("00009", "00001"):
        Image.new("RGB", (4, 4)).save(images_dir / f"{name}.png")
        np.save(depths_dir / f"{name}.npy", np.ones((4, 4), dtype=np.float32))
    # Unpaired image (no depth) must be skipped.
    Image.new("RGB", (4, 4)).save(images_dir / "00002.png")

    frames = NyuDataset.list_nyu_frames(str(images_dir), str(depths_dir))
    assert [fr[0] for fr in frames] == ["00001", "00009"]
    for name, rgb_path, depth_path in frames:
        assert rgb_path.endswith(f"{name}.png") and depth_path.endswith(f"{name}.npy")


def test_nyu_eigen_crop_mask():
    mask = NyuDataset.eigen_crop_mask()
    assert mask.shape == (480, 640) and mask.dtype == bool
    top, bottom, left, right = NyuDataset.EIGEN_CROP
    assert mask[top:bottom, left:right].all()
    assert mask.sum() == (bottom - top) * (right - left)  # nothing outside the crop
    assert not mask[0].any() and not mask[:, 0].any()


def test_nyu_constructor_requires_dir_and_frames(tmp_path):
    with pytest.raises(ValueError, match="NYU_DIR"):
        NyuDataset(common_conf=_common_conf())
    with pytest.raises(ValueError, match="No usable NYU frames"):
        NyuDataset(common_conf=_common_conf(), NYU_DIR=str(tmp_path))


# --- NYU integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_nyu_sample_schema_and_conventions():
    ds = NyuDataset(
        common_conf=_common_conf(),
        split="test",
        NYU_DIR=NYU_DIR,
        sequences=FRAMES,
        len_test=10,
    )
    # Poses/intrinsics are not GT on disk -> not advertised; depth GT is.
    assert Modality.DEPTH in ds.available_modalities
    assert Modality.POINT_MASK in ds.available_modalities
    assert Modality.SKY_MASK in ds.available_modalities
    assert Modality.EXTRINSICS not in ds.available_modalities
    assert Modality.INTRINSICS not in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities

    n = 4  # single-frame sequence: V=4 duplicates via allow_duplicate_img
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert (np.asarray(batch["ids"]) == 0).all()          # only frame 0 exists
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    cam = np.stack(batch["cam_points"])
    pmask = np.stack(batch["point_masks"])
    sky = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    # No poses: extrinsics stay the identity through processing (crops touch K only).
    np.testing.assert_allclose(extr, np.broadcast_to(np.hstack([np.eye(3), np.zeros((3, 1))]), (v, 3, 4)), atol=1e-6)
    # Identity pose => world points == camera points.
    np.testing.assert_allclose(world[pmask], cam[pmask], atol=1e-6)
    assert (depth[depth > 0]).size > 0                    # valid metric depth
    assert not sky.any()                                  # indoor: all-False sky masks
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is False
    assert batch["seq_name"].startswith("nyu_")

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_nyu_depth_statistics_sanity():
    """No poses -> no reprojection-closure test is possible. Instead validate the
    survey-verified depth properties: metric meters at scale 1.0, dense (no
    invalid pixels), plausible indoor range (~0.5-10 m globally)."""
    ds = NyuDataset(
        common_conf=_eval_common(),
        split="test",
        NYU_DIR=NYU_DIR,
        sequences=FRAMES,
        len_test=10,
    )
    # Raw reader stats on the real files.
    for name in ds.sequence_list:
        depth = ds.read_nyu_depth(ds.data_store[name][0][1])
        assert depth.shape == (480, 640) and depth.dtype == np.float32
        assert (depth > 0).mean() == 1.0                  # dense: every pixel valid
        assert 0.3 < depth.min() and depth.max() < 12.0   # plausible metric meters
        assert 0.5 < np.median(depth) < 10.0
        # A mm- or 1/5000-scaled decode would shatter these bounds; scale must be 1.0.

    # Processed pipeline keeps the depth metric and dense.
    batch = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0]), aspect_ratio=0.75)
    depth = np.stack(batch["depths"])
    pmask = np.stack(batch["point_masks"])
    assert pmask.mean() > 0.99                            # nearest-neighbor resize stays dense
    assert 0.3 < depth[pmask].min() and depth[pmask].max() < 12.0
    assert 0.5 < np.median(depth[pmask]) < 10.0


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_nyu_getitem_tuple_index():
    ds = NyuDataset(
        common_conf=_common_conf(),
        split="test",
        NYU_DIR=NYU_DIR,
        sequences=FRAMES,
        len_test=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_nyu_full_split_enumeration():
    """Unrestricted construction enumerates the full 654-frame Eigen test split
    as 654 independent 1-frame sequences (construction is a single glob: fast)."""
    ds = NyuDataset(common_conf=_eval_common(), split="test", NYU_DIR=NYU_DIR, len_test=10)
    assert ds.sequence_list_len == 654
    assert ds.sequence_list == sorted(ds.sequence_list)
    for local_idx in (0, 100, 653):
        assert ds.sequence_num_frames(local_idx) == 1
    assert ds.native_image_size(0) == (480, 640)


# --- ComposedDataset integration (tensorization + V>1 duplicate draws) ---


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_nyu_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities; a V=4 draw from a
    1-frame sequence duplicates the single frame (allow_duplicate_img path)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _nyu_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert not sample["sky_masks"].any()
    assert "timestamps" not in sample              # never fabricated
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)
    assert (sample["ids"].numpy() == 0).all()      # all V frames are frame 0 duplicates


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ids (the training-identical inference path). The two must
    not drift. NYU sequences hold a single frame, so explicit ids exercise the
    duplicate path ([0, 0, 0]) rather than ordering."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _nyu_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 0, 0]
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
    np.testing.assert_allclose(
        sample["depths"].numpy(), np.stack(batch["depths"]).astype(np.float32)
    )


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_test), so inference.py can iterate frames."""
    from hydra.utils import instantiate

    composed = instantiate(
        _nyu_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == len(FRAMES)
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in FRAMES
        assert composed.sequence_num_frames(gi) == 1


@pytest.mark.skipif(not HAVE_NYU, reason=f"NYU data not found at {NYU_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _nyu_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # NYU native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
