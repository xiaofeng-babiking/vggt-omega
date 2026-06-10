import logging
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.neu3d import Neu3dDataset

NEU3D_DIR = "/jfs/guibiao/streamVGGT/data/eval/neu3d"
HAVE_NEU3D = os.path.isdir(NEU3D_DIR)
# This copy has exactly one sequence: scene cut_roasted_beef, camera cam00.
SEQ = "cut_roasted_beef/cam00"


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


def _neu3d_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.neu3d.Neu3dDataset",
                "split": "train",
                "NEU3D_DIR": NEU3D_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- Neu3D-specific helper unit tests (no data required) ---


def test_identity_extrinsics_shape_and_values():
    E = Neu3dDataset.identity_extrinsics()
    assert E.shape == (3, 4) and E.dtype == np.float32
    np.testing.assert_allclose(E, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=0)


def test_placeholder_intrinsics_default_override_and_error():
    # Default: focal = max(H, W), principal point = image center.
    K = Neu3dDataset.placeholder_intrinsics(1014, 1352)
    assert K.shape == (3, 3) and K.dtype == np.float32
    assert K[0, 0] == 1352.0 and K[1, 1] == 1352.0       # focal = max(H, W)
    assert K[0, 2] == 1352 / 2.0 and K[1, 2] == 1014 / 2.0  # pp = center (cx, cy)
    assert K[0, 1] == 0.0 and K[1, 0] == 0.0 and K[2, 2] == 1.0
    # Portrait input: focal is still the long side.
    K_p = Neu3dDataset.placeholder_intrinsics(1352, 1014)
    assert K_p[0, 0] == 1352.0 and K_p[0, 2] == 507.0 and K_p[1, 2] == 676.0
    # Override wins verbatim.
    K2 = Neu3dDataset.placeholder_intrinsics(1014, 1352, override=[100.0, 110.0, 50.0, 60.0])
    assert K2[0, 0] == 100.0 and K2[1, 1] == 110.0
    assert K2[0, 2] == 50.0 and K2[1, 2] == 60.0
    with pytest.raises(ValueError, match="positive"):
        Neu3dDataset.placeholder_intrinsics(0, 1352)
    with pytest.raises(ValueError, match="positive"):
        Neu3dDataset.placeholder_intrinsics(1014, -1)


def test_empty_depth_is_all_invalid():
    d = Neu3dDataset.empty_depth(4, 6)
    assert d.shape == (4, 6) and d.dtype == np.float32
    assert (d == 0).all()           # 0 = invalid everywhere (no depth modality)
    assert not (d < 0).any()        # and never sky


def test_list_frames_orders_and_filters(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("0002.png", "0000.png", "0001.png", "notaframe.png", "0003.jpg"):
        (d / name).touch()
    frames = Neu3dDataset._list_frames(str(d))
    assert [fn for _, fn in frames] == [0, 1, 2]          # numeric stems, ordered
    assert all(p.endswith(".png") for p, _ in frames)     # the .jpg is ignored
    # A missing dir lists as empty (falls below min_num_images -> skipped).
    assert Neu3dDataset._list_frames(str(tmp_path / "nope")) == []


def test_constructor_validation_data_free(tmp_path):
    with pytest.raises(ValueError, match="NEU3D_DIR"):
        Neu3dDataset(common_conf=_common_conf(), NEU3D_DIR=None)
    with pytest.raises(ValueError, match="image_variant"):
        Neu3dDataset(
            common_conf=_common_conf(), NEU3D_DIR=str(tmp_path), image_variant="bogus"
        )
    with pytest.raises(ValueError, match="No usable Neu3D sequences"):
        Neu3dDataset(common_conf=_common_conf(), NEU3D_DIR=str(tmp_path))


def _make_fake_scene(root, scene, cam, n_frames, variant="images"):
    """Synthetic Neu3D layout: construction only lists file NAMES (never decodes
    pixels), so empty touch()ed files are sufficient."""
    d = root / scene / cam / variant
    d.mkdir(parents=True)
    for i in range(n_frames):
        (d / f"{i:04d}.png").touch()


def test_short_sequences_dropped_at_construction(tmp_path, caplog):
    """Sequences with < min_num_images frames must be filtered out (with a
    warning) when the dataset is built -- the TUM/7-Scenes contract. Synthetic
    root, no /jfs needed; also locks the scene-level `sequences` matching."""
    _make_fake_scene(tmp_path, "scene_long", "cam00", 30)
    _make_fake_scene(tmp_path, "scene_short", "cam00", 5)
    (tmp_path / "scene_broken" / "cam00").mkdir(parents=True)  # no images/ -> 0 frames

    with caplog.at_level(logging.WARNING):
        ds = Neu3dDataset(
            common_conf=_common_conf(),
            split="train",
            NEU3D_DIR=str(tmp_path),
            len_train=10,
            min_num_images=24,
        )
    assert ds.sequence_list == ["scene_long/cam00"]
    assert ds.sequence_list_len == 1
    assert any("only 5 frames" in m for m in caplog.messages)   # warned, not raised
    assert any("only 0 frames" in m for m in caplog.messages)
    assert ds.sequence_num_frames(0) == 30

    # `sequences` patterns match the scene name alone or the full "scene/cam".
    ds_scene = Neu3dDataset(
        common_conf=_common_conf(), NEU3D_DIR=str(tmp_path),
        sequences=["scene_long"], min_num_images=24,
    )
    assert ds_scene.sequence_list == ["scene_long/cam00"]
    ds_full = Neu3dDataset(
        common_conf=_common_conf(), NEU3D_DIR=str(tmp_path),
        sequences=["scene_long/cam*"], min_num_images=24,
    )
    assert ds_full.sequence_list == ["scene_long/cam00"]

    # If filtering leaves nothing usable, construction fails loudly.
    with pytest.raises(ValueError, match="No usable Neu3D sequences"):
        Neu3dDataset(
            common_conf=_common_conf(),
            NEU3D_DIR=str(tmp_path),
            sequences=["scene_short"],
            min_num_images=24,
        )


# --- Neu3D integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_neu3d_sample_schema_and_conventions():
    ds = Neu3dDataset(
        common_conf=_common_conf(),
        split="train",
        NEU3D_DIR=NEU3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    # IMAGE-ONLY vendor: nothing else exists on disk, nothing else is advertised.
    assert ds.available_modalities == frozenset({Modality.IMAGE})
    assert Modality.DEPTH not in ds.available_modalities
    assert Modality.EXTRINSICS not in ds.available_modalities  # placeholder, not GT
    assert Modality.INTRINSICS not in ds.available_modalities  # placeholder, not GT
    assert Modality.WORLD_POINTS not in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities   # no clock/fps on disk

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    cam = np.stack(batch["cam_points"])
    pmask = np.stack(batch["point_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    # No depth modality: zero depth, no valid points, zeroed geometry.
    assert (depth == 0).all()
    assert not pmask.any()
    assert (world == 0).all() and (cam == 0).all()
    # Placeholder pose: exactly identity w2c (process_one_image never touches
    # extrinsics with landscape_check=False).
    for E in extr:
        np.testing.assert_allclose(E, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=0)
    assert batch["seq_name"] == "neu3d_" + SEQ
    assert batch["is_metric"] is False and batch["is_video"] is True
    assert "sky_masks" not in batch                       # not advertised, not emitted
    assert "timestamps" not in batch                      # not advertised, not fabricated
    assert "camera_ids" not in batch

    validate_sample(batch, ds.available_modalities)

    # Explicit out-of-range seq_index fails loudly (inside_random=False).
    with pytest.raises(ValueError, match="out of range"):
        ds.get_data(seq_index=99, img_per_seq=2, aspect_ratio=1.0)


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_neu3d_placeholder_geometry_shapes():
    """Substitute for the depth reprojection-closure test (Neu3D has no depth or
    poses, so there is nothing to close): the pose/K placeholders must be EXACTLY
    identity / focal=max(H,W) / center-pp at native resolution, and the processed
    K must track the crop (pp at the processed-image center)."""
    ds = Neu3dDataset(
        common_conf=_eval_common(),
        split="train",
        NEU3D_DIR=NEU3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    h, w = ds.native_image_size(0)
    assert (h, w) == (1014, 1352)                  # surveyed native size (H, W)

    K_raw = Neu3dDataset.placeholder_intrinsics(h, w)
    assert K_raw[0, 0] == K_raw[1, 1] == float(max(h, w))   # focal = max(H, W)
    assert K_raw[0, 2] == w / 2.0 and K_raw[1, 2] == h / 2.0  # pp = native center

    ids = [0, 50, 150, 299]
    b = ds.get_data(seq_name=SEQ, ids=np.array(ids), aspect_ratio=0.75)
    H, W = np.stack(b["images"]).shape[1:3]
    for E in np.stack(b["extrinsics"]):
        np.testing.assert_allclose(E, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=0)
    for K in np.stack(b["intrinsics"]):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        assert fx > 0 and fy > 0 and fx == fy             # isotropic placeholder
        assert abs(cx - W / 2) <= 2 and abs(cy - H / 2) <= 2  # pp centered by crop


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_neu3d_getitem_tuple_index():
    ds = Neu3dDataset(
        common_conf=_common_conf(),
        split="train",
        NEU3D_DIR=NEU3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_neu3d_image_variant_downsampled():
    """The downsampled_2x variant is the same 300-frame video at exactly half
    the native resolution; the placeholder K follows the variant's geometry."""
    ds_full = Neu3dDataset(
        common_conf=_eval_common(), NEU3D_DIR=NEU3D_DIR, sequences=[SEQ]
    )
    ds_half = Neu3dDataset(
        common_conf=_eval_common(), NEU3D_DIR=NEU3D_DIR, sequences=[SEQ],
        image_variant="downsampled_2x",
    )
    assert ds_full.sequence_num_frames(0) == ds_half.sequence_num_frames(0) == 300
    h, w = ds_full.native_image_size(0)
    h2, w2 = ds_half.native_image_size(0)
    assert (h2, w2) == (507, 676)
    assert (h, w) == (2 * h2, 2 * w2)


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_neu3d_through_composed_dataset():
    """ComposedDataset tensorizes the IMAGE-only batch (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _neu3d_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert sample["extrinsics"].shape == (4, 3, 4)
    assert "timestamps" not in sample              # not advertised, not fabricated
    assert "sky_masks" not in sample               # not advertised, not emitted
    assert "modalities" in sample
    assert sample["modalities"] == sorted(
        m.value for m in Neu3dDataset.AVAILABLE
    )                                              # ["images"]
    assert not bool(sample["point_masks"].any())   # zero depth -> no valid points
    assert bool((sample["depths"] == 0).all())


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested (unordered) id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _neu3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 150, 50, 299]                        # deliberately NOT sorted
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

    # Order honored verbatim: the scene is dynamic, so frame identity is testable
    # purely from pixels. In eval mode processing is deterministic, so frame k of
    # the multi-id sample must equal a fresh single-id load of ids[k].
    assert not torch.equal(sample["images"][0], sample["images"][1])  # frames differ
    for k, i in enumerate(ids):
        single = vendor.get_data(
            seq_name=composed.sequence_name(0), ids=np.array([i]), aspect_ratio=0.75
        )
        np.testing.assert_array_equal(
            np.stack(batch["images"])[k], np.stack(single["images"])[0]
        )


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    composed = instantiate(
        _neu3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 1           # one scene, one camera on disk
    assert composed.sequence_name(0) == SEQ
    vendor = composed.base_dataset.datasets[0]
    assert composed.sequence_num_frames(0) == len(vendor.data_store[SEQ]) == 300


@pytest.mark.skipif(not HAVE_NEU3D, reason=f"Neu3D data not found at {NEU3D_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _neu3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (1014, 1352)                  # Neu3D images/ native (H, W)
    assert h / w == 0.75                           # exact 3:4 aspect

    composed.set_img_size(640)
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                     # half that, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
