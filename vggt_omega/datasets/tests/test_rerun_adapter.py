import logging
import os
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.adapters import rerun_adapter
from vggt_omega.datasets.modality import Modality


def test_require_rerun_raises_helpful_error_when_missing(monkeypatch):
    # Force `import rerun` to fail even if the package is installed.
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=r"vggt-omega\[viz\]"):
        rerun_adapter._require_rerun()


def test_canonical_images_from_chw_float():
    # ComposedDataset form: (V, 3, H, W) float in [0, 1]
    arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
    arr[:, 0] = 1.0  # full red
    out = rerun_adapter._canonical_images(arr)
    assert out.shape == (2, 4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0].tolist() == [255, 0, 0]


def test_canonical_images_from_hwc_uint8():
    # Raw form: (V, H, W, 3) uint8 in [0, 255]
    arr = np.full((2, 4, 5, 3), 200, dtype=np.uint8)
    out = rerun_adapter._canonical_images(arr)
    assert out.shape == (2, 4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0].tolist() == [200, 200, 200]


def test_canonical_images_rejects_non_4d():
    with pytest.raises(ValueError, match="4D"):
        rerun_adapter._canonical_images(np.zeros((4, 5, 3), dtype=np.uint8))


def _make_raw_sample(V=2, H=8, W=8):
    """A raw get_data()-style numpy dict covering several modalities."""
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(V)]
    depths = [rng.random((H, W)).astype(np.float32) + 0.1 for _ in range(V)]
    extr = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5.0]], dtype=np.float32)
            for _ in range(V)]
    intr = [np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float32)
            for _ in range(V)]
    world = [rng.random((H, W, 3)).astype(np.float32) for _ in range(V)]
    pmask = [np.ones((H, W), dtype=bool) for _ in range(V)]
    return {
        "seq_name": "synthetic",
        "images": images,
        "depths": depths,
        "extrinsics": extr,
        "intrinsics": intr,
        "world_points": world,
        "point_masks": pmask,
        "timestamps": np.arange(V, dtype=np.float64),
        "texts": [f"frame {i}" for i in range(V)],
        "modalities": frozenset({
            Modality.IMAGE, Modality.DEPTH, Modality.EXTRINSICS, Modality.INTRINSICS,
            Modality.WORLD_POINTS, Modality.POINT_MASK, Modality.TIMESTAMP, Modality.TEXT,
        }),
    }


def _make_composed_sample(V=2, H=8, W=8):
    """A ComposedDataset-style torch dict equivalent to _make_raw_sample."""
    torch = pytest.importorskip("torch")
    raw = _make_raw_sample(V, H, W)
    images = torch.from_numpy(np.stack(raw["images"]).astype(np.float32))
    images = images.permute(0, 3, 1, 2).div(255)  # (V,3,H,W) float [0,1]
    return {
        "seq_name": "synthetic",
        "images": images,
        "depths": torch.from_numpy(np.stack(raw["depths"])),
        "extrinsics": torch.from_numpy(np.stack(raw["extrinsics"])),
        "intrinsics": torch.from_numpy(np.stack(raw["intrinsics"])),
        "world_points": torch.from_numpy(np.stack(raw["world_points"])),
        "point_masks": torch.from_numpy(np.stack(raw["point_masks"])),
        "timestamps": torch.from_numpy(raw["timestamps"]),
        "texts": list(raw["texts"]),
        "modalities": sorted(m.value for m in raw["modalities"]),
    }


def test_normalize_sample_present_and_V_from_raw():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    assert norm.V == 2
    assert "images" in norm.present and "depths" in norm.present
    assert "seq_name" not in norm.present  # non-modality keys excluded


def test_normalize_sample_canonical_shapes_match_across_forms():
    raw = rerun_adapter.normalize_sample(_make_raw_sample())
    comp = rerun_adapter.normalize_sample(_make_composed_sample())
    assert raw.present == comp.present
    for key in ("images", "depths", "extrinsics", "intrinsics", "world_points", "point_masks"):
        assert raw.data[key].shape == comp.data[key].shape, key
    # images canonical: (V,H,W,3) uint8 from both forms
    assert raw.data["images"].shape == (2, 8, 8, 3)
    assert raw.data["images"].dtype == np.uint8
    assert comp.data["images"].dtype == np.uint8
    # texts stays a list of str
    assert comp.data["texts"] == ["frame 0", "frame 1"]


def test_normalize_sample_falls_back_to_present_keys_without_modalities_field():
    raw = _make_raw_sample()
    del raw["modalities"]
    norm = rerun_adapter.normalize_sample(raw)
    assert "images" in norm.present and "depths" in norm.present


def test_extrinsic_inversion_round_trip():
    # A non-trivial rotation (90 deg about Z) + translation.
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([1.0, 2.0, 3.0])
    ext = np.concatenate([R, t[:, None]], axis=1)  # (3,4) world->cam
    R_cw, t_cw = rerun_adapter._extrinsic_to_cam_to_world(ext)
    # Apply cam->world then world->cam; must recover the original camera point.
    x_cam = np.array([0.5, -0.7, 2.0])
    x_world = R_cw @ x_cam + t_cw
    x_cam2 = R @ x_world + t
    np.testing.assert_allclose(x_cam2, x_cam, atol=1e-5)


def test_extrinsic_camera_center_for_identity_rotation():
    # world->cam X_cam = X_world + t with t=[0,0,5] => camera center at world z=-5
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5.0])
    ext = np.concatenate([R, t[:, None]], axis=1)
    _, t_cw = rerun_adapter._extrinsic_to_cam_to_world(ext)
    np.testing.assert_allclose(t_cw, [0.0, 0.0, -5.0], atol=1e-6)


def test_camera_path_uses_camera_id_when_present():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    assert rerun_adapter._camera_path(norm, 0) == "world/camera"
    norm.data["camera_ids"] = np.array([7, 7])
    norm.present.add("camera_ids")
    assert rerun_adapter._camera_path(norm, 0) == "world/camera/7"


def test_frame_hw_reads_spatial_shape():
    norm = rerun_adapter.normalize_sample(_make_raw_sample(V=2, H=8, W=8))
    assert rerun_adapter._frame_hw(norm) == (8, 8)


def test_normalize01_maps_to_unit_range():
    out = rerun_adapter._normalize01(np.array([[0.0, 5.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-6)


def test_normalize01_constant_input_is_zeros():
    out = rerun_adapter._normalize01(np.full((2, 2), 3.0, dtype=np.float32))
    assert np.all(out == 0.0)


def test_track_colors_shape_and_dtype():
    colors = rerun_adapter._track_colors(5)
    assert colors.shape == (5, 3)
    assert colors.dtype == np.uint8


def test_select_views_full_coverage():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    names = {v.name for v in rerun_adapter.select_views(norm.present)}
    assert {"camera", "rgb", "world", "depth", "text"} <= names


def test_select_views_skips_views_with_missing_requirements():
    present = {"images"}  # images only -> no camera, no depth, no world cloud
    names = {v.name for v in rerun_adapter.select_views(present)}
    assert names == {"rgb"}


def test_select_views_camera_needs_both_intrinsics_and_extrinsics():
    assert {v.name for v in rerun_adapter.select_views({"intrinsics"})} == set()
    assert {v.name for v in rerun_adapter.select_views({"intrinsics", "extrinsics"})} == {"camera"}


def test_log_batch_smoke_on_full_sample():
    rr = pytest.importorskip("rerun")
    rr.init("vggt_test_smoke", spawn=False)  # no viewer; logs to the global sink
    # Must complete without raising across every active view, both input forms.
    rerun_adapter.log_batch(_make_raw_sample(), spawn=False)
    rerun_adapter.log_batch(_make_composed_sample(), spawn=False)


def test_log_batch_is_robust_to_bad_fields(caplog):
    pytest.importorskip("rerun")
    sample = _make_raw_sample()
    sample["extrinsics"][0][:] = np.nan          # poison a pose
    sample["point_masks"] = [np.zeros((8, 8), bool) for _ in range(2)]  # empty cloud
    # Per-view isolation: the empty-cloud world view raises, _guarded catches it,
    # warns, and the run continues without raising.
    with caplog.at_level(logging.WARNING):
        rerun_adapter.log_batch(sample, spawn=False)
    assert any("failed" in r.getMessage() for r in caplog.records)


def test_log_batch_empty_sample_is_noop(caplog):
    pytest.importorskip("rerun")
    rerun_adapter.log_batch({"modalities": frozenset()}, spawn=False)  # V == 0


def _make_extra_modality_sample(V=2, H=6, W=6, N=4):
    """A sample covering the modalities the other tests don't: depth_confs,
    normals, semantics, tracks, camera_ids (plus camera + rgb so those views
    activate). Used to smoke-run the remaining loggers end-to-end."""
    rng = np.random.default_rng(1)
    return {
        "images": [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(V)],
        "intrinsics": [np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], np.float32)
                       for _ in range(V)],
        "extrinsics": [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.0]], np.float32)
                       for _ in range(V)],
        "depth_confs": [rng.random((H, W)).astype(np.float32) for _ in range(V)],
        "normals": [(rng.random((H, W, 3)).astype(np.float32) * 2 - 1) for _ in range(V)],
        "semantics": [rng.integers(0, 5, (H, W), dtype=np.int32) for _ in range(V)],
        "tracks": [(rng.random((N, 2)).astype(np.float32) * W) for _ in range(V)],
        "camera_ids": np.zeros(V, dtype=np.int32),
        "modalities": frozenset({
            Modality.IMAGE, Modality.INTRINSICS, Modality.EXTRINSICS,
            Modality.DEPTH_CONF, Modality.NORMAL, Modality.SEMANTIC,
            Modality.TRACK, Modality.CAMERA_ID,
        }),
    }


def test_log_batch_exercises_remaining_loggers(caplog):
    # depth_conf / normals / semantics / tracks (and the camera_ids path) are not
    # covered by the synthetic or TUM samples; run them and assert no view warned,
    # i.e. each logger actually succeeded rather than being swallowed by _guarded.
    pytest.importorskip("rerun").init("vggt_test_kitchen", spawn=False)
    with caplog.at_level(logging.WARNING):
        rerun_adapter.log_batch(_make_extra_modality_sample(), spawn=False)
    failures = [r.getMessage() for r in caplog.records if "failed" in r.getMessage()]
    assert not failures, failures


def test_log_batch_accumulate_path_runs_clean(caplog):
    # Drive the accumulate=True branch (world/points/{i}) and point_stride>1
    # through log_batch; assert the world view logged without warning.
    pytest.importorskip("rerun").init("vggt_test_accum", spawn=False)
    with caplog.at_level(logging.WARNING):
        rerun_adapter.log_batch(_make_raw_sample(), spawn=False, accumulate=True, point_stride=2)
    assert not [r for r in caplog.records if "failed" in r.getMessage()]


# Same data location + guard as test_tum_dataset.py
TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
HAVE_TUM = os.path.isdir(TUM_DIR)


def _tum_common_conf():
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


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_log_batch_on_real_tum_sample():
    rr = pytest.importorskip("rerun")
    pytest.importorskip("torch")
    from vggt_omega.datasets.vendors.tum import TumDataset

    ds = TumDataset(
        common_conf=_tum_common_conf(),
        split="train",
        TUM_DIR=TUM_DIR,
        sequences=["rgbd_dataset_freiburg3_sitting_halfsphere"],
        len_train=10,
    )
    sample = ds.get_data(seq_index=0, img_per_seq=2, aspect_ratio=1.0)  # raw numpy dict
    rr.init("vggt_test_tum", spawn=False)
    rerun_adapter.log_batch(sample, spawn=False)  # must complete without raising
