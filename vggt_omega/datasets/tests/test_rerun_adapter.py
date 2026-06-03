import sys

import numpy as np
import pytest

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
