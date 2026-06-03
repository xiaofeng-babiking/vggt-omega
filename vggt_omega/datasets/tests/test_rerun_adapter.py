import sys

import numpy as np
import pytest

from vggt_omega.datasets.adapters import rerun_adapter


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
