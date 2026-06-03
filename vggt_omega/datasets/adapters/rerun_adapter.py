"""Unified dataset-to-Rerun adapter.

`log_batch(sample)` logs any VGGT-Omega dataset sample to Rerun for quick
visual inspection. It accepts either the raw numpy dict from a vendor's
`get_data()` or the torch dict from `ComposedDataset.__getitem__`, normalizes
both to one canonical form, then renders every modality present (and silently
skips the rest) by consulting a registry of views keyed on the `Modality`
contract in `vggt_omega.datasets.modality`.

Rerun is an *optional* dependency (`pip install 'vggt-omega[viz]'`) and is
imported lazily, so this module imports fine without it — only `log_batch`
requires it at call time.
"""
from __future__ import annotations

import logging

import numpy as np

try:  # torch is only needed to read ComposedDataset tensors; raw numpy dicts don't need it.
    import torch
except ImportError:  # pragma: no cover - torch is present in any real dataset env
    torch = None

logger = logging.getLogger(__name__)


def _require_rerun():
    """Import and return the `rerun` module, or raise a helpful ImportError."""
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "Rerun is required for dataset visualization. "
            "Install it with: pip install 'vggt-omega[viz]'"
        ) from exc
    return rr


def _canonical_images(arr) -> np.ndarray:
    """Normalize a stacked image array to (V, H, W, 3) uint8 in [0, 255].

    Accepts channel-first (V,3,H,W) or channel-last (V,H,W,3), float in [0,1]
    or any numeric range. Channel axis is detected by position; floats whose
    max is <= 1 are assumed normalized and scaled to [0, 255].
    """
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"images must be 4D (V,...), got shape {arr.shape}")
    # Channel-first -> channel-last (only when axis 1 is the 3-channel one).
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"cannot locate 3-channel axis in images shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0 + 1e-4:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def log_batch(*args, **kwargs):  # noqa: D401 - real implementation added in a later task
    """Placeholder; implemented in a later task."""
    raise NotImplementedError("log_batch is implemented in a later task")
