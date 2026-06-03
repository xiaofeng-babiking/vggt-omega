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
from dataclasses import dataclass

import numpy as np

from vggt_omega.datasets.modality import REGISTRY

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


_MODALITY_KEYS = frozenset(spec.key for spec in REGISTRY.values())


@dataclass(frozen=True)
class NormalizedSample:
    """Canonical view of a sample: modality-key -> numpy array (or list for texts)."""
    data: dict
    present: set
    V: int


def _to_numpy(val):
    """Stack a list / detach a tensor / asarray, returning a numpy array.

    `texts` (list[str]) is returned unchanged as a Python list.
    """
    if isinstance(val, list):
        if len(val) and isinstance(val[0], str):
            return list(val)
        if len(val) and torch is not None and torch.is_tensor(val[0]):
            val = [v.detach().cpu().numpy() for v in val]
        return np.stack(val)
    if torch is not None and torch.is_tensor(val):
        return val.detach().cpu().numpy()
    if isinstance(val, str):
        return val
    return np.asarray(val)


def _resolve_present(sample: dict) -> set:
    """The set of modality keys that are both declared/known and non-None."""
    declared = sample.get("modalities")
    if declared is not None:
        keys = {m.value if hasattr(m, "value") else str(m) for m in declared}
    else:
        keys = set(_MODALITY_KEYS)
    return {k for k in keys if k in _MODALITY_KEYS and sample.get(k) is not None}


def normalize_sample(sample: dict) -> NormalizedSample:
    """Normalize a raw-numpy or ComposedDataset-torch sample to canonical form."""
    present = _resolve_present(sample)
    data: dict = {}
    for key in present:
        data[key] = _to_numpy(sample[key])
    if "images" in data:
        data["images"] = _canonical_images(data["images"])
    # Frame count V: leading dim of any per-frame array, else length of texts.
    # Every REGISTRY modality is per_frame=True, so shape[0] == V for any stacked
    # array regardless of which present key we hit first.
    V = 0
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            V = int(val.shape[0])
            break
        if isinstance(val, list):
            V = len(val)
            break
    return NormalizedSample(data=data, present=present, V=V)


def _extrinsic_to_cam_to_world(ext) -> tuple[np.ndarray, np.ndarray]:
    """Invert a world->cam (3,4) OpenCV extrinsic into cam->world (R, t).

    Returns (R_cw, t_cw) such that X_world = R_cw @ X_cam + t_cw, i.e. the
    camera's pose in world space (what Rerun's Transform3D wants).
    """
    ext = np.asarray(ext, dtype=np.float64)
    R = ext[:3, :3]
    t = ext[:3, 3]
    R_cw = R.T
    t_cw = -R.T @ t
    return R_cw.astype(np.float32), t_cw.astype(np.float32)


def log_batch(*args, **kwargs):  # noqa: D401 - real implementation added in a later task
    """Placeholder; implemented in a later task."""
    raise NotImplementedError("log_batch is implemented in a later task")
