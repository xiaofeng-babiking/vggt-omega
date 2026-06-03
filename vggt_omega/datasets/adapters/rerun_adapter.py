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


def log_batch(*args, **kwargs):  # noqa: D401 - real implementation added in a later task
    """Placeholder; implemented in a later task."""
    raise NotImplementedError("log_batch is implemented in a later task")
