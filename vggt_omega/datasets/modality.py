"""General, self-describing modality schema for VGGT-Omega datasets.

Each `Modality` member's *value* is the sample-dict key, so a sample is
self-describing. `REGISTRY` maps every modality to a `ModalitySpec` (dtype,
per-frame shape rank, derived flag) that drives validation and downstream
consumers (training, eval, the future Rerun adapter). Vendors declare
`available_modalities`; the loader carries through whatever is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch


class Modality(str, Enum):
    IMAGE = "images"
    DEPTH = "depths"
    INTRINSICS = "intrinsics"
    EXTRINSICS = "extrinsics"          # world->camera, OpenCV (3,4)
    POINT_MASK = "point_masks"
    WORLD_POINTS = "world_points"
    CAM_POINTS = "cam_points"
    SKY_MASK = "sky_masks"
    DEPTH_CONF = "depth_confs"
    NORMAL = "normals"
    TRACK = "tracks"
    TEXT = "texts"
    SEMANTIC = "semantics"
    TIMESTAMP = "timestamps"
    CAMERA_ID = "camera_ids"


@dataclass(frozen=True)
class ModalitySpec:
    modality: "Modality"
    per_frame: bool          # stacked along the view axis V
    dtype: str               # numpy dtype name, or "str"
    rank: int                # per-frame ndim (0 for scalar/str)
    derived: bool = False     # produced by BaseDataset.process_one_image
    description: str = ""

    @property
    def key(self) -> str:
        return self.modality.value


REGISTRY: dict[Modality, ModalitySpec] = {
    Modality.IMAGE:        ModalitySpec(Modality.IMAGE, True, "float32", 3, False, "RGB (3,H,W) in [0,1]"),
    Modality.DEPTH:        ModalitySpec(Modality.DEPTH, True, "float32", 2, False, "(H,W) m; 0=invalid, <0=sky"),
    Modality.INTRINSICS:   ModalitySpec(Modality.INTRINSICS, True, "float32", 2, False, "(3,3) pinhole K px"),
    Modality.EXTRINSICS:   ModalitySpec(Modality.EXTRINSICS, True, "float32", 2, False, "(3,4) world->cam OpenCV"),
    Modality.POINT_MASK:   ModalitySpec(Modality.POINT_MASK, True, "bool", 2, True, "(H,W) valid-depth mask"),
    Modality.WORLD_POINTS: ModalitySpec(Modality.WORLD_POINTS, True, "float32", 3, True, "(H,W,3) world frame"),
    Modality.CAM_POINTS:   ModalitySpec(Modality.CAM_POINTS, True, "float32", 3, True, "(H,W,3) camera frame"),
    Modality.SKY_MASK:     ModalitySpec(Modality.SKY_MASK, True, "bool", 2, True, "(H,W) depth<0"),
    Modality.DEPTH_CONF:   ModalitySpec(Modality.DEPTH_CONF, True, "float32", 2, False, "(H,W) depth confidence"),
    Modality.NORMAL:       ModalitySpec(Modality.NORMAL, True, "float32", 3, False, "(H,W,3) cam-frame normals"),
    Modality.TRACK:        ModalitySpec(Modality.TRACK, True, "float32", 2, False, "(N,2) per frame"),
    Modality.TEXT:         ModalitySpec(Modality.TEXT, True, "str", 0, False, "per-frame caption/text"),
    Modality.SEMANTIC:     ModalitySpec(Modality.SEMANTIC, True, "int32", 2, False, "(H,W) label ids"),
    Modality.TIMESTAMP:    ModalitySpec(Modality.TIMESTAMP, True, "float64", 0, False, "() seconds"),
    Modality.CAMERA_ID:    ModalitySpec(Modality.CAMERA_ID, True, "int32", 0, False, "() multi-cam id"),
}

CORE: frozenset = frozenset({
    Modality.IMAGE, Modality.DEPTH, Modality.INTRINSICS, Modality.EXTRINSICS,
    Modality.POINT_MASK, Modality.WORLD_POINTS, Modality.CAM_POINTS,
})

DERIVED: frozenset = frozenset(m for m, s in REGISTRY.items() if s.derived)


def validate_sample(sample: dict[str, Any], loaded) -> None:
    """Assert every loaded modality key is present and that the sample's
    ``modalities`` field equals ``loaded``. Raises ValueError on mismatch."""
    loaded = set(loaded)
    missing = {m for m in loaded if m.value not in sample}
    if missing:
        raise ValueError(f"sample missing loaded modalities: {sorted(m.name for m in missing)}")
    declared = set(sample.get("modalities", set()))
    if declared != loaded:
        raise ValueError(
            f"sample.modalities {sorted(d.name for d in declared)} != loaded {sorted(m.name for m in loaded)}"
        )


def carry_extra_modalities(batch: dict, sample: dict) -> dict:
    """Tensorize and copy any registered modality present in ``batch`` that the
    core ComposedDataset did not already place into ``sample`` (e.g. timestamps,
    sky_masks), plus the ``modalities`` field. Mutates and returns ``sample``."""
    for mod, spec in REGISTRY.items():
        key = spec.key
        if key in sample or key not in batch or batch[key] is None:
            continue
        val = batch[key]
        if spec.dtype == "str":
            sample[key] = list(val)
        else:
            arr = np.stack(val) if isinstance(val, list) else np.asarray(val)
            sample[key] = torch.from_numpy(arr.astype(spec.dtype))
    if "modalities" in batch:
        # Store as a sorted list of Modality values so PyTorch's default collate
        # can stack/batch the field (frozenset is not collatable).
        sample["modalities"] = sorted(m.value for m in batch["modalities"])
    return sample
