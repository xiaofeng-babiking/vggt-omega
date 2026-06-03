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

import colorsys
import logging
from dataclasses import dataclass
from typing import Callable

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


@dataclass(frozen=True)
class Ctx:
    """Per-`log_batch` context handed to every view logger."""
    rr: object
    norm: NormalizedSample
    point_stride: int
    accumulate: bool


def _camera_path(norm: NormalizedSample, i: int) -> str:
    if "camera_ids" in norm.data:
        return f"world/camera/{int(norm.data['camera_ids'][i])}"
    return "world/camera"


def _frame_hw(norm: NormalizedSample):
    """(H, W) of the per-frame rasters, or None if no spatial modality present."""
    for key in ("images", "world_points", "cam_points", "depths", "point_masks", "normals"):
        if key in norm.data:
            shape = norm.data[key].shape
            return int(shape[1]), int(shape[2])
    return None


def _log_camera(ctx: Ctx, i: int) -> None:
    rr, d = ctx.rr, ctx.norm.data
    R_cw, t_cw = _extrinsic_to_cam_to_world(d["extrinsics"][i])
    cam = _camera_path(ctx.norm, i)
    rr.log(cam, rr.Transform3D(translation=t_cw, mat3x3=R_cw))
    K = np.asarray(d["intrinsics"][i], dtype=np.float32)
    hw = _frame_hw(ctx.norm)
    if hw is None:  # no raster -> derive resolution from the principal point
        w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
    else:
        h, w = hw
    rr.log(cam + "/image", rr.Pinhole(image_from_camera=K, resolution=[w, h]))


def _log_trajectory(ctx: Ctx) -> None:
    """Log the full camera-center polyline once, as a static overview."""
    ext = ctx.norm.data["extrinsics"]
    centers = np.stack(
        [_extrinsic_to_cam_to_world(ext[j])[1] for j in range(ctx.norm.V)]
    ).astype(np.float32)
    ctx.rr.log(
        "world/trajectory",
        ctx.rr.LineStrips3D([centers], colors=[[255, 128, 0]]),
        static=True,
    )


def _log_world_points(ctx: Ctx, i: int) -> None:
    rr, d = ctx.rr, ctx.norm.data
    pts = d["world_points"][i].reshape(-1, 3)
    cols = d["images"][i].reshape(-1, 3) if "images" in d else None
    if "point_masks" in d:
        mask = d["point_masks"][i].reshape(-1).astype(bool)
    else:
        mask = np.ones(pts.shape[0], dtype=bool)
    mask &= np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    cols = cols[mask] if cols is not None else None
    stride = max(int(ctx.point_stride), 1)
    pts = pts[::stride]
    cols = cols[::stride] if cols is not None else None
    if pts.shape[0] == 0:
        raise ValueError("no valid world points after masking")
    path = f"world/points/{i}" if ctx.accumulate else "world/points"
    rr.log(path, rr.Points3D(pts, colors=cols))


def _normalize01(a) -> np.ndarray:
    """Min-max normalize finite values to [0, 1]; constant/empty -> zeros."""
    a = np.asarray(a, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _track_colors(n: int) -> np.ndarray:
    """Deterministic, well-spread per-track RGB palette (golden-ratio hues)."""
    out = np.empty((n, 3), dtype=np.uint8)
    for k in range(n):
        r, g, b = colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.8, 1.0)
        out[k] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def _log_rgb(ctx: Ctx, i: int) -> None:
    # Logged on the pinhole entity so the texture maps onto the frustum image plane.
    ctx.rr.log(_camera_path(ctx.norm, i) + "/image", ctx.rr.Image(ctx.norm.data["images"][i]))


def _log_depth(ctx: Ctx, i: int) -> None:
    dep = ctx.norm.data["depths"][i].astype(np.float32)
    vis = np.where(dep > 0, dep, 0.0)  # 0=invalid, <0=sky -> clamp to 0
    ctx.rr.log(_camera_path(ctx.norm, i) + "/image/depth", ctx.rr.DepthImage(vis, meter=1.0))


def _log_depth_conf(ctx: Ctx, i: int) -> None:
    conf = (_normalize01(ctx.norm.data["depth_confs"][i]) * 255).astype(np.uint8)
    ctx.rr.log(_camera_path(ctx.norm, i) + "/image/depth_conf", ctx.rr.Image(conf))


def _log_normals(ctx: Ctx, i: int) -> None:
    n = np.clip(ctx.norm.data["normals"][i].astype(np.float32), -1.0, 1.0)
    rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    ctx.rr.log(_camera_path(ctx.norm, i) + "/image/normals", ctx.rr.Image(rgb))


def _log_semantics(ctx: Ctx, i: int) -> None:
    seg = ctx.norm.data["semantics"][i].astype(np.int32)
    ctx.rr.log(_camera_path(ctx.norm, i) + "/image/semantics", ctx.rr.SegmentationImage(seg))


def _log_seg(name: str, key: str):
    """Factory: a boolean-mask logger that renders as a SegmentationImage."""
    def _fn(ctx: Ctx, i: int) -> None:
        m = ctx.norm.data[key][i].astype(np.uint8)
        ctx.rr.log(_camera_path(ctx.norm, i) + f"/image/{name}", ctx.rr.SegmentationImage(m))
    _fn.__name__ = f"_log_{name}"
    return _fn


def _log_tracks(ctx: Ctx, i: int) -> None:
    tr = np.asarray(ctx.norm.data["tracks"][i], dtype=np.float32)  # (N, 2)
    colors = _track_colors(tr.shape[0])
    ctx.rr.log(
        _camera_path(ctx.norm, i) + "/image/tracks",
        ctx.rr.Points2D(tr, colors=colors, radii=2.0),
    )


def _log_text(ctx: Ctx, i: int) -> None:
    txt = ctx.norm.data["texts"][i]
    ctx.rr.log(_camera_path(ctx.norm, i) + "/text", ctx.rr.TextDocument(str(txt)))


@dataclass(frozen=True)
class View:
    name: str
    requires: set
    optional: set
    log: Callable[[Ctx, int], None]


VIEWS = [
    View("camera",    {"intrinsics", "extrinsics"}, {"camera_ids"},           _log_camera),
    View("rgb",       {"images"},        set(),                               _log_rgb),
    View("world",     {"world_points"},  {"images", "point_masks"},           _log_world_points),
    View("depth",     {"depths"},        set(),                               _log_depth),
    View("depthconf", {"depth_confs"},   set(),                               _log_depth_conf),
    View("normals",   {"normals"},       set(),                               _log_normals),
    View("semantics", {"semantics"},     set(),                               _log_semantics),
    View("skymask",   {"sky_masks"},     set(),                  _log_seg("sky_mask", "sky_masks")),
    View("pointmask", {"point_masks"},   set(),              _log_seg("point_mask", "point_masks")),
    View("tracks",    {"tracks"},        {"images"},                          _log_tracks),
    View("text",      {"texts"},         set(),                               _log_text),
]


def select_views(present) -> list:
    """Views whose required modalities are all present, in registry order."""
    present = set(present)
    return [v for v in VIEWS if v.requires <= present]


def log_batch(*args, **kwargs):  # noqa: D401 - real implementation added in a later task
    """Placeholder; implemented in a later task."""
    raise NotImplementedError("log_batch is implemented in a later task")
