"""Unified dataset-to-Rerun adapter for headless visual inspection.

:class:`RerunDataset` wraps any VGGT-Omega **dataset** (a ``ComposedDataset`` or
a raw vendor) into a transparent, drop-in Rerun-logging dataset: every sample it
yields -- via ``ds[idx]`` (the training/DataLoader path) or ``ds.get_sample(...)``
(the ordered eval path) -- passes through UNCHANGED and is logged to a Rerun
recording as a side effect. So you insert visualization into any train or test
pipeline exactly like a torch Dataset / DataLoader wrapper, without touching the
pipeline's data flow. Under the hood it calls ``log_sample`` per fetched sample.

``log_sample(sample, recording)`` is that per-sample primitive: it logs one
VGGT-Omega sample dict to a `Rerun <https://rerun.io>`_ recording. "Unified"
means it keys off the self-describing :mod:`vggt_omega.datasets.modality`
contract: it renders every modality the sample actually carries (RGB, depth,
world point cloud, camera frusta, masks, normals, semantics, tracks, text, ...)
and silently skips the rest, so it works for every current and future vendor
with no per-dataset code.

Each frame's modalities are stamped on the frame's **timestamp** (a ``time``
timeline in elapsed seconds, taken from the sample's ``timestamps`` modality)
*and* a plain ``frame`` index timeline, so scrubbing in either reproduces real
capture timing.

Headless workflow (this host has no display)::

    # 1. write one .rrd per sequence (no viewer needed)
    python -m vggt_omega.datasets.adapters \
        --configure vggt_omega/datasets/config/tum.yaml --out rerun_out

    # 2. serve the web viewer and open the printed URL in a browser
    rerun --serve-web rerun_out/*.rrd

Or stream live with no files -- start ``rerun --serve-web`` first, then::

    python -m vggt_omega.datasets.adapters \
        --configure vggt_omega/datasets/config/tum.yaml --connect

(``--serve`` instead hosts an in-process gRPC server and prints its URI.)

Rerun is an *optional* dependency (``pip install 'vggt-omega[viz]'``) imported
lazily inside :func:`_require_rerun`, so this module imports fine without it --
only the logging entry points need it at call time. The pure helpers
(``normalize_sample``, the math) are importable and testable without Rerun.
"""
from __future__ import annotations

import argparse
import colorsys
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from vggt_omega.datasets.modality import REGISTRY

try:  # torch is only needed to read ComposedDataset tensors; raw numpy dicts don't.
    import torch
except ImportError:  # pragma: no cover - torch is present in any real dataset env
    torch = None

logger = logging.getLogger(__name__)

# Every registry modality key (the sample-dict keys this adapter knows about).
_MODALITY_KEYS = frozenset(spec.key for spec in REGISTRY.values())


def _require_rerun():
    """Import and return the ``rerun`` module, or raise a helpful ImportError."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "Rerun is required for dataset visualization. "
            "Install it with: pip install 'vggt-omega[viz]'"
        ) from exc
    return rr


# --- Normalization shim ------------------------------------------------------
# Absorbs every difference between the two sample forms so the loggers below see
# exactly one canonical representation:
#   * raw  get_data():            numpy, per-frame lists, images (V,H,W,3) [0,255]
#   * ComposedDataset.__getitem__: torch, stacked (V,...),  images (V,3,H,W) [0,1]


@dataclass(frozen=True)
class NormalizedSample:
    """Canonical view of one sample: modality-key -> numpy array (or list[str]
    for ``texts``), the set of ``present`` modality keys, and the frame count
    ``V``."""

    data: dict
    present: set
    V: int


def _canonical_images(arr) -> np.ndarray:
    """Normalize a stacked image array to ``(V, H, W, 3)`` uint8 in ``[0, 255]``.

    Accepts channel-first ``(V,3,H,W)`` or channel-last ``(V,H,W,3)``, float in
    ``[0,1]`` or uint8. The channel axis is found by position; floats whose max
    is ``<= 1`` are assumed normalized and scaled to ``[0, 255]``.
    """
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"images must be 4D (V,...), got shape {arr.shape}")
    # Channel-first -> channel-last, but only when axis 1 is the 3-channel one
    # and the last axis is not already 3 (avoid touching genuine (V,H,W,3)).
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"cannot locate 3-channel axis in images shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        # Floats are the normalized [0,1] form (ComposedDataset divides by 255).
        # Use a generous cutoff so blend/augmentation overshoot just past 1.0
        # (e.g. 1.05) is still scaled, not clipped to black -- nothing legitimately
        # lands in (2, 50), so 2.0 cleanly separates [0,1] from already-[0,255] floats.
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        if peak <= 2.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _to_numpy(val):
    """Stack a per-frame list / detach a tensor / asarray -> numpy array.

    A ``texts`` list (``list[str]``) is returned unchanged as a Python list.
    """
    if isinstance(val, list):
        if len(val) and isinstance(val[0], str):
            return list(val)
        if len(val) and torch is not None and torch.is_tensor(val[0]):
            val = [v.detach().cpu().numpy() for v in val]
        return np.stack(val)
    if torch is not None and torch.is_tensor(val):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def _present_keys(sample: dict) -> set:
    """Registry modality keys that are actually carried (non-None, non-empty).

    Deliberately driven by *what the dict contains*, not the sample's
    ``modalities`` field: that field advertises evaluable ground-truth, whereas a
    viz tool should also render derived geometry (e.g. ``world_points``) that the
    pipeline computed but did not declare as scorable GT.
    """
    keys = set()
    for key in _MODALITY_KEYS:
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and len(val) == 0:
            continue
        keys.add(key)
    return keys


def normalize_sample(sample: dict) -> NormalizedSample:
    """Normalize a raw-numpy or ComposedDataset-torch sample to canonical form.

    For already-stacked numpy inputs the returned arrays may *alias* the input
    (no defensive copy); the loggers only read them, but callers that mutate the
    sample afterwards should copy first."""
    present = _present_keys(sample)
    data: dict = {}
    for key in present:
        data[key] = _to_numpy(sample[key])
    if "images" in data:
        data["images"] = _canonical_images(data["images"])
    # Frame count V: leading dim of any per-frame array, else length of a list.
    V = 0
    for val in data.values():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            V = int(val.shape[0])
            break
        if isinstance(val, list):
            V = len(val)
            break
    return NormalizedSample(data=data, present=present, V=V)


# --- Geometry helpers --------------------------------------------------------


def _extrinsic_to_cam_to_world(ext) -> tuple[np.ndarray, np.ndarray]:
    """Invert a world->cam ``(3,4)`` OpenCV extrinsic into cam->world ``(R, t)``.

    Returns ``(R_cw, t_cw)`` such that ``X_world = R_cw @ X_cam + t_cw`` -- the
    camera's pose in world space, which is exactly what Rerun's ``Transform3D``
    at the camera entity expects. Getting this backwards is the classic
    "cameras fly off into space" bug, so it lives in one isolated, tested place.
    """
    ext = np.asarray(ext, dtype=np.float64)
    R = ext[:3, :3]
    t = ext[:3, 3]
    R_cw = R.T
    t_cw = -R.T @ t
    return R_cw.astype(np.float32), t_cw.astype(np.float32)


def _relative_times(norm: NormalizedSample) -> Optional[np.ndarray]:
    """Per-frame elapsed seconds from the ``timestamps`` modality, offset so the
    sequence starts at 0 (TUM timestamps are absolute Unix epoch, 7-Scenes are
    already relative -- offsetting makes both read as a clean elapsed-time axis).
    Returns ``None`` when no timestamps are present (callers fall back to the
    frame-index timeline)."""
    if "timestamps" not in norm.data:
        return None
    ts = np.asarray(norm.data["timestamps"], dtype=np.float64).reshape(-1)
    if ts.size != norm.V:
        # timestamps is spec rank-0 (one scalar per frame); a length != V means a
        # malformed/multi-dim field. Don't silently misalign -- use the frame timeline.
        logger.warning(
            "timestamps length %d != V=%d; falling back to the frame index timeline",
            ts.size, norm.V,
        )
        return None
    finite = ts[np.isfinite(ts)]
    if finite.size == 0:
        return None
    # Offset by the finite minimum so a stray NaN can't poison the whole axis
    # (per-frame NaNs are then skipped in _set_time, keeping the rest on-timeline).
    return ts - float(finite.min())


# --- Per-call context + small path/shape helpers -----------------------------


@dataclass(frozen=True)
class Ctx:
    """Per-:func:`log_sample` context handed to every view logger.

    ``rec`` is the target Rerun ``RecordingStream`` (``None`` = the process-wide
    current recording). ``rr`` is the rerun module (used to build archetypes)."""

    rr: object
    rec: object
    norm: NormalizedSample
    point_stride: int
    accumulate: bool


def _emit(ctx: Ctx, path: str, archetype, *, static: bool = False) -> None:
    """Log one archetype to the context's recording (explicit ``recording=`` so
    multiple per-sequence recordings never cross-contaminate)."""
    ctx.rr.log(path, archetype, static=static, recording=ctx.rec)


def _camera_path(norm: NormalizedSample, i: int) -> str:
    """Camera entity path; namespaced by ``camera_ids`` for multi-cam rigs."""
    if "camera_ids" in norm.data:
        return f"world/camera/{int(norm.data['camera_ids'][i])}"
    return "world/camera"


def _frame_hw(norm: NormalizedSample):
    """``(H, W)`` of the per-frame rasters, or ``None`` if no spatial modality.

    Skips any candidate whose array is not at least 3-D ``(V, H, W, ...)`` so a
    malformed/degenerate modality can't raise an ``IndexError`` here."""
    for key in ("images", "world_points", "cam_points", "depths", "point_masks", "normals", "semantics"):
        arr = norm.data.get(key)
        if arr is not None and getattr(arr, "ndim", 0) >= 3:
            return int(arr.shape[1]), int(arr.shape[2])
    return None


def _normalize01(a) -> np.ndarray:
    """Min-max normalize finite values to ``[0, 1]``; constant/empty -> zeros."""
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
    out = np.empty((max(n, 0), 3), dtype=np.uint8)
    for k in range(n):
        r, g, b = colorsys.hsv_to_rgb((k * 0.61803398875) % 1.0, 0.8, 1.0)
        out[k] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


# --- Geometry loggers --------------------------------------------------------


def _log_camera(ctx: Ctx, i: int) -> None:
    rr, d = ctx.rr, ctx.norm.data
    R_cw, t_cw = _extrinsic_to_cam_to_world(d["extrinsics"][i])
    if not (np.isfinite(R_cw).all() and np.isfinite(t_cw).all()):
        raise ValueError("non-finite extrinsic")
    cam = _camera_path(ctx.norm, i)
    _emit(ctx, cam, rr.Transform3D(translation=t_cw, mat3x3=R_cw))
    K = np.asarray(d["intrinsics"][i], dtype=np.float32)
    hw = _frame_hw(ctx.norm)
    if hw is None:  # no raster -> derive resolution from the principal point
        w, h = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
    else:
        h, w = hw
    _emit(ctx, cam + "/image", rr.Pinhole(image_from_camera=K, resolution=[w, h]))


def _log_trajectory(ctx: Ctx) -> None:
    """Log the full camera-center polyline once, as a static spatial overview."""
    ext = ctx.norm.data["extrinsics"]
    centers = np.stack(
        [_extrinsic_to_cam_to_world(ext[j])[1] for j in range(ctx.norm.V)]
    ).astype(np.float32)
    centers = centers[np.isfinite(centers).all(axis=1)]
    if centers.shape[0] < 2:
        return
    _emit(ctx, "world/trajectory", ctx.rr.LineStrips3D([centers], colors=[[255, 128, 0]]), static=True)


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
    # accumulate=False: replace at one path so scrubbing shows the current
    # frame's cloud; accumulate=True: a per-frame path keeps the full scene.
    path = f"world/points/{i}" if ctx.accumulate else "world/points"
    _emit(ctx, path, rr.Points3D(pts, colors=cols))


# --- Image-plane loggers (children of the pinhole entity) --------------------


def _log_rgb(ctx: Ctx, i: int) -> None:
    # Logged on the pinhole entity so the texture maps onto the frustum image plane.
    _emit(ctx, _camera_path(ctx.norm, i) + "/image", ctx.rr.Image(ctx.norm.data["images"][i]))


def _log_depth(ctx: Ctx, i: int) -> None:
    dep = ctx.norm.data["depths"][i].astype(np.float32)
    vis = np.where(dep > 0, dep, 0.0)  # 0=invalid, <0=sky -> clamp to 0
    _emit(ctx, _camera_path(ctx.norm, i) + "/image/depth", ctx.rr.DepthImage(vis, meter=1.0))


def _log_depth_conf(ctx: Ctx, i: int) -> None:
    conf = (_normalize01(ctx.norm.data["depth_confs"][i]) * 255).astype(np.uint8)
    _emit(ctx, _camera_path(ctx.norm, i) + "/image/depth_conf", ctx.rr.Image(conf))


def _log_normals(ctx: Ctx, i: int) -> None:
    n = np.clip(ctx.norm.data["normals"][i].astype(np.float32), -1.0, 1.0)
    rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)  # [-1,1] -> [0,255]
    _emit(ctx, _camera_path(ctx.norm, i) + "/image/normals", ctx.rr.Image(rgb))


def _log_semantics(ctx: Ctx, i: int) -> None:
    seg = ctx.norm.data["semantics"][i].astype(np.int32)
    _emit(ctx, _camera_path(ctx.norm, i) + "/image/semantics", ctx.rr.SegmentationImage(seg))


def _log_seg(name: str, key: str):
    """Factory: a boolean-mask logger that renders as a SegmentationImage."""

    def _fn(ctx: Ctx, i: int) -> None:
        m = ctx.norm.data[key][i].astype(np.uint8)
        _emit(ctx, _camera_path(ctx.norm, i) + f"/image/{name}", ctx.rr.SegmentationImage(m))

    _fn.__name__ = f"_log_{name}"
    return _fn


def _log_tracks(ctx: Ctx, i: int) -> None:
    tr = np.asarray(ctx.norm.data["tracks"][i], dtype=np.float32)  # (N, 2)
    colors = _track_colors(tr.shape[0])
    _emit(
        ctx,
        _camera_path(ctx.norm, i) + "/image/tracks",
        ctx.rr.Points2D(tr, colors=colors, radii=2.0),
    )


def _log_text(ctx: Ctx, i: int) -> None:
    txt = ctx.norm.data["texts"][i]
    _emit(ctx, _camera_path(ctx.norm, i) + "/text", ctx.rr.TextDocument(str(txt)))


# --- View registry + dispatch ------------------------------------------------
# Each view declares the modalities it needs; a view runs iff its `requires` are
# all present. `cam_points` has no view on purpose: it is redundant with the
# `world_points` cloud once the camera transform is applied, and skipping it
# avoids doubling the per-frame point count. Adding a future modality = append
# one View + its `_log_*`.


@dataclass(frozen=True)
class View:
    name: str
    requires: set
    optional: set
    log: Callable[[Ctx, int], None]


VIEWS = [
    View("camera",    {"intrinsics", "extrinsics"}, {"camera_ids"},            _log_camera),
    View("rgb",       {"images"},        set(),                                _log_rgb),
    View("world",     {"world_points"},  {"images", "point_masks"},            _log_world_points),
    View("depth",     {"depths"},        set(),                                _log_depth),
    View("depthconf", {"depth_confs"},   set(),                                _log_depth_conf),
    View("normals",   {"normals"},       set(),                                _log_normals),
    View("semantics", {"semantics"},     set(),                                _log_semantics),
    View("skymask",   {"sky_masks"},     set(),                    _log_seg("sky_mask", "sky_masks")),
    View("pointmask", {"point_masks"},   set(),                _log_seg("point_mask", "point_masks")),
    View("tracks",    {"tracks"},        {"images"},                           _log_tracks),
    View("text",      {"texts"},         set(),                                _log_text),
]


# Fail fast on a mistyped modality key in any View: an unknown key in `requires`
# would never match `present`, silently dropping that modality's visualization.
for _v in VIEWS:
    _unknown = (_v.requires | _v.optional) - _MODALITY_KEYS
    if _unknown:
        raise ValueError(
            f"View {_v.name!r} references unknown modality keys {sorted(_unknown)}; "
            f"valid keys are {sorted(_MODALITY_KEYS)}"
        )


def select_views(present) -> list:
    """Views whose required modalities are all present, in registry order."""
    present = set(present)
    return [v for v in VIEWS if v.requires <= present]


# --- Timeline + orchestration ------------------------------------------------


def _set_time(ctx: Ctx, rel_times: Optional[np.ndarray], i: int) -> None:
    """Place frame ``i`` on both a ``frame`` index timeline and (when timestamps
    exist) a ``time`` elapsed-seconds timeline -- every modality logged after
    this for frame ``i`` inherits both, i.e. each modality is logged on its
    timestamp."""
    ctx.rr.set_time("frame", sequence=int(i), recording=ctx.rec)
    # Guard finiteness: rr.set_time(duration=NaN) raises (NaN->int), and this call
    # is intentionally outside _guarded, so one bad timestamp must not crash the run.
    if rel_times is not None and i < len(rel_times) and np.isfinite(rel_times[i]):
        ctx.rr.set_time("time", duration=float(rel_times[i]), recording=ctx.rec)


def _guarded(fn, *args) -> None:
    """Run a view logger; on failure warn and continue (a quick-viz tool must
    never crash on one bad field)."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - per-view isolation is intentional
        logger.warning("rerun view %s failed: %s", getattr(fn, "__name__", fn), exc)


def log_sample(
    sample: dict,
    recording=None,
    *,
    point_stride: int = 4,
    accumulate: bool = False,
) -> None:
    """Log one dataset sample (``V`` frames of one sequence) to a Rerun recording.

    Args:
        sample: a raw ``get_data()`` numpy dict OR a ``ComposedDataset`` torch dict.
        recording: target ``rerun.RecordingStream``; ``None`` uses the current
            process-wide recording (e.g. after ``rerun.init(...)``).
        point_stride: subsample factor for the world point cloud (speed / size);
            values < 1 are treated as 1 (keep all points).
        accumulate: keep every frame's cloud (``world/points/{i}``) vs. replace
            at ``world/points`` each frame for clean timeline scrubbing.

    Renders every modality the sample carries and skips the rest. Per-view
    isolation: a malformed modality warns and is skipped for that frame; the
    rest continue.
    """
    rr = _require_rerun()
    norm = normalize_sample(sample)
    if norm.V == 0:
        logger.warning("log_sample: empty sample (V=0); nothing to log")
        return
    ctx = Ctx(rr=rr, rec=recording, norm=norm, point_stride=point_stride, accumulate=accumulate)
    # OpenCV camera convention (X-right, Y-down, Z-forward) for the whole scene.
    _emit(ctx, "world", rr.ViewCoordinates.RDF, static=True)
    if "extrinsics" in norm.present:
        _guarded(_log_trajectory, ctx)
    rel_times = _relative_times(norm)
    active = select_views(norm.present)
    for i in range(norm.V):
        _set_time(ctx, rel_times, i)
        for view in active:
            _guarded(view.log, ctx, i)


def sample_to_rrd(
    sample: dict,
    path: str,
    *,
    app_id: str = "vggt_dataset",
    point_stride: int = 4,
    accumulate: bool = False,
) -> str:
    """Log one sample to a fresh recording and save it to a ``.rrd`` file.

    The headless entry point: no viewer is opened. View the result with
    ``rerun --serve-web <path>`` (or ``rerun <path>`` on a machine with a
    display). Returns the written path.
    """
    rr = _require_rerun()
    rec = rr.RecordingStream(application_id=app_id)
    log_sample(sample, recording=rec, point_stride=point_stride, accumulate=accumulate)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    rec.save(str(path))
    return str(path)


# --- Dataset wrapper: drop-in Rerun logging for any train/test pipeline -------
# log_sample / sample_to_rrd above operate on a single sample DICT. RerunDataset
# lifts that to the *dataset* level: wrap any VGGT-Omega dataset and every sample
# it yields is logged to Rerun as a side effect, while the sample itself passes
# through UNCHANGED. So it slots into a training / eval DataLoader exactly like
# the dataset it wraps -- visualization with zero changes to the pipeline.


class RerunDataset:
    """Transparent Rerun-logging wrapper around any VGGT-Omega dataset.

    Use it exactly like the dataset it wraps. It is a map-style dataset
    (``__len__`` + ``__getitem__``) and forwards every other attribute
    (``num_sequences``, ``sequence_name``, ``native_image_size``,
    ``set_img_size``, ``img_size``, ...) to the inner dataset, so it is a literal
    drop-in. Every sample fetched through ``ds[idx]`` (the DataLoader / training
    path) or ``ds.get_sample(...)`` (the ordered eval / inference path) is
    returned UNCHANGED and, as a side effect, logged to a Rerun recording::

        # 1) one .rrd file per sequence -- offline, fork-safe with DataLoader workers
        ds = RerunDataset(base_ds, out_dir="rerun_out")
        for sample in DataLoader(ds, batch_size=None, num_workers=4):
            train_step(sample)            # each sample also written to rerun_out/

        # 2) stream every sample into one live/shared recording (single process)
        rr.init("vggt", spawn=True)       # or pass recording=rr.RecordingStream(...)
        ds = RerunDataset(base_ds)        # logs to the current recording
        for sample in ds:                 # scrub the `sample` timeline in the viewer
            ...

        # 3) drop-in for the eval/inference path (same signature as ComposedDataset)
        viz = RerunDataset(dataset, out_dir="rerun_out")
        sample = viz.get_sample(seq_index, ids=frame_ids, aspect_ratio=ar)

    Logging happens at the *sample* (pre-collation) level on purpose: that is
    where shapes match :func:`log_sample` (``V`` frames, no batch dim). Wrap the
    dataset, not the DataLoader.

    DataLoader workers (``num_workers > 0``): use ``out_dir`` mode -- each sample
    is logged via its own short-lived recording, which is fork-safe, so the
    ``.rrd`` files are written correctly from worker processes. The
    ``recording=`` / current-recording modes hold a live ``RecordingStream`` that
    does NOT pickle to workers; use them single-process only (``num_workers=0``).
    Note the ``_logged`` counter and ``saved_paths`` list are per-process: a
    worker mutates its own copy, so the main process only sees what it logged
    itself (serial access, as the CLI does); two workers handling the same
    ``seq_name`` write the same file (last writer wins -- harmless for viz).

    Args:
        dataset: the dataset to wrap -- a
            :class:`~vggt_omega.datasets.composed_dataset.ComposedDataset` or any
            object whose items are VGGT-Omega sample dicts.
        recording: target ``rerun.RecordingStream`` (shared-recording mode);
            ``None`` *and* no ``out_dir`` uses the process-wide current recording
            (e.g. after ``rerun.init(...)``). Mutually exclusive with ``out_dir``.
            Single-process only (not fork-safe -- see above).
        out_dir: directory for per-sample ``.rrd`` files (file mode). Each sample
            is saved to ``<out_dir>/<seq_name>.rrd`` (a sample with no
            ``seq_name`` falls back to ``sample_<n>.rrd``). Mutually exclusive
            with ``recording``. The fork-safe mode for DataLoader workers.
        point_stride, accumulate: forwarded to :func:`log_sample`.

    Notes:
        * A logging failure on one sample is warned and swallowed -- a viz
          side-channel must never break the data pipeline -- but a *missing
          Rerun install* is surfaced (it is a setup error, not a bad sample).
        * Shared-recording mode places each sample on an outer ``sample``
          timeline (a monotonic fetch counter) so samples never collide; this
          suits *ordered* eval/inference (``get_sample``). Under a shuffled
          training loader the timeline is fetch order, not sequence order, and
          ``log_sample``'s static overview (``world`` view coords,
          ``world/trajectory``) is last-writer-wins across samples -- use
          ``out_dir`` (one recording per sequence) for per-sequence fidelity.
    """

    def __init__(
        self,
        dataset,
        *,
        recording=None,
        out_dir: Optional[str] = None,
        point_stride: int = 4,
        accumulate: bool = False,
    ):
        if recording is not None and out_dir is not None:
            raise ValueError(
                "pass at most one of `recording` (shared-recording mode) or "
                "`out_dir` (per-sample file mode), not both"
            )
        self.dataset = dataset
        self.recording = recording
        self.out_dir = out_dir
        self.point_stride = point_stride
        self.accumulate = accumulate
        # Monotonic count of logged samples -> the outer `sample` timeline value
        # (shared mode), so repeated / shuffled fetches occupy distinct coords.
        self._logged = 0
        # Paths written in file mode, in fetch order (lets the CLI report them).
        self.saved_paths: list = []

    # -- map-style dataset surface (everything else delegates) ----------------

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        """Fetch ``self.dataset[idx]``, log it, and return it UNCHANGED. ``idx``
        is forwarded verbatim, so the training sampler's ``(seq, n, ar)`` tuple
        works as-is."""
        sample = self.dataset[idx]
        self._log(sample)
        return sample

    def get_sample(self, seq_index, ids, aspect_ratio=1.0):
        """Delegating wrapper over the inner dataset's ordered-ids eval/inference
        path (:meth:`ComposedDataset.get_sample`) that also logs the returned
        sample. Same signature, so it is drop-in for ``inference.py``."""
        sample = self.dataset.get_sample(seq_index, ids=ids, aspect_ratio=aspect_ratio)
        self._log(sample)
        return sample

    def __getattr__(self, name):
        # Invoked ONLY for attributes missing on the wrapper itself; forward to
        # the inner dataset so num_sequences()/sequence_name()/img_size/... work.
        # Guard `dataset` to avoid infinite recursion before __init__ sets it.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __repr__(self) -> str:
        sink = (
            f"out_dir={self.out_dir!r}" if self.out_dir is not None
            else "recording=<shared>" if self.recording is not None
            else "recording=<current>"
        )
        return f"RerunDataset({self.dataset!r}, {sink})"

    # -- logging --------------------------------------------------------------

    def _log(self, sample) -> None:
        """Log one fetched sample, then bump the sample counter. A bad sample is
        warned and skipped (the pipeline continues); a missing Rerun install is
        re-raised (it is a setup error you opted into, not a data issue)."""
        try:
            if self.out_dir is not None:
                seq_name = sample.get("seq_name", f"sample_{self._logged}")
                path = os.path.join(self.out_dir, _safe_name(str(seq_name)) + ".rrd")
                sample_to_rrd(
                    sample, path, app_id=str(seq_name),
                    point_stride=self.point_stride, accumulate=self.accumulate,
                )
                self.saved_paths.append(path)
            else:
                rr = _require_rerun()
                # Outer `sample` timeline keyed by the monotonic counter so each
                # sample occupies a distinct coordinate even under shuffling.
                rr.set_time("sample", sequence=self._logged, recording=self.recording)
                log_sample(
                    sample, recording=self.recording,
                    point_stride=self.point_stride, accumulate=self.accumulate,
                )
        except ImportError:
            raise  # missing Rerun is a setup error -> surface it, don't swallow
        except Exception as exc:  # noqa: BLE001 - viz must never break the pipeline
            logger.warning("RerunDataset: failed to log sample %d: %s", self._logged, exc)
        finally:
            self._logged += 1


# --- CLI: dataset configure -> .rrd files OR live gRPC stream ----------------
# Mirrors inference.py's loader (same per-dataset --configure YAML, native-
# resolution sampling, evenly-spaced ordered frame ids) so what you visualize is
# exactly what training/inference tensorizes. Each sequence becomes its own Rerun
# recording, written to a .rrd (default) or streamed live to a viewer
# (--connect / --serve) with no file.

# Default gRPC endpoint a running `rerun --serve-web` listens on (its proxy).
DEFAULT_GRPC_URL = "rerun+http://127.0.0.1:9876/proxy"


def _effective_long_side(native_long: int, image_scale: float) -> int:
    """Native long side x ``image_scale``, snapped to a /16 multiple (ViT-friendly)."""
    return max(16, int(round(native_long * image_scale / 16)) * 16)


def _resolve_frame_ids(dataset, seq_index: int, num_frames: int) -> np.ndarray:
    """Ordered frame ids for one sequence: all frames (``num_frames<=0``) or
    evenly spaced across the sequence."""
    num_available = dataset.sequence_num_frames(seq_index)
    if num_frames <= 0 or num_frames >= num_available:
        return np.arange(num_available)
    return np.linspace(0, num_available - 1, num_frames).round().astype(int)


def _build_dataset(cfg):
    """Instantiate the ComposedDataset from a per-dataset --configure YAML, at
    the data's native long side x ``inference.image_scale`` (as inference does)."""
    from hydra.utils import instantiate

    dataset = instantiate(cfg.dataset, common_config=cfg.common_config, _recursive_=False)
    native_h, native_w = dataset.native_image_size()
    scale = float(cfg.get("inference", {}).get("image_scale", 1.0))
    dataset.set_img_size(_effective_long_side(max(native_h, native_w), scale))
    return dataset


def _safe_name(seq_name: str) -> str:
    """Sequence name -> filesystem-safe stem (7-Scenes names contain '/')."""
    return seq_name.replace("/", "__").replace(" ", "_")


def main(argv=None) -> int:
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Log VGGT-Omega dataset sequences to Rerun: write .rrd files "
        "(default) or stream live to a viewer over gRPC (--connect / --serve), for "
        "headless 'rerun --serve-web' visualization.",
    )
    parser.add_argument(
        "--configure", required=True,
        help="Per-dataset configure YAML (e.g. vggt_omega/datasets/config/tum.yaml).",
    )
    parser.add_argument("--out", default="rerun_out", help="Output dir for .rrd files (file mode).")
    parser.add_argument(
        "--seq-index", type=int, default=None,
        help="Visualize only this sequence index; default = all sequences.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Frames per sequence (<=0 = all, evenly spaced). Default: the "
        "configure's inference.num_frames. Lower it for large sequences.",
    )
    parser.add_argument(
        "--point-stride", type=int, default=8,
        help="Subsample factor for the world point cloud (higher = lighter output).",
    )
    parser.add_argument(
        "--accumulate", action="store_true",
        help="Accumulate every frame's cloud instead of replacing per frame.",
    )
    # Live sink modes (mutually exclusive). When neither is given, write .rrd files.
    sink = parser.add_mutually_exclusive_group()
    sink.add_argument(
        "--connect", nargs="?", const=DEFAULT_GRPC_URL, default=None, metavar="URL",
        help="Stream live over gRPC into an already-running 'rerun --serve-web' instead "
        f"of writing files. Bare --connect uses {DEFAULT_GRPC_URL}; pass a "
        "rerun+http://host:port/proxy URL to override. NOTE: --serve-web forwards "
        "live, so open the web viewer BEFORE streaming; use --serve to view later.",
    )
    sink.add_argument(
        "--serve", action="store_true",
        help="Host an in-process gRPC server and stream into it (no files); print the "
        "URI to point a viewer at, then block until Ctrl-C.",
    )
    parser.add_argument(
        "--grpc-port", type=int, default=None,
        help="gRPC port for --serve (default: rerun's 9876).",
    )
    args = parser.parse_args(argv)

    rr = _require_rerun()  # fail fast with the install hint if rerun is missing
    cfg = OmegaConf.load(args.configure)
    dataset = _build_dataset(cfg)

    default_nf = -1
    if "inference" in cfg and "num_frames" in cfg.inference:
        default_nf = int(cfg.inference.num_frames)
    num_frames = args.num_frames if args.num_frames is not None else default_nf

    num_seqs = dataset.num_sequences()
    if args.seq_index is not None and not (0 <= args.seq_index < num_seqs):
        parser.error(f"--seq-index {args.seq_index} out of range [0, {num_seqs - 1}]")
    indices = [args.seq_index] if args.seq_index is not None else list(range(num_seqs))

    live = args.connect is not None or args.serve
    dest = (f"gRPC {args.connect}" if args.connect is not None
            else "in-process gRPC server" if args.serve else f"{args.out}/")
    logger.info(
        "%d sequence(s) @ %s long side; num_frames=%s -> %s",
        len(indices), dataset.img_size, "all" if num_frames <= 0 else num_frames, dest,
    )
    if not live and (num_frames <= 0 or num_frames > 200):
        logger.warning(
            "logging %s frames/sequence at native-ish resolution writes large .rrd "
            "files (all rasters + point cloud per frame); lower --num-frames or raise "
            "--point-stride if the output is too big.",
            "all" if num_frames <= 0 else num_frames,
        )

    def _frame_plan(seq_index):
        """Per-sequence (name, ordered frame ids, native aspect ratio) -- the eval
        sampling plan shared by both sinks."""
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = _resolve_frame_ids(dataset, seq_index, num_frames)
        native_h, native_w = dataset.native_image_size(seq_index)
        aspect_ratio = min(native_h, native_w) / max(native_h, native_w)
        return seq_name, frame_ids, aspect_ratio

    # --- file mode (default): one .rrd per sequence --------------------------
    # Routed through RerunDataset (the dataset wrapper): get_sample() logs+saves.
    if not live:
        viz = RerunDataset(dataset, out_dir=args.out,
                           point_stride=args.point_stride, accumulate=args.accumulate)
        for seq_index in indices:
            seq_name, frame_ids, aspect_ratio = _frame_plan(seq_index)
            before = len(viz.saved_paths)
            viz.get_sample(seq_index, ids=frame_ids, aspect_ratio=aspect_ratio)
            if len(viz.saved_paths) == before:
                # _log swallowed a failure (no .rrd written) -- report and move on
                # rather than read a stale/absent saved_paths[-1].
                logger.warning("[%s] logging failed; skipped (see warning above)", seq_name)
                continue
            path = viz.saved_paths[-1]
            size_mb = os.path.getsize(path) / 1e6
            logger.info("[%s] %d frames -> %s (%.1f MB)", seq_name, len(frame_ids), path, size_mb)
        if viz.saved_paths:
            print(f"\nWrote {len(viz.saved_paths)} recording(s) to {args.out}/")
            print("View in a browser (headless) with:")
            print(f"    rerun --serve-web {' '.join(viz.saved_paths)}")
        return 0

    # --- live mode (--connect / --serve): per-sequence recording into a gRPC sink ---
    # Each sequence is its own RecordingStream (distinct application_id), so the
    # viewer lists them separately -- same model as one-.rrd-per-sequence. For
    # --serve the first stream starts the in-process server and the rest connect
    # to it; all streams are kept alive (the server dies when its stream is GC'd).
    recs = []
    server_uri = None
    for seq_index in indices:
        seq_name, frame_ids, aspect_ratio = _frame_plan(seq_index)
        sample = dataset.get_sample(seq_index, ids=frame_ids, aspect_ratio=aspect_ratio)
        rec = rr.RecordingStream(application_id=seq_name)
        if args.serve:
            if server_uri is None:
                server_uri = rec.serve_grpc(grpc_port=args.grpc_port)
            else:
                rec.connect_grpc(server_uri)
        else:
            rec.connect_grpc(args.connect)
        log_sample(sample, recording=rec,
                   point_stride=args.point_stride, accumulate=args.accumulate)
        rec.flush()  # hand all data to the sink before moving to the next sequence
        recs.append(rec)  # keep the stream (and, for --serve, its server) alive
        logger.info("[%s] %d frames -> streamed", seq_name, len(frame_ids))

    if args.serve:
        print(f"\nServing {len(recs)} recording(s) over gRPC at:\n    {server_uri}")
        print("View it:")
        print(f"    rerun {server_uri}    # native viewer (needs a display)")
        print("    # headless browser: run 'rerun --serve-web' separately, then re-run")
        print("    #   this with --connect instead, and open http://<host>:9090")
        print("Press Ctrl-C to stop the server.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nstopped.")
        return 0

    print(f"\nStreamed {len(recs)} recording(s) to {args.connect}")
    print("Watch it in the web viewer of the running 'rerun --serve-web' (http://<host>:9090).")
    print("If nothing shows, the viewer wasn't connected during streaming -- open it first,")
    print("or use --serve (buffers in-process) to attach a viewer afterwards.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
