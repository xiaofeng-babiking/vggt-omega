"""Tests for the dataset-to-Rerun adapter.

Pure-logic tests (normalization shim, extrinsic inversion, helpers, view
selection) run WITHOUT rerun installed. Smoke/robustness tests that actually
emit log calls are guarded by ``pytest.importorskip("rerun")`` and write to a
temp ``.rrd`` (no live viewer). The TUM integration test is additionally guarded
by the dataset being present on disk, mirroring ``test_tum_dataset.py``.
"""
import os
import sys
import types

import numpy as np
import pytest

from vggt_omega.datasets.adapters import rerun_adapter
from vggt_omega.datasets.modality import Modality

TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
HAVE_TUM = os.path.isdir(TUM_DIR)


# --- Synthetic sample factories (cover several modalities, both input forms) --


def _make_raw_sample(V=2, H=8, W=8):
    """A raw get_data()-style numpy dict (per-frame lists, (H,W,3) uint8 images)."""
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(V)]
    depths = [rng.random((H, W)).astype(np.float32) + 0.1 for _ in range(V)]
    extr = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5.0]], dtype=np.float32) for _ in range(V)]
    intr = [np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float32) for _ in range(V)]
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
        "timestamps": np.array([10.0, 10.5][:V] + [11.0] * max(0, V - 2), dtype=np.float64),
        "texts": [f"frame {i}" for i in range(V)],
        "modalities": frozenset(
            {
                Modality.IMAGE, Modality.DEPTH, Modality.EXTRINSICS, Modality.INTRINSICS,
                Modality.WORLD_POINTS, Modality.POINT_MASK, Modality.TIMESTAMP, Modality.TEXT,
            }
        ),
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


# --- Lazy import guard --------------------------------------------------------


def test_require_rerun_raises_helpful_error_when_missing(monkeypatch):
    # Force `import rerun` to fail even if the package is installed.
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=r"vggt-omega\[viz\]"):
        rerun_adapter._require_rerun()


# --- _canonical_images --------------------------------------------------------


def test_canonical_images_from_chw_float():
    arr = np.zeros((2, 3, 4, 5), dtype=np.float32)
    arr[:, 0] = 1.0  # full red
    out = rerun_adapter._canonical_images(arr)
    assert out.shape == (2, 4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0].tolist() == [255, 0, 0]


def test_canonical_images_from_hwc_uint8():
    arr = np.full((2, 4, 5, 3), 200, dtype=np.uint8)
    out = rerun_adapter._canonical_images(arr)
    assert out.shape == (2, 4, 5, 3) and out.dtype == np.uint8
    assert out[0, 0, 0].tolist() == [200, 200, 200]


def test_canonical_images_rejects_non_4d():
    with pytest.raises(ValueError, match="4D"):
        rerun_adapter._canonical_images(np.zeros((4, 5, 3), dtype=np.uint8))


# --- normalize_sample ---------------------------------------------------------


def test_normalize_sample_present_and_V_from_raw():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    assert norm.V == 2
    assert {"images", "depths", "world_points"} <= norm.present
    assert "seq_name" not in norm.present  # non-modality keys excluded


def test_normalize_sample_canonical_shapes_match_across_forms():
    raw = rerun_adapter.normalize_sample(_make_raw_sample())
    comp = rerun_adapter.normalize_sample(_make_composed_sample())
    assert raw.present == comp.present
    for key in ("images", "depths", "extrinsics", "intrinsics", "world_points", "point_masks"):
        assert raw.data[key].shape == comp.data[key].shape, key
    assert raw.data["images"].shape == (2, 8, 8, 3)
    assert raw.data["images"].dtype == np.uint8 and comp.data["images"].dtype == np.uint8
    assert comp.data["texts"] == ["frame 0", "frame 1"]


def test_normalize_sample_is_driven_by_dict_not_modalities_field():
    # Mimic the real vendors (TUM/7-Scenes): world_points IS in the sample dict
    # but is NOT declared in `modalities` (it is derived geometry, not scorable
    # GT). A viz tool must still render it, so `present` is dict-driven.
    raw = _make_raw_sample()
    raw["modalities"] = frozenset({Modality.IMAGE, Modality.DEPTH})  # narrower than the dict
    norm = rerun_adapter.normalize_sample(raw)
    assert "world_points" in norm.present  # present despite not being declared


def test_normalize_sample_without_modalities_field():
    raw = _make_raw_sample()
    del raw["modalities"]
    norm = rerun_adapter.normalize_sample(raw)
    assert {"images", "depths"} <= norm.present


# --- _extrinsic_to_cam_to_world ----------------------------------------------


def test_extrinsic_inversion_round_trip():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)  # 90deg about Z
    t = np.array([1.0, 2.0, 3.0])
    ext = np.concatenate([R, t[:, None]], axis=1)  # (3,4) world->cam
    R_cw, t_cw = rerun_adapter._extrinsic_to_cam_to_world(ext)
    x_cam = np.array([0.5, -0.7, 2.0])
    x_world = R_cw @ x_cam + t_cw
    x_cam2 = R @ x_world + t
    np.testing.assert_allclose(x_cam2, x_cam, atol=1e-5)


def test_extrinsic_camera_center_for_identity_rotation():
    ext = np.concatenate([np.eye(3), np.array([[0.0], [0.0], [5.0]])], axis=1)
    _, t_cw = rerun_adapter._extrinsic_to_cam_to_world(ext)
    np.testing.assert_allclose(t_cw, [0.0, 0.0, -5.0], atol=1e-6)


# --- timeline + small helpers -------------------------------------------------


def test_relative_times_offsets_to_zero():
    norm = rerun_adapter.normalize_sample(_make_raw_sample(V=2))
    rel = rerun_adapter._relative_times(norm)
    assert rel is not None
    assert rel[0] == 0.0
    np.testing.assert_allclose(rel[1], 0.5, atol=1e-9)


def test_relative_times_none_without_timestamps():
    raw = _make_raw_sample()
    del raw["timestamps"]
    norm = rerun_adapter.normalize_sample(raw)
    assert rerun_adapter._relative_times(norm) is None


def test_relative_times_all_nan_returns_none():
    raw = _make_raw_sample(V=2)
    raw["timestamps"] = np.array([np.nan, np.nan], dtype=np.float64)
    norm = rerun_adapter.normalize_sample(raw)
    assert rerun_adapter._relative_times(norm) is None


def test_relative_times_partial_nan_offsets_by_finite_min():
    raw = _make_raw_sample(V=2)
    raw["timestamps"] = np.array([np.nan, 10.5], dtype=np.float64)
    norm = rerun_adapter.normalize_sample(raw)
    rel = rerun_adapter._relative_times(norm)
    assert rel is not None
    assert np.isnan(rel[0]) and rel[1] == 0.0  # offset by the finite min (10.5)


def test_relative_times_length_mismatch_falls_back_to_none():
    raw = _make_raw_sample(V=2)
    raw["timestamps"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)  # size 4 != V=2
    norm = rerun_adapter.normalize_sample(raw)
    assert rerun_adapter._relative_times(norm) is None


def test_frame_hw_skips_low_rank_arrays():
    # A degenerate 2-D modality must not raise (IndexError on shape[2]); returns None.
    bad = rerun_adapter.NormalizedSample(data={"point_masks": np.zeros((2, 8))}, present={"point_masks"}, V=2)
    assert rerun_adapter._frame_hw(bad) is None
    good = rerun_adapter.NormalizedSample(data={"depths": np.zeros((2, 8, 8))}, present={"depths"}, V=2)
    assert rerun_adapter._frame_hw(good) == (8, 8)


def test_canonical_images_scales_slight_overshoot():
    # Float image overshooting 1.0 (e.g. from blending) is still scaled, not clipped to black.
    arr = np.full((1, 2, 2, 3), 1.05, dtype=np.float32)
    out = rerun_adapter._canonical_images(arr)
    assert out.max() == 255  # scaled by 255 then clipped, not floored to 1


def test_views_only_reference_known_modality_keys():
    for v in rerun_adapter.VIEWS:
        assert (v.requires | v.optional) <= rerun_adapter._MODALITY_KEYS, v.name


def test_camera_path_uses_camera_id_when_present():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    assert rerun_adapter._camera_path(norm, 0) == "world/camera"
    norm.data["camera_ids"] = np.array([7, 7])
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
    assert colors.shape == (5, 3) and colors.dtype == np.uint8


# --- select_views -------------------------------------------------------------


def test_select_views_full_coverage():
    norm = rerun_adapter.normalize_sample(_make_raw_sample())
    names = {v.name for v in rerun_adapter.select_views(norm.present)}
    assert {"camera", "rgb", "world", "depth", "text"} <= names


def test_select_views_skips_views_with_missing_requirements():
    assert {v.name for v in rerun_adapter.select_views({"images"})} == {"rgb"}


def test_select_views_camera_needs_both_intrinsics_and_extrinsics():
    assert {v.name for v in rerun_adapter.select_views({"intrinsics"})} == set()
    assert {v.name for v in rerun_adapter.select_views({"intrinsics", "extrinsics"})} == {"camera"}


# --- rerun-guarded: log into a recording, save .rrd ---------------------------


def _log_to_rrd(sample, tmp_path, name, **kw):
    """Log a sample into a fresh recording and save -> return the .rrd size."""
    path = rerun_adapter.sample_to_rrd(sample, str(tmp_path / f"{name}.rrd"), app_id=name, **kw)
    assert os.path.isfile(path)
    return os.path.getsize(path)


def test_log_sample_smoke_raw_and_composed(tmp_path):
    pytest.importorskip("rerun")
    assert _log_to_rrd(_make_raw_sample(), tmp_path, "raw") > 0
    assert _log_to_rrd(_make_composed_sample(), tmp_path, "composed") > 0


def test_log_sample_accumulate_mode(tmp_path):
    pytest.importorskip("rerun")
    assert _log_to_rrd(_make_raw_sample(V=3), tmp_path, "accum", accumulate=True) > 0


def test_log_sample_is_robust_to_bad_fields(tmp_path):
    pytest.importorskip("rerun")
    sample = _make_raw_sample()
    sample["extrinsics"][0][:] = np.nan                                  # poison a pose
    sample["point_masks"] = [np.zeros((8, 8), bool) for _ in range(2)]   # empty cloud
    # Per-view isolation: warns and continues, never raises; still writes a file.
    assert _log_to_rrd(sample, tmp_path, "robust") > 0


def test_log_sample_survives_nan_timestamp(tmp_path):
    pytest.importorskip("rerun")
    sample = _make_raw_sample(V=3)
    sample["timestamps"] = np.array([10.0, np.nan, 11.0], dtype=np.float64)  # poison one stamp
    # Must not raise: the unguarded set_time() skips the NaN frame's time stamp.
    assert _log_to_rrd(sample, tmp_path, "nan_ts") > 0


def test_log_sample_empty_is_noop(tmp_path):
    rr = pytest.importorskip("rerun")
    rec = rr.RecordingStream(application_id="empty")
    rerun_adapter.log_sample({"modalities": frozenset()}, recording=rec)  # V == 0, no raise


def test_partial_sample_images_only(tmp_path):
    rr = pytest.importorskip("rerun")
    raw = _make_raw_sample()
    sample = {"images": raw["images"]}  # only RGB -> only the rgb view runs
    assert _log_to_rrd(sample, tmp_path, "imgonly") > 0


# --- RerunDataset wrapper -----------------------------------------------------
# A minimal map-style dataset mimicking ComposedDataset's surface (indexing with
# either an int or the training sampler's (seq, n, ar) tuple, plus get_sample /
# num_sequences / sequence_name), so we can test the wrapper without real data.


class _FakeDataset:
    def __init__(self, n=3, V=2):
        self._samples = []
        for k in range(n):
            s = _make_raw_sample(V=V)
            s["seq_name"] = f"seq_{k}"
            self._samples.append(s)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        i = idx[0] if isinstance(idx, tuple) else idx
        return self._samples[i]  # IndexError past the end -> stops for-iteration

    def get_sample(self, seq_index, ids, aspect_ratio=1.0):
        return self._samples[seq_index]

    def num_sequences(self):
        return len(self._samples)

    def sequence_name(self, i):
        return self._samples[i]["seq_name"]


def test_rerun_dataset_rejects_both_sinks():
    with pytest.raises(ValueError, match="at most one"):
        rerun_adapter.RerunDataset(_FakeDataset(), recording=object(), out_dir="x")


def test_rerun_dataset_passes_through_and_delegates(monkeypatch, tmp_path):
    # Stub the per-sample logger so this runs without rerun installed.
    calls = []

    def _fake_rrd(sample, path, **kw):
        calls.append((sample, path, kw))
        open(path, "w").close()  # the wrapper records saved_paths; make it real
        return path

    monkeypatch.setattr(rerun_adapter, "sample_to_rrd", _fake_rrd)
    ds = _FakeDataset(n=3)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path), point_stride=8)

    assert len(viz) == 3                       # __len__ delegates
    out = viz[1]
    assert out is ds._samples[1]               # sample passes through UNCHANGED
    assert viz.num_sequences() == 3            # __getattr__ delegation
    assert viz.sequence_name(2) == "seq_2"     # __getattr__ delegation

    assert len(calls) == 1
    sample, path, kw = calls[0]
    assert sample is ds._samples[1]
    assert path == os.path.join(str(tmp_path), "seq_1.rrd")
    assert kw["point_stride"] == 8             # forwarded
    assert kw["app_id"] == "seq_1"             # file mode names the recording per seq
    assert viz.saved_paths == [path]


def test_rerun_dataset_get_sample_logs_and_passes_through(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        rerun_adapter, "sample_to_rrd",
        lambda sample, path, **kw: (calls.append(path), open(path, "w").close(), path)[-1],
    )
    ds = _FakeDataset(n=3)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path))
    out = viz.get_sample(2, ids=[0, 1], aspect_ratio=0.75)
    assert out is ds._samples[2]               # eval path also passes through
    assert calls == [os.path.join(str(tmp_path), "seq_2.rrd")]


def test_rerun_dataset_shared_mode_advances_sample_timeline(monkeypatch):
    # No real rerun: fake the module + the per-sample logger, assert the wrapper
    # sets an outer `sample` timeline that advances 0, 1, ... per fetched sample.
    set_time_calls = []
    fake_rr = types.SimpleNamespace(
        set_time=lambda *a, **k: set_time_calls.append((a, k))
    )
    logged = []
    monkeypatch.setattr(rerun_adapter, "_require_rerun", lambda: fake_rr)
    monkeypatch.setattr(rerun_adapter, "log_sample", lambda s, **kw: logged.append(s))

    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds)       # current-recording (shared) mode
    a, b = viz[0], viz[1]
    assert a is ds._samples[0] and b is ds._samples[1]
    assert logged == [ds._samples[0], ds._samples[1]]
    sample_seqs = [k["sequence"] for (args, k) in set_time_calls if args and args[0] == "sample"]
    assert sample_seqs == [0, 1]


def test_rerun_dataset_log_failure_is_soft(monkeypatch, tmp_path):
    # A logging error on one sample must NOT break the pipeline: warn + continue,
    # still return the sample, still advance the counter.
    def _boom(*a, **k):
        raise RuntimeError("bad field")

    monkeypatch.setattr(rerun_adapter, "sample_to_rrd", _boom)
    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path))
    out = viz[0]                                # does not raise
    assert out is ds._samples[0]
    assert viz._logged == 1
    assert viz.saved_paths == []               # nothing recorded for the failure


def test_rerun_dataset_missing_rerun_is_surfaced(monkeypatch):
    # A missing Rerun install is a setup error -> surfaced, not swallowed.
    def _no_rerun():
        raise ImportError("vggt-omega[viz]")

    monkeypatch.setattr(rerun_adapter, "_require_rerun", _no_rerun)
    viz = rerun_adapter.RerunDataset(_FakeDataset(n=1))  # shared mode
    with pytest.raises(ImportError, match=r"viz"):
        viz[0]


def test_rerun_dataset_smoke_writes_one_rrd_per_sample(tmp_path):
    pytest.importorskip("rerun")
    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path), point_stride=8)
    for _ in viz:  # for-iteration via __getitem__ until IndexError
        pass
    assert len(viz.saved_paths) == 2
    for path in viz.saved_paths:
        assert os.path.getsize(path) > 0


def test_rerun_dataset_smoke_shared_recording(tmp_path):
    rr = pytest.importorskip("rerun")
    rec = rr.RecordingStream(application_id="shared")
    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds, recording=rec, point_stride=8)
    viz[0]
    viz[1]
    out = str(tmp_path / "shared.rrd")
    rec.save(out)
    assert os.path.getsize(out) > 0
    assert viz._logged == 2


def test_rerun_dataset_forwards_tuple_index_verbatim(monkeypatch, tmp_path):
    # The training sampler (DynamicBatchSampler) indexes with a
    # (seq_idx, num_images, aspect_ratio) TUPLE -- the wrapper must forward it
    # verbatim to the inner dataset and still log + pass the sample through.
    seen = []

    class _TupleSpy(_FakeDataset):
        def __getitem__(self, idx):
            seen.append(idx)
            return super().__getitem__(idx)

    monkeypatch.setattr(rerun_adapter, "sample_to_rrd",
                        lambda s, p, **k: (open(p, "w").close(), p)[-1])
    ds = _TupleSpy(n=2)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path))
    out = viz[(1, 4, 0.75)]
    assert seen == [(1, 4, 0.75)]              # exact tuple reached the dataset
    assert out is ds._samples[1]               # sample passes through unchanged


def test_rerun_dataset_get_sample_shared_mode_advances_timeline(monkeypatch):
    # get_sample (the eval path) must also drive the outer `sample` timeline in
    # shared-recording mode, not just __getitem__.
    set_time_calls = []
    fake_rr = types.SimpleNamespace(set_time=lambda *a, **k: set_time_calls.append((a, k)))
    logged = []
    monkeypatch.setattr(rerun_adapter, "_require_rerun", lambda: fake_rr)
    monkeypatch.setattr(rerun_adapter, "log_sample", lambda s, **kw: logged.append(s))

    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds)       # shared (current-recording) mode
    a = viz.get_sample(0, ids=[0, 1])
    b = viz.get_sample(1, ids=[0, 1])
    assert a is ds._samples[0] and b is ds._samples[1]
    assert logged == [ds._samples[0], ds._samples[1]]
    sample_seqs = [k["sequence"] for (args, k) in set_time_calls if args and args[0] == "sample"]
    assert sample_seqs == [0, 1]
    assert viz._logged == 2


def test_rerun_dataset_file_mode_saved_paths_atomicity(monkeypatch, tmp_path):
    # A mid-stream logging failure must not pollute saved_paths or stall the
    # counter: saved_paths skips the failed sample, _logged counts every fetch.
    def _rrd(sample, path, **kw):
        if sample["seq_name"] == "seq_1":
            raise RuntimeError("boom on the middle sample")
        open(path, "w").close()
        return path

    monkeypatch.setattr(rerun_adapter, "sample_to_rrd", _rrd)
    ds = _FakeDataset(n=3)
    viz = rerun_adapter.RerunDataset(ds, out_dir=str(tmp_path))
    for _ in viz:
        pass
    assert viz.saved_paths == [
        os.path.join(str(tmp_path), "seq_0.rrd"),
        os.path.join(str(tmp_path), "seq_2.rrd"),
    ]                                          # the failed seq_1 left no entry
    assert viz._logged == 3                    # every fetch still counted


def test_rerun_dataset_shared_mode_counter_advances_on_failure(monkeypatch):
    # In shared mode a non-ImportError logging failure is swallowed and the
    # counter still advances (so timelines stay monotonic).
    fake_rr = types.SimpleNamespace(set_time=lambda *a, **k: None)
    monkeypatch.setattr(rerun_adapter, "_require_rerun", lambda: fake_rr)

    def _boom(*a, **k):
        raise RuntimeError("bad field")

    monkeypatch.setattr(rerun_adapter, "log_sample", _boom)
    ds = _FakeDataset(n=1)
    viz = rerun_adapter.RerunDataset(ds)
    out = viz[0]                                # does not raise
    assert out is ds._samples[0]
    assert viz._logged == 1


def test_rerun_dataset_file_mode_name_fallback(monkeypatch, tmp_path):
    # A sample with no `seq_name` falls back to a counter-based filename.
    paths = []
    monkeypatch.setattr(rerun_adapter, "sample_to_rrd",
                        lambda s, p, **k: (paths.append(p), open(p, "w").close(), p)[-1])

    class _NoName:
        def __len__(self): return 2
        def __getitem__(self, i): return {"images": _make_raw_sample()["images"]}

    viz = rerun_adapter.RerunDataset(_NoName(), out_dir=str(tmp_path))
    viz[0]
    viz[1]
    assert paths == [
        os.path.join(str(tmp_path), "sample_0.rrd"),
        os.path.join(str(tmp_path), "sample_1.rrd"),
    ]


def test_rerun_dataset_getattr_delegates_and_guards():
    ds = _FakeDataset(n=2)
    viz = rerun_adapter.RerunDataset(ds, out_dir="x")
    assert viz.dataset is ds                    # stored attribute, not delegated
    assert viz.num_sequences() == 2             # missing-on-wrapper -> delegated
    with pytest.raises(AttributeError):         # unknown on both -> raised, no recursion
        viz.does_not_exist_anywhere


def test_rerun_dataset_repr_covers_all_modes():
    ds = _FakeDataset(n=1)
    assert "out_dir=" in repr(rerun_adapter.RerunDataset(ds, out_dir="x"))
    assert "recording=<shared>" in repr(rerun_adapter.RerunDataset(ds, recording=object()))
    assert "recording=<current>" in repr(rerun_adapter.RerunDataset(ds))


# --- TUM integration (requires the dataset on disk) ---------------------------


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_log_sample_on_real_tum_sequence(tmp_path):
    rr = pytest.importorskip("rerun")
    pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from vggt_omega.datasets.vendors.tum import TumDataset

    common = OmegaConf.create(
        {
            "img_size": 256, "patch_size": 16, "training": False, "inside_random": False,
            "allow_duplicate_img": False, "get_nearby": False, "rescale": True,
            "rescale_aug": False, "landscape_check": False, "augs": {"scales": None},
        }
    )
    ds = TumDataset(
        common_conf=common, split="train", TUM_DIR=TUM_DIR,
        sequences=["rgbd_dataset_freiburg3_sitting_halfsphere"], len_train=10,
    )
    sample = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=0.75)  # raw numpy dict
    path = rerun_adapter.sample_to_rrd(
        sample, str(tmp_path / "tum.rrd"), app_id="tum_seq", point_stride=8
    )
    assert os.path.getsize(path) > 0
