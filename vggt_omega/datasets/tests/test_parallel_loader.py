"""Tests for the chunked-parallel inference loading path.

Units run against a deterministic in-memory FakeVendor (no data needed);
the TUM integration test verifies serial vs parallel `get_sample` equality
on real frames and is skipped when the data dir is absent.
"""
import os

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.modality import Modality
from vggt_omega.datasets.parallel_loader import (
    merge_chunk_batches,
    parallel_get_data,
    resolve_num_workers,
    split_ids,
)

TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
HAVE_TUM = os.path.isdir(TUM_DIR)


def _eval_common():
    """Deterministic eval-mode common_config (mirrors inference.py configs)."""
    return OmegaConf.create(
        {
            "img_size": 32,
            "patch_size": 16,
            "training": False,
            "inside_random": False,
            "allow_duplicate_img": False,
            "get_nearby": False,
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "fix_img_num": -1,
            "fix_aspect_ratio": 1.0,
            "load_track": False,
            "track_num": 16,
            "load_depth": True,
            "debug": False,
            "repeat_batch": False,
            "img_nums": [2, 6],
            "max_img_per_gpu": 12,
            "augs": {
                "scales": None,
                "aspects": [1.0, 1.0],
                "cojitter": False,
                "cojitter_ratio": 0.3,
                "color_jitter": None,
                "gray_scale": False,
                "gau_blur": False,
            },
        }
    )


class FakeVendor(BaseDataset):
    """Deterministic synthetic vendor following the standard get_data contract.

    Every per-frame array is a pure function of the frame id, so chunked
    results can be compared bit-for-bit against a single serial call.
    """

    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
            Modality.WORLD_POINTS,
            Modality.CAM_POINTS,
            Modality.SKY_MASK,
            Modality.TIMESTAMP,
        }
    )

    def __init__(self, common_conf, num_frames=64, h=32, w=32, fail_on_ids=()):
        super().__init__(common_conf=common_conf)
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.get_nearby = common_conf.get_nearby
        self.num_frames = num_frames
        self.h, self.w = h, w
        self.sequence_list = ["seq0"]
        self.sequence_list_len = 1
        self.len_train = num_frames
        self.available_modalities = self.AVAILABLE
        self.fail_on_ids = set(fail_on_ids)
        self.call_log = []  # one entry per get_data call (list.append is GIL-atomic)

    def sequence_num_frames(self, local_idx):
        return self.num_frames

    def native_image_size(self, local_idx=0):
        return (self.h, self.w)

    def _frame(self, i):
        i = int(i)
        h, w = self.h, self.w
        grid = np.arange(h * w * 3, dtype=np.int64).reshape(h, w, 3)
        image = ((grid * (i + 3)) % 256).astype(np.uint8)
        depth = np.full((h, w), float(i + 1), dtype=np.float32)
        extri = np.hstack([np.eye(3), np.full((3, 1), float(i))]).astype(np.float32)
        intri = np.array(
            [[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        world = np.full((h, w, 3), float(i), dtype=np.float32)
        cam = np.full((h, w, 3), float(i) + 0.5, dtype=np.float32)
        pmask = (grid[..., 0] % 2 == i % 2)
        return image, depth, extri, intri, world, cam, pmask

    def get_data(
        self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0
    ):
        # Record full call kwargs so chunk fan-out forwarding bugs (wrong
        # seq_name / dropped aspect_ratio) are visible to the unit tests.
        self.call_log.append(
            {
                "seq_name": seq_name,
                "ids": np.asarray(ids).tolist(),
                "aspect_ratio": aspect_ratio,
            }
        )
        out = {
            "images": [],
            "depths": [],
            "extrinsics": [],
            "intrinsics": [],
            "cam_points": [],
            "world_points": [],
            "point_masks": [],
            "sky_masks": [],
            "original_sizes": [],
        }
        timestamps = []
        for i in ids:
            if int(i) in self.fail_on_ids:
                raise RuntimeError(f"injected failure on frame {int(i)}")
            image, depth, extri, intri, world, cam, pmask = self._frame(i)
            out["images"].append(image)
            out["depths"].append(depth)
            out["extrinsics"].append(extri)
            out["intrinsics"].append(intri)
            out["world_points"].append(world)
            out["cam_points"].append(cam)
            out["point_masks"].append(pmask)
            out["sky_masks"].append(depth < 0)
            out["original_sizes"].append(np.array([self.h, self.w]))
            timestamps.append(float(i) / 30.0)
        out.update(
            {
                "seq_name": "fake_seq0",
                "ids": np.array(ids),
                "frame_num": len(out["images"]),
                "timestamps": np.array(timestamps, dtype=np.float64),
                "is_metric": True,
                "is_video": True,
                "modalities": set(self.available_modalities),
            }
        )
        return out


def _composed(common=None, **vendor_kwargs):
    cfg = {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.tests.test_parallel_loader.FakeVendor",
                **vendor_kwargs,
            }
        ],
    }
    return instantiate(cfg, common_config=common or _eval_common(), _recursive_=False)


# --- split_ids ---------------------------------------------------------------


def test_split_ids_covers_ids_in_order_with_warmup_first():
    ids = np.arange(100, 0, -1)  # non-trivial order must be preserved verbatim
    chunks = split_ids(ids, num_workers=8)
    assert all(len(c) >= 1 for c in chunks)
    np.testing.assert_array_equal(chunks[0], ids[:1])  # serial warm-up chunk
    np.testing.assert_array_equal(np.concatenate(chunks), ids)


def test_split_ids_more_workers_than_ids():
    ids = np.array([5, 4, 3, 2])
    chunks = split_ids(ids, num_workers=64)
    assert all(len(c) >= 1 for c in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), ids)


# --- merge_chunk_batches -----------------------------------------------------


def _mini_batch(ids):
    return {
        "seq_name": "s",
        "ids": np.array(ids),
        "frame_num": len(ids),
        "images": [np.full((4, 4, 3), i, dtype=np.uint8) for i in ids],
        "texts": [f"frame{i}" for i in ids],
        "ragged": [np.zeros((int(i) + 1,), dtype=np.float32) for i in ids],
        "timestamps": np.array([float(i) for i in ids]),
        "tracks": None,
        "is_metric": True,
        "modalities": {Modality.IMAGE},
    }


def test_merge_stacks_lists_concats_arrays_and_sums_frame_num():
    merged = merge_chunk_batches([_mini_batch([3, 1]), _mini_batch([2]), _mini_batch([7, 9])])
    np.testing.assert_array_equal(merged["ids"], [3, 1, 2, 7, 9])
    assert merged["frame_num"] == 5
    # same-shape ndarray lists come back PRE-STACKED in chunk order
    assert isinstance(merged["images"], np.ndarray)
    assert merged["images"].shape == (5, 4, 4, 3)
    assert merged["images"].dtype == np.uint8
    np.testing.assert_array_equal(merged["images"][:, 0, 0, 0], [3, 1, 2, 7, 9])
    # string lists stay lists, concatenated in order
    assert merged["texts"] == ["frame3", "frame1", "frame2", "frame7", "frame9"]
    # ragged ndarray lists cannot stack: stay a concatenated list
    assert isinstance(merged["ragged"], list)
    assert [len(r) for r in merged["ragged"]] == [4, 2, 3, 8, 10]
    np.testing.assert_array_equal(merged["timestamps"], [3.0, 1.0, 2.0, 7.0, 9.0])
    # scalars / sets / None: take-first
    assert merged["seq_name"] == "s"
    assert merged["is_metric"] is True
    assert merged["tracks"] is None
    assert merged["modalities"] == {Modality.IMAGE}


def test_merge_single_chunk_is_identity_modulo_stacking():
    merged = merge_chunk_batches([_mini_batch([0, 1, 2])])
    assert merged["frame_num"] == 3
    np.testing.assert_array_equal(merged["ids"], [0, 1, 2])
    assert merged["images"].shape == (3, 4, 4, 3)


def test_merge_rejects_inconsistent_scalars():
    a, b = _mini_batch([0]), _mini_batch([1])
    b["seq_name"] = "DIFFERENT"
    with pytest.raises(ValueError, match="seq_name"):
        merge_chunk_batches([a, b])


def test_merge_rejects_mixed_none():
    a, b = _mini_batch([0]), _mini_batch([1])
    b["tracks"] = [np.zeros((2, 2), dtype=np.float32)]
    with pytest.raises(ValueError, match="tracks"):
        merge_chunk_batches([a, b])
    # ... in either chunk order (list first, None later)
    with pytest.raises(ValueError, match="tracks"):
        merge_chunk_batches([b, a])


def test_merge_rejects_mismatched_keys():
    a, b = _mini_batch([0]), _mini_batch([1])
    del b["texts"]
    with pytest.raises(ValueError, match="keys"):
        merge_chunk_batches([a, b])


# --- parallel_get_data -------------------------------------------------------


def test_parallel_matches_serial_bit_for_bit():
    common = _eval_common()
    vendor = FakeVendor(common, num_frames=64)
    ids = np.arange(63, 0, -2)  # 32 frames, reversed order
    serial = vendor.get_data(seq_name="seq0", ids=ids, aspect_ratio=0.75)
    parallel = parallel_get_data(vendor, "seq0", ids, aspect_ratio=0.75, num_workers=8)

    assert len(vendor.call_log) > 2  # actually fanned out
    # The single-frame warm-up runs (serially) before any fanned-out chunk;
    # call_log[0] is this test's own serial baseline call.
    assert vendor.call_log[1]["ids"] == [int(ids[0])]
    # seq_name / aspect_ratio forwarded verbatim to every chunk call.
    assert all(c["seq_name"] == "seq0" for c in vendor.call_log)
    assert all(c["aspect_ratio"] == 0.75 for c in vendor.call_log)
    np.testing.assert_array_equal(parallel["ids"], serial["ids"])
    assert parallel["frame_num"] == serial["frame_num"]
    for key in (
        "images",
        "depths",
        "extrinsics",
        "intrinsics",
        "cam_points",
        "world_points",
        "point_masks",
        "sky_masks",
    ):
        np.testing.assert_array_equal(np.asarray(parallel[key]), np.stack(serial[key]))
    np.testing.assert_array_equal(parallel["timestamps"], serial["timestamps"])
    assert parallel["seq_name"] == serial["seq_name"]
    assert parallel["modalities"] == serial["modalities"]


def test_parallel_serial_fallback_for_one_worker_and_tiny_loads():
    vendor = FakeVendor(_eval_common())
    parallel_get_data(vendor, "seq0", np.arange(16), aspect_ratio=1.0, num_workers=1)
    assert len(vendor.call_log) == 1  # one serial call, no chunking

    vendor.call_log.clear()
    parallel_get_data(vendor, "seq0", np.arange(2), aspect_ratio=1.0, num_workers=8)
    assert len(vendor.call_log) == 1  # too few frames to fan out


@pytest.mark.parametrize("flag", ["training", "get_nearby", "landscape_check", "rescale_aug"])
def test_parallel_serial_fallback_for_every_rng_bearing_flag(flag):
    """All four flags draw from the shared global RNG inside get_data (training
    and get_nearby directly; landscape_check / rescale_aug even with
    training=False), so each one must force the serial fallback."""
    common = _eval_common()
    common[flag] = True
    vendor = FakeVendor(common)
    with pytest.warns(UserWarning, match="serial"):
        parallel_get_data(vendor, "seq0", np.arange(32), aspect_ratio=1.0, num_workers=8)
    assert len(vendor.call_log) == 1


def test_parallel_propagates_chunk_exceptions_and_restores_cv2():
    import cv2

    before = cv2.getNumThreads()
    vendor = FakeVendor(_eval_common(), fail_on_ids=(17,))
    with pytest.raises(RuntimeError, match="frame 17"):
        parallel_get_data(vendor, "seq0", np.arange(32), aspect_ratio=1.0, num_workers=8)
    assert cv2.getNumThreads() == before  # restored on the exception path too


def test_cv2_thread_cap_set_during_fanout_and_restored():
    import cv2

    original = cv2.getNumThreads()
    # Force a multi-threaded baseline so the assertions are meaningful even on
    # single-core runners (where getNumThreads() may already be 1).
    cv2.setNumThreads(8)
    try:
        observed = []

        class ProbeVendor(FakeVendor):
            def get_data(self, **kwargs):
                observed.append(cv2.getNumThreads())
                return super().get_data(**kwargs)

        vendor = ProbeVendor(_eval_common())
        parallel_get_data(vendor, "seq0", np.arange(32), aspect_ratio=1.0, num_workers=8)
        assert cv2.getNumThreads() == 8  # restored after the parallel window
        # inside the fan-out, cv2's internal pool is disabled (0 -> reported as 1)
        assert all(n <= 1 for n in observed)

        observed.clear()
        parallel_get_data(vendor, "seq0", np.arange(2), aspect_ratio=1.0, num_workers=8)
        assert observed == [8]  # serial fallback leaves cv2 threading alone
    finally:
        cv2.setNumThreads(original)


def test_cv2_guard_nests_and_restores_outermost_value():
    import cv2

    from vggt_omega.datasets.parallel_loader import _cv2_single_threaded

    original = cv2.getNumThreads()
    cv2.setNumThreads(8)
    try:
        with _cv2_single_threaded():
            assert cv2.getNumThreads() == 1
            with _cv2_single_threaded():  # inner window must not capture the 1
                assert cv2.getNumThreads() == 1
            assert cv2.getNumThreads() == 1  # still inside the outer window
        assert cv2.getNumThreads() == 8  # outermost value restored
    finally:
        cv2.setNumThreads(original)


def test_resolve_num_workers(monkeypatch):
    import vggt_omega.datasets.parallel_loader as pl

    assert resolve_num_workers(4) == 4
    assert resolve_num_workers(0) == 1  # the --loader_workers "0 = serial" contract
    monkeypatch.setattr(pl.os, "cpu_count", lambda: 64)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    assert resolve_num_workers(None) == 32  # min(32, cores)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert resolve_num_workers(None) == 4  # divided across this node's ranks
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "64")
    assert resolve_num_workers(None) == 1  # never below 1


# --- ComposedDataset.get_sample(num_workers=...) -----------------------------


def test_get_sample_parallel_equals_serial_on_fake_vendor():
    composed_serial, composed_par = _composed(), _composed()
    serial = composed_serial.get_sample(0, ids=np.arange(48), aspect_ratio=1.0, num_workers=1)
    par = composed_par.get_sample(0, ids=np.arange(48), aspect_ratio=1.0, num_workers=8)
    # Guard against comparing a code path to itself: num_workers must actually
    # select serial (one get_data call) vs fan-out (many).
    assert len(composed_serial.base_dataset.datasets[0].call_log) == 1
    assert len(composed_par.base_dataset.datasets[0].call_log) > 2
    assert set(serial.keys()) == set(par.keys())
    for key, val in serial.items():
        if isinstance(val, torch.Tensor):
            assert torch.equal(val, par[key]), key
        else:
            assert val == par[key], key


def test_tensorize_accepts_prestacked_arrays():
    composed = _composed()
    vendor = composed.base_dataset.datasets[0]
    batch = vendor.get_data(seq_name="seq0", ids=np.arange(6), aspect_ratio=1.0)
    from_lists = composed._tensorize({**batch})
    stacked = {
        k: np.stack(v) if isinstance(v, list) and isinstance(v[0], np.ndarray) else v
        for k, v in batch.items()
    }
    from_arrays = composed._tensorize(stacked)
    for key, val in from_lists.items():
        if isinstance(val, torch.Tensor):
            assert torch.equal(val, from_arrays[key]), key


# --- real-data integration ----------------------------------------------------


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_tum_get_sample_parallel_equals_serial():
    cfg = {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.tum.TumDataset",
                "split": "train",
                "TUM_DIR": TUM_DIR,
                "sequences": ["rgbd_dataset_freiburg3_sitting_halfsphere"],
                "len_train": 20,
            }
        ],
    }
    common = _eval_common()
    common.img_size = 64  # keep the test fast; geometry identical either way
    composed = instantiate(cfg, common_config=common, _recursive_=False)
    ids = np.linspace(0, composed.sequence_num_frames(0) - 1, 24).round().astype(int)
    serial = composed.get_sample(0, ids=ids, aspect_ratio=0.75, num_workers=1)
    par = composed.get_sample(0, ids=ids, aspect_ratio=0.75, num_workers=8)
    assert set(serial.keys()) == set(par.keys())
    for key, val in serial.items():
        if isinstance(val, torch.Tensor):
            assert torch.equal(val, par[key]), key
        else:
            assert val == par[key], key
