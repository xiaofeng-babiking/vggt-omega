# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Chunked-parallel frame loading for the explicit-ids inference path.

Every vendor's ``get_data`` loads frames in a serial ``for i in ids:`` loop
(~120 ms/frame: image decode + depth decode + ``process_one_image``), so a
1000-frame sequence costs ~2 minutes before the model ever runs. With the
deterministic eval flags (``training=False``, ``inside_random=False``,
``get_nearby=False``) that loop is per-frame independent and RNG-free, so the
ids can be split into ordered chunks and loaded by concurrent ``get_data``
calls, then merged in chunk order — bit-identical to one serial call.

Threads, not processes: the per-frame outputs are large (~10 MB: image, depth,
cam/world point maps), so process IPC would re-copy gigabytes, while the hot
work (cv2/PIL decode + resize, numpy unprojection) releases the GIL and scales
across threads.

Chunk 0 is the single-frame warm-up ``ids[:1]``, run serially BEFORE the
fan-out: vendors lazily build per-sequence frame lists / annotation caches
inside ``get_data`` (idempotent ``self._cache[seq] = ...`` writes), and warming
once avoids K threads redundantly re-listing the same directory over NFS.

The executor is created per call and never cached on a dataset instance:
``ComposedDataset`` is forked into DataLoader workers during training, and
cached executor threads would not survive the fork.
"""
from __future__ import annotations

import contextlib
import os
import threading
import warnings
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

import cv2
import numpy as np

# Below this many frames the fan-out overhead is not worth it.
_MIN_PARALLEL_FRAMES = 4
# Chunks per worker: small chunks smooth over per-frame latency jitter (NFS).
_CHUNKS_PER_WORKER = 4
# Auto worker cap; beyond this the loaders contend on memory bandwidth / GIL.
_MAX_AUTO_WORKERS = 32


# Refcounted guard for cv2's PROCESS-GLOBAL thread setting: with N loader
# threads each cv2 call would otherwise fan out onto cv2's own 64-thread pool
# (N x 64 logical workers thrash; measured 21s -> 13s on the 1000-frame TUM
# load with the pool disabled). Restored when the last parallel window exits.
_cv2_lock = threading.Lock()
_cv2_depth = 0
_cv2_saved_threads = 1


@contextlib.contextmanager
def _cv2_single_threaded():
    global _cv2_depth, _cv2_saved_threads
    with _cv2_lock:
        if _cv2_depth == 0:
            _cv2_saved_threads = cv2.getNumThreads()
            cv2.setNumThreads(0)
        _cv2_depth += 1
    try:
        yield
    finally:
        with _cv2_lock:
            _cv2_depth -= 1
            if _cv2_depth == 0:
                cv2.setNumThreads(_cv2_saved_threads)


def resolve_num_workers(num_workers=None) -> int:
    """Explicit ``num_workers`` (min 1), or the auto default: ``min(32, cores)``
    divided across the torchrun ranks sharing this node (``LOCAL_WORLD_SIZE``)."""
    if num_workers is not None:
        return max(1, int(num_workers))
    cores = os.cpu_count() or 1
    local_ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    return max(1, min(_MAX_AUTO_WORKERS, cores) // local_ranks)


def split_ids(ids, num_workers):
    """Split ``ids`` into ordered chunks that concatenate back to ``ids`` exactly.

    Chunk 0 is always the single-frame warm-up ``ids[:1]`` (run serially by
    :func:`parallel_get_data` to populate vendors' lazy per-sequence caches).
    """
    ids = np.asarray(ids)
    chunks = [ids[:1]]
    rest = ids[1:]
    if len(rest):
        num_chunks = min(len(rest), num_workers * _CHUNKS_PER_WORKER)
        chunks += list(np.array_split(rest, num_chunks))
    return chunks


def parallel_get_data(vendor, seq_name, ids, aspect_ratio=1.0, num_workers=None) -> dict:
    """Load ``vendor.get_data(seq_name, ids, aspect_ratio)`` with chunked threads.

    Returns a batch dict equivalent to the serial call, except per-frame lists
    of same-shape arrays come back pre-stacked as ``(V, ...)`` ndarrays (which
    ``ComposedDataset._tensorize`` / ``carry_extra_modalities`` accept without
    re-copying). Falls back to one serial call when parallelism is off
    (``num_workers <= 1``), the load is tiny, or the vendor has RNG-bearing
    flags enabled (``training`` / ``get_nearby``), whose shared global RNG
    stream would make chunked results scheduling-dependent.
    """
    ids = np.asarray(ids)
    workers = resolve_num_workers(num_workers)
    if workers <= 1 or len(ids) < _MIN_PARALLEL_FRAMES:
        return vendor.get_data(seq_name=seq_name, ids=ids, aspect_ratio=aspect_ratio)
    rng_flags = [
        flag
        for flag in ("training", "get_nearby", "landscape_check", "rescale_aug")
        if getattr(vendor, flag, False)
    ]
    if rng_flags:
        warnings.warn(
            f"{type(vendor).__name__} has {'/'.join(rng_flags)} enabled; these draw "
            "from the shared global RNG inside get_data, so chunked parallel "
            "loading would be scheduling-dependent. Falling back to serial get_data.",
            stacklevel=2,
        )
        return vendor.get_data(seq_name=seq_name, ids=ids, aspect_ratio=aspect_ratio)

    chunks = split_ids(ids, workers)
    with _cv2_single_threaded():
        # Warm-up chunk runs serially so lazy per-sequence caches are built once.
        batches = [vendor.get_data(seq_name=seq_name, ids=chunks[0], aspect_ratio=aspect_ratio)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    vendor.get_data, seq_name=seq_name, ids=chunk, aspect_ratio=aspect_ratio
                )
                for chunk in chunks[1:]
            ]
            # Fail fast: on the first chunk error, cancel the still-queued
            # chunks instead of loading the rest of the sequence first.
            wait(futures, return_when=FIRST_EXCEPTION)
            failed = next((f for f in futures if f.done() and f.exception()), None)
            if failed is not None:
                for future in futures:
                    future.cancel()
                failed.result()  # re-raises the chunk's exception
            batches += [f.result() for f in futures]
            return merge_chunk_batches(batches, executor=executor)


def merge_chunk_batches(chunk_batches, executor=None) -> dict:
    """Merge per-chunk ``get_data`` dicts (in chunk order) into one batch dict.

    Type-driven rules, so any vendor's key set works:
      - ``frame_num``           -> summed;
      - list of same-shape arrays -> pre-stacked ``(V, ...)`` ndarray (the
        copies run on ``executor`` when given — numpy memcpy releases the GIL);
      - other lists (str, ragged) -> concatenated list;
      - ndarray (``ids``, ``timestamps``) -> concatenated along axis 0;
      - everything else (str/scalar/set/None) -> must be identical across
        chunks; kept as-is. Mismatch raises ValueError naming the key.
    """
    keys = chunk_batches[0].keys()
    for batch in chunk_batches[1:]:
        if batch.keys() != keys:
            raise ValueError(
                f"chunk batches disagree on keys: {sorted(keys)} vs {sorted(batch.keys())}"
            )
    return {key: _merge_key(key, [b[key] for b in chunk_batches], executor) for key in keys}


def _merge_key(key, values, executor):
    first = values[0]
    if key == "frame_num":
        return sum(values)
    if isinstance(first, list):
        if not all(isinstance(v, list) for v in values):
            raise ValueError(
                f"chunk batches disagree on '{key}': mixed list and non-list values"
            )
        return _merge_lists(values, executor)
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    for other in values[1:]:
        if not _equal(first, other):
            raise ValueError(f"chunk batches disagree on '{key}': {first!r} != {other!r}")
    return first


def _equal(a, b):
    if (a is None) != (b is None):
        return False
    return a is None or a == b


def _merge_lists(values, executor):
    flat = [item for chunk in values for item in chunk]
    if not flat or not all(isinstance(item, np.ndarray) for item in flat):
        return flat
    shape, dtype = flat[0].shape, flat[0].dtype
    if not all(item.shape == shape and item.dtype == dtype for item in flat):
        return flat
    return _stack_chunks(values, len(flat), shape, dtype, executor)


def _stack_chunks(values, total, shape, dtype, executor):
    """Stack the chunks' per-frame arrays into one preallocated (V, ...) array.

    Each chunk's ``np.stack(chunk, out=out[a:b])`` is a plain memcpy that
    releases the GIL, so on an executor the chunks copy concurrently.
    """
    out = np.empty((total, *shape), dtype=dtype)
    offsets = np.cumsum([0] + [len(chunk) for chunk in values])

    def copy_chunk(index):
        np.stack(values[index], axis=0, out=out[offsets[index]:offsets[index + 1]])

    indices = [i for i in range(len(values)) if len(values[i])]
    if executor is None:
        for i in indices:
            copy_chunk(i)
    else:
        for future in [executor.submit(copy_chunk, i) for i in indices]:
            future.result()
    return out
