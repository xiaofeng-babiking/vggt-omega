#!/usr/bin/env python
"""TEMPORARY: pre-generate per-sequence poses_cache.npz files in parallel.

For per-frame-file vendors, the first ``get_poses()`` call combines N small
per-frame pose files into one ``<seq_dir>/poses_cache.npz`` (then later runs do a
single read). This script warms that cache for every sequence in a training/eval
config up front, in parallel, with a progress bar -- so the first training epoch
doesn't pay the per-sequence combine cost on the data path.

Usage::

    OPENCV_IO_ENABLE_OPENEXR=1 .venv/bin/python scripts/gen_pose_cache.py \
        --config vggt_omega/training/config/train_100k.yaml \
        --workers 32 [--vendors scannet,co3d] [--overwrite]

Reads dataset_configs from the config (resolving env interpolations), discovers
every sequence per vendor, and builds its pose cache. Idempotent: sequences whose
cache already exists are skipped unless --overwrite. Vendors that do not store one
pose per file (get_poses_cache_file() == "") are skipped automatically.

This file is a throwaway utility (not imported by the package); delete after use.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from omegaconf import OmegaConf
from tqdm import tqdm

from vggt_omega.datasets.sequence_dataset import _resolve_sequence_cls


def _vendor_specs(config_path):
    """Yield (vendor_name, SequenceClass, data_root, sequences_globs, sequence_kwargs)
    for each dataset_config entry that uses SequenceDataset."""
    cfg = OmegaConf.load(config_path)
    node = OmegaConf.select(cfg, "data.train.dataset") or cfg.get("dataset")
    if node is None:
        raise SystemExit(f"{config_path}: no data.train.dataset or dataset block found")
    for dc in node.dataset_configs:
        dc = OmegaConf.to_container(dc, resolve=True)
        if not str(dc.get("_target_", "")).endswith("SequenceDataset"):
            continue
        cls = _resolve_sequence_cls(dc["sequence_cls"]["path"])
        name = cls.__name__.replace("Sequence", "").lower()
        yield (
            name,
            cls,
            dc["data_root"],
            list(dc.get("sequences") or ["*"]),
            dict(dc.get("sequence_kwargs") or {}),
        )


# Worker globals (set once per process to avoid re-pickling the class each task).
_W = {}


def _init_worker(cls, data_root, sequence_kwargs):
    _W["cls"] = cls
    _W["data_root"] = data_root
    _W["sequence_kwargs"] = sequence_kwargs


def _build_one(seq_id, overwrite):
    """Construct one sequence and build its poses cache. Returns (seq_id, status)."""
    cls, data_root, skw = _W["cls"], _W["data_root"], _W["sequence_kwargs"]
    try:
        seq = cls(data_root, seq_id, **skw)
        sensor = seq.get_sensors()[0]
        cache = seq.get_poses_cache_file(sensor)
        if not cache:
            return (seq_id, "skip-no-cache")
        if os.path.exists(cache) and not overwrite:
            return (seq_id, "skip-exists")
        if overwrite and os.path.exists(cache):
            try:
                os.remove(cache)
            except OSError:
                pass
        n = len(seq.get_poses(sensor))  # builds + writes the cache
        return (seq_id, f"ok:{n}")
    except Exception as exc:  # dirty sequence: report, do not abort the pool
        return (seq_id, f"ERR:{type(exc).__name__}:{str(exc)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vggt_omega/training/config/train_100k.yaml")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--vendors", default="", help="comma list to limit (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()

    only = {v.strip() for v in args.vendors.split(",") if v.strip()}
    specs = [s for s in _vendor_specs(args.config) if not only or s[0] in only]
    if not specs:
        raise SystemExit("no matching SequenceDataset vendors in config")

    totals = {"ok": 0, "skip-exists": 0, "skip-no-cache": 0, "err": 0}
    for name, cls, data_root, globs, skw in specs:
        try:
            seq_ids = list(cls.discover(data_root, globs))
        except Exception as exc:
            print(f"[{name}] discover failed: {type(exc).__name__}: {exc}")
            continue
        if not seq_ids:
            print(f"[{name}] no sequences (root={data_root}, globs={globs})")
            continue

        errors = []
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(cls, data_root, skw)
        ) as ex:
            futs = {ex.submit(_build_one, sid, args.overwrite): sid for sid in seq_ids}
            with tqdm(total=len(futs), desc=f"{name:14s}", unit="seq", dynamic_ncols=True) as bar:
                for fut in as_completed(futs):
                    sid, status = fut.result()
                    if status.startswith("ok"):
                        totals["ok"] += 1
                    elif status.startswith("ERR"):
                        totals["err"] += 1
                        errors.append((sid, status))
                    elif status in totals:
                        totals[status] += 1
                    bar.update(1)
                    bar.set_postfix(ok=totals["ok"], skip=totals["skip-exists"], err=totals["err"])
        for sid, status in errors[:10]:
            print(f"  [{name}] {sid}: {status}")
        if len(errors) > 10:
            print(f"  [{name}] ... and {len(errors) - 10} more errors")

    print(
        f"\nDONE: built={totals['ok']} "
        f"skipped_exist={totals['skip-exists']} "
        f"skipped_no_cache={totals['skip-no-cache']} "
        f"errors={totals['err']}"
    )


if __name__ == "__main__":
    main()
