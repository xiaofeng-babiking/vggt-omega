"""Standalone dataloader benchmark for VGGT-Omega — find the data bottleneck.

Isolates the data pipeline from training (no DDP, no GPU, no model) and measures:

  1. Per-vendor SINGLE-sample latency, broken into stages:
       - se3      : SequenceDataset._se3_sample_ids (loads ALL poses of the window)
       - get_data : per-frame RGB+depth read + process_one_image
       - track    : build_tracks_by_depth (on-the-fly 3D track synthesis)
       - tensorize: torch conversion + color aug
  2. End-to-end DataLoader throughput at several num_workers (samples/s, imgs/s),
     to tell a TEMPORARY warmup stall from a PERSISTENT data bound.

Usage (from repo root, with the venv):

    OPENCV_IO_ENABLE_OPENEXR=1 .venv/bin/python -m vggt_omega.training.bench_dataloader \
        --config vggt_omega/training/config/train_100k.yaml \
        --per_vendor_samples 5 --loader_batches 40 --workers 0,4,8

All knobs have defaults; run with no args for a quick pass.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.utils import instantiate


@contextmanager
def timed(acc, key):
    t0 = time.perf_counter()
    yield
    acc[key] = acc.get(key, 0.0) + (time.perf_counter() - t0)


def _init_singleproc_group(out_dir="/tmp/opencode"):
    """DynamicDistributedSampler needs an initialized process group even at world=1."""
    if dist.is_initialized():
        return
    store = os.path.join(out_dir, ".bench_dist_init")
    if os.path.exists(store):
        os.remove(store)
    dist.init_process_group(backend="gloo", init_method=f"file://{store}", rank=0, world_size=1)


def build_composed(cfg):
    dcfg = cfg.data.train.dataset
    common = cfg.data.train.common_config
    t0 = time.perf_counter()
    ds = instantiate(dcfg, common_config=common, _recursive_=False)
    build_s = time.perf_counter() - t0
    return ds, common, build_s


def bench_per_vendor(ds, common, n_samples):
    """Time the staged cost of one sample per vendor, bypassing inside_random so we
    hit a KNOWN vendor each call (we drive the vendor's get_data + the composed
    tensorize/track path directly, mirroring __getitem__)."""
    import random as _random
    from vggt_omega.datasets.base_sequence import Modality

    img_lo, img_hi = common.img_nums
    results = []
    for vendor in ds.base_dataset.datasets:
        name = vendor.sequence_cls.__name__.replace("Sequence", "")
        nseq = vendor.sequence_list_len
        acc = {}
        frames_total = 0
        ok = 0
        for _ in range(n_samples):
            try:
                local = _random.randint(0, nseq - 1)
                seq_name = vendor.sequence_list[local]
                seq = vendor._sequence(seq_name)
                sensor = vendor._sensor(seq)
                num = _random.randint(img_lo, min(img_hi, seq.get_length(sensor)))
                with timed(acc, "se3"):
                    ids = vendor._se3_sample_ids(seq, sensor, num)
                with timed(acc, "get_data"):
                    batch = vendor.get_data(seq_name=seq_name, ids=ids, aspect_ratio=1.0)
                # tensorize + (optional) track build, mirroring ComposedDataset.__getitem__
                t0 = time.perf_counter()
                sample = ds._tensorize(batch)
                acc["tensorize+track"] = acc.get("tensorize+track", 0.0) + (time.perf_counter() - t0)
                frames_total += int(batch["frame_num"])
                ok += 1
            except Exception as exc:
                acc["errors"] = acc.get("errors", 0) + 1
        results.append((name, nseq, ok, frames_total, acc))
    return results


def bench_loader(cfg, workers_list, n_batches):
    """End-to-end DataLoader throughput for each worker count. Reports steady-state
    (skips the first few batches as warmup) vs first-batch latency."""
    from vggt_omega.training.collate import train_collate

    out = []
    for nw in workers_list:
        cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg2.data.train.num_workers = int(nw)
        data = instantiate(cfg2.data.train, collate_fn=train_collate, _recursive_=False)
        loader = data.get_loader(0)
        it = iter(loader)
        times, sizes = [], []
        t_first0 = time.perf_counter()
        first_lat = None
        for i in range(n_batches):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            dt = time.perf_counter() - t0
            if first_lat is None:
                first_lat = time.perf_counter() - t_first0
            times.append(dt)
            B, S = batch["images"].shape[:2]
            sizes.append((B, S))
        del it, loader, data
        out.append((nw, first_lat, times, sizes))
    return out


def fmt_vendor(results):
    print("\n=== PER-VENDOR SINGLE-SAMPLE COST (mean ms over OK samples) ===")
    header = f"{'vendor':<20}{'#seq':>7}{'ok':>4}{'se3':>9}{'get_data':>10}{'tens+trk':>10}{'frames':>8}{'err':>5}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, nseq, ok, frames, acc in results:
        d = max(ok, 1)
        se3 = 1e3 * acc.get("se3", 0) / d
        gd = 1e3 * acc.get("get_data", 0) / d
        tt = 1e3 * acc.get("tensorize+track", 0) / d
        fr = frames / d
        err = acc.get("errors", 0)
        rows.append((gd + se3 + tt, name, nseq, ok, se3, gd, tt, fr, err))
        print(f"{name:<20}{nseq:>7}{ok:>4}{se3:>9.1f}{gd:>10.1f}{tt:>10.1f}{fr:>8.1f}{err:>5}")
    print("\n--- slowest vendors by total single-sample latency ---")
    for total, name, *_ in sorted(rows, reverse=True)[:5]:
        print(f"  {name:<20} {total:8.1f} ms/sample")


def fmt_loader(out):
    print("\n=== END-TO-END DATALOADER THROUGHPUT ===")
    print(f"{'workers':>8}{'first_batch_s':>15}{'steady_ms/b':>13}{'p50_ms':>9}{'p90_ms':>9}{'samp/s':>9}{'imgs/s':>9}")
    for nw, first_lat, times, sizes in out:
        if not times:
            print(f"{nw:>8}  (no batches)")
            continue
        warm = times[3:] if len(times) > 6 else times
        arr = np.array(warm)
        mean = arr.mean()
        p50 = np.percentile(arr, 50)
        p90 = np.percentile(arr, 90)
        avgB = np.mean([b for b, _ in sizes])
        avgS = np.mean([s for _, s in sizes])
        samp_s = 1.0 / mean
        imgs_s = avgB * avgS / mean
        print(f"{nw:>8}{first_lat:>15.2f}{1e3*mean:>13.1f}{1e3*p50:>9.1f}{1e3*p90:>9.1f}{samp_s:>9.2f}{imgs_s:>9.1f}")
    print("\nInterpretation:")
    print("  - If steady ms/b << first_batch_s -> the stall was TEMPORARY (warmup: cold cache + worker spin-up).")
    print("  - If steady ms/b stays high and imgs/s barely rises with workers -> PERSISTENT data bound.")
    print("  - Compare imgs/s to the model's needed imgs/s (max_img_per_gpu / GPU step time).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vggt_omega/training/config/train_100k.yaml")
    ap.add_argument("--per_vendor_samples", type=int, default=5)
    ap.add_argument("--loader_batches", type=int, default=40)
    ap.add_argument("--workers", default="0,4,8", help="comma list of num_workers to sweep")
    ap.add_argument("--skip_vendor", action="store_true")
    ap.add_argument("--skip_loader", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    _init_singleproc_group()
    cfg = OmegaConf.load(args.config)

    ds, common, build_s = build_composed(cfg)
    print(f"ComposedDataset built in {build_s:.1f}s | vendors={len(ds.base_dataset.datasets)} "
          f"| total real seqs={ds.num_sequences()} | load_track={common.load_track} "
          f"| max_img_per_gpu={cfg.data.train.max_img_per_gpu} | img_nums={list(common.img_nums)}")

    if not args.skip_vendor:
        res = bench_per_vendor(ds, common, args.per_vendor_samples)
        fmt_vendor(res)

    if not args.skip_loader:
        workers = [int(x) for x in args.workers.split(",") if x.strip() != ""]
        out = bench_loader(cfg, workers, args.loader_batches)
        fmt_loader(out)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
