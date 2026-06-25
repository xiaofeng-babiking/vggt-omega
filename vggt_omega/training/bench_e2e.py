"""Full end-to-end training-step benchmark for VGGT-Omega.

Mirrors the REAL training setup: builds the model, four-term loss, AdamW and the
dynamic dataloader through :class:`Trainer` (so the recipe is identical), then runs
N steps on a single GPU with CUDA-synchronized per-stage timing:

    data        : pull the next batch from the dynamic loader (read + sample +
                  tensorize + track build, across the DataLoader workers)
    h2d         : copy the batch to the GPU (host->device)
    forward     : model(images)  (aggregator bf16 + heads fp32)
    loss        : four-term loss compute
    backward    : loss.backward()
    grad_clip   : clip_grad_norm_
    optim       : optimizer.step + zero_grad + scheduler.step
    --------------------------------------------------------------
    step_total  : sum of the above (wall time per training step)

Reports a per-stage elapsed-time table (mean / p50 / p90 over steady-state steps),
plus first-step latency and a data-vs-compute verdict.

Usage (single GPU, real recipe):

    OPENCV_IO_ENABLE_OPENEXR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vggt_omega.training.bench_e2e \
        --config vggt_omega/training/config/train_100k.yaml --steps 30 --warmup 5
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from vggt_omega.training.trainer import Trainer


class CudaTimer:
    """Wall-clock timer that CUDA-synchronizes so async kernels are attributed to
    the right stage. On CPU it degrades to plain perf_counter."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.device = device

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.dt = time.perf_counter() - self.t0


def run(args):
    cfg = OmegaConf.load(args.config)
    # Single-process: keep WORLD_SIZE unset so Trainer takes the non-DDP path but
    # still inits a 1-proc group for the DistributedSampler.
    cfg.run.output_dir = os.path.join("/tmp/opencode", "bench_e2e_run")
    os.environ.setdefault("RANK", "0")

    trainer = Trainer(cfg)
    device = trainer.device
    model = trainer.model
    model.train()
    loss_computer = trainer.loss_computer
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    grad_clip = float(cfg.optim.grad_clip)

    stages = ["data", "h2d", "forward", "loss", "backward", "grad_clip", "optim"]
    rows = {s: [] for s in stages}
    rows["step_total"] = []
    shapes = []
    peak_mem = []

    loader = trainer.data.get_loader(0)
    it = iter(loader)
    timer = lambda: CudaTimer(device)

    total_steps = args.warmup + args.steps
    print(f"running {total_steps} steps ({args.warmup} warmup + {args.steps} measured) on {device} ...",
          flush=True)

    first_step_t0 = time.perf_counter()
    for step in range(total_steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_step0 = time.perf_counter()

        # --- data ---
        with timer() as t:
            batch = next(it)
        t_data = t.dt

        # --- h2d ---
        with timer() as t:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
        t_h2d = t.dt

        images = batch["images"]
        match_on = loss_computer.weights.get("match", 0) > 0 and "tracks" in batch

        # --- forward ---
        with timer() as t:
            predictions = model(images, return_last_patch_tokens=match_on)
        t_fwd = t.dt

        # --- loss ---
        with timer() as t:
            losses = loss_computer(predictions, batch, tuple(images.shape[-2:]))
        t_loss = t.dt

        # --- backward ---
        with timer() as t:
            losses["total"].backward()
        t_bwd = t.dt

        # --- grad clip ---
        with timer() as t:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        t_clip = t.dt

        # --- optim ---
        with timer() as t:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        t_opt = t.dt

        step_total = time.perf_counter() - t_step0
        first_lat = (time.perf_counter() - first_step_t0) if step == 0 else None

        if step >= args.warmup:
            rows["data"].append(t_data)
            rows["h2d"].append(t_h2d)
            rows["forward"].append(t_fwd)
            rows["loss"].append(t_loss)
            rows["backward"].append(t_bwd)
            rows["grad_clip"].append(t_clip)
            rows["optim"].append(t_opt)
            rows["step_total"].append(step_total)
            B, S = images.shape[:2]
            shapes.append((B, S, images.shape[-2], images.shape[-1]))
            if device.type == "cuda":
                peak_mem.append(torch.cuda.max_memory_allocated(device) / 1e9)

        tag = " (first, includes warmup)" if step == 0 else ""
        slow = ""
        if t_data > 5.0:
            sn = batch.get("seq_name", "?")
            sn = sn[0] if isinstance(sn, (list, tuple)) else sn
            slow = f"  <<< SLOW DATA from vendor sample: {sn}"
        print(f"  step {step:3d} | total {step_total:6.3f}s | data {t_data:6.3f} | fwd {t_fwd:6.3f} "
              f"| bwd {t_bwd:6.3f} | B{images.shape[0]}xS{images.shape[1]}@{images.shape[-2]}x{images.shape[-1]}{tag}{slow}",
              flush=True)

    print_table(rows, shapes, peak_mem, first_lat=time.perf_counter() - first_step_t0)


def print_table(rows, shapes, peak_mem, first_lat):
    measured = rows["step_total"]
    if not measured:
        print("no measured steps")
        return
    order = ["data", "h2d", "forward", "loss", "backward", "grad_clip", "optim", "step_total"]
    total_mean = np.mean(rows["step_total"])

    print("\n=== PER-STAGE ELAPSED TIME (steady-state, ms) ===")
    head = f"{'stage':<12}{'mean':>9}{'p50':>9}{'p90':>9}{'%step':>8}"
    print(head)
    print("-" * len(head))
    for s in order:
        a = np.array(rows[s]) * 1e3
        pct = 100.0 * np.mean(rows[s]) / total_mean if s != "step_total" else 100.0
        print(f"{s:<12}{a.mean():>9.1f}{np.percentile(a,50):>9.1f}{np.percentile(a,90):>9.1f}{pct:>7.1f}%")

    avgB = np.mean([b for b, *_ in shapes])
    avgS = np.mean([s for _, s, *_ in shapes])
    imgs_per_step = avgB * avgS
    samp_s = 1.0 / total_mean
    imgs_s = imgs_per_step / total_mean

    print("\n=== THROUGHPUT (single GPU) ===")
    print(f"  avg batch          : B={avgB:.1f} x S={avgS:.1f}  (~{imgs_per_step:.1f} imgs/step)")
    print(f"  step time          : {total_mean*1e3:.1f} ms  (p50 {np.percentile(measured,50)*1e3:.1f}, p90 {np.percentile(measured,90)*1e3:.1f})")
    print(f"  throughput         : {samp_s:.2f} steps/s | {imgs_s:.1f} imgs/s")
    if peak_mem:
        print(f"  peak GPU mem       : {np.mean(peak_mem):.1f} GB (max {np.max(peak_mem):.1f} GB)")
    print(f"  first-step latency : {first_lat:.1f} s (model build excluded; loader warmup + step 0)")

    # verdict
    data_pct = 100.0 * np.mean(rows["data"]) / total_mean
    compute = np.mean(rows["forward"]) + np.mean(rows["backward"]) + np.mean(rows["loss"])
    compute_pct = 100.0 * compute / total_mean
    print("\n=== VERDICT ===")
    print(f"  data   share of step : {data_pct:.1f}%")
    print(f"  compute share of step: {compute_pct:.1f}%  (fwd+loss+bwd)")
    if data_pct > 55:
        print("  -> DATA-BOUND: the loader is the bottleneck. With persistent_workers + more")
        print("     num_workers the data stage overlaps compute and this share should drop.")
        print("     If it does NOT drop with more workers, the bound is PERSISTENT (per-sample CPU cost).")
    elif compute_pct > 55:
        print("  -> COMPUTE-BOUND: GPU forward/backward dominates; data feeding keeps up.")
    else:
        print("  -> BALANCED: neither stage dominates; tuning workers gives marginal gains.")
    print("\n  NOTE: this serial loop does NOT overlap data with compute (no prefetch hiding).")
    print("        Real training with num_workers>0 prefetches the NEXT batch during the GPU")
    print("        forward/backward, so the effective data cost is max(0, data - compute), not data.")
    print("        Compare 'data' mean vs 'forward+backward' mean: if data < fwd+bwd, prefetch fully hides it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="vggt_omega/training/config/train_100k.yaml")
    ap.add_argument("--steps", type=int, default=30, help="measured steps after warmup")
    ap.add_argument("--warmup", type=int, default=5, help="warmup steps (excluded from stats)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
