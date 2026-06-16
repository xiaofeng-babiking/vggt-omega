"""CUDA-synchronized phase profile of the train step (the perf acceptance instrument).

Usage (single GPU):
    .venv/bin/python -m vggt_omega.training.profile_step --config <yaml> [--warm 8] [--steps 40]
Under DDP (rank 0 reports):
    .venv/bin/torchrun --standalone --nproc_per_node=N -m vggt_omega.training.profile_step --config <yaml>

Phases: data_wait | h2d_copy | forward | loss | backward | opt_step. Synchronized timing
removes CPU/GPU overlap, so the phase-sum slightly exceeds the async step time.
"""
import argparse
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from vggt_omega.training.trainer import Trainer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training config YAML")
    p.add_argument("--warm", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--output_dir", default="outputs/profile_step")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = OmegaConf.load(args.config)
    cfg.run.output_dir = f"{args.output_dir}_w{os.environ.get('WORLD_SIZE', '1')}"
    cfg.run.val_at_start = False
    cfg.run.val_interval = 0
    cfg.run.ckpt_interval = 10**9
    cfg.run.max_steps = args.warm + args.steps + 10
    cfg.model.checkpoint = None  # random init: identical compute, no 4.6 GB load

    t = Trainer(cfg)
    model, lc = t.model, t.loss_computer
    it = iter(t.data.get_loader(0))
    model.train()

    def cuda_t():
        if t.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    rows = []
    for i in range(args.warm + args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        t_data = time.perf_counter() - t0
        t0 = cuda_t(); batch = t._to_device(batch); t_h2d = cuda_t() - t0
        images = batch["images"]
        match_on = lc.weights.get("match", 0) > 0 and "tracks" in batch
        t0 = cuda_t(); preds = model(images, return_last_patch_tokens=match_on); t_fwd = cuda_t() - t0
        t0 = cuda_t(); losses = lc(preds, batch, tuple(images.shape[-2:])); t_loss = cuda_t() - t0
        t0 = cuda_t(); losses["total"].backward(); t_bwd = cuda_t() - t0
        t0 = cuda_t()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.cfg.optim.grad_clip)
        t.optimizer.step(); t.optimizer.zero_grad(set_to_none=True); t.scheduler.step()
        t_opt = cuda_t() - t0
        if i >= args.warm:
            rows.append((t_data, t_h2d, t_fwd, t_loss, t_bwd, t_opt, images.shape[0] * images.shape[1]))

    if int(os.environ.get("RANK", "0")) == 0:
        a = np.array([r[:6] for r in rows])
        imgs = np.array([r[6] for r in rows])
        tot = a.sum(1)
        print(f"world={os.environ.get('WORLD_SIZE', '1')} steps={args.steps} "
              f"phase-sum mean {tot.mean():.2f}s | imgs/s/gpu {imgs.sum() / tot.sum():.2f}")
        for j, n in enumerate(["data_wait", "h2d_copy", "forward", "loss", "backward", "opt_step"]):
            c = a[:, j]
            print(f"{n:<10} mean={c.mean()*1000:6.0f}ms median={np.median(c)*1000:6.0f}ms "
                  f"p90={np.percentile(c, 90)*1000:6.0f}ms share={c.mean()/tot.mean()*100:5.1f}%")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
