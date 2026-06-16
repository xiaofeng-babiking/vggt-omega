"""Peak-memory sweep over max_img_per_gpu at the two worst-case batch shapes.

    CUDA_VISIBLE_DEVICES=<free> .venv/bin/python -m vggt_omega.training.sweep_capacity --caps 24,28,32,36,40

For each cap, runs forward+backward+optimizer-step at (B=1, S=cap) (token/attention peak)
and (B=cap, S=1) (dense-head peak) on synthetic 512x512 inputs (aspect 1.0 = worst H) with
full gradient checkpointing and optimizer state allocated, and reports peak GiB. Pick the
largest cap with >= 8 GiB headroom and set it in train_default.yaml (BOTH places:
data.train.max_img_per_gpu and data.train.common_config.max_img_per_gpu).

Synthetic peaks omit the matching-loss path (tracks + retained last-layer patch tokens)
and DDP bucket/allocator effects: real multi-GPU training measured ~9 GiB above the
synthetic number at the same cap. Treat the verdicts as a lower bound and confirm the
chosen cap with a short real run (watch perf/peak_mem_gb in TensorBoard).
"""
import argparse

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.training.losses import TrainLossComputer
from vggt_omega.training.optim import build_param_groups
from vggt_omega.training.trainer import init_model_from_scratch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caps", type=lambda s: [int(x) for x in s.split(",")], default=[24, 28, 32, 36, 40])
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--embed-dim", type=int, default=1024)
    return p.parse_args(argv)


def _synth_batch(B, S, hw, device):
    H = W = hw
    return {
        "images": torch.rand(B, S, 3, H, W, device=device),
        "depths": torch.rand(B, S, H, W, device=device) + 0.5,
        "extrinsics": torch.eye(4, device=device)[:3].expand(B, S, 3, 4).contiguous(),
        "intrinsics": torch.tensor([[256.0, 0, 256], [0, 256.0, 256], [0, 0, 1]], device=device).expand(B, S, 3, 3).contiguous(),
        "world_points": torch.rand(B, S, H, W, 3, device=device),
        "point_masks": torch.ones(B, S, H, W, dtype=torch.bool, device=device),
    }


def main(argv=None):
    args = parse_args(argv)
    assert torch.cuda.is_available(), "capacity sweep is a GPU tool"
    device = torch.device("cuda", 0)
    model = VGGTOmega(embed_dim=args.embed_dim)
    init_model_from_scratch(model)
    model.aggregator.gradient_checkpointing = True
    model = model.to(device).train()
    opt = torch.optim.AdamW(build_param_groups(model, 0.05), lr=1e-8, fused=True)
    lc = TrainLossComputer(weights={"camera": 5.0, "depth": 1.0, "point": 0.5, "match": 0.0})

    # Prime the lazily-allocated AdamW state (exp_avg/exp_avg_sq, ~8.5 GiB fp32 at
    # 1B params) so the FIRST measured shape already sees steady-state memory —
    # without this the first reading under-reports by the full state size.
    batch = _synth_batch(1, 1, args.img_size, device)
    preds = model(batch["images"])
    lc(preds, batch, (args.img_size, args.img_size))["total"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"GPU total {total:.1f} GiB | shapes: (1, cap) and (cap, 1) at {args.img_size}^2")
    for cap in args.caps:
        worst = 0.0
        for B, S in ((1, cap), (cap, 1)):
            try:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                batch = _synth_batch(B, S, args.img_size, device)
                preds = model(batch["images"])
                losses = lc(preds, batch, (args.img_size, args.img_size))
                losses["total"].backward()
                opt.step()
                peak = torch.cuda.max_memory_allocated(device) / 2**30
                worst = max(worst, peak)
                print(f"  cap={cap:3d} (B={B:2d}, S={S:2d}): peak {peak:6.1f} GiB")
            except torch.OutOfMemoryError:
                print(f"  cap={cap:3d} (B={B:2d}, S={S:2d}): OOM")
                worst = float("inf")
                break
        verdict = "OK (>=8 GiB headroom)" if total - worst >= 8 else "TOO TIGHT"
        print(f"  cap={cap:3d} worst {worst:6.1f} GiB -> {verdict}")


if __name__ == "__main__":
    main()
