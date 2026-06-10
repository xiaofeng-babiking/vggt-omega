"""Multi-rank NCCL parity check: RingAttention vs AllGatherKVAttention vs SDPA.

The CPU/gloo unit tests cannot exercise the FlashAttention ring path
(`aten._scaled_dot_product_flash_attention` is CUDA-only), so a flash-path
regression is invisible to `pytest` — THIS script is the gate that catches it.
Run it on any change to `vggt_omega/distributed/attention.py`:

    torchrun --standalone --nproc_per_node=4 \
        vggt_omega/distributed/tests/gpu_parity_check.py

Every rank builds the SAME full-sequence q/k/v from a fixed seed, runs each
strategy on its (uneven) shard, and compares against its slice of the full
SDPA reference computed locally. Modes:
  (a) bf16 inputs            -> flash ring path
  (b) fp32 inputs + autocast -> autocast cast then flash ring path
  (c) fp32 inputs, no autocast -> chunked manual ring path (camera-head regime)

Exits non-zero on any parity failure.
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vggt_omega.distributed.attention import AllGatherKVAttention, RingAttention  # noqa: E402

B, H, D = 1, 16, 64
TOKENS = 1217  # ~one 640x480 frame at patch 16


def frames_for(world: int) -> list[int]:
    """Uneven per-rank frame counts, e.g. [3, 2, 2, 1] at world=4."""
    if world == 1:
        return [3]
    return [3] + [2] * (world - 2) + [1]


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    frames = frames_for(world)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    dev = f"cuda:{rank}"

    g = torch.Generator().manual_seed(7)
    total = sum(frames) * TOKENS
    full = [torch.randn(B, H, total, D, generator=g) for _ in range(3)]
    start = sum(frames[:rank]) * TOKENS
    stop = start + frames[rank] * TOKENS

    failures = []
    for mode in ("bf16", "autocast", "fp32_manual"):
        in_dtype = torch.bfloat16 if mode == "bf16" else torch.float32
        qf, kf, vf = (t.to(dev, in_dtype) for t in full)
        q, k, v = (t[:, :, start:stop].contiguous() for t in (qf, kf, vf))
        ctx = torch.autocast("cuda", torch.bfloat16) if mode == "autocast" else torch.no_grad()
        with torch.no_grad(), ctx:
            ref = F.scaled_dot_product_attention(qf, kf, vf)[:, :, start:stop]
            ring = RingAttention()(q, k, v, dist.group.WORLD)
            agkv = AllGatherKVAttention()(q, k, v, dist.group.WORLD)
        for name, got in (("ring", ring), ("all_gather_kv", agkv)):
            diff = (got.float() - ref.float()).abs().max()
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)
            tol = 2e-4 if mode == "fp32_manual" else 2e-2
            status = "OK " if diff.item() < tol else "FAIL"
            if diff.item() >= tol:
                failures.append((mode, name, diff.item()))
            if rank == 0:
                print(f"[{status}] {mode:12s} {name:14s} dtype_out={got.dtype} "
                      f"max|diff| = {diff.item():.2e} (tol {tol:.0e})", flush=True)

    dist.barrier()
    if rank == 0:
        print("RESULT:", "FAIL" if failures else "ALL PARITY CHECKS PASSED", failures, flush=True)
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
