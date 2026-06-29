"""Frame-sharded (context-parallel) distributed inference for long 3D sequences.

Mirrors inference.py but shards each sequence's frames across the torchrun world
and computes the cross/global attention across ranks. Each rank embeds only its
frames, so memory and the O((N*P)^2) global-attention compute scale by 1/world.

Launch (G GPUs on one node):
    torchrun --standalone --nproc_per_node=G distributed_inference.py \
        --configure vggt_omega/datasets/config/tum.yaml \
        --checkpoint /path/to/vggt_omega_1b_512.pt \
        --cp_strategy all_gather_kv

Single-GPU inference.py is unchanged. Depth/conf PNGs are written per-rank
(filenames carry the global frame_id); depth metrics are reduced across ranks;
camera-pose (ATE/RPE) runs on rank 0 over the gathered trajectory.
"""
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

import gflags

from vggt_omega import inference_common as common
from vggt_omega.inference_common import FLAGS
from vggt_omega.evaluates.scene import score_camera_pose, score_depth_frames, assemble_metrics
from vggt_omega.distributed.attention import build_strategy
from vggt_omega.distributed.eval_reduce import gather_pose_enc_to_rank0, reduce_depth
from vggt_omega.distributed.model import build_cp_model
from vggt_omega.distributed.process_group import cp_group, init_distributed
from vggt_omega.distributed.shard import frame_counts_for, shard_frame_ids
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = get_logger("vggt_omega.distributed_inference")

gflags.DEFINE_string(
    "cp_strategy", "all_gather_kv",
    "Distributed global-attention strategy: 'all_gather_kv' (gathers the full "
    "K/V per rank; O(N) memory) or 'ring' (flash-tiled blockwise attention, "
    "batched P2P on a dedicated communicator; O(N/world) K/V memory for long "
    "sequences).",
)
gflags.DEFINE_boolean(
    "profile", False,
    "Profile the first sequence's forward with torch.profiler: log the rank-0 op "
    "table (by CUDA time) and write a per-rank chrome trace (trace_rank{r}.json).",
)


def run_local_inference(model, images, device):
    """Forward on the local frame shard; returns per-frame prediction arrays (numpy)."""
    images = images.contiguous().to(device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    logger.info(
        f"[rank {dist.get_rank()}] {images.shape[1]} local frames | "
        f"forward {time.perf_counter() - t0:.1f} s | "
        f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    )
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    return {
        "pose_enc": predictions["pose_enc"],  # tensor (1, n_local, 9) -- gathered later
        "pred_depth": predictions["depth"].float().cpu().numpy()[0],
        "pred_conf": predictions["depth_conf"].float().cpu().numpy()[0],
        "images_pred": predictions["images"].float().cpu().numpy()[0],
        "pred_extrinsics": extrinsics.float().cpu().numpy()[0],
        "pred_intrinsics": intrinsics.float().cpu().numpy()[0],
    }


def dump_local_shard(output_dir, frame_ids_local, frame_index_offset, pred, conf_percentile, max_points):
    """Write this rank's depth/conf PNGs (named by global frame_id) and a partial PLY."""
    depth_dir = os.path.join(output_dir, "depth")
    conf_dir = os.path.join(output_dir, "conf")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    pred_depth = pred["pred_depth"]
    pred_depth_2d = pred_depth[..., 0]
    pred_conf = pred["pred_conf"]
    images_hwc = np.transpose(pred["images_pred"], (0, 2, 3, 1))

    valid = np.isfinite(pred_depth_2d) & (pred_depth_2d > 0)
    depth_max = float(pred_depth_2d[valid].max()) if valid.any() else 1.0
    depth_scale = 65535.0 / depth_max if depth_max > 0 else 1.0
    finite_conf = np.isfinite(pred_conf)
    conf_max = float(pred_conf[finite_conf].max()) if finite_conf.any() else 1.0
    conf_scale = 65535.0 / conf_max if conf_max > 0 else 1.0

    for i in range(len(frame_ids_local)):
        name = f"frame_{frame_index_offset + i:04d}.png"
        common.save_uint16_image(pred_depth_2d[i], depth_scale, os.path.join(depth_dir, name))
        common.save_uint16_image(pred_conf[i], conf_scale, os.path.join(conf_dir, name))

    world_points = common.unproject_depth_map_to_point_map(
        pred_depth, pred["pred_extrinsics"], pred["pred_intrinsics"]
    )
    points = world_points.reshape(-1, 3)
    colors = (images_hwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_flat = pred_conf.reshape(-1)
    depth_flat = pred_depth_2d.reshape(-1)
    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf_flat) & (depth_flat > 0)
    if conf_percentile > 0 and mask.any():
        thr = np.percentile(conf_flat[mask], conf_percentile)
        mask &= conf_flat >= thr
    points, colors = points[mask], colors[mask]
    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[keep], colors[keep]
    rank = dist.get_rank()
    common.write_ply(os.path.join(output_dir, f"pointcloud_rank{rank}.ply"), points, colors)


def main():
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}"
    group = cp_group()

    cfg = common.load_config()
    inf = cfg.inference
    dataset = common.build_dataset(cfg)
    strategy = build_strategy(FLAGS.cp_strategy)
    model = build_cp_model(FLAGS.checkpoint, group, strategy, device)

    num_seqs = dataset.num_sequences()
    if rank == 0:
        logger.info(f"{num_seqs} sequence(s), world_size={world_size}, strategy={FLAGS.cp_strategy}")

    for seq_index in range(num_seqs):
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = common.resolve_frame_ids(dataset, seq_index, inf.num_frames)  # global, ordered
        local_ids = shard_frame_ids(frame_ids, rank, world_size)
        frame_index_offset = sum(frame_counts_for(len(frame_ids), world_size)[:rank])

        sample = common.load_sample(dataset, seq_index, local_ids) if len(local_ids) else None

        # Every rank must agree on (H, W) so per-frame token counts match across
        # ranks (the global-attention all_gather shapes must be identical). Rank 0
        # always holds >=1 frame when the sequence is non-empty (remainder goes to
        # the lowest ranks first), so broadcast its resolution to any empty ranks.
        if sample is not None:
            hw = torch.tensor([sample["images"].shape[-2], sample["images"].shape[-1]],
                              dtype=torch.long, device=device)
        else:
            hw = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(hw, src=0, group=group)
        height, width = int(hw[0]), int(hw[1])

        if sample is not None:
            images = sample["images"].unsqueeze(0)               # (1, n_local, 3, H, W)
        else:
            images = torch.zeros(1, 0, 3, height, width)         # empty shard, correct H/W

        output_dir = os.path.join(FLAGS.output_root, seq_name)

        # --profile: time the first sequence's forward. Warm up once (cudnn autotune
        # / allocator) so the trace is representative, then profile. Both the warmup
        # and profiled forward run on EVERY rank (the model forward issues
        # collectives) to stay rank-symmetric and avoid deadlock.
        do_profile = FLAGS.profile and seq_index == 0
        if do_profile:
            run_local_inference(model, images, device)  # warmup (all ranks)
        prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if do_profile else nullcontext()
        with prof_ctx as prof:
            pred = run_local_inference(model, images, device)
        if do_profile:
            if rank == 0:
                logger.info(
                    "profiler top ops (rank 0, by CUDA time):\n"
                    + prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
                )
            os.makedirs(output_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(output_dir, f"trace_rank{rank}.json"))

        if len(local_ids):
            dump_local_shard(output_dir, local_ids, frame_index_offset, pred,
                             FLAGS.conf_percentile, FLAGS.max_points)

        # --- distributed eval (shared scene scorer + cross-rank reduce) ---
        if sample is not None:
            gt = common.gt_from_sample(sample)
        else:
            gt = {"gt_depth": np.zeros((0, 1, 1), np.float32),
                  "gt_extrinsics": np.zeros((0, 3, 4), np.float32), "modalities": []}
        modalities = set(gt.get("modalities") or [])

        per_frame = (score_depth_frames(gt["gt_depth"], pred["pred_depth"][..., 0],
                                        modalities=modalities) if len(local_ids) else [])
        mono_depth = reduce_depth(per_frame, group)  # same dict (or None) on every rank

        full_pose = gather_pose_enc_to_rank0(pred["pose_enc"], group)
        if sample is not None:
            gt_ext_t = torch.from_numpy(gt["gt_extrinsics"]).reshape(1, -1, 12).to(device)
        else:
            gt_ext_t = torch.zeros(1, 0, 12, device=device)
        full_gt_ext = gather_pose_enc_to_rank0(gt_ext_t, group)
        if rank == 0:
            full_ext, _ = encoding_to_camera(full_pose.to(device), images.shape[-2:])
            full_ext = full_ext.float().cpu().numpy()[0]
            gt_ext = full_gt_ext.float().cpu().numpy()[0].reshape(-1, 3, 4)
            metrics_dir = os.path.join(output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            camera_pose = score_camera_pose(
                common.world_to_camera_to_camera_to_world(gt_ext),
                common.world_to_camera_to_camera_to_world(full_ext),
                modalities=modalities, num_frames=len(frame_ids),
                vis_path=os.path.join(metrics_dir, "camera_pose"),
            )
            all_metrics = assemble_metrics(
                seq_name, len(frame_ids), camera_pose, mono_depth,
                world_size=world_size, cp_strategy=FLAGS.cp_strategy,
            )
            with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(
                f"[{seq_name}] {len(frame_ids)} frames -> {output_dir}\n"
                f"  ATE rmse = {camera_pose['ate']['rmse']:.4f} m\n"
                f"  Abs Rel  = {mono_depth['abs_rel_mean']:.4f}"
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUse --help for the full flag list.")
    main()
