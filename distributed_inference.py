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
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

import gflags

# Reuse all single-GPU helpers verbatim.
import inference as single
from inference import FLAGS  # the same gflags definitions (checkpoint, configure, ...)
from vggt_omega.distributed.attention import build_strategy
from vggt_omega.distributed.eval_reduce import gather_pose_enc_to_rank0, reduce_depth_means
from vggt_omega.distributed.model import build_cp_model
from vggt_omega.distributed.process_group import cp_group, init_distributed
from vggt_omega.distributed.shard import frame_counts_for, shard_frame_ids
from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = get_logger("vggt_omega.distributed_inference")

gflags.DEFINE_string(
    "cp_strategy", "all_gather_kv",
    "Distributed global-attention strategy: 'all_gather_kv' (single-node) or 'ring' (multi-node).",
)
gflags.DEFINE_boolean(
    "profile", False,
    "Profile the first sequence's forward with torch.profiler: log the rank-0 op "
    "table (by CUDA time) and write a per-rank chrome trace (trace_rank{r}.json).",
)


def run_local_inference(model, images, device):
    """Forward on the local frame shard; returns per-frame prediction arrays (numpy)."""
    images = images.contiguous().to(device)
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
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
        single.save_uint16_image(pred_depth_2d[i], depth_scale, os.path.join(depth_dir, name))
        single.save_uint16_image(pred_conf[i], conf_scale, os.path.join(conf_dir, name))

    world_points = single.unproject_depth_map_to_point_map(
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
    single.write_ply(os.path.join(output_dir, f"pointcloud_rank{rank}.ply"), points, colors)


def main():
    rank, world_size, local_rank = init_distributed()
    device = f"cuda:{local_rank}"
    group = cp_group()

    cfg = single.load_config()
    inf = cfg.inference
    dataset = single.build_dataset(cfg)
    strategy = build_strategy(FLAGS.cp_strategy)
    model = build_cp_model(FLAGS.checkpoint, group, strategy, device)

    num_seqs = dataset.num_sequences()
    if rank == 0:
        logger.info(f"{num_seqs} sequence(s), world_size={world_size}, strategy={FLAGS.cp_strategy}")

    for seq_index in range(num_seqs):
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = single.resolve_frame_ids(dataset, seq_index, inf.num_frames)  # global, ordered
        local_ids = shard_frame_ids(frame_ids, rank, world_size)
        frame_index_offset = sum(frame_counts_for(len(frame_ids), world_size)[:rank])

        sample = single.load_sample(dataset, seq_index, local_ids) if len(local_ids) else None

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

        # --- distributed eval ---
        gt = single.gt_from_sample(sample) if sample is not None else {"gt_depth": np.zeros((0,))}
        per_frame = []
        if len(local_ids):
            for i in range(len(local_ids)):
                res = MonoDepthMetric(gt["gt_depth"][i], pred["pred_depth"][..., 0][i], align="median").run()
                per_frame.append({"abs_rel": float(res["abs_rel"]["mean"]),
                                  "delta1": float(res["delta"]["delta1"])})
        depth_means = reduce_depth_means(per_frame, ["abs_rel", "delta1"], group)

        full_pose = gather_pose_enc_to_rank0(pred["pose_enc"], group)
        # GT extrinsics: each rank already loaded its shard -> gather (flattened to
        # (1, n_local, 12)) to rank 0. Avoids re-loading all frames' GT on rank 0.
        if sample is not None:
            gt_ext_t = torch.from_numpy(gt["gt_extrinsics"]).reshape(1, -1, 12).to(device)
        else:
            gt_ext_t = torch.zeros(1, 0, 12, device=device)
        full_gt_ext = gather_pose_enc_to_rank0(gt_ext_t, group)
        if rank == 0:
            full_ext, _ = encoding_to_camera(full_pose.to(device), images.shape[-2:])
            full_ext = full_ext.float().cpu().numpy()[0]
            gt_ext = full_gt_ext.float().cpu().numpy()[0].reshape(-1, 3, 4)
            gt_c2w = single.world_to_camera_to_camera_to_world(gt_ext)
            pred_c2w = single.world_to_camera_to_camera_to_world(full_ext)
            metrics_dir = os.path.join(output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            pose_metrics = CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run(
                vis_path=os.path.join(metrics_dir, "camera_pose"))
            all_metrics = {"scene": seq_name, "num_frames": int(len(frame_ids)),
                           "world_size": world_size, "cp_strategy": FLAGS.cp_strategy,
                           "camera_pose": pose_metrics, "mono_depth": depth_means}
            with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"[{seq_name}] {len(frame_ids)} frames -> {output_dir}\n"
                        f"  ATE rmse = {pose_metrics['ate']['rmse']:.4f} m\n"
                        f"  Abs Rel  = {depth_means['abs_rel']:.4f}")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUse --help for the full flag list.")
    main()
