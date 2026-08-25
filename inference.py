"""Run VGGT-Omega on a dataset's sequences and evaluate camera-pose + mono-depth.

Drives inference from the *training* :class:`~vggt_omega.datasets.composed_dataset.ComposedDataset`,
instantiated from a per-dataset configure (``--configure``) loaded with OmegaConf,
so each frame is tensorized through the exact same contract the model is trained
on. The configure file carries the ``dataset`` + ``common_config`` (with the eval
overrides baked in) plus an ``inference`` block (``num_frames``, ``image_scale``)
that shapes the frames/resolution the dataset yields; the checkpoint and the
output/fusion knobs stay as command-line flags. Per sequence we:

  1. predict **camera poses** and **monocular depth** (+ a fused point cloud);
  2. dump them (depth/conf PNGs, ``cameras.json``, ``pointcloud.ply``);
  3. evaluate against the TUM ground truth -- camera pose (ATE / RPE) and mono
     depth (Abs Rel / delta);
  4. with ``--enable_3dgs``, also decode the **3D gaussian** scene, rasterize the
     input views back out of it, score that render (PSNR / SSIM / LPIPS) and dump
     the fused ``gaussians.ply`` plus a per-view ground-truth | rendered | error
     sheet (ported from the sibling MUSA repo).

The 3DGS stage needs a checkpoint whose gaussian head was trained (``enable_3dgs:
true`` in its training config) and ``--gs_sh_degree`` equal to the degree it was
trained at. Note what its metrics are and are not: the views rendered are the same
views fed in, so a **self-render** score can be met by geometry that is wrong
everywhere the model was not looked at. Read it as reconstruction quality of the
given views, not as novel-view synthesis.

TUM has no independent point-cloud ground truth: its "world points" are only the
GT depth re-projected through the GT poses, so the fused cloud is exported for
visualization but NOT scored (scoring it against re-projected depth is circular).

Conventions (see ``vggt_omega/datasets``): extrinsics are world-to-camera OpenCV
``[R|t]``; depth is metres with ``0`` = invalid. VGGT predicts geometry only up to
a global scale, so depth uses a per-image median scale and poses a Umeyama
``Sim3`` alignment before scoring.

Usage (the dataset, sequences, frame count and resolution come from the
``--configure`` file; the checkpoint and output/fusion knobs are flags), single GPU::

    python inference.py
    python inference.py --configure vggt_omega/datasets/config/tum.yaml
    python inference.py --checkpoint /path/to/model.pt --output_root /tmp/out

``--help`` lists every flag.
"""

import json
import os
import sys
import time

import gflags
import numpy as np
import torch
from tqdm import tqdm

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.inference_common import (
    FLAGS,
    unproject_depth_map_to_point_map,
    world_to_camera_to_camera_to_world,
    save_uint16_image,
    write_ply,
    load_config,
    build_dataset,
    resolve_frame_ids,
    resolve_checkpoint,
    load_sample,
    gt_from_sample,
)
from vggt_omega.evaluates.scene import (
    score_camera_pose,
    score_depth_frames,
    depth_sums,
    aggregate_depth_from_sums,
    assemble_metrics,
)

logger = get_logger("vggt_omega.inference")

# --- command-line flags (inference.py-specific) ------------------------------
gflags.DEFINE_boolean(
    "profile_mfu",
    False,
    "After the timed forward, count forward FLOPs (FlopCounterMode) and log "
    "model-FLOPs utilization vs the A100 bf16 peak. Adds one extra forward pass.",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_status(tag: str) -> None:
    """Log current / peak GPU memory and free/total (no-op on CPU)."""
    if device != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    logger.info(
        f"[GPU {tag}] alloc={torch.cuda.memory_allocated() / 1e9:.1f}G "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}G "
        f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}G "
        f"free={free / 1e9:.1f}G / {total / 1e9:.1f}G"
    )


A100_BF16_PEAK = 312e12  # dense bf16 tensor-core peak FLOP/s (A100 80GB)


def _log_mfu(model, images: torch.Tensor, infer_secs: float) -> None:
    """Count forward FLOPs once (FlopCounterMode, FMA=2) and log throughput +
    model-FLOPs utilization against the A100 bf16 peak.

    The rate uses the CLEAN forward time ``infer_secs``; the counted pass here is
    for FLOPs only (its own wall time carries dispatch overhead, so it is not
    timed). FlopCounterMode tallies real mm / conv / SDPA ops, so attention's
    S^2 term is included -- which the 2*N*D analytic shortcut would miss for
    VGGT's alternating frame/global attention. Adds one extra forward (opt-in)."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        model(images)
    flops = fc.get_total_flops()
    logger.info(
        f"[MFU] {flops / 1e12:.1f} TFLOP fwd / {infer_secs:.1f}s = "
        f"{flops / infer_secs / 1e12:.1f} TFLOP/s -> "
        f"{flops / infer_secs / A100_BF16_PEAK * 100:.1f}% of A100 bf16 "
        f"({A100_BF16_PEAK / 1e12:.0f} TF/s peak)"
    )


# --- 2. Model + inference ----------------------------------------------------
def build_model() -> VGGTOmega:
    """Build VGGT-Omega once and load the checkpoint.

    The gaussian head is built only under ``--enable_3dgs``; a checkpoint that
    carries one then loads strictly, and one that does not fails on its missing
    ``gs_dpt_head.*`` keys -- the right outcome, since there is nothing to render.
    """
    model = VGGTOmega(
        enable_3dgs=FLAGS.enable_3dgs, gs_sh_degree=FLAGS.gs_sh_degree
    ).to(device).eval()
    model.load_state_dict(torch.load(resolve_checkpoint(FLAGS.checkpoint), map_location="cpu"))
    gpu_status("model loaded")
    return model


def run_inference(model: VGGTOmega, images: torch.Tensor) -> dict:
    """Run the forward pass on one sequence's frames and extract prediction arrays.

    ``images`` is the training-identical ``(S,3,H,W)`` tensor in ``[0,1]`` produced
    by the dataset loader -- no hand-rolled normalization here.
    """
    images = images.contiguous().to(device)
    logger.info(
        f"running inference on {images.shape[0]} frames @ {images.shape[-2]}x{images.shape[-1]} (HxW) ..."
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_infer = time.time()
    try:
        # inference_mode is the faster context, but its tensors are barred from
        # autograd bookkeeping FOREVER -- and gsplat's rasterizer is an
        # autograd.Function that calls save_for_backward unconditionally, so
        # rendering the decoded gaussians later would die on "Inference tensors
        # cannot be saved for backward". no_grad gives the same gradient-free
        # forward without poisoning the outputs.
        with torch.no_grad() if FLAGS.enable_3dgs else torch.inference_mode():
            predictions = model(images, decode_gaussians=FLAGS.enable_3dgs)
        if device == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        gpu_status("OOM")
        total = torch.cuda.mem_get_info()[1] / 1e9
        # native-VGA cost ~= 5.9 + 0.087*N GB (README curve scaled to ~1200 tok/frame).
        fits = int(max(0, (total - 5.9) / 0.087))
        raise SystemExit(
            f"\nCUDA OOM on {images.shape[0]} frames @ {images.shape[-2]}x{images.shape[-1]}. "
            f"A single {total:.0f}G GPU fits ~{fits} native-VGA frames in one pass. "
            f"Lower --num_frames (<=~{fits}) or drop --image_scale."
        )
    infer_secs = time.time() - t_infer
    logger.info(f"inference done in {infer_secs:.1f}s")
    gpu_status("after forward")
    if FLAGS.profile_mfu:
        _log_mfu(model, images, infer_secs)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )

    # Pull to CPU/numpy (float() guards against bf16, which numpy can't hold).
    extracted = {
        "pred_depth": predictions["depth"].float().cpu().numpy()[0],  # (S, H, W, 1)
        "pred_conf": predictions["depth_conf"].float().cpu().numpy()[0],  # (S, H, W)
        "images_pred": predictions["images"]
        .float()
        .cpu()
        .numpy()[0],  # (S, 3, H, W) [0,1]
        "pred_extrinsics": extrinsics.float().cpu().numpy()[0],  # (S, 3, 4) world->cam
        "pred_intrinsics": intrinsics.float().cpu().numpy()[0],  # (S, 3, 3) pixels
    }
    if "gaussians" in predictions:
        # The 3DGS stage rasterizes these, so they stay TENSORS on the device --
        # round-tripping millions of splats through numpy would cost more than
        # the render itself. The camera tensors ride along for the same reason.
        extracted.update({
            "gaussians": predictions["gaussians"],
            "gaussians_valid": predictions["gaussians_valid"],
            "gaussians_pixel_mask": predictions["gaussians_pixel_mask"],
            "images_t": predictions["images"],
            "extrinsics_t": extrinsics,
            "intrinsics_t": intrinsics,
        })
    return extracted


# --- 2b. 3D gaussian splatting: decode -> render -> score -> dump -------------
#: Byte budget for gsplat's spherical-harmonic expansion when --gs_render_chunk
#: is auto. Deliberately a constant rather than a fraction of free memory, so a
#: run behaves the same on a busy GPU as on an idle one.
_SH_BUDGET_BYTES = 8 << 30


def render_views_chunked(gaussians, extrinsics, intrinsics, size_hw, rasterize_mode, chunk=0):
    """Rasterize the views in groups, concatenating along the view axis.

    gsplat expands spherical harmonics to (views, gaussians, coeffs, 3) in one
    allocation, so peak memory is linear in the view count -- 16 views of a
    quarter-resolution garden scene asks for 43 GiB and dies, while the same
    scene renders comfortably a few views at a time. Each camera rasterizes the
    same gaussian set independently, so the split changes nothing but the number
    of kernel launches; the output is bit-identical to a single call.

    ``chunk<=0`` sizes the group from ``_SH_BUDGET_BYTES``.
    """
    from gaussian_splat.render.render_by_gsplat import render_by_gsplat

    num_views = extrinsics.shape[1]
    if chunk <= 0:
        # fp32 coefficients plus a same-sized scratch buffer for the evaluation.
        per_view = gaussians.means.shape[1] * gaussians.harmonics.shape[-1] * 3 * 4 * 2
        chunk = max(1, min(num_views, int(_SH_BUDGET_BYTES // max(per_view, 1))))
    if chunk < num_views:
        logger.info(f"[3dgs] rendering {num_views} views in chunks of {chunk} (SH memory)")

    rendered = []
    for start in range(0, num_views, chunk):
        stop = min(start + chunk, num_views)
        rendered.append(
            render_by_gsplat(
                gaussians, extrinsics[:, start:stop], intrinsics[:, start:stop],
                size_hw, rasterize_mode=rasterize_mode,
            ).color.clamp(0, 1)
        )
    return torch.cat(rendered, dim=1)


def dump_gaussian_scene(output_dir: str, pred: dict, frame_ids) -> dict:
    """Rasterize the decoded gaussian scene back into the input views and dump it.

    Products, all under ``output_dir``:

      ``gaussians.ply``          the fused scene, INRIA/SuperSplat layout
      ``render_compare/``        one sheet per view -- ground truth | rendered |
                                 absolute error -- titled with that view's PSNR,
                                 SSIM and LPIPS
      ``metrics/gaussians.json`` per-view and pooled scores

    These are SELF-RENDER scores: the views rendered are the views fed in. A high
    number here is consistent with geometry that is wrong everywhere the model was
    not looked at, so read it as reconstruction quality, not novel-view synthesis.

    Both a per-view mean PSNR and a pooled PSNR (from the total MSE) are reported.
    They answer different questions -- the mean is what novel-view papers quote,
    while the pooled figure is dominated by the worst view -- and a wide gap
    between them means the views disagree badly.
    """
    from gaussian_splat.utils import dump_gaussians_ply
    from vggt_omega.evaluates.photometric import dump_render_comparison, per_view_metrics

    gaussians = pred["gaussians"]
    images, extrinsics, intrinsics = pred["images_t"], pred["extrinsics_t"], pred["intrinsics_t"]
    size_hw = tuple(images.shape[-2:])

    t_render = time.time()
    with torch.no_grad():
        rendered = render_views_chunked(
            gaussians, extrinsics, intrinsics, size_hw,
            FLAGS.gs_rasterize_mode, FLAGS.gs_render_chunk,
        )
        # LPIPS holds a full trunk activation stack per image, so it is scored a
        # view at a time (per_view_metrics already loops).
        metrics = per_view_metrics(rendered, images, lpips_net=FLAGS.gs_lpips_net)
    logger.info(f"[3dgs] rendered + scored {images.shape[1]} views in {time.time() - t_render:.1f}s")

    ply_path = os.path.join(output_dir, "gaussians.ply")
    if FLAGS.gs_dump_ply:
        num_splats = dump_gaussians_ply(ply_path, gaussians, mask=pred["gaussians_valid"])
    else:
        # The count the dump would have written: the valid splats of batch item 0.
        num_splats = int(pred["gaussians_valid"][0].sum())
        ply_path += " (skipped: --nogs_dump_ply)"
    sheets = dump_render_comparison(
        os.path.join(output_dir, "render_compare"),
        rendered, images, metrics, frame_ids=list(frame_ids),
    )

    # Aggregate over FINITE views only, and say how many were dropped. A single
    # bad view must not turn a whole metric into NaN, but it must not vanish
    # quietly either -- the count is reported next to the mean.
    def finite_mean(values):
        finite = values[np.isfinite(values)]
        return (float(finite.mean()) if finite.size else float("nan")), int(values.size - finite.size)

    psnr_mean, psnr_bad = finite_mean(metrics["psnr"])
    ssim_mean, ssim_bad = finite_mean(metrics["ssim"])
    lpips_mean, lpips_bad = finite_mean(metrics["lpips"])
    pooled_mse, _ = finite_mean(metrics["mse"])
    stats = {
        "num_splats": int(num_splats),
        "rasterize_mode": FLAGS.gs_rasterize_mode,
        "lpips_net": FLAGS.gs_lpips_net,
        "sh_degree": int(FLAGS.gs_sh_degree),
        "note": "SELF-RENDER: the views scored are the views the model was given.",
        "psnr_mean": psnr_mean,
        "psnr_pooled": float(10.0 * np.log10(1.0 / max(pooled_mse, 1e-10))),
        "ssim_mean": ssim_mean,
        "lpips_mean": lpips_mean,
        "nonfinite_views": {"psnr": psnr_bad, "ssim": ssim_bad, "lpips": lpips_bad},
        "pixels_kept_by_2d_filter": float(pred["gaussians_pixel_mask"].float().mean()),
        "per_view": [
            {
                "index": int(i),
                "frame_id": int(frame_ids[i]),
                "psnr": float(metrics["psnr"][i]),
                "ssim": float(metrics["ssim"][i]),
                "lpips": float(metrics["lpips"][i]),
            }
            for i in range(len(metrics["psnr"]))
        ],
    }
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "gaussians.json"), "w") as f:
        json.dump(stats, f, indent=2)

    stats["_ply_path"] = ply_path
    stats["_sheets"] = sheets
    return stats


def dump_and_eval(
    seq_name: str,
    output_dir: str,
    frame_ids,
    loaded: dict,
    pred: dict,
    conf_percentile: float,
    max_points: int,
) -> None:
    """Dump predictions, fuse the PLY, evaluate metrics, and report for one sequence."""
    pred_depth = pred["pred_depth"]
    pred_conf = pred["pred_conf"]
    images_np = pred["images_pred"]
    pred_extrinsics = pred["pred_extrinsics"]
    pred_intrinsics = pred["pred_intrinsics"]
    gt_depth = loaded["gt_depth"]
    gt_extrinsics = loaded["gt_extrinsics"]

    num_f, height, width = pred_depth.shape[:3]
    pred_depth_2d = pred_depth[..., 0]  # (S, H, W)
    images_hwc_pred = np.transpose(images_np, (0, 2, 3, 1))  # (S, H, W, 3)

    # --- 3. Dump predicted depth/conf PNGs + cameras.json ------------------------
    depth_dir = os.path.join(output_dir, "depth")
    conf_dir = os.path.join(output_dir, "conf")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    valid_depth = np.isfinite(pred_depth_2d) & (pred_depth_2d > 0)
    depth_max = float(pred_depth_2d[valid_depth].max()) if valid_depth.any() else 1.0
    depth_scale = 65535.0 / depth_max if depth_max > 0 else 1.0
    finite_conf = np.isfinite(pred_conf)
    conf_max = float(pred_conf[finite_conf].max()) if finite_conf.any() else 1.0
    conf_scale = 65535.0 / conf_max if conf_max > 0 else 1.0

    frames_meta = []
    for i in tqdm(range(num_f), desc="dump depth/conf", unit="frame"):
        name = f"frame_{i:04d}.png"
        save_uint16_image(pred_depth_2d[i], depth_scale, os.path.join(depth_dir, name))
        save_uint16_image(pred_conf[i], conf_scale, os.path.join(conf_dir, name))
        frames_meta.append(
            {
                "index": int(i),
                "frame_id": int(frame_ids[i]),
                "depth": os.path.join("depth", name),
                "conf": os.path.join("conf", name),
                "intrinsics": pred_intrinsics[i].tolist(),
                "extrinsics": pred_extrinsics[i].tolist(),
            }
        )

    camera_meta = {
        "scene": seq_name,
        "image_width": int(width),
        "image_height": int(height),
        "num_frames": int(num_f),
        "depth_scale": depth_scale,
        "depth_max": depth_max,
        "conf_scale": conf_scale,
        "conf_max": conf_max,
        "depth_unit": "uint16_value / depth_scale",
        "extrinsics_convention": "world_to_camera (OpenCV), 3x4 [R|t]",
        "intrinsics_convention": "pixels, 3x3 K",
        "frames": frames_meta,
    }
    with open(os.path.join(output_dir, "cameras.json"), "w") as f:
        json.dump(camera_meta, f, indent=2)

    # --- 4. Fuse predicted depth + RGB into a world-frame PLY --------------------
    pred_world_points = unproject_depth_map_to_point_map(
        pred_depth, pred_extrinsics, pred_intrinsics
    )  # (S, H, W, 3)
    points = pred_world_points.reshape(-1, 3)
    colors = (images_hwc_pred.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_flat = pred_conf.reshape(-1)
    depth_flat = pred_depth_2d.reshape(-1)

    mask = np.isfinite(points).all(axis=1) & np.isfinite(conf_flat) & (depth_flat > 0)
    if conf_percentile > 0 and mask.any():
        threshold = np.percentile(conf_flat[mask], conf_percentile)
        mask &= conf_flat >= threshold
    points, colors = points[mask], colors[mask]

    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[keep], colors[keep]

    ply_path = os.path.join(output_dir, "pointcloud.ply")
    write_ply(ply_path, points, colors)

    # --- 4b. Optional 3DGS: self-render the decoded scene, score it, dump it ------
    gs_stats = dump_gaussian_scene(output_dir, pred, frame_ids) if "gaussians" in pred else None

    # --- 5. Evaluate against the dataset's advertised GT modalities (shared path) -
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    modalities = set(loaded.get("modalities") or [])

    camera_pose_metrics = score_camera_pose(
        world_to_camera_to_camera_to_world(gt_extrinsics),
        world_to_camera_to_camera_to_world(pred_extrinsics),
        modalities=modalities,
        num_frames=num_f,
        vis_path=os.path.join(metrics_dir, "camera_pose"),
    )
    mono_depth_metrics = aggregate_depth_from_sums(
        *depth_sums(score_depth_frames(gt_depth, pred_depth_2d, modalities=modalities))
    )

    all_metrics = assemble_metrics(
        seq_name, num_f, camera_pose_metrics, mono_depth_metrics,
        resolution=[int(height), int(width)],
    )
    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- 6. Report ---------------------------------------------------------------
    report = [
        f"[{seq_name}] {num_f} frames @ {height}x{width} -> {output_dir}",
        f"  point cloud: {points.shape[0]} points -> {ply_path}",
    ]
    if camera_pose_metrics is not None:
        report += [
            "  camera pose (Sim3-aligned):",
            f"    ATE  rmse = {camera_pose_metrics['ate']['rmse']:.4f} m",
            f"    RPE  trans rmse = {camera_pose_metrics['rpe_trans']['rmse']:.4f} m",
            f"    RPE  rot   rmse = {camera_pose_metrics['rpe_rot']['rmse']:.4f} deg",
        ]
    else:
        report.append("  camera pose: skipped (no EXTRINSICS ground truth)")
    if mono_depth_metrics is not None:
        report += [
            "  mono depth (median-aligned, mean over frames):",
            f"    Abs Rel = {mono_depth_metrics['abs_rel_mean']:.4f}",
            f"    delta1  = {mono_depth_metrics['delta1']:.4f}",
        ]
    else:
        report.append("  mono depth: skipped (no DEPTH ground truth)")
    if gs_stats is not None:
        report += [
            f"  3dgs: {gs_stats['num_splats']:,} splats -> {gs_stats['_ply_path']}",
            f"  3dgs self-render ({gs_stats['rasterize_mode']}, lpips={gs_stats['lpips_net']}):",
            f"    PSNR  {gs_stats['psnr_mean']:.2f} dB per-view mean "
            f"({gs_stats['psnr_pooled']:.2f} dB pooled)",
            f"    SSIM  {gs_stats['ssim_mean']:.4f}",
            f"    LPIPS {gs_stats['lpips_mean']:.4f}",
        ]
        dropped = {k: v for k, v in gs_stats["nonfinite_views"].items() if v}
        if dropped:
            report.append(
                "    non-finite views excluded from the means: "
                + ", ".join(f"{k} {v}/{len(gs_stats['per_view'])}" for k, v in dropped.items())
            )
        report.append(
            f"    {gs_stats['_sheets']} per-view sheets -> {os.path.join(output_dir, 'render_compare')}"
        )
    report.append(f"  full metrics -> {os.path.join(metrics_dir, 'metrics.json')}")
    logger.info("\n".join(report))


def main():
    cfg = load_config()
    inf = cfg.inference
    dataset = build_dataset(cfg)
    model = build_model()

    num_seqs = dataset.num_sequences()
    logger.info(
        f"{num_seqs} sequence(s) @ {dataset.img_size} long side (scale {inf.image_scale}), "
        f"num_frames={'all' if inf.num_frames <= 0 else inf.num_frames}"
    )

    for seq_index in range(num_seqs):
        seq_name = dataset.sequence_name(seq_index)
        frame_ids = resolve_frame_ids(dataset, seq_index, inf)
        logger.info(
            f"[{seq_name}] ({seq_index + 1}/{num_seqs}) {len(frame_ids)} frames: "
            f"{frame_ids.tolist()}"
        )

        sample = load_sample(dataset, seq_index, frame_ids)
        pred = run_inference(model, sample["images"])

        output_dir = os.path.join(FLAGS.output_root, seq_name)
        dump_and_eval(
            seq_name,
            output_dir,
            frame_ids,
            gt_from_sample(sample),
            pred,
            FLAGS.conf_percentile,
            FLAGS.max_points,
        )


if __name__ == "__main__":
    try:
        FLAGS(sys.argv)  # parse command-line flags
    except gflags.FlagsError as err:
        sys.exit(f"{err}\nUse --help for the full flag list.")
    main()
