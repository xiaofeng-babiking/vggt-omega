"""Supervised end-to-end trainer for VGGT-Omega (arXiv 2605.15195 recipe).

Owns model build (checkpoint | correct from-scratch init), optional DDP, the
step-based train loop (no outer autocast — bf16 lives inside the model; no
GradScaler), TensorBoard logging, bare-state-dict checkpoints with a trainer
sidecar, and rank-0 validation through the existing eval metrics.
"""

import glob
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from vggt_omega.models import VGGTOmega
from vggt_omega.training.collate import train_collate
from vggt_omega.training.losses import TrainLossComputer
from vggt_omega.training.optim import build_param_groups, build_warmup_cosine
from vggt_omega.utils.logger import get_logger
from vggt_omega.utils.pose_enc import encoding_to_camera

logger = get_logger("vggt_omega.trainer")


def resolve_comm_hook(name):
    """Map cfg.optim.grad_compression to a DDP comm hook (None = fp32 default)."""
    if name in (None, "none"):
        return None
    if name == "bf16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        return default_hooks.bf16_compress_hook
    raise ValueError(f"unknown grad_compression {name!r} (expected 'bf16' or 'none')")


def init_model_from_scratch(model: nn.Module) -> None:
    """Checkpoint-less init: several parameters are created with ``torch.empty``
    (LayerScale.gamma, the ViT cls/storage/mask tokens) and ``LinearKMaskedBias``
    buffers are NaN-by-design, so a bare ``VGGTOmega(...)`` poisons the forward.

    Sweep ``reset_parameters`` + the modules' own ``init_weights`` (ViT tokens,
    RoPE, camera/register tokens), then apply the ``init_weights_vit`` bias_mask
    convention everywhere (ones with the K third zeroed — NOT all-ones).
    """
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    for module in model.modules():
        init = getattr(module, "init_weights", None)
        if callable(init):
            init()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear) and getattr(module, "bias_mask", None) is not None:
                o = module.out_features
                module.bias_mask.fill_(1)
                module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
        for p in model.parameters():
            if not torch.isfinite(p).all():
                p.normal_(std=0.02)


class Trainer:
    """Build everything from one OmegaConf cfg; ``fit()`` runs to ``run.max_steps``.

    Attributes used by tests/entrypoint: ``global_step``, ``loss_history``
    (list[float] of the weighted total loss per step).
    """

    def __init__(self, cfg, data_override=None):
        self.cfg = cfg
        self.global_step = 0
        self.loss_history = []
        self._epoch = 0

        self._setup_distributed()
        self._seed_everything()

        self.out_dir = str(cfg.run.output_dir)
        self.writer = None
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb"))

        self._build_model()
        self.loss_computer = TrainLossComputer(
            weights=OmegaConf.to_container(cfg.loss.weights),
            alpha=cfg.loss.alpha,
            temperature=cfg.loss.temperature,
            patch_size=int(OmegaConf.select(cfg, "model.patch_size", default=16)),
        )
        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.optim.weight_decay),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
        )
        self.scheduler = build_warmup_cosine(
            self.optimizer, max_steps=int(cfg.run.max_steps), warmup_frac=cfg.optim.warmup_frac
        )
        self._build_data(data_override)

    # --- setup ----------------------------------------------------------------
    def _setup_distributed(self):
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1:
            from vggt_omega.distributed.process_group import init_distributed

            self.rank, self.world_size, self.local_rank = init_distributed()
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0
            self.device = (
                torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            )
        # The dataloader worker_init_fn reads RANK unconditionally — set before any build.
        os.environ.setdefault("RANK", "0")

    def _seed_everything(self):
        seed = int(self.cfg.run.seed) + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self):
        model = VGGTOmega(embed_dim=int(self.cfg.model.embed_dim))
        if self.cfg.model.checkpoint:
            model.load_state_dict(torch.load(self.cfg.model.checkpoint, map_location="cpu"))
        else:
            init_model_from_scratch(model)
        model.aggregator.gradient_checkpointing = bool(self.cfg.model.gradient_checkpointing)
        model = model.to(self.device)
        if self.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            hook = resolve_comm_hook(OmegaConf.select(self.cfg, "optim.grad_compression", default="none"))
            if hook is not None:
                model.register_comm_hook(state=None, hook=hook)
        self.model = model

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _build_data(self, data_override):
        if data_override is not None:
            self.data = data_override
            return
        # DynamicDistributedSampler needs an initialized process group even at world=1.
        if not dist.is_initialized():
            store_path = os.path.join(os.path.abspath(self.out_dir), ".dist_init")
            if os.path.exists(store_path):
                os.remove(store_path)  # stale rendezvous file from a previous run hangs the store
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{store_path}",
                rank=0,
                world_size=1,
            )
        from hydra.utils import instantiate

        self.data = instantiate(self.cfg.data.train, collate_fn=train_collate, _recursive_=False)

    # --- train loop -------------------------------------------------------------
    def fit(self):
        cfg_run = self.cfg.run
        max_steps = int(cfg_run.max_steps)
        log_interval = int(cfg_run.log_interval)
        img_log_interval = int(OmegaConf.select(self.cfg, "run.img_log_interval", default=0) or 0)
        val_interval = int(OmegaConf.select(self.cfg, "run.val_interval", default=0) or 0)
        ckpt_interval = int(cfg_run.ckpt_interval)

        if bool(OmegaConf.select(self.cfg, "run.val_at_start", default=False)) and val_interval:
            self._validate(self.global_step)  # pretrained / resume baseline

        self.model.train()
        while self.global_step < max_steps:
            loader = self.data.get_loader(self._epoch)
            made_progress = False
            for batch in loader:
                made_progress = True
                losses, predictions, batch, grad_norm, step_time = self._train_step(batch)
                self.global_step += 1
                step = self.global_step
                self.loss_history.append(float(losses["total"].detach()))
                if self.writer is not None and log_interval and step % log_interval == 0:
                    self._log(
                        step, losses, predictions, batch, grad_norm, step_time,
                        log_images=bool(img_log_interval and step % img_log_interval == 0),
                    )
                if val_interval and step % val_interval == 0:
                    self._validate(step)
                    self.model.train()
                if ckpt_interval and step % ckpt_interval == 0:
                    self._save(step)
                if step >= max_steps:
                    break
            if not made_progress:
                logger.warning("data loader yielded no batches; stopping early")
                break
            self._epoch += 1
        if self.writer is not None:
            self.writer.flush()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _to_device(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _train_step(self, batch):
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch = self._to_device(batch)
        images = batch["images"]
        match_on = self.loss_computer.weights.get("match", 0) > 0 and "tracks" in batch
        predictions = self.model(images, return_last_patch_tokens=match_on)
        losses = self.loss_computer(predictions, batch, tuple(images.shape[-2:]))
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.optim.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return losses, predictions, batch, grad_norm, time.time() - t0

    # --- logging ------------------------------------------------------------------
    def _log(self, step, losses, predictions, batch, grad_norm, step_time, log_images=False):
        w = self.writer
        for k in ("total", "camera", "depth", "point", "match"):
            w.add_scalar(f"train/loss_{k}", float(losses[k].detach()), step)
        w.add_scalar("train/gt_scale", float(losses["gt_scale"]), step)
        w.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], step)
        w.add_scalar("train/grad_norm", float(grad_norm), step)
        B, S = batch["images"].shape[:2]
        w.add_scalar("train/frames_per_sample", S, step)
        w.add_scalar("train/batch_size", B, step)
        w.add_scalar("perf/step_time", step_time, step)
        w.add_scalar("perf/imgs_per_sec", B * S / max(step_time, 1e-9), step)
        if self.device.type == "cuda":
            w.add_scalar("perf/peak_mem_gb", torch.cuda.max_memory_allocated(self.device) / 1e9, step)
        total = float(losses["total"].detach())
        for vendor in {str(s).split("_")[0] for s in batch.get("seq_name", []) if s}:
            w.add_scalar(f"train/loss_total_by_vendor/{vendor}", total, step)
        if log_images:
            self._log_images(step, predictions, batch, losses)

    def _log_images(self, step, predictions, batch, losses):
        def norm01(x):
            x = x - x.min()
            return (x / x.max().clamp(min=1e-8)).expand(3, -1, -1)

        rgb = batch["images"][0, 0].detach().float().cpu()
        pred_depth = predictions["depth"][0, 0, ..., 0].detach().float().cpu()
        conf = predictions["depth_conf"][0, 0].detach().float().cpu()
        gt_norm = batch["depths"][0, 0].detach().float().cpu() / max(float(losses["gt_scale"]), 1e-8)
        err = (pred_depth - gt_norm).abs()
        grid = torch.stack([rgb, norm01(pred_depth[None]), norm01(conf[None]), norm01(err[None])])
        self.writer.add_images("train/sample0_rgb_depth_conf_err", grid, step)

    # --- checkpointing --------------------------------------------------------------
    def _save(self, step):
        if self.rank != 0:
            return
        torch.save(
            self._unwrapped_model().state_dict(),
            os.path.join(self.out_dir, f"model_step{step:06d}.pt"),
        )
        state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, os.path.join(self.out_dir, f"trainer_step{step:06d}.pt"))
        keep_last = int(self.cfg.run.keep_last)
        for prefix in ("model_step", "trainer_step"):
            ckpts = sorted(glob.glob(os.path.join(self.out_dir, f"{prefix}*.pt")))
            for old in ckpts[:-keep_last] if keep_last > 0 else []:
                os.remove(old)

    def resume(self, trainer_ckpt_path: str):
        state = torch.load(trainer_ckpt_path, map_location="cpu", weights_only=False)
        model_path = trainer_ckpt_path.replace("trainer_step", "model_step")
        if os.path.exists(model_path):
            self._unwrapped_model().load_state_dict(torch.load(model_path, map_location="cpu"))
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        cuda_rng = state.get("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available() and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.global_step = int(state["step"])
        logger.info(f"resumed from {trainer_ckpt_path} at step {self.global_step}")

    # --- validation -----------------------------------------------------------------
    def _validate(self, step):
        if self.rank != 0:
            return
        # Lazy imports: `inference` registers gflags at import (train.py keeps its
        # flag names disjoint) and `evaluates` pulls in evo/matplotlib.
        import inference
        from hydra.utils import instantiate
        from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric

        model = self._unwrapped_model()
        model.eval()
        val_cfg = self.cfg.val
        for cfg_path in val_cfg.configures:
            vendor = os.path.splitext(os.path.basename(str(cfg_path)))[0]
            try:
                ycfg = OmegaConf.load(str(cfg_path))
                dataset = instantiate(ycfg.dataset, common_config=ycfg.common_config, _recursive_=False)
            except Exception as exc:
                logger.warning(f"[val {vendor}] dataset unavailable, skipping: {exc}")
                continue
            ate, rpe_rot, abs_rel, delta1 = [], [], [], []
            num_seqs = min(dataset.num_sequences(), int(val_cfg.max_sequences))
            for seq_index in range(num_seqs):
                try:
                    num_avail = dataset.sequence_num_frames(seq_index)
                    ids = (
                        np.linspace(0, num_avail - 1, min(int(val_cfg.num_frames), num_avail))
                        .round()
                        .astype(int)
                    )
                    native_h, native_w = dataset.native_image_size(seq_index)
                    sample = dataset.get_sample(
                        seq_index, ids=ids, aspect_ratio=min(native_h, native_w) / max(native_h, native_w)
                    )
                    with torch.inference_mode():
                        predictions = model(sample["images"].contiguous().to(self.device))
                    extrinsics, _ = encoding_to_camera(
                        predictions["pose_enc"], predictions["images"].shape[-2:]
                    )
                    pred_w2c = extrinsics.float().cpu().numpy()[0]
                    pred_depth = predictions["depth"].float().cpu().numpy()[0][..., 0]
                    # depth = exp(logits): an undertrained model can overflow on
                    # OOD pixels (e28+), poisoning mean abs_rel; clip far beyond
                    # any physical depth so early-training val stays readable.
                    pred_depth = np.clip(pred_depth, 0.0, 1e6)
                    modalities = set(sample.get("modalities", []))
                    if "extrinsics" in modalities and len(ids) >= 3:
                        gt_c2w = inference.world_to_camera_to_camera_to_world(
                            sample["extrinsics"].numpy()
                        )
                        pred_c2w = inference.world_to_camera_to_camera_to_world(pred_w2c)
                        res = CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run()
                        ate.append(res["ate"]["rmse"])
                        rpe_rot.append(res["rpe_rot"]["mean"])
                    if "depths" in modalities:
                        gt_depth = sample["depths"].numpy()
                        for i in range(gt_depth.shape[0]):
                            if not (gt_depth[i] > 0).any():
                                continue
                            res = MonoDepthMetric(gt_depth[i], pred_depth[i], align="median").run()
                            abs_rel.append(res["abs_rel"]["mean"])
                            delta1.append(res["delta"]["delta1"])
                except (AssertionError, ValueError) as exc:
                    # Early-training predictions can be NaN / degenerate; skip, don't crash.
                    logger.warning(f"[val {vendor}] sequence {seq_index} skipped: {exc}")
            if self.writer is not None:
                for name, values in (
                    ("ate_rmse", ate),
                    ("rpe_rot_mean", rpe_rot),
                    ("abs_rel_mean", abs_rel),
                    ("delta1", delta1),
                ):
                    if values:
                        self.writer.add_scalar(f"val/{vendor}/{name}", float(np.mean(values)), step)
