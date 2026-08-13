"""Supervised end-to-end trainer for VGGT-Omega (arXiv 2605.15195 recipe).

Owns model build (checkpoint | correct from-scratch init), optional DDP, the
step-based train loop (no outer autocast — bf16 lives inside the model; no
GradScaler), TensorBoard logging, bare-state-dict checkpoints with a trainer
sidecar, and rank-0 validation through the existing eval metrics.
"""

import glob
import math
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


def apply_grad_compression(ddp_model, name):
    """Register the configured gradient-compression hook on a DDP-wrapped model."""
    hook = resolve_comm_hook(name)
    if hook is not None:
        ddp_model.register_comm_hook(state=None, hook=hook)
    return ddp_model


SELFSUP_MODES = ("feature", "offgrid")

#: Stand-in logged for a non-finite Sim(3) diagnostic. Large enough to be
#: unmistakable on a plot, finite enough not to break the scalar stream.
_DEGENERATE_SIM3 = 1e4


def resolve_selfsup_mode(cfg) -> str:
    """Validated ``selfsup.mode``; ``"feature"`` when unset.

    ``feature`` is the paper's EMA feature/camera/depth distillation. ``offgrid``
    is the M-context / M+N-target photometric scenario: a frozen teacher supplies
    the full trajectory, the student sees only the context views.
    """
    mode = str(OmegaConf.select(cfg, "selfsup.mode", default="feature")).lower()
    if mode not in SELFSUP_MODES:
        raise ValueError(f"selfsup.mode must be one of {SELFSUP_MODES}, got {mode!r}")
    return mode


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


def load_encoder_weights(model: nn.Module, path: str) -> int:
    """Restore ONLY the DINOv3 image encoder (``aggregator.patch_embed.*``) from a
    full VGGT-Omega state dict, leaving every other block at its current init.
    Returns the number of restored tensors."""
    sd = torch.load(path, map_location="cpu")
    enc = {k: v for k, v in sd.items() if k.startswith("aggregator.patch_embed.")}
    if not enc:
        raise ValueError(f"{path!r} has no aggregator.patch_embed.* keys")
    model.load_state_dict(enc, strict=False)
    return len(enc)


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
        self.nonfinite_grad_steps = 0
        self.degenerate_splats = 0
        #: Micro-batches seen. global_step counts OPTIMIZER steps; this counts
        #: forwards, and the two differ by accumulate_steps.
        self._micro_step = 0
        # Gradient accumulation buys effective batch size that will not fit in
        # memory: each rank runs accumulate_steps forwards before one optimizer
        # step, so the effective batch is dp_size * accumulate_steps.
        self.accumulate_steps = int(OmegaConf.select(cfg, "optim.accumulate_steps", default=1))
        if self.accumulate_steps < 1:
            raise ValueError(
                f"optim.accumulate_steps must be >= 1, got {self.accumulate_steps}"
            )

        self._setup_distributed()
        # offset=0: every rank must BUILD the same model. The per-rank offset is
        # restored below, once construction is done, so data sampling still
        # diverges. See _seed_everything.
        self._seed_everything(offset=0)
        # Read before _build_model: feature-mode selfsup freezes the geometry
        # heads there, offgrid mode does not.
        self.selfsup = bool(OmegaConf.select(cfg, "selfsup.enabled", default=False))
        self.selfsup_mode = resolve_selfsup_mode(cfg)
        self.n_target = int(OmegaConf.select(cfg, "selfsup.n_target", default=4))
        # A dedicated CPU generator, re-seeded per step, so the context/target
        # split is reproducible across a resume and independent of every other
        # RNG consumer in the step.
        self._split_rng = torch.Generator()

        self.out_dir = str(cfg.run.output_dir)
        self.writer = None
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb"))

        self._build_model()
        self._build_teacher()
        if self.selfsup:
            self.loss_computer = self._build_distill_loss()
        else:
            self.loss_computer = TrainLossComputer(
                weights=OmegaConf.to_container(cfg.loss.weights),
                alpha=cfg.loss.alpha,
                temperature=cfg.loss.temperature,
                patch_size=int(OmegaConf.select(cfg, "model.patch_size", default=16)),
            )
        fused = bool(OmegaConf.select(cfg, "optim.fused", default=False)) and self.device.type == "cuda"
        scratch_lr_mult = float(OmegaConf.select(cfg, "optim.scratch_lr_mult", default=1.0))
        self.optimizer = torch.optim.AdamW(
            build_param_groups(
                self.model, cfg.optim.weight_decay,
                lr=float(cfg.optim.lr), scratch_lr_mult=scratch_lr_mult,
            ),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            fused=fused,
        )
        if scratch_lr_mult != 1.0:
            logger.info(
                f"optim: gaussian head at {float(cfg.optim.lr) * scratch_lr_mult:.2e} "
                f"({scratch_lr_mult}x the backbone's {float(cfg.optim.lr):.2e})"
            )
        self.scheduler = build_warmup_cosine(
            self.optimizer, max_steps=int(cfg.run.max_steps), warmup_frac=cfg.optim.warmup_frac
        )
        # Model construction is finished and identical on every rank; from here on
        # the RNG must DIVERGE per dp rank or every rank draws the same window and
        # the effective batch collapses to one sample. Deliberately after the
        # optimizer/scheduler too -- neither consumes randomness, but the reseed
        # belongs immediately before the first thing that must differ.
        self._seed_everything()
        self._build_data(data_override)

    # --- setup ----------------------------------------------------------------
    def _setup_distributed(self):
        self.parallel = str(OmegaConf.select(self.cfg, "optim.parallel", default="ddp")).lower()
        if self.parallel in ("cp", "cp_fsdp"):
            raise ValueError(
                f"optim.parallel={self.parallel!r}: training-side context parallelism "
                "is not ported to this repo (the sibling MUSA repo carries it). Use "
                "'fsdp' -- the frozen-backbone offgrid recipe fits dp=8 without CP."
            )
        if self.parallel not in ("ddp", "fsdp"):
            raise ValueError(f"optim.parallel must be 'ddp' or 'fsdp', got {self.parallel!r}")
        self.cp_enabled = False
        self.fsdp_enabled = self.parallel == "fsdp"
        world = int(os.environ.get("WORLD_SIZE", "1"))
        self.mesh = None
        if world > 1:
            from vggt_omega.distributed.process_group import init_distributed

            self.rank, self.world_size, self.local_rank = init_distributed()
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.rank, self.world_size, self.local_rank = 0, 1, 0
            self.device = (
                torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            )
            if self.fsdp_enabled and not dist.is_initialized():
                # FSDP needs a real process group even at world=1. Use NCCL on CUDA,
                # gloo on CPU; both as a single-rank group.
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29555")
                if self.device.type == "cuda":
                    torch.cuda.set_device(0)
                dist.init_process_group(
                    backend="nccl" if self.device.type == "cuda" else "gloo",
                    rank=0,
                    world_size=1,
                )
        # No training-side CP in this repo: every rank is a plain dp rank. The
        # dp_* names are kept (rather than collapsing onto .rank/.world_size)
        # because seeding and data sharding are defined over the dp axis -- the
        # sibling repo's (dp, cp) mesh reduces to exactly this when cp=1.
        self.cp_group = None
        self.cp_size = 1
        self.dp_size = self.world_size
        self.dp_rank = self.rank

        if self.fsdp_enabled:
            # Deliberately the FULL world, not the mesh's dp axis: params, grads
            # and optimizer state shard 1/world instead of 1/dp, which is the
            # whole point of composing the two. The (dp, cp) mesh above governs
            # only which sample a rank loads and which frames of it it owns.
            #
            # fsdp.hybrid_shard='auto' makes the same config scale across node
            # counts with no edits: single node -> plain full shard; multi-node
            # -> HSDP, sharding within each node (fsdp.shard_size, default =
            # LOCAL_WORLD_SIZE) and replicating across nodes so only the
            # once-per-step gradient reduce crosses the inter-node link.
            from vggt_omega.training.parallel import build_dp_mesh, resolve_hybrid_shard

            local_world = os.environ.get("LOCAL_WORLD_SIZE")
            hybrid = resolve_hybrid_shard(
                OmegaConf.select(self.cfg, "fsdp.hybrid_shard", default=False),
                world_size=self.world_size,
                local_world_size=int(local_world) if local_world else None,
            )
            self.mesh = build_dp_mesh(
                self.world_size,
                hybrid_shard=hybrid,
                shard_size=OmegaConf.select(self.cfg, "fsdp.shard_size", default=None),
                device_type=self.device.type,
            )
            if self.rank == 0:
                # build_dp_mesh degrades hybrid to the 1-D mesh when everything
                # fits one shard group, so judge by the mesh, not the flag.
                if self.mesh.ndim == 2:
                    replicate, shard = tuple(self.mesh.shape)
                    logger.info(
                        f"FSDP topology: HSDP replicate={replicate} x shard={shard} -- "
                        f"params shard over {shard}-rank groups (intra-node), gradients "
                        f"all-reduce across {replicate} groups (inter-node)"
                    )
                else:
                    logger.info(f"FSDP topology: full shard over {self.world_size} ranks")
        # The dataloader worker_init_fn reads RANK unconditionally — set before any build.
        os.environ.setdefault("RANK", "0")

    def _seed_everything(self, offset=None):
        """Seed the global RNGs. ``offset=0`` for anything that must agree across
        ranks; the default ``dp_rank`` offset for anything that must differ.

        Two different requirements, and conflating them was a bug:

        * MODEL CONSTRUCTION must be IDENTICAL on every rank. Data-parallel
          training assumes all ranks start from the same weights, and FSDP shards
          whatever each rank happens to hold -- if the ranks disagree, the
          assembled model is a mix of unrelated initialisations and no collective
          ever corrects it. This matters here because the released checkpoint has
          no gaussian head: ``gs_dpt_head.*`` and ``gs_decoder.*`` (64 tensors)
          start from random init, so a per-rank seed gave every dp rank a
          different gaussian head. The pretrained backbone was unaffected -- it is
          overwritten by the checkpoint load -- which is exactly why this stayed
          invisible.
        * DATA SAMPLING must DIFFER per dp rank, or every rank trains on the same
          window and the effective batch is one sample wearing dp_size hats.

        dp_rank is the data-parallel rank; without training-side CP (not ported
        here) it equals the global rank, and the name is kept so the seeding
        contract reads the same as in the sibling repo's (dp, cp) topology.
        """
        seed = int(self.cfg.run.seed) + (self.dp_rank if offset is None else int(offset))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _build_model(self):
        enable_3dgs = bool(OmegaConf.select(self.cfg, "model.enable_3dgs", default=False))
        sel = lambda k, d: OmegaConf.select(self.cfg, k, default=d)
        arch = dict(
            embed_dim=int(self.cfg.model.embed_dim),
            enable_3dgs=enable_3dgs,
            gs_sh_degree=int(sel("model.gs_sh_degree", 4)),
            gs_opacity_initial=float(sel("model.gs_opacity_initial", 0.0)),
            gs_opacity_final=float(sel("model.gs_opacity_final", 0.0)),
            gs_opacity_warm_up=int(sel("model.gs_opacity_warm_up", 1)),
            # "softplus" (default) keeps the fov gradient alive below zero;
            # "relu" restores the original absorbing floor for reproducing runs
            # trained before the fix. The teacher is built from the same key
            # (teacher._resolve_arch) so the anchor compares like with like.
            fov_activation=str(sel("model.fov_activation", "softplus")),
        )
        model = VGGTOmega(**arch)
        if self.cfg.model.checkpoint:
            state = torch.load(self.cfg.model.checkpoint, map_location="cpu")
            # The released checkpoint predates the gaussian head, so its
            # gs_dpt_head.* keys are missing by design and start from the head's
            # own init. Unexpected keys still mean the wrong checkpoint.
            incompatible = model.load_state_dict(state, strict=not enable_3dgs)
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"checkpoint {self.cfg.model.checkpoint} has "
                    f"{len(incompatible.unexpected_keys)} unexpected keys, e.g. "
                    f"{list(incompatible.unexpected_keys)[:3]}"
                )
            if incompatible.missing_keys:
                logger.info(
                    f"checkpoint missing {len(incompatible.missing_keys)} keys "
                    f"(gaussian head starts from init), e.g. {list(incompatible.missing_keys)[:3]}"
                )
        else:
            init_model_from_scratch(model)
            encoder_src = OmegaConf.select(self.cfg, "model.init_encoder_from")
            if encoder_src:
                n = load_encoder_weights(model, str(encoder_src))
                logger.info(f"restored {n} encoder tensors from {encoder_src}")
        model.aggregator.gradient_checkpointing = bool(self.cfg.model.gradient_checkpointing)
        model = model.to(self.device)
        if self.selfsup and self.selfsup_mode == "feature":
            # Paper Sec. 3.4: freeze the geometry heads during feature-mode
            # self-supervision so only the backbone adapts. Must precede the
            # FSDP/DDP wrap below. Offgrid mode trains the full student instead.
            #
            # (An earlier version of this comment claimed offgrid was safe because
            # "the render cameras come from the frozen teacher". That is true only
            # of the TARGET render; the context render uses the student's own
            # cameras. What limits fov collapse there is the teacher anchor, plus
            # optionally selfsup.detach_render_camera -- not the teacher's poses.)
            from vggt_omega.training.selfsup import freeze_geometry_heads

            trainable, frozen = freeze_geometry_heads(model)
            logger.info(
                f"selfsup: froze geometry heads -> {trainable / 1e6:.1f}M trainable / "
                f"{frozen / 1e6:.1f}M frozen (aggregator backbone only)"
            )
        elif bool(OmegaConf.select(self.cfg, "model.freeze_backbone", default=False)):
            # Everything but the gaussian head is frozen, so the student's pose
            # and depth stay bit-identical to the teacher's and the Sim(3) fit
            # cannot drift out of the refuse gate. Also precedes the FSDP wrap.
            from vggt_omega.training.selfsup import freeze_backbone

            trainable, frozen = freeze_backbone(model)
            logger.info(
                f"froze backbone -> {trainable / 1e6:.1f}M trainable / "
                f"{frozen / 1e6:.1f}M frozen (gaussian head + decoder only)"
            )
        if self.fsdp_enabled:
            # optim.grad_compression is a DDP comm hook; under FSDP, the gradient-reduce dtype is fsdp.reduce_dtype instead.
            from vggt_omega.training.parallel import apply_fsdp

            apply_fsdp(
                model,
                self.mesh,
                reshard_after_forward=bool(
                    OmegaConf.select(self.cfg, "fsdp.reshard_after_forward", default=True)
                ),
                param_dtype=str(OmegaConf.select(self.cfg, "fsdp.param_dtype", default="none")),
                reduce_dtype=str(OmegaConf.select(self.cfg, "fsdp.reduce_dtype", default="bfloat16")),
                cpu_offload=bool(OmegaConf.select(self.cfg, "fsdp.cpu_offload", default=False)),
            )
        elif self.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            apply_grad_compression(
                model, OmegaConf.select(self.cfg, "optim.grad_compression", default="none")
            )
        self.model = model

    def _build_teacher(self):
        """Frozen pseudo-label model, or None. Built after the student so the
        shard='none' fit check sees the memory the student already took.

        Deliberately not wrapped in DDP/FSDP by the trainer: it holds no
        gradients, so there is nothing to synchronise. It does its own sharding
        internally when the checkpoint is too large to replicate.
        """
        from vggt_omega.training.teacher import build_teacher

        self.teacher = build_teacher(
            self.cfg, self.device, self.world_size, self.rank,
            mesh=self.mesh, student=self.model,
        )
        # Cached so the hot loop's EMA check is a bare attribute read.
        self.ema_teacher = self.teacher if (self.teacher is not None and self.teacher.ema) else None

    def _build_distill_loss(self):
        """Self-supervised distillation loss + augmentation config (Paper Sec. 3.4).

        Requires an EMA teacher: the targets are the moving-average student, so a
        frozen or absent teacher would have nothing to distill toward.
        """
        if self.selfsup_mode == "offgrid":
            return self._build_offgrid_loss()
        from types import SimpleNamespace

        from vggt_omega.training.selfsup import DistillLossComputer

        if self.ema_teacher is None:
            raise ValueError(
                "selfsup.enabled needs an EMA teacher; set teacher.mode: ema (and a "
                "teacher.checkpoint or it seeds from the student)."
            )
        sel = lambda k, d: OmegaConf.select(self.cfg, k, default=d)
        self.aug_cfg = SimpleNamespace(
            brightness=float(sel("selfsup.aug.brightness", 0.4)),
            contrast=float(sel("selfsup.aug.contrast", 0.4)),
            saturation=float(sel("selfsup.aug.saturation", 0.4)),
            mask_ratio=float(sel("selfsup.aug.mask_ratio", 0.3)),
            mask_patch=int(sel("selfsup.aug.mask_patch", 16)),
        )
        weights = OmegaConf.select(self.cfg, "selfsup.weights", default=None)
        weights = OmegaConf.to_container(weights) if weights is not None else {
            "feature": 1.0, "camera": 1.0, "depth": 1.0
        }
        return DistillLossComputer(
            weights=weights,
            conf_weighted_depth=bool(sel("selfsup.conf_weighted_depth", True)),
        )

    def _build_offgrid_loss(self):
        """Photometric + teacher-anchor loss for the off-the-grid M+N scenario.

        Unlike feature-mode distillation the photometric targets are the *real*
        pixels; the teacher supplies the M+N trajectory the held-out renders are
        posed on, and — over the M context views — the camera and depth the
        student is regressed onto. A frozen teacher is the intended setup: being
        frozen it cannot drift to a degenerate camera, which is the failure mode a
        jointly-optimised camera head falls into under photometric-only
        supervision (it shrinks the fov until every splat projects into a
        memorised smudge).

        Also reads ``selfsup.gate.*`` into ``self.gate_cfg`` for the step to use.
        """
        from vggt_omega.training.selfsup import OffGridLossComputer

        if self.teacher is None:
            raise ValueError(
                "selfsup.mode=offgrid needs a teacher to supply the M+N trajectory; set "
                "teacher.mode: frozen and teacher.checkpoint (teacher.mode: ema also works)."
            )
        if not bool(OmegaConf.select(self.cfg, "model.enable_3dgs", default=False)):
            raise ValueError(
                "selfsup.mode=offgrid renders the student's splats, so it needs the "
                "gaussian head; set model.enable_3dgs: true."
            )
        weights = OmegaConf.select(self.cfg, "selfsup.weights", default=None)
        weights = OmegaConf.to_container(weights) if weights is not None else {
            "l1": 0.8, "l2": 0.0, "ssim": 0.2, "lpips": 0.0
        }
        distill = OmegaConf.select(self.cfg, "selfsup.distill", default=None)
        distill = OmegaConf.to_container(distill) if distill is not None else {}
        # None disables a check; the defaults are deliberately loose, because a
        # gate that fires on every step silently turns this back into a
        # context-only run. Watch train/gate_rate and tighten from there.
        gate = OmegaConf.select(self.cfg, "selfsup.gate", default=None)
        gate = OmegaConf.to_container(gate) if gate is not None else {}
        self.gate_cfg = {
            "max_scale_log": gate.get("max_scale_log", 0.5),
            "max_rotation_deg": gate.get("max_rotation_deg", 20.0),
            "max_residual": gate.get("max_residual", 0.15),
        }
        # Whether the context render's VIEW MATRIX carries gradient. Default false
        # = every student gradient path is live (see _offgrid_step for what each
        # setting means for the objective). Set true to restore the older damped
        # behaviour, which trades a correct gradient for resistance to gauge drift.
        self.detach_render_camera = bool(
            OmegaConf.select(self.cfg, "selfsup.detach_render_camera", default=False)
        )
        return OffGridLossComputer(
            weights=weights,
            lpips_net=str(OmegaConf.select(self.cfg, "selfsup.lpips_net", default="vgg")),
            lpips_chunk_size=OmegaConf.select(self.cfg, "selfsup.lpips_chunk_size", default=None),
            ssim_backend=str(OmegaConf.select(self.cfg, "selfsup.ssim_backend", default="auto")),
            target_weight=float(OmegaConf.select(self.cfg, "selfsup.target_weight", default=1.0)),
            camera_weight=float(distill.get("camera", 1.0)),
            depth_weight=float(distill.get("depth", 1.0)),
            # Off by default: it constrains the RENDERED geometry rather than the
            # dense head's prediction, which is a different (and stronger) claim,
            # so existing configs keep their behaviour until they ask for it.
            render_depth_weight=float(distill.get("render_depth", 0.0)),
            render_depth_min_alpha=float(
                OmegaConf.select(self.cfg, "selfsup.render_depth_min_alpha", default=0.5)
            ),
            conf_weighted_depth=bool(
                OmegaConf.select(self.cfg, "selfsup.conf_weighted_depth", default=True)
            ),
        )

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _full_state_dict(self):
        """Consolidated, plain (non-DTensor) model state dict. Collective under FSDP."""
        if self.fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

            return get_model_state_dict(
                self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
            )
        return self._unwrapped_model().state_dict()

    def _full_optim_state_dict(self):
        """Consolidated optimizer state dict. Collective under FSDP."""
        if self.fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions

            return get_optimizer_state_dict(
                self.model, self.optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        return self.optimizer.state_dict()

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

        # Shard the dataset over the dp axis (== the world here; kept as dp_size/
        # dp_rank so the contract matches the sibling repo's (dp, cp) topology).
        self.data = instantiate(
            self.cfg.data.train,
            collate_fn=train_collate,
            num_replicas=self.dp_size,
            rank=self.dp_rank,
            _recursive_=False,
        )

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
                # Held only so _dump_crash_scene can write the offending sample
                # if this step goes non-finite. Rebound every micro-batch, so it
                # keeps nothing alive beyond the current one.
                self._last_batch = batch
                # A micro-batch. Only every accumulate_steps-th one closes an
                # optimizer step, and global_step counts THOSE -- the LR schedule
                # is defined over optimizer steps, not forwards.
                self._micro_step += 1
                apply_update = self._micro_step % self.accumulate_steps == 0
                losses, predictions, batch, grad_norm, step_time = self._train_step(
                    batch, apply_update=apply_update
                )
                if not apply_update:
                    continue
                self.global_step += 1
                step = self.global_step
                self.loss_history.append(float(losses["total"].detach()))
                if self.rank == 0 and log_interval and step % log_interval == 0:
                    if not self.selfsup:
                        term_keys = ("camera", "depth", "point", "match")
                    elif self.selfsup_mode == "offgrid":
                        # sim3_residual on the console, not just TensorBoard: it is
                        # what decides whether the held-out views are scored at all,
                        # and a run drifting toward the gate threshold is the thing
                        # you want to see while it happens rather than afterwards.
                        # ssim/lpips included: they are computed and weighted like
                        # the rest, and leaving them off the console made a NaN in
                        # photo_context unattributable -- l1 read finite while the
                        # weighted sum was nan, and the two terms that could have
                        # caused it were the two not being printed.
                        term_keys = (
                            "photo_context", "photo_target", "l1", "l2", "ssim", "lpips",
                            "render_depth", "camera", "depth", "gate_rate", "sim3_residual",
                            "fov_min",
                        )
                    else:
                        term_keys = ("feature", "camera", "depth")
                    terms = " ".join(
                        f"| {k} {float(losses[k].detach()):.4f} " for k in term_keys
                    )
                    logger.info(
                        f"step {step}/{max_steps} | total {float(losses['total'].detach()):.4f} "
                        f"{terms}| lr {self.optimizer.param_groups[0]['lr']:.2e} | {step_time:.2f}s"
                    )
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

    def _step_optimizer(self, grad_norm: float) -> None:
        """Apply the optimizer (and EMA) update unless the gradient is non-finite.

        ``clip_grad_norm_`` scales every gradient by ``max_norm / total_norm``, so
        one NaN or Inf anywhere makes the whole model NaN in a single step instead
        of being clipped away. gsplat's backward produces non-finite gradients on
        degenerate splats, so the update has to be dropped on a bad batch rather
        than poisoning every step after it.

        The scheduler still advances: the LR schedule is a function of step count,
        not of how many steps landed.
        """
        if math.isfinite(grad_norm):
            self.optimizer.step()
            if self.ema_teacher is not None:
                # After the step, so the average tracks the weights that were just
                # written. Shard-local, so this adds no collective and every rank
                # does the same work.
                self.ema_teacher.update(self.model, self.global_step)
        else:
            self.nonfinite_grad_steps += 1
            if self.rank == 0:
                logger.warning(
                    f"step {self.global_step}: non-finite grad norm ({grad_norm}); skipping "
                    f"the update ({self.nonfinite_grad_steps} skipped so far)"
                )
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def _selfsup_step(self, batch, apply_update: bool = True):
        """One self-supervised distillation step (Paper Sec. 3.4): two independent
        augmentations of the same frames, student (grad) vs EMA teacher (no-grad),
        feature + camera + depth distillation, then the EMA weight update."""
        from vggt_omega.training.parallel import grad_norm_to_float
        from vggt_omega.training.selfsup import augment_two_views

        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch = self._to_device(batch)
        images = batch["images"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        view_s, view_t = augment_two_views(images, self.aug_cfg)
        student = self.model(view_s, return_features=True)  # grad; heads frozen inside
        teacher = self.teacher(view_t, return_features=True)  # Teacher.forward is @no_grad
        losses = self.loss_computer(student, teacher)
        grad_norm = self._finish_micro_step(losses["total"], apply_update)
        return losses, student, batch, grad_norm, time.time() - t0

    def _offgrid_step(self, batch, apply_update: bool = True):
        """One off-the-grid distillation step.

        The student sees only the M context views and decodes a fused 3DGS scene
        from them. The frozen teacher is run twice, on two view sets, for two
        jobs that must not be conflated:

        * **over the M context views** — the same frames the student got, so the
          two predictions are in the same gauge and directly comparable. This is
          what the camera and depth regression anchors to.
        * **over all M+N views** — a different, better-informed reconstruction,
          and therefore in a *different* gauge. Its value is the N target poses,
          which is why it is Sim(3)-aligned onto the student over the M-view
          overlap before being used. Distilling the student onto *this* pass's
          context poses would fight that alignment, which is why the anchor comes
          from the M-view pass instead.

        The M context views are then rendered at the student's own cameras (a
        self-render, independent of the Sim(3)) and the N target views at the
        aligned teacher poses, both against the real frames — so the target views
        supervise geometry the student was never shown. Target views whose Sim(3)
        did not survive :func:`sim3_refuse_gate` are dropped for that batch item;
        the context term always stands.
        """
        from gaussian_splat.render.render_by_gsplat import render_by_gsplat
        from vggt_omega.training.parallel import grad_norm_to_float
        from vggt_omega.training.selfsup import (
            align_teacher_trajectory,
            sanitize_gaussians,
            sim3_refuse_gate,
            split_context_target,
        )

        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch = self._to_device(batch)
        images = batch["images"]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        image_hw = tuple(images.shape[-2:])
        rasterize_mode = str(
            OmegaConf.select(self.cfg, "selfsup.rasterize_mode", default="classic")
        )

        # Re-seeded per step (not per process) so a resumed run replays the same
        # splits it would have drawn had it never stopped.
        self._split_rng.manual_seed(int(self.cfg.run.seed) + self.global_step)
        context_idx, target_idx = split_context_target(
            images.shape[1], self.n_target, generator=self._split_rng
        )
        context_idx = context_idx.to(images.device)
        target_idx = target_idx.to(images.device)
        context_images = images[:, context_idx]

        student_input = context_images
        teacher_all = self.teacher(images)  # no-grad, all M+N views: the trajectory
        # Skipped when both anchor weights are zero -- it is a full extra forward.
        # The anchor compares the student against the teacher run on the SAME
        # context views, so the two are in the same gauge by construction (VGGT
        # expresses pose relative to the first frame of its input).
        teacher_context = None
        if self.loss_computer.distills:
            teacher_context = self.teacher(context_images)
        student = self.model(  # grad; this rank's shard of the M context views
            student_input, decode_gaussians=True, global_step=self.global_step
        )

        teacher_ext, teacher_k = encoding_to_camera(teacher_all["pose_enc"], image_hw)
        # pose_enc_full is the sibling repo's CP-gathered name; without CP the
        # plain pose_enc is already the whole context trajectory.
        student_pose_full = student.get("pose_enc_full", student["pose_enc"])
        # Two consumers, two different requirements, so build both explicitly.
        #
        # The ALIGNMENT never needs gradient: align_teacher_trajectory is
        # @torch.no_grad and its output is a per-step constant, so this detach is
        # a statement of intent rather than a restriction.
        align_ext, _ = encoding_to_camera(student_pose_full.detach(), image_hw)
        alignment = align_teacher_trajectory(teacher_ext, align_ext, context_idx)
        keep = sim3_refuse_gate(alignment, **self.gate_cfg)

        # Before any rasterization: a splat with a non-finite mean corrupts the
        # rasterizer backward (and on some backends crashes it outright).
        gaussians, degenerate = sanitize_gaussians(student["gaussians"])
        self.degenerate_splats += degenerate
        if degenerate and self.rank == 0:
            logger.warning(
                f"step {self.global_step}: neutralized {degenerate} non-finite splats "
                f"({self.degenerate_splats} total)"
            )

        # The RENDER VIEWPOINT is a real modelling choice, so it is config-driven
        # (selfsup.detach_render_camera) rather than hard-coded.
        #
        # Attached (the default): the context render is a function of the cameras
        # both through the splat unprojection AND through the view matrix, which
        # is the true photometric objective -- the student is fully trainable and
        # every gradient path the model has is live.
        #
        # Detached: the viewpoint is frozen at this step's value while the splats
        # still move with the cameras. That is a surrogate objective, and it is
        # not neutral -- it makes a coherent shift of the cameras cost image
        # error, which damps gauge drift. It also makes the fov component pay for
        # shrinking, which is why it was the original behaviour. The cost is that
        # the gradient no longer corresponds to any objective, and the camera head
        # is pushed toward "do not move" rather than "be correct".
        #
        # Note the two are NOT equivalent even at identical numeric poses: with
        # the viewpoint attached a global similarity applied to cameras and splats
        # together leaves the render unchanged, so that direction is exactly flat
        # and only the teacher anchor opposes it.
        student_ext, student_k = encoding_to_camera(
            student_pose_full.detach() if self.detach_render_camera else student_pose_full,
            image_hw,
        )
        context_truth = context_images
        render_ext, render_k = student_ext, student_k
        context_render = render_by_gsplat(
            gaussians, render_ext, render_k, image_hw, rasterize_mode=rasterize_mode
        )
        rendered_context = context_render.color
        self._log_nonfinite_render(context_render, gaussians, render_ext, "context")
        render_depth = (
            (context_render.depth, context_render.alpha, teacher_context["depth"])
            if teacher_context is not None and self.loss_computer.render_depth_weight > 0.0
            else None
        )
        target = None
        target_slice = target_idx
        if bool(keep.any()) and target_slice.numel():
            # Rendered for every item, scored only for the kept ones: subsetting the
            # gaussians by batch item would cost more than the views it saves. The
            # all-refused case skips the render entirely, which is the case that matters.
            target_render = render_by_gsplat(
                gaussians,
                alignment.extrinsics[:, target_slice],
                teacher_k[:, target_slice],
                image_hw,
                rasterize_mode=rasterize_mode,
            )
            self._log_nonfinite_render(
                target_render, gaussians, alignment.extrinsics[:, target_slice], "target"
            )
            target = (target_render.color, images[:, target_slice], keep)

        losses = self.loss_computer(
            context=(rendered_context, context_truth),
            target=target,
            distill=None if teacher_context is None else (student, teacher_context),
            render_depth=render_depth,
        )
        losses.update(self._alignment_diagnostics(alignment, keep))
        losses.update(self._fov_diagnostics(student_pose_full))
        # Attribute a non-finite total to the term that caused it, once, at the
        # point where the terms still exist. The weighted sum is all the guard in
        # _finish_micro_step can see, and "total is nan" does not say whether the
        # render went bad (l1/l2 follow it) or only the perceptual term did --
        # LPIPS returns nan on a single inf pixel while l1 merely returns inf.
        if not torch.isfinite(losses["total"]):
            bad = {
                key: float(value.detach())
                for key, value in losses.items()
                if torch.is_tensor(value) and not torch.isfinite(value)
            }
            render_finite = bool(torch.isfinite(rendered_context).all())

            def _fin(tensor):
                return "n/a" if tensor is None else str(bool(torch.isfinite(tensor).all()))

            # Which SIDE of the camera anchor is bad matters: the teacher is
            # frozen and no-grad, so a nan there is an input/sample problem,
            # while a nan only on the student side is the trainable path.
            logger.warning(
                f"step {self.global_step} rank {self.rank}: non-finite terms {bad}; "
                f"context render finite={render_finite}, "
                f"gaussian means finite={bool(torch.isfinite(gaussians.means).all())}, "
                f"student pose_enc finite={_fin(student.get('pose_enc'))}, "
                f"student pose_enc_full finite={_fin(student.get('pose_enc_full'))}, "
                f"teacher_ctx pose_enc finite="
                f"{_fin(None if teacher_context is None else teacher_context.get('pose_enc'))}, "
                f"teacher_ctx depth finite="
                f"{_fin(None if teacher_context is None else teacher_context.get('depth'))}, "
                f"student depth finite={_fin(student.get('depth'))}, "
                f"images finite={bool(torch.isfinite(images).all())}"
            )
        grad_norm = self._finish_micro_step(losses["total"], apply_update)
        return losses, student, batch, grad_norm, time.time() - t0

    def _log_nonfinite_render(self, render, gaussians, extrinsics, which: str) -> None:
        """Report a render the rasterizer returned non-finite, and why it might be.

        render_by_gsplat now neutralises those pixels, so they no longer reach the
        loss and no longer cost the whole CP group its step. That is the fix, but it
        also means the failure is otherwise INVISIBLE -- hence this.

        The two numbers logged are chosen to separate the competing explanations,
        because the obvious one does not survive scrutiny. Plain SH-magnitude
        overflow yields inf, and the clamp in render_by_gsplat already absorbed inf
        long before this was a problem; the renders come back NAN, which needs a
        cancellation. So:

        * ``sh_max`` -- the largest harmonic coefficient. GSDecoder leaves SH
          unbounded (every other gaussian attribute is squashed), so two huge
          opposite-sign colours summing to inf + (-inf) = nan in the tile
          accumulator is one candidate. A large value here implicates it.
        * ``min_cam_dist`` -- the closest splat-to-camera-centre distance. gsplat
          normalises the view direction, and a splat AT a camera centre gives the
          zero vector, then rsqrtf(0) = inf and 0 * inf = nan inside the SH kernel.
          near_plane is 1e-10 (AnySplat's value), which does not cull it. A value
          near zero here implicates that instead, and it is independent of SH.

        Whichever is implicated decides the real fix; guessing between them is how
        this bug survived several runs.
        """
        count = render.nonfinite_pixels
        if not count:
            return
        self.nonfinite_render_pixels = getattr(self, "nonfinite_render_pixels", 0) + count
        self.nonfinite_render_steps = getattr(self, "nonfinite_render_steps", 0) + 1
        with torch.no_grad():
            sh_max = float(gaussians.harmonics.abs().max())
            scale_min = float(gaussians.scales.min())
            # Camera centre of a world-from-camera inverse: for extrinsics that map
            # world -> camera, the centre is -R^T t.
            ext = extrinsics.reshape(-1, *extrinsics.shape[-2:])[:, :3, :]
            centres = -torch.einsum("vji,vj->vi", ext[:, :, :3], ext[:, :, 3])
            means = gaussians.means.reshape(-1, 3)
            if means.numel() and centres.numel():
                min_dist = float(torch.cdist(centres[:, None, :], means[None]).min())
                mean_absmax = float(means.abs().max())
            else:
                min_dist = mean_absmax = float("nan")
            ext_finite = bool(torch.isfinite(extrinsics).all())
        logger.warning(
            f"step {self.global_step} rank {self.rank} [{which}]: rasterizer returned "
            f"{count} non-finite px (color {render.nonfinite_color_px}, "
            f"depth {render.nonfinite_depth_px}) of {render.color[0, 0, 0].numel()} per view; "
            f"sh_max {sh_max:.3e}, scale_min {scale_min:.3e}, mean_absmax {mean_absmax:.3e}, "
            f"min_cam_dist {min_dist:.3e}, ext_finite {ext_finite} "
            f"({self.nonfinite_render_steps} steps, {self.nonfinite_render_pixels} px total)"
        )

    @staticmethod
    def _fov_diagnostics(pose_enc):
        """Predicted field of view, as loggable scalars.

        Nothing recorded this before, which meant the fov collapse the configs
        warn about ("memorised dust on pencil cameras") was invisible while it
        happened and only diagnosable afterwards from splat scales. ``fov_min`` is
        the number that matters: the focal is ``(H/2)/tan(fov/2)``, so small fov is
        a telephoto, and a camera pinned near FOV_MIN is a pencil camera. For
        scale, the pretrained checkpoint measures 0.61-1.21 rad on mip-NeRF-360
        (verify/fov_range_probe.py), i.e. ~61x the 0.01 floor.

        Diagnostics only -- detached, nothing backpropagates through them.
        """
        fov = pose_enc[..., 7:].detach().float()
        return {"fov_min": fov.min(), "fov_mean": fov.mean()}

    @staticmethod
    def _alignment_diagnostics(alignment, keep):
        """Sim(3) health as loggable scalars, alongside the losses.

        ``gate_rate`` is the fraction of batch items whose target views survived;
        a run sitting near zero is a context-only run wearing an off-the-grid
        config, and is the thing to notice. The three Sim(3) magnitudes say *why*
        it refused. They are diagnostics, not losses: nothing backpropagates
        through them.

        Infinities (a degenerate fit) are mapped to a large finite sentinel so a
        single bad step cannot poison the scalar stream — the value is meaningless
        as a magnitude, and unambiguous as "refused".
        """
        finite = lambda x: torch.nan_to_num(x.float().mean(), nan=_DEGENERATE_SIM3, posinf=_DEGENERATE_SIM3)
        return {
            "gate_rate": keep.float().mean(),
            "sim3_scale_log": finite(alignment.scale_log),
            "sim3_rotation_deg": finite(alignment.rotation_deg),
            "sim3_residual": finite(alignment.residual),
        }

    def _dump_crash_scene(self, loss):
        """Preserve the FIRST non-finite step's inputs, once per run, per rank.

        The guards around this one keep training alive through a non-finite
        step, which is what a long run needs -- but it also means the evidence
        scrolls past and the offending sample is gone. Three 2-node runs died on
        the same ranks and there was nothing on disk afterwards to inspect.

        Only the first event is written, and only by the ranks that actually saw
        a non-finite loss locally. The vote in the caller is a group MIN, so
        every rank enters that branch when any one of them is bad; dumping from
        all of them would write 16 copies of mostly-innocent batches and hide
        which rank was the source. Small (the batch, not the model) and never
        overwritten, so the primary crash scene survives whatever the run does
        afterwards.
        """
        if getattr(self, "_crash_dumped", False):
            return
        if bool(torch.isfinite(loss)):
            return  # this rank is fine; another rank tripped the group vote
        self._crash_dumped = True
        batch = getattr(self, "_last_batch", None)
        if batch is None:
            return
        path = os.path.join(self.out_dir, f"crash_scene_step{self.global_step:06d}_rank{self.rank}.pt")
        try:
            payload = {
                "step": self.global_step,
                "rank": self.rank,
                "dp_rank": self.dp_rank,
                "loss": float(loss.detach()),
                "seq_name": batch.get("seq_name"),
                "ids": batch.get("ids"),
                "images": batch["images"].detach().cpu(),
            }
            torch.save(payload, path)
            logger.warning(
                f"crash scene saved: {path} (step {self.global_step}, rank {self.rank}, "
                f"dp_rank {self.dp_rank}, scene {batch.get('seq_name')}). Replay it "
                "against the checkpoint nearest this step."
            )
        except Exception as exc:  # never let debugging aid kill the run
            logger.warning(f"crash scene dump failed ({type(exc).__name__}: {exc}); training continues")

    def _finish_micro_step(self, loss, apply_update: bool):
        """Backward one micro-batch, and update only on an accumulation boundary.

        The loss is divided by ``accumulate_steps`` so the accumulated gradient is
        the MEAN over micro-batches -- i.e. exactly the gradient a single batch of
        that size would produce, which is the point of accumulating rather than
        just stepping more often.

        Everything else is deferred to the boundary, and each for a reason:

        * clipping, because clipping each micro-gradient and then adding them
          bounds nothing (N clipped gradients can sum to N times the limit);
        * the optimizer and scheduler, because ``global_step`` counts OPTIMIZER
          steps -- the warmup-cosine schedule is built over ``run.max_steps``, so
          advancing it per micro-batch would compress the whole schedule by N.

        Returns the grad norm, or NaN on a non-boundary micro-step where no norm
        has been computed. NaN rather than 0.0 so it cannot be mistaken for a
        real measurement if it ever reaches a log.
        """
        from vggt_omega.training.parallel import grad_norm_to_float

        # A non-finite loss must not reach backward. It is not merely a wasted
        # step: autograd propagates the nan into every parameter gradient, the
        # optimizer writes it into the weights, and from then on the run produces
        # nan forever while still "training".
        #
        # Skipping keeps the gradients already accumulated from earlier
        # micro-steps and lets the run continue, which is right when the cause is
        # one pathological sample. A run where it keeps happening is not
        # recoverable and should stop rather than quietly train on a fraction of
        # its data, hence the consecutive-failure ceiling.
        # The decision MUST be unanimous. backward() under FSDP fires per-block
        # all-gathers and a reduce-scatter; a rank that returns early stops
        # contributing to collectives its peers are already blocking on, and the
        # step either deadlocks or completes with a reduction missing a term. So
        # every rank votes and the group skips together -- one rank's bad sample
        # costs the whole group its step, which is the cheap half of the trade.
        finite = torch.tensor(
            [float(torch.isfinite(loss))], device=loss.device, dtype=torch.float32
        )
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() == 0.0:
            self.nonfinite_losses = getattr(self, "nonfinite_losses", 0) + 1
            self._dump_crash_scene(loss)
            consecutive = getattr(self, "_consecutive_nonfinite", 0) + 1
            self._consecutive_nonfinite = consecutive
            mine = "this rank" if not torch.isfinite(loss) else "another rank"
            logger.warning(
                f"step {self.global_step}: non-finite loss ({float(loss.detach())}) "
                f"on {mine}, whole group skipping backward "
                f"({self.nonfinite_losses} total, {consecutive} in a row)"
            )
            if consecutive >= 20:
                raise RuntimeError(
                    f"{consecutive} consecutive non-finite losses -- the run is not "
                    "recovering. Check the photometric terms (ssim/lpips are logged) "
                    "and the render for non-finite pixels."
                )
            # Zero the SEED rather than returning early. Returning skipped
            # backward, and backward is what frees the graph: the caller still
            # holds `losses`/`predictions`, so the graph stayed alive while the
            # next forward built another one. Two live graphs is roughly double
            # the activations, and the run OOM'd after four skipped steps.
            #
            # nan_to_num's backward passes gradient only where the input was
            # finite, so a zeroed seed contributes nothing from this rank while
            # every collective still fires in step with its peers.
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._consecutive_nonfinite = 0

        (loss / self.accumulate_steps).backward()
        # A zero seed does not guarantee finite gradients: 0 * nan is still nan,
        # so a nan sitting in an INTERMEDIATE activation reaches the parameters
        # regardless of what the seed was. Scrub them before they can be reduced
        # or written into the weights -- one nan gradient poisons a parameter
        # permanently, and the run keeps "training" with it.
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if not apply_update:
            return float("nan")
        grad_norm = grad_norm_to_float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        )
        self._step_optimizer(grad_norm)
        return grad_norm

    def _train_step(self, batch, apply_update: bool = True):
        if self.selfsup:
            if self.selfsup_mode == "offgrid":
                return self._offgrid_step(batch, apply_update)
            return self._selfsup_step(batch, apply_update)
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch = self._to_device(batch)
        images = batch["images"]
        match_on = self.loss_computer.weights.get("match", 0) > 0 and "tracks" in batch
        predictions = self.model(images, return_last_patch_tokens=match_on)
        losses = self.loss_computer(predictions, batch, tuple(images.shape[-2:]))
        grad_norm = self._finish_micro_step(losses["total"], apply_update)
        return losses, predictions, batch, grad_norm, time.time() - t0

    # --- logging ------------------------------------------------------------------
    def _log(self, step, losses, predictions, batch, grad_norm, step_time, log_images=False):
        w = self.writer
        if not self.selfsup:
            keys = ("total", "camera", "depth", "point", "match")
        elif self.selfsup_mode == "offgrid":
            keys = (
                "total", "l1", "l2", "ssim", "lpips",
                "photo_context", "photo_target", "camera", "depth", "render_depth",
                "gate_rate", "sim3_scale_log", "sim3_rotation_deg", "sim3_residual",
                "fov_min", "fov_mean",
            )
        else:
            keys = ("total", "feature", "camera", "depth")
        for k in keys:
            w.add_scalar(f"train/loss_{k}", float(losses[k].detach()), step)
        if not self.selfsup:
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
        if log_images and not self.selfsup:
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
        # Gather BEFORE the rank guard: under FSDP these are collective and every
        # rank must participate (rank 0 alone would deadlock).
        model_sd = self._full_state_dict()
        optim_sd = self._full_optim_state_dict()
        # An EMA teacher is training state, not a derived artifact: skipping it
        # would silently reset the targets to the student on resume.
        from vggt_omega.training.teacher import teacher_state_dict

        teacher_sd = teacher_state_dict(self.ema_teacher)
        if self.rank != 0:
            return
        torch.save(model_sd, os.path.join(self.out_dir, f"model_step{step:06d}.pt"))
        if teacher_sd is not None:
            torch.save(teacher_sd, os.path.join(self.out_dir, f"teacher_step{step:06d}.pt"))
        state = {
            "step": step,
            "epoch": self._epoch,
            "optimizer": optim_sd,
            "scheduler": self.scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            # The sampler's frame-draw Generator (DynamicBatchSampler.np_rng) is
            # rebuilt deterministically from the restored epoch in get_loader, so
            # it is not stored here.
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, os.path.join(self.out_dir, f"trainer_step{step:06d}.pt"))
        keep_last = int(self.cfg.run.keep_last)
        for prefix in ("model_step", "trainer_step", "teacher_step"):
            ckpts = sorted(glob.glob(os.path.join(self.out_dir, f"{prefix}*.pt")))
            for old in ckpts[:-keep_last] if keep_last > 0 else []:
                os.remove(old)

    def _fill_missing_optim_state(self, optim_sd):
        """Give every trainable parameter an entry in a loaded optimizer state dict.

        A parameter that is trainable but never receives a gradient never gets Adam
        state, so ``get_optimizer_state_dict`` writes no entry for it -- while
        ``_split_optim_state_dict`` on the way back in looks up EVERY trainable
        parameter and raises ``KeyError`` on the first one missing. The checkpoint is
        not corrupt; the two halves of torch's API just disagree about optional state.

        Hit on the unfrozen offgrid run at ``dense_head.proj_conf.weight``: the dense
        head's confidence branch runs in the forward, but the off-the-grid loss never
        consumes the student's own confidence (only the teacher's, via
        ``conf_weighted_depth``), so no gradient reaches it. Unfreezing is what
        exposed this -- with a frozen backbone those parameters are not trainable and
        never enter the optimizer at all.

        An empty dict is the correct filler: Adam treats a state with no ``step`` as
        uninitialised and populates it lazily on the next update, which is exactly the
        state the parameter was in when the checkpoint was written.
        """
        state = optim_sd.get("state")
        if not isinstance(state, dict):
            return optim_sd
        expected = {
            name for name, p in self._unwrapped_model().named_parameters()
            if p.requires_grad
        }
        missing = sorted(expected - set(state))
        if missing:
            if self.rank == 0:
                logger.warning(
                    f"resume: {len(missing)} trainable parameter(s) had no optimizer "
                    f"state in the checkpoint (never received a gradient); "
                    f"re-initialising them lazily: {', '.join(missing[:4])}"
                    + (" ..." if len(missing) > 4 else "")
                )
            for fqn in missing:
                state[fqn] = {}
        return optim_sd

    def resume(self, trainer_ckpt_path: str):
        state = torch.load(trainer_ckpt_path, map_location="cpu", weights_only=False)
        model_path = trainer_ckpt_path.replace("trainer_step", "model_step")
        if not os.path.exists(model_path):
            # A sidecar without its paired weights is unrecoverable: restoring the
            # optimizer/scheduler against the init weights silently diverges.
            raise FileNotFoundError(
                f"missing model checkpoint {model_path} for sidecar {trainer_ckpt_path}"
            )
        if self.fsdp_enabled:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_model_state_dict,
                set_optimizer_state_dict,
            )

            opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            # Only rank 0 needs the full weights on disk; broadcast_from_rank0
            # scatters them to every rank's shards, so non-rank-0 ranks need no
            # access to model_path (no shared-FS requirement for the weights file).
            model_sd = torch.load(model_path, map_location="cpu") if self.rank == 0 else {}
            set_model_state_dict(self.model, model_sd, options=opts)
            # optim state dict is the 3rd POSITIONAL arg; optimizer already wraps params.
            set_optimizer_state_dict(
                self.model, self.optimizer,
                self._fill_missing_optim_state(state["optimizer"]),
                options=opts,
            )
        else:
            self._unwrapped_model().load_state_dict(torch.load(model_path, map_location="cpu"))
            self.optimizer.load_state_dict(state["optimizer"])
        if self.ema_teacher is not None:
            from vggt_omega.training.teacher import load_teacher_state_dict

            teacher_path = trainer_ckpt_path.replace("trainer_step", "teacher_step")
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(
                    f"missing EMA teacher checkpoint {teacher_path} for sidecar "
                    f"{trainer_ckpt_path}. Resuming without it would silently reset the "
                    "teacher to the student's current weights."
                )
            load_teacher_state_dict(self.ema_teacher, teacher_path, rank=self.rank)
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        cuda_rng = state.get("cuda_rng")
        if cuda_rng is not None and torch.cuda.is_available() and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.global_step = int(state["step"])
        self._epoch = int(state.get("epoch", 0))
        logger.info(f"resumed from {trainer_ckpt_path} at step {self.global_step} (epoch {self._epoch})")

    # --- validation -----------------------------------------------------------------
    def _validate(self, step):
        if self.fsdp_enabled:
            # The param all-gather is collective: EVERY rank must call the gather,
            # then only rank 0 evaluates a transient unsharded copy. A rank-0-only
            # forward on the sharded model would deadlock.
            full_sd = self._full_state_dict()
            if self.rank != 0:
                return
            model = VGGTOmega(embed_dim=int(self.cfg.model.embed_dim)).to(self.device)
            model.load_state_dict(full_sd)
        else:
            if self.rank != 0:
                return
            model = self._unwrapped_model()
        # Lazy imports: `evaluates` pulls in evo/matplotlib; the geometry helper is
        # light but kept lazy alongside it to keep this val-only path out of import.
        from hydra.utils import instantiate
        from vggt_omega.evaluates import CameraPoseMetric, MonoDepthMetric
        from vggt_omega.utils.geometry import world_to_camera_to_camera_to_world

        model.eval()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # release training-shape blocks before the val forward
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
                        gt_c2w = world_to_camera_to_camera_to_world(
                            sample["extrinsics"].numpy()
                        )
                        pred_c2w = world_to_camera_to_camera_to_world(pred_w2c)
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
                except torch.OutOfMemoryError:
                    # Validation must never kill a long training run; free the
                    # cache and skip this sequence.
                    torch.cuda.empty_cache()
                    logger.warning(f"[val {vendor}] sequence {seq_index} skipped: CUDA OOM")
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
