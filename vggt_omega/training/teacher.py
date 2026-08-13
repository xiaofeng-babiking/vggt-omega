# vggt_omega/training/teacher.py
"""Frozen VGGT-Omega teacher running in parallel with training.

A pretrained -- and optionally much larger -- VGGT-Omega that predicts pose and
depth for the same images the student sees, without ever taking a gradient.
Because it is frozen, data-parallel training needs no collectives for it at all:
every rank runs its own replica on its own microbatch. That stops working only
when the checkpoint is too large to replicate, which is what ``shard`` is for.

``teacher.shard`` modes:

``none``
    Full parameters resident on every rank, no communication. Requires the whole
    teacher to fit alongside the student.
``fsdp``
    Parameters sharded over a data-parallel mesh and all-gathered per block on
    demand, then freed. Roughly ``1/world_size`` resident, paid for with a
    model-sized all-gather every step.
``cpu_offload``
    Sharded parameters held in pinned host memory, streamed in per block.
    Smallest GPU footprint, PCIe-bound.

Sharding the teacher is independent of how the *student* is parallelised: a DDP
student with an FSDP-sharded teacher is fine, because a DeviceMesh is only a
process-group wrapper.

Parameters stay fp32 in every mode. Casting the module to bf16 is tempting for a
frozen model, but ``CameraHead``/``DenseHead`` upcast their input with ``.float()``
before each LayerNorm, so bf16 *weights* against fp32 *inputs* raise a dtype
mismatch in the forward -- the same hazard ``parallel.build_mp_policy`` documents
for FSDP's ``param_dtype``. bf16 compute still happens: the model applies its own
autocast around the aggregator.
"""
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel

from vggt_omega.models.aggregator import _RESNET_MEAN, _RESNET_STD
from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.training.parallel import apply_fsdp, build_dp_mesh, resolve_hybrid_shard

logger = logging.getLogger(__name__)

SHARD_MODES = ("none", "fsdp", "cpu_offload")
TEACHER_MODES = ("frozen", "ema")

#: Buffers registered ``persistent=False`` are absent from the state dict, so a
#: meta-initialised model cannot restore them by loading -- ``to_empty()`` leaves
#: them as uninitialised memory and the forward silently normalises with garbage.
#: They are derived constants, so rebuild them explicitly. Keyed by attribute name.
_NONPERSISTENT_BUFFERS = {"_resnet_mean": _RESNET_MEAN, "_resnet_std": _RESNET_STD}

#: Keys the teacher hands back. Everything else the model returns is dropped so a
#: caller can never accidentally hold a reference into the teacher's graph.
TEACHER_KEYS = ("pose_enc", "depth", "depth_conf")


def _gib(n_bytes):
    return n_bytes / (1024 ** 3)


def _free_bytes(device):
    """Free memory on ``device``, or None when the backend cannot report it."""
    backend = getattr(torch, device.type, None)
    mem_get_info = getattr(backend, "mem_get_info", None)
    if mem_get_info is None:
        return None
    try:
        free, _total = mem_get_info(device)
    except (RuntimeError, TypeError):
        return None
    return free


def _param_bytes(arch):
    """Size a VGGTOmega without allocating it: meta tensors carry shape + dtype."""
    with torch.device("meta"):
        probe = VGGTOmega(**arch)
    return sum(p.numel() * p.element_size() for p in probe.parameters())


def _materialize_nonpersistent_buffers(model):
    """Rebuild buffers that ``load_state_dict`` cannot restore.

    Raises rather than guessing if the model grows a non-persistent buffer this
    module does not know how to rebuild -- silently leaving one uninitialised
    would corrupt the teacher's predictions without any error.
    """
    persistent = set(model.state_dict().keys())
    unhandled = []
    for name, buf in model.named_buffers():
        if name in persistent:
            continue
        parent_path, _, attr = name.rpartition(".")
        value = _NONPERSISTENT_BUFFERS.get(attr)
        if value is None:
            unhandled.append(name)
            continue
        parent = model.get_submodule(parent_path) if parent_path else model
        rebuilt = torch.tensor(value, dtype=torch.float32, device=buf.device)
        setattr(parent, attr, rebuilt.view(buf.shape))
    if unhandled:
        raise RuntimeError(
            "teacher: cannot rebuild non-persistent buffer(s) "
            f"{unhandled} after meta-init. Add them to _NONPERSISTENT_BUFFERS in "
            "vggt_omega/training/teacher.py, or the teacher will predict from "
            "uninitialised memory."
        )


def _resolve_arch(cfg):
    arch = OmegaConf.select(cfg, "teacher.arch", default=None)
    arch = dict(OmegaConf.to_container(arch, resolve=True)) if arch is not None else {}
    arch.setdefault("patch_size", int(OmegaConf.select(cfg, "model.patch_size", default=16)))
    arch.setdefault("embed_dim", int(OmegaConf.select(cfg, "model.embed_dim", default=1024)))
    # The teacher only ever supplies pose + depth; building the other heads would
    # add parameters the released checkpoints do not carry.
    arch["enable_camera"] = True
    arch["enable_depth"] = True
    arch["enable_alignment"] = False
    # Must match the student's: the camera anchor is an L1 between the two
    # pose_enc tensors, so a teacher on a different fov activation would shift
    # the target the student is regressed onto. Identical on the pretrained
    # operating range either way, but the anchor should not depend on that.
    arch.setdefault(
        "fov_activation", str(OmegaConf.select(cfg, "model.fov_activation", default="softplus"))
    )
    return arch


def _check_fits(arch, device, world_size):
    """Refuse an unreplicable ``shard='none'`` teacher with numbers and a remedy."""
    need = _param_bytes(arch)
    free = _free_bytes(device)
    if free is None or need <= free:
        return
    per_rank = _gib(need / max(world_size, 1))
    raise RuntimeError(
        f"teacher shard='none' needs {_gib(need):.1f} GiB but only {_gib(free):.1f} GiB "
        f"free on {device} after the student was built.\n"
        f"Set teacher.shard='fsdp' (world={world_size} -> ~{per_rank:.1f} GiB/rank) "
        "or 'cpu_offload'."
    )


def _build_sharded_skeleton(arch, device, mesh, cpu_offload):
    """Meta-init then shard, leaving parameter *values* undefined.

    Split out from loading because an EMA teacher gets its values from the
    student's shards, not from a file -- copying shard-to-shard needs no
    collective at all, whereas loading a full state dict would.
    """
    with torch.device("meta"):
        model = VGGTOmega(**arch)
    apply_fsdp(
        model,
        mesh,
        reshard_after_forward=True,  # free each block right after use; keeping them
        param_dtype="none",          # resident would just be a replica again
        reduce_dtype="none",         # no gradients here, so nothing to reduce
        cpu_offload=cpu_offload,
    )
    model.to_empty(device=device)
    _materialize_nonpersistent_buffers(model)
    return model


def _build_sharded(arch, checkpoint, device, mesh, rank, cpu_offload):
    """Shard, then stream the checkpoint in from rank 0 only.

    Every rank calling ``torch.load`` on a large checkpoint would need the whole
    file in host memory per rank; ``broadcast_from_rank0`` scatters rank 0's copy
    straight into each rank's shards instead. Same pattern as Trainer.resume.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model = _build_sharded_skeleton(arch, device, mesh, cpu_offload)
    state = torch.load(checkpoint, map_location="cpu") if rank == 0 else {}
    set_model_state_dict(
        model, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    )
    _materialize_nonpersistent_buffers(model)
    return model


def _unwrap(model):
    """The bare module, whichever parallel wrapper is on it. ``fully_shard`` edits
    in place and keeps parameter names, so only DDP needs unwrapping."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class Teacher(nn.Module):
    """A frozen VGGT-Omega running alongside training: always in eval, never
    accumulates grad, and returns detached fp32 pose/depth for whatever images it
    is given.

    ``ema=False`` (the default) is a fixed reference -- typically the pretrained
    prior, and free to be a different size than the student. ``ema=True`` makes the
    weights an exponential moving average of the student's, anchoring to the
    *recent past* rather than a fixed point: it slows drift instead of preventing
    it, because a student that walks somewhere degenerate drags the average along
    behind it. An EMA teacher alone will not stop a collapse -- pair it with a
    fixed reference, or with the stop-grad/centering machinery BYOL and DINO use
    for exactly this reason.

    In EMA mode the update is deliberately shard-local: each rank interpolates only
    the parameter shards it already holds, so there is no collective and no
    synchronisation point. That is only sound when teacher and student are
    parallelised identically, which ``update`` verifies rather than assumes.

    ``update`` is the one mode-gated method: it raises in frozen mode, because a
    fixed teacher may be a different architecture entirely (a 10B teacher over a 1B
    student), so lerping it toward the student would crash or corrupt it.
    """

    def __init__(self, model, shard="none", ema=False, decay=0.999, start_decay=0.9, warmup_steps=0):
        super().__init__()
        self.shard = shard
        self.ema = bool(ema)
        self.decay = float(decay)
        self.start_decay = float(start_decay)
        self.warmup_steps = int(warmup_steps)
        model.requires_grad_(False)
        self.model = model
        # nn.Module.__init__ leaves *this* wrapper at training=True even though the
        # inner model is in eval, so pin the whole thing after the child is attached.
        self.eval()

    def train(self, mode=True):
        """Pin to eval. The Trainer calls ``.train()`` on what it owns each epoch;
        a teacher that follows the student into train mode would start updating
        norm statistics and stop being a fixed reference."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, return_features=False):
        out = self.model(images, return_features=return_features)
        missing = [k for k in TEACHER_KEYS if k not in out]
        if missing:
            raise RuntimeError(
                f"teacher forward produced no {missing}; the teacher must be built "
                "with enable_camera=True and enable_depth=True."
            )
        result = {k: out[k].detach().float() for k in TEACHER_KEYS}
        if return_features:
            # Detached fp32 targets for the distillation feature-matching loss.
            result["features"] = [f.detach().float() for f in out["features"]]
        return result

    def decay_at(self, step):
        """Ramp the decay in. An EMA initialised from the student and immediately
        given decay=0.999 barely moves for thousands of steps, so early targets
        are stale exactly when the student is changing fastest."""
        if step is None or self.warmup_steps <= 0:
            return self.decay
        ramp = min(1.0, (step + 1) / self.warmup_steps)
        return self.start_decay + (self.decay - self.start_decay) * ramp

    @torch.no_grad()
    def update(self, student, step=None, decay=None):
        """Pull an EMA teacher a fraction of the way toward the student.

        ``decay=0`` copies the student outright, which is how the teacher is
        initialised. Raises in frozen mode -- see the class docstring.
        """
        if not self.ema:
            raise RuntimeError(
                "update() called on a frozen (ema=False) teacher. A frozen teacher is "
                "a fixed reference and may be a different architecture than the student; "
                "lerping it toward the student would corrupt it. Build with ema=True "
                "(teacher.mode='ema') to make the teacher trackable."
            )
        d = self.decay_at(step) if decay is None else float(decay)
        student = _unwrap(student)

        student_params = dict(student.named_parameters())
        for name, tp in self.model.named_parameters():
            sp = student_params.get(name)
            if sp is None:
                raise RuntimeError(
                    f"EMA teacher has parameter {name!r} the student does not. An EMA "
                    "teacher is an average of the student, so the two must share an "
                    "architecture -- teacher.arch cannot differ in mode='ema'."
                )
            # DTensor.shape is the *global* shape, so a global mismatch is an
            # architecture difference while a local-only mismatch is a sharding
            # difference. Reporting them as one error sends you hunting the wrong bug.
            if sp.shape != tp.shape:
                raise RuntimeError(
                    f"EMA teacher and student disagree on the shape of {name!r} "
                    f"(teacher {tuple(tp.shape)} vs student {tuple(sp.shape)}). An EMA "
                    "teacher is an average of the student, so it must be built with the "
                    "student's architecture -- check cfg.model, and use mode='frozen' if "
                    "you wanted a differently sized teacher."
                )
            if isinstance(sp, DTensor) != isinstance(tp, DTensor) or _local(sp).shape != _local(tp).shape:
                raise RuntimeError(
                    f"EMA teacher and student disagree on the sharding of {name!r} "
                    f"(teacher {type(tp).__name__}{tuple(_local(tp).shape)} vs student "
                    f"{type(sp).__name__}{tuple(_local(sp).shape)}). The update is "
                    "shard-local, so teacher.shard must match the student's parallel mode."
                )
            if d <= 0.0:
                # Not lerp_(sp, 1.0): an EMA teacher is initialised on top of
                # to_empty() memory, and lerp computes tp + 1.0 * (sp - tp), so a
                # single inf in that garbage yields inf + -inf = NaN forever.
                tp.copy_(sp.detach())
            else:
                tp.lerp_(sp.detach(), 1.0 - d)

        # Buffers are constants here (the ImageNet mean/std, the masked-bias
        # pattern). Interpolating them would be meaningless; track them exactly.
        student_buffers = dict(student.named_buffers())
        for name, tb in self.model.named_buffers():
            sb = student_buffers.get(name)
            if sb is not None and _local(sb).shape == _local(tb).shape:
                tb.copy_(sb)


def _build_replica_skeleton(arch, device):
    """A full-size model whose parameter values are undefined, for an EMA teacher
    that will take them from the student rather than from a file."""
    with torch.device("meta"):
        model = VGGTOmega(**arch)
    model.to_empty(device=device)
    _materialize_nonpersistent_buffers(model)
    return model


def build_teacher(cfg, device, world_size=1, rank=0, mesh=None, student=None):
    """Build the teacher described by ``cfg.teacher``, or None if unset.

    ``mode='frozen'`` needs a checkpoint and never changes. ``mode='ema'`` tracks
    the student, so it starts either from a checkpoint or -- with none given --
    from the student's current weights, and needs ``student`` either way.

    Call this *after* the student exists so the ``shard='none'`` fit check sees
    the memory the student already took. Pass the student's ``mesh`` when it has
    one so both models shard over the same process groups instead of building a
    second, duplicate set.
    """
    if OmegaConf.select(cfg, "teacher", default=None) is None:
        return None

    mode = str(OmegaConf.select(cfg, "teacher.mode", default="frozen")).lower()
    if mode not in TEACHER_MODES:
        raise ValueError(f"teacher.mode must be one of {TEACHER_MODES}, got {mode!r}")
    checkpoint = OmegaConf.select(cfg, "teacher.checkpoint", default=None)
    if mode == "frozen" and not checkpoint:
        return None

    shard = str(OmegaConf.select(cfg, "teacher.shard", default="none")).lower()
    if shard not in SHARD_MODES:
        raise ValueError(f"teacher.shard must be one of {SHARD_MODES}, got {shard!r}")
    if mode == "ema" and student is None:
        raise RuntimeError(
            "teacher.mode='ema' averages the student's weights, so build_teacher "
            "needs student=<the trainable model>."
        )
    if mode == "ema" and OmegaConf.select(cfg, "teacher.arch", default=None) is not None:
        raise ValueError(
            "teacher.arch is meaningless with mode='ema': an average of the student "
            "necessarily has the student's architecture. Remove it (the arch is taken "
            "from cfg.model), or use mode='frozen' for an independently sized teacher."
        )

    arch = _resolve_arch(cfg)
    device = torch.device(device)
    cpu_offload = shard == "cpu_offload"

    if shard == "none":
        _check_fits(arch, device, world_size)
        if checkpoint:
            model = VGGTOmega(**arch)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            model = model.to(device)
        else:
            model = _build_replica_skeleton(arch, device)
    else:
        if not dist.is_initialized():
            raise RuntimeError(
                f"teacher.shard={shard!r} needs an initialised process group; "
                "run under torchrun or set teacher.shard='none'."
            )
        if mesh is None:
            # Mirror the student's fsdp.* topology (incl. HSDP) so a teacher
            # built without a student mesh -- e.g. a DDP student with a sharded
            # teacher -- still shards intra-node and replicates across nodes.
            local_world = os.environ.get("LOCAL_WORLD_SIZE")
            mesh = build_dp_mesh(
                world_size,
                hybrid_shard=resolve_hybrid_shard(
                    OmegaConf.select(cfg, "fsdp.hybrid_shard", default=False),
                    world_size=world_size,
                    local_world_size=int(local_world) if local_world else None,
                ),
                shard_size=OmegaConf.select(cfg, "fsdp.shard_size", default=None),
                device_type=device.type,
            )
        model = (
            _build_sharded(arch, checkpoint, device, mesh, rank, cpu_offload)
            if checkpoint
            else _build_sharded_skeleton(arch, device, mesh, cpu_offload)
        )

    teacher = Teacher(
        model,
        shard=shard,
        ema=(mode == "ema"),
        decay=float(OmegaConf.select(cfg, "teacher.decay", default=0.999)),
        start_decay=float(OmegaConf.select(cfg, "teacher.start_decay", default=0.9)),
        warmup_steps=int(OmegaConf.select(cfg, "teacher.warmup_steps", default=0)),
    )
    if mode == "ema" and not checkpoint:
        teacher.update(student, decay=0.0)

    logger.info(
        "teacher: %s params, mode=%s, shard=%s, from %s",
        f"{sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B",
        mode,
        shard,
        checkpoint or "student weights",
    )
    return teacher


def teacher_state_dict(teacher):
    """Consolidated teacher weights, or None. Collective when the teacher is
    sharded, so every rank must call it -- not just rank 0."""
    if teacher is None:
        return None
    if teacher.shard == "none":
        return teacher.model.state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(
        teacher.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def load_teacher_state_dict(teacher, path, rank=0):
    """Restore teacher weights saved by :func:`teacher_state_dict`.

    Takes a path rather than a loaded state dict so that *this* function decides
    which ranks read the file. When the teacher is sharded the restore is a
    collective: every rank must call in, and only rank 0 supplies the weights. A
    caller that pre-loaded on rank 0 and passed None elsewhere would strand the
    other ranks outside the collective and hang the job.

    An EMA teacher that is never restored silently restarts from the student,
    which looks like training continuing normally while the targets have reset.
    """
    if teacher is None:
        return
    if teacher.shard == "none":
        teacher.model.load_state_dict(torch.load(path, map_location="cpu"))
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    set_model_state_dict(
        teacher.model,
        torch.load(path, map_location="cpu") if rank == 0 else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
