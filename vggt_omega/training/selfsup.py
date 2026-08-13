"""Self-supervised (EMA teacher-student) distillation for VGGT-Omega.

Implements the RGB-only protocol from the VGGT-Omega paper (arXiv 2605.15195,
Sec. 3.4): student and EMA teacher are both initialised from the supervised
checkpoint; the same frames go to both under *independent* stochastic
augmentation; the student is trained to match the teacher via an l2 feature
match across cached aggregator layers plus regression on the (frozen) camera and
depth heads. The teacher is an EMA of the student (see
:class:`vggt_omega.training.teacher.Teacher`, ``mode="ema"``).

The geometry heads are frozen during self-supervision, so only the aggregator
backbone adapts -- the heads still backprop their regression signal into the
backbone, but their own weights stay at the pretrained function, which is what
keeps camera intrinsics / scale from drifting to a degenerate solution.

This v1 uses geometry-preserving appearance augmentations only (colour jitter,
patch masking), so the student and teacher predictions are directly comparable
with no output-space alignment. Geometric augmentations from the paper (90-deg
rotation, frame reordering) change the output frame and need the inverse applied
to the teacher's geometry before regression; they are deferred.
"""
import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# --- head freezing -----------------------------------------------------------
def freeze_backbone(model) -> tuple[int, int]:
    """Freeze everything except the from-scratch gaussian modules.

    Stronger than :func:`freeze_geometry_heads`, which keeps the aggregator
    trainable. Here the ONLY trainable parameters are the ones the released
    checkpoint does not carry -- ``gs_dpt_head`` and ``gs_decoder`` (see
    optim.FROM_SCRATCH_PREFIXES).

    What this buys is structural, not just cheaper: gaussian centres are
    unprojected from ``predictions["depth"]`` and ``pose_enc``, so freezing the
    backbone freezes the GEOMETRY. Two distinct consequences, which are worth
    keeping apart:

    * The **camera/depth anchors go to exactly zero**. They compare the student
      against ``teacher_context`` -- the teacher run on the SAME context views --
      so identical weights on identical input give an identical prediction.
    * The **Sim(3) alignment becomes stationary, NOT the identity**. It is fitted
      between the teacher's 24-view trajectory (subset to the context indices)
      and the student's 16-view one -- two reconstructions in genuinely different
      gauges, since VGGT expresses pose relative to the first frame of its input.
      The transform stays a real scale/rotation/translation. What changes is that
      both sides are now fixed, so the fit and its residual are constant per
      sample instead of degrading as the student's geometry moves.

    That second point is what killed the previous run: the residual climbed
    0.004 -> 0.78 and gate_rate fell to 0/20 by step 800 as the student left the
    teacher's gauge, and the held-out views stopped being scored at all. Frozen,
    the residual sits at the pretrained model's own 16-view-vs-24-view
    consistency (~0.004 measured) and stays there.

    What still trains is appearance -- opacity, scale, rotation and spherical
    harmonics through ``gs_map``. ``render_depth`` keeps its gradient too, since
    the rendered depth is an alpha-weighted composite and alpha is trainable.

    Must run BEFORE the model is wrapped in FSDP/DDP (requires_grad has to be set
    on the bare parameters). Returns ``(trainable, frozen)`` param counts.
    """
    from vggt_omega.training.optim import FROM_SCRATCH_PREFIXES

    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(FROM_SCRATCH_PREFIXES))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "freeze_backbone left NOTHING trainable: no parameter matched "
            f"{FROM_SCRATCH_PREFIXES}. The gaussian head is probably disabled "
            "(model.enable_3dgs), so the run would optimise an empty parameter set."
        )
    return trainable, frozen


def freeze_geometry_heads(model) -> tuple[int, int]:
    """Freeze the camera + dense (depth) heads; keep the aggregator trainable.

    Must run BEFORE the model is wrapped in FSDP/DDP (requires_grad has to be set
    on the bare parameters). Returns ``(trainable, frozen)`` param counts.
    """
    for p in model.parameters():
        p.requires_grad_(True)
    for head in (
        getattr(model, "camera_head", None),
        getattr(model, "dense_head", None),
        getattr(model, "text_alignment_head", None),
    ):
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


# --- dual-view augmentation --------------------------------------------------
_LUMA = (0.299, 0.587, 0.114)


def _per_frame(images: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """One uniform scalar in [lo, hi] per (batch, frame), broadcastable over CHW."""
    b, s = images.shape[:2]
    return torch.empty(b, s, 1, 1, 1, device=images.device, dtype=images.dtype).uniform_(lo, hi)


def _augment_one(images: torch.Tensor, cfg) -> torch.Tensor:
    """One independent geometry-preserving appearance augmentation of (B,S,3,H,W)
    images in [0,1]. Colour jitter (brightness/contrast/saturation) + random patch
    masking; the scene geometry is untouched, so the output geometry is unchanged."""
    x = images
    if cfg.brightness > 0:
        x = x * _per_frame(x, 1 - cfg.brightness, 1 + cfg.brightness)
    if cfg.contrast > 0:
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        x = (x - mean) * _per_frame(x, 1 - cfg.contrast, 1 + cfg.contrast) + mean
    if cfg.saturation > 0:
        gray = (x * x.new_tensor(_LUMA).view(1, 1, 3, 1, 1)).sum(2, keepdim=True)
        x = gray + (x - gray) * _per_frame(x, 1 - cfg.saturation, 1 + cfg.saturation)
    x = x.clamp(0, 1)
    if cfg.mask_ratio > 0:
        b, s, _, h, w = x.shape
        p = max(1, int(cfg.mask_patch))
        gh, gw = (h + p - 1) // p, (w + p - 1) // p
        keep = (torch.rand(b, s, 1, gh, gw, device=x.device) >= cfg.mask_ratio).float()
        mask = F.interpolate(keep.view(b * s, 1, gh, gw), size=(h, w), mode="nearest").view(b, s, 1, h, w)
        x = x * mask
    return x


def augment_two_views(images: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Two *independent* appearance augmentations of the same frames -> (student, teacher)."""
    return _augment_one(images, cfg), _augment_one(images, cfg)


# --- distillation loss -------------------------------------------------------
def _feature_l2(student_feats, teacher_feats) -> torch.Tensor:
    """Mean over cached layers of the per-token MSE between student and teacher
    aggregator features. Teacher features are already detached fp32."""
    if len(student_feats) != len(teacher_feats):
        raise ValueError(
            f"student/teacher cached {len(student_feats)}/{len(teacher_feats)} feature "
            "layers; they must share an architecture (the EMA teacher is a copy of the student)."
        )
    per_layer = [F.mse_loss(s.float(), t.float()) for s, t in zip(student_feats, teacher_feats)]
    return torch.stack(per_layer).mean()


def _depth_regression(student_depth, teacher_depth, teacher_conf, conf_weighted) -> torch.Tensor:
    """L1 of student vs teacher depth, optionally weighted by the teacher's own
    confidence so the student follows the teacher where the teacher is sure."""
    err = (student_depth.float() - teacher_depth.float()).abs()
    if conf_weighted and teacher_conf is not None:
        w = teacher_conf.float().unsqueeze(-1)  # (B,S,H,W) -> (B,S,H,W,1)
        return (w * err).sum() / w.sum().clamp(min=1.0)
    return err.mean()


class DistillLossComputer:
    """Student->teacher distillation: feature l2 + camera + depth regression, no GT.

    weights: dict(feature=1.0, camera=1.0, depth=1.0).
    __call__(student, teacher) -> dict of scalar tensors: feature, camera, depth, total.
      student: dict with ``features`` (list), ``pose_enc``, ``depth`` (grad-carrying).
      teacher: same keys + ``depth_conf`` (all detached fp32, from the EMA teacher).
    """

    def __init__(self, weights, conf_weighted_depth=True):
        self.weights = dict(weights)
        self.conf_weighted_depth = conf_weighted_depth

    def __call__(self, student: dict, teacher: dict) -> dict:
        out = {
            "feature": _feature_l2(student["features"], teacher["features"]),
            # Sum over the 9 pose-encoding components then mean over (B, S) -- the
            # same normalisation the supervised camera loss uses, so the term keeps
            # its weight relative to the per-pixel depth term.
            "camera": (student["pose_enc"] - teacher["pose_enc"]).abs().sum(dim=-1).mean(),
            "depth": _depth_regression(
                student["depth"], teacher["depth"], teacher.get("depth_conf"), self.conf_weighted_depth
            ),
        }
        out["total"] = sum(self.weights.get(k, 0.0) * out[k] for k in ("feature", "camera", "depth"))
        return out


# --- off-the-grid photometric loss -------------------------------------------
PHOTOMETRIC_TERMS = ("l1", "l2", "ssim", "lpips")


class OffGridLossComputer:
    """Off-the-grid supervision: photometric on real pixels, anchored to a teacher.

    ``__call__(context, target, distill)``; see that method for the arguments.
    Returns ``{l1, l2, ssim, lpips, photo_context, photo_target, camera, depth,
    total}``, every value a scalar tensor.

    Three signals, doing three different jobs:

    * **photo_context** — the M views the student was actually given, rendered at
      the student's *own* cameras. Self-consistent by construction, so it says
      only "do the splats reproduce what you were shown".
    * **photo_target** — the N held-out views, rendered at teacher poses mapped
      into the student's frame. These the student has never seen, so they are the
      term that can tell memorised dust from geometry — and the only one that
      depends on the Sim(3), which is why it is the one the refuse gate drops.
    * **camera / depth** — regression onto a *frozen* teacher run over the same M
      context views. Photometric loss alone cannot hold a camera in place: the
      gaussian centres are unprojected from the student's own depth and cameras
      (with gradient), so shrinking the fov while inflating depth leaves the
      render unchanged and the geometry ruined. The frozen teacher is what makes
      that trade cost something.

    The photometric part is a thin adapter over
    ``gaussian_splat.utils.photometric_loss``. The four ``PHOTOMETRIC_TERMS`` are
    always reported, unweighted, over the *context* views — a fixed view count and
    independent of the weights, so they stay comparable across steps and runs (the
    underlying helper omits zero-weight terms, which would KeyError the trainer's
    fixed logging keys).
    """

    def __init__(
        self,
        weights,
        lpips_net: str = "vgg",
        lpips_chunk_size: int | None = None,
        ssim_backend: str = "auto",
        target_weight: float = 1.0,
        camera_weight: float = 1.0,
        depth_weight: float = 1.0,
        render_depth_weight: float = 0.0,
        render_depth_min_alpha: float = 0.5,
        conf_weighted_depth: bool = True,
    ):
        from gaussian_splat.utils import LossWeights

        weights = dict(weights)
        unknown = set(weights) - set(PHOTOMETRIC_TERMS)
        if unknown:
            raise ValueError(
                f"unknown selfsup.weights keys {sorted(unknown)}; expected any of {PHOTOMETRIC_TERMS}"
            )
        if not any(float(v) > 0.0 for v in weights.values()):
            raise ValueError("every selfsup.weights entry is zero — nothing to optimize")
        self.weights = LossWeights(**{k: float(v) for k, v in weights.items()})
        self.lpips_net = lpips_net
        #: Images per LPIPS trunk forward. None runs all B*V at once, which is
        #: what the memory budget cannot afford here: the VGG trunk runs in fp32
        #: (its calibration is fp32-only) over every context view at full training
        #: resolution, and those activations land on top of a step that is already
        #: activation-bound. Chunking trades a little speed for a bounded peak.
        self.lpips_chunk_size = None if lpips_chunk_size in (None, 0) else int(lpips_chunk_size)
        self.ssim_backend = ssim_backend
        self.target_weight = float(target_weight)
        self.camera_weight = float(camera_weight)
        self.depth_weight = float(depth_weight)
        self.render_depth_weight = float(render_depth_weight)
        self.render_depth_min_alpha = float(render_depth_min_alpha)
        self.conf_weighted_depth = bool(conf_weighted_depth)

    @property
    def distills(self) -> bool:
        """Whether ANY teacher-anchor term is on. The trainer reads this to
        decide if the extra context-view teacher forward is worth running --
        render_depth included, since it needs that same teacher depth."""
        return (
            self.camera_weight > 0.0
            or self.depth_weight > 0.0
            or self.render_depth_weight > 0.0
        )

    def _photometric(self, rendered: torch.Tensor, target: torch.Tensor):
        from gaussian_splat.utils import photometric_loss

        return photometric_loss(
            rendered,
            target,
            weights=self.weights,
            ssim_backend=self.ssim_backend,
            lpips_net=self.lpips_net,
            lpips_chunk_size=self.lpips_chunk_size,
        )

    def _target_photometric(self, target, zero):
        """Mean photometric loss over the batch items the refuse gate kept.

        Per item rather than one call on the gathered tensor, because averaging a
        gathered ``(kept, N, ...)`` tensor and averaging per-item losses differ
        once an item is dropped, and only the latter keeps a kept item's weight
        independent of how many of its neighbours were refused.
        """
        if target is None:
            return zero
        rendered, images, keep = target
        kept = keep.nonzero(as_tuple=False).flatten().tolist()
        if not kept:
            return zero
        per_item = [self._photometric(rendered[i : i + 1], images[i : i + 1])[0] for i in kept]
        return torch.stack(per_item).mean()

    def render_depth_loss(self, rendered_depth, rendered_alpha, teacher_depth):
        """L1 between the RENDERED depth and the teacher's, where the render has cover.

        Different from the ``depth`` anchor, which regresses the dense head's raw
        prediction. This one constrains the geometry that actually gets drawn:
        the splats after voxel fusion, opacity and alpha compositing. The two can
        disagree -- a correct depth map whose splats are too transparent or fused
        away renders a different surface -- and only this term notices.

        Masked by alpha, and that is not optional. gsplat's depth is an
        alpha-weighted sum along the ray, so where nothing was splatted it is not
        "far away", it is undefined (0 with 0 alpha). Regressing those pixels
        toward the teacher's real depth would train the model to explain empty
        space, which is exactly backwards: it rewards spraying geometry into
        holes rather than filling them with the right surface.

        ``rendered_depth``/``rendered_alpha`` are (B, V, H, W); ``teacher_depth``
        is (B, V, H, W, 1) as the dense head emits it.
        """
        teacher_depth = teacher_depth.squeeze(-1).float()
        rendered_depth = rendered_depth.float()
        covered = (rendered_alpha.float() > self.render_depth_min_alpha).float()
        error = (rendered_depth - teacher_depth).abs() * covered
        return error.sum() / covered.sum().clamp(min=1.0)

    def _distill(self, distill, zero):
        if distill is None:
            return zero, zero
        student, teacher = distill
        student_pose, teacher_pose = student["pose_enc"], teacher["pose_enc"]

        # Drop components the teacher could not produce. The teacher is a frozen
        # ORACLE: where it returns a non-finite pose it has no opinion, and
        # regressing toward it turns the whole anchor into nan -- one bad element
        # poisons the sum, the total, and then every gradient in the step.
        #
        # Substituting the student's own DETACHED value (rather than masking the
        # difference afterwards) is what makes this safe: the difference is then
        # exactly zero with exactly zero gradient at those positions. Masking
        # after the subtraction does not work -- the |.| backward still evaluates
        # sign(nan) there, and 0 * nan is nan, so the poison survives the mask.
        #
        # Observed on MUSA at 8 ranks: the teacher emits nan pose_enc for part of
        # one CP group's context set, every step, while the same teacher on the
        # same weights is finite for that sample in a single process and bit-
        # identical across ranks. That is an SDK-level fault under memory load,
        # not a modelling decision, so it is neutralised rather than trained on.
        finite = torch.isfinite(teacher_pose)
        dropped = int((~finite).sum())
        if dropped:
            teacher_pose = torch.where(finite, teacher_pose, student_pose.detach())
            self.dropped_pose_components = getattr(self, "dropped_pose_components", 0) + dropped

        # Sum over the 9 pose-encoding components then mean over (B, S) -- the
        # same normalisation the supervised camera loss uses.
        camera = (student_pose - teacher_pose).abs().sum(dim=-1).mean()
        depth = _depth_regression(
            student["depth"], teacher["depth"], teacher.get("depth_conf"), self.conf_weighted_depth
        )
        return camera, depth

    def __call__(self, context, target=None, distill=None, render_depth=None) -> dict:
        """Combine the three signals into one weighted total.

        Args:
            context: ``(rendered, images)``, both ``(B, M, 3, H, W)`` in [0, 1].
            target: ``(rendered, images, keep)`` for the held-out views, where
                ``keep`` is the :func:`sim3_refuse_gate` mask ``(B,)``; or None
                when every item was refused and the render was skipped entirely.
            distill: ``(student, teacher)`` prediction dicts over the *context*
                views — the student's carrying gradient, the frozen teacher's
                detached — or None to run photometric-only.
        """
        rendered, images = context
        photo_context, terms = self._photometric(rendered, images)
        zero = rendered.new_zeros(())

        out = {name: terms.get(name, zero) for name in PHOTOMETRIC_TERMS}
        out["photo_context"] = photo_context
        out["photo_target"] = self._target_photometric(target, zero)
        out["camera"], out["depth"] = self._distill(distill, zero)
        out["render_depth"] = (
            self.render_depth_loss(*render_depth)
            if render_depth is not None and self.render_depth_weight > 0.0
            else zero
        )
        out["total"] = (
            out["photo_context"]
            + self.target_weight * out["photo_target"]
            + self.camera_weight * out["camera"]
            + self.depth_weight * out["depth"]
            + self.render_depth_weight * out["render_depth"]
        )
        return out


# --- off-the-grid view split + trajectory alignment ---------------------------
def split_context_target(num_views: int, n_target: int, generator=None):
    """Partition ``num_views`` frame indices into (context, target).

    Frame 0 always stays in the context set. VGGT anchors its output frame to
    input view 0, so keeping it shared puts the student's M-view prediction and
    the teacher's M+N prediction in the same gauge — the Sim(3) that follows is
    then a small correction (mostly scale) rather than an arbitrary rigid motion,
    which is what keeps it well conditioned. It also means the student's splats
    already live in the frame the aligned teacher poses are mapped into.

    Returns ``(context_idx, target_idx)``, both sorted int64 tensors on CPU.
    """
    if n_target < 1:
        raise ValueError(f"selfsup.n_target must be >= 1, got {n_target}")
    if num_views < n_target + 2:
        raise ValueError(
            f"a context/target split needs at least n_target + 2 = {n_target + 2} views "
            f"(frame 0 plus one more context frame), got S={num_views}. Lower "
            "selfsup.n_target or raise the sampler's img_nums."
        )
    drawn = torch.randperm(num_views - 1, generator=generator)[:n_target] + 1
    target_idx = drawn.sort().values
    keep = torch.ones(num_views, dtype=torch.bool)
    keep[target_idx] = False
    return keep.nonzero(as_tuple=False).squeeze(-1), target_idx


#: Gaussian fields checked before rasterization. ``covariances`` is derived from
#: scales/rotations, so it is guarded too rather than trusted.
_GAUSSIAN_FIELDS = ("means", "covariances", "harmonics", "opacities", "scales", "rotations")


def sanitize_gaussians(gaussians):
    """Neutralize non-finite splats so they cannot reach the rasterizer.

    ``scales`` are already clamped and ``opacities`` come from a bounded
    mapping, but ``means`` are unprojected from predicted depth: a depth that
    collapses toward zero or blows up puts a splat at an extreme coordinate, and
    a NaN there propagates into gsplat's tile arithmetic. On MUSA that surfaced
    as ``NonzeroOut MUDNN failed`` / ``copy D2H code700`` -- an illegal device
    address inside the backward, which kills the job outright rather than being
    caught by the non-finite-gradient guard.

    Replaces non-finite values with zeros and forces the opacity of any affected
    splat to zero, so it contributes nothing to the render instead of poisoning
    it. Gradients for well-behaved splats are untouched; a zeroed splat simply
    stops receiving photometric gradient, which is what should happen to one
    whose position is undefined.

    Returns ``(gaussians, num_neutralized)`` -- the count is worth logging, since
    a rising one means the geometry is degenerating even while the loss looks
    healthy.
    """
    import dataclasses

    bad = None
    for name in _GAUSSIAN_FIELDS:
        value = getattr(gaussians, name, None)
        if value is None:
            continue
        # Reduce every trailing dimension down to one flag per splat (B, N).
        flags = ~torch.isfinite(value)
        while flags.dim() > 2:
            flags = flags.any(dim=-1)
        bad = flags if bad is None else (bad | flags)

    if bad is None or not bool(bad.any()):
        return gaussians, 0

    fields = {}
    for name in _GAUSSIAN_FIELDS:
        value = getattr(gaussians, name, None)
        if value is None:
            continue
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if name == "opacities":
            value = value * (~bad).to(value.dtype)
        fields[name] = value
    return dataclasses.replace(gaussians, **fields), int(bad.sum())


class TrajectoryAlignment(NamedTuple):
    """A teacher trajectory moved into the student's frame, and how far it moved.

    ``extrinsics`` is ``(B, S, 3, 4)``. The three diagnostics are one scalar per
    batch item, and exist so the caller can decide whether to *trust* the
    alignment (see :func:`sim3_refuse_gate`) rather than only apply it:

    ``scale_log``
        ``|log s|``. How far apart the two gauges' scales were. A student solving
        M views and a teacher solving M+N normalise the scene differently, so a
        small value here is expected; a large one means they disagree about more
        than scale.
    ``rotation_deg``
        The rotation angle of ``R``, in degrees. Both trajectories are anchored to
        view 0, so this should be near zero — a large rotation means the two
        reconstructions have genuinely different shapes and the fit is spinning
        one onto the other.
    ``residual``
        RMS alignment error over the context overlap as a fraction of the student
        trajectory's own RMS radius, so it is scale-free. This is the direct
        measure of whether one Sim(3) can explain the overlap at all — and hence
        of whether extrapolating it to the held-out target views means anything.
        Infinite when the student's camera centres carry no spread to normalise by.
    """

    extrinsics: torch.Tensor
    scale_log: torch.Tensor
    rotation_deg: torch.Tensor
    residual: torch.Tensor


def _alignment_diagnostics(scale, rotation, translation, src, dst):
    """``(scale_log, rotation_deg, residual)`` for one fitted Sim(3), as float64
    CPU scalars. Computed alongside :func:`umeyama_sim3`, which does its own
    linear algebra in float64 on CPU for the same conditioning reasons."""
    src = src.double().cpu()
    dst = dst.double().cpu()
    scale = scale.double().cpu()
    rotation = rotation.double().cpu()
    translation = translation.double().cpu()

    inf = torch.tensor(float("inf"), dtype=torch.float64)
    scale_log = scale.log().abs() if bool(scale > 0) else inf

    cosine = ((torch.diagonal(rotation).sum() - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation_deg = torch.rad2deg(torch.arccos(cosine))

    fitted = scale * torch.einsum("ij,nj->ni", rotation, src) + translation
    error = (fitted - dst).norm(dim=-1)
    radius = (dst - dst.mean(dim=0)).norm(dim=-1)
    spread = radius.square().mean().sqrt()
    residual = error.square().mean().sqrt() / spread if bool(spread > 1e-12) else inf
    return scale_log, rotation_deg, residual


@torch.no_grad()
def align_teacher_trajectory(teacher_ext, student_ext, context_idx) -> TrajectoryAlignment:
    """Move a teacher's ``(B, S, 3, 4)`` trajectory into the student's frame.

    Fits one Sim(3) per batch item from the teacher's *context* camera centres
    onto the student's, then applies it to all S teacher poses. The student's
    splats never move; only the cameras do.

    Detached by construction: the aligned poses are per-step constants, so the
    photometric gradient reaches the student through the splat parameters alone.
    Backpropagating through the covariance SVD would be the alternative, and it
    is numerically fragile exactly where these trajectories live (near-colinear
    camera centres).
    """
    from vggt_omega.utils.geometry import (
        apply_sim3_to_extrinsics,
        camera_centers,
        umeyama_sim3,
    )

    context_idx = context_idx.to(teacher_ext.device)
    aligned, diagnostics = [], []
    inf64 = torch.tensor(float("inf"), dtype=torch.float64)
    for item in range(teacher_ext.shape[0]):
        src = camera_centers(teacher_ext[item, context_idx])
        dst = camera_centers(student_ext[item])

        # A non-finite student trajectory has no meaningful alignment, and both
        # halves of that need handling explicitly rather than left to propagate.
        #
        # The TRANSFORM: identity, not the fitted-then-nan one. These extrinsics
        # are handed to the rasterizer for EVERY item whenever any item survives
        # the gate (see Trainer._offgrid_step), so a nan here would reach the
        # renderer even for items that were refused. Identity keeps the teacher's
        # own finite poses.
        #
        # The DIAGNOSTICS: infinite, which makes sim3_refuse_gate refuse this
        # item (it tests `value.isfinite()`). That is the intended behaviour, not
        # a side effect -- the held-out target views are dropped and the step
        # trains on the context self-render alone, which is exactly what an
        # unusable alignment should cost. Forced here rather than left to emerge
        # from nan arithmetic so it is deterministic and testable.
        if not bool(torch.isfinite(src).all() and torch.isfinite(dst).all()):
            logger.warning(
                "align_teacher_trajectory: non-finite camera centres "
                f"(teacher {int((~torch.isfinite(src)).sum())}, student "
                f"{int((~torch.isfinite(dst)).sum())}) in batch item {item}. Using an "
                "identity alignment and refusing this item's target views; the step keeps "
                "its context supervision. Persisting means the student is diverging."
            )
            # Match umeyama_sim3's contract: it returns in the INPUT dtype and
            # device, and apply_sim3_to_extrinsics assumes that.
            opts = {"dtype": src.dtype, "device": src.device}
            aligned.append(
                apply_sim3_to_extrinsics(
                    teacher_ext[item],
                    torch.tensor(1.0, **opts),
                    torch.eye(3, **opts),
                    torch.zeros(3, **opts),
                )
            )
            diagnostics.append((inf64, inf64, inf64))
            continue

        scale, rotation, translation = umeyama_sim3(src, dst)
        aligned.append(apply_sim3_to_extrinsics(teacher_ext[item], scale, rotation, translation))
        diagnostics.append(_alignment_diagnostics(scale, rotation, translation, src, dst))

    stacked = [
        torch.stack(values).to(device=teacher_ext.device, dtype=torch.float32)
        for values in zip(*diagnostics)
    ]
    return TrajectoryAlignment(torch.stack(aligned), *stacked)


def sim3_refuse_gate(
    alignment: TrajectoryAlignment,
    max_scale_log: float | None = None,
    max_rotation_deg: float | None = None,
    max_residual: float | None = None,
) -> torch.Tensor:
    """Per-batch-item mask: may the target views be posed from this Sim(3)?

    The context views are rendered at the student's own cameras and so never
    depend on the alignment; the target views are the student's splats seen from
    a *teacher* pose mapped through the fitted Sim(3). When that fit is poor the
    target render is of the right scene from the wrong place, and the photometric
    loss then asks the student to move geometry to explain an error the geometry
    did not cause. Refusing those views is strictly better than that: the step
    keeps its context supervision and simply learns nothing from the held-out
    frames.

    Each threshold is skipped when None or non-finite. A non-finite *diagnostic*
    always refuses — that is a degenerate fit (colinear or coincident camera
    centres), which is exactly the case worth refusing.

    Returns a bool tensor ``(B,)`` on the alignment's device.
    """
    keep = torch.ones_like(alignment.residual, dtype=torch.bool)
    for value, limit in (
        (alignment.scale_log, max_scale_log),
        (alignment.rotation_deg, max_rotation_deg),
        (alignment.residual, max_residual),
    ):
        if limit is None or not math.isfinite(float(limit)):
            continue
        keep &= value.isfinite() & (value <= float(limit))
    return keep
