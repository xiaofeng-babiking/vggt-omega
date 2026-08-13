# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Photometric and perceptual losses over rendered vs. ground-truth images.

Four terms, all taking ``(..., 3, H, W)`` images in [0, 1] — the layout
``gaussian_splat.render.render_by_gsplat`` emits as ``RenderOutput.color``
(B, V, 3, H, W) — and returning a scalar:

    ``l1_loss``     mean |pred - target|, optional pixel mask
    ``l2_loss``     mean (pred - target)^2 (MSE), optional pixel mask
    ``ssim_loss``   1 - SSIM, backed by the fused CUDA/MUSA kernel
    ``lpips_loss``  learned perceptual distance (AlexNet/VGG trunk)

plus ``photometric_loss``, which combines them under ``LossWeights`` and
returns the per-term breakdown for logging.

Optional native/perceptual backends (both optional; zero-weight terms are
never evaluated, and SSIM has an exact pure-torch fallback):

  - ``fused_ssim`` — rahul-goel's fully-fused differentiable SSIM CUDA
    kernel. Not installed in this repo's env today; ``backend="auto"`` then
    runs the torch reference, which reproduces the kernel op-for-op (see
    below). Install it to reclaim the speed; the import is probed, never
    required.
  - ``lpips`` (PyPI, richzhang's package) — the calibrated linear weights
    ship in the wheel; the *trunk* (AlexNet / VGG16 ImageNet weights) is
    fetched by torchvision into ``$TORCH_HOME/hub/checkpoints`` (default
    ``~/.cache/torch``) on first use — 553 MB for VGG16, via the proxy on
    this host.

Reference points in AnySplat@6dced92, the source of the ported head
and renderer — the numbers its shipped configs actually train with:

  - ``src/loss/loss_lpips.py`` — ``LPIPS(net="vgg")`` called with
    ``normalize=True`` on [0, 1] images, weight 0.05 (``config/loss/lpips.yaml``)
  - ``src/loss/loss_mse.py`` — plain masked MSE, weight 1.0; the experiment
    configs compose ``mse + lpips + depth_consis``, so the photometric part
    is ``1.0*MSE + 0.05*LPIPS``
  - ``src/post_opt/simple_trainer.py:858`` — per-scene refinement instead uses
    3DGS's ``0.8*L1 + 0.2*(1 - fused_ssim(..., padding="valid"))``
  - ``src/loss/loss_ssim.py`` — a vendored pytorch-msssim, dead code: its one
    import is shadowed before use, so no SSIM term reaches AnySplat training

Design notes:

  - SSIM/LPIPS are plain functions over a module-level cache rather than
    ``nn.Module`` members. An LPIPS submodule hung off the model would put
    its frozen trunk into ``state_dict()`` and hand it to FSDP2 as a shard
    candidate; the cache keeps it out of both. (AnySplat solves the same
    problem from the other side, rewriting LPIPS's parameters into
    non-persistent buffers via ``convert_to_buffer``.)
  - ``padding="same"`` is the default, matching the original SSIM paper and
    the INRIA 3DGS implementation. AnySplat's post-opt trainer and gsplat's
    pass ``padding="valid"``, which drops the 5-pixel border where the window
    reads zeros from outside the image — a fair bit stricter on small crops.
  - no ``nan_to_num`` wrapper. AnySplat clamps every loss to 0 on NaN/Inf,
    which keeps a run alive but turns a diverged render into a silent no-op
    gradient. Here a NaN render surfaces as a NaN loss; the only NaN this
    module removes by construction is the empty-mask 0/0.
  - ``fused_ssim`` only backpropagates into its *first* argument, so
    gradients reach ``pred`` and never ``target``. That is what training
    against fixed GT wants; if you need both sides differentiable (e.g.
    distilling one render against another), force ``backend="torch"``.
  - the pure-torch SSIM fallback reproduces the fused kernel exactly — same
    11-tap sigma=1.5 window, zero padding outside the image, variances
    clamped at 0 — so CPU tests and GPU training agree (verified to ~1e-6 in
    ``tests/test_losses.py``). Holding that guarantee is why ``ssim`` pins
    itself to fp32 with autocast disabled. The fused kernel has accepted
    bf16/fp16 since fused-ssim!1 and accumulates its statistics in fp32
    whatever the input dtype, so the pin gives up no accuracy — but the
    kernel returns the score in the *input* dtype, and bf16 steps by 3.9e-3
    across [0.5, 1), where a trained model's scores live (0.999 is not
    representable, it rounds to 1.0). Unpinned, the two backends land ~1.6e-3
    apart — a thousand times the agreement above — and ``1 - ssim`` reads
    exactly 0.0 once SSIM passes ~0.998, silently dropping the term from a
    converged loss.
  - ``mask`` applies to the per-pixel terms only. SSIM and LPIPS aggregate
    over spatial neighbourhoods, so a masked pixel still leaks into its
    neighbours' statistics; masking them would need a windowed-coverage
    correction, which neither backend exposes.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# 11-tap Gaussian, sigma = 1.5, normalized — the SSIM paper's window and a
# verbatim copy of fused-ssim's `__constant__ float cGauss[11]` (ssim.cu:12,
# csrc_musa/ssim.mu:12). Hardcoded on both sides so the two backends cannot
# drift apart through a recomputed window.
_SSIM_WINDOW = (
    0.001028380123898387,
    0.0075987582094967365,
    0.036000773310661316,
    0.10936068743467331,
    0.21300552785396576,
    0.26601171493530273,
    0.21300552785396576,
    0.10936068743467331,
    0.036000773310661316,
    0.0075987582094967365,
    0.001028380123898387,
)
_SSIM_HALO = len(_SSIM_WINDOW) // 2
_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2

_LPIPS_CACHE: dict[tuple, torch.nn.Module] = {}


def _as_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
    """Fold any leading batch/view dims into one: (..., C, H, W) -> (N, C, H, W)."""
    if x.dim() < 3:
        raise ValueError(f"{name} must be (..., C, H, W), got shape {tuple(x.shape)}")
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x.reshape(-1, *x.shape[-3:])


def _align_mask(mask: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
    """Broadcast a pixel mask to the error's shape.

    Accepts the mask with or without the channel axis — (..., H, W) or
    (..., 1, H, W) or the full (..., C, H, W) — so callers can pass a plain
    per-pixel validity map without reshaping it first.
    """
    if mask.dim() == err.dim() - 1:
        mask = mask.unsqueeze(-3)
    if mask.dim() != err.dim():
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} does not align with images of "
            f"shape {tuple(err.shape)}; expected (..., H, W) or (..., C, H, W)"
        )
    return mask.to(err.dtype)


def _reduce(err: torch.Tensor, mask: torch.Tensor | None, reduction: str) -> torch.Tensor:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    if mask is not None:
        err = err * _align_mask(mask, err)
    if reduction == "none":
        return err
    if reduction == "sum":
        return err.sum()
    if mask is None:
        return err.mean()
    # Mean over *masked* elements, counting each channel of a kept pixel. An
    # all-masked-out batch returns 0 rather than NaN, so one degenerate sample
    # cannot poison a whole accumulated gradient.
    count = _align_mask(mask, err).expand_as(err).sum()
    return err.sum() / count.clamp(min=1.0)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean absolute error between rendered and ground-truth images.

    Args:
        pred: (..., C, H, W) rendered images.
        target: same shape as ``pred``, ground truth.
        mask: optional (..., H, W) or (..., C, H, W) weights — bool or float;
            with ``reduction="mean"`` the average is over kept elements only.
        reduction: "mean", "sum" or "none" (per-element error).
    """
    return _reduce((pred - target).abs(), mask, reduction)


def l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean squared error (MSE) between rendered and ground-truth images.

    Same arguments as :func:`l1_loss`. Note this is the *squared* error, not
    its root: it is the plain MSE that 3DGS-style photometric objectives use.
    """
    return _reduce((pred - target).square(), mask, reduction)


def _gaussian_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = torch.tensor(_SSIM_WINDOW, device=device, dtype=dtype)
    kernel_2d = torch.outer(w, w)
    return kernel_2d.expand(channels, 1, *kernel_2d.shape).contiguous()


def _ssim_torch(pred: torch.Tensor, target: torch.Tensor, padding: str) -> torch.Tensor:
    """Reference SSIM in pure torch — the fused kernel's math, op for op.

    Separability is not exploited (an 11x11 grouped conv is cheap enough for a
    fallback path); what matters is matching the kernel's conventions: zero
    padding outside the image and ``max(0, E[x^2] - E[x]^2)`` variances.
    """
    channels = pred.shape[1]
    kernel = _gaussian_kernel(channels, pred.device, pred.dtype)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=_SSIM_HALO, groups=channels)

    mu1, mu2 = blur(pred), blur(target)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    # fused-ssim clamps the two variances (ssim.cu:246-247) but not the
    # covariance; float cancellation can otherwise make them slightly negative.
    sigma1_sq = (blur(pred * pred) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (blur(target * target) - mu2_sq).clamp(min=0.0)
    sigma12 = blur(pred * target) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + _SSIM_C1) * (2.0 * sigma12 + _SSIM_C2)
    denominator = (mu1_sq + mu2_sq + _SSIM_C1) * (sigma1_sq + sigma2_sq + _SSIM_C2)
    ssim_map = numerator / denominator
    if padding == "valid":
        ssim_map = ssim_map[:, :, _SSIM_HALO:-_SSIM_HALO, _SSIM_HALO:-_SSIM_HALO]
    return ssim_map.mean()


def _at_least_fp32(x: torch.Tensor) -> torch.Tensor:
    """Promote bf16/fp16 to fp32, leave fp32/fp64 alone.

    SSIM squares its inputs and subtracts near-equal means, so the variance
    terms cancel catastrophically in bf16 (~3 decimal digits of mantissa).
    The cast is differentiable, so gradients still reach a bf16 caller.
    """
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _fused_ssim_available(device: torch.device) -> bool:
    """True when the compiled fused-SSIM kernel can run on ``device``."""
    if device.type not in ("cuda", "musa"):
        return False
    try:
        import fused_ssim
    except ImportError:
        return False
    # `fused_ssim/__init__.py` binds the extension symbols inside per-backend
    # branches, so an import that found no usable device leaves them undefined
    # and only fails at call time. Probe for the symbol, not the module.
    return hasattr(fused_ssim, "fusedssim")


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    padding: str = "same",
    backend: str = "auto",
) -> torch.Tensor:
    """Mean structural similarity in [-1, 1] — 1.0 for identical images.

    Args:
        pred: (..., C, H, W) rendered images in [0, 1]. Gradients flow here.
        target: same shape as ``pred``, ground truth. Receives *no* gradient
            under the fused backend.
        padding: "same" keeps the full map (border windows see zeros outside
            the image, as the kernel does); "valid" drops the 5-pixel border,
            and so requires images larger than 10 pixels on both axes.
        backend: "auto" picks the fused kernel on CUDA/MUSA and the torch
            reference elsewhere; "fused" and "torch" force one.

    Leading batch/view dims are flattened, so (B, V, 3, H, W) scores all B*V
    images at once and averages over them.

    Computed in fp32 with autocast disabled whatever the caller's dtype, so
    the score is identical inside and outside an autocast region and the two
    backends stay interchangeable. bf16 inputs are accepted and promoted;
    gradients flow back through the cast.
    """
    if padding not in ("same", "valid"):
        raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    pred, target = _as_bchw(pred, "pred"), _as_bchw(target, "target")
    height, width = pred.shape[-2:]
    if padding == "valid" and min(height, width) <= 2 * _SSIM_HALO:
        # The crop would leave nothing. An empty mean is NaN on CPU but 0.0 on
        # musa, i.e. a *perfect* loss for images that may be nothing alike —
        # too quiet to let through.
        raise ValueError(
            f"padding='valid' drops {_SSIM_HALO} pixels per side, which empties a "
            f"{height}x{width} image; use padding='same' or images larger than "
            f"{2 * _SSIM_HALO} pixels"
        )

    if backend == "auto":
        backend = "fused" if _fused_ssim_available(pred.device) else "torch"
    if backend not in ("fused", "torch"):
        raise ValueError(f"backend must be 'auto', 'fused' or 'torch', got {backend!r}")

    # Always fp32, whatever the caller's dtype or autocast state. The fused
    # kernel accepts bf16/fp16 and keeps its own statistics in fp32 either way,
    # but it returns the score in the input dtype — and bf16 quantizes that to a
    # 3.9e-3 grid, drifting ~1.6e-3 from the torch path and flattening
    # ``1 - ssim`` to exactly 0.0 above SSIM ~0.998. The cast buys that back for
    # ~4% of the op's runtime.
    with torch.autocast(device_type=pred.device.type, enabled=False):
        if backend == "torch":
            return _ssim_torch(_at_least_fp32(pred), _at_least_fp32(target), padding)
        from fused_ssim import fused_ssim

        # `train` allocates the backward buffers; skip them under no_grad/eval.
        train = torch.is_grad_enabled() and pred.requires_grad
        return fused_ssim(
            pred.float().contiguous(), target.float().contiguous(), padding=padding, train=train
        )


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    padding: str = "same",
    backend: str = "auto",
) -> torch.Tensor:
    """``1 - ssim(pred, target)`` — 0.0 for identical images. See :func:`ssim`."""
    return 1.0 - ssim(pred, target, padding=padding, backend=backend)


def get_lpips(
    net: str = "vgg",
    device: torch.device | str | None = None,
    pretrained_trunk: bool = True,
    model_path: str | None = None,
) -> torch.nn.Module:
    """Build (once) and cache a frozen LPIPS network on ``device``.

    Cached per (net, device, trunk) so a training loop pays the construction
    and the ImageNet-weight download only once. The returned module is in
    eval mode with ``requires_grad=False`` on every parameter, and is
    deliberately *not* registered under any model, keeping it out of
    ``state_dict()`` and away from FSDP2 sharding.

    Args:
        net: trunk architecture — "vgg" (the default here: smoother gradients
            for a training loss, and the only trunk that survives a
            high-resolution forward on MUSA) or "alex" (faster, the metric
            most reported numbers use). On MUSA "alex" returns NaN at large
            spatial sizes — reproduced at 832x1296 with finite trunk
            activations and a correct CPU answer for the same input, so it is
            a device kernel fault, not bad pixels. It is fine at 512.
        device: where to place the network; defaults to CPU.
        pretrained_trunk: load ImageNet weights (needs network access on the
            first call). False gives a random trunk — for shape/gradient tests
            only, the distance it produces is meaningless.
        model_path: explicit path to the calibrated linear weights; defaults
            to the ``.pth`` files bundled in the perceptualsimilarity package.
    """
    try:
        import lpips as lpips_pkg
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires the lpips package: `pip install lpips==0.1.4`"
        ) from exc

    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (net, str(device), pretrained_trunk, model_path)
    if key not in _LPIPS_CACHE:
        model = lpips_pkg.LPIPS(
            net=net,
            pnet_rand=not pretrained_trunk,
            model_path=model_path,
            verbose=False,
        )
        _LPIPS_CACHE[key] = model.eval().requires_grad_(False).to(device)
    return _LPIPS_CACHE[key]


def lpips_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "vgg",
    pretrained_trunk: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Mean LPIPS perceptual distance — 0.0 for identical images.

    Args:
        pred: (..., 3, H, W) rendered images. CLAMPED to [0, 1] internally --
            see below. Gradients flow here.
        target: same shape as ``pred``, ground truth (detached internally).
        net: LPIPS trunk, see :func:`get_lpips`.
        pretrained_trunk: see :func:`get_lpips`.
        chunk_size: images per forward pass. The trunk activations dominate
            memory at training resolution, so a (B, V, 3, H, W) batch with
            many views may need chunking; None runs all B*V at once.

    The trunk runs in fp32 with autocast disabled: its calibration is fp32 and
    the layerwise feature normalization is numerically touchy in bf16.

    **Why the clamp.** LPIPS is defined on images in [0, 1] -- ``normalize=True``
    below rescales that range to the [-1, 1] the trunk was calibrated on. A
    rasterizer output is not bounded: gsplat returns an alpha-weighted sum, and a
    from-scratch gaussian head early in training emits pixels far outside the
    range. Those propagate through the trunk's feature normalisation
    (``x / (sqrt(sum(x^2)) + eps)``) and come back nan, which then poisons the
    whole loss.

    Measured on a 9-scene mip-NeRF-360 run: 1453 of 9000 steps (16%) went
    non-finite, 90% of them attributed to this term, evenly spread across all
    ranks and showing no decline over 9000 steps. l1 and l2 saw the same renders
    and stayed finite -- only the perceptual term broke.

    Clamping is contract-correct rather than a workaround: the term is undefined
    outside [0, 1], so feeding it out-of-range values was the bug. It costs the
    LPIPS gradient on out-of-range pixels (clamp's gradient is zero there), which
    is acceptable because l1/l2 still pull those pixels back into range -- and a
    pixel at 40.0 does not need a perceptual nudge, it needs to stop being 40.0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    pred, target = _as_bchw(pred, "pred"), _as_bchw(target, "target")
    if pred.shape[1] != 3:
        raise ValueError(f"LPIPS expects 3-channel RGB, got {pred.shape[1]} channels")

    model = get_lpips(net=net, device=pred.device, pretrained_trunk=pretrained_trunk)
    step = chunk_size if chunk_size is not None else pred.shape[0]
    if step < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    with torch.autocast(device_type=pred.device.type, enabled=False):
        # clamp BEFORE the trunk, and nan_to_num first: clamp leaves nan as nan
        # (min/max propagate it), so an already-nan pixel would still reach the
        # feature normalisation. Order matters.
        pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        target = target.float().detach().clamp(0.0, 1.0)
        # normalize=True: inputs are [0, 1] and LPIPS rescales to [-1, 1].
        # Weight each chunk by its size so the mean is over images, not chunks.
        total = sum(
            model(pred[i : i + step], target[i : i + step], normalize=True).sum()
            for i in range(0, pred.shape[0], step)
        )
    return total / pred.shape[0]


@dataclass
class LossWeights:
    """Blend of the four terms. Zero-weight terms are never evaluated.

    The defaults are 3DGS's photometric objective — ``0.8*L1 + 0.2*(1-SSIM)``
    — with perceptual and MSE terms available but off. Feed-forward splatting
    models tend to want LPIPS on (AnySplat trains MSE + 0.05*LPIPS); which
    blend suits a given schedule is a training decision, not a default.
    """

    l1: float = 0.8
    l2: float = 0.0
    ssim: float = 0.2
    lpips: float = 0.0


def photometric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights | None = None,
    mask: torch.Tensor | None = None,
    ssim_backend: str = "auto",
    lpips_net: str = "vgg",
    lpips_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the enabled terms.

    Returns ``(total, terms)`` where ``terms`` holds each *unweighted* loss
    that was computed, keyed "l1"/"l2"/"ssim"/"lpips" — log those directly and
    they stay comparable across runs that change the weights.

    ``mask`` reaches the per-pixel terms only (see the module docstring).
    """
    weights = weights or LossWeights()
    terms: dict[str, torch.Tensor] = {}

    if weights.l1:
        terms["l1"] = l1_loss(pred, target, mask=mask)
    if weights.l2:
        terms["l2"] = l2_loss(pred, target, mask=mask)
    if weights.ssim:
        terms["ssim"] = ssim_loss(pred, target, backend=ssim_backend)
    if weights.lpips:
        terms["lpips"] = lpips_loss(pred, target, net=lpips_net, chunk_size=lpips_chunk_size)

    if not terms:
        raise ValueError("every loss weight is zero — nothing to optimize")

    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
