"""Per-view photometric scores over rendered vs. ground-truth images.

Sits with the other metric families rather than in ``gaussian_splat`` because
that is what it is: scoring a render and drawing the result. ``camera_pose``
already pairs a metric with a ``vis_path`` PNG, and this package already forces
a headless matplotlib backend for exactly that reason.

**Why per-view scoring needs its own function.** Everything in
``gaussian_splat.utils.losses`` reduces to one scalar over the whole batch --
``ssim`` averages its flattened views, ``lpips_loss`` divides by the image count.
That is what a training loss wants and precisely what a per-view report cannot
use, so SSIM and LPIPS are evaluated one view at a time here.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gaussian_splat.utils.losses import lpips_loss, ssim  # noqa: E402

#: Shared upper bound for the error panel's colour scale, so a view that renders
#: badly LOOKS worse than one that does not. A per-view autoscale would normalize
#: every sheet to its own worst pixel and hide exactly that.
_ERROR_VMAX_PERCENTILE = 99.0


def per_view_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_backend: str = "auto",
    lpips_net: str = "vgg",
) -> dict[str, np.ndarray]:
    """PSNR (dB), SSIM, LPIPS and MSE for each view, as (V,) numpy arrays.

    Args:
        rendered: (..., 3, H, W) rendered images in [0, 1]; leading dims are
            flattened, so (B, V, 3, H, W) scores all B*V views.
        target: same shape, the ground-truth frames.
        ssim_backend: "auto" / "fused" / "torch", see :func:`losses.ssim`.
        lpips_net: LPIPS trunk, "alex" or "vgg". Match whatever the run was
            trained against or the number is not comparable to its loss curve.
    """
    if rendered.shape != target.shape:
        raise ValueError(
            f"shape mismatch: rendered {tuple(rendered.shape)} vs target {tuple(target.shape)}"
        )
    rendered = rendered.reshape(-1, *rendered.shape[-3:]).float()
    target = target.reshape(-1, *target.shape[-3:]).float()

    mse = (rendered - target).square().mean(dim=(-1, -2, -3))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))
    ssim_per_view = torch.stack([
        ssim(rendered[i : i + 1], target[i : i + 1], backend=ssim_backend)
        for i in range(rendered.shape[0])
    ])
    lpips_per_view = torch.stack([
        lpips_loss(rendered[i : i + 1], target[i : i + 1], net=lpips_net)
        for i in range(rendered.shape[0])
    ])
    return {
        "psnr": psnr.cpu().numpy(),
        "ssim": ssim_per_view.cpu().numpy(),
        "lpips": lpips_per_view.cpu().numpy(),
        "mse": mse.cpu().numpy(),
    }


def _to_hwc(images: torch.Tensor) -> np.ndarray:
    """(V, 3, H, W) tensor in [0, 1] -> (V, H, W, 3) float numpy, clamped."""
    flat = images.reshape(-1, *images.shape[-3:])
    return flat.permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy()


def dump_render_comparison(
    directory: str | Path,
    rendered: torch.Tensor,
    target: torch.Tensor,
    metrics: dict[str, np.ndarray],
    frame_ids: list[int] | None = None,
    dpi: int = 110,
) -> int:
    """Write one ``frame_XXXX.png`` per view: GT | rendered | absolute error.

    The per-view PSNR / SSIM / LPIPS ride in the figure title, so a sheet is
    self-describing once it leaves the output directory. The error panel shares
    one colour scale across every view (the ``_ERROR_VMAX_PERCENTILE`` of all
    errors), which is what makes the sheets comparable to each other.

    Returns the number of files written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    render_hwc, target_hwc = _to_hwc(rendered), _to_hwc(target)
    error = np.abs(render_hwc - target_hwc).mean(axis=-1)  # (V, H, W)
    vmax = float(np.percentile(error, _ERROR_VMAX_PERCENTILE)) if error.size else 1.0
    vmax = max(vmax, 1e-6)

    num_views, height, width = error.shape
    for view in range(num_views):
        # layout is pinned at CREATION, and tight_layout() is never called. Both
        # matter: importing vggt_omega.evaluates pulls in evo, which flips the
        # global figure.constrained_layout.use to True, and swapping layout
        # engines after a colorbar exists is a hard error in matplotlib >= 3.6
        # ("Colorbar layout of new layout engine not compatible"). Declaring the
        # engine up front makes this independent of whatever imported first.
        figure, axes = plt.subplots(
            1, 3, figsize=(3 * width / dpi, height / dpi + 0.75), dpi=dpi,
            layout="constrained",
        )
        axes[0].imshow(target_hwc[view])
        axes[0].set_title("ground truth", fontsize=9)
        axes[1].imshow(render_hwc[view])
        axes[1].set_title("rendered", fontsize=9)
        heat = axes[2].imshow(error[view], cmap="inferno", vmin=0.0, vmax=vmax)
        axes[2].set_title(f"|error| (0-{vmax:.2f})", fontsize=9)
        figure.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.02)
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])

        label = f"view {view:02d}"
        if frame_ids is not None:
            label += f"  (frame {int(frame_ids[view])})"
        figure.suptitle(
            f"{label}    PSNR {metrics['psnr'][view]:.2f} dB    "
            f"SSIM {metrics['ssim'][view]:.4f}    LPIPS {metrics['lpips'][view]:.4f}",
            fontsize=11,
        )
        figure.savefig(directory / f"frame_{view:04d}.png", bbox_inches="tight")
        plt.close(figure)
    return num_views
