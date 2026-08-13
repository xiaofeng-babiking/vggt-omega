"""Per-view photometric scores and the rendered-vs-GT comparison sheets."""

import numpy as np
import pytest
import torch

_V, _H, _W = 3, 24, 32
# LPIPS trunks are fetched by torchvision on first use; vgg16 is the one staged
# for this project's container, so the tests stay runnable offline.
_LPIPS_NET = "vgg"


def _images(seed=0):
    torch.manual_seed(seed)
    return torch.rand(1, _V, 3, _H, _W)


from vggt_omega.evaluates.photometric import (  # noqa: E402
    dump_render_comparison,
    per_view_metrics,
)


def test_per_view_metrics_are_per_view_not_averaged():
    """The point of the module: one score per view, and views differ."""
    target = _images()
    rendered = target.clone()
    rendered[0, 1] = torch.rand(3, _H, _W)  # corrupt exactly one view

    metrics = per_view_metrics(rendered, target, lpips_net=_LPIPS_NET)

    for key in ("psnr", "ssim", "lpips", "mse"):
        assert metrics[key].shape == (_V,), f"{key} should be per-view"
    # Untouched views are identical; the corrupted one is not.
    assert metrics["psnr"][0] > 80.0 and metrics["psnr"][2] > 80.0
    assert metrics["psnr"][1] < 30.0
    assert metrics["ssim"][0] == pytest.approx(1.0, abs=1e-4)
    assert metrics["ssim"][1] < 0.5
    assert metrics["lpips"][0] < 1e-4 < metrics["lpips"][1]


def test_per_view_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        per_view_metrics(_images(), _images()[:, :1], lpips_net=_LPIPS_NET)


def test_dump_render_comparison_writes_one_sheet_per_view(tmp_path):
    target, rendered = _images(), _images(seed=1)
    metrics = {k: np.linspace(0, 1, _V) for k in ("psnr", "ssim", "lpips")}

    written = dump_render_comparison(tmp_path, rendered, target, metrics, frame_ids=[5, 9, 13])

    assert written == _V
    sheets = sorted(tmp_path.glob("frame_*.png"))
    assert [p.name for p in sheets] == ["frame_0000.png", "frame_0001.png", "frame_0002.png"]
    assert all(p.stat().st_size > 0 for p in sheets)


def test_dump_render_comparison_survives_a_global_constrained_layout(tmp_path):
    """Regression: importing this package pulls in evo, which flips the GLOBAL
    figure.constrained_layout.use to True. Figures are then born with a
    constrained engine, and switching engines once a colorbar exists raises
    RuntimeError in matplotlib >= 3.6 -- which took down a whole inference run
    after the forward and every .ply had already been written. The rcParam is set
    here directly rather than by importing evo, so the test pins the condition
    rather than the culprit."""
    import matplotlib

    target, rendered = _images(), _images(seed=1)
    metrics = {k: np.linspace(0, 1, _V) for k in ("psnr", "ssim", "lpips")}

    with matplotlib.rc_context({"figure.constrained_layout.use": True}):
        written = dump_render_comparison(tmp_path, rendered, target, metrics)

    assert written == _V
