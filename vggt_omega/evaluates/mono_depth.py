"""Monocular-depth evaluation as a :class:`BaseMetric` family for VGGT-Omega.

:class:`MonoDepthMetric` adapts the depth evaluation in ``mono_depth/metrics.py``
to the template-method framework in :mod:`vggt_omega.evaluates.base_metric`. From
a predicted vs. ground-truth depth map it reports, over the valid pixels, the two
standard monocular-depth metric families:

* **abs_rel** -- the distribution of the per-pixel absolute relative error
  ``|pred - gt| / gt``, reduced to ``reduce_ops`` statistics. Its ``mean`` is the
  canonical Abs Rel (low good).
* **delta**   -- ``delta < 1.25^i`` accuracy: the fractions of pixels whose
  worst-direction ratio ``max(pred / gt, gt / pred)`` is below ``1.25`` / ``1.25^2``
  / ``1.25^3`` (the ``delta1`` / ``delta2`` / ``delta3`` family, high good).

VGGT predicts depth only up to a global scale, whereas ground truth is metric, so
by default the prediction is first aligned to GT by a single per-image median
scale ``s = median(gt) / median(pred)`` -- the standard scale-invariant protocol.
Pass ``align="none"`` to score the raw prediction.

Everything lives on :class:`MonoDepthMetric`; the module exposes nothing else.
Lifecycle mapping onto :class:`BaseMetric`:

* :meth:`check`      -- validate the depth maps (matching ``(H, W)`` shape,
  optional mask, ``align`` mode, threshold, ``reduce_ops``).
* :meth:`preprocess` -- build the valid-pixel mask, median-align the prediction,
  and cache the per-pixel relative error and depth ratio shared by both metrics.
* ``@metric`` methods :meth:`abs_rel` / :meth:`delta`.
* :meth:`visualize`  -- when ``run(vis_path=...)`` is given a prefix, write a
  GT / aligned-prediction / abs-rel-error panel.

Usage::

    res = MonoDepthMetric(gt_depth, pred_depth).run()
    # res == {
    #   "abs_rel": {"rmse": .., "mean": <AbsRel>, "median": .., ...},
    #   "delta":   {"delta1": .., "delta2": .., "delta3": ..},
    # }

Depth maps follow the dataset convention (``datasets/base_dataset.py``): float
``(H, W)`` where ``0`` marks invalid pixels and ``< 0`` marks sky. Only
strictly-positive, finite GT pixels with a strictly-positive, finite prediction
are scored. Argument order follows :class:`BaseMetric`: **ground truth first,
prediction second** (the reverse of the functional ``evaluate_mono_depth``).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vggt_omega.evaluates.base_metric import BaseMetric, metric  # noqa: E402


class MonoDepthMetric(BaseMetric):
    """Abs-Rel / delta monocular-depth metrics over the valid pixels.

    Args:
        gt: ground-truth depth ``(H, W)``; ``0`` = invalid, ``< 0`` = sky.
        pred: predicted depth, same shape as ``gt``.
        mask: optional boolean array of pixels to keep, AND-ed with the
            positive-finite validity mask.
        align: scale alignment before scoring -- ``"median"`` (default) rescales
            the prediction by ``median(gt) / median(pred)`` over valid pixels;
            ``"none"`` scores the raw prediction.
        reduce_ops: statistics to surface for ``abs_rel``; any subset of
            ``rmse``/``mean``/``median``/``std``/``min``/``max``/``sse``.
        delta_threshold: base ratio factor for the ``delta`` accuracies
            (canonical ``1.25``; ``delta2``/``delta3`` use its square / cube).
        min_num_valid: minimum valid pixels required to score.

    Reports (via :meth:`run`)::

        {"abs_rel": {<reduce_ops>}, "delta": {"delta1", "delta2", "delta3"}}

    The applied median ``scale`` and the number of scored pixels are available
    after :meth:`run` as ``self.depth_scale`` / ``self.num_valid``.
    """

    # canonical "delta < 1.25" factor; delta2/delta3 use its square/cube.
    _DELTA_THRESHOLD = 1.25
    _VALID_ALIGN = frozenset({"median", "none"})

    _DEFAULT_REDUCE_OPS = ("rmse", "mean", "median", "std", "min", "max")
    _VALID_STATS = frozenset(_DEFAULT_REDUCE_OPS) | {"sse"}

    def __init__(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        mask=None,
        align: str = "median",
        reduce_ops: Sequence[str] | None = None,
        delta_threshold: float = _DELTA_THRESHOLD,
        min_num_valid: int = 1,
    ):
        self.gt = np.asarray(gt, dtype=np.float64)
        self.pred = np.asarray(pred, dtype=np.float64)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self.align = align
        self.reduce_ops = (
            tuple(reduce_ops) if reduce_ops is not None else self._DEFAULT_REDUCE_OPS
        )
        self.delta_threshold = float(delta_threshold)
        self.min_num_valid = int(min_num_valid)

    # ---- lifecycle hooks --------------------------------------------------- #
    def check(self) -> None:
        """Validate the depth maps and options.

        Raises:
            AssertionError: on a non-2D map, shape mismatch, an unknown ``align``
                mode, a mask whose shape differs from the maps, a
                ``delta_threshold`` that is not greater than 1, or an unknown
                reduce op.
        """
        assert self.gt.ndim == 2, f"depth maps must be (H, W); got {self.gt.shape}"
        assert (
            self.pred.shape == self.gt.shape
        ), f"pred/gt shape mismatch: {self.pred.shape} vs {self.gt.shape}"
        assert (
            self.align in self._VALID_ALIGN
        ), f"align must be one of {sorted(self._VALID_ALIGN)}; got {self.align!r}"
        assert self.mask is None or self.mask.shape == self.gt.shape, (
            f"mask shape {None if self.mask is None else self.mask.shape} "
            f"!= depth shape {self.gt.shape}"
        )
        assert (
            self.delta_threshold > 1.0
        ), f"delta_threshold must be > 1; got {self.delta_threshold}"
        unknown = set(self.reduce_ops) - self._VALID_STATS
        assert (
            not unknown
        ), f"unknown reduce_ops {sorted(unknown)}; valid: {sorted(self._VALID_STATS)}"

    def preprocess(self) -> None:
        """Mask invalid pixels and median-align the prediction.

        Builds the validity mask (positive & finite in both maps, AND any
        caller mask -- so the dataset's ``0`` = invalid / ``< 0`` = sky
        convention is honored by the positivity test), median-aligns the
        prediction when requested, and caches the per-pixel relative error and
        worst-direction depth ratio shared by both metrics.
        """
        valid = (
            np.isfinite(self.pred)
            & np.isfinite(self.gt)
            & (self.pred > 0)
            & (self.gt > 0)
        )
        if self.mask is not None:
            valid &= self.mask
        self.valid = valid
        self.num_valid = int(valid.sum())
        assert (
            self.num_valid >= self.min_num_valid
        ), f"need >= {self.min_num_valid} valid pixels; got {self.num_valid}"

        gt = self.gt[valid]
        pred = self.pred[valid]
        self.depth_scale = 1.0
        if self.align == "median":
            self.depth_scale = float(np.median(gt) / np.median(pred))
            pred = pred * self.depth_scale
        self.rel_error = np.abs(pred - gt) / gt
        self.ratio = np.maximum(pred / gt, gt / pred)

    # ---- metrics ----------------------------------------------------------- #
    @metric
    def abs_rel(self) -> dict:
        """Absolute relative error ``|pred - gt| / gt``, reduced to ``reduce_ops``.

        The ``mean`` statistic is the canonical Abs Rel.
        """
        return self._reduce(self.rel_error)

    @metric
    def delta(self) -> dict:
        """``delta < 1.25^i`` accuracies for ``i = 1, 2, 3`` (fractions, high good)."""
        return {
            f"delta{i}": float(np.mean(self.ratio < self.delta_threshold**i))
            for i in (1, 2, 3)
        }

    def _reduce(self, values: np.ndarray) -> dict[str, float]:
        """Reduce a per-pixel error array to the ``reduce_ops`` statistics."""
        v = np.asarray(values, dtype=np.float64)
        stats = {
            "rmse": np.sqrt(np.mean(v**2)),
            "mean": np.mean(v),
            "median": np.median(v),
            "std": np.std(v),
            "min": np.min(v),
            "max": np.max(v),
            "sse": np.sum(v**2),
        }
        return {op: float(stats[op]) for op in self.reduce_ops}

    # ---- visualization ----------------------------------------------------- #
    def visualize(self, vis_path: str | None = None) -> None:
        """Write a GT / aligned-prediction / abs-rel-error panel.

        Writes ``{vis_path}_depth.png``; invalid pixels are rendered blank. No-op
        when ``vis_path`` is ``None`` (the metrics-only case).
        """
        if vis_path is None:
            return
        gt = np.where(self.valid, self.gt, np.nan)
        pred = np.where(self.valid, self.pred * self.depth_scale, np.nan)
        err = np.where(self.valid, np.abs(pred - gt) / gt, np.nan)
        self._plot_depth_panel(gt, pred, err, f"{vis_path}_depth.png")

    @staticmethod
    def _plot_depth_panel(gt, pred, err, out_path: str) -> None:
        """Render side-by-side GT / prediction / abs-rel error maps to one PNG."""
        vmax = float(np.nanmax(gt))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = (
            (axes[0], gt, "ground truth (m)", "viridis", vmax),
            (axes[1], pred, "aligned prediction (m)", "viridis", vmax),
            (axes[2], err, "abs rel error", "magma", None),
        )
        for ax, img, title, cmap, top in panels:
            im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=top)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
