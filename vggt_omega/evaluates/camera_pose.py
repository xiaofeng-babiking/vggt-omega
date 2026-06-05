"""Camera-pose evaluation as a :class:`BaseMetric` family for VGGT-Omega.

:class:`CameraPoseMetric` adapts the camera-trajectory evaluation in
``camera_pose/metrics.py`` to the template-method framework in
:mod:`vggt_omega.evaluates.base_metric`. From predicted vs. ground-truth
camera-to-world poses it reports, after a single Umeyama alignment, the
three standard trajectory metrics:

* **ate**       -- Absolute Trajectory Error: translation RMSE after a *global*
  alignment of the whole trajectory (meters).
* **rpe_trans** -- Relative Pose Error, translation: frame-to-frame relative-pose
  translation error (meters).
* **rpe_rot**   -- Relative Pose Error, rotation: frame-to-frame relative-pose
  rotation error (degrees).

Umeyama alignment, APE/RPE statistics and trajectory plotting are delegated to
`evo <https://github.com/MichaelGrupp/evo>`_. The alignment is chosen by
``align_scale``: a rigid ``se3`` alignment (rotation + translation, no scale)
when ``False``, or a scale-correcting ``sim3`` alignment when ``True``. VGGT
predicts camera poses only up to a global scale, whereas ground truth
(Sintel / TUM / ...) is metric, so ``align_scale=True`` is the apples-to-apples
setting while ``align_scale=False`` exposes any scale drift.

Everything -- constants, pose plumbing, metric computation and plotting -- lives
on :class:`CameraPoseMetric`; the module exposes nothing else. Lifecycle mapping
onto :class:`BaseMetric`:

* :meth:`check`      -- validate both pose arrays (shape, finiteness, matching
  length, minimum count) and the requested ``reduce_ops``.
* :meth:`preprocess` -- build the pose trajectories and run the Umeyama
  alignment of the prediction onto the ground truth once (the expensive step
  shared by all three metrics).
* ``@metric`` methods :meth:`ate` / :meth:`rpe_trans` / :meth:`rpe_rot` --
  process the aligned trajectories into evo APE/RPE statistics, reduced to
  ``reduce_ops``.
* :meth:`visualize`  -- when ``run(vis_path=...)`` is given a prefix, write the
  GT-vs-aligned-prediction overlay and the per-pose ATE colormap.

Usage::

    res = CameraPoseMetric(gt_c2w, pred_c2w, align_scale=True).run()
    # res == {
    #   "ate":       {"rmse": ..., "mean": ..., ...},
    #   "rpe_rot":   {"rmse": ..., ...},   # degrees
    #   "rpe_trans": {"rmse": ..., ...},   # meters
    # }

    # report only a subset of statistics, and write trajectory plots:
    CameraPoseMetric(
        gt_c2w, pred_c2w, reduce_ops=["rmse", "mean"]
    ).run(vis_path="/tmp/seq01")
    # -> /tmp/seq01_traj.png, /tmp/seq01_ate.png

Inputs are **camera-to-world** poses of shape ``(N, 3, 4)`` or ``(N, 4, 4)``
whose translation column is the camera center in world coordinates; they are
consumed directly as trajectory positions, with no internal world-to-camera
inversion. Note the argument order follows :class:`BaseMetric`: **ground truth
first, prediction second** (the reverse of the functional
``evaluate_camera_pose``).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from evo.core import metrics as evo_metrics  # noqa: E402
from evo.core.trajectory import PosePath3D  # noqa: E402
from evo.tools import plot as evo_plot  # noqa: E402

from vggt_omega.evaluates.base_metric import BaseMetric, metric  # noqa: E402


class CameraPoseMetric(BaseMetric):
    """ATE / RPE camera-trajectory metrics under a single Umeyama alignment.

    Args:
        gt: ground-truth camera-to-world poses ``(N, 3, 4)`` or ``(N, 4, 4)``.
        pred: predicted camera-to-world poses, index-aligned 1:1 with ``gt``
            (same shape/length).
        min_num_poses: minimum number of poses required; Umeyama alignment is
            under-determined with fewer than 3 correspondences.
        rpe_delta: frame gap for the Relative Pose (RPE) metrics, in frames.
        reduce_ops: statistics to surface for every metric; any subset of
            ``rmse``/``mean``/``median``/``std``/``min``/``max``/``sse``.
            Defaults to all but ``sse``.
        align_scale: if ``True`` the Umeyama alignment also corrects scale
            (``sim3``); if ``False`` it is rigid (``se3``).

    Reports (via :meth:`run`)::

        {
          "ate":       {<reduce_ops>},   # meters
          "rpe_rot":   {<reduce_ops>},   # degrees
          "rpe_trans": {<reduce_ops>},   # meters
        }

    The Umeyama ``scale`` mapping prediction onto GT is available after
    :meth:`run` as ``self.umeyama_scale`` (``1.0`` when ``align_scale=False``).
    """

    # Umeyama alignment is under-determined with fewer than 3 correspondences.
    _MIN_POSES = 3

    # Statistics evo's ``get_all_statistics`` exposes; the default ``reduce_ops``
    # drops the rarely-useful "sse".
    _DEFAULT_REDUCE_OPS = ("rmse", "mean", "median", "std", "min", "max")
    _VALID_STATS = frozenset(_DEFAULT_REDUCE_OPS) | {"sse"}

    # Homogeneous bottom row appended when promoting (N, 3, 4) -> (N, 4, 4).
    _BOTTOM_ROW = np.array([0.0, 0.0, 0.0, 1.0])

    def __init__(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        min_num_poses: int = _MIN_POSES,
        rpe_delta: int = 1,
        reduce_ops: Sequence[str] | None = None,
        align_scale: bool = False,
    ):
        self.gt = np.asarray(gt, dtype=np.float64)
        self.pred = np.asarray(pred, dtype=np.float64)
        self.min_num_poses = int(min_num_poses)
        self.rpe_delta = int(rpe_delta)
        self.reduce_ops = (
            tuple(reduce_ops) if reduce_ops is not None else self._DEFAULT_REDUCE_OPS
        )
        self.align_scale = bool(align_scale)

    # ---- lifecycle hooks --------------------------------------------------- #
    def check(self) -> None:
        """Validate both pose arrays and the requested ``reduce_ops``.

        Raises:
            AssertionError: on bad shape, non-finite values, length mismatch,
                fewer than ``min_num_poses`` poses, or an unknown reduce op.
        """
        assert self.gt.ndim == self.pred.ndim == 3, "poses must be (N, R, 4)"
        assert (
            self.gt.shape == self.pred.shape
        ), f"pred/gt shape mismatch: {self.pred.shape} vs {self.gt.shape}"
        assert self.gt.shape[1:] in {
            (3, 4),
            (4, 4),
        }, f"poses must be (N, 3, 4) or (N, 4, 4); got {self.gt.shape}"
        assert (
            self.gt.shape[0] >= self.min_num_poses
        ), f"need >= {self.min_num_poses} poses for alignment; got {self.gt.shape[0]}"
        assert (
            np.isfinite(self.gt).all() and np.isfinite(self.pred).all()
        ), "poses contain non-finite values"
        unknown = set(self.reduce_ops) - self._VALID_STATS
        assert (
            not unknown
        ), f"unknown reduce_ops {sorted(unknown)}; valid: {sorted(self._VALID_STATS)}"

    def preprocess(self) -> None:
        """Build the pose trajectories and align prediction onto GT.

        The Umeyama alignment is the expensive step shared by all three metrics,
        so it runs once here -- after :meth:`check` has validated the inputs.
        ``correct_scale`` follows ``align_scale``: rigid ``se3`` when ``False``,
        scale-correcting ``sim3`` when ``True``. The recovered rotation /
        translation / scale are cached on ``self``.
        """
        self.gt_traj = self._to_pose_path(self.gt)
        self.pred_traj = self._to_pose_path(self.pred)
        self.umeyama_rot, self.umeyama_trans, self.umeyama_scale = self.pred_traj.align(
            self.gt_traj, correct_scale=self.align_scale
        )

    # ---- metrics ----------------------------------------------------------- #
    @metric
    def ate(self) -> dict:
        """Absolute Trajectory Error (translation, meters), reduced to ``reduce_ops``."""
        return self._reduce(self._create_evo_ape())

    @metric
    def rpe_rot(self) -> dict:
        """Relative Pose Error (rotation, degrees), reduced to ``reduce_ops``."""
        return self._reduce(
            self._create_evo_rpe(evo_metrics.PoseRelation.rotation_angle_deg)
        )

    @metric
    def rpe_trans(self) -> dict:
        """Relative Pose Error (translation, meters), reduced to ``reduce_ops``."""
        return self._reduce(
            self._create_evo_rpe(evo_metrics.PoseRelation.translation_part)
        )

    # ---- visualization ----------------------------------------------------- #
    def visualize(self, vis_path: str | None = None) -> None:
        """Write trajectory plots under ``{vis_path}`` (a path *prefix*).

        Writes ``{vis_path}_traj.png`` (GT vs. aligned prediction) and
        ``{vis_path}_ate.png`` (prediction colored by per-pose ATE). No-op when
        ``vis_path`` is ``None`` (the metrics-only case).
        """
        if vis_path is None:
            return
        ate_error = np.asarray(self._create_evo_ape().error, dtype=float)
        mode = "Sim3" if self.align_scale else "SE3"
        self._plot_trajectories(
            self.gt_traj,
            self.pred_traj,
            f"{vis_path}_traj.png",
            title=f"Camera trajectory ({mode}-aligned)",
        )
        self._plot_trajectories(
            self.gt_traj,
            self.pred_traj,
            f"{vis_path}_ate.png",
            ape_error=ate_error,
        )

    # ---- evo metric construction (internal) -------------------------------- #
    def _create_evo_ape(self) -> evo_metrics.APE:
        """Processed evo APE on translation (i.e. ATE) of pred vs. GT trajectory."""
        ape = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape.process_data((self.gt_traj, self.pred_traj))
        return ape

    def _create_evo_rpe(self, relation: evo_metrics.PoseRelation) -> evo_metrics.RPE:
        """Processed evo RPE for ``relation`` at ``rpe_delta``-frame spacing."""
        rpe = evo_metrics.RPE(
            relation,
            delta=self.rpe_delta,
            delta_unit=evo_metrics.Unit.frames,
            all_pairs=False,
        )
        rpe.process_data((self.gt_traj, self.pred_traj))
        return rpe

    def _reduce(self, metric_obj) -> dict[str, float]:
        """Reduce a processed evo metric to the ``reduce_ops`` statistics."""
        stats = metric_obj.get_all_statistics()
        return {op: float(stats[op]) for op in self.reduce_ops}

    # ---- pose plumbing (internal) ------------------------------------------ #
    @classmethod
    def _to_pose_path(cls, tf_mats: np.ndarray) -> PosePath3D:
        """Build an evo ``PosePath3D`` from camera poses, promoting to 4x4.

        ``(N, 3, 4)`` inputs are promoted to ``(N, 4, 4)`` -- the homogeneous
        form evo requires -- and otherwise consumed as-is (no inversion).
        """
        tf_mats = np.asarray(tf_mats, dtype=np.float64)
        if tf_mats.ndim == 3 and tf_mats.shape[1:] == (3, 4):
            bottom = np.tile(cls._BOTTOM_ROW, (tf_mats.shape[0], 1, 1))
            tf_mats = np.concatenate([tf_mats, bottom], axis=1)
        return PosePath3D(poses_se3=list(tf_mats))

    # ---- plotting (internal) ----------------------------------------------- #
    @staticmethod
    def _plot_trajectories(
        traj_ref: PosePath3D,
        traj_est: PosePath3D,
        out_path: str,
        ape_error=None,
        plot_mode="xy",
        title: str | None = None,
    ) -> None:
        """Plot two aligned trajectories and save to ``out_path`` (one PNG).

        Args:
            traj_ref: reference (ground-truth) ``PosePath3D``.
            traj_est: aligned estimated ``PosePath3D``.
            out_path: PNG file to write.
            ape_error: optional per-pose ATE array. ``None`` draws a GT-vs-estimate
                overlay; otherwise the estimate is colored by this error.
            plot_mode: projection plane, an ``evo`` ``PlotMode`` or its name
                (``"xy"``, ``"xz"``, ``"xyz"``, ...). Default top-down ``"xy"``.
            title: figure title; a sensible default is chosen per plot type.
        """
        if isinstance(plot_mode, str):
            plot_mode = evo_plot.PlotMode[plot_mode]

        fig = plt.figure(figsize=(8, 8))
        ax = evo_plot.prepare_axis(fig, plot_mode)
        evo_plot.traj(
            ax,
            plot_mode,
            traj_ref,
            style="--",
            color="gray",
            label="reference (GT)",
            plot_start_end_markers=True,
        )

        if ape_error is None:
            evo_plot.traj(
                ax,
                plot_mode,
                traj_est,
                style="-",
                color="tab:blue",
                label="aligned prediction",
                plot_start_end_markers=True,
            )
            ax.legend(frameon=True, fontsize="small")
            ax.set_title(title or "Camera trajectory (aligned)")
        else:
            ape_error = np.asarray(ape_error, dtype=float)
            min_map = float(ape_error.min())
            # Guard the degenerate (near-)constant-error case so the colormap
            # normalization always has a non-zero span.
            max_map = max(float(ape_error.max()), min_map + 1e-12)
            evo_plot.traj_colormap(
                ax,
                traj_est,
                ape_error,
                plot_mode,
                min_map=min_map,
                max_map=max_map,
                title=title or "Absolute Trajectory Error per pose (m)",
                fig=fig,
            )

        # NOTE: don't call fig.tight_layout() here -- evo's traj_colormap installs
        # a colorbar layout engine that tight_layout refuses to override.
        # bbox_inches achieves the same trimming and is compatible with either
        # branch.
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
