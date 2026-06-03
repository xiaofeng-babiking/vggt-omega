"""Point-cloud reconstruction evaluation as a :class:`BaseMetric` family.

:class:`PointcloudMetric` adapts the surface-reconstruction evaluation in
``pointcloud/metrics.py`` to the template-method framework in
:mod:`vggt_omega.evaluates.base_metric`. From a predicted vs. ground-truth point
cloud it reports the standard DTU / Tanks-and-Temples metrics, each as a small
dict of ``reduce_ops`` statistics (so the DTU ``median`` and the usual ``mean``
come out of one pass):

* **accuracy**           -- nearest-neighbor distance pred->GT (how close the
  prediction is to the GT surface; low good).
* **completeness**       -- nearest-neighbor distance GT->pred (how well the
  prediction covers the GT surface; low good).
* **chamfer**            -- ``(accuracy + completeness) / 2`` per statistic.
* **normal_consistency** -- ``|<n_a, n_b>|`` over matched normals, both
  directions, in ``[0, 1]`` (high good; ``mean`` is the headline). Normals are
  taken from the inputs when supplied, otherwise estimated per point by local-PCA.
* **fscore** -- ``{precision, recall, fscore}`` at a distance ``threshold``
  (``NaN`` when no ``threshold`` is set).

VGGT predicts geometry only up to a global scale/pose. Pass ``align="icp"`` to
register the prediction onto GT (point-to-point ICP, optional Umeyama scale)
before scoring; the default ``align="none"`` scores the clouds as given.

Everything lives on :class:`PointcloudMetric`; the module exposes nothing else.
Lifecycle mapping onto :class:`BaseMetric`:

* :meth:`check`      -- validate the cloud shapes, optional normals, ``align``
  mode, ``reduce_ops`` and threshold.
* :meth:`preprocess` -- drop non-finite rows, optionally subsample, optionally
  ICP-register the prediction, estimate missing normals, and build the KD-trees
  for the nearest-neighbor distances shared by every metric.
* ``@metric`` methods :meth:`accuracy` / :meth:`completeness` / :meth:`chamfer` /
  :meth:`normal_consistency` / :meth:`fscore`.
* :meth:`visualize`  -- when ``run(vis_path=...)`` is given a prefix, write a
  top-down scatter of GT vs. prediction colored by accuracy.

Both clouds are ``(N, 3)`` / ``(M, 3)`` float arrays in the **same world frame**;
non-finite rows (the ``NaN`` left where depth was invalid) are dropped. Argument
order follows :class:`BaseMetric`: **ground truth first, prediction second** (the
reverse of the functional ``evaluate_pointcloud``).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from vggt_omega.evaluates.base_metric import BaseMetric, metric  # noqa: E402


class PointcloudMetric(BaseMetric):
    """Accuracy / completeness / chamfer / normal-consistency / F-score metrics.

    Args:
        gt: ground-truth cloud ``(M, 3)`` in world frame.
        pred: predicted cloud ``(N, 3)`` in the same frame.
        gt_normals: optional ``(M, 3)`` GT normals; estimated by local-PCA when
            omitted.
        pred_normals: optional ``(N, 3)`` predicted normals; estimated when
            omitted.
        reduce_ops: statistics to surface for the distance / normal metrics; any
            subset of ``rmse``/``mean``/``median``/``std``/``min``/``max``/``sse``
            (``median`` is the DTU reduction, ``mean`` the usual one).
        normal_k: neighborhood size for PCA normal estimation.
        threshold: distance for ``fscore``'s precision/recall; when ``None`` the
            F-score block is ``NaN``.
        align: ``"none"`` (default, score as given) or ``"icp"`` (register the
            prediction onto GT first).
        align_scale: when ``align="icp"``, also solve for a global scale.
        max_points: if set, randomly subsample each cloud to at most this many
            points before scoring.
        min_num_points: minimum finite points required in each cloud.
        seed: RNG seed for subsampling / ICP.

    Reports (via :meth:`run`)::

        {
          "accuracy":           {<reduce_ops>},
          "completeness":       {<reduce_ops>},
          "chamfer":            {<reduce_ops>},
          "normal_consistency": {<reduce_ops>},   # mean is the headline, in [0, 1]
          "fscore":             {"precision", "recall", "fscore"},
        }

    ``self.num_pred`` / ``self.num_gt`` (points scored) and ``self.icp_scale``
    (``1.0`` unless ``align="icp"``) are available after :meth:`run`.
    """

    # Neighbors used for local-PCA normal estimation when normals aren't supplied.
    _DEFAULT_NORMAL_K = 30
    # Clouds smaller than this can't support a 3-point local PCA / a meaningful fit.
    _MIN_POINTS = 3
    _VALID_ALIGN = frozenset({"none", "icp"})

    _DEFAULT_REDUCE_OPS = ("rmse", "mean", "median", "std", "min", "max")
    _VALID_STATS = frozenset(_DEFAULT_REDUCE_OPS) | {"sse"}

    def __init__(
        self,
        gt,
        pred,
        gt_normals=None,
        pred_normals=None,
        reduce_ops: Sequence[str] | None = None,
        normal_k: int = _DEFAULT_NORMAL_K,
        threshold: float | None = None,
        align: str = "none",
        align_scale: bool = True,
        max_points: int | None = None,
        min_num_points: int = _MIN_POINTS,
        seed: int = 0,
    ):
        self.gt = np.asarray(gt, dtype=np.float64)
        self.pred = np.asarray(pred, dtype=np.float64)
        self.gt_normals_in = (
            None if gt_normals is None else np.asarray(gt_normals, dtype=np.float64)
        )
        self.pred_normals_in = (
            None if pred_normals is None else np.asarray(pred_normals, dtype=np.float64)
        )
        self.reduce_ops = (
            tuple(reduce_ops) if reduce_ops is not None else self._DEFAULT_REDUCE_OPS
        )
        self.normal_k = int(normal_k)
        self.threshold = None if threshold is None else float(threshold)
        self.align = align
        self.align_scale = bool(align_scale)
        self.max_points = None if max_points is None else int(max_points)
        self.min_num_points = int(min_num_points)
        self.seed = int(seed)

    # ---- lifecycle hooks --------------------------------------------------- #
    def check(self) -> None:
        """Validate cloud shapes, optional normals, modes, reduce_ops, threshold.

        Raises:
            AssertionError: on a non-``(*, 3)`` cloud, normals that don't match
                their cloud, an unknown ``align`` mode or reduce op, or a
                non-positive ``threshold``.
        """
        assert (
            self.gt.ndim == 2 and self.gt.shape[1] == 3
        ), f"gt must be (M, 3); got {self.gt.shape}"
        assert (
            self.pred.ndim == 2 and self.pred.shape[1] == 3
        ), f"pred must be (N, 3); got {self.pred.shape}"
        assert self.gt_normals_in is None or self.gt_normals_in.shape == self.gt.shape, (
            f"gt_normals must match gt shape {self.gt.shape}; "
            f"got {None if self.gt_normals_in is None else self.gt_normals_in.shape}"
        )
        assert (
            self.pred_normals_in is None
            or self.pred_normals_in.shape == self.pred.shape
        ), (
            f"pred_normals must match pred shape {self.pred.shape}; "
            f"got {None if self.pred_normals_in is None else self.pred_normals_in.shape}"
        )
        assert (
            self.align in self._VALID_ALIGN
        ), f"align must be one of {sorted(self._VALID_ALIGN)}; got {self.align!r}"
        assert (
            self.threshold is None or self.threshold > 0
        ), f"threshold must be positive; got {self.threshold}"
        unknown = set(self.reduce_ops) - self._VALID_STATS
        assert (
            not unknown
        ), f"unknown reduce_ops {sorted(unknown)}; valid: {sorted(self._VALID_STATS)}"

    def preprocess(self) -> None:
        """Filter, optionally register, estimate normals, and find NN distances.

        Drops non-finite rows (keeping any supplied normals row-aligned),
        subsamples to ``max_points``, optionally ICP-registers the prediction
        onto GT (caching ``self.icp_scale``), fills in missing normals by
        local-PCA, then caches the bidirectional nearest-neighbor distances and
        matched-normal cosines that all metrics reduce. This is the expensive
        step shared by every metric, so it runs once.
        """
        gt, gt_finite = self._as_points(self.gt)
        pred, pred_finite = self._as_points(self.pred)
        gt_n = self._as_normals(self.gt_normals_in, gt_finite)
        pred_n = self._as_normals(self.pred_normals_in, pred_finite)

        rng = np.random.default_rng(self.seed)
        pred, pred_n = self._subsample_pair(pred, pred_n, self.max_points, rng)
        gt, gt_n = self._subsample_pair(gt, gt_n, self.max_points, rng)

        assert (
            pred.shape[0] >= self.min_num_points and gt.shape[0] >= self.min_num_points
        ), (
            f"need >= {self.min_num_points} finite points in each cloud; "
            f"got pred={pred.shape[0]}, gt={gt.shape[0]}"
        )

        self.icp_scale = 1.0
        if self.align == "icp":
            scale, R, t = self._register_icp(
                pred, gt, with_scale=self.align_scale, seed=self.seed
            )
            pred = (scale * (R @ pred.T)).T + t
            if pred_n is not None:
                pred_n = self._unit(pred_n @ R.T)  # rotate normals (scale/t irrelevant)
            self.icp_scale = float(scale)

        if pred_n is None:
            pred_n = self._estimate_normals(pred, self.normal_k)
        if gt_n is None:
            gt_n = self._estimate_normals(gt, self.normal_k)

        pred_tree = cKDTree(pred)
        gt_tree = cKDTree(gt)
        # Accuracy: each predicted point -> nearest GT point.
        acc_dist, acc_idx = gt_tree.query(pred, k=1, workers=-1)
        # Completeness: each GT point -> nearest predicted point.
        comp_dist, comp_idx = pred_tree.query(gt, k=1, workers=-1)

        self.gt_points, self.pred_points = gt, pred
        self.num_gt = int(gt.shape[0])
        self.num_pred = int(pred.shape[0])
        self.acc_dist = acc_dist
        self.comp_dist = comp_dist
        # |cos angle| between matched normals, per direction.
        self.acc_normal_cos = np.abs(np.sum(pred_n * gt_n[acc_idx], axis=1))
        self.comp_normal_cos = np.abs(np.sum(gt_n * pred_n[comp_idx], axis=1))

    # ---- metrics ----------------------------------------------------------- #
    @metric
    def accuracy(self) -> dict:
        """Nearest-neighbor distance prediction->GT, reduced to ``reduce_ops``."""
        return self._reduce(self.acc_dist)

    @metric
    def completeness(self) -> dict:
        """Nearest-neighbor distance GT->prediction, reduced to ``reduce_ops``."""
        return self._reduce(self.comp_dist)

    @metric
    def chamfer(self) -> dict:
        """Per-statistic mean of accuracy and completeness (``mean`` is Chamfer)."""
        acc = self._reduce(self.acc_dist)
        comp = self._reduce(self.comp_dist)
        return {op: 0.5 * (acc[op] + comp[op]) for op in self.reduce_ops}

    @metric
    def normal_consistency(self) -> dict:
        """``|cos|`` of matched normals over both directions (``mean`` in ``[0, 1]``)."""
        acc = self._reduce(self.acc_normal_cos)
        comp = self._reduce(self.comp_normal_cos)
        return {op: 0.5 * (acc[op] + comp[op]) for op in self.reduce_ops}

    @metric
    def fscore(self) -> dict:
        """Precision / recall / F-score at ``threshold`` (``NaN`` when unset)."""
        precision = self._fraction_within(self.acc_dist)
        recall = self._fraction_within(self.comp_dist)
        if np.isnan(precision) or np.isnan(recall):
            f = float("nan")
        else:
            denom = precision + recall
            f = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
        return {"precision": precision, "recall": recall, "fscore": f}

    def _fraction_within(self, dist: np.ndarray) -> float:
        """Fraction of ``dist`` below ``threshold`` (``NaN`` when no threshold)."""
        if self.threshold is None:
            return float("nan")
        return float(np.mean(dist < self.threshold))

    def _reduce(self, values: np.ndarray) -> dict[str, float]:
        """Reduce a per-point array to the ``reduce_ops`` statistics."""
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
        """Write a top-down scatter of GT vs. prediction (colored by accuracy).

        Writes ``{vis_path}_cloud.png``. No-op when ``vis_path`` is ``None``.
        """
        if vis_path is None:
            return
        self._plot_clouds(
            self.gt_points,
            self.pred_points,
            self.acc_dist,
            f"{vis_path}_cloud.png",
            threshold=self.threshold,
        )

    # ---- input plumbing (internal) ----------------------------------------- #
    @staticmethod
    def _as_points(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(finite_points, finite_row_mask)`` dropping non-finite rows."""
        finite = np.isfinite(arr).all(axis=1)
        return arr[finite], finite

    @classmethod
    def _as_normals(cls, normals, finite: np.ndarray):
        """Drop the same rows as the points and unit-normalize supplied normals."""
        if normals is None:
            return None
        return cls._unit(normals[finite])

    @staticmethod
    def _unit(vecs: np.ndarray) -> np.ndarray:
        """Row-normalize vectors to unit length (zero-length rows stay zero)."""
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norm > 0, norm, 1.0)

    # ---- normals (internal) ------------------------------------------------ #
    @classmethod
    def _estimate_normals(cls, points: np.ndarray, k: int) -> np.ndarray:
        """Estimate unit normals per point by local-PCA over ``k`` neighbors.

        The normal is the eigenvector of the local covariance with the smallest
        eigenvalue (direction of least spread). Orientation (sign) is arbitrary,
        which is fine because normal consistency uses the absolute cosine.
        """
        n = points.shape[0]
        k = int(min(max(k, 2), n))
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k, workers=-1)
        idx = np.atleast_2d(idx)

        neigh = points[idx]  # (N, k, 3)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k  # (N, 3, 3)
        # eigh: eigenvalues ascending -> column 0 is the least-spread direction.
        _, eigvecs = np.linalg.eigh(cov)
        return cls._unit(eigvecs[:, :, 0])

    # ---- registration (internal) ------------------------------------------- #
    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
        """Least-squares similarity ``(s, R, t)`` mapping ``src`` onto ``dst``."""
        mu_s = src.mean(axis=0)
        mu_d = dst.mean(axis=0)
        src_c = src - mu_s
        dst_c = dst - mu_d
        cov = (dst_c.T @ src_c) / src.shape[0]
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1.0  # reflection fix -> proper rotation
        R = U @ S @ Vt
        if with_scale:
            var_s = (src_c**2).sum() / src.shape[0]
            scale = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
        else:
            scale = 1.0
        t = mu_d - scale * R @ mu_s
        return scale, R, t

    @classmethod
    def _register_icp(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        with_scale: bool = True,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        sample: int = 50_000,
        seed: int = 0,
    ):
        """Point-to-point ICP aligning ``src`` onto ``dst``; returns ``(s, R, t)``.

        Each iteration matches every (subsampled) ``src`` point to its nearest
        ``dst`` point and refits a similarity via :meth:`_umeyama`. A coarse
        moment initialization (centroid + RMS spread) gives the narrow-basin
        point-to-point ICP a chance to lock on under an unknown global scale.
        """
        rng = np.random.default_rng(seed)
        src_fit = cls._subsample(src, sample, rng)
        dst_fit = cls._subsample(dst, sample, rng)
        tree = cKDTree(dst_fit)

        R = np.eye(3)
        if with_scale:
            spread_s = np.sqrt(((src_fit - src_fit.mean(0)) ** 2).sum(1).mean())
            spread_d = np.sqrt(((dst_fit - dst_fit.mean(0)) ** 2).sum(1).mean())
            scale = float(spread_d / spread_s) if spread_s > 0 else 1.0
        else:
            scale = 1.0
        t = dst_fit.mean(0) - scale * (R @ src_fit.mean(0))
        prev = np.inf
        for _ in range(max_iterations):
            moved = (scale * (R @ src_fit.T)).T + t
            dist, idx = tree.query(moved, k=1, workers=-1)
            scale, R, t = cls._umeyama(src_fit, dst_fit[idx], with_scale)
            mean_dist = float(dist.mean())
            if prev - mean_dist <= tolerance * max(prev, 1e-12):
                break
            prev = mean_dist
        return scale, R, t

    @staticmethod
    def _subsample(points: np.ndarray, cap: int, rng) -> np.ndarray:
        """Randomly subsample a cloud to at most ``cap`` rows (``0`` = no cap)."""
        if not cap or points.shape[0] <= cap:
            return points
        keep = rng.choice(points.shape[0], size=cap, replace=False)
        return points[keep]

    @staticmethod
    def _subsample_pair(points, normals, cap, rng):
        """Subsample a cloud to ``cap`` rows, keeping optional normals aligned."""
        if not cap or points.shape[0] <= cap:
            return points, normals
        keep = rng.choice(points.shape[0], size=cap, replace=False)
        return points[keep], (None if normals is None else normals[keep])

    # ---- plotting (internal) ----------------------------------------------- #
    @staticmethod
    def _plot_clouds(gt, pred, acc_dist, out_path: str, threshold=None) -> None:
        """Top-down (xy) scatter: GT in gray, prediction colored by accuracy."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(gt[:, 0], gt[:, 1], s=2, c="lightgray", alpha=0.5, label="GT")
        # Cap the color scale at the threshold, else the 95th percentile, so a
        # few far outliers don't wash out the map.
        vmax = threshold if threshold else float(np.percentile(acc_dist, 95))
        vmax = max(vmax, 1e-12)
        sc = ax.scatter(
            pred[:, 0], pred[:, 1], s=2, c=acc_dist, cmap="magma", vmin=0.0, vmax=vmax
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="accuracy distance")
        ax.set_aspect("equal")
        ax.set_title("Point cloud (xy) -- prediction colored by accuracy")
        ax.legend(loc="upper right", markerscale=4, fontsize="small")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
