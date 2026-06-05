"""Template-method base for VGGT-Omega evaluation metrics.

The contract is inverted from a plain ABC: ``run()`` is *concrete and final*,
and a subclass adds metrics A / B / C as ordinary methods tagged with
``@metric``. ``run()`` discovers every tagged method (via per-subclass
registration in :meth:`__init_subclass__`) and dispatches it, so the evaluation
driver never changes when metrics are added or removed.

Usage::

    class DepthMetric(BaseMetric):
        def __init__(self, gt, pred):
            self.gt, self.pred = gt, pred

        def check(self):                 # optional hook (default: no-op)
            assert self.gt.shape == self.pred.shape

        def preprocess(self):            # optional hook (default: no-op)
            self.err = np.abs(self.gt - self.pred)

        @metric
        def abs_rel(self):               # metric A
            return float((self.err / self.gt).mean())

        @metric(name="rmse")             # metric B (custom report name)
        def root_mean_square(self):
            return float(np.sqrt((self.err ** 2).mean()))

    DepthMetric(gt, pred).run()
    # -> {"abs_rel": 0.07, "rmse": 0.31}

A subclass that tries to override ``run()`` raises ``TypeError`` at class
definition time -- the whole point is that ``run()`` stays untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


def metric(fn: Callable | None = None, *, name: str | None = None) -> Callable:
    """Mark a method as a metric collected by :meth:`BaseMetric.run`.

    Tagged methods take no arguments beyond ``self`` and return the metric
    value (typically a ``float`` or a small ``dict`` of sub-statistics).

    Args:
        fn: the method, when used bare as ``@metric``.
        name: report key under which the value appears in ``run()``'s result;
            defaults to the method name. Use it when the method name and the
            reported metric name should differ (e.g. ``root_mean_square`` ->
            ``"rmse"``).

    Returns:
        The same function, tagged for discovery. Supports both ``@metric`` and
        ``@metric(name=...)``.
    """

    def tag(f: Callable) -> Callable:
        f._is_metric = True  # type: ignore[attr-defined]
        f._metric_name = name or f.__name__  # type: ignore[attr-defined]
        return f

    return tag if fn is None else tag(fn)


class BaseMetric(ABC):
    """Abstract base for every evaluation metric family.

    A subclass implements ``__init__`` (store ``gt`` / ``pred``) and one or more
    ``@metric``-tagged methods. The lifecycle hooks ``check`` / ``preprocess`` /
    ``visualize`` default to no-ops and are overridden only when needed. The
    driver method ``run()`` is final and MUST NOT be overridden.

    Class attribute:
        _metric_funcs: ordered ``{report_name: method_name}`` map, rebuilt for
            each subclass from its (and its bases') ``@metric`` methods.
    """

    _metric_funcs: dict[str, str] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect ``@metric`` methods into ``cls._metric_funcs`` and seal ``run``.

        Walks the MRO so inherited metrics are included; a name re-tagged in a
        subclass overrides the inherited one. Definition order is preserved
        (``dict`` insertion order), most-derived class last.
        """
        super().__init_subclass__(**kwargs)

        if "run" in vars(cls):
            raise TypeError(
                f"{cls.__name__} must not override BaseMetric.run(); add metrics "
                "with @metric instead."
            )

        registry: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            for attr_name, attr in vars(klass).items():
                if getattr(attr, "_is_metric", False):
                    registry[attr._metric_name] = attr_name
        cls._metric_funcs = registry

    @abstractmethod
    def __init__(self, gt: Any, pred: Any, *args: Any, **kwargs: Any):
        """Initialize with Groundtruth and Prediction inputs."""

    def __len__(self) -> int:
        """Number of metrics this class reports."""
        return len(self._metric_funcs)

    # ---- lifecycle hooks (override as needed; default no-op) --------------
    def check(self) -> None:
        """Check the sanitization of inputs. Override to validate; default no-op."""

    def preprocess(self) -> None:
        """Preprocess the inputs for metric computation (e.g. align, mask, cache
        intermediate arrays on ``self``). Override as needed; default no-op."""

    def visualize(self, *args: Any, **kwargs: Any) -> None:
        """Visualize the comparison between inputs with evaluation results.
        Override as needed; default no-op."""

    # ---- FINAL: the driver (do NOT override) ------------------------------
    def run(self, vis_path: str | None = None) -> dict[str, Any]:
        """Run the full evaluation and return ``{report_name: value}``.

        Flow: ``check()`` -> ``preprocess()`` -> call every ``@metric`` method in
        registration order, collecting its return value under its report name.
        This method is sealed (see :meth:`__init_subclass__`); subclasses extend
        behaviour only by adding metrics, never by touching this API.

        Args:
            vis_path: where ``visualize`` should write its output. When omitted
                (the metrics-only case), ``visualize`` is skipped entirely.
        """
        self.check()
        self.preprocess()

        self.metrics = {}
        for report_name, method_name in self._metric_funcs.items():
            self.metrics[report_name] = {}
            self.metrics[report_name].update(getattr(self, method_name)())

        self.visualize(vis_path)

        return self.metrics
