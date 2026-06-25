"""Evaluation metric families for VGGT-Omega.

Each metric family subclasses :class:`BaseMetric` and is driven by its sealed
``run()`` (``check`` -> ``preprocess`` -> ``@metric`` methods -> ``visualize``).
"""

# Force a headless matplotlib backend BEFORE importing camera_pose (which imports
# evo.tools.plot; at import evo calls matplotlib.use(SETTINGS.plot_backend), which
# defaults to the interactive "TkAgg" and crashes in headless / torchrun runs).
# Overriding evo's SETTINGS here makes that call select "Agg" instead.
from evo.tools.settings import SETTINGS as _EVO_SETTINGS

_EVO_SETTINGS.plot_backend = "Agg"

from vggt_omega.evaluates.base_metric import BaseMetric, metric
from vggt_omega.evaluates.camera_pose import CameraPoseMetric
from vggt_omega.evaluates.mono_depth import MonoDepthMetric
from vggt_omega.evaluates.pointcloud import PointcloudMetric

__all__ = [
    "BaseMetric",
    "metric",
    "CameraPoseMetric",
    "MonoDepthMetric",
    "PointcloudMetric",
]
