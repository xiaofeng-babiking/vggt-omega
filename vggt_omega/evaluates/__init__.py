"""Evaluation metric families for VGGT-Omega.

Each metric family subclasses :class:`BaseMetric` and is driven by its sealed
``run()`` (``check`` -> ``preprocess`` -> ``@metric`` methods -> ``visualize``).
"""

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
