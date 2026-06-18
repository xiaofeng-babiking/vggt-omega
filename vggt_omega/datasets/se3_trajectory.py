from abc import ABC, abstractmethod
from typing import List
from datasets.se3_pose import BaseSE3Pose


class BaseSE3Trajectory(ABC):
    @abstractmethod
    def __init__(self, se3_poses: List[BaseSE3Pose]):
        """Initialize SE3 trajectory with multiple SE3 poses."""

    @abstractmethod
    def fit(self):
        """Fit SE3 trajectory as SE3Spline."""

    @abstractmethod
    def align(self, other: "BaseSE3Trajectory", with_scale: bool = True):
        """Align two SE3 trajectories by Umeyama method."""

    @abstractmethod
    def transform(self, se3_tf: BaseSE3Pose, scale: float = 1.0):
        """Apply SE3/SIM3 transform on SE3 trajectory."""

    @abstractmethod
    def interpolate(self, t: float):
        """Interpolate SE3 pose at give timestamp."""

    @abstractmethod
    def sample(self, n: int):
        """Uniformly sample the SE3 trajectory."""
