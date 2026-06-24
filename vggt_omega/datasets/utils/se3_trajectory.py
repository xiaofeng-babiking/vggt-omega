from abc import ABC, abstractmethod
from typing import Any, Union, Optional, List

import numpy as np


class BaseSE3Trajectory(ABC):
    def __init__(
        self,
        timestamps: List[float],
        poses: List[Any],
        source: Optional[Union[str, int]],
        target: Optional[Union[str, int]],
    ):
        self._tstamps = timestamps  # dimension = [N], format = [t]
        self._poses = poses  # dimension = [N, 6], format = [qx, qy, qz, qw, tx, ty, tz]
        self._src = source  # source frame
        # target frame, SE(3) define the coordinate transform from source to target
        #   e.g. source="vehicle", target="world"
        self._dst = target

    @staticmethod
    def log(se3_group: Any) -> Any:
        """Log mapping from SE(3) group to se(3) algebra."""

    @staticmethod
    def exp(se3_algebra: Any) -> Any:
        """Exponential mapping from se(3) algebra to SE(3) group."""

    @staticmethod
    def left_jacobian(se3_group: Any):
        """Compute left jacobian J for SE(3) group."""

    @abstractmethod
    def transform_matrix(self) -> Any:
        """As transform matrix, dimension = [N, 4, 4]."""

    @abstractmethod
    def rotation_matrix(self) -> Any:
        """As rotation matrix, dimension = [N, 3, 3]."""

    @abstractmethod
    def rotation_vector(self) -> Any:
        """As rotation vector, dimension = [N, 3]."""

    @abstractmethod
    def quaternion(self, scalar_last: bool = True) -> Any:
        """As quaternion, dimension = [N, 4], format = [qx, qy, qz, qw]."""

    @abstractmethod
    def euler_angles(self, convention: str = "ZYX", degress: bool = True) -> Any:
        """As Euler Angles, dimension = [N, 3], e.g. ZYX -> [yaw, pitch, roll]."""

    @abstractmethod
    def se3_algebra(self) -> Any:
        """As se(3) algebra, dimension = [N, 6]."""

    @abstractmethod
    def se3_group(self) -> Any:
        """As SE(3) group, dimension = [N, 4, 4]."""

    @abstractmethod
    def norm(self) -> Any:
        """Compute SE(3) norm, i.e. norm of se(3) algebra."""

    @abstractmethod
    def apply(self, points: Any) -> Any:
        """Apply SE(3) transform to 3D points."""

    @abstractmethod
    def align(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Align 2 trajectories with Umeyama method."""

    @abstractmethod
    def relative(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Compute relative SE(3) pose between 2 trajectories."""

    @abstractmethod
    def interpolate(self, timestamps: List[float]):
        """Interpolate by specific timestamps."""

    @abstractmethod
    def compose(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Element-wise compose SE(3) poses."""

    @abstractmethod
    def inverse(self) -> "BaseSE3Trajectory":
        """Inverse the SE(3) trajectory."""

    def fit(self) -> Any:
        """Fit SE(3) trajectory to a smooth spline. Future API."""
        raise NotImplementedError

    