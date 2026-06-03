import numpy as np
import pytest

from vggt_omega.datasets.vendors.tum import (
    associate,
    quat_to_rotation,
    tum_pose_to_w2c,
    tum_intrinsics,
)


def test_associate_greedy_nearest():
    first = [0.0, 1.0, 2.0]
    second = [0.01, 1.5, 1.99]
    matches = associate(first, second, max_diff=0.02)
    assert (0.0, 0.01) in matches
    assert (2.0, 1.99) in matches
    assert all(abs(a - b) < 0.02 for a, b in matches)


def test_quat_to_rotation_identity():
    R = quat_to_rotation((0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


def test_quat_to_rotation_180_about_z():
    R = quat_to_rotation((0.0, 0.0, 1.0, 0.0))  # 180 deg about z
    np.testing.assert_allclose(R, np.diag([-1.0, -1.0, 1.0]), atol=1e-7)


def test_tum_pose_to_w2c_inverts_c2w():
    w2c = tum_pose_to_w2c(np.zeros(3), (0.0, 0.0, 0.0, 1.0))
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_tum_pose_to_w2c_translation():
    w2c = tum_pose_to_w2c(np.array([1.0, 2.0, 3.0]), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_tum_intrinsics_fr3_and_override_and_unknown():
    K = tum_intrinsics("rgbd_dataset_freiburg3_sitting_halfsphere")
    assert K.shape == (3, 3) and K[0, 0] > 0 and K[2, 2] == 1.0
    K2 = tum_intrinsics("anything", override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        tum_intrinsics("no_camera_here")
