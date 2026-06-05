import numpy as np

from vggt_omega.datasets.vendors.common import (
    associate,
    quat_to_rotation,
    read_file_list,
)


def test_read_file_list_parses_and_skips_comments(tmp_path):
    p = tmp_path / "index.txt"
    p.write_text(
        "# comment\n"
        "\n"
        "1.0 a b\n"
        "2.5 c\n"
    )
    out = read_file_list(str(p))
    assert out == {1.0: ["a", "b"], 2.5: ["c"]}


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
