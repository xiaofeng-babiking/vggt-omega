import numpy as np

from vggt_omega import inference_common as ic


def test_exports_present():
    for name in ("FLAGS", "effective_long_side", "unproject_depth_map_to_point_map",
                 "world_to_camera_to_camera_to_world", "save_uint16_image", "write_ply",
                 "load_config", "build_dataset", "resolve_frame_ids", "load_sample",
                 "gt_from_sample"):
        assert hasattr(ic, name), name


def test_effective_long_side_snaps_to_16():
    assert ic.effective_long_side(640, 0.8) == 512   # round(640*0.8/16)*16
    assert ic.effective_long_side(640, 1.0) == 640
    assert ic.effective_long_side(10, 0.1) == 16     # floor clamp to 16


def test_w2c_to_c2w_roundtrip():
    # camera at world origin looking along +z, translated: w2c [R|t], t = -R @ C.
    w2c = np.tile(np.hstack([np.eye(3), np.array([[1.0], [2.0], [3.0]])]), (2, 1, 1))
    c2w = ic.world_to_camera_to_camera_to_world(w2c)
    assert c2w.shape == (2, 4, 4)
    np.testing.assert_allclose(c2w[:, :3, 3], [[-1, -2, -3]] * 2, atol=1e-9)
