import os
import subprocess
import sys

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.blendedmvs import BlendedMvsDataset

BLENDEDMVS_DIR = "/jfs/Data_4DFF/train_data/blendedmvs"
HAVE_BMVS = os.path.isdir(BLENDEDMVS_DIR)
# Small scenes for fast integration tests (47 and 104 frames respectively).
SEQ = "000000000000000000000000"
SEQ2 = "000000000000000000000001"
# A scene with only 10 frames, for the lazy min_num_images behavior.
SMALL_SEQ = "5692a4c2adafac1f14201821"


def _common_conf():
    return OmegaConf.create(
        {
            "img_size": 512,
            "patch_size": 16,
            "training": True,
            "inside_random": False,
            "allow_duplicate_img": True,
            "get_nearby": True,
            "rescale": True,
            "rescale_aug": True,
            "landscape_check": False,
            "augs": {"scales": [0.8, 1.2]},
        }
    )


def _integration_common():
    return OmegaConf.merge(
        _common_conf(),
        OmegaConf.create(
            {
                "fix_img_num": -1,
                "fix_aspect_ratio": 1.0,
                "load_track": False,
                "track_num": 1024,
                "load_depth": True,
                "debug": False,
                "repeat_batch": False,
                "img_nums": [2, 6],
                "max_img_per_gpu": 12,
                "augs": {
                    "scales": [0.8, 1.2],
                    "aspects": [1.0, 1.0],
                    "cojitter": False,
                    "cojitter_ratio": 0.3,
                    "color_jitter": None,
                    "gray_scale": False,
                    "gau_blur": False,
                },
            }
        ),
    )


def _eval_common():
    """Deterministic eval-mode common_config: no aug, no random remap, explicit ids
    honored verbatim (matches how inference.py drives the loader)."""
    return OmegaConf.merge(
        _integration_common(),
        OmegaConf.create(
            {
                "training": False,
                "inside_random": False,
                "rescale_aug": False,
                "get_nearby": False,
                "allow_duplicate_img": False,
                "augs": {"scales": None},
            }
        ),
    )


def _bmvs_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.blendedmvs.BlendedMvsDataset",
                "split": "train",
                "BLENDEDMVS_DIR": BLENDEDMVS_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


def _write_camera_safetensor(path, rot_c2w, trans_c2w, intrinsics):
    """Write a camera .safetensor with the dataset's exact (mixed) dtypes."""
    from safetensors import numpy as safetensors_numpy

    safetensors_numpy.save_file(
        {
            "R_cam2world": np.asarray(rot_c2w, dtype=np.float64),
            "t_cam2world": np.asarray(trans_c2w, dtype=np.float32),
            "intrinsics": np.asarray(intrinsics, dtype=np.float32),
        },
        str(path),
    )


# --- BlendedMVS-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity():
    w2c = BlendedMvsDataset.blendedmvs_pose_to_w2c(np.eye(3), np.zeros(3))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_pose_to_w2c_translation():
    w2c = BlendedMvsDataset.blendedmvs_pose_to_w2c(np.eye(3), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_axis_remap():
    # camera-to-world rotates +90 deg about z: cam x-axis -> world y-axis.
    rot_c2w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w2c = BlendedMvsDataset.blendedmvs_pose_to_w2c(rot_c2w, np.zeros(3))
    np.testing.assert_allclose(w2c[:3, :3], rot_c2w.T, atol=1e-6)
    # world point (1,0,0) seen from this camera lands at (0,-1,0) in cam coords.
    np.testing.assert_allclose(w2c[:3, :3] @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-6)


def test_pose_to_w2c_rejects_non_finite():
    rot = np.eye(3)
    rot[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        BlendedMvsDataset.blendedmvs_pose_to_w2c(rot, np.zeros(3))


def test_read_camera_roundtrip(tmp_path):
    rot_c2w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trans_c2w = np.array([1.0, 2.0, 3.0])
    intri = np.array([[443.965, 0.0, 255.833], [0.0, 443.965, 191.833], [0.0, 0.0, 1.0]])
    p = tmp_path / "00000000.safetensor"
    _write_camera_safetensor(p, rot_c2w, trans_c2w, intri)

    w2c, intri_read = BlendedMvsDataset.read_blendedmvs_camera(str(p))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    assert intri_read.shape == (3, 3) and intri_read.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], rot_c2w.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -rot_c2w.T @ trans_c2w, atol=1e-6)
    np.testing.assert_allclose(intri_read, intri, atol=1e-3)
    assert intri_read[2, 2] == 1.0


def test_read_camera_missing_key_raises(tmp_path):
    from safetensors import numpy as safetensors_numpy

    p = tmp_path / "00000000.safetensor"
    safetensors_numpy.save_file(
        {"R_cam2world": np.eye(3), "t_cam2world": np.zeros(3, dtype=np.float32)}, str(p)
    )
    with pytest.raises(ValueError, match="missing key"):
        BlendedMvsDataset.read_blendedmvs_camera(str(p))


def test_read_camera_bad_shape_raises(tmp_path):
    p = tmp_path / "00000000.safetensor"
    _write_camera_safetensor(p, np.eye(3), np.zeros(3), np.eye(3))
    from safetensors import numpy as safetensors_numpy

    bad = dict(safetensors_numpy.load_file(str(p)))
    bad["t_cam2world"] = np.zeros(4, dtype=np.float32)
    safetensors_numpy.save_file(bad, str(p))
    with pytest.raises(ValueError, match="shape"):
        BlendedMvsDataset.read_blendedmvs_camera(str(p))


def test_depth_decode_exr(tmp_path):
    """Synthetic EXR roundtrip: positive kept, 0 stays 0, NaN/negative -> 0.
    Also proves the OpenCV EXR codec is enabled in this pytest process."""
    import cv2

    arr = np.array([[0.0, 1.5], [-2.0, np.nan]], dtype=np.float32)
    p = tmp_path / "00000000.exr"
    assert cv2.imwrite(str(p), arr)
    depth = BlendedMvsDataset.read_blendedmvs_depth(str(p))
    assert depth.dtype == np.float32 and depth.shape == (2, 2)
    np.testing.assert_allclose(depth, [[0.0, 1.5], [0.0, 0.0]])
    assert (depth >= 0).all()  # no sky encoding: nothing maps below 0


def test_depth_decode_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BlendedMvsDataset.read_blendedmvs_depth(str(tmp_path / "nope.exr"))


def test_exr_codec_enabled_even_when_cv2_imported_first(tmp_path):
    """PROOF for the EXR gotcha: in a fresh process where cv2 is imported BEFORE
    the vendor module (i.e. before OPENCV_IO_ENABLE_OPENEXR is set, as happens
    when dataset_util is imported first), importing the vendor and decoding an
    EXR still works -- OpenCV reads the env var lazily at the first decode."""
    import cv2

    p = tmp_path / "00000000.exr"
    assert cv2.imwrite(str(p), np.full((2, 2), 3.0, dtype=np.float32))

    code = (
        "import os, sys\n"
        "assert 'OPENCV_IO_ENABLE_OPENEXR' not in os.environ\n"
        "import vggt_omega.datasets.dataset_util  # imports cv2 BEFORE the env var is set\n"
        "import cv2\n"
        "from vggt_omega.datasets.vendors.blendedmvs import BlendedMvsDataset\n"
        "d = BlendedMvsDataset.read_blendedmvs_depth(sys.argv[1])\n"
        "assert d.shape == (2, 2) and float(d[0, 0]) == 3.0\n"
        "print('EXR_DECODE_OK')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OPENCV_IO_ENABLE_OPENEXR"}
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code, str(p)],
        env=env, cwd=repo_root, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "EXR_DECODE_OK" in proc.stdout


# --- BlendedMVS integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_sample_schema_and_conventions():
    ds = BlendedMvsDataset(
        common_conf=_common_conf(),
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    # No sky encoding and no timestamps on disk -> not advertised.
    assert Modality.SKY_MASK not in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities
    # Reprojected depth is not point-cloud GT.
    assert Modality.WORLD_POINTS not in ds.available_modalities
    assert Modality.CAM_POINTS not in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "blendedmvs_" + SEQ
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    pmask = np.stack(batch["point_masks"])
    sky = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth >= 0).all()                             # 0=invalid; no sky (<0) values
    assert (depth[depth > 0]).size > 0                    # some valid depth
    assert not sky.any()                                  # no sky encoding
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is False                    # per-scene SfM units
    assert batch["is_video"] is False                     # unordered multi-view
    assert "timestamps" not in batch                      # TIMESTAMP not advertised

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_reprojection_closure():
    """World points from two frames of the same scene must be mutually
    consistent: frame A's valid world points reprojected into frame B land at
    depths matching B's depth map (validates depth-scale x pose-convention x
    intrinsics agreement end-to-end through process_one_image). Depth is in
    SfM units, so the comparison is RELATIVE."""
    ds = BlendedMvsDataset(
        common_conf=_eval_common(),
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([0, 1]), aspect_ratio=0.75)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    wA = world[0][pmask[0]]
    E, K = extr[1], intr[1]
    camB = wA @ E[:3, :3].T + E[:3, 3]
    z = camB[:, 2]
    u = camB[:, 0] / z * K[0, 0] + K[0, 2]
    v = camB[:, 1] / z * K[1, 1] + K[1, 2]
    H, W = depth[1].shape
    ok = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.round(u[ok]).astype(int)
    vi = np.round(v[ok]).astype(int)
    measured = depth[1][vi, ui]
    valid = measured > 0
    rel_err = np.abs(z[ok][valid] - measured[valid]) / measured[valid]
    assert valid.sum() >= 500
    # The correct recipe closes at ~0.00023 median rel err on this scene/pair;
    # 0.01 keeps ~40x headroom while FAILING the two plausible wrong pose
    # conventions (measured on this exact scene/frames/aspect): treating the
    # stored c2w as w2c without inverting -> 0.0645, and an OpenGL-style
    # diag(1,-1,-1) y/z axis flip -> 0.0334. A 0.05 threshold would let the
    # axis-flip pass because this scene's valid depth clusters near its median
    # (~46), so even mislanded pixels give small RELATIVE errors. NOTE: the
    # discriminative power is pair-dependent (on scene ...0001 frames 0-1 the
    # no-invert flip closes at 0.0059); keep this scene/frame pair.
    assert np.median(rel_err) < 0.01


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_getitem_tuple_index():
    ds = BlendedMvsDataset(
        common_conf=_common_conf(),
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_construction_is_lazy():
    """Construction over ALL ~502 scenes only lists the root (no per-scene
    enumeration, which takes ~18 s on this network FS); frame listings are
    cached lazily on first access."""
    import time

    t0 = time.monotonic()
    ds = BlendedMvsDataset(
        common_conf=_common_conf(),
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        len_train=10,
    )
    elapsed = time.monotonic() - t0
    assert ds.sequence_list_len >= 400
    assert not ds._frames_cache                  # nothing enumerated yet
    assert elapsed < 15.0                        # root listing only (~0.05 s)

    local_idx = ds.sequence_list.index(SEQ)
    n = ds.sequence_num_frames(local_idx)        # triggers ONE scene listing
    assert n >= 24
    assert list(ds._frames_cache) == [SEQ]


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_min_num_images_enforced_lazily():
    """Undersized sequences (frame counts are only known lazily) raise for
    sampled ids in deterministic mode but are still served for explicit ids."""
    ds = BlendedMvsDataset(
        common_conf=_eval_common(),
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        sequences=[SMALL_SEQ],
        len_train=10,
    )
    assert ds.sequence_list == [SMALL_SEQ]       # listed (filter is lazy)
    with pytest.raises(ValueError, match="min_num_images"):
        ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
    batch = ds.get_data(seq_name=SMALL_SEQ, ids=np.array([0, 1]), aspect_ratio=1.0)
    assert batch["frame_num"] == 2


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_explicit_seq_name_never_redrawn():
    """An explicitly named undersized sequence must raise under inside_random,
    not be silently swapped for a different random scene by the redraw loop
    (TUM/7-Scenes always serve the named sequence). The redraw is only for
    sampler-derived seq_index draws."""
    conf = _common_conf()
    conf.inside_random = True
    ds = BlendedMvsDataset(
        common_conf=conf,
        split="train",
        BLENDEDMVS_DIR=BLENDEDMVS_DIR,
        sequences=[SMALL_SEQ, SEQ],   # a redraw candidate exists and is big enough
        len_train=10,
    )
    with pytest.raises(ValueError, match="min_num_images"):
        ds.get_data(seq_name=SMALL_SEQ, img_per_seq=4, aspect_ratio=1.0)
    # Explicit ids on the named undersized sequence are still served verbatim.
    batch = ds.get_data(seq_name=SMALL_SEQ, ids=np.array([0, 1]), aspect_ratio=1.0)
    assert batch["seq_name"] == "blendedmvs_" + SMALL_SEQ
    # Sampler-derived draws may still redraw away from the undersized scene.
    batch = ds.get_data(seq_index=0, img_per_seq=4, aspect_ratio=1.0)
    assert batch["seq_name"] == "blendedmvs_" + SEQ


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_blendedmvs_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _bmvs_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)
    assert "timestamps" not in sample              # BlendedMVS has none


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _bmvs_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Drift guard: the same vendor.get_data + manual tensorize must match byte-for-byte.
    vendor = composed.base_dataset.datasets[0]
    batch = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(ids), aspect_ratio=0.75
    )
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)

    # Order honored: BlendedMVS has no timestamps, so prove the UNORDERED ids
    # were served verbatim via per-frame extrinsics (untouched by eval-mode
    # processing): the batch must be the sorted-ids batch under the permutation.
    sorted_batch = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(sorted(ids)), aspect_ratio=0.75
    )
    extr = np.stack(batch["extrinsics"])
    extr_sorted = np.stack(sorted_batch["extrinsics"])
    perm = [sorted(ids).index(i) for i in ids]
    # the four cameras are pairwise distinct, so the permutation check is meaningful
    assert all(
        not np.allclose(extr_sorted[a], extr_sorted[b])
        for a in range(4) for b in range(a + 1, 4)
    )
    np.testing.assert_allclose(extr, extr_sorted[perm], atol=0)
    np.testing.assert_allclose(sample["extrinsics"].numpy(), extr, atol=0)


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, SEQ2]
    composed = instantiate(
        _bmvs_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        # ground truth straight from the directory listing
        n_jpg = sum(
            1
            for f in os.listdir(os.path.join(BLENDEDMVS_DIR, name))
            if f.endswith(".jpg")
        )
        assert composed.sequence_num_frames(gi) == n_jpg


@pytest.mark.skipif(not HAVE_BMVS, reason=f"BlendedMVS data not found at {BLENDEDMVS_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _bmvs_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (384, 512)                       # BlendedMVS native (H, W)

    composed.set_img_size(512)                        # native long side
    assert composed.img_size == 512
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (384, 512)

    composed.set_img_size(256)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (192, 256)
