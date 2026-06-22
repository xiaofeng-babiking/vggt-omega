"""Shared conformance contract for :class:`BaseSequence` vendors.

``BaseSequenceContract`` holds the interface laws every vendor must satisfy,
written purely against the abstract API. Each vendor test subclasses it and
implements :meth:`make_sequence`; the contract then runs all laws against that
concrete sequence. The mixin name does not start with ``Test`` so pytest never
collects it on its own.
"""
import numpy as np
import pytest

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose


class BaseSequenceContract:
    """Interface laws for any BaseSequence vendor. Subclass + set make_sequence()."""

    def make_sequence(self) -> BaseSequence:
        raise NotImplementedError

    # ----- identity / discovery -------------------------------------------- #
    def test_is_base_sequence(self):
        assert isinstance(self.make_sequence(), BaseSequence)

    def test_sensors_nonempty(self):
        assert len(self.make_sequence().get_sensors()) >= 1

    def test_length_positive(self):
        s = self.make_sequence()
        for sid in s.get_sensors():
            assert s.get_length(sid) > 0

    def test_modalities_subset_of_known(self):
        s = self.make_sequence()
        known = set(Modality)
        for sid in s.get_sensors():
            assert s.get_modalities(sid) <= known

    # ----- per-frame getters honour advertised modalities ------------------ #
    def test_advertised_modalities_decode(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        if Modality.RGB in mods:
            rgb = s.get_rgb(sid, 0)
            assert rgb.ndim == 3 and rgb.shape[2] == 3
        if Modality.DEPTH in mods:
            depth = s.get_depth(sid, 0)
            assert depth.ndim == 2
            if Modality.RGB in mods:
                assert depth.shape == s.get_rgb(sid, 0).shape[:2]
        if Modality.POSE in mods:
            assert isinstance(s.get_pose(sid, 0), BaseSE3Pose)
        if Modality.INTRINSIC in mods:
            K = s.get_intrinsic(sid)
            assert K.shape == (3, 3)
        if Modality.TIMESTAMP in mods:
            assert isinstance(s.get_timestamp(sid, 0), float)

    def test_timestamps_monotonic_if_present(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.TIMESTAMP not in s.get_modalities(sid):
            return
        n = min(s.get_length(sid), 50)
        ts = [s.get_timestamp(sid, i) for i in range(n)]
        assert ts == sorted(ts)

    # ----- parse (the base template) --------------------------------------- #
    def test_parse_stacks_in_order(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        n = s.get_length(sid)
        ids = [0, min(1, n - 1), min(2, n - 1)]
        out = s.parse(sid, ids, image_size=(60, 80))
        for m, arr in out.items():
            assert arr.shape[0] == len(ids)
        if Modality.RGB in out:
            assert out[Modality.RGB].shape[1:] == (60, 80, 3)
        if Modality.DEPTH in out:
            assert out[Modality.DEPTH].shape[1:] == (60, 80)

    def test_parse_only_returns_requested(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        if Modality.RGB not in mods:
            return
        out = s.parse(sid, [0, 1], modalities={Modality.RGB})
        assert set(out.keys()) == {Modality.RGB}

    def test_extrinsic_is_per_sequence_calibration(self):
        # EXTRINSIC is the static inter-sensor transform (calibration), fetched via
        # get_extrinsic(src, dst) — NOT a per-frame parse output.
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.EXTRINSIC not in s.get_modalities(sid):
            return
        from vggt_omega.datasets.se3_pose import BaseSE3Pose as _Pose
        assert isinstance(s.get_extrinsic(sid, sid), _Pose)

    def test_parse_rejects_unavailable_modality(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        missing = (set(self_per_frame()) - s.get_modalities(sid))
        if not missing:
            return
        with pytest.raises(ValueError):
            s.parse(sid, [0], modalities={next(iter(missing))})

    def test_parse_rejects_per_sequence_modality(self):
        # INTRINSIC / EXTRINSIC are per-sequence calibration; parse must reject them
        # even when the sensor advertises them.
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        mods = s.get_modalities(sid)
        for cal in (Modality.INTRINSIC, Modality.EXTRINSIC):
            if cal in mods:
                with pytest.raises(ValueError):
                    s.parse(sid, [0], modalities={cal})

    def test_scaled_intrinsic_rescales(self):
        s = self.make_sequence()
        sid = s.get_sensors()[0]
        if Modality.INTRINSIC not in s.get_modalities(sid):
            return
        K = s.get_intrinsic(sid)
        h0, w0 = s.read_image_size(s._frame_image_path(sid, 0))
        # native (no image_size) returns K unchanged.
        np.testing.assert_allclose(s.scaled_intrinsic(sid), K, atol=1e-4)
        Kr = s.scaled_intrinsic(sid, image_size=(h0 // 2, w0 // 2))
        np.testing.assert_allclose(Kr[0, 0], K[0, 0] * (w0 // 2) / w0, rtol=1e-3)
        np.testing.assert_allclose(Kr[1, 1], K[1, 1] * (h0 // 2) / h0, rtol=1e-3)


def self_per_frame():
    return BaseSequence._PER_FRAME_MODALITIES
