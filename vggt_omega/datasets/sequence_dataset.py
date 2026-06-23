"""Adapter exposing :class:`BaseSequence` vendors through the training/inference
:class:`~vggt_omega.datasets.base_dataset.BaseDataset` contract.

The new :mod:`vggt_omega.datasets.vendors` vendors are :class:`BaseSequence`
objects (lazy getters + :meth:`parse`), one instance per *sequence*. The training
loader (:class:`ComposedDataset`) and inference, however, drive a
:class:`BaseDataset` that returns the rich per-batch dict (``images`` / ``depths``
/ ``extrinsics`` / ``intrinsics`` / ``cam_points`` / ``world_points`` /
``point_masks`` / ``ids`` ...) via :meth:`get_data`, and enumerate sequences
through ``sequence_list`` / :meth:`sequence_num_frames` / :meth:`native_image_size`.

:class:`SequenceDataset` bridges the two: it discovers the vendor's sequences,
constructs a :class:`BaseSequence` per sequence on demand (cached), and implements
:meth:`get_data` by composing the sequence getters with
:meth:`BaseDataset.process_one_image` (the one numpy<-pose boundary: a frame's
camera-to-world pose becomes the w2c OpenCV ``[R|t]`` extrinsic). Frame selection
for a batch uses the **SE(3) arc-length sampler** over a *random* ``[start, end]``
window (see :func:`~vggt_omega.datasets.samplers.se3_sampler.sample_se3_trajectory`),
so batches skip low-motion stretches; the randomness is the window, exactly as the
sampler intends.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Type

import numpy as np

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.dataset_util import depth_to_world_coords_points
from vggt_omega.datasets.samplers.se3_sampler import sample_se3_trajectory


def _resolve_sequence_cls(ref) -> Type[BaseSequence]:
    """Resolve ``sequence_cls`` to a class. Accepts a class object (hydra class
    reference) or a dotted import path string (e.g.
    ``"vggt_omega.datasets.vendors.tum.TumSequence"``)."""
    if isinstance(ref, str):
        import importlib

        module_path, _, cls_name = ref.rpartition(".")
        ref = getattr(importlib.import_module(module_path), cls_name)
    return ref


class SequenceDataset(BaseDataset):
    """Drive a :class:`BaseSequence` vendor through the :class:`BaseDataset` API.

    Fully generic and config-driven: the concrete :class:`BaseSequence` backend is
    supplied via ``sequence_cls`` (a hydra class reference in YAML), so there is no
    per-vendor dataset class. Sequence discovery and per-sequence construction both
    go through that class (``sequence_cls.discover`` /
    ``sequence_cls(data_root, seq_id, **sequence_kwargs)``); frame batches are
    sampled with the SE(3) arc-length sampler over a random ``[start, end]`` window.
    """

    def __init__(
        self,
        common_conf,
        sequence_cls: Type[BaseSequence],
        data_root: str,
        sequences: Optional[List[str]] = None,
        *,
        sequence_kwargs: Optional[dict] = None,
        len_train: int = 100000,
        len_test: int = 10000,
        split: str = "train",
        min_num_images: int = 2,
        seq_name_prefix: Optional[str] = None,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.get_data / ComposedDataset rely on.
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        self.sequence_cls = _resolve_sequence_cls(sequence_cls)
        if not (isinstance(self.sequence_cls, type) and issubclass(self.sequence_cls, BaseSequence)):
            raise TypeError(f"sequence_cls must be a BaseSequence subclass, got {sequence_cls!r}")

        self.data_root = data_root
        self.sequence_kwargs = dict(sequence_kwargs or {})
        self.min_num_images = int(min_num_images)
        self.len_train = len_train if split == "train" else len_test
        if seq_name_prefix is None:
            seq_name_prefix = self.sequence_cls.__name__.replace("Sequence", "").lower() + "_"
        self.seq_name_prefix = seq_name_prefix

        # Discovery lives on the BaseSequence subclass (vendor-specific layout).
        self.sequence_list = list(self.sequence_cls.discover(data_root, sequences))
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(f"No usable sequences under {data_root} (sequences={sequences})")

        self._seq_cache: Dict[str, BaseSequence] = {}

    # ------------------------------------------------------------------ #
    # sequence access
    # ------------------------------------------------------------------ #
    def _sequence(self, name: str) -> BaseSequence:
        seq = self._seq_cache.get(name)
        if seq is None:
            seq = self.sequence_cls(self.data_root, name, **self.sequence_kwargs)
            self._seq_cache[name] = seq
        return seq

    def _sensor(self, seq: BaseSequence):
        return seq.get_sensors()[0]

    # ------------------------------------------------------------------ #
    # enumeration hooks used by ComposedDataset / inference
    # ------------------------------------------------------------------ #
    def sequence_num_frames(self, local_idx: int) -> int:
        seq = self._sequence(self.sequence_list[local_idx])
        return seq.get_length(self._sensor(seq))

    def native_image_size(self, local_idx: int = 0):
        seq = self._sequence(self.sequence_list[local_idx])
        return seq.read_image_size(seq._frame_image_path(self._sensor(seq), 0))

    # ------------------------------------------------------------------ #
    # frame sampling
    # ------------------------------------------------------------------ #
    def _se3_sample_ids(self, seq: BaseSequence, sensor, num: int) -> np.ndarray:
        """SE(3) arc-length sample ``num`` frame ids over a RANDOM ``[start, end]``
        window of the sequence (the sampler's intended source of randomness)."""
        n = seq.get_length(sensor)
        if num >= n:
            return np.arange(n)
        # random window of at least `num` frames; training picks a random span,
        # eval (inside_random False) uses the full sequence.
        if self.training and self.inside_random:
            span = random.randint(num, n)
            start = random.randint(0, n - span)
            end = start + span - 1
        else:
            start, end = 0, n - 1
        # The sampler reads poses on demand via seq.get_pose(sensor, k) for ONLY
        # the [start, end] window (was: eager load of ALL n poses per sample, one
        # disk read per frame of the whole sequence on per-frame-file vendors).
        indices, _ = sample_se3_trajectory(seq, sensor, num=num, start=start, end=end)
        return np.asarray(indices, dtype=int)

    # ------------------------------------------------------------------ #
    # the batch dict
    # ------------------------------------------------------------------ #
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and seq_name is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            if seq_index is None or not 0 <= seq_index < self.sequence_list_len:
                raise ValueError(
                    f"seq_index={seq_index} out of range [0, {self.sequence_list_len}); "
                    "set inside_random=True so the sampler index is remapped."
                )
            seq_name = self.sequence_list[seq_index]

        seq = self._sequence(seq_name)
        sensor = self._sensor(seq)
        n = seq.get_length(sensor)

        if ids is None:
            num = int(img_per_seq) if img_per_seq else min(2, n)
            ids = self._se3_sample_ids(seq, sensor, num)
        ids = np.asarray(ids, dtype=int)

        target_image_shape = self.get_target_shape(aspect_ratio)
        mods = seq.get_modalities(sensor)
        has_depth = Modality.DEPTH in mods
        K_native = seq.get_intrinsic(sensor) if Modality.INTRINSIC in mods else None

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []
        timestamps = [] if Modality.TIMESTAMP in mods else None

        for i in ids:
            i = int(i)
            image = seq.get_rgb(sensor, i)
            original_size = np.array(image.shape[:2])

            if has_depth:
                depth_map = seq.get_depth(sensor, i)
            else:
                depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # camera-to-world pose -> w2c OpenCV [R|t] extrinsic (the boundary).
            if Modality.POSE in mods:
                pose_w2c = seq.get_pose(sensor, i).inverse().transform_matrix[:3].astype(np.float32)
            else:
                pose_w2c = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)

            K = K_native.copy() if K_native is not None else self._placeholder_K(image.shape[:2])

            (image, depth_map, extri, intri, world_p, cam_p, pmask, _) = self.process_one_image(
                image, depth_map, pose_w2c.copy(), K.copy(), original_size,
                target_image_shape, filepath=f"{seq_name}:{i}",
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            original_sizes.append(original_size)
            if timestamps is not None:
                timestamps.append(seq.get_timestamp(sensor, i))

        batch = {
            "seq_name": self.seq_name_prefix + seq_name,
            "ids": np.asarray(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            # Advertise GT modalities as inference-facing key strings (plural
            # sample-dict keys) the eval path expects -- NOT the
            # BaseSequence.Modality names.
            "modalities": self._advertised_modalities(mods),
        }
        if timestamps is not None:
            batch["timestamps"] = np.asarray(timestamps, dtype=np.float64)
        return batch

    @staticmethod
    def _advertised_modalities(seq_mods) -> set:
        """Translate the backend's :class:`base_sequence.Modality` set into the
        inference-facing GT key strings (plural sample-dict keys such as
        ``"depths"`` / ``"extrinsics"``) that eval / _carry_extra_modalities
        consume. ``POSE`` is the per-frame trajectory, advertised as ``extrinsics``
        (what the eval scores)."""
        mapping = {
            Modality.RGB: "images",
            Modality.DEPTH: "depths",
            Modality.POSE: "extrinsics",
            Modality.EXTRINSIC: "extrinsics",
            Modality.INTRINSIC: "intrinsics",
            Modality.TIMESTAMP: "timestamps",
        }
        return {mapping[m] for m in seq_mods if m in mapping}

    @staticmethod
    def _placeholder_K(hw) -> np.ndarray:
        h, w = hw
        f = float(max(h, w))
        return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
