import numpy as np
import pytest

from vggt_omega.datasets.modality import (
    Modality,
    REGISTRY,
    CORE,
    DERIVED,
    validate_sample,
    carry_extra_modalities,
)


def test_registry_is_complete_and_keyed_by_value():
    for mod in Modality:
        assert mod in REGISTRY, f"{mod} missing from REGISTRY"
        assert REGISTRY[mod].key == mod.value


def test_core_and_derived_partition():
    assert CORE <= set(Modality)
    assert Modality.WORLD_POINTS in DERIVED
    assert Modality.IMAGE not in DERIVED


def test_validate_sample_accepts_matching_sample():
    loaded = {Modality.IMAGE, Modality.DEPTH}
    sample = {"images": 1, "depths": 1, "modalities": {Modality.IMAGE, Modality.DEPTH}}
    validate_sample(sample, loaded)  # no raise


def test_validate_sample_rejects_missing_key():
    loaded = {Modality.IMAGE, Modality.DEPTH}
    sample = {"images": 1, "modalities": {Modality.IMAGE, Modality.DEPTH}}
    with pytest.raises(ValueError, match="missing"):
        validate_sample(sample, loaded)


def test_validate_sample_rejects_modalities_mismatch():
    loaded = {Modality.IMAGE}
    sample = {"images": 1, "modalities": {Modality.IMAGE, Modality.DEPTH}}
    with pytest.raises(ValueError, match="modalities"):
        validate_sample(sample, loaded)


def test_validate_sample_accepts_list_of_str_modalities():
    # post-carry_extra_modalities representation: modalities is a sorted list[str]
    loaded = {Modality.IMAGE, Modality.DEPTH}
    sample = {"images": 1, "depths": 1, "modalities": ["depths", "images"]}
    validate_sample(sample, loaded)  # no raise (str-enum equality)


def test_carry_extra_modalities_tensorizes_and_preserves():
    import torch

    batch = {
        "images": [1],  # core key the caller already handled; must NOT be overwritten
        "timestamps": np.array([1.0, 2.0], dtype=np.float64),  # pre-stacked scalar modality
        "sky_masks": [np.zeros((2, 3), bool), np.ones((2, 3), bool)],  # list-of-arrays
        "texts": ["a", "b"],  # str modality -> kept as list
        "modalities": {Modality.IMAGE, Modality.TIMESTAMP, Modality.SKY_MASK, Modality.TEXT},
    }
    sample = {"images": torch.zeros(2, 3, 2, 3)}
    out = carry_extra_modalities(batch, sample)

    assert out["images"].shape == (2, 3, 2, 3)  # existing key untouched
    assert torch.is_tensor(out["timestamps"]) and out["timestamps"].dtype == torch.float64
    assert out["timestamps"].shape == (2,)
    assert torch.is_tensor(out["sky_masks"]) and out["sky_masks"].dtype == torch.bool
    assert out["sky_masks"].shape == (2, 2, 3)
    assert out["texts"] == ["a", "b"]
    assert out["modalities"] == sorted(m.value for m in batch["modalities"])  # sorted list[str]
