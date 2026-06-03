import numpy as np
import pytest

from vggt_omega.datasets.modality import (
    Modality,
    REGISTRY,
    CORE,
    DERIVED,
    validate_sample,
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
