"""All vendored modules must import cleanly under vggt_omega.datasets.*"""
import importlib

import pytest

VENDORED = [
    "vggt_omega.datasets.dataset_util",
    "vggt_omega.datasets.augmentation",
    "vggt_omega.datasets.worker_fn",
    "vggt_omega.datasets.track_util",
    "vggt_omega.datasets.base_dataset",
    "vggt_omega.datasets.composed_dataset",
    "vggt_omega.datasets.dynamic_dataloader",
]


@pytest.mark.parametrize("module", VENDORED)
def test_vendored_module_imports(module):
    assert importlib.import_module(module) is not None


def test_base_dataset_exposed():
    from vggt_omega.datasets.base_dataset import BaseDataset

    assert hasattr(BaseDataset, "process_one_image")
    assert hasattr(BaseDataset, "get_target_shape")
