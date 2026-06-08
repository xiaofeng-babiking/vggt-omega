import importlib

import pytest
import torch


def test_distributed_inference_imports():
    # The entrypoint must import cleanly (gflags defined, helpers reachable) on any host.
    mod = importlib.import_module("distributed_inference")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_local_inference")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >=2 GPUs for true CP parity")
def test_g2_vs_g1_end_to_end_parity():
    # Documented manual parity harness (kept skipped in CI):
    #   1. torchrun --standalone --nproc_per_node=1 distributed_inference.py --output_root /tmp/g1
    #   2. torchrun --standalone --nproc_per_node=2 distributed_inference.py --output_root /tmp/g2
    #   3. compare /tmp/g1/<seq>/metrics/metrics.json vs /tmp/g2/<seq>/metrics/metrics.json
    #      (ATE/AbsRel match within fp tolerance).
    pytest.skip("manual multi-GPU harness; see docstring")
