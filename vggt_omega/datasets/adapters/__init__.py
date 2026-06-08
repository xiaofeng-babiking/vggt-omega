"""Adapters bridging VGGT-Omega dataset samples to external tools.

Currently ships the Rerun adapter (:mod:`rerun_adapter`). :class:`RerunDataset`
wraps any VGGT-Omega dataset into a drop-in, Rerun-logging dataset for any
train/test pipeline; ``log_sample`` / ``sample_to_rrd`` are the per-sample dict
primitives it builds on. Rerun is an *optional* dependency
(``pip install 'vggt-omega[viz]'``) imported lazily, so importing this package
never requires it.
"""
from .rerun_adapter import RerunDataset, log_sample, normalize_sample, sample_to_rrd

__all__ = ["RerunDataset", "log_sample", "sample_to_rrd", "normalize_sample"]
