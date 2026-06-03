"""Adapters that bridge dataset samples to external tools (e.g. Rerun viz)."""
from .rerun_adapter import log_batch

__all__ = ["log_batch"]
