"""CLI entry: ``python -m vggt_omega.datasets.adapters --configure <yaml> --out <dir>``.

Thin wrapper so the package is runnable directly (avoids the double-import
warning from running the submodule as ``-m`` while the package ``__init__``
already imported it). See :mod:`vggt_omega.datasets.adapters.rerun_adapter`.
"""
import logging

from .rerun_adapter import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
