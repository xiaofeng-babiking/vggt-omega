import sys

import numpy as np
import pytest

from vggt_omega.datasets.adapters import rerun_adapter


def test_require_rerun_raises_helpful_error_when_missing(monkeypatch):
    # Force `import rerun` to fail even if the package is installed.
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=r"vggt-omega\[viz\]"):
        rerun_adapter._require_rerun()
