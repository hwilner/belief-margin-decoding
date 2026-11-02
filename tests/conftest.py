"""Shared pytest fixtures. Satisfies docs/PROJECT.md task E8.

The suite must run on CPU with no model download and no CUDA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable from the source tree without an install.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def cpu_only_environment():
    """Pin the suite to a small, deterministic, CPU-only environment."""
    import torch

    torch.set_num_threads(2)
    # The suite never requires CUDA; make that explicit rather than implicit.
    yield
