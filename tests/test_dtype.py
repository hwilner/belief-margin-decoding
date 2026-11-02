"""Tests for the dtype plumbing.

The recorded Phase-0 run used ``float32``. A second model at 1.5B parameters
needs ``bfloat16`` to fit in 8 GB of Apple Silicon memory (3.1 GB instead of
6.2 GB), so the dtype has to be selectable -- but the *default* must not
change, or the published run would silently stop being reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bmd.matrix import MatrixConfig, _cache_key
from bmd.model import resolve_dtype


def test_default_dtype_is_float32():
    """The default must stay float32: the recorded run used it."""
    assert resolve_dtype(None) is torch.float32


def test_dtype_names_resolve():
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype(torch.float16) is torch.float16


def test_unknown_dtype_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("float8")


def test_float32_cache_key_is_unchanged():
    """The recorded cache key must stay byte-identical.

    ``results/base_cache.npz`` and ``results/phase0_summary.json`` were
    recorded with this key. Changing the key format for the default dtype
    would orphan the committed cache and make the published run look like it
    had never been run.
    """
    config = MatrixConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct", n_items=200, seed=0)
    assert _cache_key(config) == "Qwen/Qwen2.5-0.5B-Instruct|n200|s0|L4-8-12-16-20-24"


def test_non_default_dtype_changes_the_cache_key():
    """A bfloat16 run is a different measurement and must not reuse the cache."""
    base = MatrixConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct", n_items=200, seed=0)
    bf16 = MatrixConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", n_items=200, seed=0, dtype="bfloat16"
    )
    assert _cache_key(bf16) != _cache_key(base)
    assert "bfloat16" in _cache_key(bf16)


def test_matrix_config_serialises_the_dtype():
    config = MatrixConfig(dtype="bfloat16")
    assert config.to_json()["dtype"] == "bfloat16"


def test_control_runner_rejects_a_dtype_mismatch(tmp_path, capsys):
    """A control run in a different dtype than the base pass is not a control.

    The cached activations and the re-run forward pass must come from the same
    numerical pipeline, otherwise the comparison is between two different
    experiments rather than between two directions.
    """
    from run_controls import main as controls_main

    # A cache in the shape the control runner expects, recorded in bfloat16.
    n_items, n_layers, hidden = 4, 2, 8
    cache = tmp_path / "base_cache.npz"
    np.savez_compressed(
        cache,
        key=np.array("test"),
        base_logprobs=np.zeros((n_items * 3, 3)),
        states=np.zeros((n_items, 3, n_layers, hidden), dtype=np.float32),
        final_states=np.zeros((n_items, 3, hidden), dtype=np.float32),
        exact=np.array(True),
        dtype=np.array("bfloat16"),
    )

    code = controls_main(
        [
            "--n-items", str(n_items),
            "--layers", "4,16",
            "--layer", "4",
            "--cache", str(cache),
            "--out-dir", str(tmp_path / "out"),
            "--dtype", "float32",
        ]
    )
    assert code == 2
    assert "recorded in bfloat16" in capsys.readouterr().err
