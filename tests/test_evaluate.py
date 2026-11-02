"""Tests for metrics and statistics. Satisfies docs/PROJECT.md task E6."""

from __future__ import annotations

import numpy as np
import pytest

from bmd.data import generate_items
from bmd.evaluate import bootstrap_ci, error_breakdown, paired_test, tom_accuracy


def _items(n: int = 10, seed: int = 0):
    return generate_items(n, seed=seed)


def test_error_breakdown_counts_each_outcome():
    items = _items(4)
    preds = np.array([it.correct_index for it in items])
    breakdown = error_breakdown(preds, items)
    assert breakdown == {"correct": 4, "world": 0, "other": 0}


def test_error_breakdown_separates_world_and_other():
    items = _items(6)
    preds = np.array([it.world_index for it in items[:3]] + [it.other_index for it in items[3:]])
    breakdown = error_breakdown(preds, items)
    assert breakdown["world"] == 3
    assert breakdown["other"] == 3
    assert breakdown["correct"] == 0


def test_tom_accuracy_matches_hand_computation():
    items = _items(8)
    preds = np.array([it.correct_index if i < 6 else it.world_index for i, it in enumerate(items)])
    assert tom_accuracy(preds, items) == pytest.approx(6 / 8)


def test_error_breakdown_rejects_length_mismatch():
    items = _items(5)
    with pytest.raises(ValueError):
        error_breakdown(np.zeros(3, dtype=int), items)


def test_bootstrap_ci_on_constant_input():
    lo, hi = bootstrap_ci(np.ones(50), n_boot=200, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.binomial(1, 0.4, size=300).astype(float)
    lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
    assert lo < values.mean() < hi


def test_paired_test_identical_samples_is_one():
    a = np.array([0.0, 1.0, 1.0, 0.0])
    stat, p = paired_test(a, a)
    assert p == pytest.approx(1.0)


def test_paired_test_detects_a_clear_difference():
    a = np.array([1.0] * 20)
    b = np.array([0.0] * 20)
    stat, p = paired_test(a, b)
    assert p < 0.05


def test_paired_test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_test(np.zeros(3), np.zeros(4))
