"""Tests for metrics and statistics. Satisfies docs/PROJECT.md task E6."""

from __future__ import annotations

import numpy as np
import pytest
import torch

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


def test_candidate_id_extraction_does_not_depend_on_tuple_arity():
    """Regression: `score_item` must not unpack `_split_ids` output by position.

    A lint autofix once turned ``for _, cand_ids, _ in encoded`` into
    ``for cand_ids, _ in encoded``, which raised ``ValueError`` at runtime in a
    code path that no unit test covered (it needs a model). The candidate ids
    are now extracted by index, so the arity of the helper's return value is
    not part of the caller's contract.
    """
    from bmd.model import _split_ids

    class _FakeTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            class _Out:
                def __init__(self, ids):
                    self.input_ids = ids

            n = max(1, len(text.split()))
            return _Out(torch.arange(n).unsqueeze(0))

    tok = _FakeTokenizer()
    encoded = [_split_ids(tok, "hello world", " drawer.")]
    assert len(encoded[0]) == 3
    cand_ids = [e[1] for e in encoded]
    assert len(cand_ids) == 1
    assert torch.equal(cand_ids[0], encoded[0][1])


def test_normal_approximation_agrees_with_scipy():
    """The SciPy-free fallback must agree with the exact test where SciPy exists.

    The fallback is what produces the p-values on any machine without SciPy —
    including the environment the recorded run was made in. If it were subtly
    wrong, nobody would find out, because it is only used where the reference
    implementation is absent.
    """
    scipy_stats = pytest.importorskip("scipy.stats")
    from bmd.evaluate import _wilcoxon_normal_approx

    rng = np.random.default_rng(20261102)
    for trial in range(25):
        n = 200
        base = rng.binomial(1, 0.4, size=n).astype(float)
        treated = np.where(rng.random(n) < 0.5, base, 1.0 - base)
        if trial % 3 == 0:  # heavy ties
            treated = np.round(treated, 0)
        stat, p_approx = _wilcoxon_normal_approx(treated, base)
        exact = scipy_stats.wilcoxon(treated, base, zero_method="wilcox", alternative="two-sided")
        # The approximation is a normal approximation: it should agree on the
        # decision, and be close in value for a non-degenerate sample.
        if 1e-3 < exact.pvalue < 0.5:
            assert p_approx == pytest.approx(float(exact.pvalue), rel=0.35), (
                f"trial {trial}: approx {p_approx:.4f} vs exact {exact.pvalue:.4f}"
            )
        assert (p_approx < 0.05) == (float(exact.pvalue) < 0.05), f"trial {trial}: decision differs"
