"""Tests for the belief-accuracy frontier tool.

The frontier is computed from the cached base pass, so its numbers have to be
reconcilable with the published Phase-0 held-out analysis. The first version
of this tool built its cross-validation folds differently from
``tools/analyze_heldout.py``, and the noise-0 row silently disagreed with the
published read curve (0.705 instead of 0.720 at L16). These tests pin that
agreement so it cannot drift again.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load_tool():
    path = ROOT / "tools" / "belief_accuracy_frontier.py"
    spec = importlib.util.spec_from_file_location("belief_accuracy_frontier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


# ---------------------------------------------------------------------------
# Label corruption
# ---------------------------------------------------------------------------


def test_corrupt_labels_keeps_the_class_balance():
    correct = np.array([0, 1, 2, 0, 1, 2])
    y = np.zeros(18, dtype=int)
    for i, c in enumerate(correct):
        y[i * 3 + c] = 1
    out = tool.corrupt_labels(y, correct, noise=0.9, seed=1)
    assert out.sum() == y.sum() == 6


def test_corrupt_labels_at_zero_noise_is_the_identity():
    correct = np.array([0, 1, 2, 0, 1, 2])
    y = np.zeros(18, dtype=int)
    for i, c in enumerate(correct):
        y[i * 3 + c] = 1
    out = tool.corrupt_labels(y, correct, noise=0.0, seed=1)
    assert np.array_equal(out, y)


def test_corrupt_labels_at_full_noise_is_always_wrong():
    correct = np.array([0, 1, 2, 0, 1, 2, 2, 0])
    y = np.zeros(24, dtype=int)
    for i, c in enumerate(correct):
        y[i * 3 + c] = 1
    out = tool.corrupt_labels(y, correct, noise=1.0, seed=3)
    for i, c in enumerate(correct):
        positive = int(np.argmax(out[i * 3 : i * 3 + 3]))
        assert positive != c
        assert out[i * 3 : i * 3 + 3].sum() == 1


def test_corrupt_labels_is_deterministic_for_a_seed():
    correct = np.array([0, 1, 2, 0, 1, 2, 2, 0, 1])
    y = np.zeros(27, dtype=int)
    for i, c in enumerate(correct):
        y[i * 3 + c] = 1
    a = tool.corrupt_labels(y, correct, noise=0.4, seed=11)
    b = tool.corrupt_labels(y, correct, noise=0.4, seed=11)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The fold protocol
# ---------------------------------------------------------------------------


def test_folds_match_the_heldout_tool():
    """The folds must be the ones ``analyze_heldout.py`` uses.

    ``fit_subspace`` splits its input with a deterministic *tail* split, so
    the rows that actually train the probe depend on the order of the caller's
    array. A different fold order is a different probe.
    """
    n, n_folds = 200, 5
    mine = tool._fold_indices(n, n_folds, seed=0)

    rng = np.random.default_rng(0 + 4242)
    order = rng.permutation(n)
    n_test = int(round(n * 0.2))
    _test_idx = np.sort(order[:n_test])
    _train_idx = np.sort(order[n_test:])
    theirs = [np.sort(f) for f in np.array_split(rng.permutation(n), n_folds)]

    for a, b in zip(mine, theirs, strict=True):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The frontier against the published analysis
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (RESULTS / "frontier" / "belief_accuracy_frontier.json").exists(),
    reason="the frontier has not been computed yet",
)
def test_noise_zero_row_reproduces_the_published_heldout_curve():
    """The frontier's noise-0 row must equal the published read curve.

    If these ever disagree, the two analyses are measuring different things
    and neither can be compared with the other.
    """
    frontier = json.loads(
        (RESULTS / "frontier" / "belief_accuracy_frontier.json").read_text()
    )
    heldout = json.loads((RESULTS / "heldout" / "heldout_analysis.json").read_text())

    clean = [r for r in frontier["rows"] if r["noise"] == 0.0]
    assert clean, "the frontier has no noise-0 row"

    for hrow in heldout["rows"]:
        frow = next(r for r in clean if r["layer"] == hrow["layer"])
        assert frow["achieved_read_acc"] == pytest.approx(hrow["cv_three_way_acc"])
        arm = frow["arms"]["margin-a4"]
        published = hrow["arms"]["margin-a4"]
        assert arm["cv_acc"] == pytest.approx(published["cv_acc"])
        assert arm["cv_acc_shuffled_control"] == pytest.approx(
            published["cv_acc_shuffled_control"]
        )
        assert arm["margin_vs_control_p"] == pytest.approx(
            published["margin_vs_control_wilcoxon"]["p_value"], rel=1e-9
        )


@pytest.mark.skipif(
    not (RESULTS / "frontier" / "belief_accuracy_frontier.json").exists(),
    reason="the frontier has not been computed yet",
)
def test_the_arm_never_significantly_beats_its_probe():
    """The central claim of the frontier, pinned.

    Across every layer, every label-noise level and every alpha, the arm's
    advantage over the probe that produced its margin must not survive a
    multiple-comparison correction.

    The Wilcoxon test is **two-sided**, so a small p-value alone does not mean
    "the arm beats the probe": with a *negative* difference it means the arm
    is significantly **worse**. The sign has to be checked alongside the
    p-value, and the arm is significantly worse than the probe in plenty of
    configurations -- which is the point of the mechanism.
    """
    frontier = json.loads(
        (RESULTS / "frontier" / "belief_accuracy_frontier.json").read_text()
    )
    comparisons = [
        arm
        for r in frontier["rows"]
        for arm in r["arms"].values()
    ]
    n = len(comparisons)
    bonferroni = 0.05 / n
    surviving = [
        a for a in comparisons
        if a["margin_minus_read"] > 0 and a["margin_vs_read_p"] < bonferroni
    ]
    assert surviving == [], (
        f"{len(surviving)} of {n} comparisons beat the probe after Bonferroni correction"
    )

    # The converse is expected and is not a failure: where the probe is
    # strong and alpha is large, the arm IS the probe, and where the probe is
    # weak the arm is dragged below it.
    worse = [
        a for a in comparisons
        if a["margin_minus_read"] < 0 and a["margin_vs_read_p"] < bonferroni
    ]
    assert worse, "the arm is never significantly worse than the probe either"


def test_break_even_reports_a_bracket_not_a_single_threshold():
    """The achieved read accuracy is not monotone in the noise rate."""
    name = "margin-a4"
    rows = [
        {
            "layer": 16,
            "achieved_read_acc": 0.35,
            "base_acc": 0.37,
            "arms": {name: {"margin_minus_base": -0.02, "cv_acc": 0.35}},
        },
        {
            "layer": 16,
            "achieved_read_acc": 0.50,
            "base_acc": 0.37,
            "arms": {name: {"margin_minus_base": 0.05, "cv_acc": 0.42}},
        },
        {
            "layer": 16,
            "achieved_read_acc": 0.70,
            "base_acc": 0.37,
            "arms": {name: {"margin_minus_base": 0.30, "cv_acc": 0.67}},
        },
    ]
    be = tool.break_even(rows, 16, 4.0, 0.37)
    assert be["n_levels"] == 3
    assert be["n_levels_beating_base"] == 2
    assert be["lowest_read_acc_that_beats_base"] == pytest.approx(0.50)
    assert be["highest_read_acc_that_fails_to_beat_base"] == pytest.approx(0.35)
    assert be["fitted_crossing_read_acc"] is not None
