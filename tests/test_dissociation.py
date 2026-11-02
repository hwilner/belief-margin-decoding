"""Tests for the read-vs-bite dissociation analysis tool.

The tool is the place where the Phase-0 headline claim is turned from prose
into a number, so it gets the same treatment as the rest of the statistics
code: the rank statistics are checked against SciPy where SciPy exists, and
the numbers it reports from the stored artefacts are pinned so that a silent
change in the analysis cannot pass unnoticed.
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
    """Import ``tools/read_bite_dissociation.py`` as a module.

    The tools directory is a set of standalone scripts rather than a package,
    so it is loaded by path.
    """
    path = ROOT / "tools" / "read_bite_dissociation.py"
    spec = importlib.util.spec_from_file_location("read_bite_dissociation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


# ---------------------------------------------------------------------------
# Rank statistics
# ---------------------------------------------------------------------------


def test_average_ranks_matches_scipy_rankdata():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    for _ in range(10):
        values = rng.integers(0, 4, size=rng.integers(3, 40)).astype(float)
        assert np.allclose(
            tool._average_ranks(values),
            scipy_stats.rankdata(values, method="average"),
        )


def test_spearman_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(11)
    for _ in range(10):
        n = int(rng.integers(4, 40))
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        if n % 3 == 0:  # force ties in both vectors
            x = np.round(x)
            y = np.round(y)
        assert tool.spearman_rho(x, y) == pytest.approx(
            float(scipy_stats.spearmanr(x, y).statistic), abs=1e-12
        )


def test_kendall_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(13)
    for _ in range(10):
        n = int(rng.integers(4, 30))
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        if n % 3 == 0:
            x = np.round(x)
            y = np.round(y)
        assert tool.kendall_tau_b(x, y) == pytest.approx(
            float(scipy_stats.kendalltau(x, y).statistic), abs=1e-12
        )


def test_exact_permutation_p_counts_all_orderings():
    """A perfectly monotone pair has exactly two extreme orderings."""
    x = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    p, rho, n_perm = tool.exact_permutation_p(x, y, "spearman")
    assert n_perm == 120  # 5!
    assert rho == pytest.approx(1.0)
    assert p == pytest.approx(2 / 120)  # identity and full reversal


def test_exact_permutation_p_is_not_significant_for_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5)
    y = rng.normal(size=5)
    p, _, _ = tool.exact_permutation_p(x, y, "spearman")
    assert p > 0.05


# ---------------------------------------------------------------------------
# The correctness-vector regression
# ---------------------------------------------------------------------------


def test_correctness_vector_converts_option_indices_to_indicators():
    correct = np.array([0, 1, 2, 1])
    preds = np.array([0, 1, 0, 2])
    out = tool.correctness_vector(preds, correct)
    assert out.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_correctness_vector_rejects_length_mismatch():
    with pytest.raises(ValueError):
        tool.correctness_vector(np.zeros(3, dtype=int), np.zeros(4, dtype=int))


def test_pairing_raw_indices_would_test_the_wrong_quantity():
    """Regression: the paired test must consume correctness, not option indices.

    The first version of this tool passed the stored option indices straight
    into ``paired_test``. Those indices are not comparable across arms --
    option 1 means different things in different arms -- so the test was
    measuring an arbitrary quantity and reported p = 0.415 for the L4
    real-vs-random comparison instead of the correct p = 0.189.
    """
    rng = np.random.default_rng(5)
    correct = rng.integers(0, 3, size=200)
    a = rng.integers(0, 3, size=200)
    b = rng.integers(0, 3, size=200)
    raw_stat, raw_p = tool.paired_test(
        tool.correctness_vector(a, correct), tool.correctness_vector(b, correct)
    )
    # The same call on the raw indices is a different (wrong) test.
    from bmd.evaluate import paired_test

    _, raw_index_p = paired_test(a.astype(float), b.astype(float))
    assert raw_p != raw_index_p
    assert raw_stat != 0.0


# ---------------------------------------------------------------------------
# The recorded Phase-0 artefacts
# ---------------------------------------------------------------------------


def _find_l4_random_draw() -> Path:
    """The layer-4 random-direction control arm, found by content.

    The file name has changed once already (the layer is now part of it, so
    that arms at different layers cannot overwrite each other), so the arm is
    located by what it *is* -- a random control at layer 4 with the recorded
    seed -- rather than by a hard-coded path.
    """
    controls = RESULTS / "controls"
    candidates = []
    for path in sorted(controls.glob("act_random*.json")):
        data = json.loads(path.read_text())
        arm = data.get("arm", {})
        if arm.get("layer") != 4:
            continue
        if data.get("control_kind") != "random" and not path.name.startswith("act_random"):
            continue
        candidates.append((data.get("direction_seed"), path, data))
    if not candidates:
        # The arm is a committed result artefact. When it is not present there
        # is nothing to pin, and the comparison test skips rather than fails.
        pytest.skip("no random-direction control arm at layer 4 is committed")
    # Prefer the draw with the recorded seed 977; fall back to the only one.
    for seed, path, _ in candidates:
        if seed == 977:
            return path
    return candidates[0][1]


@pytest.mark.skipif(
    not (RESULTS / "controls" / "act_real.json").exists(),
    reason="recorded Phase-0 control artefacts are not present",
)
def test_recorded_l4_real_vs_random_comparison_is_reproduced():
    """The tool must reproduce the comparison published in results/RESULTS.md."""
    from bmd.data import generate_items

    items = generate_items(200, seed=0)
    correct = np.array([it.correct_index for it in items])

    real = json.loads((RESULTS / "controls" / "act_real.json").read_text())
    rand = json.loads(_find_l4_random_draw().read_text())

    rc = tool.correctness_vector(real["predictions"], correct)
    ac = tool.correctness_vector(rand["predictions"], correct)

    from bmd.evaluate import paired_test

    stat, p = paired_test(rc, ac)
    # Published values: statistic 708.0, p = 0.18916127259520799.
    assert stat == pytest.approx(708.0)
    assert p == pytest.approx(0.18916127259520799, rel=1e-9)
    assert real["accuracy"] == pytest.approx(0.48)
    assert rand["accuracy"] == pytest.approx(0.43)


@pytest.mark.skipif(
    not (RESULTS / "heldout" / "heldout_analysis.json").exists(),
    reason="recorded held-out analysis is not present",
)
def test_dissociation_tool_reproduces_the_recorded_curves():
    """Read curve, bite curve, and their rank correlation, re-derived from disk."""
    heldout = RESULTS / "heldout" / "heldout_analysis.json"
    read = tool.load_read_curve(heldout)
    bite = tool.load_act_arms(RESULTS / "ablation", sorted(read))
    base = json.loads((RESULTS / "ablation" / "none.json").read_text())["accuracy"]

    from bmd.data import generate_items

    items = generate_items(200, seed=0)
    correct = np.array([it.correct_index for it in items])

    result = tool.analyse(
        read, bite, {}, base, RESULTS / "controls", RESULTS / "ablation", correct
    )

    # Recorded Phase-0 values.
    assert result["best_read_layer"] == 16
    assert result["best_read_acc"] == pytest.approx(0.720)
    assert result["best_bite_layer"] == 4
    assert result["best_bite_acc"] == pytest.approx(0.480)
    assert result["spearman"]["rho"] == pytest.approx(-0.600)
    assert result["spearman"]["n_permutations"] == 120
    # The rank correlation is negative but, with five layers, not significant.
    # The tool must report that honestly rather than rounding it to a claim.
    assert result["spearman"]["p_value_exact"] > 0.05
