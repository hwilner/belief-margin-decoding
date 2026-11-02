"""Metrics and statistics for the falsification matrix.

The error taxonomy is the point of the benchmark, so it comes first: a
prediction is scored as one of three outcomes rather than as a binary
right/wrong, because "the model answered the true location" (perspective
blind) and "the model attributed another agent's belief" are different
failures with different fixes.
"""

from __future__ import annotations

import numpy as np

from .data import ToMItem


def error_breakdown(pred_index: np.ndarray, items: list[ToMItem]) -> dict:
    """Count predictions into ``correct`` / ``world`` / ``other``.

    Parameters
    ----------
    pred_index:
        Predicted option index per item.
    items:
        The corresponding items.

    Returns
    -------
    dict
        ``{"correct": int, "world": int, "other": int}``. Any index that
        matches none of the three (which should not happen) is counted under
        ``"other"``.
    """
    pred_index = np.asarray(pred_index, dtype=int)
    if len(pred_index) != len(items):
        raise ValueError(f"{len(pred_index)} predictions for {len(items)} items")
    out = {"correct": 0, "world": 0, "other": 0}
    for p, item in zip(pred_index, items):
        if p == item.correct_index:
            out["correct"] += 1
        elif p == item.world_index:
            out["world"] += 1
        else:
            out["other"] += 1
    return out


def tom_accuracy(pred_index: np.ndarray, items: list[ToMItem]) -> float:
    """Fraction of predictions matching the protagonist's believed location."""
    if len(items) == 0:
        return float("nan")
    return error_breakdown(pred_index, items)["correct"] / len(items)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    The resampling unit is the **item**, never the token or the option.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples.

    Returns ``(statistic, p_value)``. Uses ``scipy.stats.wilcoxon`` when SciPy
    is importable and falls back to a tie-corrected normal approximation
    otherwise. The all-zero-difference case (perfectly identical samples) is
    handled explicitly and returns ``p = 1.0`` rather than raising.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must have equal shape, got {a.shape} and {b.shape}")
    d = a - b
    if np.allclose(d, 0.0):
        return (0.0, 1.0)
    try:
        from scipy.stats import wilcoxon  # type: ignore

        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return (float(res.statistic), float(res.pvalue))
    except Exception:
        pass
    # Normal approximation with tie correction.
    n = len(d)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return (0.0, 1.0)
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_correction = float(np.sum(counts**3 - counts)) if counts.size else 0.0
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0)
    if sigma <= 0:
        return (w_plus, 1.0)
    z = (w_plus - mu) / sigma
    from math import erf, sqrt

    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return (w_plus, float(min(1.0, p)))
