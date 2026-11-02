"""Tests for the belief subspace. Satisfies docs/PROJECT.md tasks E2 and E3."""

from __future__ import annotations

import numpy as np
import pytest

from bmd.subspace import BeliefSubspace, fit_subspace, normalize_margins


def _separable(n: int = 100, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack([
        rng.normal(loc=[2.0] * d, scale=0.2, size=(half, d)),
        rng.normal(loc=[-2.0] * d, scale=0.2, size=(n - half, d)),
    ])
    y = np.array([1] * half + [0] * (n - half))
    return X, y


def test_lda_separates_trivially_separable_data():
    X, y = _separable()
    sub = fit_subspace(X, y, layer=0, kind="lda")
    assert sub.heldout_acc == 1.0
    assert sub.train_acc == 1.0
    assert sub.kind == "lda"


def test_logreg_separates_trivially_separable_data():
    X, y = _separable()
    sub = fit_subspace(X, y, layer=0, kind="logreg")
    assert sub.heldout_acc > 0.95


def test_save_load_round_trip(tmp_path):
    X, y = _separable()
    sub = fit_subspace(X, y, layer=7, kind="lda")
    path = tmp_path / "subspace.npz"
    sub.save(path)
    loaded = BeliefSubspace.load(path)
    assert np.allclose(loaded.w, sub.w)
    assert loaded.b == pytest.approx(sub.b)
    assert loaded.layer == sub.layer
    assert loaded.train_acc == pytest.approx(sub.train_acc)
    assert loaded.heldout_acc == pytest.approx(sub.heldout_acc)


def test_margin_shape_and_sign():
    X, y = _separable()
    sub = fit_subspace(X, y, layer=0, kind="lda")
    margins = sub.margin(X)
    assert margins.shape == (X.shape[0],)
    assert margins[y == 1].mean() > margins[y == 0].mean()


def test_margin_rejects_wrong_width():
    X, y = _separable()
    sub = fit_subspace(X, y, layer=0, kind="lda")
    with pytest.raises(ValueError):
        sub.margin(np.zeros((3, 7)))


def test_fit_rejects_single_class():
    X, _ = _separable()
    with pytest.raises(ValueError):
        fit_subspace(X, np.zeros(len(X), dtype=int), layer=0)


def test_fit_rejects_length_mismatch():
    X, y = _separable()
    with pytest.raises(ValueError):
        fit_subspace(X, y[:-1], layer=0)


def test_fit_rejects_unknown_kind():
    X, y = _separable()
    with pytest.raises(ValueError):
        fit_subspace(X, y, layer=0, kind="nope")


def test_normalize_is_scale_and_shift_invariant():
    rng = np.random.default_rng(0)
    m = rng.normal(size=200)
    a = normalize_margins(m)
    b = normalize_margins(m * 17.0 + 3.0)
    assert np.allclose(a, b)
    assert a.mean() == pytest.approx(0.0, abs=1e-9)
    assert a.std() == pytest.approx(1.0, abs=1e-9)


def test_normalize_constant_input_is_zero():
    out = normalize_margins(np.full(10, 4.2))
    assert np.allclose(out, 0.0)
