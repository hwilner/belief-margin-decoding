"""Tests for the intervention mechanisms. Satisfies docs/PROJECT.md tasks E4 and E5."""

from __future__ import annotations

import numpy as np
import pytest

from bmd.intervene import ActivationSteerer, LogitBias, margin_scores
from bmd.subspace import BeliefSubspace


def test_activation_steering_adds_scaled_direction():
    direction = np.ones(4)
    hidden = np.zeros(4)
    out = ActivationSteerer(direction, layer=2, alpha=0.5).apply(hidden)
    assert np.allclose(out, np.full(4, 0.5))


def test_activation_steering_scale_multiplies():
    out = ActivationSteerer(np.ones(4), layer=2, alpha=2.0, scale=3.0).apply(np.zeros(4))
    assert np.allclose(out, np.full(4, 6.0))


def test_activation_steering_zero_alpha_is_identity():
    hidden = np.array([1.0, 2.0, 3.0])
    out = ActivationSteerer(np.ones(3), layer=0, alpha=0.0).apply(hidden)
    assert np.allclose(out, hidden)


def test_activation_steering_rejects_width_mismatch():
    with pytest.raises(ValueError):
        ActivationSteerer(np.ones(4), layer=0, alpha=1.0).apply(np.zeros(7))


def test_logit_bias_with_zero_vector_is_identity():
    logits = np.array([1.0, 2.0, 3.0])
    out = LogitBias(np.zeros(3), alpha=1.0).apply(logits)
    assert np.allclose(out, logits)


def test_logit_bias_adds_exactly_alpha_times_vector():
    logits = np.array([1.0, 2.0, 3.0])
    vec = np.array([0.5, -1.0, 2.0])
    out = LogitBias(vec, alpha=4.0).apply(logits)
    assert np.allclose(out, logits + 4.0 * vec)


def test_logit_bias_rejects_width_mismatch():
    with pytest.raises(ValueError):
        LogitBias(np.zeros(3), alpha=1.0).apply(np.zeros(5))


def test_margin_scores_match_subspace_margin():
    w = np.array([1.0, -1.0])
    b = 0.5
    sub = BeliefSubspace(w=w, b=b, layer=0, train_acc=1.0, heldout_acc=1.0)
    states = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected = sub.margin(states)
    assert np.allclose(margin_scores(sub, states), expected)
