"""The three intervention mechanisms compared in the Phase-0 matrix.

    none     no intervention (base model)
    prompt   the partner belief is stated in the prompt -- the mechanism the
             literature actually uses
    act      activation steering: add ``alpha * direction`` to the hidden
             state at one layer
    margin   margin decoding: add ``alpha * margin(candidate)`` to each
             candidate's score, where the margin is the signed distance of the
             candidate's representation to the belief boundary
    logit    logit bias: add ``alpha * LMHead(norm(direction))`` to the logits

``act``, ``margin`` and ``logit`` are the training-free, representation-level
mechanisms this repository exists to test.
"""

from __future__ import annotations

import numpy as np


class ActivationSteerer:
    """Adds a scaled direction to a hidden state.

    Parameters
    ----------
    direction:
        Unit-free direction, shape ``(d,)``.
    layer:
        Model layer at which the intervention is applied.
    alpha:
        Intervention strength.
    scale:
        Additional multiplier, typically set to the mean hidden-state norm at
        that layer so that ``alpha`` is comparable across layers.
    """

    def __init__(self, direction: np.ndarray, layer: int, alpha: float, scale: float = 1.0):
        self.direction = np.asarray(direction, dtype=np.float64)
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.scale = float(scale)

    def apply(self, hidden: np.ndarray) -> np.ndarray:
        """Return ``hidden + alpha * scale * direction``."""
        h = np.asarray(hidden, dtype=np.float64)
        if h.shape[-1] != self.direction.shape[0]:
            raise ValueError(
                f"hidden width {h.shape[-1]} does not match direction width {self.direction.shape[0]}"
            )
        return h + self.alpha * self.scale * self.direction

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ActivationSteerer(layer={self.layer}, alpha={self.alpha}, scale={self.scale})"


class LogitBias:
    """Adds a scaled vocabulary-space bias to the logits."""

    def __init__(self, vocab_vector: np.ndarray, alpha: float):
        self.vocab_vector = np.asarray(vocab_vector, dtype=np.float64)
        self.alpha = float(alpha)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Return ``logits + alpha * vocab_vector``."""
        z = np.asarray(logits, dtype=np.float64)
        if z.shape[-1] != self.vocab_vector.shape[0]:
            raise ValueError(
                f"logit width {z.shape[-1]} does not match vocab vector width {self.vocab_vector.shape[0]}"
            )
        return z + self.alpha * self.vocab_vector

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LogitBias(alpha={self.alpha})"


def margin_scores(subspace, states: np.ndarray) -> np.ndarray:
    """Signed distance of each candidate representation to the belief boundary.

    Parameters
    ----------
    subspace:
        A :class:`bmd.subspace.BeliefSubspace`.
    states:
        Candidate representations, shape ``(k, d)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)``. Positive means belief-consistent.
    """
    return subspace.margin(np.asarray(states, dtype=np.float64))
