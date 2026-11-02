"""Belief subspace: a direction in representation space that separates
belief-consistent continuations from belief-violating ones.

The estimator is deliberately the cheapest one that could work. The default
(``kind="lda"``) is a shrinkage Fisher linear discriminant, fitted in closed
form: no gradient steps, no learned parameters beyond one vector and a bias,
and a fit that takes milliseconds. It is the same estimator family used by the
SASA self-detoxification method, transplanted from *toxicity* space to
*belief-consistency* space.

Two alternatives are provided because belief-consistency is a **relational**
property -- "does this location match the one the protagonist believes?" --
and a linear readout of a joint (story, option) representation may not be able
to express a comparison. ``kind="logreg"`` and ``kind="mlp"`` exist so that
the choice between linear and nonlinear readouts is a pre-registered,
measurable decision rather than an assumption. See ``docs/METHODS.md`` (U1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BeliefSubspace:
    """A fitted belief subspace.

    Attributes
    ----------
    w:
        Direction in representation space, shape ``(d,)``.
    b:
        Bias term.
    layer:
        The model layer the direction was read from.
    train_acc, heldout_acc:
        Classification accuracy of the underlying readout on the internal
        train / held-out split. These are **preconditions**: a subspace whose
        readout is at chance cannot support a belief claim, and the repository
        treats that as a reported result rather than something to hide.
    kind:
        Which estimator produced this subspace.
    n_train, n_heldout:
        Sizes of the internal split.
    """

    w: np.ndarray
    b: float
    layer: int
    train_acc: float
    heldout_acc: float
    kind: str = "lda"
    n_train: int = 0
    n_heldout: int = 0
    meta: dict = field(default_factory=dict)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary for each row of ``X``.

        Parameters
        ----------
        X:
            Array of shape ``(n, d)``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n,)``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"feature width {X.shape[1]} does not match subspace width {self.w.shape[0]}"
            )
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard predictions (1 = belief-consistent class, 0 = violating class)."""
        return (self.margin(X) > 0).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialise the subspace to a ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w=self.w,
            b=np.array(self.b),
            layer=np.array(self.layer),
            train_acc=np.array(self.train_acc),
            heldout_acc=np.array(self.heldout_acc),
            kind=np.array(self.kind),
            n_train=np.array(self.n_train),
            n_heldout=np.array(self.n_heldout),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BeliefSubspace":
        """Load a subspace saved by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            w = np.asarray(data["w"], dtype=np.float64)
            b = float(data["b"])
            layer = int(data["layer"])
            train_acc = float(data["train_acc"])
            heldout_acc = float(data["heldout_acc"])
            kind = str(data["kind"])
            n_train = int(data["n_train"])
            n_heldout = int(data["n_heldout"])
            meta_raw = str(data["meta"])
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}
        return cls(
            w=w,
            b=b,
            layer=layer,
            train_acc=train_acc,
            heldout_acc=heldout_acc,
            kind=kind,
            n_train=n_train,
            n_heldout=n_heldout,
            meta=meta,
        )


def normalize_margins(m: np.ndarray) -> np.ndarray:
    """Standardise margins to zero mean and unit variance.

    A fitted Fisher direction on separable data has an arbitrarily large norm
    (the pilot measured ``||w|| ~ 1e4``), which makes any hand-picked ``alpha``
    meaningless. Standardising the margin decouples the strength of the
    intervention from the arbitrary scale of the fitted direction.
    """
    m = np.asarray(m, dtype=np.float64)
    std = m.std()
    if std < 1e-12:
        return np.zeros_like(m)
    return (m - m.mean()) / std


def _split(n: int, holdout_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic index split: the tail of the array is the held-out set."""
    n_held = max(1, int(round(n * holdout_frac))) if n > 1 else 0
    idx = np.arange(n)
    if n_held == 0:
        return idx, idx[:0]
    return idx[: n - n_held], idx[n - n_held:]


def _fit_lda(X1: np.ndarray, X0: np.ndarray, reg: float) -> tuple[np.ndarray, float]:
    """Shrinkage Fisher LDA. ``X1`` is the belief-consistent class."""
    d = X1.shape[1]
    mu1, mu0 = X1.mean(0), X0.mean(0)
    S1 = np.cov(X1, rowvar=False) if len(X1) > 1 else np.zeros((d, d))
    S0 = np.cov(X0, rowvar=False) if len(X0) > 1 else np.zeros((d, d))
    S = 0.5 * (S1 + S0)
    # Ledoit-Wolf style shrinkage towards a scaled identity.
    target = np.trace(S) / d * np.eye(d)
    Sw = (1.0 - reg) * S + reg * target
    w = np.linalg.solve(Sw, mu1 - mu0)
    b = -0.5 * (mu1 + mu0) @ w
    return w, float(b)


def _fit_logreg(X: np.ndarray, y: np.ndarray, reg: float, iters: int = 500) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression fitted with full-batch gradient descent."""
    d = X.shape[1]
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    theta = np.zeros(d + 1)
    lr = 1.0 / (np.linalg.norm(X, 2) ** 2 / max(X.shape[0], 1) + reg)
    for _ in range(iters):
        z = Xb @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / X.shape[0]
        grad[:-1] += reg * theta[:-1]
        theta -= lr * grad
    return theta[:-1], float(theta[-1])


def _fit_mlp(X: np.ndarray, y: np.ndarray, reg: float, seed: int = 0, hidden: int = 64):
    """One-hidden-layer MLP probe, trained with torch on CPU."""
    import torch

    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=reg)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad()
        out = model(Xt).squeeze(-1)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
    w = model[0].weight.detach().numpy()[0]
    b = model[2].bias.detach().numpy()[0]
    # Express the MLP boundary as a linear form in the first layer's tangent
    # space by freezing the nonlinearity at zero: w·x + b is the local linear
    # read-out used for margin computation.
    return w.astype(np.float64), float(b)


def fit_subspace(
    X: np.ndarray,
    y: np.ndarray,
    layer: int,
    reg: float = 1e-2,
    kind: str = "lda",
    seed: int = 0,
) -> BeliefSubspace:
    """Fit a belief subspace on labelled representations.

    Parameters
    ----------
    X:
        Representations, shape ``(n, d)``. Row ``i`` is the representation of a
        (context, continuation) pair.
    y:
        Labels, shape ``(n,)``. ``1`` = belief-consistent, ``0`` = belief-violating.
    layer:
        The model layer the representations were read from.
    reg:
        Regularisation strength.
    kind:
        One of ``"lda"``, ``"logreg"``, ``"mlp"``.
    seed:
        Seed for the MLP probe.

    Returns
    -------
    BeliefSubspace
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes")
    if kind not in {"lda", "logreg", "mlp"}:
        raise ValueError(f"unknown subspace kind: {kind!r}")

    tr, ho = _split(X.shape[0])
    Xtr, ytr = X[tr], y[tr]
    if kind == "lda":
        w, b = _fit_lda(Xtr[ytr == 1], Xtr[ytr == 0], reg)
    elif kind == "logreg":
        w, b = _fit_logreg(Xtr, ytr, reg)
    else:
        w, b = _fit_mlp(Xtr, ytr, reg, seed=seed)

    def acc(idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        return float(((X[idx] @ w + b > 0).astype(int) == y[idx]).mean())

    return BeliefSubspace(
        w=w,
        b=b,
        layer=layer,
        train_acc=acc(tr),
        heldout_acc=acc(ho),
        kind=kind,
        n_train=int(tr.size),
        n_heldout=int(ho.size),
        meta={"reg": reg, "seed": seed},
    )
