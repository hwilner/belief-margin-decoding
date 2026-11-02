"""Matched controls for the activation-steering arms.

The Phase-0 matrix contains one control for the *scoring* mechanisms
(``margin_shuffled``). The ``act`` mechanism, which changes the forward pass,
has no matched control in the matrix, and that is a real weakness: an
activation-steering arm that "helps" is indistinguishable from generic
confidence steering unless something with the same magnitude and the same
layer is shown *not* to help.

This script provides the two standard controls, both matched to a chosen real
arm in **layer**, **direction norm**, and **alpha**:

``random``
    A uniformly random unit direction. Same norm, same layer, same alpha.
    If the real direction beats this, the effect is about the *belief
    subspace* rather than about perturbing the residual stream at that layer.

``reversed``
    The negation of the fitted belief direction. Same norm, same layer, same
    alpha. If pushing *against* the belief subspace hurts as much as pushing
    with it helps, the arm is measuring a generic confidence effect.

Both controls reuse the cached activations from the base pass, so only the
model re-run is repeated, and both are reported with the same statistics as
every other arm.

Usage::

    python src/run_controls.py --layer 12 --alpha 1.0 --out-dir results/controls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bmd.data import generate_items, render_candidate, render_prompt  # noqa: E402
from bmd.evaluate import bootstrap_ci, error_breakdown, paired_test, tom_accuracy  # noqa: E402
from bmd.intervene import ActivationSteerer  # noqa: E402
from bmd.model import load_model, score_item  # noqa: E402
from bmd.subspace import fit_subspace  # noqa: E402


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(v) for v in value.split(",") if v.strip())


def _label_matrix(items) -> np.ndarray:
    y = np.zeros(len(items) * 3, dtype=int)
    for i, item in enumerate(items):
        y[i * 3 + item.correct_index] = 1
    return y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Matched random-direction and sign-flip controls.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=_parse_floats, default=(4, 8, 12, 16, 20, 24))
    parser.add_argument("--layer", type=int, required=True, help="layer at which to run the controls")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--probe-kind", default="lda", choices=["lda", "logreg", "mlp"])
    parser.add_argument("--cache", default="results/base_cache.npz")
    parser.add_argument("--out-dir", default="results/controls")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    layer = int(args.layer)
    layers = [int(v) for v in args.layers]
    if layer not in layers:
        layers = sorted({*layers, layer})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = generate_items(args.n_items, seed=args.seed)
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"error: cache {cache_path} not found. Run src/run_phase0.py first.", file=sys.stderr)
        return 2
    with np.load(cache_path, allow_pickle=False) as data:
        states = np.asarray(data["states"])
        base_logprobs = np.asarray(data["base_logprobs"])
    if states.shape[2] != len(layers):
        raise SystemExit(
            f"cache was built for {states.shape[2]} layers {layers}, but the controls need "
            f"{len(layers)}. Re-run the base pass with --layers {','.join(str(ly) for ly in layers)}."
        )
    li = layers.index(layer)
    X = states[:, :, li, :].reshape(-1, states.shape[-1]).astype(np.float64)
    y = _label_matrix(items)

    sub = fit_subspace(X, y, layer=layer, kind=args.probe_kind, seed=args.seed)
    real_dir = sub.w / np.linalg.norm(sub.w)

    rng = np.random.default_rng(args.seed + 977)
    rand_dir = rng.normal(size=real_dir.shape)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)

    scale = float(np.mean(np.linalg.norm(X, axis=1)))
    device = args.device or "cpu"
    model, tokenizer = load_model(args.model_id, device=device)

    base_pred = base_logprobs.argmax(axis=1)
    correct = np.array([it.correct_index for it in items])
    base_correct = (base_pred == correct).astype(float)

    def record(label: str, direction: np.ndarray, note: str) -> dict:
        steerer = ActivationSteerer(direction, layer=layer, alpha=args.alpha, scale=scale)
        preds = []
        t0 = time.time()
        for i, item in enumerate(items):
            sc = score_item(
                model,
                tokenizer,
                render_prompt(item),
                [render_candidate(item, j) for j in range(3)],
                layers,
                device,
                steerer=steerer,
            )
            preds.append(int(np.argmax([c.logprob for c in sc.candidates])))
            if (i + 1) % 50 == 0:
                print(f"[control] {label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        pred = np.array(preds)
        per_item = (pred == correct).astype(float)
        lo, hi = bootstrap_ci(per_item, seed=args.seed)
        stat, pval = paired_test(per_item, base_correct)
        payload = {
            "arm": {"label": label, "mechanism": "act_control", "layer": layer, "alpha": args.alpha},
            "n_items": len(items),
            "accuracy": tom_accuracy(pred, items),
            "accuracy_ci95": [lo, hi],
            "error_breakdown": error_breakdown(pred, items),
            "world_rate": error_breakdown(pred, items)["world"] / len(items),
            "other_rate": error_breakdown(pred, items)["other"] / len(items),
            "vs_none_wilcoxon": {"statistic": stat, "p_value": pval},
            "note": note,
            "scale": scale,
            "wall_clock_s": round(time.time() - t0, 1),
            # Per-item predictions, so any statistic can be recomputed exactly
            # later without re-running the model.
            "predictions": [int(v) for v in pred],
        }
        (out_dir / f"{label}.json").write_text(json.dumps(payload, indent=2))
        print(
            f"[control] {label:22s} acc={payload['accuracy']:.3f} "
            f"world={payload['world_rate']:.3f} other={payload['other_rate']:.3f} p={pval:.3g}",
            flush=True,
        )
        return payload

    results = {}
    results["act_real"] = record(
        "act_real",
        real_dir,
        "the fitted belief direction, re-run at the control layer so that all three "
        "arms are produced by the same code path on the same day",
    )
    results["act_random"] = record(
        "act_random",
        rand_dir,
        "CONTROL: uniformly random direction, matched norm, layer and alpha to the real arm",
    )
    results["act_reversed"] = record(
        "act_reversed",
        -real_dir,
        "CONTROL: negation of the fitted belief direction, matched norm, layer and alpha",
    )

    summary = {
        "config": {
            "model_id": args.model_id,
            "n_items": args.n_items,
            "seed": args.seed,
            "layer": layer,
            "alpha": args.alpha,
            "probe_kind": args.probe_kind,
            "layers": layers,
        },
        "probe": {
            "train_acc": sub.train_acc,
            "heldout_acc": sub.heldout_acc,
            "w_norm": float(np.linalg.norm(sub.w)),
        },
        "scale": scale,
        "base_accuracy": tom_accuracy(base_pred, items),
        "arms": results,
    }
    (out_dir / "controls_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'controls_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
