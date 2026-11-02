#!/usr/bin/env python3
"""Recompute the scoring arms and test each one against its matched control.

The matrix runner tests every arm against the **base model**. That is the
wrong comparison for the scoring mechanisms, and it is worth being explicit
about why.

The question the ``margin`` arm answers is not "does adding a probe margin
help?" in the abstract. It is "does adding the margin of a subspace fitted on
the *real* belief labels help **more than** adding the margin of a subspace
fitted on *shuffled* labels?". The second quantity is what rules out generic
confidence steering, and the two arms must be compared **paired by item**, at
the **same layer and the same alpha**, to answer it.

This script re-derives the ``margin``, ``margin_shuffled`` and ``logit``
arms from the cached base pass — no model forward pass is needed for the
margin arms — and reports, for every (layer, alpha):

* the accuracy of each arm,
* the **paired** Wilcoxon test of ``margin`` against ``margin_shuffled``,
* the same test of each arm against the base model,
* the fraction of the oracle's gain that each arm recovers.

It also stores the per-item predictions, so the numbers can be recomputed
again later without re-running anything.

Usage::

    python tools/analyze_control_pairs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bmd.data import generate_items  # noqa: E402
from bmd.evaluate import bootstrap_ci, error_breakdown, paired_test, tom_accuracy  # noqa: E402
from bmd.matrix import _final_norm_weight, _gram_matrix, _label_matrix  # noqa: E402
from bmd.model import load_model  # noqa: E402
from bmd.subspace import fit_subspace, normalize_margins  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--cache", default=str(ROOT / "results" / "base_cache.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "control_pairs"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kind", default="lda")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        print(f"error: {cache} not found. Run src/run_phase0.py first.", file=sys.stderr)
        return 2
    with np.load(cache, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        base_logprobs = np.asarray(data["base_logprobs"], dtype=np.float64)
        final_states = np.asarray(data["final_states"], dtype=np.float32)
        oracle_logprobs = np.asarray(data["oracle_logprobs"], dtype=np.float64)
        key = str(data["key"])

    items = generate_items(args.n_items, seed=args.seed)
    if len(items) * 3 != base_logprobs.shape[0]:
        print(
            f"error: cache holds {base_logprobs.shape[0]//3} items, --n-items says {args.n_items}",
            file=sys.stderr,
        )
        return 2

    layers = list(range(states.shape[2]))
    n_layer = states.shape[3]
    y = _label_matrix(items)
    correct = np.array([it.correct_index for it in items])
    base_pred = base_logprobs.argmax(axis=1)
    base_correct = (base_pred == correct).astype(float)
    oracle_acc = tom_accuracy(oracle_logprobs.argmax(axis=1), items)
    base_acc = tom_accuracy(base_pred, items)
    headroom = oracle_acc - base_acc

    device = args.device or "cpu"
    model, _tokenizer = load_model(args.model_id, device=device)
    G = _gram_matrix(model)
    gamma = _final_norm_weight(model)

    rng = np.random.default_rng(args.seed + 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for li, layer in enumerate(layers):
        X = states[:, :, li, :].reshape(-1, n_layer).astype(np.float64)
        sub = fit_subspace(X, y, layer=layer, kind=args.probe_kind, seed=args.seed)
        sub_shuf = fit_subspace(X, rng.permutation(y), layer=layer, kind=args.probe_kind, seed=args.seed)

        marg = normalize_margins(sub.margin(X)).reshape(len(items), 3)
        marg_shuf = normalize_margins(sub_shuf.margin(X)).reshape(len(items), 3)

        if gamma is not None:
            rms = np.linalg.norm(sub.w) / np.sqrt(sub.w.shape[0])
            n_vec = (sub.w / (rms + 1e-12)) * gamma
            g_vec = G @ n_vec
            bias = (final_states.reshape(-1, n_layer).astype(np.float64) @ g_vec).reshape(len(items), 3)
        else:
            bias = np.zeros((len(items), 3))

        for alpha in (0.25, 0.5, 1.0, 2.0, 4.0):
            preds = {
                "margin": (base_logprobs + alpha * marg).argmax(axis=1),
                "margin_shuffled": (base_logprobs + alpha * marg_shuf).argmax(axis=1),
                "logit": (base_logprobs + alpha * bias).argmax(axis=1),
            }
            per_item = {k: (v == correct).astype(float) for k, v in preds.items()}
            stat_ctl, p_ctl = paired_test(per_item["margin"], per_item["margin_shuffled"])
            stat_base, p_base = paired_test(per_item["margin"], base_correct)
            accs = {k: tom_accuracy(v, items) for k, v in preds.items()}
            lo_m, hi_m = bootstrap_ci(per_item["margin"], seed=args.seed)
            lo_c, hi_c = bootstrap_ci(per_item["margin_shuffled"], seed=args.seed)
            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "accuracy": accs,
                    "ci95": {
                        "margin": [lo_m, hi_m],
                        "margin_shuffled": [lo_c, hi_c],
                    },
                    "probe": {"train_acc": sub.train_acc, "heldout_acc": sub.heldout_acc,
                              "shuffled_heldout_acc": sub_shuf.heldout_acc},
                    "margin_vs_control_wilcoxon": {"statistic": stat_ctl, "p_value": p_ctl},
                    "margin_vs_base_wilcoxon": {"statistic": stat_base, "p_value": p_base},
                    "oracle_gain_recovered": {
                        k: ((v - base_acc) / headroom) if abs(headroom) > 1e-9 else None
                        for k, v in accs.items()
                    },
                    "error_breakdown": {k: error_breakdown(v, items) for k, v in preds.items()},
                    "predictions": {k: [int(x) for x in v] for k, v in preds.items()},
                }
            )
            print(
                f"L{layer:<3d} a={alpha:<5g} margin={accs['margin']:.3f} "
                f"control={accs['margin_shuffled']:.3f} logit={accs['logit']:.3f} "
                f"p(margin vs control)={p_ctl:.3g} p(margin vs base)={p_base:.3g}",
                flush=True,
            )

    summary = {
        "cache_key": key,
        "model_id": args.model_id,
        "n_items": len(items),
        "seed": args.seed,
        "probe_kind": args.probe_kind,
        "base_accuracy": base_acc,
        "oracle_accuracy": oracle_acc,
        "oracle_headroom": headroom,
        "rows": rows,
    }
    (out_dir / "control_pairs.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "| layer | alpha | margin | control | margin − control | p (margin vs control) | p (margin vs base) | oracle gain recovered |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rec = r["oracle_gain_recovered"]["margin"]
        rec_s = "n/a" if rec is None else f"{rec:+.2f}"
        lines.append(
            f"| L{r['layer']} | {r['alpha']:g} | {r['accuracy']['margin']:.3f} | "
            f"{r['accuracy']['margin_shuffled']:.3f} | "
            f"{r['accuracy']['margin'] - r['accuracy']['margin_shuffled']:+.3f} | "
            f"{r['margin_vs_control_wilcoxon']['p_value']:.3g} | "
            f"{r['margin_vs_base_wilcoxon']['p_value']:.3g} | {rec_s} |"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "control_pairs.md").write_text(table)
    print(f"\nwrote {out_dir/'control_pairs.json'} and {out_dir/'control_pairs.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
