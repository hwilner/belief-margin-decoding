#!/usr/bin/env python3
"""Leakage-controlled re-analysis of the scoring arms.

The Phase-0 matrix fits one probe on 80 % of the (item, candidate) pairs and
evaluates the ``margin`` and ``logit`` arms on **all** pairs. The arms
therefore score items whose pairs were in the probe's training split, and the
numbers they produce are partly the probe *memorising* its training pairs
rather than the intervention doing anything.

This script re-does the analysis with the split at the level that matters:

* items are split 80 / 20 into a **training** and a **held-out** set
  (deterministic, seeded);
* the probe is fitted on the training items' pairs only;
* every arm is scored on the **held-out items only**.

It reports, per layer and per alpha:

* the probe's pair-wise held-out accuracy, alongside the trivial baseline for
  a pair-wise score (2/3 — always answering "not the belief-consistent
  candidate" is right two thirds of the time, because two of the three
  candidates in an item are negatives);
* the **3-way** held-out accuracy: given the three candidates of an item,
  does the probe rank the belief-consistent one first? Chance is 1/3;
* the held-out accuracy of each scoring arm, so the in-sample number in the
  matrix summary can be compared directly with its honest counterpart.

The probe's pair-wise accuracy and its 3-way ranking accuracy are different
questions, and only the second one is what a belief-margin intervention
actually relies on.

Usage::

    python tools/analyze_heldout.py
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
from bmd.evaluate import bootstrap_ci, paired_test  # noqa: E402
from bmd.subspace import fit_subspace, normalize_margins  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(ROOT / "results" / "base_cache.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "heldout"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kind", default="lda")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--layers", default="4,8,12,16,20,24",
                    help="the layer numbers the cache was built with, in cache order")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    cache = Path(args.cache)
    with np.load(cache, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        base_logprobs = np.asarray(data["base_logprobs"], dtype=np.float64)
        key = str(data["key"])

    items = generate_items(args.n_items, seed=args.seed)
    n = len(items)
    # The cache stores one slice per REQUESTED layer, in the order they were
    # requested. The axis index is not the layer number.
    layer_numbers = [int(v) for v in args.layers.split(",")]
    if len(layer_numbers) != states.shape[2]:
        print(
            f"error: the cache holds {states.shape[2]} layer slices but --layers lists "
            f"{len(layer_numbers)}",
            file=sys.stderr,
        )
        return 2
    dim = states.shape[3]

    # ---- item-level split (the probe never sees a held-out item)
    rng = np.random.default_rng(args.seed + 4242)
    order = rng.permutation(n)
    n_test = int(round(n * args.holdout_frac))
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    print(f"items: {n} -> train {len(train_idx)} / held-out {len(test_idx)}")

    # ---- k-fold cross-validation over items.
    #
    # A single 80/20 split leaves only 40 items, which is too few to resolve
    # the differences the matrix is trying to measure. The k-fold estimate
    # uses every item exactly once as a held-out item, so it is an
    # out-of-fold prediction over all 200 items, and the bootstrap resamples
    # over those 200 items.
    folds = [np.sort(f) for f in np.array_split(rng.permutation(n), args.n_folds)]

    correct = np.array([it.correct_index for it in items])
    y_all = np.zeros(n * 3, dtype=int)
    for i, item in enumerate(items):
        y_all[i * 3 + item.correct_index] = 1

    train_pairs = (train_idx[:, None] * 3 + np.arange(3)).ravel()
    test_pairs = (test_idx[:, None] * 3 + np.arange(3)).ravel()

    rows = []
    for li, layer in enumerate(layer_numbers):
        X = states[:, :, li, :].reshape(-1, dim).astype(np.float64)

        # --- k-fold cross-validation: out-of-fold predictions for every item
        oof_correct = np.zeros(n, dtype=bool)
        oof_correct_shuf = np.zeros(n, dtype=bool)
        oof_arm = {f"margin-a{a:g}": np.zeros(n, dtype=bool) for a in alphas}
        oof_arm_shuf = {f"margin-a{a:g}": np.zeros(n, dtype=bool) for a in alphas}
        oof_base = np.zeros(n, dtype=bool)
        for fold in folds:
            held = np.concatenate([f for f in folds if not np.array_equal(f, fold)])
            train_p = (held[:, None] * 3 + np.arange(3)).ravel()
            sub_k = fit_subspace(X[train_p], y_all[train_p], layer=layer, kind=args.probe_kind, seed=args.seed)
            sub_k_shuf = fit_subspace(
                X[train_p],
                np.random.default_rng(args.seed + 7).permutation(y_all[train_p]),
                layer=layer, kind=args.probe_kind, seed=args.seed,
            )
            marg_k = normalize_margins(sub_k.margin(X)).reshape(n, 3)
            marg_k_shuf = normalize_margins(sub_k_shuf.margin(X)).reshape(n, 3)
            oof_correct[fold] = marg_k[fold].argmax(axis=1) == correct[fold]
            oof_correct_shuf[fold] = marg_k_shuf[fold].argmax(axis=1) == correct[fold]
            oof_base[fold] = base_logprobs[fold].argmax(axis=1) == correct[fold]
            for a in alphas:
                oof_arm[f"margin-a{a:g}"][fold] = (
                    (base_logprobs[fold] + a * marg_k[fold]).argmax(axis=1) == correct[fold]
                )
                oof_arm_shuf[f"margin-a{a:g}"][fold] = (
                    (base_logprobs[fold] + a * marg_k_shuf[fold]).argmax(axis=1) == correct[fold]
                )

        cv_three_way = oof_correct.astype(float)
        cv_shuf = oof_correct_shuf.astype(float)
        cv_base = oof_base.astype(float)
        cv_lo, cv_hi = bootstrap_ci(cv_three_way, seed=args.seed)
        cv_stat, cv_p = paired_test(cv_three_way, cv_base)
        cv_stat_ctl, cv_p_ctl = paired_test(cv_three_way, cv_shuf)

        # --- the single 80/20 split, kept as a second, independent estimate
        sub = fit_subspace(X[train_pairs], y_all[train_pairs], layer=layer, kind=args.probe_kind, seed=args.seed)
        marg_all = normalize_margins(sub.margin(X)).reshape(n, 3)

        # pair-wise held-out accuracy, and the trivial baseline for it.
        # The prediction is the sign of the margin; the score is whether it
        # matches the label. A probe that always answered "not the
        # belief-consistent candidate" would be right two times in three,
        # because two of an item's three candidates are negatives.
        pair_pred = (sub.margin(X[test_pairs]) > 0).astype(int)
        pair_acc = float((pair_pred == y_all[test_pairs]).mean())
        trivial_pair = float(max(np.bincount(y_all[test_pairs], minlength=2)) / len(test_pairs))

        # 3-way held-out ranking accuracy: does the margin rank the right
        # candidate first among the three?
        three_way = (marg_all[test_idx].argmax(axis=1) == correct[test_idx]).astype(float)
        three_way_acc = float(three_way.mean())

        base_test = (base_logprobs[test_idx].argmax(axis=1) == correct[test_idx]).astype(float)
        lo3, hi3 = bootstrap_ci(three_way, seed=args.seed)
        stat3, p3 = paired_test(three_way, base_test)

        row = {
            "layer": layer,
            "n_folds": args.n_folds,
            "cv_three_way_acc": float(cv_three_way.mean()),
            "cv_three_way_ci95": [cv_lo, cv_hi],
            "cv_three_way_shuffled_acc": float(cv_shuf.mean()),
            "cv_base_acc": float(cv_base.mean()),
            "cv_vs_base_wilcoxon": {"statistic": cv_stat, "p_value": cv_p},
            "cv_vs_shuffled_wilcoxon": {"statistic": cv_stat_ctl, "p_value": cv_p_ctl},
            "split80_20_pairwise_heldout_acc": pair_acc,
            "split80_20_pairwise_trivial_baseline": trivial_pair,
            "split80_20_three_way_acc": three_way_acc,
            "split80_20_base_acc": float(base_test.mean()),
            "split80_20_vs_base_wilcoxon": {"statistic": stat3, "p_value": p3},
            "arms": {},
        }
        for alpha in alphas:
            name = f"margin-a{alpha:g}"
            per_item = oof_arm[name].astype(float)
            stat_ctl, p_ctl = paired_test(per_item, oof_arm_shuf[name].astype(float))
            stat_base, p_base_arm = paired_test(per_item, cv_base)
            lo_a, hi_a = bootstrap_ci(per_item, seed=args.seed)
            row["arms"][name] = {
                "cv_acc": float(per_item.mean()),
                "cv_ci95": [lo_a, hi_a],
                "cv_acc_shuffled_control": float(oof_arm_shuf[name].mean()),
                "margin_minus_control": float(per_item.mean() - oof_arm_shuf[name].mean()),
                "margin_vs_control_wilcoxon": {"statistic": stat_ctl, "p_value": p_ctl},
                "margin_vs_base_wilcoxon": {"statistic": stat_base, "p_value": p_base_arm},
            }
        rows.append(row)
        print(
            f"L{layer:<3d} CV 3-way={row['cv_three_way_acc']:.3f} "
            f"[{cv_lo:.3f},{cv_hi:.3f}] (shuffled {row['cv_three_way_shuffled_acc']:.3f}, "
            f"base {row['cv_base_acc']:.3f}) p={cv_p:.3g} | "
            f"80/20 3-way={three_way_acc:.3f} pair={pair_acc:.3f} (trivial {trivial_pair:.3f})"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cache_key": key,
        "n_items": n,
        "seed": args.seed,
        "probe_kind": args.probe_kind,
        "split": f"item-level {args.n_folds}-fold cross-validation",
        "n_train_items": len(train_idx),
        "n_heldout_items": len(test_idx),
        "rows": rows,
    }
    (out_dir / "heldout_analysis.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"### Belief probe, {args.n_folds}-fold cross-validation over items (out-of-fold)",
        "",
        "| layer | 3-way ranking acc | 95% CI | shuffled control | base model | p vs base | p vs shuffled |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| L{r['layer']} | **{r['cv_three_way_acc']:.3f}** | "
            f"[{r['cv_three_way_ci95'][0]:.3f}, {r['cv_three_way_ci95'][1]:.3f}] | "
            f"{r['cv_three_way_shuffled_acc']:.3f} | {r['cv_base_acc']:.3f} | "
            f"{r['cv_vs_base_wilcoxon']['p_value']:.3g} | "
            f"{r['cv_vs_shuffled_wilcoxon']['p_value']:.3g} |"
        )
    lines.append("")
    lines.append("### The scoring arms, out-of-fold")
    lines.append("")
    lines.append("| layer | alpha | margin | margin_shuffled | difference | p vs control | p vs base |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        for alpha, v in r["arms"].items():
            a = alpha.split("-")[1]
            lines.append(
                f"| L{r['layer']} | {a} | {v['cv_acc']:.3f} | "
                f"{v['cv_acc_shuffled_control']:.3f} | {v['margin_minus_control']:+.3f} | "
                f"{v['margin_vs_control_wilcoxon']['p_value']:.3g} | "
                f"{v['margin_vs_base_wilcoxon']['p_value']:.3g} |"
            )
    lines.append("")
    lines.append("### The single 80/20 split, as a second independent estimate")
    lines.append("")
    lines.append(
        "| layer | 3-way ranking acc (40 items) | pair-wise acc | pair-wise trivial baseline | base | p vs base |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| L{r['layer']} | {r['split80_20_three_way_acc']:.3f} | "
            f"{r['split80_20_pairwise_heldout_acc']:.3f} | "
            f"{r['split80_20_pairwise_trivial_baseline']:.3f} | {r['split80_20_base_acc']:.3f} | "
            f"{r['split80_20_vs_base_wilcoxon']['p_value']:.3g} |"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "heldout_analysis.md").write_text(table)
    print(f"\nwrote {out_dir/'heldout_analysis.json'} and {out_dir/'heldout_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
