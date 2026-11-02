#!/usr/bin/env python3
"""Recompute every reported statistic from stored per-item predictions.

This is the audit tool for the repository. It does not run the model.

* the ``none`` and ``prompt_oracle`` arms are re-derived from the cached
  log-probabilities;
* the ``margin``, ``margin_shuffled`` and ``logit`` arms are re-derived from
  the cached activations and the cached head weights;
* the ``act`` arms are read from the per-item predictions stored in their own
  JSON files;
* the three matched control arms are read from ``results/controls/``.

Every arm's accuracy, error breakdown, bootstrap CI and Wilcoxon p-value
against the base model are recomputed and written back into the arm JSON, so
the numbers in ``results/RESULTS.md`` can be regenerated from first
principles at any time.

It also writes the paired comparison between the activation arm and its two
matched controls, which is the comparison the whole repository turns on.

Usage::

    python tools/recompute_all_stats.py
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
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kind", default="lda")
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-logit", action="store_true",
                    help="skip the logit arms, which need the model head")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    layer_numbers = [int(v) for v in args.layers.split(",")]
    results_dir = Path(args.results_dir)
    arm_dir = results_dir / "ablation"

    with np.load(args.cache, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        base_logprobs = np.asarray(data["base_logprobs"], dtype=np.float64)
        oracle_logprobs = np.asarray(data["oracle_logprobs"], dtype=np.float64)
        final_states = np.asarray(data["final_states"], dtype=np.float32)

    items = generate_items(args.n_items, seed=args.seed)
    n = len(items)
    dim = states.shape[3]
    correct = np.array([it.correct_index for it in items])
    y = _label_matrix(items)

    # The head is only needed for the logit mechanism, so it is loaded lazily:
    # recomputing the margin arms must not require a second copy of the model
    # in memory.
    model = _tok = None
    G = gamma = None

    # ------------------------------------------------------------------ preds
    preds: dict[str, np.ndarray] = {}
    preds["none"] = base_logprobs.argmax(axis=1)
    preds["prompt_oracle"] = oracle_logprobs.argmax(axis=1)

    rng = np.random.default_rng(args.seed + 1)
    for li, layer in enumerate(layer_numbers):
        X = states[:, :, li, :].reshape(-1, dim).astype(np.float64)
        sub = fit_subspace(X, y, layer=layer, kind=args.probe_kind, seed=args.seed)
        sub_shuf = fit_subspace(X, rng.permutation(y), layer=layer, kind=args.probe_kind, seed=args.seed)
        marg = normalize_margins(sub.margin(X)).reshape(n, 3)
        marg_shuf = normalize_margins(sub_shuf.margin(X)).reshape(n, 3)
        for alpha in alphas:
            preds[f"margin-L{layer}-a{alpha:g}"] = (base_logprobs + alpha * marg).argmax(axis=1)
            preds[f"margin_shuffled-L{layer}-a{alpha:g}"] = (base_logprobs + alpha * marg_shuf).argmax(axis=1)
        if args.skip_logit:
            continue
        if model is None:
            model, _tok = load_model(args.model_id, device=args.device or "cpu")
            G = _gram_matrix(model)
            gamma = _final_norm_weight(model)
        if gamma is not None:
            rms = np.linalg.norm(sub.w) / np.sqrt(sub.w.shape[0])
            g_vec = G @ (sub.w / (rms + 1e-12) * gamma)
            bias = (final_states.reshape(-1, dim).astype(np.float64) @ g_vec).reshape(n, 3)
        else:
            bias = np.zeros((n, 3))
        for alpha in alphas:
            preds[f"logit-L{layer}-a{alpha:g}"] = (base_logprobs + alpha * bias).argmax(axis=1)

    # activation arms: read the stored per-item predictions. The control run
    # re-ran the real direction at layer 4 and stored its predictions, so an
    # activation arm written before the predictions were added can be filled
    # in from there.
    controls_dir = results_dir / "controls"
    control_preds: dict[str, np.ndarray] = {}
    if controls_dir.exists():
        for path in sorted(controls_dir.glob("act_*.json")):
            payload = json.loads(path.read_text())
            control_preds[payload["arm"]["label"]] = np.array(payload["predictions"], dtype=int)
    for path in sorted(arm_dir.glob("act-L*.json")):
        payload = json.loads(path.read_text())
        label = payload["arm"]["label"]
        stored = payload.get("predictions")
        if stored is None:
            if "act_real" in control_preds and payload["arm"].get("layer") == 4:
                preds[label] = control_preds["act_real"]
                print(f"  ~ {label} has no stored predictions; using the control run's act_real")
                continue
            print(f"  ! {label} has no stored predictions; leaving it alone")
            continue
        preds[label] = np.array(stored, dtype=int)

    # the three matched control arms are recomputed on the same footing as
    # every other arm
    for label, pred in control_preds.items():
        preds.setdefault(label, pred)

    base_correct = (preds["none"] == correct).astype(float)

    # -------------------------------------------------------------- recompute
    rows = []
    for label, pred in sorted(preds.items()):
        per_item = (pred == correct).astype(float)
        lo, hi = bootstrap_ci(per_item, seed=args.seed)
        stat, pval = paired_test(per_item, base_correct)
        breakdown = error_breakdown(pred, items)
        payload = {
            "arm": {"label": label},
            "n_items": n,
            "accuracy": tom_accuracy(pred, items),
            "accuracy_ci95": [lo, hi],
            "error_breakdown": breakdown,
            "world_rate": breakdown["world"] / n,
            "other_rate": breakdown["other"] / n,
            "vs_none_wilcoxon": {"statistic": stat, "p_value": pval},
            "predictions": [int(v) for v in pred],
        }
        rows.append((label, payload))

    # ---------------------------------------------- paired control comparisons
    comparisons = {}
    if controls_dir.exists():
        cpreds = dict(control_preds)
        pairs = [
            ("act_real", "act_random"),
            ("act_real", "act_reversed"),
            ("act_random", "act_reversed"),
        ]
        for a, b in pairs:
            if a not in cpreds or b not in cpreds:
                continue
            pa = (cpreds[a] == correct).astype(float)
            pb = (cpreds[b] == correct).astype(float)
            stat, pval = paired_test(pa, pb)
            dlo, dhi = bootstrap_ci(pa - pb, seed=args.seed)
            comparisons[f"{a}_vs_{b}"] = {
                "accuracy_a": float(pa.mean()),
                "accuracy_b": float(pb.mean()),
                "difference": float(pa.mean() - pb.mean()),
                "difference_ci95": [dlo, dhi],
                "wilcoxon": {"statistic": stat, "p_value": pval},
                "a_better": int(((pa == 1) & (pb == 0)).sum()),
                "b_better": int(((pb == 1) & (pa == 0)).sum()),
            }

    # ---------------------------------------------- the random-direction sweep
    # A single random draw is not a distribution. Where several draws exist at
    # a layer, the real arm is compared against all of them, and the
    # distribution is summarised rather than a single number being quoted.
    random_by_layer: dict[int, list[tuple[str, np.ndarray]]] = {}
    if controls_dir.exists():
        for path in sorted(controls_dir.glob("act_random*.json")):
            payload = json.loads(path.read_text())
            if payload.get("control_kind") != "random":
                continue
            layer = payload.get("arm", {}).get("layer")
            if layer is None or "predictions" not in payload:
                continue
            label = payload["arm"]["label"]
            preds_vec = np.array(payload["predictions"], dtype=int)
            random_by_layer.setdefault(int(layer), []).append((label, preds_vec))

    distribution = {}
    for layer, draws in sorted(random_by_layer.items()):
        real_label = f"act-L{layer}-a1"
        real_preds = preds.get(real_label)
        if real_preds is None:
            continue
        rp = (real_preds == correct).astype(float)
        entries = []
        for label, draw_preds in draws:
            dp = (draw_preds == correct).astype(float)
            stat, pval = paired_test(rp, dp)
            dlo, dhi = bootstrap_ci(rp - dp, seed=args.seed)
            entries.append(
                {
                    "random_label": label,
                    "random_accuracy": float(dp.mean()),
                    "difference": float(rp.mean() - dp.mean()),
                    "difference_ci95": [dlo, dhi],
                    "wilcoxon": {"statistic": stat, "p_value": pval},
                }
            )
        accs = [e["random_accuracy"] for e in entries]
        # Aggregate test over the independent draws: the real arm's paired
        # difference is positive against every one of them. Under the null
        # that the real arm is exchangeable with the draws, each sign is a
        # fair coin, so P(all n positive) = 2^-n and the two-sided exact
        # binomial p-value is 2 * 2^-n for n >= 1.
        #
        # This is the right aggregate statement: the individual paired tests
        # are correlated (they share the real arm and the base predictions),
        # so quoting the smallest of them would be a multiple-comparison
        # selection. The sign test uses every draw and does not cherry-pick.
        n_pos = sum(1 for e in entries if e["difference"] > 0)
        n_draws_total = len(entries)
        sign_p = None
        if n_draws_total > 0:
            from math import comb

            k = max(n_pos, n_draws_total - n_pos)
            tail = sum(comb(n_draws_total, i) for i in range(k, n_draws_total + 1)) / (2**n_draws_total)
            sign_p = min(1.0, 2.0 * tail)
        distribution[f"L{layer}"] = {
            "real_label": real_label,
            "real_accuracy": float(rp.mean()),
            "n_draws": len(accs),
            "random_accuracies": accs,
            "random_mean": float(np.mean(accs)),
            "random_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "random_min": float(np.min(accs)),
            "random_max": float(np.max(accs)),
            "n_random_beating_real": int(sum(1 for a in accs if a > rp.mean())),
            "n_random_tieing_real": int(sum(1 for a in accs if a == rp.mean())),
            "n_draws_the_real_arm_beats": int(n_pos),
            "sign_test_p": sign_p,
            "paired_vs_each": entries,
        }
    if distribution:
        comparisons["act_random_distribution"] = distribution

    # ------------------------------------------------------------------ write
    if not args.dry_run:
        for label, payload in rows:
            path = arm_dir / f"{label}.json"
            if not path.exists():
                continue
            old = json.loads(path.read_text())
            old.update(payload)
            old["arm"] = {**old.get("arm", {}), **payload["arm"]}
            path.write_text(json.dumps(old, indent=2))
        # the three matched control arms live in their own directory and carry
        # the same fields, so they get the same treatment
        for path in sorted(controls_dir.glob("act_*.json")) if controls_dir.exists() else []:
            payload = json.loads(path.read_text())
            label = payload["arm"]["label"]
            if label not in preds:
                continue
            pred = preds[label]
            per_item = (pred == correct).astype(float)
            lo, hi = bootstrap_ci(per_item, seed=args.seed)
            stat, pval = paired_test(per_item, base_correct)
            breakdown = error_breakdown(pred, items)
            payload["accuracy"] = tom_accuracy(pred, items)
            payload["accuracy_ci95"] = [lo, hi]
            payload["error_breakdown"] = breakdown
            payload["world_rate"] = breakdown["world"] / n
            payload["other_rate"] = breakdown["other"] / n
            payload["vs_none_wilcoxon"] = {"statistic": stat, "p_value": pval}
            payload["predictions"] = [int(v) for v in pred]
            path.write_text(json.dumps(payload, indent=2))
        out = results_dir / "control_comparisons.json"
        out.write_text(json.dumps(comparisons, indent=2))
        print(f"wrote {out}")

    print(f"\n{'arm':28s} {'acc':>6s} {'world':>7s} {'other':>7s} {'p vs base':>12s}")
    for label, payload in sorted(rows, key=lambda r: -r[1]["accuracy"]):
        print(
            f"{label:28s} {payload['accuracy']:6.3f} {payload['world_rate']:7.3f} "
            f"{payload['other_rate']:7.3f} {payload['vs_none_wilcoxon']['p_value']:12.3g}"
        )
    print("\npaired control comparisons:")
    for name, c in comparisons.items():
        if name == "act_random_distribution":
            continue
        print(
            f"  {name:34s} diff={c['difference']:+.3f} "
            f"CI=[{c['difference_ci95'][0]:+.3f}, {c['difference_ci95'][1]:+.3f}] "
            f"p={c['wilcoxon']['p_value']:.4g} "
            f"({c['a_better']} vs {c['b_better']})"
        )
    dist = comparisons.get("act_random_distribution")
    if dist:
        print("\nthe random-direction distribution (one draw is not a distribution):")
        for layer, d in sorted(dist.items(), key=lambda kv: int(kv[0][1:])):
            sign_txt = (
                f"sign-test p={d['sign_test_p']:.4g}"
                if d["sign_test_p"] is not None
                else "sign-test n/a"
            )
            print(
                f"  {layer:4s} real={d['real_accuracy']:.3f} | "
                f"{d['n_draws']} draws: mean={d['random_mean']:.3f} "
                f"sd={d['random_std']:.3f} min={d['random_min']:.3f} max={d['random_max']:.3f} | "
                f"beats real: {d['n_random_beating_real']}/{d['n_draws']} | "
                f"real beats draws: {d['n_draws_the_real_arm_beats']}/{d['n_draws']} | "
                f"{sign_txt}"
            )
            for e in d["paired_vs_each"]:
                print(
                    f"        vs {e['random_label']:26s} acc={e['random_accuracy']:.3f} "
                    f"diff={e['difference']:+.3f} "
                    f"CI=[{e['difference_ci95'][0]:+.3f}, {e['difference_ci95'][1]:+.3f}] "
                    f"p={e['wilcoxon']['p_value']:.4g}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
