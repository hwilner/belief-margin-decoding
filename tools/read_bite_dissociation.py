#!/usr/bin/env python3
"""Quantify the read-vs-bite dissociation from stored artefacts.

Phase 0 produced two curves over the same layers:

* **read** -- how well a linear probe can recover the partner belief from the
  residual stream at layer *L* (5-fold cross-validation over *items*, 3-way
  ranking accuracy, reported out-of-fold).
* **bite** -- how much the forward pass changes when the fitted belief
  direction is added to the residual stream at layer *L* (the ``act`` arm at
  alpha = 1, measured against the base model).

The headline result was that these curves are **inverted**: the belief reads
best at L16 (0.720) and steers worst there (0.400), while the layer with the
*worst* read (L4, 0.420) is the only layer that steers at all (0.480). That
was written up in prose. This script turns it into a number with a test
attached, and it needs **no model**: every number is re-derived from the
per-item predictions already stored on disk.

Two statistics are reported over the steerable layers:

1. **Spearman's rho** between read accuracy and bite accuracy. With five
   layers the asymptotic p-value is meaningless, so the p-value is an
   **exact permutation test** over all 5! = 120 orderings of the bite vector.
2. **Kendall's tau-b**, likewise with an exact permutation p-value.

The random-direction controls are folded in as well: for each layer, the
distribution of accuracies obtained by steering with *uniformly random*
directions matched in norm, layer and alpha. The real arm is then placed in
that distribution -- how many random draws beat it, and where it sits in
percentile -- which is the comparison the single-draw Phase-0 control could
not make.

Usage::

    python tools/read_bite_dissociation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bmd.data import generate_items  # noqa: E402
from bmd.evaluate import paired_test  # noqa: E402


def correctness_vector(pred_indices, correct: np.ndarray) -> np.ndarray:
    """Convert predicted option indices into per-item 0/1 correctness.

    The stored ``predictions`` are *which option was chosen* (0, 1 or 2). A
    paired test needs the per-item *correctness* indicator, because that is the
    unit the test resamples over. Passing the raw option indices to
    ``paired_test`` silently tests the wrong quantity -- the indices are not
    comparable across arms, and the difference ``a - b`` means nothing.
    """
    pred = np.asarray(pred_indices, dtype=int)
    correct = np.asarray(correct, dtype=int)
    if pred.shape != correct.shape:
        raise ValueError(f"prediction vector has shape {pred.shape}, expected {correct.shape}")
    return (pred == correct).astype(float)


# --------------------------------------------------------------------------
# Rank statistics (numpy only; the scipy equivalence is asserted in tests)
# --------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved to the average rank of the tie group.

    Same convention as ``scipy.stats.rankdata`` with ``method="average"``.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho with average ranks for ties."""
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b (ties corrected)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    concordant = discordant = 0
    tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = np.sqrt((n * (n - 1) / 2 - tx) * (n * (n - 1) / 2 - ty))
    if denom < 1e-12:
        return float("nan")
    return float((concordant - discordant) / denom)


def exact_permutation_p(x: np.ndarray, y: np.ndarray, statistic: str) -> tuple[float, float, int]:
    """Two-sided exact permutation p-value for a rank statistic.

    Permutes the *second* vector over all orderings (at most 5! = 120 for the
    layer count used here) and reports the fraction of permutations at least
    as extreme as the observed value.
    """
    fn = spearman_rho if statistic == "spearman" else kendall_tau_b
    observed = fn(x, y)
    if not np.isfinite(observed):
        return float("nan"), observed, 0
    count = 0
    total = 0
    for perm in permutations(range(len(y))):
        total += 1
        if abs(fn(x, np.array(y)[list(perm)])) >= abs(observed) - 1e-12:
            count += 1
    return count / total, observed, total


# --------------------------------------------------------------------------
# Artefact loading
# --------------------------------------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_read_curve(heldout_path: Path) -> dict[int, dict]:
    """Read accuracy per layer, from the item-level cross-validated probe."""
    data = _load(heldout_path)
    rows = data["rows"] if isinstance(data.get("rows"), list) else []
    out = {}
    for row in rows:
        out[int(row["layer"])] = {
            "read_acc": float(row["cv_three_way_acc"]),
            "read_shuffled_acc": float(row["cv_three_way_shuffled_acc"]),
            "read_ci95": row.get("cv_three_way_ci95"),
            "read_vs_base_p": row.get("cv_vs_base_wilcoxon", {}).get("p_value"),
            "read_vs_shuffled_p": row.get("cv_vs_shuffled_wilcoxon", {}).get("p_value"),
        }
    return out


def load_act_arms(ablation_dir: Path, layers: list[int]) -> dict[int, dict]:
    """Bite accuracy per layer: the ``act`` arm at alpha = 1."""
    out = {}
    for layer in layers:
        p = ablation_dir / f"act-L{layer}-a1.json"
        if not p.exists():
            continue
        d = _load(p)
        out[layer] = {
            "bite_acc": float(d["accuracy"]),
            "bite_ci95": d.get("accuracy_ci95"),
            "bite_vs_base_p": d.get("vs_none_wilcoxon", {}).get("p_value"),
            "error_breakdown": d.get("error_breakdown"),
            "predictions": d.get("predictions"),
            "probe": d.get("probe"),
            "scale": d.get("scale"),
        }
    return out


def load_random_controls(controls_dir: Path) -> dict[int, list[dict]]:
    """Random-direction control arms per layer.

    Handles both the legacy single-draw file (``act_random.json``, which has
    no ``control_kind`` field) and the repeated-draw files
    (``act_random_s<seed>.json``). Arms are de-duplicated by direction seed,
    so a re-run of the same seed does not get counted twice.
    """
    out: dict[int, list[dict]] = {}
    seen: dict[int, set] = {}
    for p in sorted(controls_dir.glob("act_random*.json")):
        d = _load(p)
        arm = d.get("arm", {})
        layer = arm.get("layer")
        if layer is None:
            continue
        layer = int(layer)
        kind = d.get("control_kind") or ("random" if p.name.startswith("act_random") else None)
        if kind != "random":
            continue
        preds = d.get("predictions")
        # De-duplicate by identity of the *arm*, not by filename. The legacy
        # single-draw file (``act_random.json``) and the repeated-draw file
        # for the same seed (``act_random_s977.json``) are the same direction
        # run through the same code path, and counting both would silently
        # weight that draw twice.
        if preds is not None:
            identity = ("preds", hash(tuple(preds)))
        else:
            identity = ("seed", str(d.get("direction_seed", p.name)))
        bucket = seen.setdefault(layer, set())
        if identity in bucket:
            continue
        bucket.add(identity)
        out.setdefault(layer, []).append(
            {
                "label": arm.get("label", p.stem),
                "file": p.name,
                "direction_seed": d.get("direction_seed"),
                "accuracy": float(d["accuracy"]),
                "accuracy_ci95": d.get("accuracy_ci95"),
                "vs_none_p": d.get("vs_none_wilcoxon", {}).get("p_value"),
                "cos_to_real_direction": d.get("cos_to_real_direction"),
                "predictions": preds,
            }
        )
    for layer in out:
        out[layer].sort(key=lambda r: (str(r["direction_seed"])))
    return out


def find_real_arm(layer: int, controls_dir: Path, ablation_dir: Path) -> dict | None:
    """The real (belief-direction) steering arm for a layer.

    Prefers the control runner's own layer-named arm (``act_real_L<layer>.json``,
    produced by the same code path as the controls), then the legacy
    ``act_real.json`` when its recorded layer matches, and finally the matrix's
    ``act-L<layer>-a1.json``.
    """
    candidates = [
        (controls_dir / f"act_real_L{layer}.json", True),
        (controls_dir / "act_real.json", True),
        (ablation_dir / f"act-L{layer}-a1.json", False),
    ]
    for p, check_layer in candidates:
        if not p.exists():
            continue
        d = _load(p)
        if check_layer:
            recorded = d.get("arm", {}).get("layer")
            if recorded is None or int(recorded) != layer:
                continue
        return {
            "label": d.get("arm", {}).get("label", p.stem),
            "file": p.name,
            "accuracy": float(d["accuracy"]),
            "accuracy_ci95": d.get("accuracy_ci95"),
            "predictions": d.get("predictions"),
        }
    return None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def analyse(read: dict[int, dict], bite: dict[int, dict], controls: dict[int, list[dict]],
            base_acc: float, controls_dir: Path, ablation_dir: Path,
            correct: np.ndarray) -> dict:
    layers = sorted(set(read) & set(bite))
    if len(layers) < 3:
        raise SystemExit(f"need at least 3 layers with both a read and a bite arm, got {layers}")

    rows = []
    for layer in layers:
        r, b = read[layer], bite[layer]
        row = {
            "layer": layer,
            "read_acc": r["read_acc"],
            "read_shuffled_acc": r["read_shuffled_acc"],
            "read_vs_base_p": r["read_vs_base_p"],
            "bite_acc": b["bite_acc"],
            "bite_ci95": b["bite_ci95"],
            "bite_vs_base_p": b["bite_vs_base_p"],
            "bite_minus_base": b["bite_acc"] - base_acc,
            "bite_minus_read": b["bite_acc"] - r["read_acc"],
        }
        ctl = controls.get(layer, [])
        if ctl:
            accs = [c["accuracy"] for c in ctl]
            row["n_random_draws"] = len(accs)
            row["random_acc_mean"] = float(np.mean(accs))
            row["random_acc_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            row["random_acc_min"] = float(np.min(accs))
            row["random_acc_max"] = float(np.max(accs))
            row["random_beats_bite"] = int(sum(1 for a in accs if a > b["bite_acc"]))
            row["random_ties_bite"] = int(sum(1 for a in accs if a == b["bite_acc"]))
            real = find_real_arm(layer, controls_dir, ablation_dir)
            row["real_arm"] = real
        else:
            row["n_random_draws"] = 0
            row["real_arm"] = None
        rows.append(row)

    read_vec = np.array([r["read_acc"] for r in rows])
    bite_vec = np.array([r["bite_acc"] for r in rows])

    p_rho, rho, n_perm = exact_permutation_p(read_vec, bite_vec, "spearman")
    p_tau, tau, _ = exact_permutation_p(read_vec, bite_vec, "kendall")

    best_read = max(rows, key=lambda r: r["read_acc"])
    best_bite = max(rows, key=lambda r: r["bite_acc"])

    # Paired real-vs-each-random-draw tests, where per-item predictions exist.
    paired = []
    for row in rows:
        real = row.get("real_arm")
        ctl = controls.get(row["layer"], [])
        if not real or not real.get("predictions") or not ctl:
            continue
        rp = correctness_vector(real["predictions"], correct)
        for c in ctl:
            if not c.get("predictions"):
                continue
            cp = correctness_vector(c["predictions"], correct)
            stat, pval = paired_test(rp, cp)
            paired.append(
                {
                    "layer": row["layer"],
                    "real_label": real["label"],
                    "random_label": c["label"],
                    "real_accuracy": real["accuracy"],
                    "random_accuracy": c["accuracy"],
                    "delta": real["accuracy"] - c["accuracy"],
                    "wilcoxon": {"statistic": stat, "p_value": pval},
                }
            )

    n_real_wins = sum(1 for p in paired if p["delta"] > 0)
    n_random_wins = sum(1 for p in paired if p["delta"] < 0)

    return {
        "layers": layers,
        "base_accuracy": base_acc,
        "rows": rows,
        "spearman": {"rho": rho, "p_value_exact": p_rho, "n_permutations": n_perm},
        "kendall": {"tau_b": tau, "p_value_exact": p_tau},
        "best_read_layer": best_read["layer"],
        "best_read_acc": best_read["read_acc"],
        "best_bite_layer": best_bite["layer"],
        "best_bite_acc": best_bite["bite_acc"],
        "paired_real_vs_random": paired,
        "paired_summary": {
            "n_comparisons": len(paired),
            "real_wins": n_real_wins,
            "random_wins": n_random_wins,
        },
    }


def _fmt_p(p) -> str:
    if p is None:
        return "n/a"
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if v < 1e-3:
        return f"{v:.2e}"
    return f"{v:.3f}"


def render_markdown(result: dict) -> str:
    lines = []
    lines.append("# Read-vs-bite dissociation")
    lines.append("")
    lines.append(
        "Every number below is re-derived from per-item predictions stored on disk; "
        "no model was run to produce this file."
    )
    lines.append("")
    lines.append(
        f"Base model accuracy: **{result['base_accuracy']:.3f}**. "
        f"Layers: {', '.join('L' + str(layer) for layer in result['layers'])}."
    )
    lines.append("")
    lines.append("## Per layer")
    lines.append("")
    lines.append(
        "| layer | read (5-fold item CV, 3-way) | read shuffled control | "
        "bite (act alpha=1) | bite - base | bite - read | p(bite vs base) | random draws |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        lines.append(
            f"| L{r['layer']} | {r['read_acc']:.3f} | {r['read_shuffled_acc']:.3f} | "
            f"{r['bite_acc']:.3f} | {r['bite_minus_base']:+.3f} | {r['bite_minus_read']:+.3f} | "
            f"{_fmt_p(r['bite_vs_base_p'])} | {r['n_random_draws']} |"
        )
    lines.append("")
    lines.append("## Rank correlation between the two curves")
    lines.append("")
    sp = result["spearman"]
    kt = result["kendall"]
    lines.append(
        f"- Spearman rho = **{sp['rho']:+.3f}**, exact permutation p = "
        f"**{_fmt_p(sp['p_value_exact'])}** ({sp['n_permutations']} permutations, 5! orderings)."
    )
    lines.append(
        f"- Kendall tau-b = **{kt['tau_b']:+.3f}**, exact permutation p = "
        f"**{_fmt_p(kt['p_value_exact'])}**."
    )
    lines.append(
        f"- Best read: **L{result['best_read_layer']}** ({result['best_read_acc']:.3f}). "
        f"Best bite: **L{result['best_bite_layer']}** ({result['best_bite_acc']:.3f})."
    )
    lines.append("")
    lines.append(
        "A **negative** correlation means the layers where the partner belief is most "
        "readable are the layers where steering it does least. That is the dissociation "
        "the Phase-0 matrix reported in prose."
    )
    lines.append("")

    if result["paired_real_vs_random"]:
        lines.append("## Real direction vs each random-direction draw (paired, by item)")
        lines.append("")
        lines.append("| layer | real | random | real - random | p (paired Wilcoxon) |")
        lines.append("|---|---|---|---|---|")
        for p in result["paired_real_vs_random"]:
            lines.append(
                f"| L{p['layer']} | {p['real_accuracy']:.3f} ({p['real_label']}) | "
                f"{p['random_accuracy']:.3f} ({p['random_label']}) | {p['delta']:+.3f} | "
                f"{_fmt_p(p['wilcoxon']['p_value'])} |"
            )
        s = result["paired_summary"]
        lines.append("")
        lines.append(
            f"Real direction is ahead in **{s['real_wins']}** of {s['n_comparisons']} "
            f"pairwise comparisons; the random direction is ahead in {s['random_wins']}."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", default=str(ROOT / "results" / "heldout" / "heldout_analysis.json"))
    ap.add_argument("--ablation-dir", default=str(ROOT / "results" / "ablation"))
    ap.add_argument("--controls-dir", default=str(ROOT / "results" / "controls"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "dissociation"))
    ap.add_argument("--base-accuracy", type=float, default=None)
    args = ap.parse_args(argv)

    heldout = Path(args.heldout)
    ablation_dir = Path(args.ablation_dir)
    controls_dir = Path(args.controls_dir)
    out_dir = Path(args.out_dir)

    if not heldout.exists():
        print(f"error: {heldout} not found. Run tools/analyze_heldout.py first.", file=sys.stderr)
        return 2

    base_acc = args.base_accuracy
    if base_acc is None:
        none_path = ablation_dir / "none.json"
        if none_path.exists():
            base_acc = float(_load(none_path)["accuracy"])
        else:
            print(f"error: {none_path} not found and --base-accuracy not given.", file=sys.stderr)
            return 2

    read = load_read_curve(heldout)
    bite = load_act_arms(ablation_dir, sorted(read))
    controls = load_random_controls(controls_dir)

    # The ground truth per item, needed to turn stored option indices into the
    # per-item correctness indicators the paired test actually consumes.
    heldout_meta = _load(heldout)
    n_items = int(heldout_meta.get("n_items", 0))
    seed = int(heldout_meta.get("seed", 0))
    if n_items <= 0:
        print("error: could not read n_items from the heldout analysis.", file=sys.stderr)
        return 2
    items = generate_items(n_items, seed=seed)
    correct = np.array([it.correct_index for it in items])

    result = analyse(read, bite, controls, base_acc, controls_dir, ablation_dir, correct)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "read_bite_dissociation.json").write_text(json.dumps(result, indent=2))
    (out_dir / "read_bite_dissociation.md").write_text(render_markdown(result))
    print(render_markdown(result))
    print(f"wrote {out_dir/'read_bite_dissociation.json'} and {out_dir/'read_bite_dissociation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
