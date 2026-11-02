#!/usr/bin/env python3
"""Recompute the headline statistics from the stored per-item predictions.

Every arm JSON written by the matrix runner and by the control runner stores
the per-item predictions. That makes the statistics **auditable and
recomputable**: this script re-derives the accuracy, the bootstrap CI and the
Wilcoxon p-value from the stored predictions alone, with no model and no
cached activations, and writes the recomputed values into a side file.

It also reports, where SciPy is importable, the **exact** signed-rank p-value
next to the tie-corrected normal approximation that
``bmd.evaluate.paired_test`` falls back to when SciPy is missing. The two
agree to within a few percent on a sample of this size, and the CI suite
asserts that they agree (see ``tests/test_evaluate.py``).

Usage::

    python tools/recompute_stats.py --results-dir results
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
from bmd.evaluate import _wilcoxon_normal_approx, bootstrap_ci  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="where to write the recomputed table")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    items = generate_items(args.n_items, seed=args.seed)
    correct = np.array([it.correct_index for it in items])

    base_path = results_dir / "ablation" / "none.json"
    if not base_path.exists():
        print("error: results/ablation/none.json not found", file=sys.stderr)
        return 2
    base_payload = json.loads(base_path.read_text())
    base_preds = base_payload.get("predictions")
    if base_preds is None:
        print("error: the base arm does not store predictions; re-run the matrix", file=sys.stderr)
        return 2
    base_correct = (np.array(base_preds) == correct).astype(float)

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon  # type: ignore

        have_scipy = True
    except Exception:
        have_scipy = False

    rows = []
    arm_files = sorted((results_dir / "ablation").glob("*.json"))
    control_files = sorted((results_dir / "controls").glob("*.json")) if (results_dir / "controls").exists() else []
    for path in arm_files + control_files:
        payload = json.loads(path.read_text())
        preds = payload.get("predictions")
        if preds is None:
            continue
        pred = np.array(preds)
        per_item = (pred == correct).astype(float)
        lo, hi = bootstrap_ci(per_item, seed=args.seed)
        stat_approx, p_approx = _wilcoxon_normal_approx(per_item, base_correct)
        p_exact = None
        if have_scipy and not np.allclose(per_item, base_correct):
            p_exact = float(scipy_wilcoxon(per_item, base_correct, zero_method="wilcox", alternative="two-sided").pvalue)
        rows.append(
            {
                "arm": payload["arm"]["label"],
                "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "accuracy_recomputed": float(per_item.mean()),
                "accuracy_reported": payload["accuracy"],
                "ci_recomputed": [lo, hi],
                "p_normal_approx": p_approx,
                "p_exact_scipy": p_exact,
            }
        )

    if not rows:
        print("no arms with stored predictions found", file=sys.stderr)
        return 2

    rows.sort(key=lambda r: -r["accuracy_recomputed"])
    header = (
        "| arm | accuracy (recomputed) | accuracy (reported) | 95% CI | "
        "p (normal approx) | p (exact, SciPy) |"
    )
    lines = [header, "|---|---|---|---|---|---|"]
    for r in rows:
        exact = "—" if r["p_exact_scipy"] is None else f"{r['p_exact_scipy']:.4g}"
        lines.append(
            f"| `{r['arm']}` | {r['accuracy_recomputed']:.3f} | {r['accuracy_reported']:.3f} | "
            f"[{r['ci_recomputed'][0]:.3f}, {r['ci_recomputed'][1]:.3f}] | "
            f"{r['p_normal_approx']:.4g} | {exact} |"
        )
    table = "\n".join(lines) + "\n"

    drift = [r for r in rows if abs(r["accuracy_recomputed"] - r["accuracy_reported"]) > 1e-9]
    print(table)
    if drift:
        print("WARNING: recomputed accuracy differs from the reported value for:")
        for r in drift:
            print(f"  {r['arm']}: reported {r['accuracy_reported']:.6f} "
                  f"recomputed {r['accuracy_recomputed']:.6f}")
    else:
        print("All recomputed accuracies match the reported values exactly.")

    out = Path(args.out) if args.out else results_dir / "recomputed_stats.md"
    out.write_text(table)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
