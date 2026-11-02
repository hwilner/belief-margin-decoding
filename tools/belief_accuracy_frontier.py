#!/usr/bin/env python3
"""The belief-accuracy x steering-gain frontier.

Phase 0 answered: *does* a belief-margin intervention change the model's
answers? It did not -- and the honest reading was that the score-level
intervention makes the **probe** pick the answer rather than making the
**model** use the belief, because the arm's out-of-fold accuracy converged
onto the probe's own read-out accuracy.

That convergence implies a sharper question, and it is the question the
Roadmap calls Phase 2:

    **How accurate must a belief estimate be before intervening on it is
    net-positive?**

The answer matters for deployment. If the break-even point sits above what
any estimator can reach, the mechanism is dead and should be reported as
dead. If it sits at chance, the mechanism is already useless. If it sits
somewhere in between, it tells you what a belief estimator has to achieve.

Method
------
The score-level ``margin`` mechanism operates entirely on **cached**
quantities: the base log-probabilities and the probe's margin. Neither
requires a forward pass. So the whole frontier can be computed from
``results/base_cache.npz`` with **no model loaded at all**.

A probe of a given quality is simulated by corrupting its **training
labels**: for each item, with probability ``noise`` the belief-consistent
candidate is relabelled as one of the two incorrect candidates. The error
pattern is fixed per noise level (one seeded draw, reused across folds), so
the folds differ only in which items the probe was trained on -- exactly
the structure of a real estimator with a fixed error profile.

Everything else follows the Phase-0 protocol unchanged:

* the split is at the level of the **item**, 5-fold cross-validated;
* the probe is fitted on the training items' (corrupted) labels only;
* every scored quantity is **out-of-fold**;
* the metric is **3-way ranking accuracy** (chance 1/3), never the pair-wise
  accuracy whose trivial baseline is 2/3;
* every intervention arm ships its **matched shuffled-label control**;
* statistics are the Wilcoxon signed-rank test and the item-level bootstrap.

The x-axis of the frontier is the **achieved** out-of-fold read accuracy of
the degraded probe, not the nominal noise rate. Those are different numbers,
and only the achieved one is what a deployed estimator would deliver.

Usage::

    python tools/belief_accuracy_frontier.py
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

# The alpha used for the headline frontier. It is the largest value in the
# Phase-0 sweep and is fixed here *before* the frontier is computed, so the
# frontier is not a search over alpha dressed up as a curve.
HEADLINE_ALPHA = 4.0


def corrupt_labels(y: np.ndarray, correct: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Relabel the positive class with probability ``noise``.

    ``y`` holds one 1 per item (the belief-consistent candidate) and two 0s.
    A corrupted item keeps exactly one positive label, but that positive is
    one of the two *incorrect* candidates. The number of positives is
    unchanged, so the probe sees the same class balance at every noise
    level; only the label's *meaning* degrades.
    """
    rng = np.random.default_rng(seed)
    out = y.copy()
    n_items = len(correct)
    for i in range(n_items):
        if rng.random() < noise:
            negatives = [c for c in range(3) if c != correct[i]]
            out[i * 3 : i * 3 + 3] = 0
            out[i * 3 + negatives[rng.integers(0, 2)]] = 1
    return out


def _fold_indices(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """The item-level folds, identical to ``tools/analyze_heldout.py``.

    The folds MUST be the same ones the Phase-0 held-out analysis used, or
    the noise-0 row of this frontier is not comparable with the published
    read curve (it would be a different random split, and the two numbers
    would differ for no reason a reader could see).

    ``analyze_heldout.py`` draws its 80/20 split from the same generator
    *before* it draws the folds, so the second draw is what defines the
    folds. That first draw is reproduced here for the same reason: skipping
    it would shift every fold.
    """
    rng = np.random.default_rng(seed + 4242)
    _ = rng.permutation(n)  # the 80/20 split's draw, consumed in analyze_heldout.py
    return [np.sort(f) for f in np.array_split(rng.permutation(n), n_folds)]


def run_frontier(
    states: np.ndarray,
    base_logprobs: np.ndarray,
    correct: np.ndarray,
    layer_numbers: list[int],
    noise_levels: list[float],
    alphas: list[float],
    n_folds: int,
    probe_kind: str,
    seed: int,
) -> list[dict]:
    """Compute the frontier for every layer and every noise level."""
    n = len(correct)
    dim = states.shape[3]
    folds = _fold_indices(n, n_folds, seed)

    y_true = np.zeros(n * 3, dtype=int)
    for i in range(n):
        y_true[i * 3 + correct[i]] = 1

    base_correct = (base_logprobs.argmax(axis=1) == correct).astype(float)

    rows: list[dict] = []
    for li, layer in enumerate(layer_numbers):
        X = states[:, :, li, :].reshape(-1, dim).astype(np.float64)

        for noise in noise_levels:
            # One fixed corruption per noise level, shared by every fold.
            y_noisy = corrupt_labels(y_true, correct, noise, seed=seed + 991 + int(noise * 1000))

            oof_read = np.zeros(n, dtype=bool)
            oof_read_shuf = np.zeros(n, dtype=bool)
            oof_probe_pred = np.zeros(n, dtype=int)
            oof_arm_pred = {f"margin-a{a:g}": np.zeros(n, dtype=int) for a in alphas}
            oof_arm_shuf = {f"margin-a{a:g}": np.zeros(n, dtype=bool) for a in alphas}

            for fold in folds:
                # The training rows are assembled in the SAME order as
                # `tools/analyze_heldout.py` assembles them: the other folds
                # concatenated, not the items in ascending index order.
                #
                # This matters, and not for a trivial reason.
                # `bmd.subspace.fit_subspace` splits its input with a
                # deterministic *tail* split: the last 10 % of the rows it is
                # given are held out and the probe is fitted on the rest. So the
                # rows that actually train the probe depend on the ORDER of
                # the caller's array. Pass the same items in a different order
                # and you get a different probe -- and a different read curve.
                # Matching the published order is what makes the noise-0 row of
                # this frontier comparable with `results/heldout/`.
                held = np.concatenate([f for f in folds if not np.array_equal(f, fold)])
                train_p = (held[:, None] * 3 + np.arange(3)).ravel()

                sub = fit_subspace(
                    X[train_p], y_noisy[train_p], layer=layer, kind=probe_kind, seed=seed
                )
                sub_shuf = fit_subspace(
                    X[train_p],
                    np.random.default_rng(seed + 7).permutation(y_noisy[train_p]),
                    layer=layer,
                    kind=probe_kind,
                    seed=seed,
                )
                marg = normalize_margins(sub.margin(X)).reshape(n, 3)
                marg_shuf = normalize_margins(sub_shuf.margin(X)).reshape(n, 3)

                oof_read[fold] = marg[fold].argmax(axis=1) == correct[fold]
                oof_read_shuf[fold] = marg_shuf[fold].argmax(axis=1) == correct[fold]
                oof_probe_pred[fold] = marg[fold].argmax(axis=1)
                for a in alphas:
                    oof_arm_pred[f"margin-a{a:g}"][fold] = (
                        base_logprobs[fold] + a * marg[fold]
                    ).argmax(axis=1)
                    oof_arm_shuf[f"margin-a{a:g}"][fold] = (
                        (base_logprobs[fold] + a * marg_shuf[fold]).argmax(axis=1) == correct[fold]
                    )

            read = oof_read.astype(float)
            lo, hi = bootstrap_ci(read, seed=seed)
            stat, pval = paired_test(read, base_correct)

            # How much of the base model's own preference survives the
            # intervention? The arm is `base_logprobs + alpha * margin`; when
            # the margin term dominates, the arm's prediction IS the probe's
            # prediction, and the "intervention" has become the probe.
            base_spread = np.ptp(base_logprobs, axis=1)

            row = {
                "layer": layer,
                "noise": noise,
                "achieved_read_acc": float(read.mean()),
                "achieved_read_ci95": [lo, hi],
                "read_shuffled_acc": float(oof_read_shuf.mean()),
                "read_vs_base_p": pval,
                "base_acc": float(base_correct.mean()),
                "arms": {},
            }
            for a in alphas:
                name = f"margin-a{a:g}"
                per_item = (oof_arm_pred[name] == correct).astype(float)
                per_item_ctl = oof_arm_shuf[name].astype(float)
                s_ctl, p_ctl = paired_test(per_item, per_item_ctl)
                s_base, p_base = paired_test(per_item, base_correct)
                # The intervention's own contribution: the arm against the
                # probe that produced its margin. The arm's accuracy is the
                # probe's accuracy plus whatever the intervention adds, and
                # this is the quantity that says whether it adds anything.
                s_read, p_read = paired_test(per_item, read)
                lo_a, hi_a = bootstrap_ci(per_item, seed=seed)
                row["arms"][name] = {
                    "cv_acc": float(per_item.mean()),
                    "cv_ci95": [lo_a, hi_a],
                    "cv_acc_shuffled_control": float(per_item_ctl.mean()),
                    "margin_minus_control": float(per_item.mean() - per_item_ctl.mean()),
                    "margin_vs_control_p": p_ctl,
                    "margin_vs_base_p": p_base,
                    "margin_minus_base": float(per_item.mean() - base_correct.mean()),
                    "margin_minus_read": float(per_item.mean() - read.mean()),
                    "margin_vs_read_p": p_read,
                    "margin_vs_read_statistic": s_read,
                    # The share of items where the arm picks exactly what the
                    # probe picks: the intervention has become the probe.
                    "arm_equals_probe_rate": float(
                        (oof_arm_pred[name] == oof_probe_pred).mean()
                    ),
                    "base_logprob_spread_median": float(np.median(base_spread)),
                }
            rows.append(row)
            head = row["arms"][f"margin-a{HEADLINE_ALPHA:g}"]
            print(
                f"L{layer:<3d} noise={noise:<5g} read={row['achieved_read_acc']:.3f} "
                f"[{lo:.3f},{hi:.3f}] | margin(a={HEADLINE_ALPHA:g})={head['cv_acc']:.3f} "
                f"ctl={head['cv_acc_shuffled_control']:.3f} "
                f"base={row['base_acc']:.3f} p_ctl={head['margin_vs_control_p']:.3g}",
                flush=True,
            )
    return rows


def break_even(rows: list[dict], layer: int, alpha: float, base_acc: float) -> dict:
    """Where the intervention stops being net-positive.

    The break-even is read in terms of the **achieved read accuracy**, which
    is the x-axis of the frontier, not the nominal noise rate.

    The achieved read accuracy is *not* monotone in the noise rate: each noise
    level uses one corruption draw, so a noisier level can happen to produce a
    slightly more accurate probe. The break-even is therefore reported as a
    **bracket** -- the lowest read accuracy at which the arm still beat the
    base, and the highest read accuracy at which it did not -- plus a
    least-squares crossing as a single summary number.

    Both are descriptive of the measured points. Neither is a claim about
    unmeasured read accuracies.
    """
    name = f"margin-a{alpha:g}"
    layer_rows = sorted(
        (r for r in rows if r["layer"] == layer), key=lambda r: r["achieved_read_acc"]
    )
    if not layer_rows:
        return {"layer": layer, "alpha": alpha, "n_levels": 0}

    beats = [r for r in layer_rows if r["arms"][name]["margin_minus_base"] > 0]
    fails = [r for r in layer_rows if r["arms"][name]["margin_minus_base"] <= 0]

    # Least-squares crossing of the arm-accuracy line with the base line.
    x = np.array([r["achieved_read_acc"] for r in layer_rows])
    y = np.array([r["arms"][name]["cv_acc"] for r in layer_rows])
    crossing = None
    if len(x) >= 2 and np.ptp(x) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        if abs(slope) > 1e-9:
            crossing = float((base_acc - intercept) / slope)

    return {
        "layer": layer,
        "alpha": alpha,
        "n_levels": len(layer_rows),
        "n_levels_beating_base": len(beats),
        "lowest_read_acc_that_beats_base": beats[0]["achieved_read_acc"] if beats else None,
        "highest_read_acc_that_fails_to_beat_base": (
            max(r["achieved_read_acc"] for r in fails) if fails else None
        ),
        "fitted_crossing_read_acc": crossing,
        "note": (
            "achieved read accuracy is not monotone in the noise rate (one corruption "
            "draw per level), so the break-even is a bracket, not a single measured "
            "threshold"
        ),
    }


def render_markdown(result: dict) -> str:
    lines = []
    lines.append("# Belief-accuracy x steering-gain frontier")
    lines.append("")
    lines.append(
        "Computed from the cached base pass: **no model was loaded**. The probe's quality is "
        "simulated by corrupting its training labels, and every quantity is out-of-fold under "
        "item-level 5-fold cross-validation."
    )
    lines.append("")
    lines.append(f"Base model accuracy: **{result['base_accuracy']:.3f}**.")
    lines.append("")
    lines.append("## The frontier at the best-reading layer")
    lines.append("")
    lines.append(
        f"Layer **{result['headline_layer']}** (the layer with the best Phase-0 belief readout), "
        f"alpha = {result['headline_alpha']:g} (fixed before the sweep)."
    )
    lines.append("")
    lines.append(
        "| label noise | achieved read acc (3-way) | 95% CI | margin arm | shuffled control | "
        "arm - base | p vs control | p vs base |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in result["headline_rows"]:
        a = r["arms"][f"margin-a{result['headline_alpha']:g}"]
        lines.append(
            f"| {r['noise']:g} | {r['achieved_read_acc']:.3f} | "
            f"[{r['achieved_read_ci95'][0]:.3f}, {r['achieved_read_ci95'][1]:.3f}] | "
            f"{a['cv_acc']:.3f} | {a['cv_acc_shuffled_control']:.3f} | "
            f"{a['margin_minus_base']:+.3f} | {a['margin_vs_control_p']:.3g} | "
            f"{a['margin_vs_base_p']:.3g} |"
        )
    lines.append("")
    be = result["break_even"]
    if be.get("lowest_read_acc_that_beats_base") is None:
        lines.append(
            "**Break-even:** the intervention never beats the base model at this layer and alpha, "
            "at any belief-accuracy level tested."
        )
    else:
        bracket = ""
        if be.get("highest_read_acc_that_fails_to_beat_base") is not None:
            bracket = (
                f", and fails from "
                f"{be['highest_read_acc_that_fails_to_beat_base']:.3f} upwards"
            )
        fitted = ""
        if be.get("fitted_crossing_read_acc") is not None:
            fitted = (
                f" The least-squares crossing of the fitted arm-accuracy line with the base "
                f"model is at a read accuracy of **{be['fitted_crossing_read_acc']:.3f}**."
            )
        lines.append(
            f"**Break-even:** the intervention still beats the base model down to an achieved "
            f"read accuracy of **{be['lowest_read_acc_that_beats_base']:.3f}** "
            f"({be['n_levels_beating_base']} of {be['n_levels']} noise levels){bracket}.{fitted}"
        )
    lines.append("")
    lines.append("### What the intervention adds over the probe that produced it")
    lines.append("")
    lines.append(
        "The arm's accuracy is the probe's accuracy plus whatever the intervention contributes. "
        "This column is that contribution."
    )
    lines.append("")
    lines.append("| label noise | achieved read acc | margin arm | arm - probe | p (paired) | arm picks what the probe picks |")
    lines.append("|---|---|---|---|---|---|")
    for r in result["headline_rows"]:
        a = r["arms"][f"margin-a{result['headline_alpha']:g}"]
        lines.append(
            f"| {r['noise']:g} | {r['achieved_read_acc']:.3f} | {a['cv_acc']:.3f} | "
            f"{a['margin_minus_read']:+.3f} | {a['margin_vs_read_p']:.3g} | "
            f"{a['arm_equals_probe_rate']:.3f} |"
        )
    gaps = [
        r["arms"][f"margin-a{result['headline_alpha']:g}"]["margin_minus_read"]
        for r in result["headline_rows"]
    ]
    if gaps:
        lines.append("")
        lines.append(
            f"Mean contribution over the noise levels: **{float(np.mean(gaps)):+.3f}** "
            f"(min {min(gaps):+.3f}, max {max(gaps):+.3f})."
        )
    lines.append("")

    # The alpha grid at the cleanest probe. This is the mechanism: as alpha
    # grows, the margin term overwhelms the base log-probabilities, the arm's
    # prediction becomes the probe's prediction, and the "intervention"
    # degenerates into the probe.
    clean_rows = [r for r in result["headline_rows"] if r["noise"] == min(result["noise_levels"])]
    if clean_rows:
        r0 = clean_rows[0]
        lines.append("### The alpha grid, at the cleanest probe")
        lines.append("")
        lines.append(
            f"Layer L{r0['layer']}, label noise {r0['noise']:g}, "
            f"probe read accuracy {r0['achieved_read_acc']:.3f}, base model {r0['base_acc']:.3f}."
        )
        lines.append("")
        lines.append(
            "| alpha | arm accuracy | arm - probe | arm picks what the probe picks | p vs base |"
        )
        lines.append("|---|---|---|---|---|")
        for a in result["alphas"]:
            name = f"margin-a{a:g}"
            if name not in r0["arms"]:
                continue
            arm = r0["arms"][name]
            lines.append(
                f"| {a:g} | {arm['cv_acc']:.3f} | {arm['margin_minus_read']:+.3f} | "
                f"{arm['arm_equals_probe_rate']:.3f} | {arm['margin_vs_base_p']:.3g} |"
            )
        lines.append("")
        lines.append(
            "The agreement column is the mechanism. At large alpha the margin term dominates the "
            "base log-probabilities, the arm's prediction *is* the probe's prediction, and the "
            "arm's accuracy is the probe's accuracy. The intervention is then not changing the "
            "model's answer; it is substituting the probe's answer for it."
        )
    lines.append("")

    # The global check: across every layer, every noise level and every alpha,
    # does the arm EVER beat the probe that produced its margin?
    #
    # The answer depends on how good the probe is, and that is the point. At a
    # layer where the probe is strong, the arm slides from the base model's
    # ranking to the probe's and never passes it: the margin term simply
    # replaces the model's contribution. At a layer where the probe is *weak*,
    # the base model's ranking still carries information, and base + probe can
    # beat probe alone. So the check is reported split by probe strength, and
    # with a multiple-comparison correction, because 180 comparisons that are
    # each nominally significant at 0.05 are not 180 findings.
    beats = []
    for r in result["rows"]:
        for name, arm in r["arms"].items():
            if arm["margin_minus_read"] > 0:
                beats.append((r, name, arm))
    n_configs = sum(len(r["arms"]) for r in result["rows"])
    strong_cut = 0.5
    strong = [b for b in beats if b[0]["achieved_read_acc"] >= strong_cut]
    weak = [b for b in beats if b[0]["achieved_read_acc"] < strong_cut]
    n_sig = sum(1 for b in beats if b[2]["margin_vs_read_p"] < 0.05)
    bonf = 0.05 / max(n_configs, 1)
    n_bonf = sum(1 for b in beats if b[2]["margin_vs_read_p"] < bonf)
    # The Wilcoxon test is two-sided, so a small p-value on a *negative*
    # difference means the arm is significantly WORSE than the probe. Both
    # directions are counted, because only the direction is the claim.
    all_arms = [(r, name, arm) for r in result["rows"] for name, arm in r["arms"].items()]
    n_bonf_worse = sum(
        1 for _, _, arm in all_arms
        if arm["margin_minus_read"] < 0 and arm["margin_vs_read_p"] < bonf
    )

    lines.append("### Does the intervention ever beat its own probe?")
    lines.append("")
    lines.append(
        f"Across all {n_configs} configurations (every layer x every label-noise level x every "
        f"alpha), the arm beats the probe that produced its margin in **{len(beats)}** of them, "
        f"and the beat is nominally significant (paired p < 0.05) in **{n_sig}**."
    )
    lines.append("")
    lines.append(
        f"- **Where the probe is strong** (achieved read accuracy >= {strong_cut:g}): "
        f"{len(strong)} of the beats."
    )
    lines.append(
        f"- **Where the probe is weak** (achieved read accuracy < {strong_cut:g}): "
        f"{len(weak)} of the beats."
    )
    lines.append(
        f"- Surviving a Bonferroni correction over all {n_configs} comparisons "
        f"(p < {bonf:.2e}): **{n_bonf}** beat the probe; in the other direction, "
        f"**{n_bonf_worse}** are significantly **worse** than the probe."
    )
    lines.append("")
    lines.append(
        "The paired Wilcoxon test is two-sided, so a small p-value on a negative difference "
        "means the arm is significantly *worse* than the probe. Only the sign carries the claim."
    )
    lines.append("")
    if strong:
        lines.append(
            f"At a strong-probe layer the arm slides from the base model's ranking to the probe's "
            f"and does not pass it: the margin term replaces the model's contribution instead of "
            f"adding to it. The largest beat at a strong-probe layer is "
            f"{max(b[2]['margin_minus_read'] for b in strong):+.3f}."
        )
    else:
        lines.append(
            "At a strong-probe layer the arm slides from the base model's ranking to the probe's "
            "and never passes it: the margin term replaces the model's contribution instead of "
            "adding to it."
        )
    if weak:
        lines.append("")
        lines.append(
            f"Where the probe is weak, the base model's ranking still carries information, and "
            f"base + probe can beat probe alone. The largest such beat is "
            f"{max(b[2]['margin_minus_read'] for b in weak):+.3f} "
            f"(L{weak[0][0]['layer']}, noise {weak[0][0]['noise']:g}, {weak[0][1]}), where the "
            f"probe reads {weak[0][0]['achieved_read_acc']:.3f} and the base model reads "
            f"{weak[0][0]['base_acc']:.3f} -- an arm that beats the probe but gains little over "
            "the base model it was supposed to improve on."
        )
    lines.append("")
    lines.append("## Every layer")
    lines.append("")
    lines.append(
        "| layer | noise | achieved read acc | margin arm (alpha=4) | shuffled control | arm - base |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in result["rows"]:
        a = r["arms"][f"margin-a{result['headline_alpha']:g}"]
        lines.append(
            f"| L{r['layer']} | {r['noise']:g} | {r['achieved_read_acc']:.3f} | "
            f"{a['cv_acc']:.3f} | {a['cv_acc_shuffled_control']:.3f} | "
            f"{a['margin_minus_base']:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(ROOT / "results" / "base_cache.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "frontier"))
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-kind", default="lda")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--alphas", default="0.25,0.5,1,2,4")
    ap.add_argument(
        "--noise",
        default="0,0.15,0.3,0.45,0.6,0.75",
        help="comma-separated label-noise rates; the probe's TRAINING labels are corrupted",
    )
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="re-render the markdown from the saved JSON without recomputing anything",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    json_path = out_dir / "belief_accuracy_frontier.json"

    if args.render_only:
        if not json_path.exists():
            print(f"error: {json_path} not found; run without --render-only first.", file=sys.stderr)
            return 2
        result = json.loads(json_path.read_text())
        (out_dir / "belief_accuracy_frontier.md").write_text(render_markdown(result))
        print(render_markdown(result))
        print(f"wrote {out_dir/'belief_accuracy_frontier.md'}")
        return 0

    cache = Path(args.cache)
    if not cache.exists():
        print(f"error: {cache} not found. Run src/run_phase0.py first.", file=sys.stderr)
        return 2
    with np.load(cache, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        base_logprobs = np.asarray(data["base_logprobs"], dtype=np.float64)
        key = str(data["key"])

    items = generate_items(args.n_items, seed=args.seed)
    correct = np.array([it.correct_index for it in items])
    layer_numbers = [int(v) for v in args.layers.split(",")]
    if len(layer_numbers) != states.shape[2]:
        print(
            f"error: the cache holds {states.shape[2]} layer slices but --layers lists "
            f"{len(layer_numbers)}",
            file=sys.stderr,
        )
        return 2

    alphas = [float(a) for a in args.alphas.split(",")]
    noise_levels = [float(v) for v in args.noise.split(",")]

    rows = run_frontier(
        states,
        base_logprobs,
        correct,
        layer_numbers,
        noise_levels,
        alphas,
        args.n_folds,
        args.probe_kind,
        args.seed,
    )

    # The headline layer is the one with the best Phase-0 belief readout. That
    # was fixed and published in the Phase-0 read curve, so choosing it here is
    # not a search over the frontier.
    clean = [r for r in rows if r["noise"] == 0.0]
    headline_layer = max(clean, key=lambda r: r["achieved_read_acc"])["layer"] if clean else layer_numbers[0]

    base_accuracy = float((base_logprobs.argmax(axis=1) == correct).mean())

    result = {
        "cache_key": key,
        "n_items": args.n_items,
        "seed": args.seed,
        "probe_kind": args.probe_kind,
        "n_folds": args.n_folds,
        "split": f"item-level {args.n_folds}-fold cross-validation, out-of-fold",
        "base_accuracy": base_accuracy,
        "headline_layer": headline_layer,
        "headline_alpha": HEADLINE_ALPHA,
        "noise_levels": noise_levels,
        "alphas": alphas,
        "rows": rows,
        "break_even": break_even(rows, headline_layer, HEADLINE_ALPHA, base_accuracy),
    }
    result["headline_rows"] = [r for r in rows if r["layer"] == headline_layer]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "belief_accuracy_frontier.json").write_text(json.dumps(result, indent=2))
    (out_dir / "belief_accuracy_frontier.md").write_text(render_markdown(result))
    print()
    print(render_markdown(result))
    print(f"wrote {out_dir/'belief_accuracy_frontier.json'} and {out_dir/'belief_accuracy_frontier.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
