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

    **One draw is not a distribution.** A single random direction gives a
    single number, and a single number cannot answer "is the real direction
    special?" — only "is it better than *this* random direction?". The
    Phase-0 run used exactly one draw, and the resulting paired test came
    out at p = 0.189 (not significant). That is an underpowered comparison,
    not evidence of absence. ``--random-draws N`` therefore runs N
    independent random directions, each with its own recorded seed, so the
    real arm can be compared against the *distribution* of random
    directions rather than against one sample from it.

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


def random_control_label(draw_seed: int, layer: int) -> str:
    """The label -- and filename stem -- of a random-direction control arm.

    **The layer must be part of the name.** The same direction seed is reused
    at every layer (the direction lives in the activation space, which has the
    same width at every depth), so a name without the layer is overwritten by
    the next layer's run. That is not hypothetical: the first version of the
    sweep used ``act_random_s<seed>.json`` and silently lost the layer-4 and
    layer-8 arms -- including their per-item predictions -- when the layer-12
    run overwrote them.
    """
    return f"act_random_s{int(draw_seed)}_L{int(layer)}"


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
    parser.add_argument(
        "--random-draws",
        type=int,
        default=1,
        help="number of independent random control directions (seeds are seed+977, +978, ...)",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="do not re-run the real belief direction (reuse the stored arm for pairing)",
    )
    parser.add_argument(
        "--skip-reversed",
        action="store_true",
        help="do not re-run the sign-flipped control",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-run control arms whose output file already exists",
    )
    parser.add_argument("--cache", default="results/base_cache.npz")
    parser.add_argument("--out-dir", default="results/controls")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="model dtype; must match the dtype of the base pass that built the cache",
    )
    args = parser.parse_args(argv)

    layer = int(args.layer)
    layers = [int(v) for v in args.layers]
    if layer not in layers:
        layers = sorted({*layers, layer})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.random_draws < 1:
        print("error: --random-draws must be >= 1", file=sys.stderr)
        return 2

    items = generate_items(args.n_items, seed=args.seed)
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"error: cache {cache_path} not found. Run src/run_phase0.py first.", file=sys.stderr)
        return 2
    with np.load(cache_path, allow_pickle=False) as data:
        states = np.asarray(data["states"])
        base_logprobs = np.asarray(data["base_logprobs"])
        # The dtype the base pass was recorded in. Every arm in a comparison
        # must be produced by the same numerical pipeline as the arm it is
        # compared against, so a control run in a different dtype than the
        # base pass is not a control -- it is a different experiment.
        cache_dtype = str(data["dtype"]) if "dtype" in data.files else None
    if cache_dtype and cache_dtype != "float32":
        if args.dtype and args.dtype != cache_dtype:
            print(
                f"error: the cache was recorded in {cache_dtype} but --dtype is {args.dtype}. "
                "Re-run the base pass in the same dtype, or omit --dtype to use the cache's.",
                file=sys.stderr,
            )
            return 2
        args.dtype = cache_dtype
    if args.dtype is None:
        args.dtype = "float32"
    if states.shape[2] != len(layers):
        raise SystemExit(
            f"cache was built for {states.shape[2]} layers {layers}, but the controls need "
            f"{len(layers)}. Re-run the base pass with --layers {','.join(str(ly) for ly in layers)}."
        )
    if states.shape[0] != len(items):
        # The cached activations are per (item, candidate). If the item count
        # does not match, the control would be steered with a direction fitted
        # on one item set and scored on another, and the mismatch would only
        # show up as a shape error inside fit_subspace -- or worse, as a
        # silently wrong arm if the shapes happened to line up.
        raise SystemExit(
            f"cache holds {states.shape[0]} items but --n-items is {args.n_items}. "
            "Re-run the base pass with the same --n-items, or point --cache at it."
        )
    li = layers.index(layer)
    X = states[:, :, li, :].reshape(-1, states.shape[-1]).astype(np.float64)
    y = _label_matrix(items)

    sub = fit_subspace(X, y, layer=layer, kind=args.probe_kind, seed=args.seed)
    real_dir = sub.w / np.linalg.norm(sub.w)

    def random_direction(draw: int) -> tuple[np.ndarray, int]:
        """Draw an independent random unit direction.

        The seed is ``args.seed + 977 + draw`` so that draw 0 reproduces the
        single random control that the Phase-0 run recorded, and every later
        draw is an independent direction with a recorded, replayable seed.
        """
        draw_seed = args.seed + 977 + draw
        rng = np.random.default_rng(draw_seed)
        vec = rng.normal(size=real_dir.shape)
        vec = vec / np.linalg.norm(vec)
        return vec, draw_seed

    scale = float(np.mean(np.linalg.norm(X, axis=1)))
    device = args.device or "cpu"
    model, tokenizer = load_model(args.model_id, device=device, dtype=args.dtype)

    base_pred = base_logprobs.argmax(axis=1)
    correct = np.array([it.correct_index for it in items])
    base_correct = (base_pred == correct).astype(float)

    def record(label: str, direction: np.ndarray, note: str, extra: dict | None = None) -> dict:
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
            "dtype": args.dtype,
            "wall_clock_s": round(time.time() - t0, 1),
            # Per-item predictions, so any statistic can be recomputed exactly
            # later without re-running the model.
            "predictions": [int(v) for v in pred],
        }
        if extra:
            payload.update(extra)
        (out_dir / f"{label}.json").write_text(json.dumps(payload, indent=2))
        print(
            f"[control] {label:22s} acc={payload['accuracy']:.3f} "
            f"world={payload['world_rate']:.3f} other={payload['other_rate']:.3f} p={pval:.3g}",
            flush=True,
        )
        return payload

    results = {}
    if args.skip_real:
        print("[control] skipping the real arm (--skip-real)")
    else:
        # The layer is in the filename for the same reason it is in the random
        # draws' filenames: the same label at two layers would overwrite.
        results[f"act_real_L{layer}"] = record(
            f"act_real_L{layer}",
            real_dir,
            "the fitted belief direction, re-run at the control layer so that all three "
            "arms are produced by the same code path on the same day",
            extra={"control_kind": "real"},
        )

    for draw in range(args.random_draws):
        rand_dir, draw_seed = random_direction(draw)
        # The layer MUST be in the filename. The same seed is reused at every
        # layer (the direction is drawn in the activation space, which is the
        # same at every depth), so a name without the layer would be
        # overwritten by the next layer's run and the earlier layer's arm --
        # including its per-item predictions -- would be lost. That is not
        # hypothetical: it is exactly what happened the first time this
        # sweep was run.
        label = random_control_label(draw_seed, layer)
        out_path = out_dir / f"{label}.json"
        if out_path.exists() and not args.no_skip_existing:
            try:
                existing = json.loads(out_path.read_text())
                same = (
                    int(existing.get("arm", {}).get("layer", -1)) == layer
                    and existing.get("direction_seed") == draw_seed
                    and len(existing.get("predictions", [])) == len(items)
                )
            except (ValueError, OSError):
                same = False
            if same:
                print(f"[control] {label} (cached)", flush=True)
                results[label] = existing
                continue
        results[label] = record(
            label,
            rand_dir,
            "CONTROL: uniformly random direction, matched norm, layer and alpha to the real arm",
            extra={
                "control_kind": "random",
                "draw_index": draw,
                "direction_seed": draw_seed,
                # The control is matched in norm, layer and alpha, but it is *not*
                # matched in alignment with the belief direction -- that is the
                # point. Recording the cosine makes the degree of de-alignment
                # auditable instead of assumed.
                "cos_to_real_direction": float(np.dot(rand_dir, real_dir)),
            },
        )

    if args.skip_reversed:
        print("[control] skipping the reversed arm (--skip-reversed)")
    else:
        results[f"act_reversed_L{layer}"] = record(
            f"act_reversed_L{layer}",
            -real_dir,
            "CONTROL: negation of the fitted belief direction, matched norm, layer and alpha",
            extra={"control_kind": "reversed"},
        )

    random_accs = [
        v["accuracy"] for k, v in results.items() if v.get("control_kind") == "random"
    ]
    summary = {
        "config": {
            "model_id": args.model_id,
            "n_items": args.n_items,
            "seed": args.seed,
            "layer": layer,
            "alpha": args.alpha,
            "probe_kind": args.probe_kind,
            "layers": layers,
            "dtype": args.dtype,
            "random_draws": args.random_draws,
            "skip_real": bool(args.skip_real),
            "skip_reversed": bool(args.skip_reversed),
        },
        "probe": {
            "train_acc": sub.train_acc,
            "heldout_acc": sub.heldout_acc,
            "w_norm": float(np.linalg.norm(sub.w)),
        },
        "scale": scale,
        "base_accuracy": tom_accuracy(base_pred, items),
        # The distribution of random-direction accuracies, which is the object
        # the real arm has to beat. A single draw is not a distribution.
        "random_accuracy_summary": (
            {
                "n": len(random_accs),
                "mean": float(np.mean(random_accs)),
                "std": float(np.std(random_accs, ddof=1)) if len(random_accs) > 1 else 0.0,
                "min": float(np.min(random_accs)),
                "max": float(np.max(random_accs)),
            }
            if random_accs
            else None
        ),
        "arms": results,
    }
    (out_dir / "controls_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/'controls_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
