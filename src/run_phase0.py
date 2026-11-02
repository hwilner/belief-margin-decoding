"""CLI entry point for the Phase-0 falsification matrix.

Examples
--------
Smoke test (a handful of items, one layer, seconds)::

    python src/run_phase0.py --n-items 8 --layers 12 --act-alphas 1.0 \
        --score-alphas 1.0 --out-dir results/smoke

The full run::

    python src/run_phase0.py --n-items 200 --out-dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bmd.matrix import MatrixConfig, run_matrix  # noqa: E402


def _parse_layers(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(v) for v in value.split(",") if v.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the belief-margin-decoding Phase-0 matrix.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=_parse_layers, default=(4, 8, 12, 16, 20, 24))
    parser.add_argument("--act-alphas", type=_parse_floats, default=(1.0, 3.0))
    parser.add_argument("--score-alphas", type=_parse_floats, default=(0.5, 1.0, 2.0))
    parser.add_argument("--probe-kind", default="lda", choices=["lda", "logreg", "mlp"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args(argv)

    config = MatrixConfig(
        model_id=args.model_id,
        n_items=args.n_items,
        seed=args.seed,
        layers=tuple(args.layers),
        act_alphas=tuple(args.act_alphas),
        score_alphas=tuple(args.score_alphas),
        probe_kind=args.probe_kind,
        out_dir=args.out_dir,
        device=args.device,
        skip_existing=not args.no_skip_existing,
    )
    summary = run_matrix(config)
    best = max(
        (a for a in summary["arms"].values() if a["arm"]["mechanism"] != "none"),
        key=lambda a: a["accuracy"],
        default=None,
    )
    if best is not None:
        print(f"\nbest arm: {best['arm']['label']} accuracy={best['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
