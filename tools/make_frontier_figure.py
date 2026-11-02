#!/usr/bin/env python3
"""Render the belief-accuracy frontier figure.

Standard library only, for the same two reasons as ``tools/make_phase0_figures.py``:
the output is vector SVG with no font-embedding baggage, and CI carries no
plotting dependency.

The chart is the clearest statement of the phase's central finding. The x-axis
is the **achieved** out-of-fold read accuracy of the probe; the y-axis is
accuracy. Three references are drawn:

* the **diagonal** ``y = x`` -- what the probe alone achieves;
* a **horizontal** line at the base model's accuracy -- what the model achieves
  unaided;
* the **arm** -- what the intervention achieves at each probe quality.

The arm curve lies *on* the diagonal as the probe gets better, and crosses the
base-model line at the **break-even belief accuracy**. Everything below that
crossing is the intervention making things worse.

Usage::

    python tools/make_frontier_figure.py

Writes ``docs/figures/phase0b_frontier.svg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

C_ARM = "#b91c1c"
C_PROBE = "#0f766e"
C_BASE = "#4338ca"
C_CTRL = "#c2410c"
C_GRID = "#e2e8f0"
C_AXIS = "#334155"
C_TEXT = "#0f172a"
C_MUTED = "#64748b"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chart_frontier(frontier: dict) -> str:
    layer = int(frontier["headline_layer"])
    alpha = float(frontier["headline_alpha"])
    name = f"margin-a{alpha:g}"

    rows = [r for r in frontier["rows"] if r["layer"] == layer]
    rows.sort(key=lambda r: r["achieved_read_acc"])
    if not rows:
        raise SystemExit(f"no rows for layer {layer}")

    base_acc = float(frontier["base_accuracy"])
    xs = [r["achieved_read_acc"] for r in rows]
    ys = [r["arms"][name]["cv_acc"] for r in rows]
    ctl = [r["arms"][name]["cv_acc_shuffled_control"] for r in rows]

    lo = min(xs + ys + ctl + [base_acc]) - 0.05
    hi = max(xs + ys + ctl + [base_acc]) + 0.05
    lo = max(0.0, lo)
    hi = min(1.0, hi)

    W, H = 1000, 500
    ml, mr, mt, mb = 78, 252, 96, 64
    pw, ph = W - ml - mr, H - mt - mb

    def x(v: float) -> float:
        return ml + pw * ((v - lo) / (hi - lo))

    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - lo) / (hi - lo))

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="26" font-size="16" font-weight="bold" fill="{C_TEXT}">'
        "Phase 0b — the belief-margin intervention is a dial, not a gain</text>",
        f'<text x="24" y="44" font-size="11.5" fill="{C_MUTED}">'
        f"Layer {layer}, alpha = {alpha:g}. Each point is a probe of a different quality; the x-axis is its "
        "achieved out-of-fold</text>",
        f'<text x="24" y="60" font-size="11.5" fill="{C_MUTED}">'
        "read accuracy (3-way, chance 1/3). The arm curve lies ON the probe diagonal: the intervention does not</text>",
        f'<text x="24" y="76" font-size="11.5" fill="{C_MUTED}">'
        "improve on the probe that produced its margin, and it crosses the base model only just above chance.</text>",
    ]

    # grid and axes
    steps = 5
    for i in range(steps + 1):
        v = lo + (hi - lo) * i / steps
        p.append(
            f'<line x1="{ml}" y1="{y(v):.1f}" x2="{ml+pw}" y2="{y(v):.1f}" '
            f'stroke="{C_GRID}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{ml-10}" y="{y(v)+4:.1f}" font-size="11" fill="{C_MUTED}" '
            f'text-anchor="end">{v:.2f}</text>'
        )
        p.append(
            f'<text x="{x(v):.1f}" y="{mt+ph+20:.1f}" font-size="11" fill="{C_MUTED}" '
            f'text-anchor="middle">{v:.2f}</text>'
        )
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')

    # the probe diagonal, y = x
    p.append(
        f'<line x1="{x(lo):.1f}" y1="{y(lo):.1f}" x2="{x(hi):.1f}" y2="{y(hi):.1f}" '
        f'stroke="{C_PROBE}" stroke-width="2" stroke-dasharray="7 4" opacity="0.85"/>'
    )

    # the base model's accuracy
    p.append(
        f'<line x1="{ml}" y1="{y(base_acc):.1f}" x2="{ml+pw}" y2="{y(base_acc):.1f}" '
        f'stroke="{C_BASE}" stroke-width="2" stroke-dasharray="3 3"/>'
    )

    # the shuffled-label control, as a thin dotted polyline
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ctl, strict=True))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{C_CTRL}" stroke-width="1.6" opacity="0.7"/>')

    # the arm
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ys, strict=True))
    p.append(f'<polyline points="{pts}" fill="none" stroke="{C_ARM}" stroke-width="2.6"/>')

    # the points, labelled with the label-noise rate
    for r, xv, yv in zip(rows, xs, ys, strict=True):
        cx, cy = x(xv), y(yv)
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{C_ARM}" stroke="#ffffff" stroke-width="1.6"/>')
        p.append(
            f'<text x="{cx:.1f}" y="{cy-12:.1f}" font-size="10" fill="{C_MUTED}" '
            f'text-anchor="middle">noise {r["noise"]:g}</text>'
        )

    # the break-even crossing
    be = frontier.get("break_even") or {}
    crossing = be.get("fitted_crossing_read_acc")
    if crossing is not None and lo <= crossing <= hi:
        cx, cy = x(crossing), y(base_acc)
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="none" stroke="{C_BASE}" stroke-width="2.2"/>')

    # axis labels
    p.append(
        f'<text x="{ml+pw/2:.1f}" y="{H-18}" font-size="12" fill="{C_TEXT}" text-anchor="middle">'
        "achieved probe read accuracy (3-way ranking, out-of-fold)</text>"
    )
    p.append(
        f'<text x="22" y="{mt+ph/2:.1f}" font-size="12" fill="{C_TEXT}" text-anchor="middle" '
        f'transform="rotate(-90 22 {mt+ph/2:.1f})">accuracy</text>'
    )

    # legend
    lx, ly = ml + pw + 24, mt + 14
    entries = [
        (C_ARM, f"margin arm (alpha = {alpha:g})", "the intervention"),
        (C_PROBE, "probe diagonal (y = x)", "what the probe alone achieves"),
        (C_BASE, f"base model ({base_acc:.3f})", "the model unaided"),
        (C_CTRL, "shuffled-label control", "the matched control"),
    ]
    for i, (colour, label, sub) in enumerate(entries):
        yy = ly + i * 42
        p.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+26}" y2="{yy}" stroke="{colour}" stroke-width="2.6"/>')
        p.append(f'<text x="{lx+34}" y="{yy+4}" font-size="11.5" fill="{C_TEXT}">{_esc(label)}</text>')
        p.append(f'<text x="{lx+34}" y="{yy+18}" font-size="10" fill="{C_MUTED}">{_esc(sub)}</text>')

    if crossing is not None and lo <= crossing <= hi:
        yy = ly + len(entries) * 42
        p.append(
            f'<circle cx="{lx+13}" cy="{yy}" r="6.5" fill="none" stroke="{C_BASE}" stroke-width="2.2"/>'
        )
        p.append(
            f'<text x="{lx+34}" y="{yy+4}" font-size="11.5" fill="{C_TEXT}">'
            f"break-even belief accuracy {crossing:.2f}</text>"
        )
        p.append(
            f'<text x="{lx+34}" y="{yy+18}" font-size="10" fill="{C_MUTED}">'
            "where the arm crosses base</text>"
        )

    p.append("</svg>")
    return "\n".join(p) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frontier",
        default=str(ROOT / "results" / "frontier" / "belief_accuracy_frontier.json"),
    )
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "phase0b_frontier.svg"))
    args = ap.parse_args()

    frontier_path = Path(args.frontier)
    if not frontier_path.exists():
        print(f"error: {frontier_path} not found. Run tools/belief_accuracy_frontier.py first.", file=sys.stderr)
        return 2
    frontier = json.loads(frontier_path.read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(chart_frontier(frontier))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
