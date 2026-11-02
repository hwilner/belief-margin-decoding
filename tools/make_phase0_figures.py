#!/usr/bin/env python3
"""Render the Phase-0 figures and tables from ``results/phase0_summary.json``.

Standard library only: no matplotlib. Two reasons.

1. The figures are vector SVG, so they stay crisp in the README and in the PDF
   build, and they carry no font-embedding baggage.
2. The CI runner has a small disk and a small memory budget. A plotting
   dependency is not worth it for three charts.

Usage::

    python tools/make_phase0_figures.py --summary results/phase0_summary.json

Writes ``docs/figures/phase0_*.svg`` and ``results/tables.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Palette chosen to stay distinguishable in greyscale print and for the most
# common forms of colour-vision deficiency (no red/green-only pairings).
C_NONE = "#64748b"
C_ORACLE = "#b91c1c"
C_READ = "#0f766e"
C_BITE = "#4338ca"
C_CTRL = "#c2410c"
C_GRID = "#e2e8f0"
C_AXIS = "#334155"
C_TEXT = "#0f172a"
C_MUTED = "#64748b"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_alpha(a: float) -> str:
    return f"{a:g}"


def _mechanisms(summary: dict) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for label, payload in summary["arms"].items():
        mech = payload["arm"]["mechanism"]
        out.setdefault(mech, []).append((label, payload))
    for v in out.values():
        v.sort(key=lambda t: (t[1]["arm"].get("layer") or 0, t[1]["arm"].get("alpha") or 0.0))
    return out


def _layers(summary: dict) -> list[int]:
    return sorted(int(k) for k in summary["subspaces"])


# --------------------------------------------------------------------------- charts


def chart_read_bite(summary: dict) -> str:
    """The two curves: probe read-out accuracy per layer vs. best biting arm."""
    layers = _layers(summary)
    subs = summary["subspaces"]
    arms = summary["arms"]
    read = [(int(l), subs[str(l)]["heldout_acc"]) for l in layers]
    read_train = [(int(l), subs[str(l)]["train_acc"]) for l in layers]

    # best "bite" per layer: the margin arm with the highest accuracy at alpha=1
    bite: list[tuple[int, float, str]] = []
    for l in layers:
        best = None
        for label, payload in arms.items():
            mech = payload["arm"]["mechanism"]
            if mech not in {"margin", "act", "logit"}:
                continue
            if payload["arm"].get("layer") != l:
                continue
            a = payload["arm"].get("alpha") or 0.0
            if mech == "margin" and a != 1.0:
                continue
            if best is None or payload["accuracy"] > best[1]:
                best = (label, payload["accuracy"])
        if best is not None:
            bite.append((l, best[1], best[0]))

    base_acc = arms["none"]["accuracy"]
    oracle_acc = arms["prompt_oracle"]["accuracy"]

    W, H = 880, 470
    ml, mr, mt, mb = 70, 250, 50, 60
    pw, ph = W - ml - mr, H - mt - mb

    def x(layer: int) -> float:
        if len(layers) == 1:
            return ml + pw / 2
        return ml + pw * (layers.index(layer) / (len(layers) - 1))

    def y(v: float) -> float:
        return mt + ph * (1.0 - v)

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="26" font-size="16" font-weight="bold" fill="{C_TEXT}">'
        "Phase 0 — the read curve and the bite curve</text>",
        f'<text x="24" y="44" font-size="11.5" fill="{C_MUTED}">'
        "Read = held-out accuracy of the belief probe. Bite = accuracy of the best scoring arm at that layer. "
        "If the curves coincide, the probe is measuring what the intervention moves.</text>",
    ]
    # grid + axes
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = frac
        p.append(f'<line x1="{ml}" y1="{y(v):.1f}" x2="{ml+pw}" y2="{y(v):.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        p.append(
            f'<text x="{ml-10}" y="{y(v)+4:.1f}" font-size="11" fill="{C_MUTED}" text-anchor="end">{v:.2f}</text>'
        )
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    for l in layers:
        p.append(
            f'<text x="{x(l):.1f}" y="{mt+ph+20}" font-size="11" fill="{C_MUTED}" text-anchor="middle">L{l}</text>'
        )
    p.append(
        f'<text x="{ml+pw/2:.1f}" y="{mt+ph+42}" font-size="11.5" fill="{C_AXIS}" text-anchor="middle">'
        "decoder layer</text>"
    )
    p.append(
        f'<text x="18" y="{mt+ph/2:.1f}" font-size="11.5" fill="{C_AXIS}" text-anchor="middle" '
        f'transform="rotate(-90 18 {mt+ph/2:.1f})">accuracy</text>'
    )
    # reference lines
    for val, col, name in ((base_acc, C_NONE, "base"), (oracle_acc, C_ORACLE, "ORACLE")):
        p.append(
            f'<line x1="{ml}" y1="{y(val):.1f}" x2="{ml+pw}" y2="{y(val):.1f}" stroke="{col}" '
            f'stroke-width="1.6" stroke-dasharray="7 4"/>'
        )
        p.append(
            f'<text x="{ml+pw+6}" y="{y(val)+4:.1f}" font-size="11" fill="{col}">{name} {val:.3f}</text>'
        )

    def polyline(pts, col, dash=None):
        d = " ".join(f"{x(l):.1f},{y(v):.1f}" for l, v in pts)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2.4"{extra}/>')
        for l, v in pts:
            p.append(f'<circle cx="{x(l):.1f}" cy="{y(v):.1f}" r="4" fill="#ffffff" stroke="{col}" stroke-width="2.2"/>')

    polyline(read_train, C_READ, "5 4")
    polyline(read, C_READ)
    if bite:
        polyline([(l, v) for l, v, _ in bite], C_BITE)

    # legend
    lx, ly = ml + pw + 130, mt + 6
    p.append(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="bold" fill="{C_TEXT}">Legend</text>')
    entries = [
        (C_READ, "read — probe held-out acc", None),
        (C_READ, "read — probe train acc", "5 4"),
        (C_BITE, "bite — best scoring arm", None),
        (C_CTRL, "control — margin_shuffled", "5 4"),
    ]
    for i, (col, text, dash) in enumerate(entries):
        yy = ly + 22 + i * 22
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+24}" y2="{yy}" stroke="{col}" stroke-width="2.4"{extra}/>')
        p.append(f'<text x="{lx+31}" y="{yy+4}" font-size="11" fill="{C_TEXT}">{_esc(text)}</text>')

    # per-layer detail
    yy = ly + 22 + len(entries) * 22 + 16
    p.append(f'<text x="{lx}" y="{yy}" font-size="12" font-weight="bold" fill="{C_TEXT}">Per layer</text>')
    p.append(
        f'<text x="{lx}" y="{yy+17}" font-size="10" fill="{C_MUTED}">layer  read   bite</text>'
    )
    for i, (l, rv) in enumerate(read):
        bv = next((v for ll, v, _ in bite if ll == l), float("nan"))
        p.append(
            f'<text x="{lx}" y="{yy+34+i*15}" font-size="10.5" fill="{C_TEXT}">L{l:<3d} '
            f"{rv:.3f}  {bv:.3f}</text>"
        )
    p.append("</svg>")
    return "\n".join(p)


def chart_alpha_sweep(summary: dict) -> str:
    """Accuracy against alpha, one line per layer, for the three scoring arms."""
    layers = _layers(summary)
    arms = summary["arms"]
    alphas = sorted({p["arm"].get("alpha") for p in arms.values() if p["arm"]["mechanism"] in {"margin", "logit"}})
    base_acc = arms["none"]["accuracy"]

    W, H = 880, 460
    ml, mr, mt, mb = 70, 210, 46, 60
    pw, ph = W - ml - mr, H - mt - mb
    allv = [base_acc]
    for payload in arms.values():
        if payload["arm"]["mechanism"] in {"margin", "logit", "margin_shuffled"}:
            allv.append(payload["accuracy"])
    lo, hi = min(allv), max(allv)
    span = max(hi - lo, 1e-3)
    lo -= span * 0.15
    hi += span * 0.15

    def x(a: float) -> float:
        if len(alphas) == 1:
            return ml + pw / 2
        return ml + pw * (alphas.index(a) / (len(alphas) - 1))

    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - lo) / (hi - lo))

    shades = ["#4338ca", "#6366f1", "#8b5cf6", "#a855f7", "#c026d3", "#d946ef"]
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="26" font-size="16" font-weight="bold" fill="{C_TEXT}">Alpha sweep — margin, logit and the shuffled control</text>',
        f'<text x="24" y="44" font-size="11.5" fill="{C_MUTED}">'
        "Solid = margin (probe-weighted), dashed = margin_shuffled (control), dotted = logit. "
        "A curve that tracks its dashed twin is not an effect.</text>",
    ]
    for frac in range(5):
        v = lo + (hi - lo) * frac / 4
        p.append(f'<line x1="{ml}" y1="{y(v):.1f}" x2="{ml+pw}" y2="{y(v):.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        p.append(f'<text x="{ml-10}" y="{y(v)+4:.1f}" font-size="11" fill="{C_MUTED}" text-anchor="end">{v:.3f}</text>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1.2"/>')
    for a in alphas:
        p.append(f'<text x="{x(a):.1f}" y="{mt+ph+20}" font-size="11" fill="{C_MUTED}" text-anchor="middle">{_fmt_alpha(a)}</text>')
    p.append(f'<text x="{ml+pw/2:.1f}" y="{mt+ph+42}" font-size="11.5" fill="{C_AXIS}" text-anchor="middle">alpha</text>')
    p.append(f'<text x="18" y="{mt+ph/2:.1f}" font-size="11.5" fill="{C_AXIS}" text-anchor="middle" transform="rotate(-90 18 {mt+ph/2:.1f})">accuracy</text>')
    p.append(
        f'<line x1="{ml}" y1="{y(base_acc):.1f}" x2="{ml+pw}" y2="{y(base_acc):.1f}" stroke="{C_NONE}" stroke-width="1.6" stroke-dasharray="7 4"/>'
    )
    p.append(f'<text x="{ml+pw+6}" y="{y(base_acc)+4:.1f}" font-size="11" fill="{C_NONE}">base {base_acc:.3f}</text>')

    def series(mech: str, layer: int) -> list[tuple[float, float]]:
        pts = []
        for payload in arms.values():
            if payload["arm"]["mechanism"] == mech and payload["arm"].get("layer") == layer:
                pts.append((payload["arm"]["alpha"], payload["accuracy"]))
        return sorted(pts)

    for i, l in enumerate(layers):
        col = shades[i % len(shades)]
        for mech, dash, width in (("margin", None, 2.2), ("margin_shuffled", "5 4", 1.6), ("logit", "2 3", 1.8)):
            pts = series(mech, l)
            if not pts:
                continue
            d = " ".join(f"{x(a):.1f},{y(v):.1f}" for a, v in pts)
            extra = f' stroke-dasharray="{dash}"' if dash else ""
            p.append(
                f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{width}"'
                f' opacity="{1.0 if dash is None else 0.75}"{extra}/>'
            )
            for a, v in pts:
                if dash is None:
                    p.append(f'<circle cx="{x(a):.1f}" cy="{y(v):.1f}" r="3.2" fill="#ffffff" stroke="{col}" stroke-width="1.8"/>')

    lx, ly = ml + pw + 22, mt + 6
    p.append(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="bold" fill="{C_TEXT}">Layer</text>')
    for i, l in enumerate(layers):
        col = shades[i % len(shades)]
        yy = ly + 20 + i * 18
        p.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" stroke="{col}" stroke-width="2.2"/>')
        p.append(f'<text x="{lx+28}" y="{yy+4}" font-size="11" fill="{C_TEXT}">L{l}</text>')
    yy = ly + 20 + len(layers) * 18 + 12
    for text, dash in (("margin", None), ("margin_shuffled", "5 4"), ("logit", "2 3")):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" stroke="{C_AXIS}" stroke-width="2"{extra}/>')
        p.append(f'<text x="{lx+28}" y="{yy+4}" font-size="11" fill="{C_TEXT}">{text}</text>')
        yy += 18
    p.append("</svg>")
    return "\n".join(p)


def chart_arm_matrix(summary: dict) -> str:
    """A compact table-as-figure: every arm, its accuracy, and its p-value."""
    arms = summary["arms"]
    rows = sorted(arms.items(), key=lambda kv: -kv[1]["accuracy"])
    W = 880
    rh = 20
    H = 70 + rh * len(rows) + 34
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="26" font-size="16" font-weight="bold" fill="{C_TEXT}">Every arm, ranked by accuracy</text>',
        f'<text x="24" y="44" font-size="11.5" fill="{C_MUTED}">'
        "All arms are shown, including the ones that did not work. "
        "p is the Wilcoxon signed-rank test of the arm against the base model, paired by item.</text>",
        f'<text x="24" y="64" font-size="11" fill="{C_MUTED}">arm</text>'
        f'<text x="330" y="64" font-size="11" fill="{C_MUTED}">accuracy</text>'
        f'<text x="410" y="64" font-size="11" fill="{C_MUTED}">95% CI</text>'
        f'<text x="520" y="64" font-size="11" fill="{C_MUTED}">world</text>'
        f'<text x="580" y="64" font-size="11" fill="{C_MUTED}">other</text>'
        f'<text x="650" y="64" font-size="11" fill="{C_MUTED}">p vs base</text>'
        f'<text x="770" y="64" font-size="11" fill="{C_MUTED}">beat control?</text>',
        f'<line x1="24" y1="70" x2="{W-24}" y2="70" stroke="{C_AXIS}" stroke-width="1"/>',
    ]
    for i, (label, payload) in enumerate(rows):
        yy = 70 + rh * i + 15
        mech = payload["arm"]["mechanism"]
        col = {
            "none": C_NONE,
            "prompt_oracle": C_ORACLE,
            "margin_shuffled": C_CTRL,
        }.get(mech, C_TEXT)
        if mech == "prompt_oracle":
            label = label + "  (ORACLE)"
        p.append(f'<text x="24" y="{yy}" font-size="11.5" fill="{col}">{_esc(label)}</text>')
        p.append(f'<text x="330" y="{yy}" font-size="11.5" fill="{C_TEXT}">{payload["accuracy"]:.3f}</text>')
        lo, hi = payload["accuracy_ci95"]
        p.append(f'<text x="410" y="{yy}" font-size="11.5" fill="{C_MUTED}">[{lo:.3f}, {hi:.3f}]</text>')
        p.append(f'<text x="520" y="{yy}" font-size="11.5" fill="{C_TEXT}">{payload["world_rate"]:.3f}</text>')
        p.append(f'<text x="580" y="{yy}" font-size="11.5" fill="{C_TEXT}">{payload["other_rate"]:.3f}</text>')
        p.append(f'<text x="650" y="{yy}" font-size="11.5" fill="{C_TEXT}">{payload["vs_none_wilcoxon"]["p_value"]:.3g}</text>')
        verdict = "—"
        if mech == "margin":
            ctl = arms.get(label.replace("margin-", "margin_shuffled-"))
            if ctl is not None:
                verdict = "yes" if payload["accuracy"] > ctl["accuracy"] else "NO"
        elif mech == "margin_shuffled":
            verdict = "control"
        elif mech == "none":
            verdict = "base"
        elif mech == "prompt_oracle":
            verdict = "upper bound"
        p.append(f'<text x="770" y="{yy}" font-size="11.5" fill="{C_TEXT}">{verdict}</text>')
    p.append("</svg>")
    return "\n".join(p)


# -------------------------------------------------------------------------- tables


def markdown_tables(summary: dict) -> str:
    arms = summary["arms"]
    layers = _layers(summary)
    subs = summary["subspaces"]
    base = arms["none"]
    oracle = arms["prompt_oracle"]
    lines: list[str] = []

    lines.append("### Baselines\n")
    lines.append("| arm | accuracy | 95% CI (bootstrap) | world rate | other rate | p vs base (Wilcoxon) |")
    lines.append("|---|---|---|---|---|---|")
    for label in ("none", "prompt_oracle"):
        a = arms[label]
        lo, hi = a["accuracy_ci95"]
        name = "base (`none`)" if label == "none" else "**`prompt_oracle` (ORACLE)**"
        lines.append(
            f"| {name} | {a['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{a['world_rate']:.3f} | {a['other_rate']:.3f} | "
            f"{a['vs_none_wilcoxon']['p_value']:.3g} |"
        )

    lines.append("\n### The read curve — belief probe, per layer\n")
    lines.append("| layer | probe train acc | probe held-out acc | shuffled control held-out acc | ‖w‖ |")
    lines.append("|---|---|---|---|---|")
    for l in layers:
        s = subs[str(l)]
        lines.append(
            f"| L{l} | {s['train_acc']:.3f} | **{s['heldout_acc']:.3f}** | "
            f"{s['shuffled_heldout_acc']:.3f} | {s['w_norm']:.3e} |"
        )

    lines.append("\n### Scoring arms at alpha = 1.0 — margin, its control, and logit\n")
    lines.append("| layer | margin | margin_shuffled (control) | logit | margin − control |")
    lines.append("|---|---|---|---|---|")
    for l in layers:
        m = arms.get(f"margin-L{l}-a1")
        c = arms.get(f"margin_shuffled-L{l}-a1")
        g = arms.get(f"logit-L{l}-a1")
        if not (m and c and g):
            continue
        lines.append(
            f"| L{l} | {m['accuracy']:.3f} | {c['accuracy']:.3f} | {g['accuracy']:.3f} | "
            f"{m['accuracy'] - c['accuracy']:+.3f} |"
        )

    lines.append("\n### Alpha sweep (margin mechanism)\n")
    alphas = sorted({p["arm"]["alpha"] for p in arms.values() if p["arm"]["mechanism"] == "margin"})
    header = "| layer | " + " | ".join(f"a={_fmt_alpha(a)}" for a in alphas) + " |"
    lines.append(header)
    lines.append("|---" * (len(alphas) + 1) + "|")
    for l in layers:
        cells = []
        for a in alphas:
            payload = arms.get(f"margin-L{l}-a{_fmt_alpha(a)}")
            cells.append(f"{payload['accuracy']:.3f}" if payload else "—")
        lines.append(f"| L{l} | " + " | ".join(cells) + " |")

    lines.append("\n### Activation-steering arms (alpha = 1.0)\n")
    lines.append("| layer | accuracy | 95% CI | world rate | other rate | p vs base |")
    lines.append("|---|---|---|---|---|---|")
    for l in layers:
        payload = arms.get(f"act-L{l}-a1")
        if not payload:
            continue
        lo, hi = payload["accuracy_ci95"]
        lines.append(
            f"| L{l} | {payload['accuracy']:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{payload['world_rate']:.3f} | {payload['other_rate']:.3f} | "
            f"{payload['vs_none_wilcoxon']['p_value']:.3g} |"
        )

    lines.append("\n### Every arm\n")
    lines.append("| arm | mechanism | layer | alpha | accuracy | world | other | p vs base |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, payload in sorted(arms.items(), key=lambda kv: -kv[1]["accuracy"]):
        a = payload["arm"]
        lines.append(
            f"| `{label}` | {a['mechanism']} | {a.get('layer') if a.get('layer') is not None else '—'} | "
            f"{_fmt_alpha(a['alpha']) if a.get('alpha') is not None else '—'} | {payload['accuracy']:.3f} | "
            f"{payload['world_rate']:.3f} | {payload['other_rate']:.3f} | "
            f"{payload['vs_none_wilcoxon']['p_value']:.3g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(ROOT / "results" / "phase0_summary.json"))
    ap.add_argument("--out-figures", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--out-tables", default=str(ROOT / "results" / "tables.md"))
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    figdir = Path(args.out_figures)
    figdir.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("phase0_read_bite.svg", chart_read_bite),
        ("phase0_alpha_sweep.svg", chart_alpha_sweep),
        ("phase0_arm_matrix.svg", chart_arm_matrix),
    ):
        path = figdir / name
        path.write_text(fn(summary))
        print(f"wrote {path}")

    tpath = Path(args.out_tables)
    tpath.write_text(markdown_tables(summary))
    print(f"wrote {tpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
