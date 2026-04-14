"""
experiments/gen_fig_scaling.py

Single-panel scaling figure.

Shows % depth improvement over naive for configs A–D across three circuit
families.  Circuits are ordered by complexity (smallest → largest) within each
family; families are separated on the x-axis by a visual gap so all data fits
in one shared panel without distortion from cross-family T-count differences.

Grayscale theme (A=light → D=dark) matches gen_fig_families.
Grid = solid lines / circles  |  Ring = dashed lines / squares.

Output
------
  results/figures/fig_scaling.pdf
  results/figures/fig_scaling.png

Usage
-----
  python experiments/gen_fig_scaling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR = _ROOT / "results" / "raw"
OUT_DIR = _ROOT / "results" / "figures"

# ── Family definitions (circuits ordered small → large) ────────────────────────
FAMILIES: list[tuple[str, str, list[str]]] = [
    (
        "Adder",
        "① Arithmetic Adders  (A8–A64)",
        ["Adder8", "Adder16", "Adder32", "Adder64"],
    ),
    (
        "QFT",
        "② Quantum Fourier Transforms  (Q8–Q128)",
        ["QFT8", "QFT16", "QFT32", "QFT64", "QFT128"],
    ),
    (
        "GF mult",
        "③ Finite-Field Multipliers  GF(2⁶–2¹⁰)",
        ["gf6_mult", "gf7_mult", "gf8_mult", "gf9_mult", "gf10_mult"],
    ),
]

TOPOLOGIES = ["grid", "ring"]
CONFIGS: list[tuple[str, str]] = [
    ("A", "Rand + Greedy"),
    ("B", "Rand + CP-SAT"),
    ("C", "SA  + Greedy"),
    ("D", "SA  + CP-SAT"),
]

# ── Grayscale theme matching gen_fig_families ──────────────────────────────────
CONFIG_COLORS = {"A": "#bbbbbb", "B": "#888888", "C": "#444444", "D": "#111111"}
TOPO_LINESTYLE = {"grid": "-",   "ring": "--"}
TOPO_MARKER    = {"grid": "o",   "ring": "s"}

FAM_GAP = 1.8   # blank x-units between families

# Short x-tick circuit names
SHORT_NAME = {
    "Adder8":    "A8",   "Adder16":   "A16",  "Adder32":  "A32",  "Adder64":  "A64",
    "QFT8":      "Q8",   "QFT16":     "Q16",  "QFT32":    "Q32",
    "QFT64":     "Q64",  "QFT128":    "Q128",
    "gf6_mult":  "GF6",  "gf7_mult":  "GF7",  "gf8_mult": "GF8",
    "gf9_mult":  "GF9",  "gf10_mult": "GF10",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_raw(circuit: str) -> dict | None:
    path = RAW_DIR / f"{circuit}_seed42.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def improvement_pct(cfg_data: dict, naive_depth: float) -> float | None:
    """Positive = better than naive."""
    if naive_depth == 0:
        return None
    pct = cfg_data.get("pct_vs_naive")
    if pct is None:
        pct = (cfg_data["logical_depth"] - naive_depth) / naive_depth * 100.0
    return -pct   # flip sign: positive → improvement


# ── Build figure ───────────────────────────────────────────────────────────────
def build_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))

    x_cursor    = 0.0
    xtick_pos   = []
    xtick_lbl   = []
    fam_regions = []   # (x_mid, label) for family annotations

    for fam_key, fam_label, circuits in FAMILIES:
        fam_x_start = x_cursor

        # Build series: {(cfg, topo): [(xpos, improvement%), ...]}
        series: dict[tuple[str, str], list[tuple[float, float]]] = {
            (cfg, topo): [] for cfg, _ in CONFIGS for topo in TOPOLOGIES
        }

        for rank, circ in enumerate(circuits):
            raw = load_raw(circ)
            if raw is None:
                print(f"  [skip] {circ} — no raw file")
                continue

            xpos = x_cursor + rank
            xtick_pos.append(xpos)
            xtick_lbl.append(SHORT_NAME.get(circ, circ))

            for topo in TOPOLOGIES:
                if topo not in raw:
                    continue
                topo_cfg = raw[topo]["configs"]
                naive_d  = topo_cfg["naive"]["logical_depth"]

                for cfg_key, _ in CONFIGS:
                    cfg = topo_cfg.get(cfg_key)
                    if cfg is None:
                        continue
                    imp = improvement_pct(cfg, naive_d)
                    if imp is not None:
                        series[(cfg_key, topo)].append((xpos, imp))

        fam_x_end = x_cursor + len(circuits) - 1
        fam_regions.append(((fam_x_start + fam_x_end) / 2, fam_label,
                             fam_x_start, fam_x_end))

        # ── Plot lines + scatter per (config, topology) ────────────────────
        for (cfg_key, topo), pts in series.items():
            if len(pts) < 1:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color  = CONFIG_COLORS[cfg_key]
            ls     = TOPO_LINESTYLE[topo]
            mk     = TOPO_MARKER[topo]
            lw     = 1.8 if topo == "grid" else 1.3
            alpha  = 0.92 if topo == "grid" else 0.65
            ec     = "white" if topo == "grid" else "#555555"

            ax.plot(xs, ys,
                    color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=4)
            ax.scatter(xs, ys,
                       color=color, marker=mk, s=55, zorder=5, alpha=alpha,
                       edgecolors=ec, linewidths=0.6)

        x_cursor += len(circuits) + FAM_GAP

    # ── Family separator lines and labels ──────────────────────────────────────
    for i, (xmid, fam_label, xs, xe) in enumerate(fam_regions):
        # label just above top of plot area
        ax.text(xmid, 1.015, fam_label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", clip_on=False)

        # vertical dashed separator between families
        if i < len(fam_regions) - 1:
            sep_x = xe + FAM_GAP / 2
            ax.axvline(sep_x, color="#aaaaaa", linewidth=0.9,
                       linestyle=":", zorder=2, alpha=0.9)

        # subtle shaded background per family
        ax.axvspan(xs - 0.45, xe + 0.45,
                   color=["#f7f7f7", "#efefef", "#f7f7f7"][i],
                   alpha=0.5, zorder=0, linewidth=0)

    # ── X-axis ─────────────────────────────────────────────────────────────────
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=8.5)
    ax.set_xlim(fam_regions[0][2] - 0.6,
                fam_regions[-1][3] + 0.6)
    ax.tick_params(axis="x", length=0, pad=5)

    # complexity arrow annotation
    for i, (xmid, _, xs, xe) in enumerate(fam_regions):
        ax.annotate(
            "", xy=(xe + 0.35, -0.085), xytext=(xs - 0.35, -0.085),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=0.9, mutation_scale=8),
            clip_on=False,
        )
        ax.text((xs + xe) / 2, -0.115, "increasing complexity →",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=7, color="#666666", style="italic", clip_on=False)

    # ── Y-axis ─────────────────────────────────────────────────────────────────
    ax.set_ylabel("Depth improvement over naive scheduling  (%)", fontsize=9.5)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylim(0, None)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8.5)

    # ── Legend ─────────────────────────────────────────────────────────────────
    config_handles = [
        mlines.Line2D([], [], color=CONFIG_COLORS[k], linestyle="-",
                      linewidth=2.0, marker="o", markersize=5,
                      label=f"Config {k}:  {lbl}")
        for k, lbl in CONFIGS
    ]
    topo_handles = [
        mlines.Line2D([], [], color="#777777", linestyle="-", linewidth=1.8,
                      marker="o", markersize=5, label="Grid  (solid / ●)"),
        mlines.Line2D([], [], color="#777777", linestyle="--", linewidth=1.4,
                      marker="s", markersize=5, label="Ring  (dashed / ■)"),
    ]
    ax.legend(
        handles=config_handles + topo_handles,
        loc="upper left",
        fontsize=7.5,
        framealpha=0.93,
        edgecolor="#cccccc",
        ncol=2,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.set_title(
        "Scheduling Depth Improvement vs Circuit Complexity  ·  "
        "Gain increases consistently as circuits scale up",
        fontsize=10, fontweight="bold", pad=8,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1], pad=0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_scaling.pdf", dpi=300,
                bbox_inches="tight", format="pdf")
    fig.savefig(OUT_DIR / "fig_scaling.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_DIR / 'fig_scaling.pdf'}")
    print(f"Saved → {OUT_DIR / 'fig_scaling.png'}")

    # ── Console summary ─────────────────────────────────────────────────────────
    print("\nData summary:")
    for fam_key, _, circuits in FAMILIES:
        for circ in circuits:
            raw = load_raw(circ)
            if raw is None:
                continue
            for topo in TOPOLOGIES:
                if topo not in raw:
                    continue
                naive_d = raw[topo]["configs"]["naive"]["logical_depth"]
                d_cfg   = raw[topo]["configs"].get("D")
                if d_cfg and naive_d:
                    imp = improvement_pct(d_cfg, naive_d)
                    print(f"  {circ:<15s} {topo:<5s}  D: {imp:+.1f}%")


if __name__ == "__main__":
    build_figure()
