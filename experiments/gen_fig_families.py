"""
experiments/gen_fig_families.py

Multi-family depth comparison — single-column, three stacked panels.

Layout
------
Three vertically stacked panels, one per family (Adder / QFT / GF mult).
Within each panel:
  - x-axis  : circuits ordered by size, labelled by qubit/bit count
  - bars     : configs A–D, normalised to SC baseline depth (y = depth / sc_depth)
               Grid = solid fill   |   Ring = hatched (//)
               Gray scale A→D: lightest → darkest (prints cleanly in B&W)
  - y-axis   : depth / SC-baseline depth (log scale, independent per panel)
  - solid black line at y = 1.0  — SC baseline anchor
  - dashed line per circuit      — naive overhead level
  - improvement label on each bar — "−XX%" reduction vs naive, rotated 90°

Output
------
  results/figures/fig_family_depth.pdf
  results/figures/fig_family_depth.png

Usage
-----
  python experiments/gen_fig_families.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR = _ROOT / "results" / "raw"
SC_PATH = _ROOT / "results" / "sc_baseline" / "sc_baseline.json"
OUT_DIR = _ROOT / "results" / "figures"

# ── Family definitions ────────────────────────────────────────────────────────
FAMILIES: dict[str, list[str]] = {
    "Adder":   ["Adder8",   "Adder16",  "Adder32",  "Adder64"],
    "QFT":     ["QFT8",     "QFT16",    "QFT32",    "QFT64",   "QFT128"],
    "GF mult": ["gf6_mult", "gf7_mult", "gf8_mult", "gf9_mult","gf10_mult"],
}

# Human-readable x-tick labels per circuit
X_LABEL: dict[str, str] = {
    "Adder8":    "8-bit",  "Adder16":   "16-bit",
    "Adder32":   "32-bit", "Adder64":   "64-bit",
    "QFT8":      "8q",     "QFT16":     "16q",
    "QFT32":     "32q",    "QFT64":     "64q",    "QFT128": "128q",
    "gf6_mult":  "n=6",    "gf7_mult":  "n=7",
    "gf8_mult":  "n=8",    "gf9_mult":  "n=9",    "gf10_mult": "n=10",
}

TOPOLOGIES = ["grid", "ring"]

CONFIGS: list[tuple[str, str]] = [
    ("A", "Rand+Greedy"),
    ("B", "Rand+CP-SAT"),
    ("C", "LaM+Greedy"),
    ("D", "LaM+CP-SAT"),
]

# ── Grayscale palette — prints cleanly in B&W ─────────────────────────────────
CONFIG_GRAY = {
    "A": "#cccccc",   # light gray
    "B": "#888888",   # medium gray
    "C": "#444444",   # dark gray
    "D": "#111111",   # near black
}
HATCH = {"grid": "", "ring": "///"}

# ── Typography ────────────────────────────────────────────────────────────────
FS_TITLE   = 13
FS_AXLABEL = 11
FS_TICK    = 10
FS_LEGEND  = 9
FS_BAR     = 7      # improvement labels on bars
FS_NAIVE   = 8      # "naive" annotation

# ── Bar geometry ──────────────────────────────────────────────────────────────
BAR_W       = 0.17
GROUP_PAD   = 0.06   # gap between grid-group and ring-group within a circuit
CIRCUIT_PAD = 0.42   # gap between adjacent circuits


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_sc_depths() -> dict[str, dict]:
    if not SC_PATH.exists():
        return {}
    rows = json.loads(SC_PATH.read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    # keep latest entry per circuit
    out: dict[str, dict] = {}
    for r in rows:
        c = r["circuit"]
        if c not in out or r.get("timestamp","") > out[c].get("timestamp",""):
            out[c] = r
    return out


def load_raw(circuit: str) -> dict | None:
    p = RAW_DIR / f"{circuit}_seed42.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Build figure ──────────────────────────────────────────────────────────────
def build_figure() -> None:
    sc_map = load_sc_depths()   # circuit → sc_row dict

    n_fam = len(FAMILIES)
    fig, axes = plt.subplots(
        n_fam, 1,
        figsize=(7.0, 3.8 * n_fam),
        squeeze=True,
    )
    if n_fam == 1:
        axes = [axes]

    all_norm_vals:  list[float] = []   # config bar values
    all_naive_vals: list[float] = []   # naive line values — kept separate for ylim

    for ax, (fam, circuits) in zip(axes, FAMILIES.items()):

        # ── collect entries ───────────────────────────────────────────────────
        entries = []
        for circ in circuits:
            raw = load_raw(circ)
            if raw is None:
                print(f"  [skip] {circ} — no raw file")
                continue
            sc_row = sc_map.get(circ)
            if sc_row is None or sc_row.get("sc_logical_depth", 0) == 0:
                print(f"  [skip] {circ} — no SC baseline")
                continue
            sc_d = float(sc_row["sc_logical_depth"])
            entries.append({"circuit": circ, "raw": raw, "sc_d": sc_d})

        if not entries:
            ax.set_visible(False)
            continue

        # ── x layout: collect positions while drawing bars ────────────────────
        xticks_pos: list[float] = []
        naive_spans: list[tuple[float, float, float]] = []  # (x_lo, x_hi, ratio)
        # three-row x-axis labels: (x_gm, x_rm, x_ctr, circ_lbl, hw_g, hw_r)
        xlabel_rows: list[tuple] = []
        cursor = 0.0

        for entry in entries:
            circ  = entry["circuit"]
            raw   = entry["raw"]
            sc_d  = entry["sc_d"]

            grid_w  = len(CONFIGS) * BAR_W
            ring_w  = len(CONFIGS) * BAR_W
            group_w = grid_w + GROUP_PAD + ring_w

            x_grid_start = cursor
            x_ring_start = cursor + grid_w + GROUP_PAD
            x_grid_mid   = cursor + grid_w / 2
            x_ring_mid   = x_ring_start + ring_w / 2
            x_center     = cursor + group_w / 2

            xticks_pos.append(x_center)

            # Build compact hw labels from raw fields
            g_topo = raw.get("grid", {})
            r_topo = raw.get("ring", {})
            gr = g_topo.get("grid_rows"); gc = g_topo.get("grid_cols")
            g_nb = g_topo.get("n_blocks", "?")
            g_fr = g_topo.get("fill_rate")
            r_nb = r_topo.get("n_blocks", "?")
            r_fr = r_topo.get("fill_rate")
            hw_g = (f"{gr}×{gc}B, {g_fr*100:.0f}%"
                    if gr and gc and g_fr is not None
                    else f"{g_nb}B")
            hw_r = (f"{r_nb}B, {r_fr*100:.0f}%"
                    if r_fr is not None else f"{r_nb}B")

            xlabel_rows.append((x_grid_mid, x_ring_mid, x_center,
                                 X_LABEL.get(circ, circ), hw_g, hw_r))

            # naive depth for reference line + label y-anchor
            naive_g = (raw.get("grid", {})
                          .get("configs", {})
                          .get("naive", {})
                          .get("logical_depth"))
            naive_ratio = (naive_g / sc_d) if naive_g else None
            if naive_g:
                x_lo = cursor - BAR_W * 0.4
                x_hi = cursor + group_w + BAR_W * 0.4
                naive_spans.append((x_lo, x_hi, naive_ratio))
                all_naive_vals.append(naive_ratio)   # track for ylim

            # ── draw bars ─────────────────────────────────────────────────────
            for topo in TOPOLOGIES:
                if topo not in raw:
                    continue
                topo_data = raw[topo]
                x_start   = x_grid_start if topo == "grid" else x_ring_start

                for cfg_idx, (cfg_key, _) in enumerate(CONFIGS):
                    cfg = topo_data.get("configs", {}).get(cfg_key)
                    if cfg is None:
                        continue
                    depth = cfg["logical_depth"]
                    norm  = depth / sc_d
                    pct   = cfg.get("pct_vs_naive")
                    all_norm_vals.append(norm)

                    xc = x_start + cfg_idx * BAR_W + BAR_W / 2

                    ax.bar(
                        xc, norm,
                        width=BAR_W * 0.84,
                        color=CONFIG_GRAY[cfg_key],
                        hatch=HATCH[topo],
                        edgecolor="#333333" if topo == "ring" else "white",
                        linewidth=0.55,
                        alpha=0.93,
                        zorder=3,
                    )

                    # ── pct label: just below the naive line ──────────────────
                    if pct is not None and naive_ratio is not None:
                        label_str = f"−{abs(pct):.0f}%"
                        # place at 88 % of naive in log space → just below line
                        y_lbl = naive_ratio * (10 ** -0.055)
                        ax.text(
                            xc, y_lbl, label_str,
                            ha="center", va="top",
                            fontsize=FS_BAR, rotation=90,
                            color="#111111", zorder=4,
                            fontweight="bold" if cfg_key == "D" else "normal",
                        )

            cursor += group_w + CIRCUIT_PAD

        # ── SC baseline anchor ────────────────────────────────────────────────
        ax.axhline(1.0, color="black", linewidth=1.8, linestyle="-",
                   zorder=7, label="SC baseline  (1×)")

        # ── Naive lines per circuit ───────────────────────────────────────────
        for i, (x_lo, x_hi, naive_ratio) in enumerate(naive_spans):
            ax.hlines(
                naive_ratio, x_lo, x_hi,
                colors="#555555", linewidth=1.4, linestyle="--",
                zorder=5,
            )
            if i == 0:
                ax.text(
                    x_hi + 0.04, naive_ratio, "naive",
                    va="center", ha="left",
                    fontsize=FS_NAIVE, color="#555555", style="italic",
                )

        # ── Three-row x-axis labels ───────────────────────────────────────────
        # Row 1 (−9 pt):   "Grid" | "Ring"  (topology type)
        # Row 2 (−19 pt):  hw detail  e.g. "2×2 (4B, 52%)" | "ring (3B, 70%)"
        # Row 3 (−34 pt):  circuit name  e.g. "8-bit"  (bold, largest)
        for x_gm, x_rm, x_ctr, circ_lbl, hw_g, hw_r in xlabel_rows:
            # Row 1 — topology name
            for xm, lbl in [(x_gm, "Grid"), (x_rm, "Ring")]:
                ax.annotate(
                    lbl,
                    xy=(xm, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=FS_BAR + 0.5, color="#333333",
                    fontweight="semibold",
                )
            # Row 2 — hw detail
            for xm, lbl in [(x_gm, hw_g), (x_rm, hw_r)]:
                ax.annotate(
                    lbl,
                    xy=(xm, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -19), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=FS_BAR - 0.5, color="#666666",
                    style="italic",
                )
            # Row 3 — circuit name
            ax.annotate(
                circ_lbl,
                xy=(x_ctr, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -34), textcoords="offset points",
                ha="center", va="top",
                fontsize=FS_TICK, fontweight="bold", color="black",
            )

        # ── Axes formatting ───────────────────────────────────────────────────
        ax.set_yscale("log")
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels([""] * len(xticks_pos))   # labels drawn manually above
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(-BAR_W * 1.5, cursor - CIRCUIT_PAD + BAR_W * 1.5)
        PANEL_TITLE = {
            "Adder":   "① Arithmetic Adders  (Adder8–64)",
            "QFT":     "② Quantum Fourier Transforms  (QFT8–128)",
            "GF mult": "③ Finite-Field Multipliers  GF(2\u2076\u20132\u00b9\u2070)",
        }
        ax.set_ylabel("Depth  /  SC baseline  (log scale)", fontsize=FS_AXLABEL)
        ax.set_title(PANEL_TITLE.get(fam, fam), fontsize=FS_TITLE,
                     fontweight="bold", pad=8, loc="left")

        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=10)
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _: f"{int(v)}×" if v == int(v) else f"{v:.1f}×"
            )
        )
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="y", labelsize=FS_TICK)

        ax.yaxis.grid(True, which="major", linestyle="-",  linewidth=0.45, alpha=0.45, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    # ── set y-limits per panel ────────────────────────────────────────────────
    # y-max: just enough to clear the highest naive line — no rounding to a clean
    # tick so the axis ends naturally without adding an empty decade at the top.
    all_vals = all_norm_vals + all_naive_vals
    ymax_global = max(all_vals) * 1.10 if all_vals else 50
    for ax in axes:
        ax.set_ylim(0.80, ymax_global)

    # ── shared legend (bottom of figure) ──────────────────────────────────────
    config_patches = [
        mpatches.Patch(
            facecolor=CONFIG_GRAY[k], edgecolor="white",
            label=f"Config {k}:  {lbl}  (Grid)",
        )
        for k, lbl in CONFIGS
    ]
    topo_patches = [
        mpatches.Patch(facecolor="#888888", edgecolor="white",
                       hatch="",    label="Grid topology   (solid)"),
        mpatches.Patch(facecolor="#888888", edgecolor="#333333",
                       hatch="///", label="Ring topology   (hatched)"),
    ]
    from matplotlib.lines import Line2D
    ref_lines = [
        Line2D([0],[0], color="black",   linestyle="-",  linewidth=1.8,
               label="SC baseline  (1×)"),
        Line2D([0],[0], color="#555555", linestyle="--", linewidth=1.4,
               label="Naive  (random + sequential)"),
    ]
    all_handles = ref_lines + config_patches + topo_patches

    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=3,
        fontsize=FS_LEGEND,
        framealpha=0.95,
        edgecolor="#aaaaaa",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1], pad=0.4, h_pad=1.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_family_depth.pdf", dpi=300,
                bbox_inches="tight", format="pdf")
    fig.savefig(OUT_DIR / "fig_family_depth.png", dpi=180,
                bbox_inches="tight")
    print(f"Saved → {OUT_DIR / 'fig_family_depth.pdf'}")
    print(f"Saved → {OUT_DIR / 'fig_family_depth.png'}")


if __name__ == "__main__":
    build_figure()
