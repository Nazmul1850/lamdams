"""
experiments/gen_fig_families.py

Multi-family depth comparison — two-column, three-row layout.

Layout
------
3 rows × 2 columns:
  Row    = circuit family  (Adder / QFT / GF mult)
  Left col  = Grid topology
  Right col = Ring topology

Within each panel (one family × one topology):
  - x-axis  : circuits ordered by size, labelled by circuit name + hw detail
  - bars     : configs A–D, normalised to SC baseline depth (y = depth / sc_depth)
               Grayscale A→D: lightest → darkest (prints cleanly in B&W)
  - y-axis   : depth / SC-baseline depth (log scale, shared per row)
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
COL_TITLE  = {"grid": "Grid Topology", "ring": "Ring Topology"}

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
FS_TITLE    = 12
FS_AXLABEL  = 10
FS_TICK     = 9.5
FS_LEGEND   = 11
FS_BAR      = 8.5   # improvement labels on bars
FS_NAIVE    = 8.5   # "naive" annotation
FS_COLTITLE = 12    # column header
FS_XLBL     = 9.5   # circuit name below bars
FS_XHWLBL   = 8.0   # hw detail below circuit name

# ── Bar geometry ──────────────────────────────────────────────────────────────
BAR_W       = 0.30   # width of each bar — wide enough for % labels on top
CIRCUIT_PAD = 0.22   # gap between adjacent circuits


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_sc_depths() -> dict[str, dict]:
    if not SC_PATH.exists():
        return {}
    rows = json.loads(SC_PATH.read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    out: dict[str, dict] = {}
    for r in rows:
        c = r["circuit"]
        if c not in out or r.get("timestamp", "") > out[c].get("timestamp", ""):
            out[c] = r
    return out


def load_raw(circuit: str) -> dict | None:
    p = RAW_DIR / f"{circuit}_seed42.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── Draw one panel (family × topology) ────────────────────────────────────────
def _draw_panel(
    ax,
    circuits: list[str],
    topo: str,
    sc_map: dict,
    all_norm_vals: list[float],
    all_naive_vals: list[float],
) -> None:
    entries = []
    for circ in circuits:
        raw = load_raw(circ)
        if raw is None:
            continue
        if topo not in raw:
            continue
        sc_row = sc_map.get(circ)
        if sc_row is None or sc_row.get("sc_logical_depth", 0) == 0:
            continue
        sc_d = float(sc_row["sc_logical_depth"])
        entries.append({"circuit": circ, "raw": raw, "sc_d": sc_d})

    if not entries:
        ax.set_visible(False)
        return

    n_cfg    = len(CONFIGS)
    grp_w    = n_cfg * BAR_W           # width of one circuit's bar cluster
    cursor   = 0.0
    xtick_pos: list[float] = []
    naive_spans: list[tuple[float, float, float]] = []
    xlabel_info: list[tuple[float, str, str]] = []   # (x_ctr, circ_lbl, hw_lbl)

    for entry in entries:
        circ = entry["circuit"]
        raw  = entry["raw"]
        sc_d = entry["sc_d"]
        topo_data = raw[topo]

        x_ctr = cursor + grp_w / 2
        xtick_pos.append(x_ctr)

        # hw label
        td = raw.get(topo, {})
        if topo == "grid":
            gr = td.get("grid_rows"); gc = td.get("grid_cols")
            fr = td.get("fill_rate")
            hw_lbl = (f"{gr}×{gc}B, {fr*100:.0f}%" if gr and gc and fr is not None
                      else f"{td.get('n_blocks','?')}B")
        else:
            nb = td.get("n_blocks", "?"); fr = td.get("fill_rate")
            hw_lbl = f"{nb}B, {fr*100:.0f}%" if fr is not None else f"{nb}B"

        xlabel_info.append((x_ctr, X_LABEL.get(circ, circ), hw_lbl))

        # naive depth for this topology
        naive_d = topo_data.get("configs", {}).get("naive", {}).get("logical_depth")
        naive_ratio = (naive_d / sc_d) if naive_d else None
        if naive_d:
            x_lo = cursor - BAR_W * 0.5
            x_hi = cursor + grp_w + BAR_W * 0.5
            naive_spans.append((x_lo, x_hi, naive_ratio))
            all_naive_vals.append(naive_ratio)

        # bars
        for cfg_idx, (cfg_key, _) in enumerate(CONFIGS):
            cfg = topo_data.get("configs", {}).get(cfg_key)
            if cfg is None:
                continue
            depth = cfg["logical_depth"]
            norm  = depth / sc_d
            pct   = cfg.get("pct_vs_naive")
            all_norm_vals.append(norm)

            xc = cursor + cfg_idx * BAR_W + BAR_W / 2
            ax.bar(
                xc, norm,
                width=BAR_W * 0.84,
                color=CONFIG_GRAY[cfg_key],
                hatch=HATCH[topo],
                edgecolor="#333333" if topo == "ring" else "white",
                linewidth=0.5,
                alpha=0.97,
                zorder=3,
            )

            if pct is not None:
                label_str = f"−{abs(pct):.0f}%"
                # place just above bar top in log space
                y_lbl = norm * (10 ** 0.04)
                ax.text(
                    xc, y_lbl, label_str,
                    ha="center", va="bottom",
                    fontsize=FS_BAR, rotation=0,
                    color="#111111", zorder=4,
                    fontweight="bold" if cfg_key == "D" else "normal",
                )

        cursor += grp_w + CIRCUIT_PAD

    # SC baseline
    ax.axhline(1.0, color="black", linewidth=1.6, linestyle="-", zorder=7)

    # naive lines
    for i, (x_lo, x_hi, naive_ratio) in enumerate(naive_spans):
        ax.hlines(naive_ratio, x_lo, x_hi,
                  colors="#555555", linewidth=1.2, linestyle="--", zorder=5)
        if i == 0:
            ax.text(x_hi + 0.03, naive_ratio, "naive",
                    va="center", ha="left",
                    fontsize=FS_NAIVE, color="#555555", style="italic")

    # x-axis labels — 2 rows:
    #   Row 1 (−9 pt):  circuit name  e.g. "16-bit"  (bold)
    #   Row 2 (−20 pt): hw detail     e.g. "4B, 52%"  (italic, muted)
    for x_ctr, circ_lbl, hw_lbl in xlabel_info:
        ax.annotate(
            circ_lbl,
            xy=(x_ctr, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -9), textcoords="offset points",
            ha="center", va="top",
            fontsize=FS_XLBL, fontweight="bold", color="black",
        )
        ax.annotate(
            hw_lbl,
            xy=(x_ctr, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -20), textcoords="offset points",
            ha="center", va="top",
            fontsize=FS_XHWLBL, color="#444444", style="italic",
        )

    # axes formatting
    ax.set_yscale("log")
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([""] * len(xtick_pos))
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-BAR_W, cursor - CIRCUIT_PAD + BAR_W)

    ax.yaxis.set_major_locator(
        mticker.LogLocator(base=10, subs=[1.0, 2.0, 5.0], numticks=8)
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{int(v)}×" if v == int(v) else f"{v:.1f}×"
        )
    )
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ── Build figure ──────────────────────────────────────────────────────────────
def build_figure() -> None:
    sc_map   = load_sc_depths()
    fam_list = list(FAMILIES.items())
    n_fam    = len(fam_list)

    fig, axes = plt.subplots(
        n_fam, 2,
        figsize=(16.0, 2.6 * n_fam),
        sharey="row",
        squeeze=False,
    )

    all_norm_vals:  list[float] = []
    all_naive_vals: list[float] = []

    PANEL_TITLE = {
        "Adder":   "① Adders  (8–64 bit)",
        "QFT":     "② QFT  (8–128 qubit)",
        "GF mult": "③ GF Mult  (n=6–10)",
    }

    for row, (fam, circuits) in enumerate(fam_list):
        for col, topo in enumerate(TOPOLOGIES):
            ax = axes[row][col]
            _draw_panel(ax, circuits, topo, sc_map,
                        all_norm_vals, all_naive_vals)

            # row label: left panel only
            if col == 0:
                ax.set_ylabel("Depth / SC baseline  (log)", fontsize=FS_AXLABEL)
                ax.set_title(PANEL_TITLE.get(fam, fam),
                             fontsize=FS_TITLE, fontweight="bold",
                             pad=6, loc="left")

    # ── column headers ────────────────────────────────────────────────────────
    for col, topo in enumerate(TOPOLOGIES):
        axes[0][col].set_title(
            COL_TITLE[topo],
            fontsize=FS_COLTITLE, fontweight="bold",
            color="#222222", pad=10,
            loc="center",
        )

    # ── global y-limits (shared per row via sharey) ───────────────────────────
    all_vals  = all_norm_vals + all_naive_vals
    ymax_glob = max(all_vals) * 1.12 if all_vals else 50
    for row in range(n_fam):
        axes[row][0].set_ylim(0.80, ymax_glob)

    # ── shared legend at bottom ───────────────────────────────────────────────
    config_patches = [
        mpatches.Patch(facecolor=CONFIG_GRAY[k], edgecolor="white",
                       label=f"Config {k}:  {lbl}")
        for k, lbl in CONFIGS
    ]
    topo_patches = [
        mpatches.Patch(facecolor="#888888", edgecolor="white",
                       hatch="",    label="Grid  (solid fill)"),
        mpatches.Patch(facecolor="#888888", edgecolor="#333333",
                       hatch="///", label="Ring  (hatched)"),
    ]
    from matplotlib.lines import Line2D
    ref_lines = [
        Line2D([0], [0], color="black",   linestyle="-",  linewidth=1.6,
               label="SC baseline  (1×)"),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2,
               label="Naive  (random + sequential)"),
    ]
    fig.legend(
        handles=ref_lines + config_patches + topo_patches,
        loc="lower center",
        ncol=4,
        fontsize=FS_LEGEND,
        framealpha=0.95,
        edgecolor="#aaaaaa",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.0,
        handleheight=1.2,
        borderpad=0.8,
        labelspacing=0.5,
    )

    fig.tight_layout(rect=[0, 0.11, 1, 1], pad=0.4, h_pad=2.0, w_pad=1.0)

    # ── save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_family_depth.pdf", dpi=300,
                bbox_inches="tight", format="pdf")
    fig.savefig(OUT_DIR / "fig_family_depth.png", dpi=180,
                bbox_inches="tight")
    print(f"Saved → {OUT_DIR / 'fig_family_depth.pdf'}")
    print(f"Saved → {OUT_DIR / 'fig_family_depth.png'}")


if __name__ == "__main__":
    build_figure()
