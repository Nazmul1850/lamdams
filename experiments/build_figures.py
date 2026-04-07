# -*- coding: utf-8 -*-
"""
Build all paper figures from raw experiment results in results/raw/.

Figures produced:
  fig_paper_1_inter_block.png   — SA vs baselines: inter-block rotations
  fig_paper_2_robustness.png    — Box plots over 30 seeds (Config E)
  fig_paper_3_depth.png         — End-to-end logical depth, 4 configs × 2 circuits
  fig_paper_4_scatter.png       — Compile time vs depth scatter (scheduler tradeoff)
  fig_paper_5_scaling.png       — Depth reduction % vs block count

Usage:
    python experiments/build_figures.py [--all] [--fig 1] [--fig 3] ...
    python experiments/build_figures.py --table1   # print Table 1 to stdout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR   = os.path.join(_ROOT, "runs")
SC_DIR     = os.path.join(_ROOT, "results", "sc_baseline")
FIG_DIR    = os.path.join(_ROOT, "results", "figures")

# ── Style constants ───────────────────────────────────────────────────────────

MAPPING_LABELS = {"random": "Random", "sa": "SA (ours)"}
SCHEDULER_LABELS = {"greedy_critical": "Greedy", "cpsat": "CP-SAT"}

CONFIG_LABELS = {
    ("random", "greedy_critical"): "A: Rand+Greedy",
    ("random", "cpsat"):           "B: Rand+CPSAT",
    ("sa",     "greedy_critical"): "C: SA+Greedy",
    ("sa",     "cpsat"):           "D: SA+CPSAT",
}
CONFIG_COLORS = {
    ("random", "greedy_critical"): "#d62728",
    ("random", "cpsat"):           "#ff7f0e",
    ("sa",     "greedy_critical"): "#1f77b4",
    ("sa",     "cpsat"):           "#2ca02c",
}

CIRCUIT_DISPLAY = {
    "gf2_16_mult":    "GF(2^16) mult",
    "qft_100_approx": "QFT-100",
    "random_ct_500q_10k": "Rand-500q/10k",
}

_RC = {
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  8,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_results(
    circuit=None, mapping=None, scheduler=None, seed=None
) -> List[dict]:
    records = []
    if not os.path.isdir(RUNS_DIR):
        return records
    for fname in sorted(os.listdir(RUNS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RUNS_DIR, fname)) as f:
            rec = json.load(f)
        # backwards-compat: old files used "placement" key
        if "mapping" not in rec and "placement" in rec:
            rec["mapping"] = rec["placement"]
        if circuit   and rec.get("circuit")  != circuit:  continue
        if mapping   and rec.get("mapping")  != mapping:  continue
        if scheduler and rec.get("scheduler") != scheduler: continue
        if seed is not None and rec.get("seed") != seed:   continue
        records.append(rec)
    return records


def _single(circuit, mapping, scheduler, seed=42) -> dict | None:
    recs = load_results(circuit=circuit, mapping=mapping, scheduler=scheduler, seed=seed)
    return recs[0] if recs else None


def _require(circuit, mapping, scheduler, seed=42) -> dict:
    rec = _single(circuit, mapping, scheduler, seed)
    if rec is None:
        raise FileNotFoundError(
            f"Missing result: {circuit} | {mapping} | {scheduler} | seed={seed}\n"
            f"Run: python experiments/run_experiment.py --circuit {circuit}"
            f" --mapping {mapping} --scheduler {scheduler} --seed {seed}"
        )
    return rec


def _display(circuit: str) -> str:
    return CIRCUIT_DISPLAY.get(circuit, circuit)


# ── Figure 1: inter-block rotations ──────────────────────────────────────────

def build_fig1(circuits=None, save=True):
    """
    Grouped bar chart: inter_block_rotations for random / greedy / sa
    across benchmark circuits.  Shows that SA placement reduces inter-block
    communication (the cost that drives logical depth).
    """
    if circuits is None:
        circuits = ["gf2_16_mult", "qft_100_approx"]

    placements = ["random", "greedy", "sa"]
    colors     = ["#d62728", "#ff7f0e", "#2ca02c"]

    x = np.arange(len(circuits))
    width = 0.25

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6, 3.5))

        any_data = False
        for i, (mapping, color) in enumerate(zip(placements, colors)):
            vals = []
            for circuit in circuits:
                rec = _single(circuit, mapping, "greedy_critical") or \
                      _single(circuit, mapping, "cpsat")
                vals.append(rec["inter_block_rotations"] if rec else 0)
            if any(v > 0 for v in vals):
                any_data = True
            bars = ax.bar(
                x + i * width, vals, width,
                label=MAPPING_LABELS[mapping],
                color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
            )

        ax.set_xlabel("Benchmark circuit")
        ax.set_ylabel("Inter-block T-rotations")
        ax.set_title("Figure 1 — SA placement reduces inter-block communication")
        ax.set_xticks(x + width)
        ax.set_xticklabels([_display(c) for c in circuits])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

    if not any_data:
        print("[fig1] WARNING: no data found — figure will be empty")
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "fig_paper_1_inter_block.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig1] saved -> {path}")
    return fig


# ── Figure 2: robustness (box plots over 30 seeds) ───────────────────────────

def build_fig2(circuit="qft_100_approx", save=True):
    """
    Box plots of logical_depth over 30 SA seeds (Config E).
    Compares SA+Greedy and SA+CPSAT boxes against single-seed baseline bars.
    """
    configs_box = [("sa", "greedy_critical"), ("sa", "cpsat")]
    configs_bar = [("random", "greedy_critical"), ("random", "cpsat")]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        # Box plots (SA, 30 seeds)
        box_data = []
        box_labels = []
        for mapping, scheduler in configs_box:
            recs = load_results(circuit=circuit, mapping=mapping, scheduler=scheduler)
            depths = [r["logical_depth"] for r in recs if r.get("logical_depth", 0) > 0]
            if depths:
                box_data.append(depths)
                box_labels.append(f"SA+{SCHEDULER_LABELS[scheduler]}")

        # Baseline bars (single seed, shown as horizontal lines)
        baseline_lines = []
        for mapping, scheduler in configs_bar:
            rec = _single(circuit, mapping, scheduler)
            if rec and rec.get("logical_depth", 0) > 0:
                baseline_lines.append((
                    rec["logical_depth"],
                    f"{MAPPING_LABELS[mapping]}+{SCHEDULER_LABELS[scheduler]}",
                    CONFIG_COLORS[(mapping, scheduler)],
                ))

        positions = list(range(1, len(box_data) + 1))
        if box_data:
            bp = ax.boxplot(
                box_data, positions=positions, patch_artist=True,
                widths=0.4, medianprops=dict(color="black", linewidth=1.5),
            )
            box_colors = ["#1f77b4", "#2ca02c"]
            for patch, color in zip(bp["boxes"], box_colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(box_labels)
        else:
            print("[fig2] WARNING: no SA multi-seed data — run Config E first")

        for depth, label, color in baseline_lines:
            ax.axhline(depth, linestyle="--", color=color, linewidth=1.2, label=label, alpha=0.9)

        ax.set_ylabel("Logical depth")
        ax.set_title(f"Figure 2 — SA robustness over 30 seeds ({_display(circuit)})")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "fig_paper_2_robustness.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig2] saved -> {path}")
    return fig


# ── Figure 3: end-to-end logical depth ───────────────────────────────────────

def build_fig3(circuits=None, save=True):
    """
    Grouped bar chart of logical_depth for all 4 configs across circuits.
    Demonstrates the compound benefit of SA placement + CP-SAT scheduling.
    """
    if circuits is None:
        circuits = ["gf2_16_mult", "qft_100_approx"]

    configs = [
        ("random", "greedy_critical"),
        ("random", "cpsat"),
        ("sa",     "greedy_critical"),
        ("sa",     "cpsat"),
    ]

    n_cfg    = len(configs)
    x        = np.arange(len(circuits))
    width    = 0.18
    offsets  = np.linspace(-(n_cfg - 1) / 2, (n_cfg - 1) / 2, n_cfg) * width

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 4))

        any_data = False
        for cfg, offset in zip(configs, offsets):
            mapping, scheduler = cfg
            vals = []
            for circuit in circuits:
                rec = _single(circuit, mapping, scheduler)
                vals.append(rec["logical_depth"] if rec else 0)
            if any(v > 0 for v in vals):
                any_data = True
            ax.bar(
                x + offset, vals, width,
                label=CONFIG_LABELS[cfg],
                color=CONFIG_COLORS[cfg], alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )

        ax.set_xlabel("Benchmark circuit")
        ax.set_ylabel("Logical depth")
        ax.set_title("Figure 3 — End-to-end logical depth by config")
        ax.set_xticks(x)
        ax.set_xticklabels([_display(c) for c in circuits])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend(ncol=2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

    if not any_data:
        print("[fig3] WARNING: no data found — run Phase 1 first")
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "fig_paper_3_depth.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig3] saved -> {path}")
    return fig


# ── Figure 4: scheduler tradeoff scatter ─────────────────────────────────────

def build_fig4(circuits=None, save=True):
    """
    Scatter plot: compile_time_sec (x) vs logical_depth (y).
    Each point is one (circuit, placement, scheduler, seed=42) run.
    Shows that CP-SAT costs more time but achieves lower depth.
    """
    if circuits is None:
        circuits = ["gf2_16_mult", "qft_100_approx"]

    configs = [
        ("random", "greedy_critical"),
        ("random", "cpsat"),
        ("sa",     "greedy_critical"),
        ("sa",     "cpsat"),
    ]
    markers = {"gf2_16_mult": "o", "qft_100_approx": "s", "random_ct_500q_10k": "^"}

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4))

        any_data = False
        plotted_configs = set()
        for circuit in circuits:
            marker = markers.get(circuit, "D")
            for cfg in configs:
                mapping, scheduler = cfg
                rec = _single(circuit, mapping, scheduler)
                if rec and rec.get("logical_depth", 0) > 0 and rec.get("compile_time_sec", 0) > 0:
                    any_data = True
                    label = CONFIG_LABELS[cfg] if cfg not in plotted_configs else None
                    ax.scatter(
                        rec["compile_time_sec"],
                        rec["logical_depth"],
                        color=CONFIG_COLORS[cfg],
                        marker=marker,
                        s=70, alpha=0.85,
                        label=label,
                        zorder=3,
                    )
                    ax.annotate(
                        _display(circuit),
                        (rec["compile_time_sec"], rec["logical_depth"]),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 2), textcoords="offset points",
                    )
                    plotted_configs.add(cfg)

        ax.set_xlabel("Compile time (s)")
        ax.set_ylabel("Logical depth")
        ax.set_title("Figure 4 — Scheduler depth–time tradeoff")
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)
        fig.tight_layout()

    if not any_data:
        print("[fig4] WARNING: no data found — run Phase 1 first")
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "fig_paper_4_scatter.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig4] saved -> {path}")
    return fig


# ── Figure 5: scaling sweep ───────────────────────────────────────────────────

def build_fig5(save=True):
    """
    Line plot of depth reduction % vs number of qLDPC blocks.
    reduction = (Rand+Greedy depth - SA+CPSAT depth) / (Rand+Greedy depth) × 100
    Uses QFT scaling circuits from Phase 2.
    """
    scaling_circuits = [
        "qft_22_approx", "qft_33_approx", "qft_44_approx",
        "qft_66_approx", "qft_99_approx",
    ]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        blocks_list = []
        reductions  = []

        for circuit in scaling_circuits:
            baseline = _single(circuit, "random", "greedy_critical")
            best     = _single(circuit, "sa",     "cpsat")
            if not baseline or not best:
                continue
            if baseline["logical_depth"] == 0:
                continue
            n_blocks   = baseline["n_blocks"]
            reduction  = (
                (baseline["logical_depth"] - best["logical_depth"])
                / baseline["logical_depth"]
            ) * 100
            blocks_list.append(n_blocks)
            reductions.append(reduction)

        if blocks_list:
            ax.plot(blocks_list, reductions, "o-", color="#2ca02c",
                    linewidth=2, markersize=6, label="SA+CPSAT vs Rand+Greedy")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Number of qLDPC blocks")
            ax.set_ylabel("Depth reduction (%)")
            ax.set_title("Figure 5 — SA depth reduction vs. scale")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.legend()
            ax.grid(linestyle="--", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "No scaling data yet\n(Run Phase 2)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Figure 5 — SA depth reduction vs. scale (no data)")

        fig.tight_layout()

    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        path = os.path.join(FIG_DIR, "fig_paper_5_scaling.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig5] saved -> {path}")
    return fig


# ── Table 1: SC baseline vs qLDPC comparison ─────────────────────────────────

def _parse_sc_stats(stats_file: str) -> dict:
    """
    Parse lsqecc stats output file.
    Looks for lines like:
        Slices: 1234
        Qubit tiles: 567
    Returns {"sc_depth": int, "sc_tiles": int} or empty dict on failure.
    """
    result = {}
    if not os.path.exists(stats_file):
        return result
    with open(stats_file) as f:
        for line in f:
            low = line.lower().strip()
            if "slices" in low or "time steps" in low or "depth" in low:
                parts = low.split()
                for part in parts:
                    if part.isdigit():
                        result["sc_depth"] = int(part)
                        break
            if "qubit tiles" in low or "qubits" in low:
                parts = low.split()
                for part in parts:
                    if part.isdigit():
                        result["sc_tiles"] = int(part)
                        break
    return result


def print_table1(circuits=None):
    """
    Print Table 1: SC baseline vs qLDPC (Rand+Greedy) vs qLDPC+SA+CPSAT.
    """
    if circuits is None:
        circuits = ["gf2_16_mult", "qft_100_approx"]

    D2 = 144   # physical qubits per logical qubit for [[144,12,12]] Gross code

    header = (
        f"{'Circuit':<22} {'Qubits':>7} {'T-count':>8} "
        f"{'SC depth':>10} {'SC phys':>10} "
        f"{'qLDPC naive':>12} {'qLDPC SA+CP':>12} {'Reduction':>10}"
    )
    print()
    print("Table 1 — Surface code vs qLDPC compiler comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for circuit in circuits:
        # qLDPC results
        naive_rec = _single(circuit, "random", "sequential")
        best_rec  = _single(circuit, "sa", "cpsat")

        n_qubits = naive_rec["n_qubits"] if naive_rec else "?"
        t_count  = naive_rec["t_count"]  if naive_rec else "?"
        n_blocks = naive_rec["n_blocks"] if naive_rec else "?"
        naive_depth = naive_rec["logical_depth"] if naive_rec else "?"
        best_depth  = best_rec["logical_depth"]  if best_rec  else "?"

        # SC baseline (parsed from lsqecc stats)
        sc_file  = os.path.join(SC_DIR, f"{circuit}_stats.txt")
        sc_stats = _parse_sc_stats(sc_file)
        sc_depth = sc_stats.get("sc_depth", "—")
        sc_tiles = sc_stats.get("sc_tiles", "—")
        sc_phys  = f"{sc_tiles * D2:,}" if isinstance(sc_tiles, int) else "—"

        reduction = "—"
        if isinstance(naive_depth, int) and isinstance(best_depth, int) and naive_depth > 0:
            reduction = f"{(naive_depth - best_depth) / naive_depth * 100:.1f}%"

        print(
            f"{_display(circuit):<22} {str(n_qubits):>7} {str(t_count):>8} "
            f"{str(sc_depth):>10} {sc_phys:>10} "
            f"{str(naive_depth):>12} {str(best_depth):>12} {reduction:>10}"
        )

    print("=" * len(header))
    print()
    print("SC phys = SC qubit tiles × 144 (physical qubits per logical for [[144,12,12]] Gross code)")
    print("qLDPC naive = random placement + greedy scheduler")
    print("qLDPC SA+CP = SA placement + CP-SAT scheduler (this work)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build paper figures from raw results")
    parser.add_argument("--all",    action="store_true", help="Build all figures")
    parser.add_argument("--fig",    type=int, action="append", metavar="N",
                        help="Build specific figure (1–5); can repeat")
    parser.add_argument("--table1", action="store_true", help="Print Table 1 to stdout")
    args = parser.parse_args()

    if not any([args.all, args.fig, args.table1]):
        parser.print_help()
        sys.exit(0)

    figs = set(args.fig or [])

    if args.all or 1 in figs:
        build_fig1()
    if args.all or 2 in figs:
        build_fig2()
    if args.all or 3 in figs:
        build_fig3()
    if args.all or 4 in figs:
        build_fig4()
    if args.all or 5 in figs:
        build_fig5()
    if args.all or args.table1:
        print_table1()


if __name__ == "__main__":
    main()
