"""
Generate Figure 3: Grouped bar chart of logical depth (normalized to SC baseline).

Y-axis = depth / sc_baseline_depth, so SC baseline = 1.0 for every circuit.
Both circuits are on the same scale; each bar is annotated with the absolute
depth and the overhead multiple (e.g. "19,352 / 10.8× SC").

Reads directly from runs/ JSON files. Re-run this script when missing data arrives.

Usage:
    python experiments/gen_fig3.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR  = os.path.join(_ROOT, "runs")
OUT_DIR   = os.path.join(_ROOT, "results", "figures")
OUT_PNG   = os.path.join(OUT_DIR, "fig3_depth_comparison.png")
OUT_CSV   = os.path.join(_ROOT, "results", "fig_3_depth.csv")
SC_BASE   = os.path.join(_ROOT, "results", "sc_baseline", "sc_baseline.json")

# ── Config definitions ─────────────────────────────────────────────────────────

CONFIGS = [
    ("random", "greedy_critical", "Random + Greedy",  "#d62728"),  # red
    ("random", "cpsat",           "Random + CP-SAT",  "#ff7f0e"),  # orange
    ("sa",     "greedy_critical", "SA + Greedy",       "#1f77b4"),  # blue
    ("sa",     "cpsat",           "SA + CP-SAT",       "#2ca02c"),  # green
]

CIRCUITS = [
    ("gf2_16_mult",    r"$\mathrm{GF}(2^{16})$ mult" + "\n(48 qubits, 6 blocks)"),
    ("qft_100_approx", "QFT-100 (approx)\n(100 qubits, 12 blocks)"),
]

# ── Load helpers ──────────────────────────────────────────────────────────────

def load_run(circuit: str, placement: str, scheduler: str, seed: int = 42):
    path = os.path.join(RUNS_DIR, f"{circuit}_{placement}_{scheduler}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sc_depth(circuit: str) -> int:
    with open(SC_BASE) as f:
        sc_data = {r["circuit"]: r for r in json.load(f)}
    return sc_data[circuit]["sc_logical_depth"]


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(depths: dict) -> bool:
    """
    depths[circuit][(placement, scheduler)] = depth | None
    Checks:
      1. random+greedy >= sa+greedy   (SA placement is better)
      2. random+cpsat  >= sa+cpsat    (SA placement is better)
      3. cpsat <= greedy within same placement
    """
    ok = True
    print("\n=== CONFIG SANITY CHECK ===")
    for circuit, label in CIRCUITS:
        d = depths[circuit]
        rg = d.get(("random", "greedy_critical"))
        rc = d.get(("random", "cpsat"))
        sg = d.get(("sa",     "greedy_critical"))
        sc = d.get(("sa",     "cpsat"))

        print(f"{circuit}:")
        for (pl, sched), val in [
            (("random", "greedy_critical"), rg),
            (("random", "cpsat"),           rc),
            (("sa",     "greedy_critical"), sg),
            (("sa",     "cpsat"),           sc),
        ]:
            print(f"  {pl}+{sched}: {val if val is not None else 'MISSING'}")

        checks = [
            ("random+greedy >= sa+greedy", rg, sg),
            ("random+cpsat  >= sa+cpsat",  rc, sc),
            ("random: cpsat <= greedy",    rc, rg, "le"),
            ("sa:     cpsat <= greedy",    sc, sg, "le"),
        ]
        for check in checks:
            name = check[0]
            a, b = check[1], check[2]
            mode = check[3] if len(check) > 3 else "ge"
            if a is None or b is None:
                print(f"  Check {name}: SKIP (data missing)")
                continue
            passed = (a >= b) if mode == "ge" else (a <= b)
            status = "PASS" if passed else "FAIL"
            print(f"  Check {name}: {status}  ({a} vs {b})")
            if not passed:
                ok = False
        print()
    print("===========================\n")
    return ok


# ── Figure generation ─────────────────────────────────────────────────────────

def make_figure(depths: dict, sc_depths: dict):
    """
    Y-axis: depth normalized to SC baseline (SC = 1.0).
    Both circuits share the same Y scale.
    Each bar is annotated with the absolute depth and overhead ratio (e.g. "19,352\n10.8× SC").
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    n_circuits = len(CIRCUITS)
    n_configs  = len(CONFIGS)
    bar_w      = 0.18
    group_gap  = 0.30
    group_w    = n_configs * bar_w + group_gap
    group_centers = np.arange(n_circuits) * group_w

    all_norm_vals = []

    # Draw bars (normalized heights)
    for ci, (placement, scheduler, label, color) in enumerate(CONFIGS):
        offsets = (ci - (n_configs - 1) / 2) * bar_w
        xpos    = group_centers + offsets

        for gi, (circuit, _) in enumerate(CIRCUITS):
            d    = depths[circuit].get((placement, scheduler))
            sc_d = sc_depths[circuit]
            if d is None:
                continue
            norm = d / sc_d
            all_norm_vals.append(norm)

            bar = ax.bar(
                xpos[gi], norm,
                width=bar_w,
                color=color,
                alpha=1.0,
                label=label if gi == 0 else "_nolegend_",
                zorder=3,
            )

            # Annotate every bar: absolute depth on top, overhead ratio below it
            ax.text(
                xpos[gi],
                norm * 1.25,
                f"{d:,}\n({norm:.1f}× SC)",
                ha="center", va="bottom",
                fontsize=7.5, color="black",
                linespacing=1.3,
            )

    # SC baseline: single horizontal dashed line at y=1, spanning full plot
    x_lo = group_centers[0]  - (n_configs / 2) * bar_w - 0.15
    x_hi = group_centers[-1] + (n_configs / 2) * bar_w + 0.15
    ax.axhline(y=1.0, xmin=0, xmax=1, linestyle="--", linewidth=1.6,
               color="black", zorder=5, label="SC baseline (Litinski, d=12)")

    # Axes — log scale
    ax.set_yscale("log")
    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in CIRCUITS], fontsize=11)
    ax.set_xlabel("Circuit", fontsize=11)
    ax.set_ylabel("Logical Depth (normalized to SC baseline, log scale)", fontsize=10)
    ax.set_title(
        "Logical Depth Overhead vs Surface Code Baseline\n"
        "Placement (Random / SA) × Scheduler (Greedy / CP-SAT)",
        fontsize=11,
    )

    # Y-axis ticks: show "1×", "2×", ... as multiples of SC baseline
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 1 else f"{x:.1f}×"
    ))
    # Extend Y-axis top to leave room for annotations
    if all_norm_vals:
        ax.set_ylim(bottom=0.5, top=max(all_norm_vals) * 4.5)

    # Grid: y only
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Legend — deduplicate (bar() adds one entry per bar; we only want one per config)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=10, framealpha=0.9)

    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {OUT_PNG}")
    return fig


# ── CSV output ────────────────────────────────────────────────────────────────

def write_csv(depths: dict, compile_times: dict):
    lines = ["circuit,placement,scheduler,logical_depth,compile_time_sec"]
    for circuit, _ in CIRCUITS:
        for placement, scheduler, _, _ in CONFIGS:
            d = depths[circuit].get((placement, scheduler))
            t = compile_times[circuit].get((placement, scheduler))
            lines.append(
                f"{circuit},{placement},{scheduler},"
                f"{d if d is not None else 'NA'},"
                f"{t if t is not None else 'NA'}"
            )
    with open(OUT_CSV, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"CSV saved   → {OUT_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all depths and compile times
    depths        = {circuit: {} for circuit, _ in CIRCUITS}
    compile_times = {circuit: {} for circuit, _ in CIRCUITS}
    sc_depths     = {}

    missing = []
    for circuit, _ in CIRCUITS:
        sc_depths[circuit] = load_sc_depth(circuit)
        for placement, scheduler, _, _ in CONFIGS:
            run = load_run(circuit, placement, scheduler)
            if run is not None:
                depths[circuit][(placement, scheduler)]        = run["logical_depth"]
                compile_times[circuit][(placement, scheduler)] = run.get("compile_time_sec")
            else:
                depths[circuit][(placement, scheduler)]        = None
                compile_times[circuit][(placement, scheduler)] = None
                missing.append(f"{circuit} {placement}+{scheduler}")

    if missing:
        print(f"WARNING: missing run data for: {', '.join(missing)}")
        print("  Those bars will be absent from the figure.")
        print("  Re-run this script once the data is available.\n")

    # Sanity check
    ok = sanity_check(depths)
    if not ok:
        print("ABORT: sanity check failed — fix data before generating figure.")
        sys.exit(1)

    make_figure(depths, sc_depths)
    write_csv(depths, compile_times)

    # Summary line
    for circuit, _ in CIRCUITS:
        sc_d  = sc_depths[circuit]
        cpsat = depths[circuit].get(("sa", "cpsat"))
        if cpsat:
            ratio = cpsat / sc_d
            print(
                f"  {circuit}: SA+CP-SAT depth={cpsat:,} vs SC baseline={sc_d:,} "
                f"— {ratio:.1f}× overhead remaining."
            )

    print("\nFigure 3 complete.")


if __name__ == "__main__":
    main()
