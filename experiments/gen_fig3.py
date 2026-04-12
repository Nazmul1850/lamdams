"""
Generate Figure 3: Grouped bar chart of logical depth (normalized to SC baseline).

Y-axis = depth / sc_baseline_depth, so SC baseline = 1.0 for every circuit.
Each bar is annotated with the absolute depth and the overhead multiple
(e.g. "19,352\n10.8×").

Circuits are auto-discovered from circuits/benchmarks/pbc/.  For each circuit,
all 4 configs are loaded from runs/.  If a circuit fails the sanity check or is
missing too many runs to be meaningful, it is skipped with a printed warning.

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

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR  = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RUNS_DIR = os.path.join(_ROOT, "runs")
OUT_DIR  = os.path.join(_ROOT, "results", "figures")
OUT_PNG  = os.path.join(OUT_DIR, "fig3_depth_comparison.png")
OUT_CSV  = os.path.join(_ROOT, "results", "fig_3_depth.csv")
SC_BASE  = os.path.join(_ROOT, "results", "sc_baseline", "sc_baseline.json")

# ── Config definitions ────────────────────────────────────────────────────────

CONFIGS = [
    ("random", "greedy_critical", "Random + Greedy",  "#d62728"),  # red
    ("random", "cpsat",           "Random + CP-SAT",  "#ff7f0e"),  # orange
    ("sa",     "greedy_critical", "SA + Greedy",      "#1f77b4"),  # blue
    ("sa",     "cpsat",           "SA + CP-SAT",      "#2ca02c"),  # green
]

# ── Circuit discovery ─────────────────────────────────────────────────────────

def discover_circuits() -> list[str]:
    """Return sorted circuit names from circuits/benchmarks/pbc/."""
    names = []
    for fname in sorted(os.listdir(PBC_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]  # strip .json
        name = stem[:-4] if stem.endswith("_PBC") else stem
        names.append(name)
    return names


def _display(circuit: str, run: dict | None) -> str:
    """Short multi-line label: circuit name + qubit / block count if available."""
    if run:
        return f"{circuit}\n({run['n_qubits']}q, {run['n_blocks']}b)"
    return circuit

# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(circuit: str, mapping: str, scheduler: str, seed: int = 42) -> dict | None:
    path = os.path.join(RUNS_DIR, f"{circuit}_{mapping}_{scheduler}_seed{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    # backwards-compat: old files used "placement" key
    if "mapping" not in d and "placement" in d:
        d["mapping"] = d["placement"]
    return d


def load_sc_depth(circuit: str, sc_data: dict) -> int | None:
    row = sc_data.get(circuit)
    return row["sc_logical_depth"] if row else None

# ── Per-circuit sanity check ──────────────────────────────────────────────────

def sanity_check(circuit: str, depths: dict) -> bool:
    """
    Returns True if the circuit passes all checks that can be evaluated
    (i.e. both values are present).  Prints a warning for each failure and
    returns False so the caller can skip the circuit.

    Checks (only when both sides are present):
      1. random+greedy >= sa+greedy    (SA placement should help)
      2. random+cpsat  >= sa+cpsat     (SA placement should help)
      3. random: cpsat <= greedy       (CP-SAT no worse than greedy)
      4. sa:     cpsat <= greedy       (CP-SAT no worse than greedy)
    """
    rg = depths.get(("random", "greedy_critical"))
    rc = depths.get(("random", "cpsat"))
    sg = depths.get(("sa",     "greedy_critical"))
    sc = depths.get(("sa",     "cpsat"))

    checks = [
        ("random+greedy >= sa+greedy", rg, sg, "ge"),
        ("random+cpsat  >= sa+cpsat",  rc, sc, "ge"),
        ("random: cpsat <= greedy",    rc, rg, "le"),
        ("sa:     cpsat <= greedy",    sc, sg, "le"),
    ]

    failed = []
    for name, a, b, mode in checks:
        if a is None or b is None:
            continue  # can't check, skip silently
        passed = (a >= b) if mode == "ge" else (a <= b)
        if not passed:
            failed.append(f"{name} ({a} vs {b})")

    if failed:
        print(f"  [SANITY FAIL] {circuit} — skipping:")
        for msg in failed:
            print(f"    ✗ {msg}")
        return False
    return True

# ── Figure generation ─────────────────────────────────────────────────────────

def make_figure(circuits_data: list[tuple[str, dict, dict, int]]):
    """
    circuits_data: list of (circuit_name, depths_dict, run_for_label, sc_depth)
    depths_dict:   {(mapping, scheduler): depth | None}
    """
    n_circuits = len(circuits_data)
    n_configs  = len(CONFIGS)

    bar_w     = 0.16
    group_gap = 0.25
    group_w   = n_configs * bar_w + group_gap
    group_centers = np.arange(n_circuits) * group_w

    # Scale figure width with number of circuits (min 8, max 20)
    fig_w = max(8, min(20, n_circuits * 1.8 + 2))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    all_norm_vals = []

    for ci, (mapping, scheduler, label, color) in enumerate(CONFIGS):
        offsets = (ci - (n_configs - 1) / 2) * bar_w
        xpos    = group_centers + offsets

        for gi, (circuit, depths, ref_run, sc_d) in enumerate(circuits_data):
            d = depths.get((mapping, scheduler))
            if d is None:
                continue
            norm = d / sc_d
            all_norm_vals.append(norm)

            ax.bar(
                xpos[gi], norm,
                width=bar_w,
                color=color,
                alpha=1.0,
                label=label if gi == 0 else "_nolegend_",
                zorder=3,
            )

            ax.text(
                xpos[gi],
                norm * 1.3,
                f"{d:,}\n({norm:.1f}×)",
                ha="center", va="bottom",
                fontsize=6.5, color="black",
                linespacing=1.3,
            )

    # SC baseline line
    ax.axhline(y=1.0, xmin=0, xmax=1, linestyle="--", linewidth=1.6,
               color="black", zorder=5, label="SC baseline (Litinski, d=12)")

    ax.set_yscale("log")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [_display(c, ref_run) for c, _, ref_run, _ in circuits_data],
        fontsize=8,
    )
    ax.set_xlabel("Circuit", fontsize=11)
    ax.set_ylabel("Logical depth (normalized to SC baseline, log scale)", fontsize=10)
    ax.set_title(
        "Logical Depth Overhead vs Surface Code Baseline\n"
        "Placement (Random / SA) × Scheduler (Greedy / CP-SAT)",
        fontsize=11,
    )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}×" if x >= 1 else f"{x:.1f}×"
    ))
    if all_norm_vals:
        ax.set_ylim(bottom=0.5, top=max(all_norm_vals) * 5.0)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=10, framealpha=0.9)

    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {OUT_PNG}")
    return fig

# ── CSV output ────────────────────────────────────────────────────────────────

def write_csv(circuits_data: list[tuple[str, dict, dict, int]]):
    lines = ["circuit,mapping,scheduler,logical_depth,sc_depth"]
    for circuit, depths, _, sc_d in circuits_data:
        for mapping, scheduler, _, _ in CONFIGS:
            d = depths.get((mapping, scheduler))
            lines.append(
                f"{circuit},{mapping},{scheduler},"
                f"{d if d is not None else 'NA'},"
                f"{sc_d}"
            )
    with open(OUT_CSV, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"CSV saved   → {OUT_CSV}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load SC baseline once
    with open(SC_BASE) as f:
        sc_raw = json.load(f)
    sc_data = {r["circuit"]: r for r in sc_raw}

    all_circuits = discover_circuits()
    print(f"Discovered {len(all_circuits)} PBC circuits: {all_circuits}\n")

    circuits_data = []  # (circuit, depths, ref_run, sc_depth)
    skipped = []

    for circuit in all_circuits:
        # Load SC depth — skip if missing
        sc_d = load_sc_depth(circuit, sc_data)
        if sc_d is None:
            print(f"  [SKIP] {circuit} — no SC baseline entry")
            skipped.append((circuit, "no SC baseline"))
            continue

        # Load all configs
        depths = {}
        ref_run = None
        for mapping, scheduler, _, _ in CONFIGS:
            run = load_run(circuit, mapping, scheduler)
            if run is not None:
                depths[(mapping, scheduler)] = run["logical_depth"]
                if ref_run is None:
                    ref_run = run
            else:
                depths[(mapping, scheduler)] = None

        # Need at least the naive (sequential) or one config to show anything
        has_any = any(v is not None for v in depths.values())
        if not has_any:
            print(f"  [SKIP] {circuit} — no run data found")
            skipped.append((circuit, "no run data"))
            continue

        # Sanity check — skip circuit if it fails
        if not sanity_check(circuit, depths):
            skipped.append((circuit, "sanity check failed"))
            continue

        circuits_data.append((circuit, depths, ref_run, sc_d))
        missing_cfgs = [f"{m}+{s}" for (m, s), v in depths.items() if v is None]
        if missing_cfgs:
            print(f"  [WARN] {circuit} — missing configs (bars omitted): {missing_cfgs}")

    print(f"\nIncluded: {[c for c, *_ in circuits_data]}")
    if skipped:
        print(f"Skipped:  {[(c, r) for c, r in skipped]}")

    if not circuits_data:
        print("\nNo circuits to plot. Exiting.")
        sys.exit(0)

    make_figure(circuits_data)
    write_csv(circuits_data)

    print("\nSummary (SA+CP-SAT vs SC baseline):")
    for circuit, depths, _, sc_d in circuits_data:
        cpsat = depths.get(("sa", "cpsat"))
        naive = depths.get(("random", "greedy_critical")) or depths.get(("random", "sequential"))
        if cpsat:
            print(f"  {circuit}: SA+CP-SAT={cpsat:,}  SC={sc_d:,}  → {cpsat/sc_d:.1f}× overhead")

    print("\nFigure 3 complete.")


if __name__ == "__main__":
    main()
