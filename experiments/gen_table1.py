"""
Generate Table 1: SC vs qLDPC resource comparison — all 5 configs.

Discovers circuits automatically from:
  - circuits/benchmarks/pbc/*_PBC.json
  - experiment_circuits/*.qasm  (if no PBC counterpart)

Configs shown per circuit:
  naive : random + sequential      (baseline)
  A     : random + greedy_critical
  B     : random + cpsat
  C     : sa     + greedy_critical
  D     : sa     + cpsat

A row is rendered if SC baseline + naive run are present.
All other configs are optional — missing ones show "—".

Usage:
    python experiments/gen_table1.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

# ── Constants ─────────────────────────────────────────────────────────────────
CODE_DISTANCE     = 12
PHYS_PER_TILE     = 2 * CODE_DISTANCE**2 - 1   # 287 for d=12
PHYS_PER_BLOCK    = 288
ANCILLA_PER_BLOCK = 103
BRIDGE_PER_COUP   = 24

SEED = 42

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR     = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
EXP_DIR     = os.path.join(_ROOT, "experiment_circuits")
RUNS_DIR    = os.path.join(_ROOT, "runs")
SC_BASELINE = os.path.join(_ROOT, "results", "sc_baseline", "sc_baseline.json")
OUT_PATH    = os.path.join(_ROOT, "results", "table1", "table1.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circuit_name_from_pbc(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith("_PBC") else stem


def discover_circuits() -> list[str]:
    """Return sorted list of all circuit names present in pbc/ or experiment_circuits/."""
    names: set[str] = set()
    if os.path.isdir(PBC_DIR):
        for f in os.listdir(PBC_DIR):
            if f.endswith(".json"):
                names.add(_circuit_name_from_pbc(f))
    if os.path.isdir(EXP_DIR):
        for f in os.listdir(EXP_DIR):
            if f.endswith(".qasm"):
                names.add(os.path.splitext(f)[0])
    return sorted(names)


def run_path(circuit: str, mapping: str, scheduler: str) -> str:
    return os.path.join(RUNS_DIR, f"{circuit}_{mapping}_{scheduler}_seed{SEED}.json")


def load_run(circuit: str, mapping: str, scheduler: str) -> dict | None:
    p = run_path(circuit, mapping, scheduler)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def grid_shape(n_blocks: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_blocks))
    rows = math.ceil(n_blocks / cols)
    return rows, cols


def n_grid_couplers(n_blocks: int) -> int:
    rows, cols = grid_shape(n_blocks)
    return rows * (cols - 1) + (rows - 1) * cols


def qldpc_physical_qubits(n_blocks: int) -> tuple[int, int, int]:
    """Return (total_phys, n_couplers, block_qubits)."""
    n_couplers    = n_grid_couplers(n_blocks)
    block_qubits  = n_blocks * (PHYS_PER_BLOCK + ANCILLA_PER_BLOCK)
    bridge_qubits = n_couplers * BRIDGE_PER_COUP
    return block_qubits + bridge_qubits, n_couplers, block_qubits


# ── Load SC baseline ──────────────────────────────────────────────────────────

if not os.path.exists(SC_BASELINE):
    print(f"ERROR: SC baseline not found at {SC_BASELINE}")
    print("  Run: python experiments/sc_baseline.py")
    raise SystemExit(1)

with open(SC_BASELINE) as f:
    sc_data = {r["circuit"]: r for r in json.load(f)}


# ── Discover circuits ─────────────────────────────────────────────────────────

all_circuits = discover_circuits()

# ── Status report ─────────────────────────────────────────────────────────────

print("=" * 90)
print("TABLE 1 — STATUS REPORT")
print("=" * 90)
print(f"\n  {'Circuit':<28}  {'SC':^4}  {'naive':^5}  {'A:R+Grd':^7}  {'B:R+CP':^6}  {'C:SA+Grd':^8}  {'D:SA+CP':^7}  ready?")
print("  " + "-" * 80)

for c in all_circuits:
    has_sc         = c in sc_data
    naive_run      = load_run(c, "random", "sequential")
    rg_run         = load_run(c, "random", "greedy_critical")
    rc_run         = load_run(c, "random", "cpsat")
    sa_greedy_run  = load_run(c, "sa",     "greedy_critical")
    sa_cpsat_run   = load_run(c, "sa",     "cpsat")

    def sym(r): return "✓" if r else "—"

    sc_sym    = "✓" if has_sc    else "✗"
    naive_sym = "✓" if naive_run else "✗"
    ready     = "✓ ROW" if (has_sc and naive_run) else "✗ skip"

    print(f"  {c:<28}  {sc_sym:^4}  {naive_sym:^5}  {sym(rg_run):^7}  {sym(rc_run):^6}  {sym(sa_greedy_run):^8}  {sym(sa_cpsat_run):^7}  {ready}")

# Missing commands
missing_sc       = [c for c in all_circuits if c not in sc_data]
missing_naive    = [c for c in all_circuits if not load_run(c, "random", "sequential")]
missing_rg       = [c for c in all_circuits if not load_run(c, "random", "greedy_critical")]
missing_rc       = [c for c in all_circuits if not load_run(c, "random", "cpsat")]
missing_sa_greedy = [c for c in all_circuits if not load_run(c, "sa", "greedy_critical")]
missing_sa       = [c for c in all_circuits if not load_run(c, "sa", "cpsat")]

if missing_sc:
    print("\n[TO DO — SC baseline]")
    for c in missing_sc:
        print(f"  python experiments/sc_baseline.py --circuit {c}")

if missing_naive:
    print("\n[TO DO — naive (random+sequential)]")
    for c in missing_naive:
        print(f"  python experiments/run_experiment.py --circuit {c} --mapping random --scheduler sequential --seed {SEED}")

if missing_rg:
    print("\n[TO DO — config A (random+greedy)]")
    for c in missing_rg:
        print(f"  python experiments/run_experiment.py --circuit {c} --mapping random --scheduler greedy_critical --seed {SEED}")

if missing_rc:
    print("\n[TO DO — config B (random+cpsat)]")
    for c in missing_rc:
        print(f"  python experiments/run_experiment.py --circuit {c} --mapping random --scheduler cpsat --seed {SEED}")

if missing_sa_greedy:
    print("\n[TO DO — config C (sa+greedy)]")
    for c in missing_sa_greedy:
        print(f"  python experiments/run_experiment.py --circuit {c} --mapping sa --scheduler greedy_critical --seed {SEED}")

if missing_sa:
    print("\n[TO DO — config D (sa+cpsat)]")
    for c in missing_sa:
        print(f"  python experiments/run_experiment.py --circuit {c} --mapping sa --scheduler cpsat --seed {SEED}")

print()


# ── Build table rows ──────────────────────────────────────────────────────────

rows = []
for circuit in all_circuits:
    if circuit not in sc_data:
        continue
    naive_run = load_run(circuit, "random", "sequential")
    if naive_run is None:
        continue

    sc            = sc_data[circuit]
    rg_run        = load_run(circuit, "random", "greedy_critical")
    rc_run        = load_run(circuit, "random", "cpsat")
    sa_greedy_run = load_run(circuit, "sa",     "greedy_critical")
    sa_run        = load_run(circuit, "sa",     "cpsat")

    # Use sc_baseline as authoritative t_count — sequential runs on PBC circuits
    # record t_count=0 because the sequential scheduler doesn't recount T-gates.
    t_count  = sc["t_count"]
    n_qubits = sc["n_qubits"]
    n_blocks = naive_run["n_blocks"]

    qldpc_phys, n_couplers, block_qubits = qldpc_physical_qubits(n_blocks)

    sc_data_tiles  = 6 * n_qubits
    sc_magic_tiles = 3 * (t_count + 1)
    sc_phys_data   = sc_data_tiles * PHYS_PER_TILE
    sc_phys_total  = sc["sc_tile_count"] * PHYS_PER_TILE

    sc_depth        = sc["sc_logical_depth"]
    naive_depth     = naive_run["logical_depth"]
    rg_depth        = rg_run["logical_depth"]        if rg_run        else None
    rc_depth        = rc_run["logical_depth"]        if rc_run        else None
    sa_greedy_depth = sa_greedy_run["logical_depth"] if sa_greedy_run else None
    sa_depth        = sa_run["logical_depth"]        if sa_run        else None

    def _pct(d):
        if d is None: return None
        return round((naive_depth - d) / naive_depth * 100, 1)

    qubit_reduction = round(sc_phys_data / qldpc_phys, 2)
    naive_depth_oh  = round(naive_depth / sc_depth, 2) if sc_depth else None

    rows.append({
        "circuit":                  circuit,
        "n_logical_qubits":         n_qubits,
        "t_count":                  t_count,
        "n_blocks":                 n_blocks,
        "n_couplers":               n_couplers,

        # SC columns
        "sc_data_tiles":            sc_data_tiles,
        "sc_magic_tiles":           sc_magic_tiles,
        "sc_total_tiles":           sc["sc_tile_count"],
        "sc_physical_qubits":       sc_phys_data,
        "sc_physical_qubits_total": sc_phys_total,
        "sc_depth":                 sc_depth,
        "code_distance_d":          CODE_DISTANCE,
        "phys_per_tile":            PHYS_PER_TILE,

        # qLDPC columns
        "qldpc_block_qubits":       block_qubits,
        "qldpc_bridge_qubits":      n_couplers * BRIDGE_PER_COUP,
        "qldpc_physical_qubits":    qldpc_phys,

        # All 5 configs
        "qldpc_naive_depth":        naive_depth,           # naive: random+sequential
        "qldpc_rg_depth":           rg_depth,              # A: random+greedy
        "qldpc_rc_depth":           rc_depth,              # B: random+cpsat
        "qldpc_sa_greedy_depth":    sa_greedy_depth,       # C: sa+greedy
        "qldpc_sa_cpsat_depth":     sa_depth,              # D: sa+cpsat

        # Derived — all vs naive
        "qubit_reduction":          qubit_reduction,
        "naive_depth_overhead":     naive_depth_oh,
        "rg_depth_reduction_pct":   _pct(rg_depth),
        "rc_depth_reduction_pct":   _pct(rc_depth),
        "sa_greedy_depth_reduction_pct": _pct(sa_greedy_depth),
        "sa_cpsat_depth_reduction_pct":  _pct(sa_depth),
    })


# ── Write JSON ────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
output = {
    "description": "Table 1: SC vs qLDPC physical qubit and depth comparison",
    "methodology": {
        "sc_physical_qubits":    "data-only tiles × (2·d²-1); data tiles = 6·n_qubits",
        "sc_data_tiles":         "3 rows × 2·n_qubits cols  (SimplePreDistilledStates layout)",
        "sc_magic_tiles":        "3 rows × (T_count+1) cols  (excluded from qubit comparison)",
        "qldpc_physical_qubits": "n_blocks × (288+103) + n_couplers × 24",
        "sc_depth":              "T-count (Litinski transform, each T = 1 lattice-surgery round)",
        "configs": {
            "naive": "random placement + sequential scheduler",
            "A":     "random placement + greedy_critical scheduler",
            "B":     "random placement + CP-SAT scheduler",
            "C":     "SA placement + greedy_critical scheduler",
            "D":     "SA placement + CP-SAT scheduler",
        },
        "depth_reduction_pct":   "(naive - config_depth) / naive × 100",
        "code_distance":         CODE_DISTANCE,
        "t_count_source":        "sc_baseline.json (authoritative; sequential runs do not recount)",
    },
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)


# ── Print table ───────────────────────────────────────────────────────────────

def _fmt(v, fmt=",") -> str:
    if v is None:
        return "—"
    if fmt == "pct":
        return f"{v:.1f}%"
    if fmt == "x":
        return f"{v:.1f}x"
    return f"{v:{fmt}}"


if rows:
    # ── Header ────────────────────────────────────────────────────────────────
    W = 160
    print("=" * W)
    print(
        f"  {'Circuit':<22} {'Q':>4} {'B':>3} {'T':>6}  "
        f"{'SC PhysQ':>9} {'SC D':>6}  "
        f"{'qLDPC PhysQ':>11}  "
        f"{'Naive':>7} {'A:R+Grd':>8} {'B:R+CP':>7} {'C:SA+Grd':>9} {'D:SA+CP':>8}  "
        f"{'Q×':>5} {'NaiveOH':>7}  "
        f"{'A%':>6} {'B%':>6} {'C%':>6} {'D%':>6}"
    )
    print("  " + "-" * (W - 2))
    for r in rows:
        print(
            f"  {r['circuit']:<22} {r['n_logical_qubits']:>4} {r['n_blocks']:>3} {r['t_count']:>6,}  "
            f"{r['sc_physical_qubits']:>9,} {r['sc_depth']:>6,}  "
            f"{r['qldpc_physical_qubits']:>11,}  "
            f"{r['qldpc_naive_depth']:>7,} "
            f"{_fmt(r['qldpc_rg_depth']):>8} "
            f"{_fmt(r['qldpc_rc_depth']):>7} "
            f"{_fmt(r['qldpc_sa_greedy_depth']):>9} "
            f"{_fmt(r['qldpc_sa_cpsat_depth']):>8}  "
            f"{_fmt(r['qubit_reduction'], 'x'):>5} "
            f"{_fmt(r['naive_depth_overhead'], 'x'):>7}  "
            f"{_fmt(r['rg_depth_reduction_pct'], 'pct'):>6} "
            f"{_fmt(r['rc_depth_reduction_pct'], 'pct'):>6} "
            f"{_fmt(r['sa_greedy_depth_reduction_pct'], 'pct'):>6} "
            f"{_fmt(r['sa_cpsat_depth_reduction_pct'], 'pct'):>6}"
        )
    print("=" * W)

    # ── Summary counts ────────────────────────────────────────────────────────
    counts = {
        "A": sum(1 for r in rows if r["qldpc_rg_depth"] is not None),
        "B": sum(1 for r in rows if r["qldpc_rc_depth"] is not None),
        "C": sum(1 for r in rows if r["qldpc_sa_greedy_depth"] is not None),
        "D": sum(1 for r in rows if r["qldpc_sa_cpsat_depth"] is not None),
    }
    print(f"\n  {len(rows)} circuit(s)  |  "
          + "  |  ".join(f"Config {k}: {v}/{len(rows)}" for k, v in counts.items()))

    # ── Mean reduction row (only configs with data) ───────────────────────────
    def _mean_pct(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{sum(vals)/len(vals):.1f}%" if vals else "—"

    print(f"\n  Mean depth reduction vs naive:")
    print(f"    A (Rand+Greedy): {_mean_pct('rg_depth_reduction_pct')}")
    print(f"    B (Rand+CPSAT):  {_mean_pct('rc_depth_reduction_pct')}")
    print(f"    C (SA+Greedy):   {_mean_pct('sa_greedy_depth_reduction_pct')}")
    print(f"    D (SA+CPSAT):    {_mean_pct('sa_cpsat_depth_reduction_pct')}")
else:
    print("No rows to render — run sc_baseline.py and naive runs first.")

print(f"\nWritten → {OUT_PATH}")
