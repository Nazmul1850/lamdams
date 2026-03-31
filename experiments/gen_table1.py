"""
Generate Table 1: SC vs qLDPC resource comparison.

SC physical qubits  = sc_tile_count × (2·d² - 1)    (rotated surface code, d=12)
qLDPC physical qubits = n_blocks × (288 + 103) + n_couplers × 24
  where:
    288          = data qubits per Gross-code [[288,12,12]] block
    103          = ancilla qubits per block (syndrome extraction)
    24           = bridge qubits per inter-block coupler
    n_couplers   = edges in the hardware grid topology

Depth:
  sc_depth      = T-count  (Litinski model: each T-gate = 1 lattice-surgery round)
  qldpc_depth   = logical depth under sequential (naive) scheduling
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

# ── Constants ─────────────────────────────────────────────────────────────────
CODE_DISTANCE   = 12          # surface code distance
PHYS_PER_TILE   = 2 * CODE_DISTANCE**2 - 1   # 287 for d=12 (rotated SC)
PHYS_PER_BLOCK  = 288         # data qubits in one [[288,12,12]] Gross block
ANCILLA_PER_BLOCK = 103       # ancilla qubits per block (syndrome extraction)
BRIDGE_PER_COUP = 24          # bridge physical qubits per inter-block coupler

RUNS_DIR    = "runs"
SC_BASELINE = "results/sc_baseline/sc_baseline.json"
OUT_PATH    = "results/table1/table1.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def grid_shape(n_blocks: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_blocks))
    rows = math.ceil(n_blocks / cols)
    return rows, cols


def n_grid_couplers(n_blocks: int) -> int:
    """Number of edges in the most-square grid for n_blocks."""
    rows, cols = grid_shape(n_blocks)
    # actual blocks rounded up to fill rectangle
    n_actual = rows * cols
    return rows * (cols - 1) + (rows - 1) * cols


def qldpc_physical_qubits(n_blocks: int) -> tuple[int, int, int]:
    """Return (total_phys, n_couplers, block_qubits)."""
    n_couplers  = n_grid_couplers(n_blocks)
    block_qubits = n_blocks * (PHYS_PER_BLOCK + ANCILLA_PER_BLOCK)
    bridge_qubits = n_couplers * BRIDGE_PER_COUP
    return block_qubits + bridge_qubits, n_couplers, block_qubits


# ── Load SC baseline ──────────────────────────────────────────────────────────

with open(SC_BASELINE) as f:
    sc_data = {r["circuit"]: r for r in json.load(f)}


# ── Load sequential-scheduler run results ─────────────────────────────────────

def load_run(circuit: str, placement: str, scheduler: str, seed: int) -> dict:
    path = os.path.join(RUNS_DIR, f"{circuit}_{placement}_{scheduler}_seed{seed}.json")
    with open(path) as f:
        return json.load(f)


# ── Build table rows ──────────────────────────────────────────────────────────

ENTRIES = [
    ("gf2_16_mult",    "random", "sequential", 42),
    ("qft_100_approx", "random", "sequential", 42),
]

rows = []
for circuit, placement, scheduler, seed in ENTRIES:
    sc  = sc_data[circuit]
    run = load_run(circuit, placement, scheduler, seed)

    n_blocks   = run["n_blocks"]
    n_qubits   = run["n_qubits"]
    t_count    = run["t_count"]
    qldpc_phys, n_couplers, block_qubits = qldpc_physical_qubits(n_blocks)

    # SC data-only tiles: layout cols = 2*n_qubits (data) + T_count+1 (magic)
    # Exclude magic-state patches so we compare only logical-data qubit footprints.
    sc_data_tiles = 6 * n_qubits                   # = 3 rows × 2·n_qubits cols
    sc_magic_tiles = 3 * (t_count + 1)             # = 3 rows × (T_count+1) cols
    sc_phys_data  = sc_data_tiles  * PHYS_PER_TILE
    sc_phys_total = sc["sc_tile_count"] * PHYS_PER_TILE  # kept for reference

    sc_depth    = sc["sc_logical_depth"]
    qldpc_depth = run["logical_depth"]

    rows.append({
        "circuit":                  circuit,
        "n_logical_qubits":         n_qubits,
        "t_count":                  t_count,

        # Surface code columns (data patches only, magic excluded)
        "sc_data_tiles":            sc_data_tiles,
        "sc_magic_tiles":           sc_magic_tiles,
        "sc_total_tiles":           sc["sc_tile_count"],
        "sc_physical_qubits":       sc_phys_data,   # data-only, for fair comparison
        "sc_physical_qubits_total": sc_phys_total,  # including magic, for reference
        "sc_depth":                 sc_depth,
        "code_distance_d":          CODE_DISTANCE,
        "phys_per_tile":            PHYS_PER_TILE,

        # qLDPC columns
        "n_blocks":                 n_blocks,
        "n_couplers":               n_couplers,
        "qldpc_block_qubits":       block_qubits,
        "qldpc_bridge_qubits":      n_couplers * BRIDGE_PER_COUP,
        "qldpc_physical_qubits":    qldpc_phys,
        "qldpc_naive_depth":        qldpc_depth,
        "qldpc_placement":          placement,
        "qldpc_scheduler":          scheduler,
        "seed":                     seed,

        # Derived ratios  (SC data-only vs qLDPC)
        "qubit_reduction":          round(sc_phys_data / qldpc_phys, 2),
        "depth_overhead":           round(qldpc_depth / sc_depth, 2),
    })

# ── Write output ──────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
output = {
    "description": "Table 1: SC vs qLDPC physical qubit and depth comparison",
    "methodology": {
        "sc_physical_qubits":    "data-only tiles × (2·d² - 1); data tiles = 6·n_qubits (magic patches excluded for fair comparison)",
        "sc_data_tiles":         "3 rows × 2·n_qubits cols  (SimplePreDistilledStates layout)",
        "sc_magic_tiles":        "3 rows × (T_count + 1) cols  (excluded from qubit comparison)",
        "qldpc_physical_qubits": "n_blocks × (288 + 103) + n_couplers × 24",
        "sc_depth":              "T-count (Litinski transform: each T = 1 lattice-surgery round)",
        "qldpc_naive_depth":     "logical depth under sequential (one-at-a-time) scheduling",
        "code_distance":         CODE_DISTANCE,
    },
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

# ── Print table ───────────────────────────────────────────────────────────────

print(f"\n{'Circuit':<22} {'SC Phys Q':>10} {'qLDPC Phys Q':>13} {'Qubit Red.':>11} "
      f"{'SC Depth':>9} {'qLDPC Seq D':>12} {'Depth OH':>9}")
print(f"{'':22} {'(data only)':>10} {'':>13}")
print("-" * 92)
for r in rows:
    print(f"{r['circuit']:<22} {r['sc_physical_qubits']:>10,} {r['qldpc_physical_qubits']:>13,} "
          f"{r['qubit_reduction']:>10.1f}x {r['sc_depth']:>9,} {r['qldpc_naive_depth']:>12,} "
          f"{r['depth_overhead']:>8.1f}x")
    print(f"  SC: {r['sc_data_tiles']} data tiles + {r['sc_magic_tiles']} magic tiles (excluded)")

print(f"\nWritten → {OUT_PATH}")
