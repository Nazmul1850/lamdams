"""
Generate Table 1: SC vs qLDPC resource comparison — all 5 configs.

Data sources (in priority order per circuit):
  1. results/raw/{circuit}_seed42.json   — new unified format (grid + ring + all configs)
  2. runs/{circuit}_*_seed42.json        — legacy per-run files (fallback)

SC baseline always from results/sc_baseline/sc_baseline.json.

A row is rendered for every circuit that has SC baseline + at least naive depth.
All other configs show "—" when missing.

Topology shown: GRID (primary). Ring depths included in JSON output as _ring columns.

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
RAW_DIR     = os.path.join(_ROOT, "results", "raw")
SC_BASELINE = os.path.join(_ROOT, "results", "sc_baseline", "sc_baseline.json")
OUT_PATH    = os.path.join(_ROOT, "results", "table1", "table1.json")

# Config key → (mapping, scheduler) in the new raw format
RAW_CONFIGS = {
    "naive": ("random", "sequential"),
    "A":     ("random", "greedy_critical"),
    "B":     ("random", "cpsat"),
    "C":     ("sa",     "greedy_critical"),
    "D":     ("sa",     "cpsat"),
}
# Config key → legacy runs/ filename fragments
LEGACY_CONFIGS = {
    "naive": ("random", "sequential"),
    "A":     ("random", "greedy_critical"),
    "B":     ("random", "cpsat"),
    "C":     ("sa",     "greedy_critical"),
    "D":     ("sa",     "cpsat"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _circuit_name_from_pbc(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith("_PBC") else stem


def discover_circuits() -> list[str]:
    """Union of: raw/ files, pbc/ files, experiment_circuits/ files."""
    names: set[str] = set()
    # primary: raw/ directory
    if os.path.isdir(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(f"_seed{SEED}.json"):
                names.add(f[: -(len(f"_seed{SEED}.json"))])
    # secondary: pbc/ JSONs
    if os.path.isdir(PBC_DIR):
        for f in os.listdir(PBC_DIR):
            if f.endswith(".json"):
                names.add(_circuit_name_from_pbc(f))
    # tertiary: experiment_circuits/ .qasm
    if os.path.isdir(EXP_DIR):
        for f in os.listdir(EXP_DIR):
            if f.endswith(".qasm"):
                names.add(os.path.splitext(f)[0])
    return sorted(names)


def load_raw_file(circuit: str) -> dict | None:
    """Load results/raw/{circuit}_seed42.json (new unified format)."""
    p = os.path.join(RAW_DIR, f"{circuit}_seed{SEED}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_legacy_run(circuit: str, mapping: str, scheduler: str) -> dict | None:
    """Load runs/{circuit}_{mapping}_{scheduler}_seed42.json (old format)."""
    p = os.path.join(RUNS_DIR, f"{circuit}_{mapping}_{scheduler}_seed{SEED}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    if "mapping" not in d and "placement" in d:
        d["mapping"] = d["placement"]
    return d


def grid_shape(n_blocks: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n_blocks))
    rows = math.ceil(n_blocks / cols)
    return rows, cols


def n_grid_couplers(n_blocks: int) -> int:
    rows, cols = grid_shape(n_blocks)
    return rows * (cols - 1) + (rows - 1) * cols


def qldpc_physical_qubits(n_blocks: int, n_couplers: int | None = None) -> tuple[int, int, int]:
    """Return (total_phys, n_couplers, block_qubits)."""
    if n_couplers is None:
        n_couplers = n_grid_couplers(n_blocks)
    block_qubits  = n_blocks * (PHYS_PER_BLOCK + ANCILLA_PER_BLOCK)
    bridge_qubits = n_couplers * BRIDGE_PER_COUP
    return block_qubits + bridge_qubits, n_couplers, block_qubits


# ── Per-circuit data extraction ───────────────────────────────────────────────

class CircuitData:
    """Unified view of one circuit's depths across all configs and topologies."""

    def __init__(self, circuit: str, sc: dict):
        self.circuit  = circuit
        self.sc       = sc
        self.source   = None   # "raw" or "legacy"

        # depths[topology][config_key] = int | None
        self.depths:    dict[str, dict[str, int | None]]   = {"grid": {}, "ring": {}}
        self.n_blocks:  dict[str, int | None]               = {"grid": None, "ring": None}
        self.n_couplers: dict[str, int | None]              = {"grid": None, "ring": None}
        self.hw_label:  dict[str, str]                      = {"grid": "", "ring": ""}
        self.fill_rate: dict[str, float | None]             = {"grid": None, "ring": None}

        raw = load_raw_file(circuit)
        if raw is not None:
            self.source = "raw"
            self._load_raw(raw)
        else:
            self.source = "legacy"
            self._load_legacy()

    def _load_raw(self, raw: dict) -> None:
        for topo in ("grid", "ring"):
            if topo not in raw:
                continue
            td = raw[topo]
            self.n_blocks[topo]   = td.get("n_blocks")
            self.n_couplers[topo] = td.get("n_couplers")
            self.hw_label[topo]   = td.get("hw_label", "")
            self.fill_rate[topo]  = td.get("fill_rate")
            for cfg_key in RAW_CONFIGS:
                cfg = td.get("configs", {}).get(cfg_key)
                self.depths[topo][cfg_key] = cfg["logical_depth"] if cfg else None

    def _load_legacy(self) -> None:
        """Pull from runs/ flat files into grid slot only."""
        for cfg_key, (mapping, scheduler) in LEGACY_CONFIGS.items():
            rec = load_legacy_run(self.circuit, mapping, scheduler)
            depth = None
            if rec:
                depth = rec.get("logical_depth")
                if self.n_blocks["grid"] is None:
                    self.n_blocks["grid"] = rec.get("n_blocks")
                if self.n_couplers["grid"] is None:
                    self.n_couplers["grid"] = rec.get("n_couplers")
            self.depths["grid"][cfg_key] = depth

    def has_naive(self, topo: str = "grid") -> bool:
        return self.depths.get(topo, {}).get("naive") is not None

    def depth(self, cfg: str, topo: str = "grid") -> int | None:
        return self.depths.get(topo, {}).get(cfg)

    def grid_n_blocks(self) -> int | None:
        return self.n_blocks["grid"]

    def grid_n_couplers(self) -> int | None:
        return self.n_couplers["grid"]


# ── Load SC baseline ──────────────────────────────────────────────────────────

if not os.path.exists(SC_BASELINE):
    print(f"ERROR: SC baseline not found at {SC_BASELINE}")
    raise SystemExit(1)

with open(SC_BASELINE) as f:
    _raw_sc = json.load(f)
    # keep only best (latest) entry per circuit name
    sc_data: dict[str, dict] = {}
    for r in _raw_sc:
        c = r["circuit"]
        if c not in sc_data or r.get("timestamp", "") > sc_data[c].get("timestamp", ""):
            sc_data[c] = r


# ── Discover + load all circuits ──────────────────────────────────────────────

all_circuit_names = discover_circuits()
circuit_objects: list[CircuitData] = []
for name in all_circuit_names:
    if name not in sc_data:
        continue
    cd = CircuitData(name, sc_data[name])
    if cd.has_naive("grid"):
        circuit_objects.append(cd)


# ── Status report ─────────────────────────────────────────────────────────────

print("=" * 100)
print("TABLE 1 — STATUS REPORT")
print("=" * 100)
print(
    f"\n  {'Circuit':<30} {'Src':^5} {'SC':^4}  "
    f"{'naive':^5} {'A':^4} {'B':^4} {'C':^4} {'D':^4}  "
    f"{'ring-naive':^10} {'ring-D':^8}  ready?"
)
print("  " + "-" * 90)

all_names_with_sc = [n for n in all_circuit_names if n in sc_data]
for name in all_names_with_sc:
    cd   = CircuitData(name, sc_data[name])
    src  = cd.source or "—"

    def sym(v): return "✓" if v is not None else "—"

    row_ready = "✓ ROW" if cd.has_naive("grid") else "✗ skip"
    print(
        f"  {name:<30} {src:^5} ✓     "
        f"{sym(cd.depth('naive')):^5} "
        f"{sym(cd.depth('A')):^4} "
        f"{sym(cd.depth('B')):^4} "
        f"{sym(cd.depth('C')):^4} "
        f"{sym(cd.depth('D')):^4}  "
        f"{sym(cd.depth('naive','ring')):^10} "
        f"{sym(cd.depth('D','ring')):^8}  "
        f"{row_ready}"
    )

# To-do: circuits with SC baseline but no naive run
missing_naive = [n for n in all_names_with_sc
                 if not CircuitData(n, sc_data[n]).has_naive("grid")]
if missing_naive:
    print(f"\n[TO DO — missing naive grid run for {len(missing_naive)} circuit(s)]")
    for n in missing_naive:
        print(f"  python experiments/run_experiment.py --circuit {n} --all-configs --seed {SEED}")

print()


# ── Build table rows ──────────────────────────────────────────────────────────

rows = []
for cd in circuit_objects:
    sc       = cd.sc
    t_count  = sc["t_count"]
    n_qubits = sc["n_qubits"]

    nb = cd.grid_n_blocks()
    nc = cd.grid_n_couplers()
    if nb is None:
        continue

    qldpc_phys, n_couplers_calc, block_qubits = qldpc_physical_qubits(nb, nc)
    if nc is None:
        nc = n_couplers_calc

    sc_data_tiles  = 6 * n_qubits
    sc_magic_tiles = 3 * (t_count + 1)
    sc_phys_data   = sc_data_tiles * PHYS_PER_TILE
    sc_phys_total  = sc["sc_tile_count"] * PHYS_PER_TILE
    sc_depth       = sc["sc_logical_depth"]

    naive_d = cd.depth("naive")
    rg_d    = cd.depth("A")
    rc_d    = cd.depth("B")
    sag_d   = cd.depth("C")
    sa_d    = cd.depth("D")

    # Ring topology (optional)
    naive_d_ring = cd.depth("naive", "ring")
    rg_d_ring    = cd.depth("A",     "ring")
    rc_d_ring    = cd.depth("B",     "ring")
    sag_d_ring   = cd.depth("C",     "ring")
    sa_d_ring    = cd.depth("D",     "ring")

    def _pct(d, base=None):
        b = base if base is not None else naive_d
        if d is None or b is None or b == 0:
            return None
        return round((b - d) / b * 100, 1)

    qubit_reduction = round(sc_phys_data / qldpc_phys, 2) if qldpc_phys else None
    naive_depth_oh  = round(naive_d / sc_depth, 2) if sc_depth and naive_d else None

    rows.append({
        "circuit":            cd.circuit,
        "source":             cd.source,
        "n_logical_qubits":   n_qubits,
        "t_count":            t_count,
        "hw_label_grid":      cd.hw_label.get("grid", ""),
        "hw_label_ring":      cd.hw_label.get("ring", ""),
        "n_blocks":           nb,
        "n_couplers":         nc,
        "fill_rate_grid":     cd.fill_rate.get("grid"),
        "fill_rate_ring":     cd.fill_rate.get("ring"),

        # SC
        "sc_data_tiles":            sc_data_tiles,
        "sc_magic_tiles":           sc_magic_tiles,
        "sc_total_tiles":           sc["sc_tile_count"],
        "sc_physical_qubits":       sc_phys_data,
        "sc_physical_qubits_total": sc_phys_total,
        "sc_depth":                 sc_depth,
        "code_distance_d":          CODE_DISTANCE,
        "phys_per_tile":            PHYS_PER_TILE,

        # qLDPC physical
        "qldpc_block_qubits":    block_qubits,
        "qldpc_bridge_qubits":   nc * BRIDGE_PER_COUP,
        "qldpc_physical_qubits": qldpc_phys,

        # Grid depths
        "grid_naive_depth": naive_d,
        "grid_A_depth":     rg_d,
        "grid_B_depth":     rc_d,
        "grid_C_depth":     sag_d,
        "grid_D_depth":     sa_d,

        # Ring depths
        "ring_naive_depth": naive_d_ring,
        "ring_A_depth":     rg_d_ring,
        "ring_B_depth":     rc_d_ring,
        "ring_C_depth":     sag_d_ring,
        "ring_D_depth":     sa_d_ring,

        # Derived (grid, vs naive_grid)
        "qubit_reduction":       qubit_reduction,
        "naive_depth_overhead":  naive_depth_oh,
        "grid_A_pct": _pct(rg_d),
        "grid_B_pct": _pct(rc_d),
        "grid_C_pct": _pct(sag_d),
        "grid_D_pct": _pct(sa_d),

        # Derived (ring, vs ring naive)
        "ring_A_pct": _pct(rg_d_ring,  naive_d_ring),
        "ring_B_pct": _pct(rc_d_ring,  naive_d_ring),
        "ring_C_pct": _pct(sag_d_ring, naive_d_ring),
        "ring_D_pct": _pct(sa_d_ring,  naive_d_ring),

        # Legacy field aliases (backward compat)
        "qldpc_naive_depth":             naive_d,
        "qldpc_rg_depth":                rg_d,
        "qldpc_rc_depth":                rc_d,
        "qldpc_sa_greedy_depth":         sag_d,
        "qldpc_sa_cpsat_depth":          sa_d,
        "rg_depth_reduction_pct":        _pct(rg_d),
        "rc_depth_reduction_pct":        _pct(rc_d),
        "sa_greedy_depth_reduction_pct": _pct(sag_d),
        "sa_cpsat_depth_reduction_pct":  _pct(sa_d),
    })


# ── Write JSON ────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
output = {
    "description": "Table 1: SC vs qLDPC physical qubit and depth comparison — all families",
    "methodology": {
        "sc_physical_qubits":    "data-only tiles × (2·d²-1); data tiles = 6·n_qubits",
        "qldpc_physical_qubits": "n_blocks × (288+103) + n_couplers × 24",
        "sc_depth":              "T-count (Litinski transform)",
        "configs": {
            "naive": "random placement + sequential scheduler",
            "A":     "random placement + greedy_critical scheduler",
            "B":     "random placement + CP-SAT scheduler",
            "C":     "SA placement + greedy_critical scheduler",
            "D":     "SA placement + CP-SAT scheduler",
        },
        "depth_pct":       "(naive - config) / naive × 100  [grid vs grid naive, ring vs ring naive]",
        "code_distance":   CODE_DISTANCE,
        "t_count_source":  "sc_baseline.json (authoritative)",
        "data_sources":    "results/raw/{circuit}_seed42.json (primary), runs/ (legacy fallback)",
    },
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)


# ── Print table (grid topology) ───────────────────────────────────────────────

def _fmt(v, fmt=",") -> str:
    if v is None:
        return "—"
    if fmt == "pct":
        return f"{v:+.1f}%"
    if fmt == "x":
        return f"{v:.1f}×"
    return f"{v:{fmt}}"


if rows:
    W = 172
    print("=" * W)
    print(
        f"  {'Circuit':<22} {'Q':>4} {'B':>3} {'T':>6}  "
        f"{'SC PhysQ':>9} {'SC D':>6}  "
        f"{'qLDPC PhysQ':>11}  "
        f"{'Naive':>7} {'A:R+Grd':>8} {'B:R+CP':>7} {'C:SA+Grd':>9} {'D:SA+CP':>8}  "
        f"{'Q×':>5} {'SC OH':>6}  "
        f"{'A%':>6} {'B%':>6} {'C%':>6} {'D%':>6}  src"
    )
    print("  " + "-" * (W - 2))

    prev_family = None
    def _family(c: str) -> str:
        if c.startswith("Adder"):       return "Adder"
        if c.upper().startswith("QFT"): return "QFT"
        if c.startswith("gf"):          return "GF"
        return "Other"

    for r in rows:
        fam = _family(r["circuit"])
        if fam != prev_family:
            if prev_family is not None:
                print("  " + "·" * (W - 2))
            prev_family = fam

        print(
            f"  {r['circuit']:<22} {r['n_logical_qubits']:>4} {r['n_blocks']:>3} {r['t_count']:>6,}  "
            f"{r['sc_physical_qubits']:>9,} {r['sc_depth']:>6,}  "
            f"{r['qldpc_physical_qubits']:>11,}  "
            f"{_fmt(r['grid_naive_depth']):>7} "
            f"{_fmt(r['grid_A_depth']):>8} "
            f"{_fmt(r['grid_B_depth']):>7} "
            f"{_fmt(r['grid_C_depth']):>9} "
            f"{_fmt(r['grid_D_depth']):>8}  "
            f"{_fmt(r['qubit_reduction'], 'x'):>5} "
            f"{_fmt(r['naive_depth_overhead'], 'x'):>6}  "
            f"{_fmt(r['grid_A_pct'], 'pct'):>6} "
            f"{_fmt(r['grid_B_pct'], 'pct'):>6} "
            f"{_fmt(r['grid_C_pct'], 'pct'):>6} "
            f"{_fmt(r['grid_D_pct'], 'pct'):>6}  "
            f"{r['source']}"
        )

    print("=" * W)

    # Summary by family
    for fam_name, prefix in [("Adder", "Adder"), ("QFT", "QFT"), ("GF", "gf"), ("Other", None)]:
        fam_rows = [r for r in rows if _family(r["circuit"]) == fam_name]
        if not fam_rows:
            continue
        def _mean(key):
            vals = [r[key] for r in fam_rows if r[key] is not None]
            return f"{sum(vals)/len(vals):.1f}%" if vals else "—"
        print(f"\n  {fam_name} ({len(fam_rows)} circuits)  "
              f"mean D reduction:  A={_mean('grid_A_pct')}  B={_mean('grid_B_pct')}  "
              f"C={_mean('grid_C_pct')}  D={_mean('grid_D_pct')}")

    total_counts = {k: sum(1 for r in rows if r[f"grid_{k}_depth"] is not None)
                    for k in ("A","B","C","D")}
    print(f"\n  Total: {len(rows)} circuits  |  "
          + "  |  ".join(f"{k}: {v}/{len(rows)}" for k, v in total_counts.items()))
else:
    print("No rows — check SC baseline and raw/runs data.")

print(f"\nWritten → {OUT_PATH}")


# ── Figure 3 gap report ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("FIGURE 3 GAP REPORT (results/raw/ coverage)")
print("=" * 60)
raw_circuits = set()
if os.path.isdir(RAW_DIR):
    for f in os.listdir(RAW_DIR):
        if f.endswith(f"_seed{SEED}.json"):
            raw_circuits.add(f[: -(len(f"_seed{SEED}.json"))])

all_with_sc = set(sc_data.keys())
in_raw      = raw_circuits & all_with_sc
missing_raw = all_with_sc - raw_circuits

print(f"\n  Have raw file ({len(in_raw)}): {sorted(in_raw)}")
if missing_raw:
    print(f"\n  Missing raw file ({len(missing_raw)}) — needed for fig 3:")
    for c in sorted(missing_raw):
        print(f"    {c}")
else:
    print("\n  All SC-baseline circuits have raw files. Fig 3 is complete.")
