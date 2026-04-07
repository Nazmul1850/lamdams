"""
Steps 5-6: Surface code baseline via lsqecc Python API.

Auto-discovers circuits from:
  - experiment_circuits/*.qasm   (lsqecc QASM pipeline)
  - circuits/benchmarks/pbc/*_PBC.json  (direct PBC, no lsqecc needed)

For each circuit computes:
  sc_logical_depth  -- T-gate count after Litinski transform (each T = 1 round)
  sc_tile_count     -- number of surface-code patches (layout formula)
  sc_rows, sc_cols  -- lattice dimensions from SimplePreDistilledStates layout

Usage:
    python experiments/sc_baseline.py          # process PBC files + QASM-only circuits
    python experiments/sc_baseline.py --pbc    # PBC dir only
    python experiments/sc_baseline.py --qasm   # experiment_circuits/ QASM only
    python experiments/sc_baseline.py --circuit Adder16   # single circuit (auto-detects source)
    python experiments/sc_baseline.py --force  # re-run all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from fractions import Fraction

# -- Patch sys.modules BEFORE any lsqecc import ------------------------------
# qiskit.opflow was removed in Qiskit 1.0+; lsqecc still imports it at module
# level for type annotations and its state-vector simulator.  We stub it so
# the import succeeds, then never invoke the simulator (we only call fast
# structural methods that work without it).

class _Stub:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _Stub()
    def __xor__(self, o): return self
    def __rxor__(self, o): return self
    def __add__(self, o): return self
    def __mul__(self, o): return self
    def adjoint(self): return self
    def eval(self, *a, **kw): return 0.0
    @property
    def num_qubits(self): return 1
    def to_dict_fn(self): return self
    def to_matrix(self): return [[1, 0], [0, 1]]

class _OpflowStub(types.ModuleType):
    def __getattr__(self, name):
        cls = type(name, (_Stub,), {})
        setattr(type(self), name, cls)
        return cls

_opflow_stub = _OpflowStub("qiskit.opflow")
import qiskit as _qk
_qk.opflow = _opflow_stub
sys.modules["qiskit.opflow"] = _opflow_stub

# Stub qiskit.exceptions -- missing from this environment's namespace package.
# QiskitError must be a real Exception subclass so it can appear in except clauses.
class QiskitError(Exception): pass
_exceptions_stub = types.ModuleType("qiskit.exceptions")
_exceptions_stub.QiskitError = QiskitError
sys.modules["qiskit.exceptions"] = _exceptions_stub

# Stub qiskit.quantum_info -- used only inside simulation functions we never call.
_qinfo_stub = _OpflowStub("qiskit.quantum_info")
sys.modules["qiskit.quantum_info"] = _qinfo_stub

# Stub qiskit.qasm.node -- legacy AST submodule removed in Qiskit 2.x.
_qasm_node      = types.ModuleType("qiskit.qasm.node")
_qasm_node_node = types.ModuleType("qiskit.qasm.node.node")
class _QNode: pass
_qasm_node.node       = _qasm_node_node
_qasm_node.If         = _QNode
_qasm_node.Measure    = _QNode
_qasm_node.Barrier    = _QNode
_qasm_node_node.Node  = _QNode
sys.modules["qiskit.qasm.node"]      = _qasm_node
sys.modules["qiskit.qasm.node.node"] = _qasm_node_node

# -- Project root on path -----------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# -- lsqecc imports (after stub injection) -----------------------------------
from lsqecc.pauli_rotations.circuit import PauliOpCircuit
import lsqecc.logical_lattice_ops.logical_lattice_ops as llops
import pyzx as zx

# -- Project imports ----------------------------------------------------------
from modqldpc.core.trace import Trace

# -- Directory layout ---------------------------------------------------------
EXP_CIRCUITS = os.path.join(_ROOT, "experiment_circuits")
PBC_DIR      = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RESULTS_DIR  = os.path.join(_ROOT, "results")
SC_DIR       = os.path.join(RESULTS_DIR, "sc_baseline")
_trace       = Trace(os.path.join(RESULTS_DIR, "trace.jsonl"))

def discover_qasm_circuits() -> dict:
    """Scan experiment_circuits/ and return {circuit_name: qasm_path} for all .qasm files."""
    result = {}
    if not os.path.isdir(EXP_CIRCUITS):
        return result
    for fname in sorted(os.listdir(EXP_CIRCUITS)):
        if fname.endswith(".qasm"):
            name = os.path.splitext(fname)[0]
            result[name] = os.path.join(EXP_CIRCUITS, fname)
    return result


def discover_pbc_circuits() -> dict:
    """Scan circuits/benchmarks/pbc/ and return {circuit_name: pbc_path} for all *_PBC.json."""
    result = {}
    if not os.path.isdir(PBC_DIR):
        return result
    for fname in sorted(os.listdir(PBC_DIR)):
        if fname.endswith(".json"):
            name = _circuit_name_from_pbc_path(os.path.join(PBC_DIR, fname))
            result[name] = os.path.join(PBC_DIR, fname)
    return result


def run_sc_baseline(circuit_name: str, qasm_path: str | None = None) -> dict:
    """
    Compute surface-code baseline metrics for circuit_name.

    Pipeline:
      QASM -> PyZX -> lsqecc PauliOpCircuit -> to_y_free -> LogicalLatticeComputation
      -> extract T_count (= sc_logical_depth after Litinski transform)
      -> derive tile layout from SimplePreDistilledStates formula.

    sc_logical_depth = count_magic_states()
        Each T/T/Tdg gate requires one magic state injection = one lattice-surgery
        round.  After the Litinski transform all Clifford gates are absorbed
        into Pauli-frame tracking, so T-count is the exact temporal depth.

    sc_tile_count = 3 x (2*n_qubits + T_count + 1)
        From SimplePreDistilledStates layout initialiser:
          rows = 3 (min_rows)
          data cols = 2*n_qubits  (patches at every other column)
          magic-state cols = T_count  (one pre-distilled T-state per T gate)
          +1 for the right-side distillery anchor
    """
    if qasm_path is None:
        qasm_path = os.path.join(EXP_CIRCUITS, f"{circuit_name}.qasm")

    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"QASM not found: {qasm_path}")

    with open(qasm_path) as f:
        qasm_str = f.read()

    _trace.event("sc_start", circuit=circuit_name, source=qasm_path)
    t0 = time.perf_counter()

    # 1. QASM -> PyZX -> lsqecc PauliOpCircuit (Y-free)
    pyzx_circ = zx.Circuit.from_qasm(qasm_str)
    circ      = PauliOpCircuit.load_from_pyzx(pyzx_circ)
    circ      = circ.to_y_free_equivalent()
    print(f"  [{circuit_name}] parsed: {circ.qubit_num}q, {len(circ)} Pauli ops")

    # 2. Build logical lattice computation (fast -- no state simulation)
    lc = llops.LogicalLatticeComputation(circ)
    n_logical_ops = len(lc.ops)
    t_count       = lc.count_magic_states()   # T + T/Tdg gates
    n_qubits      = circ.qubit_num
    print(f"  [{circuit_name}] logical ops: {n_logical_ops}  T-count: {t_count}")

    # 3. Derive SC metrics (SimplePreDistilledStates layout formula)
    #
    #   sc_logical_depth = T_count
    #     After Litinski transform all Clifford ops collapse into frame;
    #     only T-gate injection rounds remain on the critical path.
    #
    #   sc_tile_count = rows x cols
    #     rows = 3  (min_rows from make_computation)
    #     cols = 2*n_qubits + t_count + 1  (data + magic-state columns)
    sc_rows          = 3
    sc_cols          = 2 * n_qubits + t_count + 1
    sc_logical_depth = t_count
    sc_tile_count    = sc_rows * sc_cols

    elapsed = time.perf_counter() - t0

    _trace.event(
        "sc_done", circuit=circuit_name,
        sc_logical_depth=sc_logical_depth,
        sc_tile_count=sc_tile_count,
        sc_rows=sc_rows, sc_cols=sc_cols,
        lc_ops_naive=n_logical_ops,
        elapsed_sec=round(elapsed, 3),
    )

    rec = {
        "circuit":          circuit_name,
        "n_qubits":         n_qubits,
        "t_count":          t_count,
        "sc_logical_depth": sc_logical_depth,
        "sc_tile_count":    sc_tile_count,
        "sc_rows":          sc_rows,
        "sc_cols":          sc_cols,
        "lc_ops_naive":     n_logical_ops,
        "elapsed_sec":      round(elapsed, 3),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    print(
        f"  [{circuit_name}] sc_logical_depth={sc_logical_depth:,}  "
        f"sc_tile_count={sc_tile_count:,}  ({sc_rows}x{sc_cols})  "
        f"({elapsed:.2f}s)"
    )
    return rec


def _circuit_name_from_pbc_path(pbc_path: str) -> str:
    """Derive a clean circuit name from a PBC JSON filename.

    Examples:
        Adder16_PBC.json  -> Adder16
        QFT32_PBC.json    -> QFT32
        rand_500_10k.json -> rand_500_10k
    """
    stem = os.path.splitext(os.path.basename(pbc_path))[0]
    if stem.endswith("_PBC"):
        stem = stem[:-4]
    return stem


def run_sc_baseline_from_pbc(pbc_path: str) -> dict:
    """Compute SC baseline directly from a PBC JSON file.

    The PBC files produced by this pipeline contain *only* π/8 (T-type)
    Pauli rotations — all Clifford gates have been absorbed into the frame.
    Therefore:
        t_count          = len(rotations)   (every entry is a π/8 rotation)
        sc_logical_depth = t_count
        sc_tile_count    = 3 × (2·n_qubits + t_count + 1)

    No lsqecc or PyZX required.
    """
    circuit_name = _circuit_name_from_pbc_path(pbc_path)

    with open(pbc_path) as f:
        data = json.load(f)

    rotations = data["rotations"]
    if not rotations:
        raise ValueError(f"No rotations in {pbc_path}")

    n_qubits = len(rotations[0][1])          # length of any Pauli string
    t_count  = sum(1 for r in rotations if r[2] == 8)   # π/8 only

    _trace.event("sc_pbc_start", circuit=circuit_name, source=pbc_path)
    t0 = time.perf_counter()

    sc_rows          = 3
    sc_cols          = 2 * n_qubits + t_count + 1
    sc_logical_depth = t_count
    sc_tile_count    = sc_rows * sc_cols
    n_logical_ops    = len(rotations)        # all are π/8, same as t_count

    elapsed = time.perf_counter() - t0

    _trace.event(
        "sc_pbc_done", circuit=circuit_name,
        sc_logical_depth=sc_logical_depth,
        sc_tile_count=sc_tile_count,
        elapsed_sec=round(elapsed, 3),
    )

    rec = {
        "circuit":          circuit_name,
        "n_qubits":         n_qubits,
        "t_count":          t_count,
        "sc_logical_depth": sc_logical_depth,
        "sc_tile_count":    sc_tile_count,
        "sc_rows":          sc_rows,
        "sc_cols":          sc_cols,
        "lc_ops_naive":     n_logical_ops,
        "elapsed_sec":      round(elapsed, 3),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "source":           "pbc",
    }

    print(
        f"  [{circuit_name}] n_qubits={n_qubits}  t_count={t_count:,}  "
        f"sc_depth={sc_logical_depth:,}  tiles={sc_tile_count:,}  ({sc_rows}x{sc_cols})"
    )
    return rec


def _print_run_status(all_circuits: list[str], existing_sc: dict) -> None:
    """
    Print a status report:
      - Which circuits are missing SC baseline
      - Which runs/{circuit}_random_sequential_seed42.json are missing
      - A table of all available SC baseline data
    """
    RUNS_DIR = os.path.join(_ROOT, "runs")
    PLACEMENT, SCHEDULER, SEED = "random", "sequential", 42

    print("\n" + "=" * 70)
    print("STATUS REPORT")
    print("=" * 70)

    missing_sc   = [c for c in all_circuits if c not in existing_sc]
    missing_runs = []
    for c in all_circuits:
        run_fname = f"{c}_{PLACEMENT}_{SCHEDULER}_seed{SEED}.json"
        run_path  = os.path.join(RUNS_DIR, run_fname)
        if not os.path.exists(run_path):
            missing_runs.append((c, run_fname))

    if missing_sc:
        print("\n[MISSING SC baseline] — re-run sc_baseline.py for these:")
        for c in missing_sc:
            print(f"  python experiments/sc_baseline.py --circuit {c}")
    else:
        print("\n[SC baseline] all circuits covered.")

    if missing_runs:
        print(f"\n[MISSING runs] — no runs/{'{circuit}'}_{PLACEMENT}_{SCHEDULER}_seed{SEED}.json for:")
        for c, fname in missing_runs:
            print(f"  {fname}")
        print("\n  To generate them:")
        for c, _ in missing_runs:
            print(
                f"  python experiments/run_experiment.py"
                f" --circuit {c} --mapping {PLACEMENT}"
                f" --scheduler {SCHEDULER} --seed {SEED}"
            )
    else:
        print(f"\n[runs] all {PLACEMENT}/{SCHEDULER}/seed{SEED} runs present.")

    # Available SC baseline table
    if existing_sc:
        print("\n=== Available SC Baseline Data ===")
        print(f"  {'Circuit':<25}  {'Qubits':>6}  {'T-count':>8}  {'SC depth':>10}  {'Tiles':>8}  {'Grid'}")
        print("  " + "-" * 70)
        for r in existing_sc.values():
            run_fname = f"{r['circuit']}_{PLACEMENT}_{SCHEDULER}_seed{SEED}.json"
            run_ok = "✓" if os.path.exists(os.path.join(RUNS_DIR, run_fname)) else "✗ run missing"
            print(
                f"  {r['circuit']:<25}  {r['n_qubits']:>6}  "
                f"{r['t_count']:>8,}  {r['sc_logical_depth']:>10,}  "
                f"{r['sc_tile_count']:>8,}  {r['sc_rows']}x{r['sc_cols']}  {run_ok}"
            )
    else:
        print("\n[No SC baseline data available yet]")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Surface code baseline — auto-discovers circuits from experiment_circuits/ and circuits/benchmarks/pbc/"
    )
    parser.add_argument(
        "--circuit", default=None,
        help="Process a single circuit by name (auto-detects PBC or QASM source)",
    )
    parser.add_argument(
        "--pbc", action="store_true",
        help="Process PBC files from circuits/benchmarks/pbc/ only",
    )
    parser.add_argument(
        "--qasm", action="store_true",
        help="Process QASM files from experiment_circuits/ only",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if result already exists",
    )
    args = parser.parse_args()

    os.makedirs(SC_DIR, exist_ok=True)
    results_path = os.path.join(SC_DIR, "sc_baseline.json")

    # Load existing results
    existing: dict = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = {r["circuit"]: r for r in json.load(f)}

    # Discover all circuit sources
    pbc_circuits  = discover_pbc_circuits()   # {name: pbc_path}
    qasm_circuits = discover_qasm_circuits()  # {name: qasm_path}
    all_circuits  = sorted(set(pbc_circuits) | set(qasm_circuits))

    # Determine what to process
    if args.circuit:
        to_process_pbc  = {args.circuit: pbc_circuits[args.circuit]}  if args.circuit in pbc_circuits  else {}
        to_process_qasm = {args.circuit: qasm_circuits[args.circuit]} if args.circuit in qasm_circuits else {}
        if not to_process_pbc and not to_process_qasm:
            print(f"ERROR: '{args.circuit}' not found in pbc/ or experiment_circuits/")
            print(f"  PBC circuits:  {sorted(pbc_circuits)}")
            print(f"  QASM circuits: {sorted(qasm_circuits)}")
            return
    elif args.pbc:
        to_process_pbc, to_process_qasm = pbc_circuits, {}
    elif args.qasm:
        to_process_pbc, to_process_qasm = {}, qasm_circuits
    else:
        # Default: PBC first, then QASM-only circuits (not already covered by PBC)
        to_process_pbc  = pbc_circuits
        to_process_qasm = {k: v for k, v in qasm_circuits.items() if k not in pbc_circuits}

    # --- Process PBC circuits ---
    if to_process_pbc:
        _trace.event("sc_pbc_phase_start", n_files=len(to_process_pbc))
        print(f"Processing {len(to_process_pbc)} PBC file(s) from {PBC_DIR}\n")
        for name, pbc_path in sorted(to_process_pbc.items()):
            if not args.force and name in existing:
                print(f"  [{name}] already done -- skipping")
                _trace.event("sc_cache_hit", circuit=name)
                continue
            rec = run_sc_baseline_from_pbc(pbc_path)
            existing[name] = rec

    # --- Process QASM-only circuits ---
    if to_process_qasm:
        _trace.event("sc_phase_start", circuits=list(to_process_qasm))
        print(f"\nProcessing {len(to_process_qasm)} QASM circuit(s) from {EXP_CIRCUITS}\n")
        for name, qasm_path in sorted(to_process_qasm.items()):
            if not args.force and name in existing:
                print(f"  [{name}] already done -- skipping")
                _trace.event("sc_cache_hit", circuit=name)
                continue
            print(f"  SC baseline for {name} ...")
            rec = run_sc_baseline(name, qasm_path)
            existing[name] = rec

    with open(results_path, "w") as f:
        json.dump(list(existing.values()), f, indent=2)
    _trace.event("sc_saved", path=results_path, n=len(existing))
    print(f"\nSaved: {results_path}")

    _print_run_status(all_circuits, existing)


if __name__ == "__main__":
    main()
