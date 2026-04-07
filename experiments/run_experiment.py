"""
Experiment orchestrator for the PBC compiler paper evaluation.

Runs (circuit × mapping × scheduler × seed) configurations and saves
a JSON record per run to results/raw/.

Usage examples:

  # Phase 1: core benchmarks (circuits 1 & 2, all configs, 1 seed)
  python experiments/run_experiment.py --phase 1

  # Config E: 30-seed robustness run on qft_100_approx
  python experiments/run_experiment.py --circuit qft_100_approx --mapping sa --scheduler cpsat --seeds 1-30

  # Phase 2: scaling sweep (QFT sizes, configs A and D only)
  python experiments/run_experiment.py --phase 2

  # Single run
  python experiments/run_experiment.py --circuit gf2_16_mult --mapping sa --scheduler cpsat --seed 42

  # Generate PBC cache files from QASM (do this before running experiments)
  python experiments/run_experiment.py --build-pbc
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.frontend.qasm_reader import load_qasm_file
from modqldpc.mapping.mapper import MappingConfig, MappingProblem, get_mapper
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.policy import (
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ChooseMagicBlockMinId,
    ShortestPathGatherRouting,
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.core.types import PauliRotation
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem
from modqldpc.pipeline.profiling import collect_circuit_profile, collect_layer_profile
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.core.trace import Trace

# ── Directory layout ──────────────────────────────────────────────────────────

_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR      = os.path.join(_ROOT, "circuits", "benchmarks")
PBC_DIR        = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
EXP_CIRCUITS   = os.path.join(_ROOT, "experiment_circuits")
RUNS_DIR       = os.path.join(_ROOT, "runs")
RESULTS_DIR    = os.path.join(_ROOT, "results")

# Trace writer — one JSONL file per orchestrator session
_trace = Trace(os.path.join(RESULTS_DIR, "trace.jsonl"))

# ── Name mappings ─────────────────────────────────────────────────────────────

PLACEMENT_TO_MAPPER = {
    "random":  "pure_random",
    "greedy":  "auto_round_robin_mapping",
    "sa":      "simulated_annealing",
}

SCHEDULER_TO_FACTORY = {
    "sequential":      "sequential_scheduler",
    "greedy_critical": "greedy_critical",
    "cpsat":           "cp_sat",
}

# ── Circuits registry ─────────────────────────────────────────────────────────

# Maps circuit name → (qasm filename, expected n_qubits hint)
CIRCUITS = {
    "gf2_16_mult":             ("gf2_16_mult.qasm",             48),
    "qft_100_approx":          ("qft_100_approx.qasm",         100),
    "rand_500_10k":            ("random_ct_500q_10k.qasm",     500),
    "random_ct_500q_10k":      ("random_ct_500q_10k.qasm",     500),
    # Scaling sweep
    "qft_22_approx":           ("qft_22_approx.qasm",           22),
    "qft_33_approx":           ("qft_33_approx.qasm",           33),
    "qft_44_approx":           ("qft_44_approx.qasm",           44),
    "qft_66_approx":           ("qft_66_approx.qasm",           66),
    "qft_99_approx":           ("qft_99_approx.qasm",           99),
}


# ── Inter-block rotation counting ─────────────────────────────────────────────

def count_inter_block_rotations(rotations, plan) -> int:
    """
    Count rotations whose Pauli support spans ≥ 2 qLDPC blocks after mapping.
    Works for any mapper (does not rely on plan.meta).

    The axis string has rightmost character = qubit 0.
    """
    count = 0
    for rot in rotations:
        axis = rot.axis.lstrip("+-")
        n = len(axis)
        blocks = set()
        for qubit_idx in range(n):
            char = axis[n - 1 - qubit_idx]   # rightmost = qubit 0
            if char != "I":
                b = plan.logical_to_block.get(qubit_idx)
                if b is not None:
                    blocks.add(b)
        if len(blocks) >= 2:
            count += 1
    return count


# ── PBC discovery ────────────────────────────────────────────────────────────

def _circuit_name_from_pbc_fname(fname: str) -> str:
    """'Adder16_PBC.json' → 'Adder16',  'gf2_16_mult.json' → 'gf2_16_mult'."""
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith("_PBC") else stem


def pbc_path_for(circuit_name: str) -> str | None:
    """
    Return the path to an existing PBC JSON for circuit_name, or None.
    Tries (in order):
      1. circuits/benchmarks/pbc/{circuit_name}_PBC.json   (externally compiled)
      2. circuits/benchmarks/pbc/{circuit_name}.json       (built by build_pbc)
    """
    candidates = [
        os.path.join(PBC_DIR, f"{circuit_name}_PBC.json"),
        os.path.join(PBC_DIR, f"{circuit_name}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def discover_pbc_circuits() -> dict[str, str]:
    """
    Scan circuits/benchmarks/pbc/ and return {circuit_name: pbc_path}
    for every JSON file present.
    """
    result = {}
    if not os.path.isdir(PBC_DIR):
        return result
    for fname in sorted(os.listdir(PBC_DIR)):
        if fname.endswith(".json"):
            name = _circuit_name_from_pbc_fname(fname)
            result[name] = os.path.join(PBC_DIR, fname)
    return result


# ── PBC cache management ──────────────────────────────────────────────────────

def build_pbc(circuit_name: str) -> str:
    """
    Convert QASM → PBC JSON (compact format) and save to circuits/benchmarks/pbc/.
    Returns path to the PBC JSON.  Skips if already exists.
    """
    fname, _ = CIRCUITS[circuit_name]
    qasm_path = os.path.join(BENCH_DIR, fname)
    pbc_path  = os.path.join(PBC_DIR, f"{circuit_name}.json")

    if os.path.exists(pbc_path):
        _trace.event("pbc_cache_hit", circuit=circuit_name, path=pbc_path)
        return pbc_path

    # Prefer experiment_circuits/ over legacy benchmarks dir
    exp_path = os.path.join(EXP_CIRCUITS, fname)
    if os.path.exists(exp_path):
        qasm_path = exp_path
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(
            f"QASM not found in experiment_circuits/ or circuits/benchmarks/: {fname}\n"
            f"Run: python circuits/generate_benchmarks.py --all"
        )

    _trace.event("pbc_convert_start", circuit=circuit_name, source=qasm_path)
    qasm_str = load_qasm_file(qasm_path)
    conv     = GoSCConverter(verbose=False)
    program  = conv.convert_qasm(qasm_str)
    _        = conv.greedy_layering()

    t_count  = sum(1 for r in program.rotations if abs(r.angle) < math.pi / 2 - 1e-9)
    _trace.event(
        "pbc_convert_done", circuit=circuit_name,
        n_qubits=conv.num_qubits, t_count=t_count,
        n_layers=len(conv.layers), n_rotations=len(program.rotations),
    )

    os.makedirs(PBC_DIR, exist_ok=True)
    payload = conv.to_compact_payload()
    with open(pbc_path, "w") as f:
        json.dump(payload, f)
    _trace.event("pbc_saved", circuit=circuit_name, path=pbc_path)
    return pbc_path


def load_pbc(circuit_name: str, pbc_path: str | None = None):
    """
    Load a pre-built PBC JSON and return a GoSCConverter with data loaded.

    Resolution order (when pbc_path is not given explicitly):
      1. circuits/benchmarks/pbc/{circuit_name}_PBC.json  (externally compiled)
      2. circuits/benchmarks/pbc/{circuit_name}.json      (built by build_pbc)
      3. Build from QASM via build_pbc() (requires circuit in CIRCUITS registry)
    """
    if pbc_path is None:
        pbc_path = pbc_path_for(circuit_name)
        if pbc_path is None:
            pbc_path = build_pbc(circuit_name)
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)
    return conv


def qasm_stats(circuit_name: str):
    """Return (n_qubits, t_count) by loading the PBC (lightweight)."""
    conv = load_pbc(circuit_name)
    first_rot = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    t_count = sum(
        1 for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9
    )
    return n_logicals, t_count


# ── Single-config runner ──────────────────────────────────────────────────────

def run_one_config(
    circuit_name: str,
    mapping: str,
    scheduler: str,
    seed: int,
    *,
    pbc_path: str | None = None,
    sa_steps: int = 50_000,
    sa_t0: float = 1e5,
    sa_tend: float = 0.05,
    cp_sat_time_limit: float = 120.0,
    n_data: int = 11,
    topology: str = "grid",
) -> dict:
    """
    Run one (circuit, mapping, scheduler, seed) configuration.
    Returns a metrics dict matching the paper's output schema.
    pbc_path: explicit path to PBC JSON (optional; auto-resolved when None).
    """
    mapper_name = PLACEMENT_TO_MAPPER[mapping]
    sched_name  = SCHEDULER_TO_FACTORY[scheduler]

    # ── Load PBC ──────────────────────────────────────────────────────────────
    conv = load_pbc(circuit_name, pbc_path=pbc_path)
    first_rot  = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    t_count    = sum(
        1 for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    hw, hw_spec = make_hardware(
        n_logicals, topology=topology, sparse_pct=0.0,
        n_data=n_data, coupler_capacity=1,
    )
    n_blocks = len(hw.blocks)

    _trace.event(
        "run_start", circuit=circuit_name, mapping=mapping,
        scheduler=scheduler, seed=seed, n_blocks=n_blocks, mapper=mapper_name,
    )

    t_start = time.perf_counter()

    # ── mapping ───────────────────────────────────────────────────────────────
    map_cfg = MappingConfig(
        seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend,
    )
    plan = get_mapper(mapper_name).solve(
        MappingProblem(n_logicals=n_logicals),
        hw, map_cfg,
        {"rotations": conv.program.rotations, "verbose": False, "debug": False},
    )

    t_after_map = time.perf_counter()

    # ── Count inter-block rotations (mapping-only metric) ───────────────────
    # Use only π/8 (T-gate) rotations as they are the primary cost
    t_rotations = [r for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9]
    inter_block = count_inter_block_rotations(t_rotations, plan)

    # ── Lowering policies ─────────────────────────────────────────────────────
    cost_fn  = make_gross_actual_cost_fn(plan, n_data=n_data)
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    # ── Pipeline loop ─────────────────────────────────────────────────────────
    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    total_depth = 0

    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )
        sched_problem = SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": cp_sat_time_limit,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        )
        S = get_scheduler(sched_name).solve(sched_problem)

        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next  = [effective_rotations[i] for i in next_idxs]
        ex = executor.execute_layer(
            layer=layer_id, dag=res.dag, schedule=S,
            frame_in=frame, next_layer_rotations=rot_next,
        )
        for r in ex.next_rotations_effective:
            effective_rotations[r.idx] = r
        frame = ex.frame_after
        total_depth += ex.depth

    t_end            = time.perf_counter()
    mapping_time     = round(t_after_map - t_start, 3)
    scheduling_time  = round(t_end - t_after_map, 3)
    compile_time     = round(t_end - t_start, 3)

    _trace.event(
        "run_done", circuit=circuit_name, mapping=mapping,
        scheduler=scheduler, seed=seed,
        inter_block_rotations=inter_block, logical_depth=total_depth,
        mapping_time_sec=mapping_time,
        scheduling_time_sec=scheduling_time,
        compile_time_sec=compile_time,
    )

    return {
        "circuit":               circuit_name,
        "n_qubits":              n_logicals,
        "n_blocks":              n_blocks,
        "t_count":               t_count,
        "mapping":             mapping,
        "scheduler":             scheduler,
        "seed":                  seed,
        "inter_block_rotations": inter_block,
        "logical_depth":         total_depth,
        "sa_iterations":         sa_steps if mapping == "sa" else None,
        "mapping_time_sec":      mapping_time,
        "scheduling_time_sec":   scheduling_time,
        "compile_time_sec":      compile_time,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }


# ── Result persistence ────────────────────────────────────────────────────────

_VALID_GATES = {"h", "s", "sdg", "t", "tdg", "cx"}


def validate_gate_set(circuit_name: str) -> bool:
    """
    Check every gate in the circuit's QASM is in {h, s, sdg, t, tdg, cx}.
    Returns True on pass, False on fail.  Writes a trace event either way.
    """
    fname, _ = CIRCUITS[circuit_name]
    qasm_path = os.path.join(EXP_CIRCUITS, fname)
    bad = []
    with open(qasm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("OPENQASM") \
                    or line.startswith("include") or line.startswith("qreg") \
                    or line.startswith("creg"):
                continue
            gate = line.split("(")[0].split()[0].rstrip(";")
            if gate and gate not in _VALID_GATES:
                bad.append(gate)
    passed = len(bad) == 0
    _trace.event(
        "gate_set_check", circuit=circuit_name,
        result="PASS" if passed else "FAIL",
        invalid_gates=list(set(bad))[:10],
    )
    status = "PASS" if passed else f"FAIL {set(bad)}"
    print(f"  [gate-set] {circuit_name}: {status}")
    return passed


# ── Result persistence ────────────────────────────────────────────────────────

def result_path(circuit: str, mapping: str, scheduler: str, seed: int) -> str:
    return os.path.join(
        RUNS_DIR, f"{circuit}_{mapping}_{scheduler}_seed{seed}.json"
    )


def save_result(rec: dict) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = result_path(rec["circuit"], rec["mapping"], rec["scheduler"], rec["seed"])
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    _trace.event("result_saved", path=os.path.basename(path))
    return path


def load_results(
    circuit: str | None = None,
    mapping: str | None = None,
    scheduler: str | None = None,
) -> List[dict]:
    """Load all runs/ JSONs, optionally filtered."""
    records = []
    if not os.path.isdir(RUNS_DIR):
        return records
    for fname in sorted(os.listdir(RUNS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RUNS_DIR, fname)) as f:
            rec = json.load(f)
        if circuit   and rec.get("circuit")   != circuit:   continue
        if mapping and rec.get("mapping") != mapping: continue
        if scheduler and rec.get("scheduler") != scheduler: continue
        records.append(rec)
    return records


# ── Phase definitions ─────────────────────────────────────────────────────────

PHASE1_CIRCUITS = ["gf2_16_mult", "qft_100_approx"]
PHASE1_CONFIGS  = [
    ("random", "greedy_critical"),   # A
    ("random", "cpsat"),             # B
    ("sa",     "greedy_critical"),   # C
    ("sa",     "cpsat"),             # D
]

PHASE2_CIRCUITS = [f"qft_{n}_approx" for n in [22, 33, 44, 66, 99]]
PHASE2_CONFIGS  = [
    ("random", "greedy_critical"),   # A – baseline
    ("sa",     "cpsat"),             # D – best
]

ROBUSTNESS_SEEDS = list(range(1, 31))   # Config E: 30 seeds


def _run_and_save(circuit, mapping, scheduler, seed, skip_existing=True,
                  pbc_path=None, **kwargs):
    path = result_path(circuit, mapping, scheduler, seed)
    if skip_existing and os.path.exists(path):
        _trace.event("run_skipped", path=os.path.basename(path))
        print(f"  [skip] {os.path.basename(path)} already exists")
        return
    rec = run_one_config(circuit, mapping, scheduler, seed,
                         pbc_path=pbc_path, **kwargs)
    save_result(rec)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PBC compiler experiment runner")
    parser.add_argument("--phase",     type=int, choices=[1, 2, 3],
                        help="Run a full experiment phase")
    parser.add_argument("--robustness", action="store_true",
                        help="Run Config E: 30 seeds of SA+CP-SAT on qft_100_approx")
    parser.add_argument("--all-pbc",  action="store_true",
                        help="Run all circuits found in circuits/benchmarks/pbc/ "
                             "(requires --mapping and --scheduler)")
    parser.add_argument("--circuit",   help="Single circuit name")
    parser.add_argument("--mapping", choices=list(PLACEMENT_TO_MAPPER), help="mapping strategy")
    parser.add_argument("--scheduler", choices=list(SCHEDULER_TO_FACTORY), help="Scheduler")
    parser.add_argument("--seed",      type=int, default=42, help="RNG seed (single run)")
    parser.add_argument("--seeds",     help="Seed range, e.g. '1-30'")
    parser.add_argument("--sa-steps",  type=int, default=50_000, help="SA iterations")
    parser.add_argument("--cp-time",   type=float, default=120.0, help="CP-SAT time limit (s)")
    parser.add_argument("--build-pbc", action="store_true",
                        help="Build PBC cache files for all registered circuits that have QASM")
    parser.add_argument("--force",     action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()

    skip = not args.force
    kwargs = dict(sa_steps=args.sa_steps, cp_sat_time_limit=args.cp_time)

    if args.all_pbc:
        if not args.mapping or not args.scheduler:
            parser.error("--all-pbc requires --mapping and --scheduler")
        pbc_circuits = discover_pbc_circuits()
        if not pbc_circuits:
            print(f"No PBC files found in {PBC_DIR}")
            return
        seeds = [args.seed]
        if args.seeds:
            lo, hi = args.seeds.split("-")
            seeds = list(range(int(lo), int(hi) + 1))
        _trace.event("phase_start", phase="all_pbc",
                     n_circuits=len(pbc_circuits),
                     mapping=args.mapping, scheduler=args.scheduler)
        print(f"Running {len(pbc_circuits)} PBC circuit(s) with "
              f"{args.mapping}/{args.scheduler}/seed(s)={seeds}")
        for name, path in sorted(pbc_circuits.items()):
            for seed in seeds:
                print(f"\n[{name}] mapping={args.mapping} scheduler={args.scheduler} seed={seed}")
                _run_and_save(name, args.mapping, args.scheduler, seed=seed,
                              skip_existing=skip, pbc_path=path, **kwargs)
        _trace.event("phase_done", phase="all_pbc")
        return

    if args.build_pbc:
        _trace.event("phase_start", phase="build_pbc")
        for name in CIRCUITS:
            fname, _ = CIRCUITS[name]
            exp_path  = os.path.join(EXP_CIRCUITS, fname)
            bench_path = os.path.join(BENCH_DIR, fname)
            if os.path.exists(exp_path) or os.path.exists(bench_path):
                try:
                    build_pbc(name)
                except Exception as e:
                    _trace.event("pbc_error", circuit=name, error=str(e))
        return

    if args.phase == 1:
        _trace.event("phase_start", phase=1)
        for circuit in PHASE1_CIRCUITS:
            for mapping, scheduler in PHASE1_CONFIGS:
                _run_and_save(circuit, mapping, scheduler, seed=42,
                              skip_existing=skip, **kwargs)
        _trace.event("phase_done", phase=1)
        return

    if args.robustness:
        _trace.event("phase_start", phase="robustness_E")
        for seed in ROBUSTNESS_SEEDS:
            _run_and_save("qft_100_approx", "sa", "cpsat", seed=seed,
                          skip_existing=skip, **kwargs)
        _trace.event("phase_done", phase="robustness_E")
        return

    if args.phase == 2:
        _trace.event("phase_start", phase=2)
        for circuit in PHASE2_CIRCUITS:
            for mapping, scheduler in PHASE2_CONFIGS:
                _run_and_save(circuit, mapping, scheduler, seed=42,
                              skip_existing=skip, **kwargs)
        _trace.event("phase_done", phase=2)
        return

    if args.phase == 3:
        _trace.event("phase_start", phase=3)
        for mapping, scheduler in PHASE2_CONFIGS:
            _run_and_save("random_ct_500q_10k", mapping, scheduler, seed=42,
                          skip_existing=skip, **kwargs)
        _trace.event("phase_done", phase=3)
        return

    # Single or multi-seed run
    if args.circuit and args.mapping and args.scheduler:
        # Resolve PBC path — works for both registered and PBC-only circuits
        explicit_pbc = pbc_path_for(args.circuit)
        if explicit_pbc is None and args.circuit not in CIRCUITS:
            parser.error(
                f"Circuit '{args.circuit}' not found in CIRCUITS registry "
                f"and no PBC file exists in {PBC_DIR}.\n"
                f"Available PBC circuits: {sorted(discover_pbc_circuits())}"
            )
        seeds = [args.seed]
        if args.seeds:
            lo, hi = args.seeds.split("-")
            seeds = list(range(int(lo), int(hi) + 1))
        for seed in seeds:
            _run_and_save(args.circuit, args.mapping, args.scheduler, seed=seed,
                          skip_existing=skip, pbc_path=explicit_pbc, **kwargs)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
