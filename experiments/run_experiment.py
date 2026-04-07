"""
Experiment runner for the PBC compiler paper evaluation.

Source of truth for circuits: circuits/benchmarks/pbc/  (all *_PBC.json files).
Circuits whose names start with "rand" are skipped (too slow for first iteration).

Configs
-------
  naive : random + sequential     (Table 1 naive-depth baseline)
  A     : random + greedy_critical
  B     : random + cpsat
  C     : sa     + greedy_critical
  D     : sa     + cpsat

SA hyperparameters (fixed)
--------------------------
  sa_steps = 25_000
  sa_t0    = 1e5
  sa_tend  = 5e-2

Efficiency
----------
  Configs C and D share ONE SA mapping pass (SA runs once per circuit).
  Configs naive, A and B share ONE random mapping pass.

Output
------
  runs/{circuit}_{mapping}_{scheduler}_seed{seed}.json
  Fields: circuit, n_qubits, n_blocks, t_count, mapping, scheduler, seed,
          inter_block_rotations, logical_depth, sa_iterations,
          mapping_time_sec, scheduling_time_sec, compile_time_sec, timestamp

Usage
-----
  # Run all PBC circuits, all 5 configs, SA-once efficiency:
  python experiments/run_experiment.py --all-pbc --seed 42

  # Single circuit, single config:
  python experiments/run_experiment.py --circuit Adder8 --mapping sa --scheduler cpsat

  # Single circuit, all configs (SA runs once for C+D):
  python experiments/run_experiment.py --circuit Adder8 --all-configs

  # 30-seed robustness on a specific circuit:
  python experiments/run_experiment.py --circuit qft_100_approx --all-configs --seeds 1-30

  # Force re-run even if file exists:
  python experiments/run_experiment.py --all-pbc --force
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
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.core.trace import Trace

# ── Directory layout ──────────────────────────────────────────────────────────

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR      = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
EXP_CIRCUITS = os.path.join(_ROOT, "experiment_circuits")
BENCH_DIR    = os.path.join(_ROOT, "circuits", "benchmarks")
RUNS_DIR     = os.path.join(_ROOT, "runs")
RESULTS_DIR  = os.path.join(_ROOT, "results")

_trace = Trace(os.path.join(RESULTS_DIR, "trace.jsonl"))

# ── SA hyperparameters (fixed for all paper runs) ─────────────────────────────

SA_STEPS = 25_000
SA_T0    = 1e5
SA_TEND  = 5e-2

# ── Circuits to skip (prefix match) ──────────────────────────────────────────

SKIP_PREFIXES = ("rand",)   # random circuits are too slow for first iteration

# ── Name mappings ─────────────────────────────────────────────────────────────

MAPPING_TO_MAPPER = {
    "random": "pure_random",
    "sa":     "simulated_annealing",
}

SCHEDULER_TO_FACTORY = {
    "sequential":      "sequential_scheduler",
    "greedy_critical": "greedy_critical",
    "cpsat":           "cp_sat",
}

# Configs that share a mapping pass
RANDOM_SCHEDULERS = ["sequential", "greedy_critical", "cpsat"]  # naive + A + B
SA_SCHEDULERS     = ["greedy_critical", "cpsat"]                 # C + D


# ── PBC discovery ─────────────────────────────────────────────────────────────

def _circuit_name_from_pbc(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith("_PBC") else stem


def pbc_path_for(circuit_name: str) -> str | None:
    """Return existing PBC path for circuit, trying _PBC.json then .json."""
    for candidate in [
        os.path.join(PBC_DIR, f"{circuit_name}_PBC.json"),
        os.path.join(PBC_DIR, f"{circuit_name}.json"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def discover_pbc_circuits() -> dict[str, str]:
    """Return {circuit_name: pbc_path} for every JSON in circuits/benchmarks/pbc/."""
    result = {}
    if not os.path.isdir(PBC_DIR):
        return result
    for fname in sorted(os.listdir(PBC_DIR)):
        if fname.endswith(".json"):
            name = _circuit_name_from_pbc(fname)
            result[name] = os.path.join(PBC_DIR, fname)
    return result


def _should_skip(circuit_name: str) -> bool:
    return any(circuit_name.startswith(p) for p in SKIP_PREFIXES)


# ── PBC loading ───────────────────────────────────────────────────────────────

def _build_pbc_from_qasm(circuit_name: str, qasm_path: str) -> str:
    """Convert QASM → PBC JSON and save to PBC_DIR. Returns pbc_path."""
    pbc_path = os.path.join(PBC_DIR, f"{circuit_name}.json")
    if os.path.exists(pbc_path):
        return pbc_path
    _trace.event("pbc_convert_start", circuit=circuit_name, source=qasm_path)
    qasm_str = load_qasm_file(qasm_path)
    conv = GoSCConverter(verbose=False)
    conv.convert_qasm(qasm_str)
    conv.greedy_layering()
    os.makedirs(PBC_DIR, exist_ok=True)
    with open(pbc_path, "w") as f:
        json.dump(conv.to_compact_payload(), f)
    _trace.event("pbc_saved", circuit=circuit_name, path=pbc_path)
    return pbc_path


def load_pbc(circuit_name: str, pbc_path: str | None = None) -> GoSCConverter:
    """
    Load PBC JSON into a GoSCConverter.
    Resolution: explicit path → _PBC.json → .json → build from QASM.
    """
    if pbc_path is None:
        pbc_path = pbc_path_for(circuit_name)
    if pbc_path is None:
        # Try building from experiment_circuits/
        for fname in os.listdir(EXP_CIRCUITS) if os.path.isdir(EXP_CIRCUITS) else []:
            if os.path.splitext(fname)[0] == circuit_name and fname.endswith(".qasm"):
                pbc_path = _build_pbc_from_qasm(
                    circuit_name, os.path.join(EXP_CIRCUITS, fname)
                )
                break
    if pbc_path is None:
        raise FileNotFoundError(
            f"No PBC file found for '{circuit_name}' in {PBC_DIR}.\n"
            f"Available: {sorted(discover_pbc_circuits())}"
        )
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)
    return conv


# ── Inter-block rotation counting ─────────────────────────────────────────────

def count_inter_block_rotations(rotations, plan) -> int:
    count = 0
    for rot in rotations:
        axis = rot.axis.lstrip("+-")
        n = len(axis)
        blocks = set()
        for qubit_idx in range(n):
            if axis[n - 1 - qubit_idx] != "I":
                b = plan.logical_to_block.get(qubit_idx)
                if b is not None:
                    blocks.add(b)
        if len(blocks) >= 2:
            count += 1
    return count


# ── Core pipeline: mapping phase ──────────────────────────────────────────────

def _run_mapping(
    circuit_name: str,
    mapping: str,
    seed: int,
    conv: GoSCConverter,
    hw,
    n_logicals: int,
) -> tuple:
    """
    Run the mapping phase and return:
        (plan, inter_block_rotations, mapping_time_sec)
    hw is mutated in-place to the best mapping state.
    """
    mapper_name = MAPPING_TO_MAPPER[mapping]
    map_cfg = MappingConfig(seed=seed, sa_steps=SA_STEPS, sa_t0=SA_T0, sa_tend=SA_TEND)

    _trace.event("mapping_start", circuit=circuit_name, mapping=mapping, seed=seed)
    t0 = time.perf_counter()

    plan = get_mapper(mapper_name).solve(
        MappingProblem(n_logicals=n_logicals),
        hw, map_cfg,
        {"rotations": conv.program.rotations, "verbose": False, "debug": False},
    )

    mapping_time = round(time.perf_counter() - t0, 3)

    t_rotations = [r for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9]
    inter_block = count_inter_block_rotations(t_rotations, plan)

    _trace.event(
        "mapping_done", circuit=circuit_name, mapping=mapping, seed=seed,
        inter_block_rotations=inter_block, mapping_time_sec=mapping_time,
    )
    return plan, inter_block, mapping_time


# ── Core pipeline: scheduling phase ──────────────────────────────────────────

CPSAT_TIME_LIMIT     = 1000.0   # seconds per layer; None = no limit
CPSAT_FALLBACK_SCHED = "greedy_critical"


def _run_scheduling(
    conv: GoSCConverter,
    hw,
    plan,
    scheduler: str,
    seed: int,
    *,
    cp_sat_time_limit: float | None = CPSAT_TIME_LIMIT,
    n_data: int = 11,
) -> tuple:
    """
    Run the scheduling phase for a fixed mapping (plan/hw) and return:
        (logical_depth, scheduling_time_sec, n_cpsat_fallbacks)

    If scheduler == "cpsat" and a layer hits the time limit or is infeasible,
    that layer falls back to greedy_critical and execution continues.
    """
    sched_name     = SCHEDULER_TO_FACTORY[scheduler]
    fallback_name  = SCHEDULER_TO_FACTORY[CPSAT_FALLBACK_SCHED]
    is_cpsat       = scheduler == "cpsat"

    cost_fn    = make_gross_actual_cost_fn(plan, n_data=n_data)
    policies   = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    total_depth      = 0
    n_cpsat_fallbacks = 0

    t0 = time.perf_counter()
    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )
        meta = {
            "start_time":        0,
            "layer_idx":         layer_id,
            "tie_breaker":       "duration",
            "cp_sat_time_limit": cp_sat_time_limit,
            "debug_decode":      False,
            "safe_fill":         True,
            "cp_sat_log":        False,
        }
        sched_problem = SchedulingProblem(
            dag=res.dag, hw=hw, seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta=meta,
        )

        try:
            S = get_scheduler(sched_name).solve(sched_problem)
        except Exception as exc:
            if is_cpsat:
                print(
                    f"  [cpsat fallback] layer {layer_id}: {exc!r} "
                    f"— falling back to {CPSAT_FALLBACK_SCHED}"
                )
                _trace.event(
                    "cpsat_fallback", layer=layer_id, reason=str(exc)
                )
                n_cpsat_fallbacks += 1
                S = get_scheduler(fallback_name).solve(sched_problem)
            else:
                raise

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

    scheduling_time = round(time.perf_counter() - t0, 3)
    return total_depth, scheduling_time, n_cpsat_fallbacks


# ── Result persistence ─────────────────────────────────────────────────────────

def result_path(circuit: str, mapping: str, scheduler: str, seed: int) -> str:
    return os.path.join(RUNS_DIR, f"{circuit}_{mapping}_{scheduler}_seed{seed}.json")


def _make_record(
    circuit_name: str,
    mapping: str,
    scheduler: str,
    seed: int,
    n_logicals: int,
    n_blocks: int,
    t_count: int,
    inter_block: int,
    logical_depth: int,
    mapping_time: float,
    scheduling_time: float,
    n_cpsat_fallbacks: int = 0,
) -> dict:
    return {
        "circuit":               circuit_name,
        "n_qubits":              n_logicals,
        "n_blocks":              n_blocks,
        "t_count":               t_count,
        "mapping":               mapping,
        "scheduler":             scheduler,
        "seed":                  seed,
        "inter_block_rotations": inter_block,
        "logical_depth":         logical_depth,
        "sa_iterations":         SA_STEPS if mapping == "sa" else None,
        "mapping_time_sec":      mapping_time,
        "scheduling_time_sec":   scheduling_time,
        "compile_time_sec":      round(mapping_time + scheduling_time, 3),
        "cpsat_fallback_layers": n_cpsat_fallbacks if scheduler == "cpsat" else None,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    }


def save_result(rec: dict) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = result_path(rec["circuit"], rec["mapping"], rec["scheduler"], rec["seed"])
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    _trace.event("result_saved", path=os.path.basename(path))
    print(f"  [saved] {os.path.basename(path)}")
    return path


def load_results(
    circuit: str | None = None,
    mapping: str | None = None,
    scheduler: str | None = None,
) -> List[dict]:
    """Load all runs/ JSONs, optionally filtered. Handles old 'placement' key."""
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
        if circuit   and rec.get("circuit")   != circuit:   continue
        if mapping   and rec.get("mapping")   != mapping:   continue
        if scheduler and rec.get("scheduler") != scheduler: continue
        records.append(rec)
    return records


# ── High-level runners ────────────────────────────────────────────────────────

def run_mapping_then_schedulers(
    circuit_name: str,
    mapping: str,
    schedulers: list[str],
    seed: int,
    *,
    pbc_path: str | None = None,
    skip_existing: bool = True,
    cp_sat_time_limit: float | None = CPSAT_TIME_LIMIT,
    n_data: int = 11,
    topology: str = "grid",
) -> None:
    """
    Run ONE mapping pass, then run each scheduler against that mapping.
    Saves one JSON file per scheduler. Skips any scheduler whose output
    file already exists (unless skip_existing=False).

    This is the efficiency primitive: SA runs once for C+D, random runs
    once for naive+A+B.
    """
    # Check which schedulers still need running
    pending = []
    for sched in schedulers:
        path = result_path(circuit_name, mapping, sched, seed)
        if skip_existing and os.path.exists(path):
            print(f"  [skip] {os.path.basename(path)}")
        else:
            pending.append(sched)

    if not pending:
        return

    # Load PBC
    conv = load_pbc(circuit_name, pbc_path)
    first_rot  = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    t_count    = sum(1 for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9)

    # Build hardware
    hw, _ = make_hardware(
        n_logicals, topology=topology, sparse_pct=0.0,
        n_data=n_data, coupler_capacity=1,
    )
    n_blocks = len(hw.blocks)

    print(f"\n[{circuit_name}] mapping={mapping} seed={seed}  "
          f"({n_logicals}q / {n_blocks} blocks / {t_count} T-gates)")

    # Mapping phase — runs once
    plan, inter_block, mapping_time = _run_mapping(
        circuit_name, mapping, seed, conv, hw, n_logicals
    )
    print(f"  mapping done in {mapping_time:.1f}s  inter_block={inter_block}")

    # Scheduling phase — once per pending scheduler
    for sched in pending:
        print(f"  scheduling: {sched} ...", end="", flush=True)
        depth, sched_time, n_fallbacks = _run_scheduling(
            conv, hw, plan, sched, seed,
            cp_sat_time_limit=cp_sat_time_limit, n_data=n_data,
        )
        fallback_msg = f"  ({n_fallbacks} fallback layers)" if n_fallbacks else ""
        print(f" depth={depth:,}  time={sched_time:.1f}s{fallback_msg}")
        rec = _make_record(
            circuit_name, mapping, sched, seed,
            n_logicals, n_blocks, t_count,
            inter_block, depth, mapping_time, sched_time, n_fallbacks,
        )
        _trace.event(
            "run_done", circuit=circuit_name, mapping=mapping,
            scheduler=sched, seed=seed,
            inter_block_rotations=inter_block, logical_depth=depth,
            mapping_time_sec=mapping_time, scheduling_time_sec=sched_time,
        )
        save_result(rec)


def run_all_configs(
    circuit_name: str,
    seed: int,
    *,
    pbc_path: str | None = None,
    skip_existing: bool = True,
    cp_sat_time_limit: float | None = CPSAT_TIME_LIMIT,
    n_data: int = 11,
    topology: str = "grid",
) -> None:
    """
    Run all 5 configs for one circuit:
      - random mapping once  → sequential (naive), greedy_critical (A), cpsat (B)
      - SA mapping once      → greedy_critical (C), cpsat (D)
    """
    kwargs = dict(
        pbc_path=pbc_path, skip_existing=skip_existing,
        cp_sat_time_limit=cp_sat_time_limit, n_data=n_data, topology=topology,
    )
    run_mapping_then_schedulers(circuit_name, "random", RANDOM_SCHEDULERS, seed, **kwargs)
    run_mapping_then_schedulers(circuit_name, "sa",     SA_SCHEDULERS,     seed, **kwargs)


# ── Single-config runner (public API, backwards-compat) ───────────────────────

def run_one_config(
    circuit_name: str,
    mapping: str,
    scheduler: str,
    seed: int,
    *,
    pbc_path: str | None = None,
    cp_sat_time_limit: float | None = CPSAT_TIME_LIMIT,
    n_data: int = 11,
    topology: str = "grid",
) -> dict:
    """Run a single (circuit, mapping, scheduler, seed) config and return the record."""
    conv = load_pbc(circuit_name, pbc_path)
    first_rot  = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    t_count    = sum(1 for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9)

    hw, _ = make_hardware(
        n_logicals, topology=topology, sparse_pct=0.0,
        n_data=n_data, coupler_capacity=1,
    )
    n_blocks = len(hw.blocks)

    plan, inter_block, mapping_time = _run_mapping(
        circuit_name, mapping, seed, conv, hw, n_logicals
    )
    depth, sched_time, n_fallbacks = _run_scheduling(
        conv, hw, plan, scheduler, seed,
        cp_sat_time_limit=cp_sat_time_limit, n_data=n_data,
    )
    return _make_record(
        circuit_name, mapping, scheduler, seed,
        n_logicals, n_blocks, t_count,
        inter_block, depth, mapping_time, sched_time, n_fallbacks,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PBC compiler experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All circuits, all configs (SA runs once per circuit for C+D):
  python experiments/run_experiment.py --all-pbc

  # Single circuit, all configs:
  python experiments/run_experiment.py --circuit Adder8 --all-configs

  # Single circuit, single config:
  python experiments/run_experiment.py --circuit Adder8 --mapping sa --scheduler cpsat

  # 30-seed robustness (all configs):
  python experiments/run_experiment.py --circuit qft_100_approx --all-configs --seeds 1-30
""",
    )
    parser.add_argument("--all-pbc",     action="store_true",
                        help="Run all circuits in circuits/benchmarks/pbc/ (skips rand* circuits)")
    parser.add_argument("--all-configs", action="store_true",
                        help="Run all 5 configs for the selected circuit (SA runs once for C+D)")
    parser.add_argument("--circuit",     help="Single circuit name")
    parser.add_argument("--mapping",     choices=list(MAPPING_TO_MAPPER),
                        help="Mapping strategy (for single-config runs)")
    parser.add_argument("--scheduler",   choices=list(SCHEDULER_TO_FACTORY),
                        help="Scheduler (for single-config runs)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--seeds",       help="Seed range, e.g. '1-30'")
    parser.add_argument("--cp-time",     type=float, default=None,
                        help="CP-SAT time limit per layer in seconds (default: no limit)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()

    skip    = not args.force
    cp_time = args.cp_time

    seeds = [args.seed]
    if args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))

    # ── --all-pbc: run every discovered circuit, all configs ──────────────────
    if args.all_pbc:
        pbc_circuits = discover_pbc_circuits()
        if not pbc_circuits:
            print(f"No PBC files found in {PBC_DIR}")
            return
        skipped = [n for n in pbc_circuits if _should_skip(n)]
        to_run  = {n: p for n, p in pbc_circuits.items() if not _should_skip(n)}
        if skipped:
            print(f"Skipping rand* circuits: {skipped}")
        print(f"Running {len(to_run)} circuit(s) × {len(seeds)} seed(s), all configs\n")
        _trace.event("phase_start", phase="all_pbc", n_circuits=len(to_run))
        for name, path in sorted(to_run.items()):
            for seed in seeds:
                run_all_configs(
                    name, seed, pbc_path=path,
                    skip_existing=skip, cp_sat_time_limit=cp_time,
                )
        _trace.event("phase_done", phase="all_pbc")
        return

    # ── --circuit + --all-configs: all 5 configs for one circuit ─────────────
    if args.circuit and args.all_configs:
        explicit_pbc = pbc_path_for(args.circuit)
        for seed in seeds:
            run_all_configs(
                args.circuit, seed, pbc_path=explicit_pbc,
                skip_existing=skip, cp_sat_time_limit=cp_time,
            )
        return

    # ── --circuit + --mapping + --scheduler: single config ───────────────────
    if args.circuit and args.mapping and args.scheduler:
        explicit_pbc = pbc_path_for(args.circuit)
        if explicit_pbc is None:
            # Check experiment_circuits/ as fallback
            qasm = os.path.join(EXP_CIRCUITS, f"{args.circuit}.qasm")
            if not os.path.exists(qasm):
                parser.error(
                    f"Circuit '{args.circuit}' has no PBC file in {PBC_DIR} "
                    f"and no QASM in {EXP_CIRCUITS}.\n"
                    f"Available PBC circuits: {sorted(discover_pbc_circuits())}"
                )
        for seed in seeds:
            path = result_path(args.circuit, args.mapping, args.scheduler, seed)
            if skip and os.path.exists(path):
                print(f"[skip] {os.path.basename(path)}")
                continue
            rec = run_one_config(
                args.circuit, args.mapping, args.scheduler, seed,
                pbc_path=explicit_pbc, cp_sat_time_limit=cp_time,
            )
            save_result(rec)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
