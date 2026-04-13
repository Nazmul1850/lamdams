"""
experiments/run_experiment.py — PBC compiler paper evaluation.

Hardware (Gross code): n_data=11, coupler_capacity=1, sparse_pct=0.0.
Topologies: grid AND ring, both evaluated per circuit.
SA hyperparameters: tuned values from sensitivity analysis (phases 1–3).

Configs (per topology):
  naive : random + sequential
  A     : random + greedy_critical
  B     : random + cpsat
  C     : sa     + greedy_critical
  D     : sa     + cpsat

Efficiency: SA runs ONCE per topology (shared by C+D).
            Random runs ONCE per topology (shared by naive+A+B).

Output:
  results/raw/{circuit}_seed{seed}.json

Usage:
  # List all available circuits:
  python experiments/run_experiment.py --list

  # Run one circuit (both topologies, all 5 configs, seed=42):
  python experiments/run_experiment.py --circuit Adder8

  # With custom seed and CP-SAT time limit:
  python experiments/run_experiment.py --circuit Adder8 --seed 1 --cp-time 300

  # Re-run even if result file exists:
  python experiments/run_experiment.py --circuit Adder8 --force
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.mapping.algos.sa_mapping import TUNED_SA_STEPS, TUNED_SA_T0, TUNED_SA_TEND
from modqldpc.mapping.factory import get_mapper
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem
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


# ── Directory layout ──────────────────────────────────────────────────────────

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR  = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RAW_DIR  = os.path.join(_ROOT, "results", "raw")

# ── Hardware constants ────────────────────────────────────────────────────────

N_DATA           = 11   # Gross code data qubits per block
COUPLER_CAPACITY = 1    # per-coupler capacity
SPARSE_PCT       = 0.0  # dense packing

# ── SA hyperparameters (tuned, from sensitivity analysis phases 1–3) ──────────

SA_STEPS = TUNED_SA_STEPS   # 22_500
SA_T0    = TUNED_SA_T0      # 1e5
SA_TEND  = TUNED_SA_TEND    # 5e-2

# ── CP-SAT fallback ───────────────────────────────────────────────────────────

CPSAT_FALLBACK_SCHED = "greedy_critical"
DEFAULT_CPSAT_TIME   = 1000.0   # seconds per layer

# ── Config definitions ────────────────────────────────────────────────────────

CONFIGS = {
    "naive": ("random", "sequential"),
    "A":     ("random", "greedy_critical"),
    "B":     ("random", "cpsat"),
    "C":     ("sa",     "greedy_critical"),
    "D":     ("sa",     "cpsat"),
}

MAPPER_NAMES = {
    "random": "pure_random",
    "sa":     "simulated_annealing",
}

SCHED_NAMES = {
    "sequential":      "sequential_scheduler",
    "greedy_critical": "greedy_critical",
    "cpsat":           "cp_sat",
}

TOPOLOGIES = ["grid", "ring"]


# ── Circuit discovery ─────────────────────────────────────────────────────────

def _circuit_name(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem[:-4] if stem.endswith("_PBC") else stem


def discover_circuits() -> Dict[str, str]:
    """Return {circuit_name: pbc_path} for every JSON in circuits/benchmarks/pbc/."""
    if not os.path.isdir(PBC_DIR):
        return {}
    result = {}
    for fname in sorted(os.listdir(PBC_DIR)):
        if fname.endswith(".json"):
            name = _circuit_name(fname)
            result[name] = os.path.join(PBC_DIR, fname)
    return result


def find_pbc(circuit_name: str) -> Optional[str]:
    for candidate in [
        os.path.join(PBC_DIR, f"{circuit_name}_PBC.json"),
        os.path.join(PBC_DIR, f"{circuit_name}.json"),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


# ── PBC loading ───────────────────────────────────────────────────────────────

def load_pbc(circuit_name: str) -> Tuple[GoSCConverter, int, int, int]:
    """
    Load the PBC for circuit_name.
    Returns (conv, n_logicals, t_count, n_layers).
    """
    path = find_pbc(circuit_name)
    if path is None:
        available = sorted(discover_circuits())
        raise FileNotFoundError(
            f"No PBC found for '{circuit_name}' in {PBC_DIR}.\n"
            f"Available: {available}"
        )
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(path)
    first_rot  = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    t_count    = sum(1 for r in conv.program.rotations if abs(r.angle) < math.pi / 2 - 1e-9)
    n_layers   = len(conv.layers)
    return conv, n_logicals, t_count, n_layers


# ── Inter-block rotation counting ─────────────────────────────────────────────

# ── Mapping ───────────────────────────────────────────────────────────────────

def _run_mapping(
    mapping_key: str,
    seed: int,
    conv: GoSCConverter,
    hw,
    n_logicals: int,
) -> Tuple[MappingPlan, float]:
    """
    Run mapping and return (plan, mapping_time_sec).
    """
    mapper_name = MAPPER_NAMES[mapping_key]
    cfg = MappingConfig(seed=seed, sa_steps=SA_STEPS, sa_t0=SA_T0, sa_tend=SA_TEND)

    t0 = time.perf_counter()
    plan = get_mapper(mapper_name).solve(
        MappingProblem(n_logicals=n_logicals),
        hw,
        cfg,
        {"rotations": conv.program.rotations, "verbose": False, "debug": False},
    )
    mapping_time = round(time.perf_counter() - t0, 3)
    return plan, mapping_time


# ── Scheduling ────────────────────────────────────────────────────────────────

def _run_scheduling(
    conv: GoSCConverter,
    hw,
    plan: MappingPlan,
    sched_key: str,
    seed: int,
    cp_time: float,
) -> Tuple[int, float, int, int]:
    """
    Run scheduling and return (logical_depth, sched_time_sec, cpsat_fallbacks, cpsat_suboptimal).
    cpsat_fallbacks/suboptimal are 0 for non-cpsat schedulers.
    """
    sched_name    = SCHED_NAMES[sched_key]
    fallback_name = SCHED_NAMES[CPSAT_FALLBACK_SCHED]
    is_cpsat      = sched_key == "cpsat"

    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=make_gross_actual_cost_fn(plan)),
    )

    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    total_depth  = 0
    n_fallback   = 0
    n_suboptimal = 0

    t0 = time.perf_counter()
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
                "cp_sat_time_limit": cp_time,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        )
        try:
            S = get_scheduler(sched_name).solve(sched_problem)
            if is_cpsat and S.meta.get("cp_sat_status") == "FEASIBLE":
                n_suboptimal += 1
        except Exception:
            if is_cpsat:
                n_fallback += 1
                S = get_scheduler(fallback_name).solve(sched_problem)
            else:
                raise

        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next  = [effective_rotations[i] for i in next_idxs]
        ex = executor.execute_layer(
            layer=layer_id,
            dag=res.dag,
            schedule=S,
            frame_in=frame,
            next_layer_rotations=rot_next,
        )
        for r in ex.next_rotations_effective:
            effective_rotations[r.idx] = r
        frame = ex.frame_after
        total_depth += ex.depth

    sched_time = round(time.perf_counter() - t0, 3)
    return total_depth, sched_time, n_fallback, n_suboptimal


# ── Per-topology runner ───────────────────────────────────────────────────────

def _run_topology(
    circuit_name: str,
    topology: str,
    conv: GoSCConverter,
    n_logicals: int,
    seed: int,
    cp_time: float,
) -> dict:
    """
    Run all 5 configs for one topology. Returns a dict with hw info and per-config results.
    SA runs once (C+D). Random runs once (naive+A+B).
    """
    hw, spec = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=SPARSE_PCT,
        n_data=N_DATA,
        coupler_capacity=COUPLER_CAPACITY,
    )
    n_blocks   = len(hw.blocks)
    n_couplers = len(hw.couplers)

    print(f"\n  [{circuit_name}|{topology.upper()}] {spec.label()}  n_blocks={n_blocks}  n_couplers={n_couplers}")

    # ── Random mapping (shared by naive, A, B) ────────────────────────────────
    print(f"    mapping: random ...", end="", flush=True)
    # need a fresh hw for random — clone state after make_hardware
    hw_rand, _ = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=SPARSE_PCT,
        n_data=N_DATA,
        coupler_capacity=COUPLER_CAPACITY,
    )
    plan_rand, t_map_rand = _run_mapping("random", seed, conv, hw_rand, n_logicals)
    print(f" done ({t_map_rand:.1f}s)")

    # ── SA mapping (shared by C, D) ───────────────────────────────────────────
    print(f"    mapping: SA     ...", end="", flush=True)
    hw_sa, _ = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=SPARSE_PCT,
        n_data=N_DATA,
        coupler_capacity=COUPLER_CAPACITY,
    )
    plan_sa, t_map_sa = _run_mapping("sa", seed, conv, hw_sa, n_logicals)
    print(f" done ({t_map_sa:.1f}s)")

    # SA score metadata from plan
    sa_meta = plan_sa.meta or {}
    sa_score_info = {
        "sa_score_total":   sa_meta.get("best_score_total"),
        "sa_unused_blocks": sa_meta.get("unused_blocks"),
        "sa_num_multiblock":sa_meta.get("num_multiblock"),
        "sa_span_total":    sa_meta.get("span_total"),
        "sa_mst_total":     sa_meta.get("mst_total"),
    }

    # ── Scheduling: all 5 configs ─────────────────────────────────────────────
    config_results: Dict[str, dict] = {}
    naive_depth: Optional[int] = None

    for config_name, (mapping_key, sched_key) in CONFIGS.items():
        is_sa = (mapping_key == "sa")
        plan  = plan_sa   if is_sa else plan_rand
        hw    = hw_sa     if is_sa else hw_rand
        t_map = t_map_sa  if is_sa else t_map_rand

        print(f"    {config_name} ({mapping_key}+{sched_key}) ...", end="", flush=True)
        depth, t_sched, n_fb, n_sub = _run_scheduling(
            conv, hw, plan, sched_key, seed, cp_time
        )
        t_total = round(t_map + t_sched, 3)
        print(f" depth={depth:,}  ({t_sched:.1f}s)", end="")
        if n_fb:
            print(f"  [cpsat fallback: {n_fb}]", end="")
        print()

        if config_name == "naive":
            naive_depth = depth

        entry: dict = {
            "mapping":              mapping_key,
            "scheduler":            sched_key,
            "logical_depth":        depth,
            "mapping_time_sec":     t_map,
            "scheduling_time_sec":  t_sched,
            "total_time_sec":       t_total,
        }
        if is_sa:
            entry.update(sa_score_info)
        if is_sa or config_name not in ("naive",):  # cpsat quality info
            if sched_key == "cpsat":
                entry["cpsat_fallback_layers"]  = n_fb
                entry["cpsat_suboptimal_layers"] = n_sub

        config_results[config_name] = entry

    # ── Compute percent improvement vs naive ──────────────────────────────────
    if naive_depth and naive_depth > 0:
        for cname, entry in config_results.items():
            d = entry["logical_depth"]
            entry["pct_vs_naive"] = round(100.0 * (d - naive_depth) / naive_depth, 2)
    else:
        for entry in config_results.values():
            entry["pct_vs_naive"] = 0.0

    return {
        "hw_label":   spec.label(),
        "n_blocks":   n_blocks,
        "n_couplers": n_couplers,
        "fill_rate":  round(spec.actual_fill_rate, 4),
        "grid_rows":  spec.grid_rows,
        "grid_cols":  spec.grid_cols,
        "configs":    config_results,
    }


# ── Main per-circuit runner ───────────────────────────────────────────────────

def result_path(circuit_name: str, seed: int) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, f"{circuit_name}_seed{seed}.json")


def run_circuit(
    circuit_name: str,
    seed: int,
    cp_time: float,
    *,
    force: bool = False,
) -> dict:
    """
    Run all 5 configs × 2 topologies for one circuit. Saves and returns the result JSON.
    """
    out_path = result_path(circuit_name, seed)
    if not force and os.path.exists(out_path):
        print(f"[skip] {os.path.basename(out_path)} already exists (use --force to re-run)")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  Circuit: {circuit_name}  seed={seed}  cp_time={cp_time}s/layer")
    print(f"  SA: steps={SA_STEPS}  t0={SA_T0:.0e}  tend={SA_TEND}")
    print(f"  Hardware: n_data={N_DATA}  coupler_capacity={COUPLER_CAPACITY}")
    print(f"{'='*70}")

    conv, n_logicals, t_count, n_layers = load_pbc(circuit_name)
    print(f"  n_logicals={n_logicals}  t_count={t_count}  n_layers={n_layers}")

    topo_results: dict = {}
    for topology in TOPOLOGIES:
        topo_results[topology] = _run_topology(
            circuit_name, topology, conv, n_logicals, seed, cp_time
        )

    record = {
        "circuit":   circuit_name,
        "n_qubits":  n_logicals,
        "t_count":   t_count,
        "n_layers":  n_layers,
        "seed":      seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "n_data":           N_DATA,
            "coupler_capacity": COUPLER_CAPACITY,
            "sparse_pct":       SPARSE_PCT,
        },
        "sa_params": {
            "steps": SA_STEPS,
            "t0":    SA_T0,
            "tend":  SA_TEND,
        },
        "cp_sat_time_limit_per_layer": cp_time,
    }
    for topology in TOPOLOGIES:
        record[topology] = topo_results[topology]

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  [saved] {out_path}")

    # ── Quick summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {circuit_name}  seed={seed}")
    print(f"  {'Config':<8}  {'Mapping':<8}  {'Sched':<16}  "
          f"{'Grid depth':>10}  {'Grid %':>7}  {'Ring depth':>10}  {'Ring %':>7}")
    print(f"  {'-'*74}")
    for cname in CONFIGS:
        gc = topo_results["grid"]["configs"][cname]
        rc = topo_results["ring"]["configs"][cname]
        mapping_k, sched_k = CONFIGS[cname]
        print(f"  {cname:<8}  {mapping_k:<8}  {sched_k:<16}  "
              f"{gc['logical_depth']:>10,}  {gc['pct_vs_naive']:>+7.1f}%  "
              f"{rc['logical_depth']:>10,}  {rc['pct_vs_naive']:>+7.1f}%")
    print(f"{'='*70}")

    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PBC compiler paper evaluation — one circuit at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available PBC circuits:
  python experiments/run_experiment.py --list

  # Run Adder8 with defaults (seed=42, cp-time=1000s):
  python experiments/run_experiment.py --circuit Adder8

  # Custom seed and CP-SAT time limit:
  python experiments/run_experiment.py --circuit QFT32 --seed 1 --cp-time 300

  # Force re-run even if result exists:
  python experiments/run_experiment.py --circuit Adder16 --force
""",
    )
    parser.add_argument("--list",    action="store_true",
                        help="List all available PBC circuits and exit")
    parser.add_argument("--circuit", help="Circuit name to run")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cp-time", type=float, default=DEFAULT_CPSAT_TIME,
                        dest="cp_time",
                        help=f"CP-SAT time limit per layer in seconds "
                             f"(default: {DEFAULT_CPSAT_TIME})")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()

    if args.list:
        circuits = discover_circuits()
        if not circuits:
            print(f"No PBC files found in {PBC_DIR}")
            return
        print(f"Available circuits ({len(circuits)}) in {PBC_DIR}:\n")
        for name in sorted(circuits):
            print(f"  {name}")
        return

    if not args.circuit:
        parser.print_help()
        return

    run_circuit(
        args.circuit,
        seed=args.seed,
        cp_time=args.cp_time,
        force=args.force,
    )


if __name__ == "__main__":
    main()
