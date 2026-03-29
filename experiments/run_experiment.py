"""
Experiment orchestrator for the PBC compiler paper evaluation.

Runs (circuit × placement × scheduler × seed) configurations and saves
a JSON record per run to results/raw/.

Usage examples:

  # Phase 1: core benchmarks (circuits 1 & 2, all configs, 1 seed)
  python experiments/run_experiment.py --phase 1

  # Config E: 30-seed robustness run on qft_100_approx
  python experiments/run_experiment.py --circuit qft_100_approx --placement sa --scheduler cpsat --seeds 1-30

  # Phase 2: scaling sweep (QFT sizes, configs A and D only)
  python experiments/run_experiment.py --phase 2

  # Single run
  python experiments/run_experiment.py --circuit gf2_16_mult --placement sa --scheduler cpsat --seed 42

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

# ── Directory layout ──────────────────────────────────────────────────────────

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR   = os.path.join(_ROOT, "circuits", "benchmarks")
PBC_DIR     = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RESULTS_DIR = os.path.join(_ROOT, "results", "raw")

# ── Name mappings ─────────────────────────────────────────────────────────────

PLACEMENT_TO_MAPPER = {
    "random":  "pure_random",
    "greedy":  "auto_round_robin_mapping",
    "sa":      "simulated_annealing",
}

SCHEDULER_TO_FACTORY = {
    "greedy_critical": "greedy_critical",
    "cpsat":           "cp_sat",
}

# ── Circuits registry ─────────────────────────────────────────────────────────

# Maps circuit name → (qasm filename, expected n_qubits hint)
CIRCUITS = {
    "gf2_16_mult":             ("gf2_16_mult.qasm",             48),
    "qft_100_approx":          ("qft_100_approx.qasm",         100),
    "random_ct_500q_10k":      ("random_ct_500q_10k_validate.qasm", 500),
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
    Count rotations whose Pauli support spans ≥ 2 qLDPC blocks after placement.
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
        print(f"  [pbc] {circuit_name}: already cached at {pbc_path}")
        return pbc_path

    if not os.path.exists(qasm_path):
        raise FileNotFoundError(
            f"QASM not found: {qasm_path}\n"
            f"Run: python circuits/generate_benchmarks.py --all\n"
            f"  (or copy gf2_16_mult.qasm manually from VOQC-benchmarks)"
        )

    print(f"  [pbc] converting {fname} ...")
    qasm_str = load_qasm_file(qasm_path)
    conv     = GoSCConverter(verbose=False)
    program  = conv.convert_qasm(qasm_str)
    _        = conv.greedy_layering()

    t_count  = sum(1 for r in program.rotations if abs(r.angle) < math.pi / 2 - 1e-9)
    print(
        f"  [pbc] {circuit_name}: qubits={conv.num_qubits}"
        f"  T-count={t_count}  layers={len(conv.layers)}"
        f"  rotations={len(program.rotations)}"
    )

    os.makedirs(PBC_DIR, exist_ok=True)
    payload = conv.to_compact_payload()
    with open(pbc_path, "w") as f:
        json.dump(payload, f)
    print(f"  [pbc] saved → {pbc_path}")
    return pbc_path


def load_pbc(circuit_name: str):
    """Load a pre-built PBC JSON and return a GoSCConverter with data loaded."""
    pbc_path = os.path.join(PBC_DIR, f"{circuit_name}.json")
    if not os.path.exists(pbc_path):
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
    placement: str,
    scheduler: str,
    seed: int,
    *,
    sa_steps: int = 50_000,
    sa_t0: float = 1e5,
    sa_tend: float = 0.05,
    cp_sat_time_limit: float = 120.0,
    n_data: int = 11,
    topology: str = "grid",
    verbose: bool = True,
) -> dict:
    """
    Run one (circuit, placement, scheduler, seed) configuration.
    Returns a metrics dict matching the paper's output schema.
    """
    mapper_name = PLACEMENT_TO_MAPPER[placement]
    sched_name  = SCHEDULER_TO_FACTORY[scheduler]

    # ── Load PBC ──────────────────────────────────────────────────────────────
    conv = load_pbc(circuit_name)
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

    if verbose:
        print(
            f"  [{circuit_name}]  {placement}+{scheduler}  seed={seed}"
            f"  blocks={n_blocks}  mapper={mapper_name}"
        )

    t_start = time.perf_counter()

    # ── Mapping ───────────────────────────────────────────────────────────────
    map_cfg = MappingConfig(
        seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend,
    )
    plan = get_mapper(mapper_name).solve(
        MappingProblem(n_logicals=n_logicals),
        hw, map_cfg,
        {"rotations": conv.program.rotations, "verbose": False, "debug": False},
    )

    t_after_map = time.perf_counter()

    # ── Count inter-block rotations (placement-only metric) ───────────────────
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

    compile_time = time.perf_counter() - t_start

    if verbose:
        print(
            f"    inter_block={inter_block}  depth={total_depth}"
            f"  time={compile_time:.1f}s"
        )

    return {
        "circuit":               circuit_name,
        "n_qubits":              n_logicals,
        "n_blocks":              n_blocks,
        "t_count":               t_count,
        "placement":             placement,
        "scheduler":             scheduler,
        "seed":                  seed,
        "inter_block_rotations": inter_block,
        "logical_depth":         total_depth,
        "sa_iterations":         sa_steps if placement == "sa" else 0,
        "compile_time_sec":      round(compile_time, 3),
    }


# ── Result persistence ────────────────────────────────────────────────────────

def result_path(circuit: str, placement: str, scheduler: str, seed: int) -> str:
    return os.path.join(
        RESULTS_DIR, f"{circuit}_{placement}_{scheduler}_seed{seed}.json"
    )


def save_result(rec: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_path(rec["circuit"], rec["placement"], rec["scheduler"], rec["seed"])
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return path


def load_results(
    circuit: str | None = None,
    placement: str | None = None,
    scheduler: str | None = None,
) -> List[dict]:
    """Load all raw result JSONs, optionally filtered."""
    records = []
    if not os.path.isdir(RESULTS_DIR):
        return records
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            rec = json.load(f)
        if circuit  and rec.get("circuit")   != circuit:  continue
        if placement and rec.get("placement") != placement: continue
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


def _run_and_save(circuit, placement, scheduler, seed, skip_existing=True, **kwargs):
    path = result_path(circuit, placement, scheduler, seed)
    if skip_existing and os.path.exists(path):
        print(f"  [skip] {os.path.basename(path)} already exists")
        return
    rec = run_one_config(circuit, placement, scheduler, seed, **kwargs)
    out = save_result(rec)
    print(f"  [saved] {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PBC compiler experiment runner")
    parser.add_argument("--phase",     type=int, choices=[1, 2, 3],
                        help="Run a full experiment phase")
    parser.add_argument("--robustness", action="store_true",
                        help="Run Config E: 30 seeds of SA+CP-SAT on qft_100_approx")
    parser.add_argument("--circuit",   help="Single circuit name")
    parser.add_argument("--placement", choices=list(PLACEMENT_TO_MAPPER), help="Placement strategy")
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

    if args.build_pbc:
        print("Building PBC cache ...")
        for name in CIRCUITS:
            fname, _ = CIRCUITS[name]
            if os.path.exists(os.path.join(BENCH_DIR, fname)):
                try:
                    build_pbc(name)
                except Exception as e:
                    print(f"  [warn] {name}: {e}")
        return

    if args.phase == 1:
        print("=== Phase 1: Core benchmarks ===")
        for circuit in PHASE1_CIRCUITS:
            for placement, scheduler in PHASE1_CONFIGS:
                _run_and_save(circuit, placement, scheduler, seed=42,
                              skip_existing=skip, **kwargs)
        return

    if args.robustness:
        print("=== Config E: 30-seed robustness (SA+CP-SAT on qft_100_approx) ===")
        for seed in ROBUSTNESS_SEEDS:
            _run_and_save("qft_100_approx", "sa", "cpsat", seed=seed,
                          skip_existing=skip, **kwargs)
        return

    if args.phase == 2:
        print("=== Phase 2: Scaling sweep ===")
        for circuit in PHASE2_CIRCUITS:
            for placement, scheduler in PHASE2_CONFIGS:
                _run_and_save(circuit, placement, scheduler, seed=42,
                              skip_existing=skip, **kwargs)
        return

    if args.phase == 3:
        print("=== Phase 3: Large random circuit ===")
        for placement, scheduler in PHASE2_CONFIGS:
            _run_and_save("random_ct_500q_10k", placement, scheduler, seed=42,
                          skip_existing=skip, **kwargs)
        return

    # Single or multi-seed run
    if args.circuit and args.placement and args.scheduler:
        seeds = [args.seed]
        if args.seeds:
            lo, hi = args.seeds.split("-")
            seeds = list(range(int(lo), int(hi) + 1))
        for seed in seeds:
            _run_and_save(args.circuit, args.placement, args.scheduler, seed=seed,
                          skip_existing=skip, **kwargs)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
