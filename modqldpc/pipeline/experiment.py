"""
Multi-combo experiment runner.

Provides:
  _run_combo()             — run one (mapper, scheduler) pair, return profiles
  run_algo_comparison()    — sweep 9 combos (3 mappers × 3 schedulers)
  run_sparse_dense()       — compare dense vs sparse hardware for one combo
  run_topology_gallery()   — draw hardware topology for grid/ring × dense/sparse
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

from modqldpc.frontend.extract_pauli import GoSCConverter
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
from modqldpc.mapping.model import HardwareGraph
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem
from modqldpc.pipeline.profiling import (
    CircuitProfile, LayerProfile,
    collect_circuit_profile, collect_layer_profile,
)
from modqldpc.pipeline.viz import (
    plot_algo_comparison,
    plot_sparse_dense_comparison,
)

# ── Canonical combo lists ─────────────────────────────────────────────────────

MAPPER_NAMES: List[str] = [
    "auto_round_robin_mapping",
    "pure_random",
    "simulated_annealing",
]

SCHEDULER_NAMES: List[str] = [
    "sequential_scheduler",
    "greedy_critical",
    "cp_sat",
]

# Short display labels for axes
MAPPER_LABELS: Dict[str, str] = {
    "auto_round_robin_mapping": "RoundRobin",
    "pure_random":              "PureRandom",
    "simulated_annealing":      "SA",
}
SCHEDULER_LABELS: Dict[str, str] = {
    "sequential_scheduler": "Sequential",
    "greedy_critical":      "GreedyCritical",
    "cp_sat":               "CPSAT",
}


@dataclass
class ComboResult:
    mapper_name: str
    sched_name: str
    total_depth: int
    circuit_profile: CircuitProfile
    layer_profiles: List[LayerProfile]


# ── Internal: import cost-fn builder from run_one to avoid duplication ────────

def _make_cost_fn(plan):
    """Borrow the gross-code cost fn from run_one; falls back to heuristic."""
    try:
        from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
        return make_gross_actual_cost_fn(plan)
    except Exception:
        from modqldpc.pipeline.run_one import normal_rotation_cost_fn
        return normal_rotation_cost_fn


def _run_combo(
    conv: GoSCConverter,
    hw: HardwareGraph,
    mapper_name: str,
    sched_name: str,
    n_logicals: int,
    *,
    seed: int = 42,
    cp_sat_time_limit: float = 30.0,
) -> ComboResult:
    """Run one (mapper, scheduler) combo and return a ComboResult."""
    cfg = MappingConfig(seed=seed, sa_steps=10_000, sa_t0=1e5, sa_tend=1.1)
    problem = MappingProblem(n_logicals=n_logicals)
    mapper = get_mapper(mapper_name)
    plan = mapper.solve(problem, hw, cfg, {
        "rotations": conv.program.rotations,
        "verbose": False,
        "debug": False,
    })

    cost_fn = _make_cost_fn(plan)
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )

    circuit_profile = collect_circuit_profile(
        n_logicals=n_logicals,
        layers=conv.layers,
        rotations=effective_rotations,
        hw=hw,
        plan=plan,
    )
    layer_profiles: List[LayerProfile] = []
    total_depth = 0

    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )

        sched = get_scheduler(sched_name)
        sched_problem = SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time": 0,
                "layer_idx": layer_id,
                "tie_breaker": "duration",
                "cp_sat_time_limit": cp_sat_time_limit,
                "debug_decode": False,
                "safe_fill": True,
                "cp_sat_log": False,
            },
        )
        S = sched.solve(sched_problem)

        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next = [effective_rotations[i] for i in next_idxs]
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

        lp = collect_layer_profile(
            layer_id=layer_id,
            rotation_indices=layer,
            effective_rotations=effective_rotations,
            res=res,
            S=S,
            ex=ex,
            hw=hw,
            plan=plan,
        )
        layer_profiles.append(lp)
        total_depth += ex.depth

    return ComboResult(
        mapper_name=mapper_name,
        sched_name=sched_name,
        total_depth=total_depth,
        circuit_profile=circuit_profile,
        layer_profiles=layer_profiles,
    )


# ── Phase 8: 3 × 3 algorithm comparison ──────────────────────────────────────

def run_algo_comparison(
    pbc_path: str,
    fig_dir: str,
    n_logicals: int,
    *,
    sparse_pct: float = 0.0,
    seed: int = 42,
    cp_sat_time_limit: float = 30.0,
    verbose: bool = True,
) -> List[ComboResult]:
    """
    Run all 9 mapper × scheduler combos on a fixed dense grid hardware.

    Returns a list of ComboResult (one per combo).
    Saves fig_08_algo_comparison.png to fig_dir.
    """
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)
    hw, _ = make_hardware(n_logicals, topology="grid", sparse_pct=sparse_pct)

    results: List[ComboResult] = []
    n_total = len(MAPPER_NAMES) * len(SCHEDULER_NAMES)
    done = 0
    for mapper_name in MAPPER_NAMES:
        for sched_name in SCHEDULER_NAMES:
            done += 1
            if verbose:
                label = f"{MAPPER_LABELS[mapper_name]} + {SCHEDULER_LABELS[sched_name]}"
                print(f"  [{done}/{n_total}] {label} ...", end=" ", flush=True)
            cr = _run_combo(
                conv, hw, mapper_name, sched_name, n_logicals,
                seed=seed, cp_sat_time_limit=cp_sat_time_limit,
            )
            results.append(cr)
            if verbose:
                print(f"depth={cr.total_depth}")

    fig = plot_algo_comparison(
        results,
        mapper_labels=MAPPER_LABELS,
        sched_labels=SCHEDULER_LABELS,
        save_path=str(pathlib.Path(fig_dir) / "fig_08_algo_comparison.png"),
    )
    fig.clf()
    if verbose:
        print("  [done] fig_08_algo_comparison.png")

    return results


# ── Phase 9: dense vs sparse hardware comparison ──────────────────────────────

def run_sparse_dense_comparison(
    pbc_path: str,
    fig_dir: str,
    n_logicals: int,
    *,
    mapper_name: str = "auto_round_robin_mapping",
    sched_name: str = "greedy_critical",
    sparse_pct: float = 0.5,
    seed: int = 42,
    cp_sat_time_limit: float = 30.0,
    verbose: bool = True,
) -> Tuple[ComboResult, ComboResult]:
    """
    Run one combo on dense (sparse_pct=0) and sparse (sparse_pct=sparse_pct) hardware.

    Returns (dense_result, sparse_result).
    Saves fig_09_sparse_dense.png to fig_dir.
    """
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)

    hw_dense, spec_dense = make_hardware(n_logicals, topology="grid", sparse_pct=0.0)
    hw_sparse, spec_sparse = make_hardware(n_logicals, topology="grid", sparse_pct=sparse_pct)

    if verbose:
        print(f"  Dense  hardware: {spec_dense.label()}")
        print(f"  Sparse hardware: {spec_sparse.label()}")

    if verbose:
        print(f"  Running dense  ({MAPPER_LABELS[mapper_name]} + {SCHEDULER_LABELS[sched_name]}) ...", end=" ", flush=True)
    dense_result = _run_combo(
        conv, hw_dense, mapper_name, sched_name, n_logicals,
        seed=seed, cp_sat_time_limit=cp_sat_time_limit,
    )
    if verbose:
        print(f"depth={dense_result.total_depth}")

    if verbose:
        print(f"  Running sparse ({MAPPER_LABELS[mapper_name]} + {SCHEDULER_LABELS[sched_name]}) ...", end=" ", flush=True)
    sparse_result = _run_combo(
        conv, hw_sparse, mapper_name, sched_name, n_logicals,
        seed=seed, cp_sat_time_limit=cp_sat_time_limit,
    )
    if verbose:
        print(f"depth={sparse_result.total_depth}")

    fig = plot_sparse_dense_comparison(
        dense_result, sparse_result,
        dense_label=f"Dense  {spec_dense.label()}",
        sparse_label=f"Sparse {spec_sparse.label()}",
        save_path=str(pathlib.Path(fig_dir) / "fig_09_sparse_dense.png"),
    )
    fig.clf()
    if verbose:
        print("  [done] fig_09_sparse_dense.png")

    return dense_result, sparse_result


# ── Fig 10: hardware topology gallery ────────────────────────────────────────

def run_topology_gallery(
    fig_dir: str,
    n_logicals: int,
    *,
    sparse_pct: float = 0.5,
    n_data: int = 11,
    verbose: bool = True,
) -> None:
    """
    Build all 4 hardware configs (grid/ring × dense/sparse) and plot a
    node-edge topology diagram for each as fig_10_hardware_topology.png.

    No pipeline execution is needed — this only builds the HardwareGraph
    objects and visualises their topology.
    """
    from modqldpc.pipeline.viz import plot_hardware_gallery

    configs_spec = [
        ("grid", 0.0,       f"Grid · dense\n(fill=100%)"),
        ("grid", sparse_pct, f"Grid · sparse\n(fill={int((1-sparse_pct)*100)}%)"),
        ("ring", 0.0,       f"Ring · dense\n(fill=100%)"),
        ("ring", sparse_pct, f"Ring · sparse\n(fill={int((1-sparse_pct)*100)}%)"),
    ]

    configs = []
    for topo, spct, base_label in configs_spec:
        hw, spec = make_hardware(
            n_logicals, topology=topo, sparse_pct=spct, n_data=n_data,
        )
        label = f"{base_label}\n{spec.label()}"
        configs.append((hw, spec, None, label))
        if verbose:
            print(f"  {topo:4s}  sparse={spct:.0%}: {spec.label()}")

    fig = plot_hardware_gallery(
        configs,
        save_path=str(pathlib.Path(fig_dir) / "fig_10_hardware_topology.png"),
    )
    fig.clf()
    if verbose:
        print("  [done] fig_10_hardware_topology.png")
