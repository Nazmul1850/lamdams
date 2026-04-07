"""
experiments/sensitivity_analysis.py
Structured SA mapping sensitivity analysis following the diagnostic plan.

Phases
------
0  Diagnostic    - placement only (no scheduling): inter_block + SA energy convergence.
1  Hyperparam    - greedy scheduler, sweep T0 and T_end (cooling); 13 runs.
2  Robustness    - 30 seeds, best hyperparam, greedy; box-plot data.
3  CP-SAT valid  - best hyperparam, all 3 circuits, CP-SAT; cp_time configurable.
4  Correlation   - 20 random placements, inter_block vs depth scatter; compute R².
5  Scaling       - QFT family (qft_22..qft_99), configs A and D.

All results are written to results/sensitivity/<key>.json immediately after each run.
Existing result files are skipped automatically (use --force to recompute).
SA energy is logged at 0 / 25 / 50 / 75 / 100 % of iterations.
Scheduling prints progress after every layer.

Usage
-----
  python experiments/sensitivity_analysis.py --phase 0
  python experiments/sensitivity_analysis.py --phase 1 --circuit qft_100_approx
  python experiments/sensitivity_analysis.py --phase 2 --best-t0 1e5 --best-tend 0.05
  python experiments/sensitivity_analysis.py --phase 3 --cp-time 10000
  python experiments/sensitivity_analysis.py --phase 4
  python experiments/sensitivity_analysis.py --phase 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId,
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ShortestPathGatherRouting,
)
from modqldpc.mapping.algos.sa_mapping import (
    ScoreBreakdown,
    _random_move,
    _score,
    _undo_move,
)
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem, get_mapper
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem

# ── Directory layout ──────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR     = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RESULTS_DIR = os.path.join(_ROOT, "results", "sensitivity")

# ── Circuit registry ──────────────────────────────────────────────────────────
MAIN_CIRCUITS: Dict[str, str] = {
    "gf2_16_mult":    os.path.join(PBC_DIR, "gf2_16_mult.json"),
    "qft_100_approx": os.path.join(PBC_DIR, "qft_100_approx.json"),
    "rand_500_10k":   os.path.join(PBC_DIR, "rand_500_10k.json"),
}
SCALING_CIRCUITS = ["qft_22_approx", "qft_33_approx", "qft_44_approx",
                    "qft_66_approx", "qft_99_approx"]

# ── Known baselines (seed=42, production SA defaults) ─────────────────────────
BASELINES: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    ("gf2_16_mult", "random", "cpsat"):           {"depth": 18_936, "inter_block": 1_057},
    ("gf2_16_mult", "sa",     "cpsat"):           {"depth": 19_352, "inter_block": None},
    ("gf2_16_mult", "random", "greedy_critical"): {"depth": 24_893, "inter_block": 1_057},
    ("gf2_16_mult", "sa",     "greedy_critical"): {"depth": 23_689, "inter_block": None},
}

# ── SA production defaults (run_experiment.py) ────────────────────────────────
SA_PROD = dict(steps=50_000, t0=1e5, t_end=0.05)
N_DATA  = 11

# ── Phase 1 sweep grids ───────────────────────────────────────────────────────
# T0: initial temperature (5 values)
T0_SWEEP = [1e2, 1e4, 1e5, 1e6, 1e8]

# T_end: end temperature — controls effective cooling rate per step.
#   α_per_step = (T_end/T0)^(1/steps)
#   With T0=1e5, steps=50k:
#     T_end=0.001 → α≈0.99970  (aggressive, similar to prod 0.05)
#     T_end=0.05  → α≈0.99971  (production default)
#     T_end=5.0   → α≈0.99977  (milder cooling)
#     T_end=500   → α≈0.99990  (slow cooling)
#     T_end=5e4   → α≈0.99998  (very slow, nearly uniform)
TEND_SWEEP = [0.001, 0.05, 5.0, 500.0, 5e4]

# Steps sweep (use after finding best T0/T_end)
STEPS_SWEEP = [10_000, 20_000, 25_000, 30_000]


# ─────────────────────────────────────────────────────────────────────────────
# Circuit loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pbc(circuit_name: str):
    """Load PBC and return (n_logicals, rotations_list, conv)."""
    path = MAIN_CIRCUITS.get(circuit_name)
    if path is None:
        # Try scaling circuits
        path = os.path.join(PBC_DIR, f"{circuit_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PBC not found for '{circuit_name}' at {path}\n"
            "Run: python experiments/run_experiment.py --build-pbc"
        )
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(path)
    first_rot  = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    rotations  = list(conv.program.rotations)
    n_layers   = len(conv.layers)
    print(f"  [circuit] {circuit_name}  n_logicals={n_logicals}"
          f"  layers={n_layers}  rotations={len(rotations)}")
    return n_logicals, rotations, conv


# ─────────────────────────────────────────────────────────────────────────────
# Inter-block rotation count
# ─────────────────────────────────────────────────────────────────────────────

def count_inter_block(rotations: list, plan: MappingPlan, t_only: bool = False) -> int:
    """Count rotations whose Pauli support spans ≥ 2 blocks.

    t_only=True filters to T-gate (π/8) rotations only, matching run_experiment.py.
    """
    count = 0
    for rot in rotations:
        if t_only and not (abs(rot.angle) < math.pi / 2 - 1e-9):
            continue
        axis = rot.axis.lstrip("+-")
        n    = len(axis)
        blocks: set = set()
        for qi in range(n):
            if axis[n - 1 - qi] != "I":
                b = plan.logical_to_block.get(qi)
                if b is not None:
                    blocks.add(b)
        if len(blocks) >= 2:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# SA annealing with energy checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def _anneal_with_checkpoints(
    rotations: list,
    hw,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Dict[str, float],
    n_check: int = 5,
) -> Tuple[ScoreBreakdown, List[float], Dict]:
    """
    SA annealing that captures energy at n_check evenly-spaced checkpoints.

    Returns
    -------
    best_score     : ScoreBreakdown at best-ever mapping
    energy_log     : list of (pct, best_energy) tuples at each checkpoint
    best_map       : {logical_id: (block_id, local_id)} of best mapping
    """
    rng      = random.Random(seed)
    check_at = {int(i * (steps - 1) / (n_check - 1)) for i in range(n_check)}

    def snap() -> Dict:
        return {q: (hw.logical_to_block[q], hw.logical_to_local[q])
                for q in hw.logical_to_block}

    best_map = snap()
    cur      = _score(rotations, hw, **score_kwargs)
    best     = cur

    energy_log: List[Tuple[float, float]] = []  # (pct, best_energy)
    n_noop = n_accept = n_reject = 0

    for it in range(steps):
        frac = it / max(1, steps - 1)
        T    = t0 * ((t_end / t0) ** frac)

        if it in check_at:
            pct = 100.0 * it / max(1, steps - 1)
            energy_log.append((pct, best.total))
            alpha = (t_end / t0) ** (1.0 / max(1, steps - 1))
            print(f"  [SA {pct:5.1f}%]  T={T:.3f}  best={best.total:.1f}"
                  f"  cur={cur.total:.1f}  accept={n_accept}  reject={n_reject}"
                  f"  noop={n_noop}  (α≈{alpha:.6f})")
            n_noop = n_accept = n_reject = 0

        move = _random_move(hw, rng)
        if move[0] == "noop":
            n_noop += 1
            continue

        nxt   = _score(rotations, hw, **score_kwargs)
        delta = nxt.total - cur.total
        if delta <= 0 or rng.random() < math.exp(-delta / max(1e-12, T)):
            n_accept += 1
            cur = nxt
            if cur.total < best.total:
                best     = cur
                best_map = snap()
        else:
            n_reject += 1
            _undo_move(hw, move)

    # Restore best
    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    for q, (b, l) in best_map.items():
        hw.logical_to_block[q] = b
        hw.logical_to_local[q] = l

    return best, energy_log, best_map


# ─────────────────────────────────────────────────────────────────────────────
# Placement (SA or random, no scheduling)
# ─────────────────────────────────────────────────────────────────────────────

def run_placement(
    circuit_name: str,
    placement: str,
    seed: int,
    *,
    sa_steps: int = SA_PROD["steps"],
    sa_t0:    float = SA_PROD["t0"],
    sa_tend:  float = SA_PROD["t_end"],
    sa_score_kwargs: Optional[Dict[str, float]] = None,
    n_data: int = N_DATA,
) -> Dict[str, Any]:
    """Run placement only (no scheduling). Fast diagnostic tool."""
    n_logicals, rotations, conv = load_pbc(circuit_name)
    hw, _ = make_hardware(n_logicals, topology="grid", sparse_pct=0.0,
                          n_data=n_data, coupler_capacity=1)
    score_kw = sa_score_kwargs or {}
    t_sa = time.perf_counter()

    if placement == "random":
        plan = get_mapper("pure_random").solve(
            MappingProblem(n_logicals=n_logicals), hw,
            MappingConfig(seed=seed),
        )
        sa_metrics: Dict[str, Any] = {}

    elif placement == "sa":
        # Init with round-robin (same as production)
        get_mapper("auto_round_robin_mapping").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed)
        )
        init_score = _score(rotations, hw, **score_kw)
        print(f"  [SA] initial_energy={init_score.total:.1f}"
              f"  peak={init_score.peak_load_pen:.0f}"
              f"  span={init_score.span_pen:.1f}"
              f"  mst={init_score.mst_pen:.1f}")

        best_score, energy_log, best_map = _anneal_with_checkpoints(
            rotations, hw,
            steps=sa_steps, t0=sa_t0, t_end=sa_tend, seed=seed,
            score_kwargs=score_kw,
        )

        # Build MappingPlan from restored hw state
        plan = MappingPlan(
            dict(hw.logical_to_block), dict(hw.logical_to_local),
            meta={"best_score_total": best_score.total},
        )

        reduction_pct = (
            100.0 * (1.0 - best_score.total / max(1e-12, init_score.total))
            if init_score.total > 0 else 0.0
        )
        sa_metrics = {
            "sa_initial_energy": init_score.total,
            "sa_final_energy":   best_score.total,
            "sa_energy_reduction_pct": round(reduction_pct, 2),
            "sa_energy_checkpoints": [
                {"pct": pct, "best_energy": e} for pct, e in energy_log
            ],
        }
        print(f"  [SA] final_energy={best_score.total:.1f}"
              f"  reduction={reduction_pct:.1f}%")

    else:
        raise ValueError(f"Unknown placement: {placement!r}")

    elapsed = time.perf_counter() - t_sa

    iblk_all = count_inter_block(rotations, plan, t_only=False)
    iblk_t   = count_inter_block(rotations, plan, t_only=True)

    rec = {
        "circuit":                 circuit_name,
        "placement":               placement,
        "scheduler":               None,
        "seed":                    seed,
        "sa_steps":                sa_steps if placement == "sa" else None,
        "sa_t0":                   sa_t0    if placement == "sa" else None,
        "sa_tend":                 sa_tend  if placement == "sa" else None,
        "inter_block_all":         iblk_all,
        "inter_block_t_only":      iblk_t,
        "placement_time_sec":      round(elapsed, 3),
        "logical_depth":           None,
        "layer_depths":            [],
        "timestamp":               datetime.now(timezone.utc).isoformat(),
        **sa_metrics,
    }
    print(f"  [result] inter_block_all={iblk_all}  inter_block_T={iblk_t}"
          f"  time={elapsed:.1f}s")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: placement + scheduling with LayerExecutor
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    circuit_name: str,
    placement: str,
    scheduler: str,
    seed: int,
    *,
    sa_steps:         int   = SA_PROD["steps"],
    sa_t0:            float = SA_PROD["t0"],
    sa_tend:          float = SA_PROD["t_end"],
    sa_score_kwargs:  Optional[Dict[str, float]] = None,
    cp_time:          float = 120.0,
    n_data:           int   = N_DATA,
    phase:            str   = "misc",
    force:            bool  = False,
) -> Dict[str, Any]:
    """
    Full compilation run matching run_experiment.py (LayerExecutor + frame updates).
    Prints layer-by-layer scheduling progress and SA energy checkpoints.
    Result is saved to results/sensitivity/ and skipped on re-run.
    """
    rpath = _result_path(phase, circuit_name, placement, scheduler, seed,
                         sa_steps, sa_t0, sa_tend, sa_score_kwargs)
    if not force and os.path.exists(rpath):
        print(f"  [skip] {os.path.basename(rpath)}")
        with open(rpath) as f:
            return json.load(f)

    n_logicals, rotations, conv = load_pbc(circuit_name)
    hw, hw_spec = make_hardware(n_logicals, topology="grid", sparse_pct=0.0,
                                n_data=n_data, coupler_capacity=1)
    score_kw = sa_score_kwargs or {}
    t_start  = time.perf_counter()
    sa_metrics: Dict[str, Any] = {}

    # ── Placement ──────────────────────────────────────────────────────────────
    if placement == "random":
        plan = get_mapper("pure_random").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed),
        )
    elif placement == "sa":
        get_mapper("auto_round_robin_mapping").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed)
        )
        init_score = _score(rotations, hw, **score_kw)
        print(f"  [SA] initial_energy={init_score.total:.1f}")

        best_score, energy_log, _ = _anneal_with_checkpoints(
            rotations, hw,
            steps=sa_steps, t0=sa_t0, t_end=sa_tend, seed=seed,
            score_kwargs=score_kw,
        )
        plan = MappingPlan(
            dict(hw.logical_to_block), dict(hw.logical_to_local),
            meta={"best_score_total": best_score.total},
        )
        reduction_pct = (
            100.0 * (1.0 - best_score.total / max(1e-12, init_score.total))
            if init_score.total > 0 else 0.0
        )
        sa_metrics = {
            "sa_initial_energy":       init_score.total,
            "sa_final_energy":         best_score.total,
            "sa_energy_reduction_pct": round(reduction_pct, 2),
            "sa_energy_checkpoints":   [
                {"pct": pct, "best_energy": e} for pct, e in energy_log
            ],
        }
        print(f"  [SA] final_energy={best_score.total:.1f}"
              f"  reduction={reduction_pct:.1f}%")
    else:
        raise ValueError(f"Unknown placement: {placement!r}")

    t_after_map = time.perf_counter()

    # ── Inter-block count ──────────────────────────────────────────────────────
    iblk_all = count_inter_block(rotations, plan, t_only=False)
    iblk_t   = count_inter_block(rotations, plan, t_only=True)
    print(f"  [placement] inter_block_all={iblk_all}  inter_block_T={iblk_t}"
          f"  map_time={t_after_map - t_start:.1f}s")

    # ── Scheduling pipeline (matches run_experiment.py) ───────────────────────
    cost_fn  = make_gross_actual_cost_fn(plan, n_data=n_data)
    policies = LoweringPolicies(
        namer   = KeyNamer(),
        magic   = ChooseMagicBlockMinId(),
        routing = ShortestPathGatherRouting(),
        native  = HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    from modqldpc.core.types import PauliRotation
    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model = RandomOutcomeModel(seed=seed),
        frame_policy  = FrameUpdatePolicy(),
    )

    n_layers    = len(conv.layers)
    total_depth = 0
    layer_depths: List[int] = []
    sched_obj   = get_scheduler(scheduler)
    t_sched_start = time.perf_counter()

    print(f"  [sched] {scheduler}  cp_time={cp_time}s  layers={n_layers}")
    for layer_id, layer in enumerate(conv.layers):
        t_layer = time.perf_counter()
        res = lower_one_layer(
            layer_idx        = layer_id,
            rotations        = effective_rotations,
            rotation_indices = layer,
            hw               = hw,
            policies         = policies,
        )
        S = sched_obj.solve(SchedulingProblem(
            dag         = res.dag,
            hw          = hw,
            seed        = seed,
            policy_name = "incident_coupler_blocks_local",
            meta={
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": cp_time,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        ))
        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next  = [effective_rotations[i] for i in next_idxs]
        ex = executor.execute_layer(
            layer             = layer_id,
            dag               = res.dag,
            schedule          = S,
            frame_in          = frame,
            next_layer_rotations = rot_next,
        )
        for r in ex.next_rotations_effective:
            effective_rotations[r.idx] = r
        frame = ex.frame_after
        total_depth += ex.depth
        layer_depths.append(ex.depth)

        elapsed_layer = time.perf_counter() - t_layer
        elapsed_total = time.perf_counter() - t_sched_start
        print(f"  [layer {layer_id+1:>3}/{n_layers}]  "
              f"depth={ex.depth:>5}  cumulative={total_depth:>7}"
              f"  layer_time={elapsed_layer:.1f}s  total_sched={elapsed_total:.0f}s")

    compile_time = time.perf_counter() - t_start
    print(f"  [DONE]  total_depth={total_depth}  compile_time={compile_time:.1f}s")

    # ── Beat-baseline check ────────────────────────────────────────────────────
    baseline = BASELINES.get((circuit_name, "random", "cpsat"))
    if baseline and scheduler in ("cp_sat", "cpsat"):
        marker = " *** BEATS RANDOM+CPSAT ***" if total_depth < baseline["depth"] else ""
        print(f"  [baseline] random+cpsat={baseline['depth']}  "
              f"this={total_depth}{marker}")

    rec = {
        "circuit":               circuit_name,
        "placement":             placement,
        "scheduler":             scheduler,
        "seed":                  seed,
        "sa_steps":              sa_steps if placement == "sa" else None,
        "sa_t0":                 sa_t0    if placement == "sa" else None,
        "sa_tend":               sa_tend  if placement == "sa" else None,
        "sa_score_kwargs":       sa_score_kwargs,
        "inter_block_all":       iblk_all,
        "inter_block_t_only":    iblk_t,
        "logical_depth":         total_depth,
        "layer_depths":          layer_depths,
        "compile_time_sec":      round(compile_time, 3),
        "map_time_sec":          round(t_after_map - t_start, 3),
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        **sa_metrics,
    }
    _save_result(rec, rpath)
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Result caching helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sci(v: float) -> str:
    """Safe scientific notation for filenames: 1e+05 → 1ep05."""
    return f"{v:.1e}".replace("+", "p").replace("-", "m")


def _result_path(
    phase: str,
    circuit: str,
    placement: str,
    scheduler: Optional[str],
    seed: int,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    sa_score_kwargs: Optional[Dict[str, float]] = None,
) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sched_tag   = scheduler or "noscheduler"
    kw_tag      = ""
    if sa_score_kwargs:
        # Compact tag from non-default weight values
        kw_tag = "_kw=" + "-".join(
            f"{k.replace('W_','')}{_sci(v)}"
            for k, v in sorted(sa_score_kwargs.items())
            if v != 0.0
        )
    fname = (
        f"p{phase}_{circuit}_{placement}_{sched_tag}"
        f"_t0={_sci(sa_t0)}_tend={_sci(sa_tend)}_steps={sa_steps}"
        f"_seed{seed}{kw_tag}.json"
    )
    return os.path.join(RESULTS_DIR, fname)


def _save_result(rec: Dict[str, Any], path: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  [saved] {os.path.basename(path)}")


def _print_sep(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def phase0_diagnostic(circuit: str = "gf2_16_mult", seed: int = 42) -> None:
    """
    Cause diagnosis — no scheduling required (runs in minutes).

    Prints:
      Check 1: inter_block for random vs SA → determines Cause 1 vs Cause 2.
      Check 2: SA energy initial vs final → if barely decreased → Cause 1 (local min).
    """
    _print_sep(f"Phase 0 — Diagnostic  ({circuit}, seed={seed})")

    print("\n  [random placement]")
    r_random = run_placement(circuit, "random", seed=seed)

    print("\n  [SA placement  (production params)]")
    r_sa = run_placement(circuit, "sa", seed=seed,
                         sa_steps=SA_PROD["steps"],
                         sa_t0=SA_PROD["t0"],
                         sa_tend=SA_PROD["t_end"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DIAGNOSTIC SUMMARY  ({circuit})")
    print(f"{'─'*70}")
    print(f"  inter_block_all  random={r_random['inter_block_all']:>5}"
          f"   sa={r_sa['inter_block_all']:>5}"
          f"   delta={r_sa['inter_block_all'] - r_random['inter_block_all']:>+5}")
    print(f"  inter_block_T    random={r_random['inter_block_t_only']:>5}"
          f"   sa={r_sa['inter_block_t_only']:>5}"
          f"   delta={r_sa['inter_block_t_only'] - r_random['inter_block_t_only']:>+5}")

    init_e  = r_sa.get("sa_initial_energy", 0)
    final_e = r_sa.get("sa_final_energy", 0)
    red_pct = r_sa.get("sa_energy_reduction_pct", 0)
    print(f"\n  SA energy:  initial={init_e:.1f}  final={final_e:.1f}"
          f"  reduction={red_pct:.1f}%")

    if r_sa["inter_block_all"] >= r_random["inter_block_all"]:
        print("\n  DIAGNOSIS → Cause 1 (local minimum): SA inter_block >= random.")
        print("    SA is not finding a better placement. Try more steps or higher T0.")
    elif red_pct < 5.0:
        print("\n  DIAGNOSIS → Cause 1 (cooling too fast): energy barely decreased.")
        print("    SA is not exploring. Try higher sa_t0 or higher sa_tend.")
    else:
        print("\n  DIAGNOSIS → Cause 2 (energy-function misalignment) likely.")
        print("    SA found fewer inter_block rotations but CP-SAT depth was worse.")
        print("    The SA score is not well-correlated with depth on this circuit.")
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Hyperparameter sweep (greedy, fast)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_hyperparameter_sweep(
    circuit:   str   = "qft_100_approx",
    scheduler: str   = "greedy_critical",
    seed:      int   = 42,
    force:     bool  = False,
) -> List[Dict[str, Any]]:
    """
    Sweep T0 (5 values) and T_end (5 values) independently, plus 3 step counts.
    Uses greedy scheduler for speed. Total: 13 runs.
    """
    _print_sep(f"Phase 1 — Hyperparameter Sweep  ({circuit})")
    results: List[Dict[str, Any]] = []

    # ── Sweep 1: T0 (T_end and steps fixed at production) ────────────────────
    print("\n  [Sweep T0]  T_end=prod  steps=prod")
    for t0 in T0_SWEEP:
        alpha = (SA_PROD["t_end"] / t0) ** (1.0 / max(1, SA_PROD["steps"] - 1))
        print(f"\n  T0={t0:.1e}  T_end={SA_PROD['t_end']}  "
              f"α_per_step={alpha:.6f}")
        r = run_full_pipeline(
            circuit, "sa", scheduler, seed,
            sa_steps=SA_PROD["steps"], sa_t0=t0, sa_tend=SA_PROD["t_end"],
            phase="1a", force=force,
        )
        results.append(r)

    # ── Sweep 2: T_end (T0 and steps fixed at production) ────────────────────
    print("\n  [Sweep T_end]  T0=prod  steps=prod")
    for tend in TEND_SWEEP:
        alpha = (tend / SA_PROD["t0"]) ** (1.0 / max(1, SA_PROD["steps"] - 1))
        print(f"\n  T0={SA_PROD['t0']:.1e}  T_end={tend}  "
              f"α_per_step={alpha:.6f}")
        r = run_full_pipeline(
            circuit, "sa", scheduler, seed,
            sa_steps=SA_PROD["steps"], sa_t0=SA_PROD["t0"], sa_tend=tend,
            phase="1b", force=force,
        )
        results.append(r)

    # ── Sweep 3: steps (T0 and T_end fixed at production) ────────────────────
    print("\n  [Sweep steps]  T0=prod  T_end=prod")
    for steps in STEPS_SWEEP:
        print(f"\n  steps={steps}  T0={SA_PROD['t0']:.1e}  T_end={SA_PROD['t_end']}")
        r = run_full_pipeline(
            circuit, "sa", scheduler, seed,
            sa_steps=steps, sa_t0=SA_PROD["t0"], sa_tend=SA_PROD["t_end"],
            phase="1c", force=force,
        )
        results.append(r)

    _print_phase1_table(results)
    return results


def _print_phase1_table(results: List[Dict[str, Any]]) -> None:
    sep = "─" * 105
    print(f"\n{'='*105}")
    print(f"  Phase 1 results  (reference: greedy random depth from baselines)")
    print(sep)
    print(f"  {'label':<30}  {'depth':>7}  {'iblk_all':>8}  {'iblk_T':>6}"
          f"  {'sa_init':>12}  {'final':>12}  {'reduc%':>6}"
          f"  {'t_map':>7}  {'t_sched':>7}")
    print(sep)
    ref_random = BASELINES.get(("gf2_16_mult", "random", "greedy_critical"), {}).get("depth")
    for r in results:
        label = (f"t0={r.get('sa_t0',0):.1e}"
                 f" tend={r.get('sa_tend',0):.1e}"
                 f" steps={r.get('sa_steps',0)}")
        depth  = r.get("logical_depth") or 0
        iblk_a = r.get("inter_block_all", 0)
        iblk_t = r.get("inter_block_t_only", 0)
        init_e = r.get("sa_initial_energy") or 0
        fin_e  = r.get("sa_final_energy")   or 0
        red    = r.get("sa_energy_reduction_pct") or 0
        t_map  = r.get("map_time_sec") or 0
        t_tot  = r.get("compile_time_sec") or 0
        t_sch  = max(0, t_tot - t_map)
        print(f"  {label:<30}  {depth:>7}  {iblk_a:>8}  {iblk_t:>6}"
              f"  {init_e:>12.0f}  {fin_e:>12.0f}  {red:>6.1f}"
              f"  {t_map:>7.1f}  {t_sch:>7.1f}")
    print(f"{'='*105}")

    # Best by depth
    valid = [r for r in results if r.get("logical_depth")]
    if valid:
        best = min(valid, key=lambda r: r["logical_depth"])
        print(f"\n  Best depth={best['logical_depth']}"
              f"  t0={best['sa_t0']:.1e}"
              f"  tend={best['sa_tend']:.1e}"
              f"  steps={best['sa_steps']}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Robustness (30 seeds)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_robustness(
    circuit:   str   = "qft_100_approx",
    scheduler: str   = "greedy_critical",
    best_t0:   float = SA_PROD["t0"],
    best_tend: float = SA_PROD["t_end"],
    best_steps: int  = SA_PROD["steps"],
    n_seeds:   int   = 30,
    force:     bool  = False,
) -> None:
    """Run SA and random placement over n_seeds seeds. Produces box-plot data."""
    _print_sep(f"Phase 2 — Robustness  ({circuit}, {n_seeds} seeds)")

    sa_depths:  List[int] = []
    rnd_depths: List[int] = []
    sa_iblk:    List[int] = []
    rnd_iblk:   List[int] = []

    for seed in range(1, n_seeds + 1):
        print(f"\n  [seed {seed}/{n_seeds}]")

        r_sa = run_full_pipeline(
            circuit, "sa", scheduler, seed,
            sa_steps=best_steps, sa_t0=best_t0, sa_tend=best_tend,
            phase="2sa", force=force,
        )
        r_rnd = run_full_pipeline(
            circuit, "random", scheduler, seed,
            sa_steps=best_steps, sa_t0=best_t0, sa_tend=best_tend,
            phase="2rnd", force=force,
        )
        if r_sa.get("logical_depth"):
            sa_depths.append(r_sa["logical_depth"])
            sa_iblk.append(r_sa["inter_block_all"])
        if r_rnd.get("logical_depth"):
            rnd_depths.append(r_rnd["logical_depth"])
            rnd_iblk.append(r_rnd["inter_block_all"])

    _print_robustness_table(sa_depths, rnd_depths, sa_iblk, rnd_iblk)


def _print_robustness_table(sa_d, rnd_d, sa_i, rnd_i) -> None:
    def stats(xs):
        if not xs:
            return {}
        xs_s = sorted(xs)
        n = len(xs_s)
        return {
            "n": n, "min": xs_s[0], "max": xs_s[-1],
            "median": xs_s[n // 2],
            "q1": xs_s[n // 4], "q3": xs_s[3 * n // 4],
        }

    print(f"\n{'='*70}")
    print(f"  Phase 2 Robustness Summary")
    print(f"{'─'*70}")
    print(f"  {'metric':<20}  {'SA':>12}  {'random':>12}")
    print(f"{'─'*70}")
    for key, sa_vals, rnd_vals in [("depth", sa_d, rnd_d), ("inter_block", sa_i, rnd_i)]:
        ss, rs = stats(sa_vals), stats(rnd_vals)
        print(f"  {key+' min':<20}  {ss.get('min',0):>12}  {rs.get('min',0):>12}")
        print(f"  {key+' median':<20}  {ss.get('median',0):>12}  {rs.get('median',0):>12}")
        print(f"  {key+' max':<20}  {ss.get('max',0):>12}  {rs.get('max',0):>12}")
        print(f"  {key+' Q1-Q3':<20}  {ss.get('q1',0)}-{ss.get('q3',0):>7}  "
              f"{rs.get('q1',0)}-{rs.get('q3',0):>7}")
        print(f"{'─'*70}")

    # SA better than random count
    n_better = sum(1 for s, r in zip(sa_d, rnd_d) if s < r)
    print(f"  SA < random (depth): {n_better}/{len(sa_d)} seeds")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — CP-SAT validation on all 3 circuits
# ─────────────────────────────────────────────────────────────────────────────

def phase3_cpsat_validation(
    best_t0:    float = SA_PROD["t0"],
    best_tend:  float = SA_PROD["t_end"],
    best_steps: int   = SA_PROD["steps"],
    cp_time:    float = 10_000.0,
    seed:       int   = 42,
    force:      bool  = False,
) -> None:
    """
    Run configs C (SA+greedy) and D (SA+cpsat) on all 3 circuits with best
    hyperparameters. cp_time=10000 by default (generous limit for paper quality).
    Each layer is logged as it completes — safe to interrupt and resume.
    """
    _print_sep(f"Phase 3 — CP-SAT Validation  (cp_time={cp_time}s/layer)")
    circuits = list(MAIN_CIRCUITS.keys())
    results: List[Dict[str, Any]] = []

    for circuit in circuits:
        for placement, scheduler in [("sa", "greedy_critical"), ("sa", "cp_sat"),
                                      ("random", "greedy_critical"), ("random", "cp_sat")]:
            print(f"\n  [{circuit}  {placement}+{scheduler}]")
            r = run_full_pipeline(
                circuit, placement, scheduler, seed,
                sa_steps=best_steps, sa_t0=best_t0, sa_tend=best_tend,
                cp_time=cp_time, phase="3", force=force,
            )
            results.append(r)

    _print_phase3_table(results)


def _print_phase3_table(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'='*100}")
    print(f"  Phase 3 — Updated Table 1 (best hyperparameters)")
    print(f"{'─'*100}")
    print(f"  {'circuit':<20}  {'placement':>10}  {'scheduler':>15}"
          f"  {'depth':>7}  {'iblk_T':>6}  {'map_t':>7}  {'beat?':>5}")
    print(f"{'─'*100}")
    for r in results:
        depth = r.get("logical_depth") or 0
        iblk  = r.get("inter_block_t_only") or 0
        t_map = r.get("map_time_sec") or 0
        # Check vs random+cpsat baseline
        baseline = BASELINES.get((r["circuit"], "random", "cpsat"), {})
        beat = "YES" if (baseline.get("depth") and depth < baseline["depth"]) else "   "
        print(f"  {r['circuit']:<20}  {r['placement']:>10}  {r['scheduler']:>15}"
              f"  {depth:>7}  {iblk:>6}  {t_map:>7.1f}  {beat:>5}")
    print(f"{'='*100}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Energy-function correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def phase4_energy_correlation(
    circuit:   str  = "qft_100_approx",
    scheduler: str  = "greedy_critical",
    n_samples: int  = 20,
    force:     bool = False,
) -> None:
    """
    For n_samples random placements, record inter_block + depth.
    Compute Pearson R² to assess how well inter_block predicts depth.
    R² > 0.7 → energy function is fine; R² < 0.5 → misaligned (Cause 2).
    """
    _print_sep(f"Phase 4 — Energy Correlation  ({circuit}, {n_samples} samples)")
    xs: List[float] = []  # inter_block_all
    ys: List[float] = []  # logical_depth

    for seed in range(1, n_samples + 1):
        print(f"\n  [sample {seed}/{n_samples}]")
        r = run_full_pipeline(
            circuit, "random", scheduler, seed,
            sa_steps=SA_PROD["steps"], sa_t0=SA_PROD["t0"], sa_tend=SA_PROD["t_end"],
            phase="4", force=force,
        )
        if r.get("inter_block_all") and r.get("logical_depth"):
            xs.append(float(r["inter_block_all"]))
            ys.append(float(r["logical_depth"]))

    if len(xs) < 2:
        print("  Not enough data to compute correlation.")
        return

    # Pearson R
    n  = len(xs)
    mx = sum(xs) / n;  my = sum(ys) / n
    cov  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx   = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy   = math.sqrt(sum((y - my) ** 2 for y in ys))
    r    = cov / (sx * sy) if sx * sy > 0 else 0.0
    r2   = r ** 2

    print(f"\n{'─'*60}")
    print(f"  Correlation: inter_block_all vs logical_depth")
    print(f"  Pearson R = {r:.4f}   R² = {r2:.4f}  (n={n})")
    if r2 > 0.7:
        print("  → Energy function IS well-aligned (R² > 0.7).")
        print("    Problem is hyperparameter tuning (Cause 1).")
    elif r2 < 0.5:
        print("  → Energy function is NOT well-aligned (R² < 0.5).")
        print("    SA optimises the wrong proxy (Cause 2).")
    else:
        print("  → Moderate alignment (0.5 ≤ R² ≤ 0.7). Mixed causes.")
    print(f"{'─'*60}")

    # Save correlation data
    corr_path = os.path.join(RESULTS_DIR, f"p4_correlation_{circuit}.json")
    with open(corr_path, "w") as f:
        json.dump({"circuit": circuit, "scheduler": scheduler,
                   "inter_block": xs, "depth": ys, "r": r, "r2": r2}, f, indent=2)
    print(f"  [saved] {os.path.basename(corr_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Scaling sweep (QFT family)
# ─────────────────────────────────────────────────────────────────────────────

def phase5_scaling(
    best_t0:    float = SA_PROD["t0"],
    best_tend:  float = SA_PROD["t_end"],
    best_steps: int   = SA_PROD["steps"],
    seed:       int   = 42,
    cp_time:    float = 120.0,
    force:      bool  = False,
) -> None:
    """Scaling sweep over QFT family. Config A (random+greedy) and D (sa+cpsat)."""
    _print_sep("Phase 5 — Scaling Sweep  (QFT family)")

    available = [c for c in SCALING_CIRCUITS
                 if os.path.exists(os.path.join(PBC_DIR, f"{c}.json"))]
    if not available:
        print("  No QFT scaling PBC files found. Run --build-pbc first.")
        return

    print(f"  Circuits: {available}")
    results: List[Dict[str, Any]] = []

    for circuit in available:
        for placement, scheduler in [("random", "greedy_critical"),
                                      ("sa",     "cp_sat")]:
            print(f"\n  [{circuit}  {placement}+{scheduler}]")
            r = run_full_pipeline(
                circuit, placement, scheduler, seed,
                sa_steps=best_steps, sa_t0=best_t0, sa_tend=best_tend,
                cp_time=cp_time, phase="5", force=force,
            )
            results.append(r)

    print(f"\n{'─'*75}")
    print(f"  {'circuit':<22}  {'placement':>10}  {'scheduler':>15}  {'depth':>8}")
    print(f"{'─'*75}")
    for r in results:
        print(f"  {r['circuit']:<22}  {r['placement']:>10}  "
              f"{r['scheduler']:>15}  {r.get('logical_depth',0):>8}")
    print(f"{'─'*75}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured SA sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase",      type=int, required=True,
                        choices=[0, 1, 2, 3, 4, 5],
                        help="Which phase to run (0=diagnostic, 1=sweep, ...)")
    parser.add_argument("--circuit",    default="qft_100_approx",
                        help="Circuit for phases 1/2/4")
    parser.add_argument("--scheduler",  default="greedy_critical",
                        choices=["greedy_critical", "cp_sat"],
                        help="Scheduler for phases 1/2/4")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-seeds",    type=int, default=30,
                        help="Number of seeds for phase 2")
    parser.add_argument("--best-t0",    type=float, default=SA_PROD["t0"],
                        help="Best T0 from phase 1 (used in phases 2/3/5)")
    parser.add_argument("--best-tend",  type=float, default=SA_PROD["t_end"],
                        help="Best T_end from phase 1 (used in phases 2/3/5)")
    parser.add_argument("--best-steps", type=int,   default=SA_PROD["steps"],
                        help="Best step count from phase 1 (used in phases 2/3/5)")
    parser.add_argument("--cp-time",    type=float, default=10_000.0,
                        help="CP-SAT time limit per layer in seconds (phase 3)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run even if result JSON already exists")
    args = parser.parse_args()

    print(f"\n[sensitivity_analysis]  phase={args.phase}"
          f"  circuit={args.circuit}  seed={args.seed}"
          f"  force={args.force}")
    print(f"[SA defaults]  steps={SA_PROD['steps']}"
          f"  t0={SA_PROD['t0']:.1e}  t_end={SA_PROD['t_end']}")
    print(f"[results dir]  {RESULTS_DIR}\n")

    if args.phase == 0:
        # Run diagnostic on gf2_16_mult (the failing circuit) by default
        circuit = "gf2_16_mult" if args.circuit == "qft_100_approx" else args.circuit
        phase0_diagnostic(circuit=circuit, seed=args.seed)

    elif args.phase == 1:
        phase1_hyperparameter_sweep(
            circuit=args.circuit, scheduler=args.scheduler,
            seed=args.seed, force=args.force,
        )

    elif args.phase == 2:
        phase2_robustness(
            circuit=args.circuit, scheduler=args.scheduler,
            best_t0=args.best_t0, best_tend=args.best_tend,
            best_steps=args.best_steps,
            n_seeds=args.n_seeds, force=args.force,
        )

    elif args.phase == 3:
        phase3_cpsat_validation(
            best_t0=args.best_t0, best_tend=args.best_tend,
            best_steps=args.best_steps,
            cp_time=args.cp_time, seed=args.seed, force=args.force,
        )

    elif args.phase == 4:
        phase4_energy_correlation(
            circuit=args.circuit, scheduler=args.scheduler,
            force=args.force,
        )

    elif args.phase == 5:
        phase5_scaling(
            best_t0=args.best_t0, best_tend=args.best_tend,
            best_steps=args.best_steps,
            seed=args.seed, cp_time=args.cp_time, force=args.force,
        )


if __name__ == "__main__":
    main()
