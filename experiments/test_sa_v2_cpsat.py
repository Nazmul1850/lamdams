"""
experiments/test_sa_v2_cpsat.py

Single-circuit test comparing:
  mapper  : random  vs  SA-v2 (finalized weights)
  sched   : greedy_critical  vs  CPSATv2

Runs both grid and ring topologies by default.

Execution order per topology (for time efficiency):
  1. Build hw_random + hw_sa  (mapping mutates hw, so two separate instances)
  2. Run random mapping        (fast ~0.1s)
  3. Run SA-v2 mapping         (slow ~3-10s)
  4. Run greedy on random plan (fast)
  5. Run greedy on SA-v2 plan  (fast)
  6. Run cpsat on random plan  (slow)
  7. Run cpsat on SA-v2 plan   (slow)
  conv is shared across all scheduler runs (read-only during scheduling).

Compact progress → terminal.
Full details + per-layer CP-SAT status → results/sa_v2_final_tuning/per_circuit.log (appended).

Usage
-----
  python experiments/test_sa_v2_cpsat.py --list
  python experiments/test_sa_v2_cpsat.py --circuit Adder16
  python experiments/test_sa_v2_cpsat.py --circuit Adder16 --topology grid
  python experiments/test_sa_v2_cpsat.py --circuit QFT16 --cpsat-time 60
  python experiments/test_sa_v2_cpsat.py --circuit Adder16 --no-cpsat
  python experiments/test_sa_v2_cpsat.py --circuit Adder16 --sparse-pct 0.2
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

from modqldpc.core.types import PauliRotation
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId,
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ShortestPathGatherRouting,
)
from modqldpc.mapping.algos.sa_v2 import score_mapping_v2
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem, get_mapper
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.algos.cpsat_v2_scheduling import CPSATv2Scheduler
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR  = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
LOG_DIR  = os.path.join(_ROOT, "results", "sa_v2_final_tuning")
LOG_FILE = os.path.join(LOG_DIR, "per_circuit.log")

# ── Finalized SA-v2 weights ───────────────────────────────────────────────────

FINAL_SCORE_KWARGS: Dict[str, float] = {
    "W_UNUSED_BLOCKS": 1_000_000.0,
    "W_OCC_RANGE":        10_000.0,
    "W_OCC_STD":           5_000.0,
    "W_MULTI_BLOCK":           0.0,
    "W_SPAN":             10_000.0,
    "W_MST":                 500.0,
    "W_SPLIT":                10.0,
    "W_SUPPORT_PEAK":        100.0,
    "W_SUPPORT_RANGE":        20.0,
    "W_SUPPORT_STD":           0.0,
}

# ── SA / hardware defaults ────────────────────────────────────────────────────

SA_STEPS = 25_000
SA_T0    = 1e5
SA_TEND  = 5e-2
N_DATA   = 11   # Gross code data qubits per block

# ── Logging ───────────────────────────────────────────────────────────────────

_log_fh = None


def _open_log() -> None:
    global _log_fh
    os.makedirs(LOG_DIR, exist_ok=True)
    _log_fh = open(LOG_FILE, "a", encoding="utf-8")


def _close_log() -> None:
    global _log_fh
    if _log_fh is not None:
        _log_fh.flush()
        _log_fh.close()
        _log_fh = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _flog(msg: str) -> None:
    """Log file only."""
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def _tlog(msg: str) -> None:
    """Terminal only (compact progress)."""
    print(f"[{_ts()}] {msg}", flush=True)


def _both(msg: str) -> None:
    """Both terminal and log file."""
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    _flog(line)


def _fhr(char: str = "─", width: int = 80) -> None:
    _flog(char * width)


# ── Circuit helpers ───────────────────────────────────────────────────────────

def _discover_circuits() -> Dict[str, str]:
    circuits: Dict[str, str] = {}
    if not os.path.isdir(PBC_DIR):
        return circuits
    for fname in sorted(os.listdir(PBC_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = os.path.splitext(fname)[0]
        name = stem[:-4] if stem.endswith("_PBC") else stem
        circuits[name] = os.path.join(PBC_DIR, fname)
    return circuits


def _load_conv(name: str) -> GoSCConverter:
    circuits = _discover_circuits()
    if name not in circuits:
        raise FileNotFoundError(
            f"Circuit '{name}' not found.\nAvailable: {sorted(circuits)}"
        )
    c = GoSCConverter(verbose=False)
    c.load_cache_json(circuits[name])
    return c


def _circuit_stats(conv: GoSCConverter) -> Tuple[int, list, list]:
    """Return (n_logicals, all_rotations, t_rotations)."""
    first = next(iter(conv.program.rotations))
    n_logicals = len(first.axis.lstrip("+-"))
    rots = list(conv.program.rotations)
    t_rots = [r for r in rots if abs(r.angle) < math.pi / 2 - 1e-9]
    return n_logicals, rots, t_rots


def _count_interblock(rotations: list, plan: MappingPlan, *, t_only: bool = True) -> int:
    count = 0
    for rot in rotations:
        if t_only and not (abs(rot.angle) < math.pi / 2 - 1e-9):
            continue
        axis = rot.axis.lstrip("+-")
        n = len(axis)
        blocks: set = set()
        for qi in range(n):
            if axis[n - 1 - qi] != "I":
                b = plan.logical_to_block.get(qi)
                if b is not None:
                    blocks.add(b)
        if len(blocks) >= 2:
            count += 1
    return count


# ── Score logging ─────────────────────────────────────────────────────────────

def _flog_score(score, label: str, spec) -> None:
    """Write full score breakdown + hardware info to log file."""
    _fhr("·")
    _flog(f"  Score [{label}]  hw={spec.label()}")
    _flog(f"  fill={spec.actual_fill_rate*100:.1f}%  sparse_req={spec.sparse_pct*100:.1f}%  "
          f"capacity={spec.total_capacity}  blocks={spec.n_blocks}"
          + (f"  grid={spec.grid_rows}x{spec.grid_cols}" if spec.topology == "grid" else ""))
    _flog(f"  total={score.total:.2f}  unused_blocks={score.unused_blocks}  "
          f"active={score.active_blocks}")
    _flog(f"  occupancy  range={score.occupancy_range:.4f}(pen={score.occupancy_range_pen:.1f})  "
          f"std={score.occupancy_std:.4f}(pen={score.occupancy_std_pen:.1f})  "
          f"unused_pen={score.unused_block_pen:.1f}")
    _flog(f"  multiblock num={score.num_multiblock}(pen={score.multiblock_pen:.1f})  "
          f"span_total={score.span_total:.2f}(pen={score.span_pen:.1f})  "
          f"mean_blk={score.mean_blocks_touched:.3f}  max_blk={score.max_blocks_touched}")
    _flog(f"  mst        total={score.mst_total:.2f}(pen={score.mst_pen:.1f})  "
          f"mean={score.mean_mst:.3f}  max={score.max_mst:.3f}")
    _flog(f"  split      total={score.split_total:.2f}(pen={score.split_pen:.1f})  "
          f"mean={score.mean_split:.3f}  max={score.max_split:.3f}")
    _flog(f"  support    peak={score.support_peak:.2f}(pen={score.support_peak_pen:.1f})  "
          f"range={score.support_range:.2f}(pen={score.support_range_pen:.1f})  "
          f"std={score.support_std:.2f}(pen={score.support_std_pen:.1f})")


# ── Scheduling ────────────────────────────────────────────────────────────────

def _run_scheduling(
    conv: GoSCConverter,
    hw,
    plan: MappingPlan,
    scheduler_name: str,
    seed: int,
    *,
    label: str = "",
    cpsat_time_limit: float = 300.0,
    cpsat_log: bool = False,
) -> Tuple[int, float, List[dict], List[dict]]:
    """
    Run full scheduling loop over all layers.
    conv is read-only here (effective_rotations is rebuilt fresh from conv.program.rotations).
    Returns (total_depth, wall_sec, fallback_layers, suboptimal_layers).

    CP-SAT behaviour:
      OPTIMAL  → accept, log depth.
      FEASIBLE → accept as near-optimal (seeded from greedy hint, so ≥ greedy quality).
                 Record in suboptimal_layers with gap %.
      exception → fall back to greedy_critical for that layer only.
    """
    cost_fn  = make_gross_actual_cost_fn(plan, n_data=N_DATA)
    policies = LoweringPolicies(
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

    is_cpsat = (scheduler_name == "cpsat_v2")
    sched_obj      = CPSATv2Scheduler() if is_cpsat else get_scheduler(scheduler_name)
    fallback_sched = get_scheduler("greedy_critical")

    fallback_layers:   List[dict] = []
    suboptimal_layers: List[dict] = []
    total_depth = 0
    n_layers    = len(conv.layers)
    tag = f"[{label}]" if label else f"[{scheduler_name}]"

    _flog(f"\n{tag} scheduling  scheduler={scheduler_name}  layers={n_layers}"
          + (f"  limit={cpsat_time_limit}s/layer" if is_cpsat else ""))

    t_start = time.perf_counter()

    for layer_id, layer in enumerate(conv.layers):
        layer_t0 = time.perf_counter()
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )
        sched_problem = SchedulingProblem(
            dag=res.dag, hw=hw, seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": cpsat_time_limit if is_cpsat else None,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        cpsat_log,
            },
        )

        # Seed CP-SAT with greedy solution for faster convergence (Fix 3).
        if is_cpsat:
            try:
                g_hint = fallback_sched.solve(sched_problem)
                sched_problem.meta["greedy_hint"] = g_hint.meta.get("entries", {})
            except Exception:
                pass  # hint is optional; proceed without it

        try:
            S = sched_obj.solve(sched_problem)
            layer_t = round(time.perf_counter() - layer_t0, 2)

            if is_cpsat:
                status   = S.meta.get("cp_sat_status", "?")
                makespan = S.meta.get("cp_sat_makespan", "?")
                obj      = S.meta.get("cp_sat_obj", -1)
                obj_lb   = S.meta.get("cp_sat_obj_lb", -1)
                if status == "FEASIBLE":
                    gap_pct = (
                        round(100.0 * (obj - obj_lb) / max(obj_lb, 1), 1)
                        if obj >= 0 and obj_lb >= 0 else None
                    )
                    # Accept FEASIBLE as near-optimal — seeded from greedy hint so
                    # guaranteed ≥ greedy quality; discarding it would be wasteful.
                    _flog(f"  L{layer_id:03d}/{n_layers}  FEASIBLE(near-opt)"
                          f"  ms={makespan}  obj={obj}  lb={obj_lb}  gap={gap_pct}%  t={layer_t}s")
                    suboptimal_layers.append(
                        {"layer": layer_id, "obj": obj, "obj_lb": obj_lb, "gap_pct": gap_pct}
                    )
                else:
                    _flog(f"  L{layer_id:03d}/{n_layers}  {status}"
                          f"  ms={makespan}  obj={obj}  lb={obj_lb}  t={layer_t}s")

        except Exception as exc:
            if is_cpsat:
                layer_t = round(time.perf_counter() - layer_t0, 2)
                _flog(f"  L{layer_id:03d}/{n_layers}  FALLBACK→greedy  {exc!r}  t={layer_t}s")
                fallback_layers.append({"layer": layer_id, "reason": str(exc)})
                S = fallback_sched.solve(sched_problem)
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
        frame        = ex.frame_after
        total_depth += ex.depth

    wall_time = round(time.perf_counter() - t_start, 3)
    _flog(f"{tag} done  depth={total_depth}  t={wall_time}s"
          + (f"  fallbacks={len(fallback_layers)}  suboptimal={len(suboptimal_layers)}"
             if is_cpsat else ""))
    return total_depth, wall_time, fallback_layers, suboptimal_layers


# ── Mapping helper ────────────────────────────────────────────────────────────

def _run_mapping(
    mapper_name: str,
    n_logicals: int,
    conv: GoSCConverter,
    hw,          # mutated in-place
    seed: int,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    score_kwargs: Dict[str, float],
) -> Tuple[MappingPlan, float]:
    """Run mapper; return (plan, elapsed_sec). hw is mutated by the mapper."""
    cfg = MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend)
    meta: dict = {"rotations": conv.program.rotations, "verbose": False, "debug": False}
    if mapper_name == "sa_v2":
        meta["score_kwargs"] = score_kwargs
    t0   = time.perf_counter()
    plan = get_mapper(mapper_name).solve(MappingProblem(n_logicals=n_logicals), hw, cfg, meta)
    return plan, round(time.perf_counter() - t0, 3)


# ── Per-topology run ──────────────────────────────────────────────────────────

def _run_topology(
    circuit_name: str,
    conv: GoSCConverter,        # shared read-only across all scheduler calls
    rotations: list,
    n_logicals: int,
    topology: str,
    seed: int,
    score_kwargs: Dict[str, float],
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    sparse_pct: float,
    run_greedy: bool,
    run_cpsat: bool,
    cpsat_time: float,
    cpsat_log: bool,
) -> dict:
    """
    Full run for one topology.  Returns result dict with both mapper entries.

    Execution order (time-efficient):
      1. Build hw_random + hw_sa  (separate instances; mapping mutates hw)
      2. Map random  (fast)
      3. Map SA-v2   (slow)
      4. Greedy on random plan  (fast)
      5. Greedy on SA-v2 plan   (fast)
      6. CPSATv2 on random plan (slow)
      7. CPSATv2 on SA-v2 plan  (slow)
    conv is reused across steps 4-7 (read-only).
    """
    _fhr("─")
    _flog(f"\n  [{topology.upper()}]  circuit={circuit_name}  n_logicals={n_logicals}  "
          f"sparse={sparse_pct*100:.1f}%  seed={seed}")

    # ── 1. Build two hw instances (one per mapper) ────────────────────────────
    hw_rand, spec = make_hardware(
        n_logicals, topology=topology, sparse_pct=sparse_pct, n_data=N_DATA
    )
    hw_sa, _      = make_hardware(
        n_logicals, topology=topology, sparse_pct=sparse_pct, n_data=N_DATA
    )
    hw_label = spec.label()
    n_blocks = spec.n_blocks

    _flog(f"  hw={hw_label}  blocks={n_blocks}  capacity={spec.total_capacity}  "
          f"fill={spec.actual_fill_rate*100:.1f}%  "
          f"sparse_actual={((1-spec.actual_fill_rate)*100):.1f}%"
          + (f"  grid={spec.grid_rows}x{spec.grid_cols}" if topology == "grid" else ""))
    _tlog(f"  [{topology}] hw={hw_label}  mapping random+SA ...")

    # ── 2. Random mapping ─────────────────────────────────────────────────────
    plan_rand, t_rand = _run_mapping(
        "pure_random", n_logicals, conv, hw_rand, seed, sa_steps, sa_t0, sa_tend, score_kwargs,
    )
    score_rand     = score_mapping_v2(rotations, hw_rand, **score_kwargs)
    ibl_t_rand     = _count_interblock(rotations, plan_rand, t_only=True)
    ibl_all_rand   = _count_interblock(rotations, plan_rand, t_only=False)
    _flog(f"  [random] map_time={t_rand}s  score={score_rand.total:.2f}  "
          f"ibl_t={ibl_t_rand}  ibl_all={ibl_all_rand}")
    _flog_score(score_rand, f"random/{topology}", spec)

    # ── 3. SA-v2 mapping ──────────────────────────────────────────────────────
    plan_sa, t_sa = _run_mapping(
        "sa_v2", n_logicals, conv, hw_sa, seed, sa_steps, sa_t0, sa_tend, score_kwargs,
    )
    score_sa       = score_mapping_v2(rotations, hw_sa, **score_kwargs)
    ibl_t_sa       = _count_interblock(rotations, plan_sa, t_only=True)
    ibl_all_sa     = _count_interblock(rotations, plan_sa, t_only=False)
    _flog(f"  [sa_v2]  map_time={t_sa}s  score={score_sa.total:.2f}  "
          f"ibl_t={ibl_t_sa}  ibl_all={ibl_all_sa}")
    _flog_score(score_sa, f"sa_v2/{topology}", spec)

    _tlog(f"  [{topology}] random map={t_rand}s score={score_rand.total:.0f} ibl={ibl_t_rand} | "
          f"sa_v2 map={t_sa}s score={score_sa.total:.0f} ibl={ibl_t_sa}")

    # Result skeleton
    result: dict = {
        "topology":          topology,
        "hw_label":          hw_label,
        "n_blocks":          n_blocks,
        "sparse_pct_req":    sparse_pct,
        "sparse_pct_actual": round(1 - spec.actual_fill_rate, 4),
        "fill_rate":         round(spec.actual_fill_rate, 4),
        "total_capacity":    spec.total_capacity,
        "grid_rows":         spec.grid_rows,
        "grid_cols":         spec.grid_cols,
        "random": {
            "mapping_time_sec": t_rand,
            "inter_block_t":    ibl_t_rand,
            "inter_block_all":  ibl_all_rand,
            "score":            _score_dict(score_rand),
            "scheduling":       {},
        },
        "sa_v2": {
            "mapping_time_sec": t_sa,
            "inter_block_t":    ibl_t_sa,
            "inter_block_all":  ibl_all_sa,
            "score":            _score_dict(score_sa),
            "scheduling":       {},
        },
    }

    # ── 4-5. Greedy (fast; run both mappers first) ────────────────────────────
    if run_greedy:
        _tlog(f"  [{topology}] greedy_critical: random ...")
        g_d_rand, g_t_rand, _, _ = _run_scheduling(
            conv, hw_rand, plan_rand, "greedy_critical", seed,
            label=f"{topology}/random/greedy",
        )
        _tlog(f"  [{topology}] greedy_critical: sa_v2 ...")
        g_d_sa, g_t_sa, _, _ = _run_scheduling(
            conv, hw_sa, plan_sa, "greedy_critical", seed,
            label=f"{topology}/sa_v2/greedy",
        )
        result["random"]["scheduling"]["greedy_critical"] = {
            "logical_depth": g_d_rand, "scheduling_time": g_t_rand,
        }
        result["sa_v2"]["scheduling"]["greedy_critical"] = {
            "logical_depth": g_d_sa, "scheduling_time": g_t_sa,
        }
        _tlog(f"  [{topology}] greedy done  random_d={g_d_rand}({g_t_rand}s)  "
              f"sa_d={g_d_sa}({g_t_sa}s)")

    # ── 6-7. CPSATv2 (slow; run both mappers) ────────────────────────────────
    if run_cpsat:
        _tlog(f"  [{topology}] CPSATv2 (limit={cpsat_time}s/layer): random ...")
        c_d_rand, c_t_rand, fb_rand, sub_rand = _run_scheduling(
            conv, hw_rand, plan_rand, "cpsat_v2", seed,
            label=f"{topology}/random/cpsat",
            cpsat_time_limit=cpsat_time,
            cpsat_log=cpsat_log,
        )
        _tlog(f"  [{topology}] CPSATv2 random: depth={c_d_rand}  t={c_t_rand}s  "
              f"fallbacks={len(fb_rand)}  suboptimal={len(sub_rand)}")

        _tlog(f"  [{topology}] CPSATv2 (limit={cpsat_time}s/layer): sa_v2 ...")
        c_d_sa, c_t_sa, fb_sa, sub_sa = _run_scheduling(
            conv, hw_sa, plan_sa, "cpsat_v2", seed,
            label=f"{topology}/sa_v2/cpsat",
            cpsat_time_limit=cpsat_time,
            cpsat_log=cpsat_log,
        )
        _tlog(f"  [{topology}] CPSATv2 sa_v2:  depth={c_d_sa}  t={c_t_sa}s  "
              f"fallbacks={len(fb_sa)}  suboptimal={len(sub_sa)}")

        result["random"]["scheduling"]["cpsat_v2"] = {
            "logical_depth":     c_d_rand,
            "scheduling_time":   c_t_rand,
            "n_fallbacks":       len(fb_rand),
            "n_suboptimal":      len(sub_rand),
            "fallback_layers":   fb_rand,
            "suboptimal_layers": sub_rand,
        }
        result["sa_v2"]["scheduling"]["cpsat_v2"] = {
            "logical_depth":     c_d_sa,
            "scheduling_time":   c_t_sa,
            "n_fallbacks":       len(fb_sa),
            "n_suboptimal":      len(sub_sa),
            "fallback_layers":   fb_sa,
            "suboptimal_layers": sub_sa,
        }

    return result


def _score_dict(score) -> dict:
    return {
        "total":               score.total,
        "active_blocks":       score.active_blocks,
        "unused_blocks":       score.unused_blocks,
        "occupancy_range":     score.occupancy_range,
        "occupancy_std":       score.occupancy_std,
        "num_multiblock":      score.num_multiblock,
        "span_total":          score.span_total,
        "mean_blocks_touched": score.mean_blocks_touched,
        "max_blocks_touched":  score.max_blocks_touched,
        "mst_total":           score.mst_total,
        "mean_mst":            score.mean_mst,
        "max_mst":             score.max_mst,
        "split_total":         score.split_total,
        "mean_split":          score.mean_split,
        "max_split":           score.max_split,
        "support_peak":        score.support_peak,
        "support_range":       score.support_range,
        "support_std":         score.support_std,
        "penalties": {
            "unused_block_pen":    score.unused_block_pen,
            "occupancy_range_pen": score.occupancy_range_pen,
            "occupancy_std_pen":   score.occupancy_std_pen,
            "multiblock_pen":      score.multiblock_pen,
            "span_pen":            score.span_pen,
            "mst_pen":             score.mst_pen,
            "split_pen":           score.split_pen,
            "support_peak_pen":    score.support_peak_pen,
            "support_range_pen":   score.support_range_pen,
            "support_std_pen":     score.support_std_pen,
        },
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(
    circuit: str,
    seed: int,
    sparse_pct: float,
    topo_results: List[dict],
    run_greedy: bool,
    run_cpsat: bool,
) -> None:
    """Print aligned summary table to both terminal and log file."""
    lines: List[str] = []

    def emit(s: str) -> None:
        lines.append(s)
        print(s, flush=True)
        _flog(s)

    _fhr("═")
    _flog("═" * 80)

    header = (
        f"  SUMMARY  circuit={circuit}  seed={seed}  sparse={sparse_pct*100:.1f}%"
    )
    emit(header)

    # Column headers
    # fixed cols: topo(5) mapper(7) hw(21) score(12) ibl_t(5) map_s(6)
    col_hdr  = f"  {'topo':<5} {'mapper':<7} {'hardware':<21} {'score':>12} {'ibl_t':>5} {'map_s':>6}"
    col_sep  = "  " + "-" * 5 + " " + "-" * 7 + " " + "-" * 21 + " " + "-" * 12 + " " + "-" * 5 + " " + "-" * 6
    if run_greedy:
        col_hdr += f"  {'g_depth':>7} {'g_t':>5}"
        col_sep += "  " + "-" * 7 + " " + "-" * 5
    if run_cpsat:
        col_hdr += f"  {'c_depth':>7} {'c_t':>8} {'fall':>4} {'sub':>3}"
        col_sep += "  " + "-" * 7 + " " + "-" * 8 + " " + "-" * 4 + " " + "-" * 3
    if run_greedy and run_cpsat:
        col_hdr += f"  {'c/g':>6}"
        col_sep += "  " + "-" * 6

    emit(col_hdr)
    emit(col_sep)

    for r in topo_results:
        for mapper_key in ("random", "sa_v2"):
            m = r[mapper_key]
            g = m["scheduling"].get("greedy_critical", {})
            c = m["scheduling"].get("cpsat_v2", {})

            row = (
                f"  {r['topology']:<5} {mapper_key:<7} {r['hw_label']:<21} "
                f"{m['score']['total']:>12.0f} {m['inter_block_t']:>5} {m['mapping_time_sec']:>6.1f}"
            )
            if run_greedy:
                gd  = g.get("logical_depth", "—")
                gt  = g.get("scheduling_time", "—")
                row += f"  {str(gd):>7} {str(gt):>5}"
            if run_cpsat:
                cd  = c.get("logical_depth", "—")
                ct  = c.get("scheduling_time", "—")
                nfb = c.get("n_fallbacks", "—")
                nsb = c.get("n_suboptimal", "—")
                row += f"  {str(cd):>7} {str(ct):>8} {str(nfb):>4} {str(nsb):>3}"
            if run_greedy and run_cpsat:
                gd_v = g.get("logical_depth", 0)
                cd_v = c.get("logical_depth", 0)
                ratio = f"{cd_v/gd_v:.4f}" if gd_v and cd_v else "—"
                row += f"  {ratio:>6}"

            emit(row)

        # Blank line between topologies
        if r is not topo_results[-1]:
            emit("")

    _fhr("═")
    _flog("═" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare random vs SA-v2 mapping with greedy/CPSATv2 scheduling."
    )
    parser.add_argument("--circuit", "-c", default=None,
                        help="Circuit name (e.g. Adder16). Required unless --list.")
    parser.add_argument("--list", action="store_true",
                        help="List available circuits and exit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", choices=["grid", "ring", "both"], default="both",
                        help="Hardware topology (default: both).")
    parser.add_argument("--sparse-pct", type=float, default=0.0,
                        help="Fraction of hw capacity left empty, e.g. 0.2 = 20%% sparse (default: 0).")
    parser.add_argument("--cpsat-time", type=float, default=300.0,
                        help="CP-SAT time limit per layer in seconds (default: 300).")
    parser.add_argument("--no-cpsat",   action="store_true", help="Skip CP-SAT scheduling.")
    parser.add_argument("--no-greedy",  action="store_true", help="Skip greedy_critical.")
    parser.add_argument("--cpsat-log",  action="store_true",
                        help="Write CP-SAT search log to log file (verbose).")
    parser.add_argument("--weights-json", default=None,
                        help="JSON file with SA-v2 score weight overrides.")
    parser.add_argument("--sa-steps", type=int,   default=SA_STEPS)
    parser.add_argument("--sa-t0",    type=float, default=SA_T0)
    parser.add_argument("--sa-tend",  type=float, default=SA_TEND)
    args = parser.parse_args()

    if args.list:
        circuits = _discover_circuits()
        print("Available circuits:")
        for name in sorted(circuits):
            print(f"  {name}")
        return

    if args.circuit is None:
        parser.error("--circuit is required (or use --list).")

    topologies  = ["grid", "ring"] if args.topology == "both" else [args.topology]
    score_kwargs = dict(FINAL_SCORE_KWARGS)
    if args.weights_json:
        with open(args.weights_json, "r") as f:
            score_kwargs.update(json.load(f))

    _open_log()
    run_ts = datetime.now(timezone.utc).isoformat()

    try:
        # Header in log
        _fhr("═")
        _flog(f"  RUN  {run_ts}")
        _flog(f"  circuit={args.circuit}  topologies={topologies}  seed={args.seed}  "
              f"sparse_pct={args.sparse_pct*100:.1f}%")
        _flog(f"  SA: steps={args.sa_steps}  t0={args.sa_t0}  tend={args.sa_tend}")
        _flog(f"  Schedulers: greedy={'yes' if not args.no_greedy else 'no'}  "
              f"cpsat_v2={'yes' if not args.no_cpsat else 'no'}  "
              f"cpsat_time_limit={args.cpsat_time}s/layer  "
              f"feasible=near-opt(accepted)")
        _flog(f"  Weights: {json.dumps(score_kwargs)}")
        _fhr("═")

        _tlog(f"=== {args.circuit}  topologies={topologies}  seed={args.seed}  "
              f"sparse={args.sparse_pct*100:.0f}% ===")

        # Load circuit once (conv is read-only during scheduling; reuse across topologies)
        conv = _load_conv(args.circuit)
        n_logicals, rotations, t_rotations = _circuit_stats(conv)
        n_layers = len(conv.layers)
        _flog(f"  n_logicals={n_logicals}  rotations={len(rotations)}"
              f"  T-gates={len(t_rotations)}  layers={n_layers}")
        _tlog(f"  loaded: logicals={n_logicals}  T-gates={len(t_rotations)}  layers={n_layers}")

        topo_results: List[dict] = []
        for topo in topologies:
            res = _run_topology(
                circuit_name=args.circuit,
                conv=conv,
                rotations=rotations,
                n_logicals=n_logicals,
                topology=topo,
                seed=args.seed,
                score_kwargs=score_kwargs,
                sa_steps=args.sa_steps,
                sa_t0=args.sa_t0,
                sa_tend=args.sa_tend,
                sparse_pct=args.sparse_pct,
                run_greedy=not args.no_greedy,
                run_cpsat=not args.no_cpsat,
                cpsat_time=args.cpsat_time,
                cpsat_log=args.cpsat_log,
            )
            topo_results.append(res)

        _print_summary(
            circuit=args.circuit,
            seed=args.seed,
            sparse_pct=args.sparse_pct,
            topo_results=topo_results,
            run_greedy=not args.no_greedy,
            run_cpsat=not args.no_cpsat,
        )

        # Full JSON to log only (not terminal)
        full = {
            "circuit":     args.circuit,
            "seed":        args.seed,
            "sparse_pct":  args.sparse_pct,
            "n_logicals":  n_logicals,
            "n_rotations": len(rotations),
            "t_gates":     len(t_rotations),
            "n_layers":    n_layers,
            "sa_steps":    args.sa_steps,
            "sa_t0":       args.sa_t0,
            "sa_tend":     args.sa_tend,
            "score_kwargs": score_kwargs,
            "topologies":  topo_results,
            "timestamp":   run_ts,
        }
        _flog("\nFull JSON:")
        _flog(json.dumps(full, indent=2))
        _flog("")

        _tlog(f"Done. Log → {LOG_FILE}")

    finally:
        _close_log()


if __name__ == "__main__":
    main()
