"""
Temporary scratch script — SA mapping parameter + weight sensitivity with actual depth.

For each configuration, runs:
  1. SA mapping  (standard or custom score weights)
  2. Greedy scheduling + layer execution  →  total_depth (the real metric)

Then builds a 4×3 figure:
  Row 0  param → depth       (steps / t0 / tend line plots)
  Row 1  score ↔ depth       (scatter: SA score vs total_depth)
  Row 2  weight → depth      (W_PEAK / W_SPLIT / W_RANGE line plots)
  Row 3  summary bar + W_PEAK × W_SPLIT heatmap

Usage:
    python test_sa_mapping.py
    python test_sa_mapping.py --pbc path/to/PBC.json --topology ring
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId, HeuristicRepeatNativePolicy,
    LoweringPolicies, ShortestPathGatherRouting,
)
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import (
    MappingConfig, MappingPlan, MappingProblem, get_mapper,
)
from modqldpc.mapping.algos.sa_mapping import (
    _random_move, _score, _undo_move, ScoreBreakdown,
)
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem


# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_PBC = "runs/rand_50q_1kt__seed42__2026-03-12T06-06-52Z/stage_frontend/PBC.json"
OUT_FIG         = "test_sa_comparison.png"
OUT_FIG_WEIGHTS = "test_sa_weights.png"
OUT_FIG_GAP     = "test_sa_gap_sweep.png"

_SYNCH_DIR  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "modqldpc", "rotation_synch"
)


# ── Singleton: rotation synthesis DB (loaded once) ───────────────────────────
_GROSS_CACHE: Optional[Tuple[Any, Any]] = None

def _gross_synth() -> Tuple[Any, Any]:
    global _GROSS_CACHE
    if _GROSS_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "gross_clifford", os.path.join(_SYNCH_DIR, "gross_clifford.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gross_clifford"] = mod
        spec.loader.exec_module(mod)
        synth = mod.GrossCliffordSynth.load_precomputed(_SYNCH_DIR)
        _GROSS_CACHE = (mod, synth)
        print("[synth]  rotation synthesis database loaded")
    return _GROSS_CACHE


def make_cost_fn(plan: MappingPlan, n_data: int = 11):
    mod, synth = _gross_synth()
    def cost_fn(_b, ops: Dict[int, str], _h) -> int:
        del _b, _h
        chars = ["I"] * n_data
        for lid, axis in ops.items():
            local_id = plan.logical_to_local.get(lid)
            if local_id is not None and local_id < n_data:
                chars[local_id] = axis
        if all(c == "I" for c in chars):
            return 1
        return int(synth.rotation_cost(mod.pauli_to_mask("".join(chars))))
    return cost_fn


# ── Circuit loading ───────────────────────────────────────────────────────────
def load_circuit(pbc_path: str):
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)
    first_rot = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.to_label().lstrip("+-"))
    rotations  = list(conv.program.rotations)
    layers     = conv.layers          # List[List[int]]
    print(f"[circuit]  n_logicals={n_logicals}"
          f"  layers={len(layers)}  rotations={len(rotations)}")
    return n_logicals, rotations, layers


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RunResult:
    label:         str
    # SA metrics (scored under the weights used during annealing)
    sa_total:      float
    sa_peak:       float
    sa_range:      float
    sa_split:      float
    sa_mst:        float
    sa_disconnected: float
    sa_span:       float
    num_multiblock:  int
    num_disconnected: int
    # Default-weight score on the final mapping (fair cross-config comparison)
    default_total: float
    # Depth (the real metric)
    total_depth:   int
    layer_depths:  List[int] = field(default_factory=list)
    # Timing
    elapsed_sa_s:    float = 0.0
    elapsed_depth_s: float = 0.0
    # Extra info
    hw_label: str = ""
    params:   Dict[str, Any] = field(default_factory=dict)


# ── Custom annealing with configurable score weights ─────────────────────────
def _anneal_weighted(
    rotations: list,
    hw,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Dict[str, float],
    verbose: bool = False,
    report_every: int = 2_000,
) -> ScoreBreakdown:
    """
    SA identical to _anneal but passes score_kwargs to _score so that
    individual W_* weights can be varied without modifying the library.
    """
    rng = random.Random(seed)

    def snap() -> Dict:
        return {q: (hw.logical_to_block[q], hw.logical_to_local[q])
                for q in hw.logical_to_block}

    best_map = snap()
    cur      = _score(rotations, hw, **score_kwargs)
    best     = cur
    n_noop = n_accept = n_reject = 0

    for it in range(1, steps + 1):
        frac = (it - 1) / max(1, steps - 1)
        T    = t0 * ((t_end / t0) ** frac)

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

        if verbose and it % report_every == 0:
            print(f"  [SA-w] it={it:6d}  T={T:.4f}  cur={cur.total:.1f}"
                  f"  best={best.total:.1f}  accept={n_accept} reject={n_reject}")
            n_noop = n_accept = n_reject = 0

    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    for q, (b, l) in best_map.items():
        hw.logical_to_block[q] = b
        hw.logical_to_local[q] = l

    return best


# ── Depth calculation (greedy schedule over all layers) ───────────────────────
def _compute_depth(
    rotations: list,
    layers: list,
    hw,
    plan: MappingPlan,
    n_data: int,
    sched_name: str,
    seed: int,
    cp_sat_time_limit: float,
) -> Tuple[int, List[int]]:
    policies = LoweringPolicies(
        namer   = KeyNamer(),
        magic   = ChooseMagicBlockMinId(),
        routing = ShortestPathGatherRouting(),
        native  = HeuristicRepeatNativePolicy(cost_fn=make_cost_fn(plan, n_data)),
    )
    effective_rotations = {r.idx: r for r in rotations}
    total_depth  = 0
    layer_depths = []

    for layer_id, layer_rot_ids in enumerate(layers):
        res = lower_one_layer(
            layer_idx        = layer_id,
            rotations        = effective_rotations,
            rotation_indices = layer_rot_ids,
            hw               = hw,
            policies         = policies,
        )
        sched_obj = get_scheduler(sched_name)
        S = sched_obj.solve(SchedulingProblem(
            dag         = res.dag,
            hw          = hw,
            seed        = seed,
            policy_name = "incident_coupler_blocks_local",
            meta        = {
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": cp_sat_time_limit,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        ))
        # Depth = schedule makespan (max end-time); no frame execution needed.
        entries = S.meta.get("entries", {})
        depth   = max((se["end"] for se in entries.values()), default=0)
        layer_depths.append(depth)
        total_depth += depth

    return total_depth, layer_depths


# ── Single full run: SA + depth ───────────────────────────────────────────────
def run_full(
    n_logicals: int,
    rotations:  list,
    layers:     Dict[int, List[int]],
    *,
    topology:         str   = "grid",
    sparse_pct:       float = 0.0,
    n_data:           int   = 11,
    coupler_capacity: int   = 1,
    sa_steps:         int   = 10_000,
    sa_t0:            float = 1e5,
    sa_tend:          float = 1.1,
    seed:             int   = 42,
    score_kwargs:     Optional[Dict[str, float]] = None,   # None → standard SA
    sched_name:       str   = "greedy_critical",
    cp_sat_time_limit: float = 30.0,
    verbose:          bool  = False,
    label:            str   = "",
) -> RunResult:
    hw, hw_spec = make_hardware(
        n_logicals,
        topology         = topology,
        sparse_pct       = sparse_pct,
        n_data           = n_data,
        coupler_capacity = coupler_capacity,
    )

    # ── SA mapping ────────────────────────────────────────────────────────────
    t_sa = time.perf_counter()
    if score_kwargs is None:
        map_cfg = MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend)
        plan = get_mapper("simulated_annealing").solve(
            MappingProblem(n_logicals=n_logicals), hw, map_cfg,
            {"rotations": rotations, "verbose": verbose},
        )
        m = plan.meta
        sa_score = ScoreBreakdown(
            total            = m["best_score_total"],
            peak_load_pen    = m["best_score_peak_load"],
            range_load_pen   = m["best_score_range_load"],
            std_load_pen     = 0.0,
            span_pen         = m["best_score_span"],
            disconnected_pen = m["best_score_disconnected"],
            mst_pen          = m["best_score_mst"],
            split_pen        = m["best_score_split"],
            num_multiblock   = m["num_multiblock"],
            num_disconnected = m["num_disconnected"],
        )
    else:
        # Init with round-robin, then run custom-weight annealing
        get_mapper("auto_round_robin_mapping").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed)
        )
        sa_score = _anneal_weighted(
            rotations, hw,
            steps         = sa_steps,
            t0            = sa_t0,
            t_end         = sa_tend,
            seed          = seed,
            score_kwargs  = score_kwargs,
            verbose       = verbose,
        )
        plan = MappingPlan(
            logical_to_block = dict(hw.logical_to_block),
            logical_to_local = dict(hw.logical_to_local),
            meta             = {
                "best_score_total":        sa_score.total,
                "best_score_peak_load":    sa_score.peak_load_pen,
                "best_score_range_load":   sa_score.range_load_pen,
                "best_score_split":        sa_score.split_pen,
                "best_score_mst":          sa_score.mst_pen,
                "best_score_disconnected": sa_score.disconnected_pen,
                "best_score_span":         sa_score.span_pen,
                "num_multiblock":          sa_score.num_multiblock,
                "num_disconnected":        sa_score.num_disconnected,
            },
        )
    elapsed_sa = time.perf_counter() - t_sa

    # Default-weight score on the final mapping (fair comparison across all runs)
    default_score = _score(rotations, hw)

    # ── Greedy depth ──────────────────────────────────────────────────────────
    t_depth = time.perf_counter()
    total_depth, layer_depths = _compute_depth(
        rotations, layers, hw, plan, n_data, sched_name, seed, cp_sat_time_limit
    )
    elapsed_depth = time.perf_counter() - t_depth

    result = RunResult(
        label            = label or f"sa(steps={sa_steps},t0={sa_t0:.0e},tend={sa_tend})",
        sa_total         = sa_score.total,
        sa_peak          = sa_score.peak_load_pen,
        sa_range         = sa_score.range_load_pen,
        sa_split         = sa_score.split_pen,
        sa_mst           = sa_score.mst_pen,
        sa_disconnected  = sa_score.disconnected_pen,
        sa_span          = sa_score.span_pen,
        num_multiblock   = sa_score.num_multiblock,
        num_disconnected = sa_score.num_disconnected,
        default_total    = default_score.total,
        total_depth      = total_depth,
        layer_depths     = layer_depths,
        elapsed_sa_s     = elapsed_sa,
        elapsed_depth_s  = elapsed_depth,
        hw_label         = hw_spec.label(),
        params           = {
            "topology": topology, "sparse_pct": sparse_pct,
            "sa_steps": sa_steps, "sa_t0": sa_t0, "sa_tend": sa_tend,
            "seed": seed, "score_kwargs": score_kwargs or {},
        },
    )
    print(f"  {result.label:<45}  depth={total_depth:>6}"
          f"  sa_score={result.sa_total:>10.1f}"
          f"  default={result.default_total:>10.1f}"
          f"  SA={elapsed_sa:.1f}s  sched={elapsed_depth:.1f}s")
    return result


# ── Sweeps ────────────────────────────────────────────────────────────────────
def sweep_steps(n_logicals, rotations, layers, base, values) -> List[RunResult]:
    print("\n── Sweep: sa_steps ──────────────────────────────────────────────────")
    return [run_full(n_logicals, rotations, layers,
                     **{**base, "sa_steps": v}, label=f"steps={v:,}")
            for v in values]


def sweep_t0(n_logicals, rotations, layers, base, values) -> List[RunResult]:
    print("\n── Sweep: sa_t0 ─────────────────────────────────────────────────────")
    return [run_full(n_logicals, rotations, layers,
                     **{**base, "sa_t0": v}, label=f"t0={v:.0e}")
            for v in values]


def sweep_tend(n_logicals, rotations, layers, base, values) -> List[RunResult]:
    print("\n── Sweep: sa_tend ───────────────────────────────────────────────────")
    return [run_full(n_logicals, rotations, layers,
                     **{**base, "sa_tend": v}, label=f"tend={v}")
            for v in values]


def sweep_weight(
    n_logicals, rotations, layers, base,
    weight_name: str, values: list,
    default_kwargs: Dict[str, float],
) -> List[RunResult]:
    print(f"\n── Sweep: {weight_name} ─────────────────────────────────────────────")
    results = []
    for v in values:
        kw = {**default_kwargs, weight_name: v}
        results.append(run_full(
            n_logicals, rotations, layers,
            **{**base, "score_kwargs": kw},
            label=f"{weight_name}={v:.0e}",
        ))
    return results


def sweep_weight_heatmap(
    n_logicals, rotations, layers, base,
    w_peak_vals, w_split_vals,
    default_kwargs: Dict[str, float],
) -> np.ndarray:
    print("\n── Heatmap: W_PEAK × W_SPLIT ────────────────────────────────────────")
    grid = np.zeros((len(w_peak_vals), len(w_split_vals)), dtype=float)
    for i, wp in enumerate(w_peak_vals):
        for j, ws in enumerate(w_split_vals):
            kw = {**default_kwargs, "W_PEAK": wp, "W_SPLIT": ws}
            r  = run_full(n_logicals, rotations, layers,
                          **{**base, "score_kwargs": kw},
                          label=f"W_PEAK={wp:.0e} W_SPLIT={ws:.0e}")
            grid[i, j] = r.total_depth
    return grid


# ── Hierarchy presets (W_STD and W_DISCONNECTED always 0) ────────────────────
HIERARCHY_PRESETS = [
    # name                  W_PEAK  W_RANGE  W_SPAN   W_SPLIT  W_MST
    ("peak>>rest",          1e6,    1e3,     1e2,     10.0,    1.0  ),
    ("peak>range>span",     1e5,    1e4,     1e3,     1e2,     10.0 ),
    ("peak=range>>span",    1e4,    1e4,     1e3,     1e2,     1e2  ),
    ("peak>span>range",     1e5,    1e3,     1e4,     1e2,     10.0 ),
    ("span>>rest",          1e3,    1e3,     1e5,     1e2,     10.0 ),
    ("range>span>peak",     1e3,    1e5,     1e4,     1e2,     10.0 ),
    ("split>>rest",         1e3,    1e3,     1e2,     1e5,     10.0 ),
    ("mst>>rest",           1e3,    1e3,     1e2,     1e2,     1e5  ),
    ("balanced",            1e4,    1e4,     1e4,     1e4,     1e4  ),
    ("peak+span dominant",  1e5,    1e3,     1e5,     1e2,     10.0 ),
]


def sweep_hierarchy_presets(n_logicals, rotations, layers, base) -> List[RunResult]:
    print("\n── Sweep: Hierarchy presets ─────────────────────────────────────────")
    results = []
    for (name, wp, wr, wspan, wsplit, wmst) in HIERARCHY_PRESETS:
        kw = dict(W_PEAK=wp, W_RANGE=wr, W_SPAN=wspan, W_SPLIT=wsplit,
                  W_MST=wmst, W_STD=0.0, W_DISCONNECTED=0.0)
        results.append(run_full(
            n_logicals, rotations, layers,
            **{**base, "score_kwargs": kw},
            label=name,
        ))
    return results


# ── Hierarchy gap sweep: geometric step-factor parameterisation ───────────────
def sweep_gap(
    n_logicals, rotations, layers, base,
    step_factors: List[float],
    base_scales:  List[float],
) -> List[List["RunResult"]]:
    """
    Fixed ordering: PEAK > SPAN > RANGE > SPLIT > MST

    W_MST   = b
    W_SPLIT = b × s
    W_RANGE = b × s²
    W_SPAN  = b × s³
    W_PEAK  = b × s⁴

    Returns grid[i_base][i_step].
    """
    print("\n── Sweep: hierarchy gap (base × step factor) ───────────────────────")
    grid: List[List[RunResult]] = []
    for b in base_scales:
        row: List[RunResult] = []
        for s in step_factors:
            kw = dict(
                W_MST   = b,
                W_SPLIT = b * s,
                W_RANGE = b * s ** 2,
                W_SPAN  = b * s ** 3,
                W_PEAK  = b * s ** 4,
                W_STD   = 0.0,
                W_DISCONNECTED = 0.0,
            )
            row.append(run_full(
                n_logicals, rotations, layers,
                **{**base, "score_kwargs": kw},
                label=f"b={b:.0e} s={s}",
            ))
        grid.append(row)
    return grid


def build_gap_figure(
    gap_grid:     List[List["RunResult"]],
    step_factors: List[float],
    base_scales:  List[float],
    out_path:     str,
) -> None:
    """
    2×2 figure:
      [0,0] Line plot  – depth vs step_factor, one curve per base
      [0,1] Heatmap    – base × step_factor → depth
      [1,0] Best-worst – sorted bar of all 15 configs
      [1,1] Weight table – actual W_* values for the best 5 configs
    """
    n_base  = len(base_scales)
    n_steps = len(step_factors)

    # Flatten into lists for easy iteration
    all_results: List[RunResult] = [gap_grid[i][j]
                                     for i in range(n_base)
                                     for j in range(n_steps)]
    all_depths = [r.total_depth for r in all_results]

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "Hierarchy Gap Sweep  –  peak > span > range > split > mst\n"
        r"$W_k = b \times s^{\,\text{level}}$   (mst=0 … peak=4)",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.60, wspace=0.38)

    # ── [0,0]: depth vs step factor, one line per base ────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for i, b in enumerate(base_scales):
        ys = [gap_grid[i][j].total_depth for j in range(n_steps)]
        ax.plot(step_factors, ys, "o-", color=_C[i], linewidth=1.8,
                markersize=5, label=f"base={b:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("step factor  s  (log)", fontsize=7)
    ax.set_ylabel("total depth", fontsize=7)
    ax.set_title("Depth vs step factor\n(gap between hierarchy levels)", fontsize=9)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

    # ── [0,1]: heatmap base × step ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    hmap = np.array([[gap_grid[i][j].total_depth
                      for j in range(n_steps)]
                     for i in range(n_base)], dtype=float)
    im = ax.imshow(hmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"{s}" for s in step_factors], fontsize=6)
    ax.set_yticks(range(n_base))
    ax.set_yticklabels([f"{b:.0e}" for b in base_scales], fontsize=6)
    ax.set_xlabel("step factor  s", fontsize=7)
    ax.set_ylabel("base weight  b", fontsize=7)
    ax.set_title("Depth heatmap:  base × step factor\n(darker = deeper)", fontsize=9)
    for i in range(n_base):
        for j in range(n_steps):
            ax.text(j, i, str(int(hmap[i, j])), ha="center", va="center",
                    fontsize=7, color="white" if hmap[i, j] > hmap.mean() else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── [1,0]: all configs sorted best → worst ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    sorted_idx   = sorted(range(len(all_results)), key=lambda k: all_depths[k])
    sorted_depths = [all_depths[k] for k in sorted_idx]
    sorted_labels = [all_results[k].label for k in sorted_idx]
    n = len(sorted_depths)
    palette = [(0.2 + 0.6 * k / max(1, n - 1),
                0.7 - 0.5 * k / max(1, n - 1),
                0.2) for k in range(n)]
    bars = ax.bar(range(n), sorted_depths, color=palette, alpha=0.88, width=0.65)
    ax.bar_label(bars, fmt="%d", fontsize=6, padding=2)
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_labels, rotation=35, ha="right", fontsize=5)
    ax.set_ylabel("total depth", fontsize=7)
    ax.set_title("All configurations  sorted best → worst\n(green=best, red=worst)",
                 fontsize=9)
    ax.tick_params(axis="y", labelsize=7)

    # ── [1,1]: weight-value table for top-5 configs ──────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    top5 = [all_results[k] for k in sorted_idx[:5]]
    col_labels = ["config", "W_PEAK", "W_SPAN", "W_RANGE", "W_SPLIT", "W_MST", "depth"]
    rows = []
    for r in top5:
        kw = r.params.get("score_kwargs", {})
        rows.append([
            r.label,
            f"{kw.get('W_PEAK',0):.1e}",
            f"{kw.get('W_SPAN',0):.1e}",
            f"{kw.get('W_RANGE',0):.1e}",
            f"{kw.get('W_SPLIT',0):.1e}",
            f"{kw.get('W_MST',0):.1e}",
            str(r.total_depth),
        ])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.5)
    ax.set_title("Top-5 configurations (weight values)", fontsize=9, pad=14)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure]  saved → {out_path}")


# ── Print table ───────────────────────────────────────────────────────────────
def print_table(results: List[RunResult], title: str) -> None:
    sep = "─" * 110
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Label':<45}  {'depth':>6}  {'sa_score':>10}  "
          f"{'default':>10}  {'peak':>8}  {'split':>7}  {'multi':>6}  {'disc':>5}")
    print(sep)
    for r in results:
        print(f"  {r.label:<45}  {r.total_depth:>6}  {r.sa_total:>10.1f}  "
              f"{r.default_total:>10.1f}  {r.sa_peak:>8.0f}  {r.sa_split:>7.1f}"
              f"  {r.num_multiblock:>6}  {r.num_disconnected:>5}")
    print(f"{'='*110}")


# ── Plot helpers ──────────────────────────────────────────────────────────────
_C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _line(ax, xs, ys, title, xlabel, ylabel="total depth", log_x=False, color="#4C72B0"):
    ax.plot(xs, ys, "o-", color=color, linewidth=1.8, markersize=5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)
    if log_x:
        ax.set_xscale("log")


def _scatter(ax, results_groups, x_attr, title, xlabel, colors=None):
    for i, (results, label) in enumerate(results_groups):
        xs = [getattr(r, x_attr) for r in results]
        ys = [r.total_depth        for r in results]
        c  = (colors or _C)[i % len(_C)]
        ax.scatter(xs, ys, color=c, label=label, s=35, alpha=0.85, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("total depth", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6)


def _bar(ax, results: List[RunResult], title: str):
    labels = [r.label for r in results]
    depths = [r.total_depth for r in results]
    colors = [_C[i % len(_C)] for i in range(len(results))]
    bars   = ax.bar(range(len(labels)), depths, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=6, padding=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("total depth", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)


def _heatmap(ax, grid: np.ndarray, row_labels, col_labels, title):
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"{v:.0e}" for v in col_labels], fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"{v:.0e}" for v in row_labels], fontsize=6)
    ax.set_xlabel("W_SPLIT", fontsize=7)
    ax.set_ylabel("W_PEAK", fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(int(grid[i, j])), ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > grid.mean() else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)


# ── Weight-sensitivity figure ─────────────────────────────────────────────────
def _sensitivity_bar(
    ax,
    sweep_results: List[Tuple[str, List[RunResult]]],
    title: str,
) -> None:
    """Bar chart: depth range (max - min) per weight, showing how sensitive depth is."""
    names  = [name for name, _ in sweep_results]
    ranges = [max(r.total_depth for r in rs) - min(r.total_depth for r in rs)
              for _, rs in sweep_results]
    colors = [_C[i % len(_C)] for i in range(len(names))]
    bars = ax.bar(range(len(names)), ranges, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=6, padding=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("depth range\n(max − min)", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)


def build_weight_figure(
    w_peak_r:  List[RunResult],
    w_range_r: List[RunResult],
    w_mst_r:   List[RunResult],
    w_split_r: List[RunResult],
    w_span_r:  List[RunResult],
    preset_r:  List[RunResult],
    out_path:  str,
) -> None:
    """
    3-row figure for weight/hierarchy decision-making:
      Row 0: individual sweeps – W_PEAK, W_RANGE, W_MST
      Row 1: individual sweeps – W_SPLIT, W_SPAN  +  sensitivity bar
      Row 2: hierarchy presets sorted by depth (the decision chart)
    """
    fig = plt.figure(figsize=(15, 14))
    fig.suptitle(
        "SA Weight Sensitivity & Hierarchy Analysis  (metric: total depth)\n"
        "W_STD = W_DISCONNECTED = 0  (fixed)",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.65, wspace=0.38)

    # ── Row 0: W_PEAK, W_RANGE, W_MST ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _line(ax,
          [r.params["score_kwargs"].get("W_PEAK", 1e4) for r in w_peak_r],
          [r.total_depth for r in w_peak_r],
          "Depth vs W_PEAK", "W_PEAK (log)", log_x=True, color=_C[0])

    ax = fig.add_subplot(gs[0, 1])
    _line(ax,
          [r.params["score_kwargs"].get("W_RANGE", 1e4) for r in w_range_r],
          [r.total_depth for r in w_range_r],
          "Depth vs W_RANGE", "W_RANGE (log)", log_x=True, color=_C[1])

    ax = fig.add_subplot(gs[0, 2])
    _line(ax,
          [r.params["score_kwargs"].get("W_MST", 1e2) for r in w_mst_r],
          [r.total_depth for r in w_mst_r],
          "Depth vs W_MST", "W_MST (log)", log_x=True, color=_C[2])

    # ── Row 1: W_SPLIT, W_SPAN, sensitivity bar ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _line(ax,
          [r.params["score_kwargs"].get("W_SPLIT", 1e2) for r in w_split_r],
          [r.total_depth for r in w_split_r],
          "Depth vs W_SPLIT", "W_SPLIT (log)", log_x=True, color=_C[3])

    ax = fig.add_subplot(gs[1, 1])
    _line(ax,
          [r.params["score_kwargs"].get("W_SPAN", 1e3) for r in w_span_r],
          [r.total_depth for r in w_span_r],
          "Depth vs W_SPAN", "W_SPAN (log)", log_x=True, color=_C[4])

    ax = fig.add_subplot(gs[1, 2])
    _sensitivity_bar(
        ax,
        [("W_PEAK", w_peak_r), ("W_RANGE", w_range_r), ("W_MST", w_mst_r),
         ("W_SPLIT", w_split_r), ("W_SPAN", w_span_r)],
        "Per-weight depth sensitivity\n(higher bar = more impact when tuned)",
    )

    # ── Row 2: hierarchy presets sorted by depth ──────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    sorted_presets = sorted(preset_r, key=lambda r: r.total_depth)
    depths = [r.total_depth for r in sorted_presets]
    labels = [r.label for r in sorted_presets]
    n = len(sorted_presets)
    # gradient: green (best) → red (worst)
    palette = [
        (0.2 + 0.6 * i / max(1, n - 1),
         0.7 - 0.5 * i / max(1, n - 1),
         0.2)
        for i in range(n)
    ]
    bars = ax.bar(range(n), depths, color=palette, alpha=0.88, width=0.65)
    ax.bar_label(bars, fmt="%d", fontsize=7, padding=3)
    # annotate each bar with the weight values used
    for idx, r in enumerate(sorted_presets):
        kw = r.params.get("score_kwargs", {})
        ann = (f"Pk={kw.get('W_PEAK',0):.0e} "
               f"Rg={kw.get('W_RANGE',0):.0e} "
               f"Sp={kw.get('W_SPAN',0):.0e}\n"
               f"Sl={kw.get('W_SPLIT',0):.0e} "
               f"Ms={kw.get('W_MST',0):.0e}")
        ax.text(idx, depths[idx] * 0.98, ann,
                ha="center", va="top", fontsize=5, color="white",
                multialignment="center")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    ax.set_ylabel("total depth", fontsize=8)
    ax.set_title(
        "Hierarchy presets — sorted best → worst  (green=best, red=worst)\n"
        "Weight labels inside bars: Pk=W_PEAK  Rg=W_RANGE  Sp=W_SPAN  Sl=W_SPLIT  Ms=W_MST",
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=7)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure]  saved → {out_path}")


# ── Figure assembly ───────────────────────────────────────────────────────────
def build_figure(
    steps_r, t0_r, tend_r,
    w_peak_r, w_split_r, w_range_r,
    heatmap_grid, heatmap_row_vals, heatmap_col_vals,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("SA Mapping — Parameter & Weight Sensitivity  (metric: total depth)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.35)

    # ── Row 0: SA parameter → depth ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _line(ax, [r.params["sa_steps"] for r in steps_r],
          [r.total_depth for r in steps_r],
          "Depth vs sa_steps", "sa_steps")

    ax = fig.add_subplot(gs[0, 1])
    _line(ax, [r.params["sa_t0"] for r in t0_r],
          [r.total_depth for r in t0_r],
          "Depth vs sa_t0", "sa_t0 (log)", log_x=True)

    ax = fig.add_subplot(gs[0, 2])
    _line(ax, [r.params["sa_tend"] for r in tend_r],
          [r.total_depth for r in tend_r],
          "Depth vs sa_tend", "sa_tend")

    # ── Row 1: score ↔ depth scatter ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _scatter(ax,
             [(steps_r, "steps sweep"), (t0_r, "t0 sweep"), (tend_r, "tend sweep")],
             "default_total",
             "Default SA score vs depth\n(are they correlated?)",
             "default SA score")

    ax = fig.add_subplot(gs[1, 1])
    _scatter(ax,
             [(steps_r, "steps"), (t0_r, "t0"), (tend_r, "tend")],
             "num_multiblock",
             "Multiblock rotations vs depth",
             "# multiblock rotations")

    ax = fig.add_subplot(gs[1, 2])
    _scatter(ax,
             [(steps_r, "steps"), (t0_r, "t0"), (tend_r, "tend")],
             "sa_peak",
             "Peak load penalty vs depth",
             "peak load penalty")

    # ── Row 2: weight → depth ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    _line(ax,
          [r.params["score_kwargs"].get("W_PEAK", 1e4) for r in w_peak_r],
          [r.total_depth for r in w_peak_r],
          "Depth vs W_PEAK", "W_PEAK (log)", log_x=True, color=_C[0])

    ax = fig.add_subplot(gs[2, 1])
    _line(ax,
          [r.params["score_kwargs"].get("W_SPLIT", 1e2) for r in w_split_r],
          [r.total_depth for r in w_split_r],
          "Depth vs W_SPLIT", "W_SPLIT (log)", log_x=True, color=_C[1])

    ax = fig.add_subplot(gs[2, 2])
    _line(ax,
          [r.params["score_kwargs"].get("W_RANGE", 5e3) for r in w_range_r],
          [r.total_depth for r in w_range_r],
          "Depth vs W_RANGE", "W_RANGE (log)", log_x=True, color=_C[2])

    # ── Row 3: summary bar + heatmap ─────────────────────────────────────────
    summary = (
        [min(steps_r, key=lambda r: r.total_depth),
         max(steps_r, key=lambda r: r.total_depth)]
        + [min(t0_r,    key=lambda r: r.total_depth),
           max(t0_r,    key=lambda r: r.total_depth)]
        + [min(tend_r,  key=lambda r: r.total_depth),
           max(tend_r,  key=lambda r: r.total_depth)]
        + [min(w_peak_r, key=lambda r: r.total_depth),
           min(w_split_r, key=lambda r: r.total_depth),
           min(w_range_r, key=lambda r: r.total_depth)]
    )
    ax = fig.add_subplot(gs[3, :2])
    _bar(ax, summary, "Best & worst depth per sweep category")

    ax = fig.add_subplot(gs[3, 2])
    _heatmap(ax, heatmap_grid, heatmap_row_vals, heatmap_col_vals,
             "W_PEAK × W_SPLIT → depth")

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure]  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SA mapping sensitivity test")
    parser.add_argument("--pbc",              default=DEFAULT_PBC)
    parser.add_argument("--topology",         default="grid", choices=["grid", "ring"])
    parser.add_argument("--sparse_pct",       type=float, default=0.0)
    parser.add_argument("--n_data",           type=int,   default=11)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--sched",            default="greedy_critical")
    parser.add_argument("--verbose",          action="store_true")
    args = parser.parse_args()

    n_logicals, rotations, layers = load_circuit(args.pbc)

    # Base config shared across all sweeps
    base = dict(
        topology          = args.topology,
        sparse_pct        = args.sparse_pct,
        n_data            = args.n_data,
        coupler_capacity  = 1,
        sa_steps          = 10_000,
        sa_t0             = 1e5,
        sa_tend           = 1.1,
        seed              = args.seed,
        sched_name        = args.sched,
        cp_sat_time_limit = 30.0,
        verbose           = args.verbose,
    )

    # Default score weights (mirrors _score defaults in sa_mapping.py)
    W_DEFAULT = dict(W_PEAK=1e4, W_RANGE=1e4, W_STD=0.0,
                     W_DISCONNECTED=0.0, W_MST=1e2, W_SPLIT=1e2, W_SPAN=1e3)

    # ── SA parameter sweeps ───────────────────────────────────────────────────
    steps_r = sweep_steps(n_logicals, rotations, layers, base,
                          [1_000, 5_000, 10_000, 25_000])
    print_table(steps_r, "Sweep: sa_steps")

    t0_r = sweep_t0(n_logicals, rotations, layers, base,
                    [1e4, 1e5, 1e6, 1e7, 1e8])
    print_table(t0_r, "Sweep: sa_t0")

    tend_r = sweep_tend(n_logicals, rotations, layers, base,
                        [1.0, 5.0, 10.0, 50.0, 100.0])
    print_table(tend_r, "Sweep: sa_tend")

    # ── Weight (scaling factor) sweeps ────────────────────────────────────────
    # Use fewer SA steps for weight sweeps to keep runtime manageable
    base_w = {**base, "sa_steps": 5_000}

    w_peak_r = sweep_weight(n_logicals, rotations, layers, base_w,
                            "W_PEAK", [1e2, 1e3, 1e4, 1e5, 5e5], W_DEFAULT)
    print_table(w_peak_r, "Sweep: W_PEAK")

    w_split_r = sweep_weight(n_logicals, rotations, layers, base_w,
                             "W_SPLIT", [0.0, 10.0, 1e2, 1e3, 1e4], W_DEFAULT)
    print_table(w_split_r, "Sweep: W_SPLIT")

    w_range_r = sweep_weight(n_logicals, rotations, layers, base_w,
                             "W_RANGE", [0.0, 1e2, 1e3, 5e3, 1e4], W_DEFAULT)
    print_table(w_range_r, "Sweep: W_RANGE")

    w_mst_r = sweep_weight(n_logicals, rotations, layers, base_w,
                           "W_MST", [0.0, 10.0, 1e2, 1e3, 1e4], W_DEFAULT)
    print_table(w_mst_r, "Sweep: W_MST")

    w_span_r = sweep_weight(n_logicals, rotations, layers, base_w,
                            "W_SPAN", [0.0, 1e2, 1e3, 1e4, 1e5], W_DEFAULT)
    print_table(w_span_r, "Sweep: W_SPAN")

    # ── Hierarchy presets ─────────────────────────────────────────────────────
    preset_r = sweep_hierarchy_presets(n_logicals, rotations, layers, base_w)
    print_table(preset_r, "Sweep: Hierarchy presets")

    # ── Weight sensitivity figure (separate) ──────────────────────────────────
    build_weight_figure(
        w_peak_r, w_range_r, w_mst_r, w_split_r, w_span_r,
        preset_r,
        OUT_FIG_WEIGHTS,
    )

    # ── Hierarchy gap sweep ───────────────────────────────────────────────────
    gap_step_factors = [1.5, 3.0, 10.0, 30.0, 100.0]
    gap_base_scales  = [1e2, 1e3, 1e4]
    gap_grid = sweep_gap(
        n_logicals, rotations, layers, base_w,
        gap_step_factors, gap_base_scales,
    )
    build_gap_figure(gap_grid, gap_step_factors, gap_base_scales, OUT_FIG_GAP)

    # ── 3×3 heatmap: W_PEAK × W_SPLIT ────────────────────────────────────────
    hmap_peak  = [1e3, 1e4, 1e5]
    hmap_split = [10.0, 1e2, 1e3]
    heatmap_grid = sweep_weight_heatmap(
        n_logicals, rotations, layers, base_w,
        hmap_peak, hmap_split, W_DEFAULT,
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    build_figure(
        steps_r, t0_r, tend_r,
        w_peak_r, w_split_r, w_range_r,
        heatmap_grid, hmap_peak, hmap_split,
        OUT_FIG,
    )


if __name__ == "__main__":
    main()
