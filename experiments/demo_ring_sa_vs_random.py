#!/usr/bin/env python3
"""
experiments/demo_ring_sa_vs_random.py

Demonstration: SA mapping vs random placement on a 6-block ring.
Designed to produce the three-panel figure in the paper.

Setup
-----
- 6-block ring topology (Gross code blocks, n_data=11)
- 33 logical qubits at 50 % fill (3 rotations × 11 qubits each)
- Single commuting layer (disjoint supports → trivially commute)

Hard-coded random mapping  (panel a / c-left)
----------------------------------------------
  B1 [full 11/11]: R₁ ∪ R₃  ← TWO rotation colors, OVERLOADED
  B2 [  6/11    ]: R₂ only
  B3 [  6/11    ]: R₃ only
  B4 [  5/11    ]: R₁ only
  B5 [  5/11    ]: R₂ only
  B6 [  0/11    ]: EMPTY

  Hop distances:
    R₁: B1 ↔ B4 = 3 hops  (link crosses ring interior)
    R₂: B2 ↔ B5 = 3 hops  (link crosses ring interior)
    R₃: B1 ↔ B3 = 2 hops

  Because R₁ and R₃ share B1, their gadgets compete for B1's resources
  → the scheduler is forced to serialize them → higher logical depth.

SA mapping  (panel b / c-right)
---------------------------------
  Priority:   Utilization (1e6) → Interblock (1e4) → Locality (1e2) → Balance (1e0)
  Schedule:   20 000 iterations, T₀ = 1e5, T_end = 5e-2

  SA clusters each rotation into a contiguous pair of adjacent blocks,
  eliminating the B1 overload, the B6 vacancy, and the 3-hop links.
  No shared blocks → full parallelism → lower logical depth.

Why ring over grid
------------------
  On a ring, 3-hop links (B1↔B4, B2↔B5) are immediately visible as chords
  cutting across the interior.  SA's 1-hop perimeter arcs make the contrast
  obvious at figure size.  A grid arrangement would be much harder to read.

Usage
-----
  cd <repo root>
  python experiments/demo_ring_sa_vs_random.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple

# Allow import from repo root or experiments/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.core.types import PauliRotation
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId,
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ShortestPathGatherRouting,
)
from modqldpc.mapping.algos.sa_mapping import _blocks_touched, _mst_len, _rotation_support
from modqldpc.mapping.algos.sa_v2 import anneal_with_checkpoints_v2, score_mapping_v2
from modqldpc.mapping.factory import get_mapper
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.model import HardwareGraph
from modqldpc.mapping.types import MappingConfig, MappingPlan, MappingProblem
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem

# ── Constants ─────────────────────────────────────────────────────────────────

SEED     = 42
SA_STEPS = 20_000
SA_T0    = 1e5
SA_TEND  = 5e-2
N_DATA   = 11    # Gross code: 11 data-qubit slots per block
N_ROTS   = 3

# Scoring weights: sequential ×100 scaling between priority tiers
#   Tier 1 – Utilisation:  W_UNUSED_BLOCKS = 1e6
#   Tier 2 – Interblock:   W_MULTI_BLOCK = W_SPAN = 1e4
#   Tier 3 – Locality:     W_MST = 1e2
#   Tier 4 – Balance:      W_SPLIT = 1e0
SCORE_KWARGS: Dict[str, float] = {
    "W_UNUSED_BLOCKS": 1e6,
    "W_OCC_RANGE":     0.0,   # disabled — clean tier separation
    "W_OCC_STD":       0.0,
    "W_MULTI_BLOCK":   1e4,
    "W_SPAN":          1e4,
    "W_MST":           1e2,
    "W_SPLIT":         1e0,
    "W_SUPPORT_PEAK":  0.0,
    "W_SUPPORT_RANGE": 0.0,
    "W_SUPPORT_STD":   0.0,
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "ring_demo",
)

# ── Utilities ─────────────────────────────────────────────────────────────────

ROT_LABELS = ["R₁", "R₂", "R₃"]


def make_axis(n: int, support: List[int], pauli: str = "Y") -> str:
    """
    Build an n-character Pauli axis string.
    Convention: qubit k occupies position (n-1-k) from the left (qubit 0
    is the rightmost character), matching GoSCConverter / _rotation_support.
    """
    chars = ["I"] * n
    for q in support:
        chars[n - 1 - q] = pauli
    return "".join(chars)


def bfs_hops(hw: HardwareGraph, src: int, dst: int) -> int:
    """Shortest-hop distance between two blocks on the hardware graph."""
    visited: Dict[int, int] = {src: 0}
    q: deque = deque([src])
    while q:
        node = q.popleft()
        if node == dst:
            return visited[node]
        for nbr in hw.neighbors.get(node, set()):
            if nbr not in visited:
                visited[nbr] = visited[node] + 1
                q.append(nbr)
    return -1  # unreachable


def apply_mapping(hw: HardwareGraph,
                  l2b: Dict[int, int],
                  l2l: Dict[int, int]) -> None:
    """Overwrite hw logical-to-block/local assignments in-place."""
    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    hw.logical_to_block.update(l2b)
    hw.logical_to_local.update(l2l)


# ── Pretty-printing helpers ───────────────────────────────────────────────────

def print_block_assignments(hw: HardwareGraph,
                             rotations: List[PauliRotation]) -> None:
    """Print per-block qubit inventory with rotation ownership."""
    # For each qubit determine which rotation index it belongs to
    qubit_to_rot: Dict[int, int] = {}
    for i, rot in enumerate(rotations):
        for q in _rotation_support(rot):
            qubit_to_rot[q] = i

    block_to_info: Dict[int, Dict] = {}
    for b in sorted(hw.blocks.keys()):
        capacity = hw.blocks[b].num_logicals
        qubits = sorted(
            q for q, blk in hw.logical_to_block.items() if blk == b
        )
        rot_indices = sorted(set(qubit_to_rot[q] for q in qubits))
        block_to_info[b] = dict(capacity=capacity, qubits=qubits, rots=rot_indices)

    for b, info in block_to_info.items():
        fill    = f"{len(info['qubits'])}/{info['capacity']}"
        rot_lbl = " + ".join(ROT_LABELS[i] for i in info['rots']) if info['rots'] else "—"
        flags   = ""
        if len(info['rots']) >= 2:
            flags += "  ← OVERLOADED (two rotation colours)"
        if not info['qubits']:
            flags += "  ← EMPTY"
        print(f"    B{b} [{fill:>6}]:  {info['qubits']}  [{rot_lbl}]{flags}")


def print_span_analysis(rotations: List[PauliRotation],
                         hw: HardwareGraph) -> None:
    """Print inter-block span / MST for every rotation."""
    print("  Inter-block span analysis:")
    for i, rot in enumerate(rotations):
        blocks = sorted(_blocks_touched(rot, hw))
        if len(blocks) <= 1:
            desc = "intra-block (0 hops)"
        else:
            mst   = _mst_len(set(blocks), hw)
            pairs = [
                f"B{a}↔B{b}={bfs_hops(hw, a, b)}h"
                for a in blocks for b in blocks if a < b
            ]
            cross = "  ← crosses ring interior" if mst >= 3 else ""
            desc  = f"MST={mst} hop(s)  ({', '.join(pairs)}){cross}"
        print(f"    {ROT_LABELS[i]}: blocks {['B'+str(b) for b in blocks]}  →  {desc}")


def print_score_breakdown(score, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(
        f"  {tag}Score={score.total:.0f}"
        f"  | unused_blocks={score.unused_blocks}"
        f"  | num_multiblock={score.num_multiblock}"
        f"  | span_total={score.span_total:.0f}"
        f"  | mst_total={score.mst_total:.2f}"
        f"  | split_total={score.split_total:.2f}"
    )


# ── Depth computation ─────────────────────────────────────────────────────────

def _rotation_timings(
    schedule,
    rotations: List[PauliRotation],
    layer: int = 0,
) -> Dict[int, Tuple[int, int]]:
    """
    Return {ridx: (start, end)} for each rotation by scanning the schedule
    entries whose node-IDs contain the rotation tag 'L{layer:02d}_R{ridx:03d}'.
    start = earliest start across all nodes for that rotation.
    end   = latest   end   across all nodes for that rotation.
    """
    entries = schedule.meta.get("entries", {})
    timings: Dict[int, Tuple[int, int]] = {}
    for rot in rotations:
        ridx = rot.idx
        tag  = f"L{layer:02d}_R{ridx:03d}"
        starts, ends = [], []
        for nid, se in entries.items():
            if tag in nid:
                starts.append(int(se["start"]))
                ends.append(int(se["end"]))
        if starts:
            timings[ridx] = (min(starts), max(ends))
    return timings


def print_rotation_timings(
    timings: Dict[int, Tuple[int, int]],
    rotations: List[PauliRotation],
) -> None:
    """Print a table of start / end / duration for each rotation."""
    print("  Rotation scheduling timings:")
    print(f"    {'Rotation':<10}  {'Start':>6}  {'End':>6}  {'Duration':>9}")
    print(f"    {'─'*10}  {'─'*6}  {'─'*6}  {'─'*9}")
    for rot in rotations:
        ridx = rot.idx
        if ridx in timings:
            s, e = timings[ridx]
            print(f"    {ROT_LABELS[ridx]:<10}  {s:>6}  {e:>6}  {e-s:>9}")
        else:
            print(f"    {ROT_LABELS[ridx]:<10}  {'—':>6}  {'—':>6}  {'—':>9}")


def compute_depth(
    rotations: List[PauliRotation],
    hw: HardwareGraph,
    plan: MappingPlan,
    seed: int,
) -> Tuple[int, Dict[int, Tuple[int, int]]]:
    """
    Lower the single commuting layer and schedule it with greedy_critical.
    Returns (logical_depth, {ridx: (start, end)}).
    """
    cost_fn = make_gross_actual_cost_fn(plan)
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    # Single layer: all three rotations at indices [0, 1, 2]
    effective_rotations = {r.idx: r for r in rotations}
    rotation_indices    = [r.idx for r in rotations]

    res = lower_one_layer(
        layer_idx=0,
        rotations=effective_rotations,
        rotation_indices=rotation_indices,
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
            "layer_idx":         0,
            "tie_breaker":       "duration",
            "cp_sat_time_limit": None,
            "debug_decode":      False,
            "safe_fill":         True,
            "cp_sat_log":        False,
        },
    )
    schedule = get_scheduler("greedy_critical").solve(sched_problem)

    timings = _rotation_timings(schedule, rotations, layer=0)

    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    result = executor.execute_layer(
        layer=0,
        dag=res.dag,
        schedule=schedule,
        frame_in=FrameState(),
        next_layer_rotations=[],
    )
    return result.depth, timings


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "═" * 72
    SEP2 = "─" * 60

    print(SEP)
    print("  RING DEMONSTRATION:  SA Mapping vs Random Placement")
    print(SEP)

    # ── 1. Hardware ───────────────────────────────────────────────────────────
    # 33 logical qubits @ 50 % fill → 6 blocks, each n_data=11 slots
    n_logicals = N_ROTS * N_DATA   # 33
    hw, spec = make_hardware(
        n_logicals, topology="ring", sparse_pct=0.5, n_data=N_DATA
    )

    block_ids = sorted(hw.blocks.keys())
    print(f"\nHardware:  {spec.label()}")
    print(f"  n_blocks={spec.n_blocks}  n_data={spec.n_data}  "
          f"capacity={spec.total_capacity}  n_logicals={spec.n_logicals}  "
          f"fill={spec.actual_fill_rate*100:.0f}%")
    print(f"  Ring order: " + " — ".join(f"B{b}" for b in block_ids) + f" — B{block_ids[0]}")

    # ── 2. Rotations (hard-coded, single commuting layer) ─────────────────────
    n = n_logicals   # 33
    r1_qubits = list(range(0,  11))   # logical qubits 0–10
    r2_qubits = list(range(11, 22))   # logical qubits 11–21
    r3_qubits = list(range(22, 33))   # logical qubits 22–32

    rotations: List[PauliRotation] = [
        PauliRotation(
            axis=make_axis(n, r1_qubits, "Y"),
            angle=math.pi / 8, source="R1_demo", idx=0,
        ),
        PauliRotation(
            axis=make_axis(n, r2_qubits, "Y"),
            angle=math.pi / 8, source="R2_demo", idx=1,
        ),
        PauliRotation(
            axis=make_axis(n, r3_qubits, "Z"),
            angle=math.pi / 8, source="R3_demo", idx=2,
        ),
    ]

    print("\nRotations  (single commuting layer — disjoint supports):")
    for i, rot in enumerate(rotations):
        supp  = sorted(_rotation_support(rot))
        pauli = rot.axis.lstrip("I")[0] if rot.axis.lstrip("I") else "I"
        print(f"  {ROT_LABELS[i]}: qubits {supp}  (all-{pauli}, angle=π/8)")

    # ── 3. Hard-coded random mapping ──────────────────────────────────────────
    #
    # Pathological placement designed to expose both overloading and 3-hop links:
    #
    #   B1 [11/11]: R₁ slots {0..5}   + R₃ slots {6..10}  ← R₁+R₃ share B1
    #   B2 [ 6/11]: R₂ slots {0..5}
    #   B3 [ 6/11]: R₃ slots {0..5}
    #   B4 [ 5/11]: R₁ slots {0..4}
    #   B5 [ 5/11]: R₂ slots {0..4}
    #   B6 [ 0/11]: (empty)
    #
    # Hop map:  R₁: B1↔B4 = 3 hops   R₂: B2↔B5 = 3 hops   R₃: B1↔B3 = 2 hops
    # Block contention: R₁ and R₃ both require B1 → their gadgets are serialised.

    rand_l2b: Dict[int, int] = {}
    rand_l2l: Dict[int, int] = {}

    def assign(q: int, block: int, slot: int) -> None:
        rand_l2b[q] = block
        rand_l2l[q] = slot

    # R₁: first 6 qubits → B1 slots 0-5; last 5 → B4 slots 0-4
    for idx_local, q in enumerate(r1_qubits[:6]):  assign(q, 1, idx_local)
    for idx_local, q in enumerate(r1_qubits[6:]):  assign(q, 4, idx_local)

    # R₂: first 6 qubits → B2 slots 0-5; last 5 → B5 slots 0-4
    for idx_local, q in enumerate(r2_qubits[:6]):  assign(q, 2, idx_local)
    for idx_local, q in enumerate(r2_qubits[6:]):  assign(q, 5, idx_local)

    # R₃: first 5 qubits → B1 slots 6-10 (filling B1); last 6 → B3 slots 0-5
    for idx_local, q in enumerate(r3_qubits[:5]):  assign(q, 1, 6 + idx_local)
    for idx_local, q in enumerate(r3_qubits[5:]):  assign(q, 3, idx_local)

    apply_mapping(hw, rand_l2b, rand_l2l)
    rand_plan  = MappingPlan(dict(rand_l2b), dict(rand_l2l))
    rand_score = score_mapping_v2(rotations, hw, **SCORE_KWARGS)

    print(f"\n{SEP2}")
    print("PANEL (a):  RANDOM MAPPING  [hard-coded pathological placement]")
    print(SEP2)
    print("  Block assignments:")
    print_block_assignments(hw, rotations)
    print()
    print_span_analysis(rotations, hw)
    print()
    print_score_breakdown(rand_score, label="random score")

    print("\n  Scheduling layer (greedy_critical) …", flush=True)
    t0_sched = time.perf_counter()
    rand_depth, rand_timings = compute_depth(rotations, hw, rand_plan, SEED)
    print(f"  Elapsed: {time.perf_counter()-t0_sched:.1f} s")
    print()
    print_rotation_timings(rand_timings, rotations)
    print(f"\n  *** Logical depth (random mapping): {rand_depth} ***")

    # ── 4. SA mapping with progress checkpoints ───────────────────────────────
    # Reset hw, seed with auto_round_robin, then anneal
    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    get_mapper("auto_round_robin_mapping").solve(
        MappingProblem(n_logicals), hw, MappingConfig(seed=SEED)
    )

    print(f"\n{SEP2}")
    print("PANEL (b):  SA MAPPING")
    print(SEP2)
    print(
        f"  Weights:   Utilisation(1e6) → Interblock(1e4) → Locality(1e2) → Balance(1e0)\n"
        f"  Schedule:  {SA_STEPS:,} iterations,  T₀={SA_T0:.0e},  T_end={SA_TEND:.0e}\n"
    )

    # Column header for checkpoint table
    print(
        f"  {'  %':>5}  {'T':>12}  {'best score':>12}  "
        f"{'unused':>7}  {'multi':>6}  {'span':>5}  {'mst':>6}"
    )
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*6}")

    # How SA helps: each row shows the best-so-far mapping quality at that
    # fraction of the optimisation budget.  Watch unused_blocks → 0 and
    # mst → 1 as the mapping tightens around the ring perimeter.
    t0_sa = time.perf_counter()
    best_score, checkpoints, best_map = anneal_with_checkpoints_v2(
        rotations, hw,
        steps=SA_STEPS,
        t0=SA_T0,
        t_end=SA_TEND,
        seed=SEED,
        score_kwargs=SCORE_KWARGS,
        n_check=11,           # 0 % … 100 % in steps of 10 %
    )
    sa_time = time.perf_counter() - t0_sa

    for cp in checkpoints:
        print(
            f"  {cp['pct']:>5.1f}%  "
            f"{cp['temperature']:>12.4f}  "
            f"{cp['best_total']:>12.0f}  "
            f"{int(cp['best_unused_blocks']):>7}  "
            f"{int(cp['best_num_multiblock']):>6}  "
            f"{cp['best_span_total']:>5.0f}  "
            f"{cp['best_mean_mst']:>6.2f}"
        )
    print(f"\n  SA completed in {sa_time:.1f} s")

    # hw is already set to best mapping by anneal_with_checkpoints_v2
    sa_l2b  = dict(hw.logical_to_block)
    sa_l2l  = dict(hw.logical_to_local)
    sa_plan  = MappingPlan(sa_l2b, sa_l2l)
    sa_score = score_mapping_v2(rotations, hw, **SCORE_KWARGS)

    print("\n  SA block assignments:")
    print_block_assignments(hw, rotations)
    print()
    print_span_analysis(rotations, hw)
    print()
    print_score_breakdown(sa_score, label="SA score")

    print("\n  Scheduling layer (greedy_critical) …", flush=True)
    t0_sched = time.perf_counter()
    sa_depth, sa_timings = compute_depth(rotations, hw, sa_plan, SEED)
    print(f"  Elapsed: {time.perf_counter()-t0_sched:.1f} s")
    print()
    print_rotation_timings(sa_timings, rotations)
    print(f"\n  *** Logical depth (SA mapping): {sa_depth} ***")

    # ── 5. Panel-c summary ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PANEL (c):  DEPTH COMPARISON SUMMARY")
    print(SEP)

    def _pct(rv: float, sv: float) -> str:
        if sv == 0:
            return "—"
        return f"{rv/sv:.1f}×"

    rows: List[Tuple[str, float, float]] = [
        ("Mapping score (total)",    rand_score.total,            sa_score.total),
        ("Unused blocks",            float(rand_score.unused_blocks), float(sa_score.unused_blocks)),
        ("Multi-block rotations",    float(rand_score.num_multiblock), float(sa_score.num_multiblock)),
        ("Total MST hops",           rand_score.mst_total,        sa_score.mst_total),
        ("Logical depth",            float(rand_depth),           float(sa_depth)),
    ]

    print(f"\n  {'Metric':<28}  {'Random':>12}  {'SA':>12}  {'SA speedup':>10}")
    print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*10}")
    for label, rv, sv in rows:
        print(f"  {label:<28}  {rv:>12.1f}  {sv:>12.1f}  {_pct(rv, sv):>10}")

    print(
        f"\n  Depth:  {rand_depth}  →  {sa_depth}"
        f"  ({_pct(float(rand_depth), float(sa_depth))} reduction)\n"
    )

    # ── 6. Save results ───────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "hardware": {
            "topology":       "ring",
            "n_blocks":       spec.n_blocks,
            "n_data":         spec.n_data,
            "n_logicals":     spec.n_logicals,
            "fill_rate":      spec.actual_fill_rate,
        },
        "rotations": [
            {"label": ROT_LABELS[i], "support": sorted(_rotation_support(r)),
             "pauli": r.axis.lstrip("I")[0] if r.axis.lstrip("I") else "I",
             "angle_over_pi": 1/8}
            for i, r in enumerate(rotations)
        ],
        "sa_config": {
            "steps":        SA_STEPS,
            "t0":           SA_T0,
            "t_end":        SA_TEND,
            "seed":         SEED,
            "score_kwargs": SCORE_KWARGS,
        },
        "random_mapping": {
            "description":    "hard-coded pathological placement",
            "score_total":    rand_score.total,
            "unused_blocks":  rand_score.unused_blocks,
            "num_multiblock": rand_score.num_multiblock,
            "span_total":     rand_score.span_total,
            "mst_total":      rand_score.mst_total,
            "logical_depth":  rand_depth,
            "rotation_timings": {
                ROT_LABELS[ridx]: {"start": s, "end": e}
                for ridx, (s, e) in rand_timings.items()
            },
            "block_assignment": {str(q): b for q, b in rand_l2b.items()},
        },
        "sa_mapping": {
            "score_total":    sa_score.total,
            "unused_blocks":  sa_score.unused_blocks,
            "num_multiblock": sa_score.num_multiblock,
            "span_total":     sa_score.span_total,
            "mst_total":      sa_score.mst_total,
            "logical_depth":  sa_depth,
            "rotation_timings": {
                ROT_LABELS[ridx]: {"start": s, "end": e}
                for ridx, (s, e) in sa_timings.items()
            },
            "block_assignment": {str(q): b for q, b in sa_l2b.items()},
        },
        "sa_checkpoints": checkpoints,
    }
    out_path = os.path.join(RESULTS_DIR, "ring_demo_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
