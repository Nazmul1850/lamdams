#!/usr/bin/env python3
"""
experiments/demo_greedy_critical_scheduling.py

Greedy-critical scheduler demonstration on a 4-block ring.

PURPOSE
-------
Shows the internal mechanisms of GreedyCriticalScheduler and how they
reduce logical depth:

  1. Bottom-level (BL) and block-criticality computation
  2. Component ordering by criticality (most critical first)
  3. N-cycle DFS deadlock prevention  — R1 and R2 would hard-deadlock if
     admitted simultaneously; DFS detects the 2-cycle and stalls R2
  4. Safe-fill (Phase-2 legacy heuristic) — explained with exact values;
     superseded by the DFS check in the current implementation
  5. Priority ordering clears future contention: R3 (single-block, B2)
     runs in parallel with R1, freeing B2 before R2 needs it

HARDWARE  (4-block ring, Gross code blocks)
-----------
  n_data=6  (24 logical qubits, dense fill)
  Ring: B1 — B2 — B3 — B4 — B1

ROTATIONS  (single commuting layer — disjoint supports)
-----------
  NOTE on make_axis convention: qubit q is placed at tensor position n-1-q
  (rightmost = qubit 0, GoSCConverter convention).  The lowering reads tensor
  positions as logical ids directly (no reversal), so the actual hardware blocks
  depend on the logical ids at those tensor positions, not the support indices.
  With round-robin mapping (q//N_DATA+1), the resolved blocks are:

  R1 (idx 0):  support indices {0,1,2,12,13,14}
               → tensor positions / logical ids {9,10,11,21,22,23}
               → participant blocks B2, B4
               multiblock, 2-hop route B2→B1→B4  (B1 is intermediate)
               block_crit_sum = 26888  →  priority rank 1

  R2 (idx 1):  support indices {6,7,18,19}
               → tensor positions / logical ids {4,5,16,17}
               → participant blocks B1, B3
               multiblock, 2-hop route B1→B2→B3  (B2 is intermediate)
               block_crit_sum = 29072  →  priority rank 0  (admitted FIRST)

  R3 (idx 2):  support indices {8,9,10}
               → tensor positions / logical ids {13,14,15}
               → participant block B3 only
               single-block — no routing needed
               block_crit_sum = 6468   →  priority rank 2

SCHEDULING STORY  (exact times from sched_demo_results.json)
----------------
  t=0  : R2 admitted first (rank 0 — highest block_crit_sum=29072; B3 is the
           hottest block at criticality=539).
           Inits start on B1 and B3 simultaneously.
         R1 tried → DFS finds 2-cycle:
           R1.route {B1,B2,B4} ∩ R2.participants {B1,B3} = {B1} ≠ ∅
           R2.route {B1,B2,B3} ∩ R1.participants {B2,B4} = {B2} ≠ ∅  → R1 BLOCKED
         R3 tried → single-block on B3, but R2 owns B3 → R3 BLOCKED

  t=28 : R2's PZ completes (end=29) → B3 (non-magic) and B2 (intermediate)
           released.  Xm_R2 becomes ready.

  t=29 : Xm_R2 dispatched first (R2 still has highest comp_rank priority) →
           R2 removed from in_progress_multiblock the moment Xm is committed.
           R1 inits dispatched next (in_progress_multiblock now empty → DFS
           skipped → no cycle) → R1 starts on B2 and B4.
           R3 init dispatched (B3 now free) → R3 starts on B3.
           R1, R2.Xm, and R3 all overlap in [t=29, t=30].

  t=30 : R2.Xm completes → B1 (magic block) released.
  t=41 : R3 completes (PZ=39..40, Xm=40..41).
  t=59 : R1 completes (link=55..57, PZ=57..58, Xm=58..59).
           Makespan=59  vs  naive serial 30+30+12=72  →  13 steps saved (18%).

SAFE-FILL (legacy Phase-2 heuristic)
--------------------------------------
  Original design: when component B is blocked by a 2-cycle with A, still
  admit B if B.xm_completion_LB < A.link_start_LB (B finishes before A
  needs its link node — speculative but safe).
  Limitation: only caught 2-component cycles; 3+-cycles on sparse hardware
  caused hard deadlocks that safe_fill missed.
  Superseded by: N-cycle DFS detection (current) which catches all cycle
  lengths. The use_safe_fill field in GreedyCriticalScheduler is now a
  no-op; DFS runs unconditionally when in_progress_multiblock ≠ ∅.

OUTPUTS
-------
  Console: full metric tables, timeline, rotation timings, figure flow
  JSON:    results/sched_demo/sched_demo_results.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.core.types import PauliRotation
from modqldpc.lowering.ir import (
    K_FRAME_UPDATE, K_INIT_PIVOT, K_INTERBLOCK_LINK,
    K_LOCAL_COUPLE, K_MEAS_MAGIC_X, K_MEAS_PARITY_PZ,
)
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId,
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ShortestPathGatherRouting,
)
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.types import MappingPlan
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.algos.greedy_critical_scheduling import GreedyCriticalScheduler
from modqldpc.scheduling.types import SchedulingProblem

# ── Constants ─────────────────────────────────────────────────────────────────

SEED     = 42
N_DATA   = 6     # slots per block (deliberately small for a clear figure)
N_QUBITS = 24    # 4 blocks × 6 slots, dense fill

ROT_LABELS = {0: "R₁", 1: "R₂", 2: "R₃"}
KIND_SHORT = {
    K_INIT_PIVOT:      "init",
    K_LOCAL_COUPLE:    "lc  ",
    K_INTERBLOCK_LINK: "link",
    K_MEAS_PARITY_PZ:  "PZ  ",
    K_MEAS_MAGIC_X:    "Xm  ",
    K_FRAME_UPDATE:    "frm ",
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "sched_demo",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_axis(n: int, support: List[int], pauli: str = "Y") -> str:
    """Build Pauli axis string. Qubit k occupies position n-1-k (rightmost = qubit 0)."""
    chars = ["I"] * n
    for q in support:
        chars[n - 1 - q] = pauli
    return "".join(chars)


def ridx_from_nid(nid: str) -> Optional[int]:
    """Extract rotation index from a node ID like 'init_L00_R001_B2_c0'."""
    m = re.search(r"_R(\d{3})", nid)
    return int(m.group(1)) if m else None


def rot_timings(
    schedule,
    rotations: List[PauliRotation],
    layer: int = 0,
) -> Dict[int, Tuple[int, int]]:
    """Return {ridx: (earliest_start, latest_end)} across all nodes for that rotation."""
    entries = schedule.meta.get("entries", {})
    result: Dict[int, Tuple[int, int]] = {}
    for rot in rotations:
        ridx = rot.idx
        tag   = f"L{layer:02d}_R{ridx:03d}"
        starts, ends = [], []
        for nid, se in entries.items():
            if tag in nid:
                starts.append(int(se["start"]))
                ends.append(int(se["end"]))
        if starts:
            result[ridx] = (min(starts), max(ends))
    return result


def waitsfor_edges(prep) -> Dict[int, List[int]]:
    """
    Build waits-for edges among multiblock components.
    Edge A → B means: A's link route blocks intersect B's participant blocks
    (A's link node cannot run while B holds those blocks → A waits for B).
    """
    mb = [cid for cid in range(len(prep.components)) if prep.comp_is_multiblock[cid]]
    edges: Dict[int, List[int]] = {cid: [] for cid in mb}
    for a in mb:
        for b in mb:
            if a != b and (prep.comp_route_blocks[a] & prep.comp_participant_blocks[b]):
                edges[a].append(b)
    return edges


def find_cycles(edges: Dict[int, List[int]]) -> List[List[int]]:
    """Find all simple directed cycles in the waits-for graph (Johnson-like, small graphs)."""
    nodes = list(edges)
    cycles: List[List[int]] = []
    visited: Set[int] = set()

    def dfs(start: int, path: List[int], seen: Set[int]) -> None:
        for nb in edges.get(path[-1], []):
            if nb == start and len(path) > 1:
                cycles.append(list(path))
            elif nb not in seen:
                seen.add(nb)
                path.append(nb)
                dfs(start, path, seen)
                path.pop()
                seen.discard(nb)

    for node in nodes:
        if node not in visited:
            dfs(node, [node], {node})
            visited.add(node)
    # Deduplicate (same cycle, different start)
    canonical: List[List[int]] = []
    seen_sets: List[frozenset] = []
    for c in cycles:
        fs = frozenset(c)
        if fs not in seen_sets:
            seen_sets.append(fs)
            canonical.append(c)
    return canonical


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP  = "═" * 72
    SEP2 = "─" * 60

    print(SEP)
    print("  GREEDY CRITICAL SCHEDULER DEMONSTRATION")
    print(SEP)

    # ── 1. Hardware ───────────────────────────────────────────────────────────
    hw, spec = make_hardware(N_QUBITS, topology="ring", sparse_pct=0.0, n_data=N_DATA)
    block_ids = sorted(hw.blocks.keys())   # [1, 2, 3, 4]

    # Dense round-robin mapping: qubit q → block (q//N_DATA + 1), slot (q % N_DATA)
    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    for q in range(N_QUBITS):
        hw.logical_to_block[q] = q // N_DATA + 1
        hw.logical_to_local[q] = q % N_DATA

    plan = MappingPlan(dict(hw.logical_to_block), dict(hw.logical_to_local))

    print(f"\n{SEP2}")
    print("SECTION 1 — HARDWARE & MAPPING")
    print(SEP2)
    print(f"  Topology:   {spec.label()}")
    print(f"  n_blocks:   {spec.n_blocks}  (ring: " +
          " — ".join(f"B{b}" for b in block_ids) + f" — B{block_ids[0]})")
    print(f"  n_data:     {N_DATA} slots/block")
    print(f"  n_logicals: {N_QUBITS}  (fill = {spec.actual_fill_rate*100:.0f}%)")
    print(f"\n  Mapping:    round-robin  (qubit q → block (q//{N_DATA}+1), slot q%{N_DATA})")
    for b in block_ids:
        qs = sorted(q for q, blk in hw.logical_to_block.items() if blk == b)
        print(f"    B{b}: qubits {qs}")

    # ── 2. Rotations ──────────────────────────────────────────────────────────
    n = N_QUBITS

    # R1: support indices chosen so lowering resolves to B2 + B4 (opposite on ring)
    #     make_axis places qubit q at tensor pos n-1-q; lowering reads pos as logical id.
    #     indices [0,1,2,12,13,14] → logical ids [9,10,11,21,22,23] → B2, B4
    r1_support = [0, 1, 2, 12, 13, 14]
    # R2: indices [6,7,18,19] → logical ids [4,5,16,17] → B1, B3 (opposite on ring)
    #     Creates deadlock with R1: R1.route∩R2.participants={B1}, R2.route∩R1.participants={B2}
    r2_support = [6, 7, 18, 19]
    # R3: indices [8,9,10] → logical ids [13,14,15] → B3 only (single-block)
    r3_support = [8, 9, 10]

    rotations: List[PauliRotation] = [
        PauliRotation(axis=make_axis(n, r1_support, "Y"), angle=math.pi/8,
                      source="R1_sched_demo", idx=0),
        PauliRotation(axis=make_axis(n, r2_support, "Y"), angle=math.pi/8,
                      source="R2_sched_demo", idx=1),
        PauliRotation(axis=make_axis(n, r3_support, "Z"), angle=math.pi/8,
                      source="R3_sched_demo", idx=2),
    ]

    support_map = {0: r1_support, 1: r2_support, 2: r3_support}
    block_map = {0: [2, 4], 1: [1, 3], 2: [3]}    # actual participant blocks (from lowering)

    print(f"\n{SEP2}")
    print("SECTION 2 — ROTATIONS (single commuting layer — disjoint supports)")
    print(SEP2)
    for rot in rotations:
        lbl  = ROT_LABELS[rot.idx]
        supp = support_map[rot.idx]
        blks = block_map[rot.idx]
        mb   = "multiblock" if len(blks) > 1 else "single-block"
        print(f"  {lbl} (idx={rot.idx}):  support={supp}  →  blocks {['B'+str(b) for b in blks]}  [{mb}]")

    # ── 3. Lower to ExecDAG ───────────────────────────────────────────────────
    cost_fn = make_gross_actual_cost_fn(plan)
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )
    effective_rots = {r.idx: r for r in rotations}
    t0_lower = time.perf_counter()
    res = lower_one_layer(
        layer_idx=0,
        rotations=effective_rots,
        rotation_indices=[r.idx for r in rotations],
        hw=hw,
        policies=policies,
    )
    dag = res.dag
    lower_time = time.perf_counter() - t0_lower

    print(f"\n{SEP2}")
    print("SECTION 3 — DAG STRUCTURE")
    print(SEP2)
    print(f"  Lowering completed in {lower_time:.2f} s")
    print(f"  Total nodes: {len(dag.nodes)}")

    # Group nodes by rotation index
    rot_nodes: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for nid in dag.nodes:
        ridx = ridx_from_nid(nid)
        if ridx is not None and ridx in rot_nodes:
            rot_nodes[ridx].append(nid)

    for ridx, nids in rot_nodes.items():
        lbl = ROT_LABELS[ridx]
        print(f"\n  {lbl}  ({len(nids)} nodes):")
        print(f"    {'node id':<50}  {'kind':<20}  {'dur':>4}  {'blocks'}")
        print(f"    {'─'*50}  {'─'*20}  {'─'*4}  {'─'*20}")
        for nid in sorted(nids, key=lambda x: (dag.nodes[x].kind, x)):
            node = dag.nodes[nid]
            preds = sorted(dag.pred.get(nid, set()))
            print(
                f"    {nid:<50}  {node.kind:<20}  {node.duration:>4}  "
                f"{node.blocks}  {'← '+', '.join(preds) if preds else '(root)'}"
            )

    # ── 4. Preprocessing: bottom levels, block criticality ────────────────────
    scheduler = GreedyCriticalScheduler()
    sched_prob = SchedulingProblem(
        dag=dag,
        hw=hw,
        seed=SEED,
        policy_name="incident_coupler_blocks_local",
        meta={
            "start_time":        0,
            "layer_idx":         0,
            "tie_breaker":       "duration",
            "safe_fill":         True,   # passed to scheduler (currently a no-op; DFS runs)
            "debug_decode":      False,
            "cp_sat_time_limit": None,
            "cp_sat_log":        False,
        },
    )
    prep = scheduler._preprocess(sched_prob)
    comp_order = scheduler._build_component_order(prep)

    print(f"\n{SEP2}")
    print("SECTION 4 — BOTTOM LEVEL PER NODE")
    print(SEP2)
    print("  Bottom-level BL(n) = dur(n) + max(BL(child)).")
    print("  Larger BL → node sits on a longer remaining-work path → higher scheduling priority.\n")
    print(f"  {'Rotation':<10}  {'node id':<50}  {'kind':<20}  {'dur':>4}  {'BL':>6}")
    print(f"  {'─'*10}  {'─'*50}  {'─'*20}  {'─'*4}  {'─'*6}")
    for ridx in [0, 1, 2]:
        for nid in sorted(rot_nodes[ridx], key=lambda x: -prep.bottom_level[x]):
            node = dag.nodes[nid]
            kind_s = KIND_SHORT.get(node.kind, node.kind[:8])
            print(
                f"  {ROT_LABELS[ridx]:<10}  {nid:<50}  {node.kind:<20}  "
                f"{node.duration:>4}  {prep.bottom_level[nid]:>6}"
            )

    print(f"\n{SEP2}")
    print("SECTION 5 — BLOCK CRITICALITY")
    print(SEP2)
    print("  block_criticality[b] = Σ bottom_level(n) for all nodes n touching block b.")
    print("  Nodes touching high-criticality blocks get a priority boost.\n")
    print(f"  {'Block':>7}  {'Occupancy':>10}  {'Block Criticality':>18}  {'Rotation(s) using it'}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*18}  {'─'*30}")
    for b in sorted(block_ids):
        occ   = sum(1 for q, blk in hw.logical_to_block.items() if blk == b)
        crit  = prep.block_criticality.get(b, 0)
        users = sorted(set(
            ridx_from_nid(nid)
            for nid, node in dag.nodes.items()
            if b in node.blocks and ridx_from_nid(nid) is not None
        ))
        user_lbl = " + ".join(ROT_LABELS[r] for r in users)
        print(f"  B{b:>5}  {occ:>10}  {crit:>18}  {user_lbl}")

    # ── 5. Component analysis & ordering ──────────────────────────────────────
    print(f"\n{SEP2}")
    print("SECTION 6 — COMPONENT ANALYSIS & PRIORITY ORDERING")
    print(SEP2)
    print("  Component ordering key: (-bottom_max, -block_crit_sum, -bottom_sum, -duration_sum, ...)")
    print(f"\n  {'rank':>4}  {'cid':>4}  {'rotation':>9}  {'multiblock':>10}  "
          f"{'bottom_max':>10}  {'blk_crit_sum':>12}  {'participant_blks':>18}  {'route_blks'}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*18}  {'─'*18}")

    # Map component → rotation (by finding which rot nids are in each component)
    cid_to_rot: Dict[int, int] = {}
    for cid, comp in enumerate(prep.components):
        for nid in comp:
            ridx = ridx_from_nid(nid)
            if ridx is not None:
                cid_to_rot[cid] = ridx
                break

    for rank, cid in enumerate(comp_order):
        m = prep.comp_metrics[cid]
        ridx = cid_to_rot.get(cid, -1)
        rot_lbl = ROT_LABELS.get(ridx, "?")
        mb  = prep.comp_is_multiblock[cid]
        pb  = sorted(prep.comp_participant_blocks[cid])
        rb  = sorted(prep.comp_route_blocks[cid])
        inter = sorted(prep.comp_intermediate_blocks[cid])
        print(
            f"  {rank:>4}  {cid:>4}  {rot_lbl:>9}  {'yes' if mb else 'no':>10}  "
            f"{m['bottom_max']:>10}  {m['block_crit_sum']:>12}  "
            f"{'B'+str(pb):>18}  B{rb}"
        )
        if inter:
            print(f"         intermediate blocks: B{inter}  (claimed at link step)")

    # ── 6. Deadlock analysis ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("SECTION 7 — DEADLOCK ANALYSIS  (waits-for graph + DFS cycle detection)")
    print(SEP2)

    mb_cids = [cid for cid in range(len(prep.components)) if prep.comp_is_multiblock[cid]]
    wf = waitsfor_edges(prep)
    cycles = find_cycles(wf)

    print("  Waits-for edges  (A → B means A.route_blocks ∩ B.participant_blocks ≠ ∅):\n")
    print(f"  {'Edge':>12}  {'A route blocks':>18}  ∩  {'B participant blocks':>22}  →  Shared")
    print(f"  {'─'*12}  {'─'*18}     {'─'*22}     {'─'*20}")
    for a in sorted(wf):
        for b in wf[a]:
            shared = prep.comp_route_blocks[a] & prep.comp_participant_blocks[b]
            a_rot = ROT_LABELS.get(cid_to_rot.get(a, -1), "?")
            b_rot = ROT_LABELS.get(cid_to_rot.get(b, -1), "?")
            rb_a = sorted(prep.comp_route_blocks[a])
            pb_b = sorted(prep.comp_participant_blocks[b])
            edge  = f"{a_rot}→{b_rot}"
            print(f"  {edge:>12}  B{rb_a!s:>17}  ∩  B{pb_b!s:>21}  →  B{sorted(shared)}")

    print(f"\n  Detected cycles ({len(cycles)} total):")
    if cycles:
        for c in cycles:
            rot_path = " → ".join(ROT_LABELS.get(cid_to_rot.get(x, -1), "?") for x in c)
            cid_path = " → ".join(str(x) for x in c)
            print(f"    [{cid_path}]  →  {rot_path} → {ROT_LABELS.get(cid_to_rot.get(c[0], -1), '?')}")
    else:
        print("    None.")

    print("\n  DFS deadlock-prevention outcome:")
    if cycles:
        for c in cycles:
            rots = [ROT_LABELS.get(cid_to_rot.get(x, -1), "?") for x in c]
            print(f"    Cycle {rots}: all but the first-admitted component are BLOCKED")
            print(f"    until the in-progress multiblock set becomes cycle-free.")
            # Find which one gets blocked by the comp_order
            first_admitted_rank = min(
                comp_order.index(x) for x in c if x in comp_order
            )
            first_cid = comp_order[first_admitted_rank]
            first_rot = ROT_LABELS.get(cid_to_rot.get(first_cid, -1), "?")
            blocked = [ROT_LABELS.get(cid_to_rot.get(x, -1), "?")
                       for x in c if x != first_cid]
            print(f"    → {first_rot} admitted first (rank {first_admitted_rank} in comp_order)")
            print(f"    → {blocked} blocked until {first_rot} releases its participant blocks")

    # ── 7. Safe-fill analysis (legacy Phase-2 heuristic) ─────────────────────
    print(f"\n{SEP2}")
    print("SECTION 8 — SAFE-FILL  (legacy Phase-2 heuristic, superseded)")
    print(SEP2)

    # Compute LB on when the admitted rotation's link node can start
    entries_preview: Dict[str, Any] = {}  # will be populated after solve()
    print("  Safe-fill asked: could a BLOCKED component B be admitted early")
    print("  if B's Xm completion LB < A's link-node start LB for all A in cycle?\n")
    print("  Safe-fill LB formula (per blocked component B):")
    print("    link_start_LB(A) = max(entries[pred].end  if pred scheduled")
    print("                           else t + BL[pred]   for pred in preds(A.link))")
    print("    xm_end_LB(B)     = t + BL[B.init_root]")
    print()
    for c in cycles:
        in_prog_cid = comp_order[min(comp_order.index(x) for x in c if x in comp_order)]
        blocked_cids = [x for x in c if x != in_prog_cid]
        in_prog_rot = ROT_LABELS.get(cid_to_rot.get(in_prog_cid, -1), "?")
        for b_cid in blocked_cids:
            b_rot = ROT_LABELS.get(cid_to_rot.get(b_cid, -1), "?")
            bl_b_root = prep.bottom_level.get(prep.comp_root[b_cid], 0)
            link_nid_a = prep.comp_link_node.get(in_prog_cid)
            if link_nid_a:
                bl_link_preds = [
                    prep.bottom_level[pred]
                    for pred in dag.pred.get(link_nid_a, set())
                ]
                link_lb = max(bl_link_preds) if bl_link_preds else 0
            else:
                link_lb = 0
            xm_lb = bl_b_root
            safe = xm_lb < link_lb
            print(f"  Blocked component: {b_rot} (cid={b_cid})")
            print(f"    BL[{b_rot}.init_root]          = {xm_lb}  → {b_rot}.xm_end_LB ≈ {xm_lb}")
            print(f"    {in_prog_rot}.link_start_LB (from preds) = {link_lb}")
            print(f"    Safe to admit speculatively?  {'YES ← safe-fill would admit' if safe else 'NO  ← blocked regardless'}")
            print(f"    Current implementation:        always uses DFS (safe_fill field is a no-op)")

    # ── 8. Run the scheduler ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("SECTION 9 — SCHEDULE EXECUTION")
    print(SEP2)

    t0_sched = time.perf_counter()
    schedule = scheduler.solve(sched_prob)
    sched_time = time.perf_counter() - t0_sched

    entries = schedule.meta.get("entries", {})
    makespan = schedule.meta.get("makespan", 0)
    cp_lb    = schedule.meta.get("cp_lower_bound", 0)
    gap      = schedule.meta.get("gap", 0)
    n_comp   = schedule.meta.get("num_components", 0)
    n_link   = schedule.meta.get("num_link_nodes", 0)

    print(f"\n  Scheduler:    greedy_critical  (elapsed: {sched_time:.2f} s)")
    print(f"  Components:   {n_comp}  (link nodes: {n_link})")
    print(f"  Makespan:     {makespan}")
    print(f"  CP lower-bound: {cp_lb}   gap: {gap}   "
          f"({'optimal' if gap == 0 else f'sub-optimal by {gap}'})")

    # Timeline: group schedule entries by start time, show per-rotation events
    print(f"\n  Schedule Timeline (key events only):")
    print(f"  {'time':>6}  {'rotation':>10}  {'kind':<20}  {'end':>6}  {'dur':>4}  node")
    print(f"  {'─'*6}  {'─'*10}  {'─'*20}  {'─'*6}  {'─'*4}  {'─'*30}")

    t_to_events: Dict[int, List[Tuple]] = {}
    for nid, se in entries.items():
        node  = dag.nodes[nid]
        ridx  = ridx_from_nid(nid)
        start = int(se["start"])
        end   = int(se["end"])
        dur   = end - start
        if node.kind == K_FRAME_UPDATE:
            continue   # zero-duration, not meaningful for timeline
        t_to_events.setdefault(start, []).append((ridx, node.kind, end, dur, nid))

    prev_time = -1
    for t in sorted(t_to_events):
        if t != prev_time and prev_time >= 0:
            print(f"  {'·':>6}")
        prev_time = t
        for ridx, kind, end, dur, nid in sorted(t_to_events[t], key=lambda x: (x[0], x[1])):
            rot_lbl = ROT_LABELS.get(ridx, "?") if ridx is not None else "—"
            print(f"  {t:>6}  {rot_lbl:>10}  {kind:<20}  {end:>6}  {dur:>4}  {nid}")

    # ── 9. Rotation timings ───────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("SECTION 10 — ROTATION TIMINGS")
    print(SEP2)
    print("  start = earliest start of any node in the rotation's gadget")
    print("  end   = latest   end   of any node in the rotation's gadget\n")

    timings = rot_timings(schedule, rotations, layer=0)

    print(f"  {'Rotation':<12}  {'Start':>6}  {'End':>6}  {'Duration':>9}  {'Blocks used'}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*20}")
    for rot in rotations:
        ridx = rot.idx
        blks = block_map[ridx]
        if ridx in timings:
            s, e = timings[ridx]
            print(f"  {ROT_LABELS[ridx]:<12}  {s:>6}  {e:>6}  {e-s:>9}  B{blks}")
        else:
            print(f"  {ROT_LABELS[ridx]:<12}  {'—':>6}  {'—':>6}  {'—':>9}  B{blks}")

    # Also compute naive serial depth (all rotations one after another, no overlap)
    naive_depth = sum(
        timings[ridx][1] - timings[ridx][0]
        for ridx in timings
    )
    print(f"\n  Actual makespan:               {makespan}")
    print(f"  Naive serial bound (sum dur):  {naive_depth}")
    print(f"  Parallelism gain:              {naive_depth - makespan} time steps saved"
          f"  ({100*(naive_depth-makespan)/max(naive_depth,1):.1f}%)")
    print(f"\n  CP lower bound:   {cp_lb}  (tightest possible makespan given node durations)")
    print(f"  Gap to optimal:   {gap}")

    # ── 10. Summary + block ownership ─────────────────────────────────────────
    print(f"\n{SEP2}")
    print("SECTION 11 — BLOCK OWNERSHIP INTERVALS")
    print(SEP2)
    print("  Each block is exclusively owned during the gadget span that touches it.")
    print("  Non-overlapping intervals confirm deadlock was correctly prevented.\n")

    block_ownerships = schedule.meta.get("block_ownership_intervals", {})
    print(f"  {'Block':>7}  {'owner cid':>10}  {'start':>6}  {'end':>6}  {'rotation'}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*12}")
    for b in sorted(block_ids):
        intervals = block_ownerships.get(b, [])
        for iv in sorted(intervals, key=lambda x: x[0]):
            iv_start, iv_end, iv_cid = iv[0], iv[1], iv[2]
            rot_lbl = ROT_LABELS.get(cid_to_rot.get(iv_cid, -1), "?")
            print(f"  B{b:>5}  {iv_cid:>10}  {iv_start:>6}  {iv_end:>6}  {rot_lbl}")

    # ── 11. Save JSON ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "hardware": {
            "topology":   "ring",
            "n_blocks":   spec.n_blocks,
            "n_data":     N_DATA,
            "n_logicals": N_QUBITS,
            "block_ids":  block_ids,
            "ring_order": block_ids + [block_ids[0]],
        },
        "rotations": [
            {
                "label":    ROT_LABELS[rot.idx],
                "idx":      rot.idx,
                "support":  support_map[rot.idx],
                "blocks":   block_map[rot.idx],
                "multiblock": len(block_map[rot.idx]) > 1,
            }
            for rot in rotations
        ],
        "dag": {
            "n_nodes":       len(dag.nodes),
            "n_link_nodes":  n_link,
            "n_components":  n_comp,
        },
        "bottom_levels": {
            nid: prep.bottom_level[nid] for nid in dag.nodes
        },
        "block_criticality": {
            str(b): prep.block_criticality.get(b, 0) for b in block_ids
        },
        "component_analysis": [
            {
                "cid":                cid,
                "rank":               rank,
                "rotation":           ROT_LABELS.get(cid_to_rot.get(cid, -1), "?"),
                "multiblock":         prep.comp_is_multiblock[cid],
                "participant_blocks": sorted(prep.comp_participant_blocks[cid]),
                "route_blocks":       sorted(prep.comp_route_blocks[cid]),
                "intermediate_blocks": sorted(prep.comp_intermediate_blocks[cid]),
                "metrics":            prep.comp_metrics[cid],
            }
            for rank, cid in enumerate(comp_order)
        ],
        "deadlock_analysis": {
            "waitsfor_edges": {
                str(a): wf[a] for a in wf
            },
            "cycles": cycles,
            "cycle_rotations": [
                [ROT_LABELS.get(cid_to_rot.get(x, -1), "?") for x in c]
                for c in cycles
            ],
        },
        "schedule": {
            "makespan":        makespan,
            "cp_lower_bound":  cp_lb,
            "gap":             gap,
            "entries": {nid: {"start": se["start"], "end": se["end"]}
                        for nid, se in entries.items()},
        },
        "rotation_timings": {
            ROT_LABELS[ridx]: {"start": s, "end": e, "duration": e - s}
            for ridx, (s, e) in timings.items()
        },
        "block_ownership_intervals": {
            str(b): block_ownerships.get(b, []) for b in block_ids
        },
    }
    out_path = os.path.join(RESULTS_DIR, "sched_demo_results.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── 12. Figure flow suggestion ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUGGESTED FIGURE FLOW")
    print(SEP)
    print("""
  PANEL A — Ring hardware + block criticality heat map
  ─────────────────────────────────────────────────────
    Visualise: 4-block ring (circle of nodes B1-B2-B3-B4).
    Node colour = block_criticality score (use results["block_criticality"]).
    Draw rotation support arcs:
      R₁: arc B1 ↔ B3 (red, thick)   — will need B2 as intermediate
      R₂: arc B2 ↔ B4 (blue, thick)  — will need B3 as intermediate
      R₃: dot on B2   (green, small)  — single-block
    Label each block with its criticality score and occupancy.
    Caption note: "B2 and B3 are hot blocks — both multi-block routes
    pass through them — driving their high criticality score."

  PANEL B — DAG with bottom levels
  ──────────────────────────────────
    Draw the ExecDAG for all three rotations as a combined directed graph.
    Use three columns (one per rotation) and six rows (one per node kind):
      init → lc → link → PZ → Xm → frame (top to bottom).
    Annotate each node with (kind, duration, BL).
    Highlight the critical path (highest BL chain) in a bold colour.
    Use different colours per rotation.
    Show inter-rotation edges if any (there are none for disjoint supports;
    make this explicit — "no data dependency between rotations").

  PANEL C — Waits-for graph + DFS cycle detection
  ─────────────────────────────────────────────────
    Draw three component boxes: R₁, R₂, R₃.
    Draw directed waits-for edges (from results["deadlock_analysis"]):
      R₁ → R₂  (R₁ route needs B2, a participant of R₂)
      R₂ → R₁  (R₂ route needs B3, a participant of R₁)
    Highlight the 2-cycle in red with label "DEADLOCK if both admitted".
    R₃ has no outgoing edges → no cycle risk.
    Show DFS trace box: "Before admitting R₂, DFS from R₂ via R₁ returns to R₂
    → cycle detected → R₂ blocked."
    Sub-panel: safe-fill LB values (B.xm_end_LB vs A.link_start_LB) — use the
    numbers from Section 8 to annotate whether safe-fill would have admitted or
    blocked.

  PANEL D — Schedule Gantt chart (most impactful panel)
  ───────────────────────────────────────────────────────
    X-axis: time steps 0 … makespan.
    Y-axis: one lane per block (B1, B2, B3, B4).
    Colour-fill ownership intervals (from results["block_ownership_intervals"]):
      R₁ colour for B1, B2 (intermediate), B3
      R₂ colour for B2, B3 (intermediate), B4
      R₃ colour for B2
    Overlay horizontal bars per rotation spanning their [start, end] window.
    Mark the CP lower bound as a vertical dashed line.
    Annotate: "R₃ (green) and R₁ (red) overlap in time — B2 is free during
    R₁ init, so R₃ runs in parallel."
    Annotate the gap (if any) between makespan and CP-LB.
    Add a comparison inset or sibling bar: naive serial depth vs actual makespan.

  PANEL E (optional) — Priority score table
  ───────────────────────────────────────────
    Simple 4-column table from Section 6:
    Rank | Rotation | bottom_max | block_crit_sum
    Underline R₁ in the table and draw an arrow to its Gantt bar (Panel D)
    to show "highest rank → scheduled first → parallel execution unlocked."
""")


if __name__ == "__main__":
    main()
