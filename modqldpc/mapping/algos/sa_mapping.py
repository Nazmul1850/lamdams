# modqldpc/mapping/algos/sa_mapping.py
from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..base import BaseMapper
from ..types import MappingPlan, MappingProblem, MappingConfig
from ..model import HardwareGraph, BlockId, LogicalId, LocalId


# ============================================================
# Tuned score weights (from sensitivity analysis, phases 1–3)
#
# Phase 1 (one-at-a-time screening, 9 circuits × 2 topologies):
#   Selected for impact: W_SPAN, W_UNUSED_BLOCKS, W_OCC_RANGE, W_OCC_STD,
#                        W_MULTI_BLOCK, W_MST, W_SUPPORT_PEAK, W_SUPPORT_RANGE
#   Screened out (depth-span < 1%, inter-span < 1%): W_SPLIT, W_SUPPORT_STD
#
# Phase 2 (cumulative greedy selection, accept if Δ mean-depth-gain ≥ 0.2%):
#   W_SPAN:       50000 → 10000  (+2.3% gain)
#   W_OCC_STD:     1000 →  5000  (accepted refinement)
#   W_MULTI_BLOCK: 200000 → 0    (dropped — flat multiblock count conflicted
#                                  with span/MST; span alone is sufficient)
#   All others: base value confirmed.
#
# Phase 3 (hyperparameter sweep):
#   steps: 25000 → 22500   (plateau reached earlier; saves ~10% runtime)
#   t0 and t_end unchanged: 1e5 and 0.05
#
# Zero-weight terms (W_MULTI_BLOCK, W_SUPPORT_STD) are not computed.
# ============================================================

TUNED_SCORE_KWARGS: Dict[str, float] = {
    "W_UNUSED_BLOCKS": 1_000_000.0,
    "W_OCC_RANGE":        10_000.0,
    "W_OCC_STD":           5_000.0,
    "W_MULTI_BLOCK":           0.0,   # tuned to 0 → skipped in scoring
    "W_SPAN":             10_000.0,
    "W_MST":                 500.0,
    "W_SPLIT":                10.0,
    "W_SUPPORT_PEAK":        100.0,
    "W_SUPPORT_RANGE":        20.0,
    "W_SUPPORT_STD":           0.0,   # tuned to 0 → skipped in scoring
}

TUNED_SA_STEPS: int   = 22_500
TUNED_SA_T0:    float = 1e5
TUNED_SA_TEND:  float = 5e-2


# ============================================================
# Support extraction
# ============================================================

def _rotation_support(rot) -> Set[LogicalId]:
    """
    Return the set of logical qubit indices (0-based) where rot.axis is non-identity.
    rot.axis is a Pauli word string like 'IXYZ' where the RIGHTMOST character is qubit 0.
    """
    word = rot.axis.lstrip("+-")
    supp: Set[LogicalId] = set()
    for k, ch in enumerate(reversed(word)):
        if ch != "I":
            supp.add(k)
    return supp


def _blocks_touched(rot, hw: HardwareGraph) -> Set[BlockId]:
    """Return the set of blocks that a PauliRotation's support spans."""
    blocks: Set[BlockId] = set()
    for q in _rotation_support(rot):
        if q in hw.logical_to_block:
            blocks.add(hw.logical_to_block[q])
    return blocks


# ============================================================
# Occupancy helpers
# ============================================================

def _build_occupancy(hw: HardwareGraph) -> Dict[BlockId, List[Optional[LogicalId]]]:
    """occ[b][local] = logical_id or None, reflecting current hw state."""
    occ: Dict[BlockId, List[Optional[LogicalId]]] = {
        b: [None] * hw.blocks[b].num_logicals for b in hw.blocks
    }
    for q, b in hw.logical_to_block.items():
        l = hw.logical_to_local[q]
        occ[b][l] = q
    return occ


def _first_free(occ: Dict[BlockId, List[Optional[LogicalId]]], b: BlockId) -> Optional[int]:
    for i, x in enumerate(occ[b]):
        if x is None:
            return i
    return None


def _occupancy_counts(hw: HardwareGraph) -> Dict[BlockId, int]:
    """Return per-block logical qubit count."""
    counts: Dict[BlockId, int] = {b: 0 for b in hw.blocks}
    for b in hw.logical_to_block.values():
        counts[b] += 1
    return counts


# ============================================================
# Graph helpers: BFS distance + MST
# ============================================================

def _bfs_dist(hw: HardwareGraph, src: BlockId) -> Dict[BlockId, int]:
    """Unweighted BFS hop-distances from src over the full hardware graph."""
    q: deque = deque([src])
    dist: Dict[BlockId, int] = {src: 0}
    while q:
        u = q.popleft()
        for v in hw.neighbors.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _is_connected(blocks: Set[BlockId], hw: HardwareGraph) -> bool:
    """True iff the induced subgraph on `blocks` is connected in hw."""
    if len(blocks) <= 1:
        return True
    start = next(iter(blocks))
    seen = {start}
    queue: deque = deque([start])
    while queue:
        u = queue.popleft()
        for v in hw.neighbors.get(u, set()):
            if v in blocks and v not in seen:
                seen.add(v)
                queue.append(v)
    return len(seen) == len(blocks)


def _mst_len(blocks: Set[BlockId], hw: HardwareGraph) -> int:
    """
    MST length (hop-count) over `blocks` using the hw shortest-path metric.
    Returns 1_000_000 if blocks are disconnected in hw.
    """
    if len(blocks) <= 1:
        return 0
    nodes = list(blocks)

    # All-pairs shortest-path distances (BFS from each node in blocks)
    dists: Dict[Tuple[BlockId, BlockId], int] = {}
    for u in nodes:
        du = _bfs_dist(hw, u)
        for v in nodes:
            if u == v:
                continue
            if v not in du:
                return 1_000_000
            dists[(u, v)] = du[v]

    # Prim's algorithm
    start = nodes[0]
    in_mst: Set[BlockId] = {start}
    pq: List[Tuple[int, BlockId]] = [(dists[(start, v)], v) for v in nodes[1:]]
    heapq.heapify(pq)
    total = 0
    while pq and len(in_mst) < len(nodes):
        w, v = heapq.heappop(pq)
        if v in in_mst:
            continue
        in_mst.add(v)
        total += w
        for x in nodes:
            if x not in in_mst:
                heapq.heappush(pq, (dists[(v, x)], x))

    if len(in_mst) != len(nodes):
        return 1_000_000
    return total


# ============================================================
# Move operators
# ============================================================

# Move type: ("noop", (q,))
#            ("move", (q, old_b, old_l, new_b, new_l, swap_q, swap_orig_b, swap_orig_l))
Move = Tuple[str, tuple]


def _swap_logicals(hw: HardwareGraph, q1: LogicalId, q2: LogicalId) -> None:
    """Swap (block, local) of q1 and q2 in-place."""
    b1, l1 = hw.logical_to_block[q1], hw.logical_to_local[q1]
    b2, l2 = hw.logical_to_block[q2], hw.logical_to_local[q2]
    hw.logical_to_block[q1], hw.logical_to_local[q1] = b2, l2
    hw.logical_to_block[q2], hw.logical_to_local[q2] = b1, l1


def _random_move(hw: HardwareGraph, rng: random.Random) -> Move:
    """
    Pick a random logical and move it to a random (different) block.
    - If target block has a free slot: simple move.
    - Else: swap with a random occupant of the target block.
    Returns a move descriptor sufficient for undo.
    """
    qs = list(hw.logical_to_block.keys())
    q = rng.choice(qs)
    old_b = hw.logical_to_block[q]
    old_l = hw.logical_to_local[q]

    blocks = sorted(hw.blocks.keys())
    new_b = rng.choice(blocks)
    if new_b == old_b:
        return ("noop", (q,))

    occ = _build_occupancy(hw)
    free = _first_free(occ, new_b)

    if free is not None:
        # simple move into free slot
        hw.logical_to_block[q] = new_b
        hw.logical_to_local[q] = free
        return ("move", (q, old_b, old_l, new_b, free, None, None, None))

    # swap with a random occupant
    occupants = [x for x in occ[new_b] if x is not None]
    swap_q = rng.choice(occupants)
    swap_orig_b = hw.logical_to_block[swap_q]   # = new_b (before swap)
    swap_orig_l = hw.logical_to_local[swap_q]   # local of swap_q in new_b
    _swap_logicals(hw, q, swap_q)
    # after swap: q is at (new_b, swap_orig_l), swap_q is at (old_b, old_l)
    return ("move", (q, old_b, old_l, new_b, swap_orig_l, swap_q, swap_orig_b, swap_orig_l))


def _undo_move(hw: HardwareGraph, move: Move) -> None:
    mtype, args = move
    if mtype == "noop":
        return
    if mtype == "move":
        q, old_b, old_l, new_b, new_l, swap_q, swap_orig_b, swap_orig_l = args
        hw.logical_to_block[q] = old_b
        hw.logical_to_local[q] = old_l
        if swap_q is not None:
            # restore swap_q to where it was before (new_b, swap_orig_l)
            hw.logical_to_block[swap_q] = swap_orig_b
            hw.logical_to_local[swap_q] = swap_orig_l
        return
    raise ValueError(f"Unknown move type: {mtype!r}")


# ============================================================
# Scoring helpers
# ============================================================

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _support_split(rot, hw: HardwareGraph, blocks: List[BlockId]) -> float:
    """L2 split: std of per-block qubit counts across touched blocks."""
    per_block: Dict[BlockId, int] = {b: 0 for b in blocks}
    for q in _rotation_support(rot):
        b = hw.logical_to_block.get(q)
        if b in per_block:
            per_block[b] += 1
    counts = list(per_block.values())
    return _std([float(c) for c in counts])


# ============================================================
# Score dataclass and scoring function
# ============================================================

@dataclass
class ScoreBreakdown:
    total: float
    # Block utilization
    active_blocks: int
    unused_blocks: int
    occupancy_range: float
    occupancy_std: float
    # Multi-block / span
    num_multiblock: int
    span_total: float
    max_blocks_touched: int
    # Locality
    mst_total: float
    # Support balance
    split_total: float
    # Support load
    support_peak: float
    support_range: float
    # Penalty terms (for metadata / debug)
    unused_block_pen: float
    occupancy_range_pen: float
    occupancy_std_pen: float
    span_pen: float
    mst_pen: float
    split_pen: float
    support_peak_pen: float
    support_range_pen: float


def _score(
    rotations: list,
    hw: HardwareGraph,
    *,
    W_UNUSED_BLOCKS: float = TUNED_SCORE_KWARGS["W_UNUSED_BLOCKS"],
    W_OCC_RANGE:     float = TUNED_SCORE_KWARGS["W_OCC_RANGE"],
    W_OCC_STD:       float = TUNED_SCORE_KWARGS["W_OCC_STD"],
    W_MULTI_BLOCK:   float = TUNED_SCORE_KWARGS["W_MULTI_BLOCK"],  # 0.0
    W_SPAN:          float = TUNED_SCORE_KWARGS["W_SPAN"],
    W_MST:           float = TUNED_SCORE_KWARGS["W_MST"],
    W_SPLIT:         float = TUNED_SCORE_KWARGS["W_SPLIT"],
    W_SUPPORT_PEAK:  float = TUNED_SCORE_KWARGS["W_SUPPORT_PEAK"],
    W_SUPPORT_RANGE: float = TUNED_SCORE_KWARGS["W_SUPPORT_RANGE"],
    W_SUPPORT_STD:   float = TUNED_SCORE_KWARGS["W_SUPPORT_STD"],  # 0.0
) -> ScoreBreakdown:
    """
    Energy function to minimise.

    Terms and tuned weights (from sensitivity analysis):
      W_UNUSED_BLOCKS  1e6  — use all hardware blocks              (dominant)
      W_OCC_RANGE      1e4  — even out block occupancy (max-min)
      W_OCC_STD        5e3  — smooth occupancy distribution (std)
      W_SPAN           1e4  — penalise extra blocks touched (k-1)
      W_MST            5e2  — prefer spatially compact multi-block spans
      W_SPLIT          10   — balance qubit support across touched blocks
      W_SUPPORT_PEAK   1e2  — reduce busiest block under rotation load
      W_SUPPORT_RANGE  20   — flatten rotation-load range across blocks
      W_MULTI_BLOCK    0    — (screened out; span already captures this)
      W_SUPPORT_STD    0    — (screened out; low impact)

    Zero-weight terms are not computed.
    """
    occupancies = _occupancy_counts(hw)
    support_loads: Dict[BlockId, int] = {b: 0 for b in hw.blocks}

    num_multiblock  = 0
    span_total      = 0.0
    mst_total       = 0.0
    split_total     = 0.0
    max_blocks_touched = 0

    for rot in rotations:
        touched = sorted(_blocks_touched(rot, hw))
        k = len(touched)
        if k > max_blocks_touched:
            max_blocks_touched = k
        for b in touched:
            support_loads[b] += 1

        if k > 1:
            num_multiblock += 1
            span_total += float(k - 1)
            if W_MST != 0.0:
                mst_total += float(_mst_len(set(touched), hw))
            if W_SPLIT != 0.0:
                split_total += _support_split(rot, hw, touched)

    # Occupancy statistics
    occ_values = [float(v) for v in occupancies.values()]
    active_blocks = sum(1 for v in occ_values if v > 0)
    unused_blocks = len(occ_values) - active_blocks
    occupancy_range = float(max(occ_values) - min(occ_values)) if occ_values else 0.0
    occupancy_std   = _std(occ_values) if W_OCC_STD != 0.0 else 0.0

    # Support load statistics
    sup_values = [float(v) for v in support_loads.values()]
    support_peak  = float(max(sup_values)) if sup_values else 0.0
    support_range = float(max(sup_values) - min(sup_values)) if sup_values else 0.0
    # W_SUPPORT_STD == 0.0 → skip std computation

    # Penalty terms
    unused_block_pen    = W_UNUSED_BLOCKS * float(unused_blocks)
    occupancy_range_pen = W_OCC_RANGE     * occupancy_range
    occupancy_std_pen   = W_OCC_STD       * occupancy_std
    span_pen            = W_SPAN          * span_total
    mst_pen             = W_MST           * mst_total
    split_pen           = W_SPLIT         * split_total
    support_peak_pen    = W_SUPPORT_PEAK  * support_peak
    support_range_pen   = W_SUPPORT_RANGE * support_range
    # W_MULTI_BLOCK == 0.0 and W_SUPPORT_STD == 0.0 → not added

    total = (
        unused_block_pen
        + occupancy_range_pen
        + occupancy_std_pen
        + span_pen
        + mst_pen
        + split_pen
        + support_peak_pen
        + support_range_pen
    )

    return ScoreBreakdown(
        total=total,
        active_blocks=active_blocks,
        unused_blocks=unused_blocks,
        occupancy_range=occupancy_range,
        occupancy_std=occupancy_std,
        num_multiblock=num_multiblock,
        span_total=span_total,
        max_blocks_touched=max_blocks_touched,
        mst_total=mst_total,
        split_total=split_total,
        support_peak=support_peak,
        support_range=support_range,
        unused_block_pen=unused_block_pen,
        occupancy_range_pen=occupancy_range_pen,
        occupancy_std_pen=occupancy_std_pen,
        span_pen=span_pen,
        mst_pen=mst_pen,
        split_pen=split_pen,
        support_peak_pen=support_peak_pen,
        support_range_pen=support_range_pen,
    )


def print_score_debug(score: ScoreBreakdown, *, title: str = "SCORE DEBUG") -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"  total              : {score.total:.2f}")
    print(f"  active_blocks      : {score.active_blocks}  (unused={score.unused_blocks})")
    print(f"  unused_block_pen   : {score.unused_block_pen:.2f}")
    print(f"  occupancy_range    : {score.occupancy_range:.4f}  pen={score.occupancy_range_pen:.2f}")
    print(f"  occupancy_std      : {score.occupancy_std:.4f}  pen={score.occupancy_std_pen:.2f}")
    print(f"  span_total         : {score.span_total:.2f}  pen={score.span_pen:.2f}")
    print(f"  mst_total          : {score.mst_total:.2f}  pen={score.mst_pen:.2f}")
    print(f"  split_total        : {score.split_total:.2f}  pen={score.split_pen:.2f}")
    print(f"  support_peak       : {score.support_peak:.2f}  pen={score.support_peak_pen:.2f}")
    print(f"  support_range      : {score.support_range:.2f}  pen={score.support_range_pen:.2f}")
    print(f"  num_multiblock     : {score.num_multiblock}")
    print(f"  max_blocks_touched : {score.max_blocks_touched}")
    print(bar)


# ============================================================
# Debug diagnostics
# ============================================================

def _debug_state(rotations: list, hw: HardwareGraph, cfg: MappingConfig) -> None:
    """Print a state snapshot to diagnose why SA may not be improving."""
    print(f"\n[SA-debug] === State snapshot ===")
    print(f"  hw mapped logicals : {len(hw.logical_to_block)}")
    print(f"  hw blocks          : {sorted(hw.blocks.keys())}")
    print(f"  rotations          : {len(rotations)}")
    print(f"  cfg.sa_steps/t0/tend: {cfg.sa_steps} / {cfg.sa_t0} / {cfg.sa_tend}")

    multi_block = 0
    no_support  = 0
    unmatched   = 0
    for rot in rotations:
        supp   = _rotation_support(rot)
        blocks = _blocks_touched(rot, hw)
        if not supp:
            no_support += 1
        elif not blocks:
            unmatched += 1
        elif len(blocks) > 1:
            multi_block += 1

    print(f"  rotations w/ empty support   : {no_support}")
    print(f"  rotations w/ unmatched qubits: {unmatched}  <- nonzero = logical ID mismatch")
    print(f"  rotations spanning >1 block  : {multi_block}")

    load: Dict[BlockId, int] = {b: 0 for b in hw.blocks}
    for rot in rotations:
        for b in _blocks_touched(rot, hw):
            load[b] += 1
    qubit_count: Dict[BlockId, int] = {b: 0 for b in hw.blocks}
    for b in hw.logical_to_block.values():
        qubit_count[b] += 1
    print(f"  {'block':>6}  {'logicals':>8}  {'capacity':>8}  {'rot_load':>8}")
    for b in sorted(hw.blocks.keys()):
        cap = hw.blocks[b].num_logicals
        print(f"  {b:>6}  {qubit_count[b]:>8}  {cap:>8}  {load[b]:>8}")

    s = _score(rotations, hw)
    print_score_debug(s, title="Initial score")

    rng = random.Random(0)
    deltas = []
    for _ in range(20):
        move = _random_move(hw, rng)
        nxt = _score(rotations, hw)
        deltas.append(nxt.total - s.total)
        _undo_move(hw, move)
    nonzero = sum(1 for d in deltas if d != 0.0)
    print(f"  20 probe moves: {nonzero}/20 changed score  "
          f"(min delta={min(deltas):.1f}, max delta={max(deltas):.1f})")
    print(f"  <- if 0/20 changed: score is stuck (likely all noops or score=0)\n")


# ============================================================
# Simulated annealing loop
# ============================================================

def _anneal(
    rotations: list,
    hw: HardwareGraph,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    report_every: int = 2_000,
) -> Tuple[ScoreBreakdown, Dict[LogicalId, Tuple[BlockId, LocalId]]]:
    """
    SA over hw.logical_to_block / hw.logical_to_local (mutated in-place).
    Geometric temperature schedule from t0 down to t_end.
    Restores the best-ever mapping into hw before returning.

    Returns:
        (best_score, best_map)  where best_map[q] = (block, local)
    """
    rng = random.Random(seed)
    skw = score_kwargs or {}

    def _snapshot() -> Dict[LogicalId, Tuple[BlockId, LocalId]]:
        return {q: (hw.logical_to_block[q], hw.logical_to_local[q]) for q in hw.logical_to_block}

    best_map = _snapshot()
    cur  = _score(rotations, hw, **skw)
    best = cur

    n_noop   = 0
    n_accept = 0
    n_reject = 0

    for it in range(1, steps + 1):
        frac = (it - 1) / max(1, steps - 1)
        T    = t0 * ((t_end / t0) ** frac)

        move = _random_move(hw, rng)
        if move[0] == "noop":
            n_noop += 1
            continue

        nxt   = _score(rotations, hw, **skw)
        delta = nxt.total - cur.total
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(1e-12, T))

        if accept:
            n_accept += 1
            cur = nxt
            if cur.total < best.total:
                best     = cur
                best_map = _snapshot()
        else:
            n_reject += 1
            _undo_move(hw, move)

        if verbose and report_every and (it % report_every == 0):
            print(
                f"[SA-mapping] it={it:6d}  T={T:.4f}  "
                f"cur={cur.total:.1f}  best={best.total:.1f}  "
                f"(unused={cur.unused_blocks} span={cur.span_total:.1f} "
                f"mst={cur.mst_total:.1f} peak={cur.support_peak:.1f})  "
                f"accept={n_accept}  reject={n_reject}  noop={n_noop}"
            )
            n_noop = n_accept = n_reject = 0

    # Restore best mapping into hw
    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    for q, (b, l) in best_map.items():
        hw.logical_to_block[q] = b
        hw.logical_to_local[q] = l

    return best, best_map


# ============================================================
# Mapper class
# ============================================================

@dataclass(frozen=True)
class SimulatedAnnealingMapper(BaseMapper):
    """
    SA mapper using tuned scoring and hyperparameters from sensitivity analysis.

    Compatible with the factory pattern:

        mapper = get_mapper("simulated_annealing")
        plan   = mapper.solve(problem, hw, cfg, meta={"rotations": converter.program.rotations})

    meta keys:
        rotations  (required) : list[PauliRotation] from GoSCConverter.program.rotations
        verbose    (optional) : bool, print SA progress (default False)
        score_kwargs (optional): override individual score weights

    SA temperature schedule and step count are read from cfg:
        cfg.sa_steps, cfg.sa_t0, cfg.sa_tend, cfg.seed
    Tuned defaults (TUNED_SA_STEPS=22500, TUNED_SA_T0=1e5, TUNED_SA_TEND=5e-2)
    are applied in run_experiment.py via SA_STEPS / SA_T0 / SA_TEND constants.
    """
    name: str = "simulated_annealing"
    init_mapper_name: str = "auto_round_robin_mapping"

    def solve(
        self,
        problem: MappingProblem,
        hw: HardwareGraph,
        cfg: MappingConfig,
        meta: Optional[dict] = None,
    ) -> MappingPlan:
        meta = meta or {}
        rotations = meta.get("rotations")
        if rotations is None:
            raise ValueError(
                "SimulatedAnnealingMapper requires meta['rotations'] "
                "(list of PauliRotation from GoSCConverter.program.rotations)."
            )
        verbose: bool = meta.get("verbose", False)
        debug:   bool = meta.get("debug",   False)

        # Build score kwargs: start from tuned defaults, allow caller overrides
        skw = dict(TUNED_SCORE_KWARGS)
        skw.update(meta.get("score_kwargs", {}) or {})

        from ..factory import get_mapper

        # 1) Build initial plan (seeds hw.logical_to_block / hw.logical_to_local)
        init_mapper = get_mapper(self.init_mapper_name)
        init_mapper.solve(problem, hw, cfg)

        if debug:
            _debug_state(rotations, hw, cfg)

        # 2) Run SA — mutates hw in-place, restores best at end
        best_score, _ = _anneal(
            rotations,
            hw,
            steps=cfg.sa_steps,
            t0=cfg.sa_t0,
            t_end=cfg.sa_tend,
            seed=cfg.seed,
            score_kwargs=skw,
            verbose=verbose,
        )

        # 3) Build MappingPlan from hw state (already at best)
        out_b = dict(hw.logical_to_block)
        out_l = dict(hw.logical_to_local)

        return MappingPlan(
            out_b,
            out_l,
            meta={
                "mapper":               self.name,
                "init_mapper":          self.init_mapper_name,
                "sa_steps":             cfg.sa_steps,
                "sa_t0":                cfg.sa_t0,
                "sa_tend":              cfg.sa_tend,
                "score_kwargs":         skw,
                "best_score_total":     best_score.total,
                "active_blocks":        best_score.active_blocks,
                "unused_blocks":        best_score.unused_blocks,
                "occupancy_range":      best_score.occupancy_range,
                "occupancy_std":        best_score.occupancy_std,
                "num_multiblock":       best_score.num_multiblock,
                "span_total":           best_score.span_total,
                "mst_total":            best_score.mst_total,
                "split_total":          best_score.split_total,
                "support_peak":         best_score.support_peak,
                "support_range":        best_score.support_range,
                "max_blocks_touched":   best_score.max_blocks_touched,
                "n_rotations":          len(rotations),
            },
        )
