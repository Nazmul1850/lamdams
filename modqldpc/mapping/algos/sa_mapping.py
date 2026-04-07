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
# Scoring
# ============================================================

@dataclass
class RotationDebug:
    """Per-rotation penalty breakdown, collected when ENABLE_DEBUG=True."""
    ridx: int
    layer: Optional[int]
    word: str
    support: Tuple[int, ...]
    touched_blocks: Tuple[BlockId, ...]
    k: int
    support_size: int
    per_block_counts: Dict[BlockId, int]
    raw_mst: float
    raw_split: float
    multi_block_pen: float
    mst_pen: float
    split_pen: float
    total_pen: float


@dataclass
class ScoreBreakdown:
    total: float
    unused_blocks_pen: float   # W_UNUSED_BLOCKS * n_unused — primary: use all blocks
    multi_block_pen: float     # W_MULTI_BLOCK  * num_multiblock — secondary: reduce inter-block
    mst_pen: float             # W_MST          * total_mst — tertiary: spatial compactness
    split_pen: float           # W_SPLIT        * total_split — quaternary: balance support
    block_loads: Dict[BlockId, int] = field(default_factory=dict)
    num_multiblock: int = 0
    n_unused_blocks: int = 0
    max_blocks_touched: int = 0
    top_rotations: List[RotationDebug] = field(default_factory=list)


def _score(
    rotations: list,
    hw: HardwareGraph,
    *,
    W_PEAK: float = 1e4,
    W_RANGE: float = 5e3,
    W_STD: float = 0.0,
    W_DISCONNECTED: float = 0.0,
    W_MST: float = 0.0,
    W_SPLIT: float = 1e2,
    W_SPAN: float = 1e3,
    SPLIT_MODE: str = "l2",
    RETURN_TOP_K: int = 10,
    ENABLE_DEBUG: bool = False,
) -> ScoreBreakdown:
    """
    Energy to minimise.  For each PauliRotation, find S = set of blocks its
    support touches under the current mapping.

    Weight hierarchy: W_PEAK > W_RANGE > W_DISCONNECTED > W_MST > W_SPLIT
    (W_SPAN=0.0 by default — span count alone is not penalised directly)

      - peak_load_pen:     W_PEAK * max(load) over active blocks
                           PRIMARY: prevent one block becoming global bottleneck.
      - range_load_pen:    W_RANGE * (max - min) load over active blocks
                           SECONDARY: reduce load imbalance across blocks.
      - std_load_pen:      W_STD * std(load) over active blocks (off by default).
      - disconnected_pen:  W_DISCONNECTED * count of disconnected multi-block spans
                           HARD CONSTRAINT: disconnected is always worst.
      - mst_pen:           W_MST * sum of MST hop-lengths for multi-block rotations
                           TERTIARY: compactness when multi-block unavoidable.
      - split_pen:         W_SPLIT * support-imbalance over multi-block rotations
                           QUATERNARY: prefer equal qubit split across blocks.
      - span_pen:          W_SPAN * (k-1) per rotation (disabled by default).

    SPLIT_MODE: "l1" (max-min counts), "l2" (std of counts), "pairwise" (max |ci-cj|).
    """
    span = 0.0
    disconn = 0.0
    mst = 0.0
    split = 0.0

    load: Dict[BlockId, int] = {b: 0 for b in hw.blocks}
    num_multiblock = 0
    num_disconnected = 0
    max_blocks_touched = 0
    debug_rows: List[RotationDebug] = []

    for ridx, rot in enumerate(rotations):
        S = _blocks_touched(rot, hw)
        k = len(S)
        for b in S:
            load[b] += 1
        if k > max_blocks_touched:
            max_blocks_touched = k

        raw_span = float(k - 1) if k > 1 else 0.0
        raw_disconnected = 0.0
        raw_mst = 0.0
        raw_split_val = 0.0
        per_block_counts: Dict[BlockId, int] = {}

        if k > 1:
            num_multiblock += 1
            span += k - 1

            if not _is_connected(S, hw):
                disconn += 1.0
                raw_disconnected = 1.0
                num_disconnected += 1

            mst_val = _mst_len(S, hw)
            mst += mst_val
            raw_mst = float(mst_val)

            # per-block qubit support counts for split penalty
            per_block_counts = {b: 0 for b in S}
            for q in _rotation_support(rot):
                b = hw.logical_to_block.get(q)
                if b in per_block_counts:
                    per_block_counts[b] += 1
            counts = list(per_block_counts.values())

            if SPLIT_MODE == "l1":
                raw_split_val = float(max(counts) - min(counts))
            elif SPLIT_MODE == "l2":
                mean = sum(counts) / len(counts)
                raw_split_val = math.sqrt(
                    sum((c - mean) ** 2 for c in counts) / len(counts)
                )
            elif SPLIT_MODE == "pairwise":
                raw_split_val = float(
                    max(abs(counts[i] - counts[j])
                        for i in range(len(counts))
                        for j in range(i + 1, len(counts)))
                )
            split += raw_split_val

        if ENABLE_DEBUG:
            supp = _rotation_support(rot)
            word = rot.axis.lstrip("+-")
            debug_rows.append(RotationDebug(
                ridx=ridx,
                layer=getattr(rot, "layer", None),
                word=word,
                support=tuple(sorted(supp)),
                touched_blocks=tuple(sorted(S)),
                k=k,
                support_size=len(supp),
                per_block_counts=dict(per_block_counts),
                raw_span=raw_span,
                raw_disconnected=raw_disconnected,
                raw_mst=raw_mst,
                raw_split=raw_split_val,
                span_pen=W_SPAN * raw_span,
                disconnected_pen=W_DISCONNECTED * raw_disconnected,
                mst_pen=W_MST * raw_mst,
                split_pen=W_SPLIT * raw_split_val,
                total_pen=(W_SPAN * raw_span + W_DISCONNECTED * raw_disconnected
                           + W_MST * raw_mst + W_SPLIT * raw_split_val),
            ))

    # load-balance penalties over active blocks only
    active_blocks = {hw.logical_to_block[q] for q in hw.logical_to_block}
    active_loads = [load[b] for b in active_blocks] if active_blocks else [0]
    peak = float(max(active_loads))
    load_range = float(max(active_loads) - min(active_loads)) if len(active_loads) > 1 else 0.0
    if len(active_loads) > 1:
        mean_load = sum(active_loads) / len(active_loads)
        std_load = math.sqrt(
            sum((x - mean_load) ** 2 for x in active_loads) / len(active_loads)
        )
    else:
        std_load = 0.0

    top_rotations: List[RotationDebug] = []
    if ENABLE_DEBUG and debug_rows:
        top_rotations = sorted(debug_rows, key=lambda r: r.total_pen, reverse=True)[:RETURN_TOP_K]

    total = (
        W_PEAK * peak
        + W_RANGE * load_range
        + W_STD * std_load
        + W_DISCONNECTED * disconn
        + W_MST * mst
        + W_SPLIT * split
        + W_SPAN * span
    )

    return ScoreBreakdown(
        total=total,
        peak_load_pen=W_PEAK * peak,
        range_load_pen=W_RANGE * load_range,
        std_load_pen=W_STD * std_load,
        span_pen=W_SPAN * span,
        disconnected_pen=W_DISCONNECTED * disconn,
        mst_pen=W_MST * mst,
        split_pen=W_SPLIT * split,
        block_loads=load,
        num_multiblock=num_multiblock,
        num_disconnected=num_disconnected,
        max_blocks_touched=max_blocks_touched,
        top_rotations=top_rotations,
    )


def print_score_debug(
    score: ScoreBreakdown,
    *,
    title: str = "SCORE DEBUG",
    show_top_k: int = 10,
) -> None:
    """Pretty-print a ScoreBreakdown. Pass ENABLE_DEBUG=True to _score to populate top_rotations."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"  total             : {score.total:.2f}")
    print(f"  peak_load_pen     : {score.peak_load_pen:.2f}")
    print(f"  range_load_pen    : {score.range_load_pen:.2f}")
    print(f"  std_load_pen      : {score.std_load_pen:.2f}")
    print(f"  disconnected_pen  : {score.disconnected_pen:.2f}")
    print(f"  mst_pen           : {score.mst_pen:.2f}")
    print(f"  split_pen         : {score.split_pen:.2f}")
    print(f"  span_pen          : {score.span_pen:.2f}")
    print(f"  num_multiblock    : {score.num_multiblock}")
    print(f"  num_disconnected  : {score.num_disconnected}")
    print(f"  max_blocks_touched: {score.max_blocks_touched}")
    if score.block_loads:
        print(f"\n  Block loads (rotation count per block):")
        for b, cnt in sorted(score.block_loads.items()):
            print(f"    block {b:>4}: {cnt}")
    if score.top_rotations:
        k = min(show_top_k, len(score.top_rotations))
        print(f"\n  Top-{k} rotations by penalty:")
        for i, r in enumerate(score.top_rotations[:k]):
            print(
                f"    [{i+1:2d}] ridx={r.ridx} layer={r.layer} k={r.k} "
                f"supp={r.support_size} blocks={r.touched_blocks} "
                f"pen={r.total_pen:.2f} "
                f"(disc={r.raw_disconnected:.0f} mst={r.raw_mst:.0f} split={r.raw_split:.2f})"
            )
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

    # check a sample of rotation supports and which blocks they hit
    multi_block = 0
    no_support = 0
    unmatched = 0
    for rot in rotations:
        supp = _rotation_support(rot)
        blocks = _blocks_touched(rot, hw)
        if not supp:
            no_support += 1
        elif not blocks:
            unmatched += 1  # support qubits not in hw.logical_to_block
        elif len(blocks) > 1:
            multi_block += 1

    print(f"  rotations w/ empty support   : {no_support}")
    print(f"  rotations w/ unmatched qubits: {unmatched}  <- nonzero = logical ID mismatch")
    print(f"  rotations spanning >1 block  : {multi_block}")

    # per-block utilisation table
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
    active_blocks = set(hw.logical_to_block.values())
    active_loads = [load[b] for b in active_blocks] if active_blocks else [0]
    print(f"  active-block load  max={max(active_loads)}  (balance metric)")

    # initial score
    s = _score(rotations, hw, ENABLE_DEBUG=True)
    print(f"  initial score  total={s.total:.1f}  peak={s.peak_load_pen:.1f}  "
          f"range={s.range_load_pen:.1f}  disconn={s.disconnected_pen:.1f}  "
          f"mst={s.mst_pen:.1f}  split={s.split_pen:.1f}")

    # check 10 random moves and their score deltas
    import random as _random
    rng = _random.Random(0)
    deltas = []
    for _ in range(20):
        move = _random_move(hw, rng)
        nxt = _score(rotations, hw, ENABLE_DEBUG=True)
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
    T0 = t0
    Tend = t_end

    # snapshot helper
    def _snapshot() -> Dict[LogicalId, Tuple[BlockId, LocalId]]:
        return {q: (hw.logical_to_block[q], hw.logical_to_local[q]) for q in hw.logical_to_block}

    best_map = _snapshot()
    cur = _score(rotations, hw)
    best = cur

    n_noop = 0
    n_accept = 0
    n_reject = 0

    for it in range(1, steps + 1):
        frac = (it - 1) / max(1, steps - 1)
        T = T0 * ((Tend / T0) ** frac)

        move = _random_move(hw, rng)

        if move[0] == "noop":
            n_noop += 1
            continue

        nxt = _score(rotations, hw)

        delta = nxt.total - cur.total
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(1e-12, T))

        if accept:
            n_accept += 1
            cur = nxt
            if cur.total < best.total:
                best = cur
                best_map = _snapshot()
        else:
            n_reject += 1
            _undo_move(hw, move)

        if verbose and report_every and (it % report_every == 0):
            print(
                f"[SA-mapping] it={it:6d}  T={T:.4f}  "
                f"cur={cur.total:.1f}  best={best.total:.1f}  "
                f"(peak={cur.peak_load_pen:.0f} range={cur.range_load_pen:.0f} "
                f"disconn={cur.disconnected_pen:.0f} split={cur.split_pen:.0f})  "
                f"accept={n_accept}  reject={n_reject}  noop={n_noop}"
            )
            n_noop = n_accept = n_reject = 0

    # restore best mapping into hw
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
    SA mapper that minimises the number of PauliRotations spanning multiple blocks.

    Compatible with the factory pattern — no constructor args required:

        mapper = get_mapper("simulated_annealing")
        plan   = mapper.solve(problem, hw, cfg, meta={"rotations": converter.program.rotations})

    meta keys:
        rotations  (required) : list[PauliRotation] from GoSCConverter.program.rotations
        verbose    (optional) : bool, print SA progress (default False)

    SA temperature schedule and step count are read from cfg:
        cfg.sa_steps, cfg.sa_t0, cfg.sa_tend, cfg.seed
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
        debug: bool = meta.get("debug", False)

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
            verbose=verbose,
        )

        # 3) Build MappingPlan from hw state (already at best)
        out_b = dict(hw.logical_to_block)
        out_l = dict(hw.logical_to_local)

        return MappingPlan(
            out_b,
            out_l,
            meta={
                "mapper": self.name,
                "init_mapper": self.init_mapper_name,
                "sa_steps": cfg.sa_steps,
                "sa_t0": cfg.sa_t0,
                "sa_t_end": cfg.sa_tend,
                "best_score_total": best_score.total,
                "best_score_peak_load": best_score.peak_load_pen,
                "best_score_range_load": best_score.range_load_pen,
                "best_score_disconnected": best_score.disconnected_pen,
                "best_score_mst": best_score.mst_pen,
                "best_score_split": best_score.split_pen,
                "best_score_span": best_score.span_pen,
                "num_multiblock": best_score.num_multiblock,
                "num_disconnected": best_score.num_disconnected,
                "n_rotations": len(rotations),
            },
        )
