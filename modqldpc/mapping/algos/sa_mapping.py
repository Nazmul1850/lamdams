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
    rot.axis is a Qiskit Pauli; .to_label() returns a string like 'IXYZ'
    where the RIGHTMOST character is qubit 0.
    """
    word = rot.axis.to_label()
    supp: Set[LogicalId] = set()
    for k, ch in enumerate(reversed(word)):
        if ch not in ("I", "-", "+"):
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
class ScoreBreakdown:
    total: float
    span_pen: float
    disconnected_pen: float
    mst_pen: float
    intra_pen: float
    balance_pen: float


def _score(
    rotations: list,
    hw: HardwareGraph,
    *,
    W_SPAN: float = 1e4,
    W_DISCONNECTED: float = 0.0,
    W_MST: float = 1e2,
    W_INTRA: float = 1e3,
    W_BALANCE: float = 0.0,
) -> ScoreBreakdown:
    """
    Energy to minimise.  For each PauliRotation, find S = set of blocks its
    support touches under the current mapping:

      - span penalty:        (|S| - 1) per rotation   [0 when single-block]
                             PRIMARY: reduce cross-block rotations.
      - disconnected penalty: 1 per disconnected multi-block span
                             HARD CONSTRAINT: disconnected is always worst.
      - mst penalty:         MST hop-length over S per multi-block rotation
                             QUATERNARY: compactness when multi-block unavoidable.
      - intra penalty:       sum over multi-block rotations of
                             (max_block_support - min_block_support),
                             where block_support[b] = #qubits of this rotation in b.
                             SECONDARY: for unavoidable multi-block rotations,
                             prefer equal qubit split across participating blocks
                             so no block idles while the other finishes.
      - balance penalty:     max(load) over ACTIVE blocks only
                             where load[b] = #rotations touching block b.
                             TERTIARY: prevent one block becoming the global bottleneck.

    Weight hierarchy: W_DISCONNECTED >> W_SPAN >> W_INTRA >> W_BALANCE >= W_MST
    """
    span = 0.0
    disconn = 0.0
    mst = 0.0
    intra = 0.0

    # load[b] = number of rotations whose support touches block b
    load: Dict[BlockId, int] = {b: 0 for b in hw.blocks}

    for rot in rotations:
        S = _blocks_touched(rot, hw)
        k = len(S)
        for b in S:
            load[b] += 1
        if k > 1:
            span += k - 1
            if not _is_connected(S, hw):
                disconn += 1.0
            mst += _mst_len(S, hw)

            # intra: count qubits of this rotation per participating block
            per_block: Dict[BlockId, int] = {b: 0 for b in S}
            for q in _rotation_support(rot):
                b = hw.logical_to_block.get(q)
                if b in per_block:
                    per_block[b] += 1
            counts = list(per_block.values())
            intra += max(counts) - min(counts)

    # balance: max load over ACTIVE blocks (those with ≥1 logical mapped)
    # Using max(load) not range: empty blocks don't inflate the penalty,
    # so concentrating logicals into fewer blocks is not artificially penalised.
    active_blocks = {hw.logical_to_block[q] for q in hw.logical_to_block}
    active_loads = [load[b] for b in active_blocks] if active_blocks else [0]
    balance = float(max(active_loads))

    return ScoreBreakdown(
        total=W_SPAN * span + W_DISCONNECTED * disconn + W_MST * mst + W_INTRA * intra + W_BALANCE * balance,
        span_pen=W_SPAN * span,
        disconnected_pen=W_DISCONNECTED * disconn,
        mst_pen=W_MST * mst,
        intra_pen=W_INTRA * intra,
        balance_pen=W_BALANCE * balance,
    )


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
    s = _score(rotations, hw)
    print(f"  initial score  total={s.total:.1f}  span={s.span_pen:.1f}  "
          f"disconn={s.disconnected_pen:.1f}  mst={s.mst_pen:.1f}  "
          f"intra={s.intra_pen:.1f}  balance={s.balance_pen:.1f}")

    # check 10 random moves and their score deltas
    import random as _random
    rng = _random.Random(0)
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
                f"(span={cur.span_pen:.0f} intra={cur.intra_pen:.0f} bal={cur.balance_pen:.0f})  "
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
                "best_score_span": best_score.span_pen,
                "best_score_disconnected": best_score.disconnected_pen,
                "best_score_mst": best_score.mst_pen,
                "best_score_intra": best_score.intra_pen,
                "best_score_balance": best_score.balance_pen,
                "n_rotations": len(rotations),
            },
        )
