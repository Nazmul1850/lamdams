# modqldpc/lowering/policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Set

from ..core.types import PauliAxis
from .keys import KeyNamer
from .plans import LocalPauli, LocalMeasurePlan, InterblockPlan, RotationLoweringPlan
from ..mapping.model import HardwareGraph, BlockId, CouplerId


# -------------------------
# Helper: split Pauli by block (global logical ids)
# -------------------------

def split_pauli_by_block(
    pauli: PauliAxis,
    hw: HardwareGraph,
    *,
    logical_ids: Optional[List[int]] = None,
) -> Dict[BlockId, Dict[int, str]]:
    """
    Interpret pauli.tensor positions as global logical ids (0..n-1).
    Returns per-block ops: block -> {logical_id: 'X'/'Y'/'Z'} for non-I.
    """
    n = len(pauli.tensor)
    ids = logical_ids if logical_ids is not None else list(range(n))
    if len(ids) != n:
        raise ValueError("logical_ids length must match Pauli tensor length.")

    per_block: Dict[BlockId, Dict[int, str]] = {}
    for lid, ch in zip(ids, pauli.tensor):
        if ch == "I":
            continue
        b = hw.logical_to_block.get(lid)
        if b is None:
            raise KeyError(f"Logical {lid} not mapped in HardwareGraph.")
        per_block.setdefault(b, {})[lid] = ch
    return per_block


# -------------------------
# Strategy interfaces
# -------------------------

class MagicPlacementPolicy(Protocol):
    def choose_magic_block(self, *, blocks_involved: List[BlockId], hw: HardwareGraph) -> BlockId: ...


class RoutingPolicy(Protocol):
    def plan_interblock(
        self,
        *,
        blocks_involved: List[BlockId],
        magic_block: BlockId,
        hw: HardwareGraph,
    ) -> Optional[InterblockPlan]:
        ...


class NativeMeasurementPolicy(Protocol):
    """
    This is where you enforce:
      - which local Paulis are natively measurable on a block
      - how to decompose non-native measurements into native ones (+ combine rule)
      - whether gauge fix is required
    """
    def plan_local_measurement(
        self,
        *,
        target: LocalPauli,
        hw: HardwareGraph,
    ) -> LocalMeasurePlan: ...


# -------------------------
# Default minimal policies (safe starter)
# -------------------------

@dataclass(frozen=True)
class ChooseMagicBlockMinId(MagicPlacementPolicy):
    """
    Deterministic: choose smallest block id among involved.
    Replace later (nearest T-cache, Steiner hub, congestion aware, etc.).
    """
    def choose_magic_block(self, *, blocks_involved: List[BlockId], hw: HardwareGraph) -> BlockId:
        if not blocks_involved:
            raise ValueError("blocks_involved empty")
        return min(blocks_involved)


@dataclass(frozen=True)
class ShortestPathGatherRouting(RoutingPolicy):
    """
    Gather pivots to magic_block using shortest paths (one path per other block).
    This does not assume a specific topology; uses hw.shortest_path().
    """
    def plan_interblock(
        self,
        *,
        blocks_involved: List[BlockId],
        magic_block: BlockId,
        hw: HardwareGraph,
    ) -> Optional[InterblockPlan]:
        uniq = sorted(set(blocks_involved))
        if len(uniq) <= 1:
            return None

        route_paths: List[List[BlockId]] = []
        couplers: List[CouplerId] = []

        for b in uniq:
            if b == magic_block:
                continue
            path = hw.shortest_path(b, magic_block)
            if not path:
                raise ValueError(f"No route from block {b} to magic_block {magic_block}")
            route_paths.append(path)

            # derive couplers used along path
            for u, v in zip(path, path[1:]):
                cid = hw.coupler_id(u, v)
                if cid is None:
                    raise ValueError(f"Missing coupler on edge {u}-{v} in path {path}")
                couplers.append(cid)

        # de-dup couplers but keep deterministic order
        seen = set()
        couplers_dedup = []
        for c in couplers:
            if c not in seen:
                seen.add(c)
                couplers_dedup.append(c)

        return InterblockPlan(
            blocks_involved=uniq,
            magic_block=magic_block,
            route_paths=route_paths,
            couplers_used=couplers_dedup,
            meta={"routing": "shortest_path_gather"},
        )


@dataclass(frozen=True)
class NativeAllPaulisForNow(NativeMeasurementPolicy):
    """
    STARTER ONLY:
      - claims every local pauli is native.
    You will replace this with a block-native measurement set + synthesis rules.
    Keeping it as a policy means you won't touch the lowering pipeline.
    """
    def plan_local_measurement(self, *, target: LocalPauli, hw: HardwareGraph) -> LocalMeasurePlan:
        return LocalMeasurePlan(
            block=target.block,
            target=target,
            native=True,
            sequence=[dict(target.ops)],
            combine_rule={"type": "direct"},
            requires_gauge_fix=False,
        )
# cost_fn contract:
#   returns k >= 1 meaning "k native measurement primitives required"
NativeCostFn = Callable[[BlockId, Dict[int, str], HardwareGraph], int]


@dataclass(frozen=True)
class HeuristicRepeatNativePolicy(NativeMeasurementPolicy):
    """
    NativeMeasurementPolicy implementation:
      - ask cost_fn how many native measurements are needed for target.ops on this block
      - if k==1: treat as native (direct)
      - if k>1: treat as non-native; return sequence repeating the same primitive k times

    This is intentionally simplistic; it gives you a clean hook point for later
    replacement with a real synthesis policy.
    """
    cost_fn: NativeCostFn
    # Optional: attach explanation tags in combine_rule/meta
    tag: str = "heuristic_repeat"

    def plan_local_measurement(self, *, target: LocalPauli, hw: HardwareGraph) -> LocalMeasurePlan:
        b = target.block
        ops = dict(target.ops)
        k = int(self.cost_fn(b, ops, hw))
        if k <= 0:
            raise ValueError(f"cost_fn must return k>=1, got {k} for block {b} ops={ops}")

        native = (k == 1)

        # repeat exact same primitive k times (your requested rule)
        seq = [ops for _ in range(k)]

        combine_rule: Dict[str, Any] = {
            "type": "repeat_same_primitive",
            "k": k,
            "note": "Placeholder synthesis: repeat identical native primitive k times; "
                    "replace with real decomposition later.",
            "tag": self.tag,
        }

        return LocalMeasurePlan(
            block=b,
            target=target,
            native=native,
            sequence=seq,
            combine_rule=combine_rule,
            requires_gauge_fix=False,
            gauge_fix_meta={},
        )

# -------------------------
# Policy bundle
# -------------------------

@dataclass(frozen=True)
class LoweringPolicies:
    namer: KeyNamer
    magic: MagicPlacementPolicy
    routing: RoutingPolicy
    native: NativeMeasurementPolicy


# -------------------------
# Plan one rotation (no DAG emission yet)
# -------------------------

def plan_rotation_lowering(
    *,
    layer: int,
    ridx: int,
    axis: PauliAxis,
    angle: float,
    hw: HardwareGraph,
    policies: LoweringPolicies,
    logical_ids: Optional[List[int]] = None,
) -> RotationLoweringPlan:
    per_block_ops = split_pauli_by_block(axis, hw, logical_ids=logical_ids)
    blocks_involved = sorted(per_block_ops.keys())
    magic_block = policies.magic.choose_magic_block(blocks_involved=blocks_involved, hw=hw)
    magic_id = policies.namer.magic_id(layer, ridx, magic_block)

    local_plans: List[LocalMeasurePlan] = []
    for b in blocks_involved:
        lp = policies.native.plan_local_measurement(
            target=LocalPauli(block=b, ops=per_block_ops[b]),
            hw=hw,
        )
        local_plans.append(lp)

    interblock = policies.routing.plan_interblock(
        blocks_involved=blocks_involved,
        magic_block=magic_block,
        hw=hw,
    )

    return RotationLoweringPlan(
        layer=layer,
        ridx=ridx,
        axis=axis,
        angle=angle,
        blocks_involved=blocks_involved,
        magic_block=magic_block,
        magic_id=magic_id,
        local_plans=local_plans,
        interblock=interblock,
        out_bPZ=policies.namer.bPZ(layer, ridx),
        out_bXm=policies.namer.bXm(layer, ridx),
    )