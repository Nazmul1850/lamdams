# modqldpc/scheduling/policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Set

from .types import ResourceClaim, StepResourceState, BlockId, CouplerId
from ..mapping.model import HardwareGraph
from ..lowering.ir import ExecNode


class ResourceConsumptionPolicy(Protocol):
    name: str
    def claim_for_node(self, node: ExecNode, hw: HardwareGraph) -> ResourceClaim: ...
    def can_apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> bool: ...
    def apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> None: ...


def incident_blocks_of_couplers(hw: HardwareGraph, couplers: Set[CouplerId]) -> Set[BlockId]:
    blocks: Set[BlockId] = set()
    for cid in couplers:
        c = hw.couplers[cid]
        blocks.add(c.u)
        blocks.add(c.v)
    return blocks


@dataclass(frozen=True)
class SimplePortsAndCouplersPolicy:
    """
    Baseline:
      - node.blocks consumes block port
      - node.couplers consumes coupler capacity
      - coupler does NOT consume block port
      - local ops can run even if incident coupler is used (if ports allow)
    """
    name: str = "simple"

    def claim_for_node(self, node: ExecNode, hw: HardwareGraph) -> ResourceClaim:
        return ResourceClaim(blocks_touched=set(node.blocks), couplers_used=set(node.couplers))

    def can_apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> bool:
        for b in claim.blocks_touched:
            used = state.block_ports_used.get(b, 0)
            cap = hw.port_capacity.get(b, 1)
            if used + 1 > cap:
                return False

        for cid in claim.couplers_used:
            used = state.coupler_used.get(cid, 0)
            cap = hw.couplers[cid].capacity
            if used + 1 > cap:
                return False

        return True

    def apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> None:
        for b in claim.blocks_touched:
            state.block_ports_used[b] = state.block_ports_used.get(b, 0) + 1
        for cid in claim.couplers_used:
            state.coupler_used[cid] = state.coupler_used.get(cid, 0) + 1


@dataclass(frozen=True)
class IncidentCouplerBlocksLocalOpsPolicy:
    """
    Your current semantics:
      1) Using a coupler consumes endpoint block ports (and coupler capacity).
      2) If any incident coupler is used for a block in this step, that block cannot
         run any other node touching that block in this step.
    """
    name: str = "incident_coupler_blocks_local"

    def claim_for_node(self, node: ExecNode, hw: HardwareGraph) -> ResourceClaim:
        return ResourceClaim(blocks_touched=set(node.blocks), couplers_used=set(node.couplers))

    def can_apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> bool:
        # disallow local ops if block already incident-coupler-busy
        for b in claim.blocks_touched:
            if b in state.incident_coupler_busy:
                return False

        incident_blocks = incident_blocks_of_couplers(hw, claim.couplers_used)

        # if we will use a coupler incident to block b, block cannot already be used this step
        for b in incident_blocks:
            if state.block_ports_used.get(b, 0) > 0:
                return False

        # check block port capacity for union(blocks_touched, incident_blocks)
        all_blocks = set(claim.blocks_touched) | incident_blocks
        for b in all_blocks:
            used = state.block_ports_used.get(b, 0)
            cap = hw.port_capacity.get(b, 1)
            if used + 1 > cap:
                return False

        # check coupler capacity
        for cid in claim.couplers_used:
            used = state.coupler_used.get(cid, 0)
            cap = hw.couplers[cid].capacity
            if used + 1 > cap:
                return False

        return True

    def apply(self, state: StepResourceState, claim: ResourceClaim, hw: HardwareGraph) -> None:
        incident_blocks = incident_blocks_of_couplers(hw, claim.couplers_used)

        for b in incident_blocks:
            state.incident_coupler_busy.add(b)

        all_blocks = set(claim.blocks_touched) | incident_blocks
        for b in all_blocks:
            state.block_ports_used[b] = state.block_ports_used.get(b, 0) + 1

        for cid in claim.couplers_used:
            state.coupler_used[cid] = state.coupler_used.get(cid, 0) + 1


def get_resource_policy(name: str) -> ResourceConsumptionPolicy:
    reg = {
        "simple": SimplePortsAndCouplersPolicy,
        "incident_coupler_blocks_local": IncidentCouplerBlocksLocalOpsPolicy,
    }
    if name not in reg:
        raise KeyError(f"Unknown resource policy '{name}'. Available: {sorted(reg)}")
    return reg[name]()