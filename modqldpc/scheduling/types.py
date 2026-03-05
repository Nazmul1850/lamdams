# modqldpc/scheduling/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..lowering.ir import ExecDAG  # adjust path
from ..mapping.model import HardwareGraph


BlockId = int
CouplerId = str
NodeId = str

@dataclass(frozen=True)
class ScheduleEntry:
    nid: str
    start: int
    end: int

@dataclass(frozen=True)
class ResourceClaim:
    """
    What a node wants to consume THIS step (abstract).
    Policy decides how this translates into conflicts.
    """
    blocks_touched: Set[BlockId] = field(default_factory=set)
    couplers_used: Set[CouplerId] = field(default_factory=set)

@dataclass
class StepResourceState:
    """
    Resources already consumed in the current time step.
    """
    block_ports_used: Dict[BlockId, int] = field(default_factory=dict)
    coupler_used: Dict[CouplerId, int] = field(default_factory=dict)

    # optional: track "incident coupler occupied per block"
    incident_coupler_busy: Set[BlockId] = field(default_factory=set)


@dataclass(frozen=True)
class SchedulingProblem:
    dag: ExecDAG
    hw: HardwareGraph
    seed: int = 0
    policy_name: str = "incident_coupler_blocks_local"
    # optional: override per-node durations
    duration_override: Dict[NodeId, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduleStep:
    t: int
    nodes: List[NodeId]              # nodes executed in this step (parallel set)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Schedule:
    steps: List[ScheduleStep]
    # optional convenience maps
    node_to_time: Dict[NodeId, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def depth(self) -> int:
        return 0 if not self.steps else (max(s.t for s in self.steps) + 1)