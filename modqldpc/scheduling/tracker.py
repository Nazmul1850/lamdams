# modqldpc/scheduling/tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .types import StepResourceState
from .policy import ResourceConsumptionPolicy
from ..mapping.model import HardwareGraph
from ..lowering.ir import ExecNode


@dataclass
class HardwareTracker:
    """
    Interval reservation tracker:
      - resource constraints are enforced for each integer time t in [start,end).
      - maintains step_states[t] = StepResourceState after applying all reservations active at t.
    """
    hw: HardwareGraph
    policy: ResourceConsumptionPolicy
    step_states: Dict[int, StepResourceState] = field(default_factory=dict)

    def _state_at(self, t: int) -> StepResourceState:
        if t not in self.step_states:
            self.step_states[t] = StepResourceState()
        return self.step_states[t]

    def can_reserve(self, node: ExecNode, start: int, end: int) -> bool:
        claim = self.policy.claim_for_node(node, self.hw)
        for t in range(int(start), int(end)):
            st = self._state_at(t)
            if not self.policy.can_apply(st, claim, self.hw):
                return False
        return True

    def reserve(self, node: ExecNode, start: int, end: int) -> None:
        claim = self.policy.claim_for_node(node, self.hw)
        for t in range(int(start), int(end)):
            st = self._state_at(t)
            # must be valid if caller used can_reserve; still safe to assert
            if not self.policy.can_apply(st, claim, self.hw):
                raise ValueError(f"reserve() failed at t={t} for node={node.nid} under policy={self.policy.name}")
            self.policy.apply(st, claim, self.hw)