# modqldpc/mapping/algos/sa_mapping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import BaseMapper
from ..types import MappingPlan, MappingProblem, MappingConfig, MappingCostFn, AnnealingConfig
from ..model import HardwareGraph


@dataclass(frozen=True)
class SimulatedAnnealingMapper(BaseMapper):
    """
    Placeholder: plug your SA implementation here.

    Contract:
      - start from an initial plan (e.g., auto_round_robin_mapping)
      - propose local moves (swap two logicals, move one logical to a free slot, etc.)
      - accept/reject by Metropolis using cost(plan)
      - return best plan
    """
    name: str = "simulated_annealing"

    # You can inject:
    init_mapper_name: str = "auto_round_robin_mapping"
    cost_fn: Optional[MappingCostFn] = None
    anneal: AnnealingConfig = AnnealingConfig()

    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan:
        if self.cost_fn is None:
            raise ValueError(
                "SimulatedAnnealingMapper requires a cost_fn. "
                "Provide one that scores MappingPlan given (problem, hw)."
            )

        # import here to avoid circular import
        from ..factory import get_mapper

        # 1) initial plan from another mapper
        init_mapper = get_mapper(self.init_mapper_name)
        init_plan = init_mapper.solve(problem, hw, cfg)

        # 2) call user-implemented annealer (not implemented here)
        # best_plan = self._anneal(init_plan, problem, hw, self.cost_fn, self.anneal)
        # return best_plan

        # For now, just return init_plan and tag it:
        return MappingPlan(
            init_plan.logical_to_block,
            init_plan.logical_to_local,
            meta={
                **init_plan.meta,
                "mapper": self.name,
                "sa_placeholder": True,
                "init_mapper": self.init_mapper_name,
            },
        )

    # def _anneal(...): pass  # you will implement
