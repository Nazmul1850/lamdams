# modqldpc/mapping/algos/pure_random.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..base import BaseMapper
from ..types import MappingPlan, MappingProblem, MappingConfig
from ..model import HardwareGraph
from ..helpers import (
    _ensure_capacity, _iter_blocks_sorted, _init_free_slots,
    _apply_fixed, _unmapped,
)


@dataclass(frozen=True)
class PureRandomMapper(BaseMapper):
    """
    Completely random mapping: flattens all available (block, slot) pairs,
    shuffles them uniformly, and assigns logicals with no locality bias.

    Unlike RandomPackMapper (which fills one block before moving to the next),
    this gives each logical an independent, uniformly random slot across
    the entire hardware graph.
    """
    name: str = "pure_random"

    def solve(
        self,
        problem: MappingProblem,
        hw: HardwareGraph,
        cfg: MappingConfig,
        meta: Optional[dict] = None,
    ) -> MappingPlan:
        _ensure_capacity(problem, hw, cfg)
        rng = random.Random(cfg.seed)

        blocks = _iter_blocks_sorted(hw)
        free = _init_free_slots(hw, blocks)
        out_b = {}
        out_l = {}

        _apply_fixed(problem, hw, free, out_b, out_l)

        # flatten all remaining (block, local_slot) pairs and shuffle uniformly
        all_slots = [(b, slot) for b in blocks for slot in free[b]]
        rng.shuffle(all_slots)

        ids = _unmapped(problem, out_b)
        rng.shuffle(ids)

        if len(ids) > len(all_slots):
            raise ValueError(
                f"PureRandomMapper: need {len(ids)} slots but only {len(all_slots)} available."
            )

        for q, (b, local) in zip(ids, all_slots):
            out_b[q] = b
            out_l[q] = local

        hw.update_plan(out_b, out_l)
        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})
