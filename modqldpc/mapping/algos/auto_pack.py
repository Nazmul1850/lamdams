# modqldpc/mapping/algos/auto_pack.py
from __future__ import annotations

import random
from dataclasses import dataclass

from ..base import BaseMapper
from ..types import MappingPlan, MappingProblem, MappingConfig
from ..model import HardwareGraph
from ..helpers import (
    _ensure_capacity, _iter_blocks_sorted, _init_free_slots,
    _apply_fixed, _unmapped, _take_slot,
)


@dataclass(frozen=True)
class AutoPackMapper(BaseMapper):
    """
    Pack logicals into blocks in sorted block order (fill block 0, then block 1, ...).
    This maximizes locality but can stress couplers/ports depending on workload.
    """
    name: str = "auto_pack"

    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan:
        _ensure_capacity(problem, hw, cfg)
        rng = random.Random(cfg.seed)

        blocks = _iter_blocks_sorted(hw)
        free = _init_free_slots(hw, blocks)
        out_b = {}
        out_l = {}

        _apply_fixed(problem, hw, free, out_b, out_l)

        ids = _unmapped(problem, out_b)
        if cfg.shuffle_logicals:
            rng.shuffle(ids)

        bi = 0
        for q in ids:
            while bi < len(blocks) and not free[blocks[bi]]:
                bi += 1
            if bi >= len(blocks):
                raise ValueError("No available slots left during packing.")
            b = blocks[bi]
            out_b[q] = b
            out_l[q] = _take_slot(free, b)
        hw.update_plan(out_b, out_l)
        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})
