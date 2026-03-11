# modqldpc/mapping/algos/random_pack.py
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
class RandomPackMapper(BaseMapper):
    """
    Pack logicals into randomly ordered blocks.
    (Still fills each chosen block before moving to next.)
    """
    name: str = "random_pack_mapping"

    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan:
        _ensure_capacity(problem, hw, cfg)
        rng = random.Random(cfg.seed)

        blocks = _iter_blocks_sorted(hw)
        if cfg.shuffle_blocks:
            rng.shuffle(blocks)

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
                raise ValueError("No available slots left during random packing.")
            b = blocks[bi]
            out_b[q] = b
            out_l[q] = _take_slot(free, b)
        hw.update_plan(out_b, out_l)
        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})
