# modqldpc/mapping/algos/auto_round_robin.py
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
class AutoRoundRobinMapper(BaseMapper):
    """
    Spread logicals across blocks in a round-robin manner, respecting per-block capacity.
    This tends to reduce concentration / hotspots.
    """
    name: str = "auto_round_robin_mapping"

    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan:
        _ensure_capacity(problem, hw, cfg)
        rng = random.Random(cfg.seed)

        blocks = _iter_blocks_sorted(hw)
        if cfg.shuffle_blocks:
            blocks = blocks[:]
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
            for _ in range(len(blocks)):
                b = blocks[bi % len(blocks)]
                bi += 1
                if free[b]:
                    out_b[q] = b
                    out_l[q] = _take_slot(free, b)
                    break
            else:
                raise ValueError("No available slot found during round-robin placement.")
        hw.update_plan(out_b, out_l)
        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})
