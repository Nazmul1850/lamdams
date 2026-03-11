# modqldpc/mapping/algos/random_semi_pack.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from ..base import BaseMapper
from ..types import MappingPlan, MappingProblem, MappingConfig
from ..model import HardwareGraph, BlockId
from ..helpers import (
    _ensure_capacity, _iter_blocks_sorted, _init_free_slots,
    _apply_fixed, _unmapped, _take_slot,
)


@dataclass(frozen=True)
class RandomSemiPackMapper(BaseMapper):
    """
    Hybrid:
      - Pack first `pack_fraction` of logicals into a small prefix of blocks (randomized)
      - Then round-robin the remaining logicals over all blocks (randomized)
    This is useful to create controlled locality without full concentration.
    """
    name: str = "random_semi_pack"

    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan:
        _ensure_capacity(problem, hw, cfg)
        rng = random.Random(cfg.seed)

        blocks_all = _iter_blocks_sorted(hw)
        if cfg.shuffle_blocks:
            blocks_all = blocks_all[:]
            rng.shuffle(blocks_all)

        free = _init_free_slots(hw, blocks_all)
        out_b = {}
        out_l = {}

        _apply_fixed(problem, hw, free, out_b, out_l)

        ids = _unmapped(problem, out_b)
        if cfg.shuffle_logicals:
            rng.shuffle(ids)

        n = len(ids)
        k = int(round(max(0.0, min(1.0, cfg.pack_fraction)) * n))
        pack_ids = ids[:k]
        rest_ids = ids[k:]

        # choose a small prefix of blocks with enough capacity to hold k logicals
        prefix: List[BlockId] = []
        cap = 0
        for b in blocks_all:
            prefix.append(b)
            cap += len(free[b])
            if cap >= k:
                break
        if cap < k:
            raise ValueError("Not enough free capacity to satisfy pack_fraction after fixed assignments.")

        # 1) pack into prefix blocks
        bi = 0
        for q in pack_ids:
            while bi < len(prefix) and not free[prefix[bi]]:
                bi += 1
            if bi >= len(prefix):
                raise ValueError("Unexpected: ran out of slots in prefix during semi-pack.")
            b = prefix[bi]
            out_b[q] = b
            out_l[q] = _take_slot(free, b)

        # 2) spread remaining round-robin over all blocks
        bi = 0
        for q in rest_ids:
            for _ in range(len(blocks_all)):
                b = blocks_all[bi % len(blocks_all)]
                bi += 1
                if free[b]:
                    out_b[q] = b
                    out_l[q] = _take_slot(free, b)
                    break
            else:
                raise ValueError("No available slot found during semi-pack spreading.")
        hw.update_plan(out_b, out_l)
        return MappingPlan(
            out_b, out_l,
            meta={"mapper": self.name, "seed": cfg.seed, "pack_fraction": cfg.pack_fraction},
        )
