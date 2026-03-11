# modqldpc/mapping/helpers.py
from __future__ import annotations

from typing import Dict, List

from .model import HardwareGraph, BlockId, LogicalId, LocalId
from .types import MappingProblem, MappingConfig


def _block_capacity(hw: HardwareGraph, b: BlockId) -> int:
    return hw.blocks[b].num_logicals

def _total_capacity(hw: HardwareGraph, block_ids: List[BlockId]) -> int:
    return sum(_block_capacity(hw, b) for b in block_ids)

def _iter_blocks_sorted(hw: HardwareGraph) -> List[BlockId]:
    return sorted(hw.blocks.keys())

def _init_free_slots(hw: HardwareGraph, block_ids: List[BlockId]) -> Dict[BlockId, List[LocalId]]:
    return {b: list(range(_block_capacity(hw, b))) for b in block_ids}

def _take_slot(free: Dict[BlockId, List[LocalId]], b: BlockId) -> LocalId:
    if not free[b]:
        raise ValueError(f"No free slots left in block {b}")
    return free[b].pop(0)

def _apply_fixed(
    problem: MappingProblem,
    hw: HardwareGraph,
    free: Dict[BlockId, List[LocalId]],
    out_b: Dict[LogicalId, BlockId],
    out_l: Dict[LogicalId, LocalId],
) -> None:
    for q, (b, l) in problem.fixed.items():
        if b not in hw.blocks:
            raise ValueError(f"Fixed mapping references unknown block {b} for logical {q}")
        if l not in free[b]:
            raise ValueError(f"Fixed mapping slot (block={b}, local={l}) not available for logical {q}")
        out_b[q] = b
        out_l[q] = l
        free[b].remove(l)

def _ensure_capacity(problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> None:
    cap = _total_capacity(hw, _iter_blocks_sorted(hw))
    need = len(problem.ids())
    if cfg.strict_capacity and cap < need:
        raise ValueError(f"Insufficient capacity: need {need} logicals but hardware has {cap} slots.")

def _unmapped(problem: MappingProblem, out_b: Dict[LogicalId, BlockId]) -> List[LogicalId]:
    return [q for q in problem.ids() if q not in out_b]
