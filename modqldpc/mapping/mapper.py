# modqldpc/mapping/mapper.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Protocol, Callable, Iterable
import random
import math

from .model import HardwareGraph, BlockId, LogicalId, LocalId


# ============================================================
# Public data models
# ============================================================

@dataclass(frozen=True)
class MappingPlan:
    """A concrete mapping: global logical -> (block, local)."""
    logical_to_block: Dict[LogicalId, BlockId]
    logical_to_local: Dict[LogicalId, LocalId]
    meta: Dict[str, object] = field(default_factory=dict)

    def loc(self, q: LogicalId) -> Tuple[BlockId, LocalId]:
        return self.logical_to_block[q], self.logical_to_local[q]


@dataclass(frozen=True)
class MappingProblem:
    """
    What we are mapping.
    Keep this minimal and extend later:
      - n_logicals: number of logical data qubits
      - logical_ids: explicit ordering (optional)
      - fixed: pre-assigned subset (optional)
    """
    n_logicals: int
    logical_ids: Optional[List[LogicalId]] = None
    fixed: Dict[LogicalId, Tuple[BlockId, LocalId]] = field(default_factory=dict)

    def ids(self) -> List[LogicalId]:
        return list(self.logical_ids) if self.logical_ids is not None else list(range(self.n_logicals))


@dataclass(frozen=True)
class MappingConfig:
    """
    Common knobs for mapping.
    """
    seed: int = 0
    # for semi-pack
    pack_fraction: float = 0.7              # [0,1], fraction mapped by packing before spreading
    # for random_pack
    shuffle_blocks: bool = True
    shuffle_logicals: bool = True
    # for (future) annealing
    sa_steps: int = 50_000
    sa_t0: float = 1.0
    sa_alpha: float = 0.9995
    # capacity handling: if True, raise if insufficient capacity
    strict_capacity: bool = True


# ============================================================
# Strategy pattern: mapping algorithms are interchangeable
# ============================================================

class Mapper(Protocol):
    name: str
    def solve(self, problem: MappingProblem, hw: HardwareGraph, cfg: MappingConfig) -> MappingPlan: ...


_MAPPER_REGISTRY: Dict[str, type] = {}


def register_mapper(mapper_cls):
    _MAPPER_REGISTRY[mapper_cls().name] = mapper_cls
    return mapper_cls

def get_mapper(name: str):
    return _MAPPER_REGISTRY[name]()


def list_mappers() -> List[str]:
    return sorted(_MAPPER_REGISTRY)


# ============================================================
# Internal helpers (capacity-aware placement)
# ============================================================

def _block_capacity(hw: HardwareGraph, b: BlockId) -> int:
    return hw.blocks[b].num_logicals

def _total_capacity(hw: HardwareGraph, block_ids: List[BlockId]) -> int:
    return sum(_block_capacity(hw, b) for b in block_ids)

def _iter_blocks_sorted(hw: HardwareGraph) -> List[BlockId]:
    return sorted(hw.blocks.keys())

def _init_free_slots(hw: HardwareGraph, block_ids: List[BlockId]) -> Dict[BlockId, List[LocalId]]:
    # free locals are 0..num_logicals-1
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


# ============================================================
# Common mapping algorithms
# ============================================================

@register_mapper
@dataclass(frozen=True)
class AutoRoundRobinMapper:
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
        out_b: Dict[LogicalId, BlockId] = {}
        out_l: Dict[LogicalId, LocalId] = {}

        _apply_fixed(problem, hw, free, out_b, out_l)

        ids = _unmapped(problem, out_b)
        if cfg.shuffle_logicals:
            rng.shuffle(ids)

        bi = 0
        for q in ids:
            # find next block with space
            for _ in range(len(blocks)):
                b = blocks[bi % len(blocks)]
                bi += 1
                if free[b]:
                    out_b[q] = b
                    out_l[q] = _take_slot(free, b)
                    break
            else:
                # should not happen if capacity ensured
                raise ValueError("No available slot found during round-robin placement.")

        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})


@register_mapper
@dataclass(frozen=True)
class AutoPackMapper:
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
        out_b: Dict[LogicalId, BlockId] = {}
        out_l: Dict[LogicalId, LocalId] = {}

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

        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})


@register_mapper
@dataclass(frozen=True)
class RandomPackMapper:
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
        out_b: Dict[LogicalId, BlockId] = {}
        out_l: Dict[LogicalId, LocalId] = {}

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

        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed})


@register_mapper
@dataclass(frozen=True)
class RandomSemiPackMapper:
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
        out_b: Dict[LogicalId, BlockId] = {}
        out_l: Dict[LogicalId, LocalId] = {}

        _apply_fixed(problem, hw, free, out_b, out_l)

        ids = _unmapped(problem, out_b)
        if cfg.shuffle_logicals:
            rng.shuffle(ids)

        n = len(ids)
        k = int(round(max(0.0, min(1.0, cfg.pack_fraction)) * n))
        pack_ids = ids[:k]
        rest_ids = ids[k:]

        # choose a small prefix of blocks for packing: enough capacity to hold k logicals
        prefix: List[BlockId] = []
        cap = 0
        for b in blocks_all:
            prefix.append(b)
            cap += len(free[b])
            if cap >= k:
                break
        if cap < k:
            # should not happen if capacity ensured, but fixed mappings can reduce available
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

        return MappingPlan(out_b, out_l, meta={"mapper": self.name, "seed": cfg.seed, "pack_fraction": cfg.pack_fraction})


# ============================================================
# Simulated annealing placeholder (structure only)
# ============================================================

class MappingCostFn(Protocol):
    def __call__(self, plan: MappingPlan, problem: MappingProblem, hw: HardwareGraph) -> float: ...


@dataclass(frozen=True)
class AnnealingConfig:
    steps: int = 50_000
    t0: float = 1.0
    alpha: float = 0.9995
    seed: int = 0


@register_mapper
@dataclass(frozen=True)
class SimulatedAnnealingMapper:
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