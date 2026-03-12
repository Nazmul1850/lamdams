# modqldpc/mapping/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Protocol

from .model import BlockId, LogicalId, LocalId, HardwareGraph


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
    # for SA mapping
    # T0 must be large enough to accept the worst typical move early on.
    # Probe moves showed max delta ~3M; T0=1e5 gives ~5% acceptance of those.
    sa_steps: int = 50_000
    sa_t0: float = 1e5
    sa_tend: float = 0.05
    # capacity handling: if True, raise if insufficient capacity
    strict_capacity: bool = True


class MappingCostFn(Protocol):
    def __call__(self, plan: MappingPlan, problem: MappingProblem, hw: HardwareGraph) -> float: ...
