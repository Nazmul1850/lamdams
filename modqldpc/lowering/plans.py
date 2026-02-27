# modqldpc/lowering/plans.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ..core.types import PauliAxis
from ..mapping.model import BlockId, CouplerId


@dataclass(frozen=True)
class LocalPauli:
    """
    Local restriction of a global PauliString to a single block.
    ops: mapping from *global logical id* -> 'X'/'Y'/'Z' within that block.
    (Later you can switch to block-local indices if you want.)
    """
    block: BlockId
    ops: Dict[int, str]


@dataclass(frozen=True)
class LocalMeasurePlan:
    """
    How to realize a desired local Pauli on one block, given native constraints.

    - native=True: can measure directly with one primitive
    - native=False: provide a decomposition into native measurement strings and a classical combine rule
    """
    block: BlockId
    target: LocalPauli
    native: bool
    # A sequence of native measurement primitives to run (each as ops dict)
    sequence: List[Dict[int, str]] = field(default_factory=list)
    # How to combine outcomes into the target parity (e.g., xor of subset, sign flips)
    combine_rule: Dict[str, Any] = field(default_factory=dict)
    # optional: requires a gauge/basis fix step
    requires_gauge_fix: bool = False
    gauge_fix_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterblockPlan:
    """
    How to connect multiple blocks for a joint parity measurement.
    Start simple: single shortest path tree that gathers to magic_block.
    """
    blocks_involved: List[BlockId]
    magic_block: BlockId
    route_paths: List[List[BlockId]]          # e.g., list of paths to gather pivots
    couplers_used: List[CouplerId]            # flattened list (optional; can be derived)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RotationLoweringPlan:
    """
    Full plan for lowering one π/8 rotation.
    """
    layer: int
    ridx: int
    axis: PauliAxis
    angle: float

    blocks_involved: List[BlockId]
    magic_block: BlockId
    magic_id: str

    local_plans: List[LocalMeasurePlan]
    interblock: Optional[InterblockPlan]

    out_bPZ: str
    out_bXm: str