# modqldpc/mapping/mapper.py
# Backward-compatibility shim — import from the canonical locations instead.
from .types import MappingPlan, MappingProblem, MappingConfig, MappingCostFn
from .base import BaseMapper
from .factory import get_mapper, list_mappers
from .algos.auto_round_robin import AutoRoundRobinMapper
from .algos.auto_pack import AutoPackMapper
from .algos.random_pack import RandomPackMapper
from .algos.random_semi_pack import RandomSemiPackMapper
from .algos.sa_mapping import SimulatedAnnealingMapper

__all__ = [
    "MappingPlan", "MappingProblem", "MappingConfig", "MappingCostFn",
    "BaseMapper",
    "get_mapper", "list_mappers",
    "AutoRoundRobinMapper", "AutoPackMapper", "RandomPackMapper",
    "RandomSemiPackMapper", "SimulatedAnnealingMapper",
]
