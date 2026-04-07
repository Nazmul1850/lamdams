# modqldpc/mapping/factory.py
from __future__ import annotations

from typing import List

from .base import BaseMapper
from .algos.auto_round_robin import AutoRoundRobinMapper
from .algos.auto_pack import AutoPackMapper
from .algos.random_pack import RandomPackMapper
from .algos.random_semi_pack import RandomSemiPackMapper
from .algos.sa_mapping import SimulatedAnnealingMapper
from .algos.sa_v2 import SimulatedAnnealingV2Mapper
from .algos.pure_random import PureRandomMapper


def get_mapper(name: str) -> BaseMapper:
    reg = {
        "auto_round_robin_mapping": AutoRoundRobinMapper,
        "auto_pack": AutoPackMapper,
        "random_pack_mapping": RandomPackMapper,
        "random_semi_pack": RandomSemiPackMapper,
        "simulated_annealing": SimulatedAnnealingMapper,
        "sa_v2": SimulatedAnnealingV2Mapper,
        "simulated_annealing_v2": SimulatedAnnealingV2Mapper,
        "pure_random": PureRandomMapper,
    }
    if name not in reg:
        raise KeyError(f"Unknown mapper '{name}'. Available: {sorted(reg)}")
    return reg[name]()


def list_mappers() -> List[str]:
    return [
        "auto_round_robin_mapping",
        "auto_pack",
        "random_pack_mapping",
        "random_semi_pack",
        "simulated_annealing",
        "sa_v2",
        "simulated_annealing_v2",
        "pure_random",
    ]
