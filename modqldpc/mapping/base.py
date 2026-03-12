# modqldpc/mapping/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .types import MappingPlan, MappingProblem, MappingConfig
from .model import HardwareGraph


class BaseMapper(ABC):
    name: str = "base"

    @abstractmethod
    def solve(
        self,
        problem: MappingProblem,
        hw: HardwareGraph,
        cfg: MappingConfig,
        meta: Optional[dict] = None,
    ) -> MappingPlan:
        raise NotImplementedError
