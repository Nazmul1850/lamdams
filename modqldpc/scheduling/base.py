# modqldpc/scheduling/base.py
from __future__ import annotations
from abc import ABC, abstractmethod

from .types import SchedulingProblem, Schedule


class BaseScheduler(ABC):
    name: str = "base"

    @abstractmethod
    def solve(self, problem: SchedulingProblem) -> Schedule:
        raise NotImplementedError