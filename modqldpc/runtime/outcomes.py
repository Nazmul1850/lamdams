# modqldpc/runtime/outcomes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict
import random

from ..lowering.ir import ClassicalKey 


class OutcomeModel(Protocol):
    def sample_bit(self, key: ClassicalKey) -> int: ...


@dataclass
class RandomOutcomeModel:
    seed: int = 0
    _rng: random.Random = None

    def __post_init__(self):
        object.__setattr__(self, "_rng", random.Random(self.seed))

    def sample_bit(self, key: ClassicalKey) -> int:
        # bit in {0,1}
        return int(self._rng.randint(0, 1))


@dataclass
class ReplayOutcomeModel:
    """
    Deterministic playback from stored bits.
    """
    bits: Dict[str, int]

    def sample_bit(self, key: ClassicalKey) -> int:
        if key.name not in self.bits:
            raise KeyError(f"Missing replay bit for key '{key.name}'")
        return int(self.bits[key.name])