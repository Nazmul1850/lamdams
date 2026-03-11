# modqldpc/scheduling/factory.py
from __future__ import annotations

from modqldpc.scheduling.algos.naive_events import NaiveEventScheduler
from modqldpc.scheduling.algos.random_ready_pack import RandomReadyPackScheduler
from modqldpc.scheduling.algos.sa_scheduling import SimulatedAnnealingScheduler
from modqldpc.scheduling.algos.sequential import SequentialScheduler

from .base import BaseScheduler
from .algos.random_ready import RandomReadyScheduler


def get_scheduler(name: str) -> BaseScheduler:
    reg = {
        "random_ready": RandomReadyScheduler,
        "random_ready_pack": RandomReadyPackScheduler,
        "naive_event": NaiveEventScheduler,
        "sequential_scheduler": SequentialScheduler,
        'sa_scheduler': SimulatedAnnealingScheduler,
    }
    if name not in reg:
        raise KeyError(f"Unknown scheduler '{name}'. Available: {sorted(reg)}")
    return reg[name]()  # instantiate