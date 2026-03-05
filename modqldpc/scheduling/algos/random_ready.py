# modqldpc/scheduling/algos/random_ready.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set
import random

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep


@dataclass
class RandomReadyScheduler(BaseScheduler):
    name: str = "random_ready"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        rng = random.Random(problem.seed)

        # indegree for remaining nodes
        indeg: Dict[str, int] = {}
        for nid in dag.nodes:
            indeg[nid] = len(dag.pred.get(nid, set()))

        ready: List[str] = sorted([nid for nid, d in indeg.items() if d == 0])
        remaining: Set[str] = set(dag.nodes.keys())

        steps: List[ScheduleStep] = []
        node_to_time: Dict[str, int] = {}

        t = 0
        while remaining:
            if not ready:
                raise ValueError("DAG has a cycle or no ready nodes (unexpected).")

            pick = rng.choice(ready)

            # schedule single node in its own step
            steps.append(ScheduleStep(t=t, nodes=[pick], meta={"algo": self.name}))
            node_to_time[pick] = t

            # remove it
            remaining.remove(pick)
            ready.remove(pick)

            for v in dag.succ.get(pick, set()):
                indeg[v] -= 1
                if indeg[v] == 0 and v in remaining:
                    ready.append(v)

            ready.sort()
            t += 1

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={"scheduler": self.name, "seed": problem.seed},
        )