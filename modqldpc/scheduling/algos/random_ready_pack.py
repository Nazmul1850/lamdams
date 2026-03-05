# modqldpc/scheduling/algos/random_ready_pack.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set
import random

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep
from ..resources import new_step_state
from ..policy import get_resource_policy


@dataclass
class RandomReadyPackScheduler(BaseScheduler):
    """
    Greedy parallel scheduler:
      - at each time step, consider currently-ready nodes
      - shuffle them
      - add as many as possible subject to resource policy
      - commit step, remove scheduled nodes, update indegrees
    """
    name: str = "random_ready_pack"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        hw = problem.hw
        rng = random.Random(problem.seed)
        pol = get_resource_policy(problem.policy_name)

        indeg: Dict[str, int] = {nid: len(dag.pred.get(nid, set())) for nid in dag.nodes}
        remaining: Set[str] = set(dag.nodes.keys())
        ready: List[str] = [nid for nid, d in indeg.items() if d == 0]
        ready.sort()

        steps: List[ScheduleStep] = []
        node_to_time: Dict[str, int] = {}

        t = 0
        while remaining:
            if not ready:
                raise ValueError("No ready nodes: DAG cycle or inconsistent indegree bookkeeping.")

            state = new_step_state()
            candidates = ready[:]
            rng.shuffle(candidates)

            chosen: List[str] = []
            for nid in candidates:
                if nid not in remaining:
                    continue
                node = dag.nodes[nid]
                claim = pol.claim_for_node(node, hw)
                if pol.can_apply(state, claim, hw):
                    pol.apply(state, claim, hw)
                    chosen.append(nid)

            # safety: must schedule at least one node each step
            if not chosen:
                # fallback: schedule one ready node ignoring resource packing order;
                # but still must satisfy can_apply on empty state
                nid = ready[0]
                node = dag.nodes[nid]
                claim = pol.claim_for_node(node, hw)
                if not pol.can_apply(state, claim, hw):
                    raise ValueError(
                        f"Resource policy '{pol.name}' rejects even a single ready node '{nid}'. "
                        f"Check node resource annotation (blocks/couplers) or capacities."
                    )
                pol.apply(state, claim, hw)
                chosen = [nid]

            # commit step
            steps.append(ScheduleStep(t=t, nodes=sorted(chosen), meta={"algo": self.name, "policy": pol.name}))
            for nid in chosen:
                node_to_time[nid] = t

            # remove chosen from graph and update indegrees
            for u in chosen:
                remaining.remove(u)
                if u in ready:
                    ready.remove(u)
                for v in dag.succ.get(u, set()):
                    indeg[v] -= 1
                    if indeg[v] == 0 and v in remaining and v not in ready:
                        ready.append(v)

            ready.sort()
            t += 1

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={"scheduler": self.name, "seed": problem.seed, "resource_policy": pol.name},
        )