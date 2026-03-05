# modqldpc/scheduling/algos/naive_event.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import heapq

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep, ScheduleEntry
from ..policy import get_resource_policy
from ..tracker import HardwareTracker


@dataclass
class NaiveEventScheduler(BaseScheduler):
    """
    Event-based greedy scheduler with durations (correct baseline).

    - ready set: indegree_left==0 and not scheduled
    - at time t, try to start as many ready nodes as possible (resource-feasible over [t,t+dur))
    - if nothing can start, jump to next completion time (earliest end_time)
    """
    name: str = "naive_event"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        hw = problem.hw
        pol = get_resource_policy(problem.policy_name)

        # ---- indegree-left using dag.pred ----
        indeg_left: Dict[str, int] = {nid: len(dag.pred.get(nid, set())) for nid in dag.nodes}
        ready: Set[str] = {nid for nid, d in indeg_left.items() if d == 0 and getattr(dag.nodes[nid], "duration", 1) > 0}

        # active heap: (end_time, nid)
        active: List[Tuple[int, str]] = []
        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}  # start time

        tracker = HardwareTracker(hw=hw, policy=pol)

        t = int(problem.meta.get("start_time", 0))

        def node_duration(nid: str) -> int:
            if nid in problem.duration_override:
                return int(problem.duration_override[nid])
            d = getattr(dag.nodes[nid], "duration", 1)
            return int(d) if d is not None else 1

        tie = problem.meta.get("tie_breaker", "nid")  # "nid" or "duration"

        def ready_order(nid: str):
            if tie == "duration":
                # longer first as you wrote (arbitrary heuristic)
                return (-node_duration(nid), nid)
            return (nid,)

        def try_start_at_time(t_now: int) -> List[str]:
            started: List[str] = []
            # iterate deterministically over a snapshot of ready
            for nid in sorted(list(ready), key=ready_order):
                if nid in entries:
                    continue
                node = dag.nodes[nid]
                dur = node_duration(nid)
                if dur == 0:
                    entries[nid] = ScheduleEntry(nid=nid, start=t_now, end=t_now)
                    node_to_time[nid] = t_now
                    ready.remove(nid)
                    started.append(nid)

                    # immediately “finish” it: reduce indegree of children right now
                    for ch in dag.succ.get(nid, set()):
                        indeg_left[ch] -= 1
                        if indeg_left[ch] == 0 and ch not in entries:
                            ready.add(ch)
                    continue
                
                if dur < 0:
                    raise ValueError(f"Node {nid} has non-positive duration {dur}")
                end = t_now + dur

                if tracker.can_reserve(node, t_now, end):
                    tracker.reserve(node, t_now, end)
                    entries[nid] = ScheduleEntry(nid=nid, start=t_now, end=end)
                    node_to_time[nid] = t_now
                    heapq.heappush(active, (end, nid))
                    ready.remove(nid)
                    started.append(nid)

            return started

        # ---- main loop ----
        while len(entries) < len(dag.nodes):
            # release finished tasks (and update indegrees)
            while active and active[0][0] <= t:
                end_time, finished = heapq.heappop(active)
                # when finished, reduce indegree of successors
                for ch in dag.succ.get(finished, set()):
                    indeg_left[ch] -= 1
                    if indeg_left[ch] == 0 and ch not in entries:
                        ready.add(ch)

            started_now = try_start_at_time(t)

            if started_now:
                # record a ScheduleStep: nodes that START at time t
                # (this keeps your existing Schedule structure)
                # Advance time: mild progress or jump to next completion
                if active:
                    # optional: move by 1, but never beyond next event
                    t = min(t + 1, active[0][0])
                else:
                    t = t + 1
                continue

            # nothing started
            if active:
                # jump to next completion time
                t = active[0][0]
                continue

            if ready:
                raise RuntimeError(
                    f"Ready nodes exist but none can be scheduled at t={t} under policy '{pol.name}'. "
                    f"Likely resource deadlock / inconsistent node resource annotation."
                )
            raise RuntimeError("No active jobs and no ready jobs, but schedule incomplete.")

        # Build steps list from entries grouped by start time
        starts: Dict[int, List[str]] = {}
        for nid, e in entries.items():
            starts.setdefault(e.start, []).append(nid)

        steps: List[ScheduleStep] = []
        for tt in sorted(starts):
            steps.append(ScheduleStep(t=tt, nodes=sorted(starts[tt]), meta={"algo": self.name, "policy": pol.name}))

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={
                "scheduler": self.name,
                "resource_policy": pol.name,
                "entries": {nid: {"start": e.start, "end": e.end} for nid, e in entries.items()},
            },
        )