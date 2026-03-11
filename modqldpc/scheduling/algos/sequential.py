# # modqldpc/scheduling/algos/naive_event.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Set, Tuple, Optional
# import heapq
# import re

# from ..base import BaseScheduler
# from ..types import SchedulingProblem, Schedule, ScheduleStep, ScheduleEntry
# from ..policy import get_resource_policy
# from ..tracker import HardwareTracker


# @dataclass
# class SequentialScheduler(BaseScheduler):
#     """
#     Event-based greedy scheduler with durations (baseline).

#     Modified behavior:
#     - choose one ready init node
#     - lock onto its rotation id
#     - schedule only nodes from that rotation until the rotation completes
#     - then choose the next ready init node from another rotation
#     """
#     name: str = "sequential"

#     def solve(self, problem: SchedulingProblem) -> Schedule:
#         dag = problem.dag
#         hw = problem.hw
#         pol = get_resource_policy(problem.policy_name)

#         # ---- indegree-left using dag.pred ----
#         indeg_left: Dict[str, int] = {nid: len(dag.pred.get(nid, set())) for nid in dag.nodes}
#         ready: Set[str] = {nid for nid, d in indeg_left.items() if d == 0}

#         # active heap: (end_time, nid)
#         active: List[Tuple[int, str]] = []
#         entries: Dict[str, ScheduleEntry] = {}
#         node_to_time: Dict[str, int] = {}  # start time

#         tracker = HardwareTracker(hw=hw, policy=pol)

#         t = int(problem.meta.get("start_time", 0))

#         def node_duration(nid: str) -> int:
#             if nid in problem.duration_override:
#                 return int(problem.duration_override[nid])
#             d = getattr(dag.nodes[nid], "duration", 1)
#             return int(d) if d is not None else 1

#         tie = problem.meta.get("tie_breaker", "nid")  # "nid" or "duration"

#         def ready_order(nid: str):
#             if tie == "duration":
#                 return (-node_duration(nid), nid)
#             return (nid,)

#         # ------------------------------------------------------------
#         # minimal helpers for rotation-serial scheduling
#         # ------------------------------------------------------------
#         def rotation_key(nid: str) -> str:
#             """
#             Extract rotation id from nid.
#             Example:
#               init_L00_R000_B1_c0  -> L00_R000
#               lc_L00_R000_B1_c0_k0 -> L00_R000
#               PZ_L00_R000          -> L00_R000
#             """
#             m = re.search(r"(L\d+_R\d+)", nid)
#             if not m:
#                 raise ValueError(f"Could not extract rotation id from nid='{nid}'")
#             return m.group(1)

#         def is_init_node(nid: str) -> bool:
#             # Based on your naming pattern
#             return nid.startswith("init_")

#         node_to_rotation: Dict[str, str] = {nid: rotation_key(nid) for nid in dag.nodes}
#         rotation_to_nodes: Dict[str, Set[str]] = {}
#         for nid, rot in node_to_rotation.items():
#             rotation_to_nodes.setdefault(rot, set()).add(nid)

#         active_rotation: Optional[str] = None

#         def rotation_finished(rot: str) -> bool:
#             return all(nid in entries for nid in rotation_to_nodes[rot])

#         def pick_next_rotation() -> Optional[str]:
#             """
#             Pick next rotation only from currently ready independent init nodes.
#             Deterministic by ready_order.
#             """
#             ready_inits = [nid for nid in ready if is_init_node(nid) and nid not in entries]
#             if not ready_inits:
#                 return None
#             chosen_init = sorted(ready_inits, key=ready_order)[0]
#             return node_to_rotation[chosen_init]

#         def try_start_at_time(t_now: int) -> List[str]:
#             started: List[str] = []

#             # If no active rotation yet, choose one from a ready init node
#             nonlocal active_rotation
#             if active_rotation is None:
#                 active_rotation = pick_next_rotation()

#             # If still none, nothing to do at this time
#             if active_rotation is None:
#                 return started

#             # Only nodes from the active rotation are eligible
#             eligible = [
#                 nid for nid in ready
#                 if nid not in entries and node_to_rotation[nid] == active_rotation
#             ]

#             for nid in sorted(eligible, key=ready_order):
#                 node = dag.nodes[nid]
#                 dur = node_duration(nid)

#                 if dur == 0:
#                     entries[nid] = ScheduleEntry(nid=nid, start=t_now, end=t_now)
#                     node_to_time[nid] = t_now
#                     ready.remove(nid)
#                     started.append(nid)

#                     # immediately “finish” it
#                     for ch in dag.succ.get(nid, set()):
#                         indeg_left[ch] -= 1
#                         if indeg_left[ch] == 0 and ch not in entries:
#                             ready.add(ch)
#                     continue

#                 if dur < 0:
#                     raise ValueError(f"Node {nid} has non-positive duration {dur}")

#                 end = t_now + dur

#                 if tracker.can_reserve(node, t_now, end):
#                     tracker.reserve(node, t_now, end)
#                     entries[nid] = ScheduleEntry(nid=nid, start=t_now, end=end)
#                     node_to_time[nid] = t_now
#                     heapq.heappush(active, (end, nid))
#                     ready.remove(nid)
#                     started.append(nid)

#             return started

#         # ---- main loop ----
#         while len(entries) < len(dag.nodes):
#             # release finished tasks (and update indegrees)
#             while active and active[0][0] <= t:
#                 end_time, finished = heapq.heappop(active)
#                 for ch in dag.succ.get(finished, set()):
#                     indeg_left[ch] -= 1
#                     if indeg_left[ch] == 0 and ch not in entries:
#                         ready.add(ch)

#             # if current rotation is fully done, unlock and choose next later
#             if active_rotation is not None and rotation_finished(active_rotation):
#                 active_rotation = None

#             started_now = try_start_at_time(t)

#             if started_now:
#                 if active:
#                     t = min(t + 1, active[0][0])
#                 else:
#                     t = t + 1
#                 continue

#             # nothing started
#             if active:
#                 # wait for next completion; do not switch rotation while one is in progress
#                 t = active[0][0]
#                 continue

#             # no active jobs
#             if active_rotation is not None:
#                 unfinished = [
#                     nid for nid in rotation_to_nodes[active_rotation]
#                     if nid not in entries
#                 ]
#                 ready_in_active = [nid for nid in ready if nid in unfinished]

#                 if unfinished and not ready_in_active:
#                     raise RuntimeError(
#                         f"Rotation '{active_rotation}' has unfinished nodes but none are ready at t={t}. "
#                         f"Likely dependency issue inside the rotation DAG."
#                     )

#                 if ready_in_active:
#                     raise RuntimeError(
#                         f"Ready nodes exist in active rotation '{active_rotation}', "
#                         f"but none can be scheduled at t={t} under policy '{pol.name}'. "
#                         f"Likely resource deadlock / inconsistent node resource annotation."
#                     )

#             if ready:
#                 # ready exists, but maybe no ready init yet; this means structure is unexpected
#                 raise RuntimeError(
#                     f"Ready nodes exist at t={t}, but no ready init node is available to start a new rotation. "
#                     f"Ready={sorted(ready)}"
#                 )

#             raise RuntimeError("No active jobs and no ready jobs, but schedule incomplete.")

#         # Build steps list from entries grouped by start time
#         starts: Dict[int, List[str]] = {}
#         for nid, e in entries.items():
#             starts.setdefault(e.start, []).append(nid)

#         steps: List[ScheduleStep] = []
#         for tt in sorted(starts):
#             steps.append(
#                 ScheduleStep(
#                     t=tt,
#                     nodes=sorted(starts[tt]),
#                     meta={"algo": self.name, "policy": pol.name},
#                 )
#             )

#         return Schedule(
#             steps=steps,
#             node_to_time=node_to_time,
#             meta={
#                 "scheduler": self.name,
#                 "resource_policy": pol.name,
#                 "entries": {nid: {"start": e.start, "end": e.end} for nid, e in entries.items()},
#                 "node_to_rotation": dict(node_to_rotation),
#             },
#         )



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
class SequentialScheduler(BaseScheduler):
    """
    Completes one rotation at a time, but without enforcing that all nodes of a rotation must start together.
    This is a more relaxed version of the "rotation-serial" approach, which may allow better resource utilization at the cost of more complex schedules.
    Behavior:
    - choose one ready init node
    - lock onto its rotation id
    - schedule any ready nodes from that rotation as soon as they are ready, even if they don't start at the same time
    - once all nodes from that rotation are done, choose the next ready init node from another rotation
    """
    name: str = "sequential_scheduler"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        hw = problem.hw
        pol = get_resource_policy(problem.policy_name)

        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}
        tracker = HardwareTracker(hw=hw, policy=pol)

        t = int(problem.meta.get("start_time", 0))

        # ------------------------------------------------------------
        # basic helpers
        # ------------------------------------------------------------
        def node_duration(nid: str) -> int:
            if nid in problem.duration_override:
                return int(problem.duration_override[nid])
            d = getattr(dag.nodes[nid], "duration", 1)
            return int(d) if d is not None else 1

        tie = problem.meta.get("tie_breaker", "nid")  # "nid" or "duration"

        def node_priority(nid: str):
            if tie == "duration":
                return (-node_duration(nid), nid)
            return (nid,)

        def is_init_node(nid: str) -> bool:
            node = dag.nodes[nid]
            kind = getattr(node, "kind", None)
            if kind is not None:
                return kind == "init_pivot"
            return nid.startswith("init_")

        # ------------------------------------------------------------
        # global indegree (used only for identifying component roots)
        # ------------------------------------------------------------
        global_indeg: Dict[str, int] = {nid: len(dag.pred.get(nid, set())) for nid in dag.nodes}

        # ------------------------------------------------------------
        # build weakly connected components from DAG
        # each component = one rotation
        # ------------------------------------------------------------
        components: List[Set[str]] = []
        node_to_component: Dict[str, int] = {}
        seen: Set[str] = set()

        for start in dag.nodes:
            if start in seen:
                continue

            comp: Set[str] = set()
            stack = [start]
            seen.add(start)

            while stack:
                u = stack.pop()
                comp.add(u)

                neighbors = set(dag.pred.get(u, set())) | set(dag.succ.get(u, set()))
                for v in neighbors:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)

            cid = len(components)
            components.append(comp)
            for nid in comp:
                node_to_component[nid] = cid

        # ------------------------------------------------------------
        # choose one init root per component, then order components
        # ------------------------------------------------------------
        component_root: Dict[int, str] = {}

        for cid, comp in enumerate(components):
            roots = [nid for nid in comp if global_indeg[nid] == 0 and is_init_node(nid)]
            if not roots:
                raise RuntimeError(
                    f"Component {cid} has no independent init root. "
                    f"This violates the expected DAG structure for sequential scheduling."
                )

            # If more than one exists, choose deterministically.
            # This should not normally happen per your stated invariant.
            component_root[cid] = sorted(roots, key=node_priority)[0]

        component_order: List[int] = sorted(
            range(len(components)),
            key=lambda cid: node_priority(component_root[cid])
        )

        # ------------------------------------------------------------
        # schedule one component completely, then move to next
        # ------------------------------------------------------------
        for cid in component_order:
            comp = components[cid]

            # local indegree restricted to this component
            indeg_left: Dict[str, int] = {
                nid: sum(1 for p in dag.pred.get(nid, set()) if p in comp)
                for nid in comp
            }

            # local ready heap + membership
            ready_heap: List[Tuple] = []
            ready_set: Set[str] = set()

            def add_ready(nid: str) -> None:
                if nid in entries or nid in ready_set:
                    return
                ready_set.add(nid)
                heapq.heappush(ready_heap, (*node_priority(nid), nid))

            def pop_all_ready_in_order() -> List[str]:
                out: List[str] = []
                while ready_heap:
                    item = heapq.heappop(ready_heap)
                    nid = item[-1]
                    if nid in ready_set and nid not in entries:
                        ready_set.remove(nid)
                        out.append(nid)
                return out

            for nid, d in indeg_left.items():
                if d == 0:
                    add_ready(nid)

            # active heap only for this component: (end_time, nid)
            active: List[Tuple[int, str]] = []
            done_count = 0

            def schedule_zero_duration(nid: str, t_now: int) -> None:
                nonlocal done_count
                entries[nid] = ScheduleEntry(nid=nid, start=t_now, end=t_now)
                node_to_time[nid] = t_now
                done_count += 1

                for ch in dag.succ.get(nid, set()):
                    if ch in comp and ch not in entries:
                        indeg_left[ch] -= 1
                        if indeg_left[ch] == 0:
                            add_ready(ch)

            def release_finished_up_to(t_now: int) -> None:
                nonlocal done_count
                while active and active[0][0] <= t_now:
                    _, finished = heapq.heappop(active)
                    done_count += 1

                    for ch in dag.succ.get(finished, set()):
                        if ch in comp and ch not in entries:
                            indeg_left[ch] -= 1
                            if indeg_left[ch] == 0:
                                add_ready(ch)

            while done_count < len(comp):
                release_finished_up_to(t)

                started_now: List[str] = []
                blocked: List[str] = []

                candidates = pop_all_ready_in_order()

                for nid in candidates:
                    if nid in entries:
                        continue

                    node = dag.nodes[nid]
                    dur = node_duration(nid)

                    if dur < 0:
                        raise ValueError(f"Node {nid} has non-positive duration {dur}")

                    if dur == 0:
                        schedule_zero_duration(nid, t)
                        started_now.append(nid)
                        continue

                    end = t + dur
                    if tracker.can_reserve(node, t, end):
                        tracker.reserve(node, t, end)
                        entries[nid] = ScheduleEntry(nid=nid, start=t, end=end)
                        node_to_time[nid] = t
                        heapq.heappush(active, (end, nid))
                        started_now.append(nid)
                    else:
                        blocked.append(nid)

                # put blocked ready nodes back
                for nid in blocked:
                    add_ready(nid)

                if started_now:
                    if active:
                        t = min(t + 1, active[0][0])
                    else:
                        t = t + 1
                    continue

                if active:
                    t = active[0][0]
                    continue

                if ready_set:
                    raise RuntimeError(
                        f"Component {cid} has ready nodes at t={t}, but none can be scheduled under "
                        f"policy '{pol.name}'. Likely resource deadlock / inconsistent node resource annotation."
                    )

                raise RuntimeError(
                    f"Component {cid} is incomplete but has no active jobs and no ready jobs."
                )

        # ------------------------------------------------------------
        # build output
        # ------------------------------------------------------------
        starts: Dict[int, List[str]] = {}
        for nid, e in entries.items():
            starts.setdefault(e.start, []).append(nid)

        steps: List[ScheduleStep] = []
        for tt in sorted(starts):
            steps.append(
                ScheduleStep(
                    t=tt,
                    nodes=sorted(starts[tt]),
                    meta={"algo": self.name, "policy": pol.name},
                )
            )

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={
                "scheduler": self.name,
                "resource_policy": pol.name,
                "entries": {
                    nid: {"start": e.start, "end": e.end}
                    for nid, e in entries.items()
                },
                "node_to_component": dict(node_to_component),
                "component_root": dict(component_root),
                "component_order": list(component_order),
            },
        )