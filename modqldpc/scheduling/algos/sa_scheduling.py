# modqldpc/scheduling/algos/sa_scheduler.py
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Set, Tuple, Optional
import heapq

from modqldpc.lowering.ir import K_INIT_PIVOT

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep, ScheduleEntry
from ..policy import get_resource_policy
from ..tracker import HardwareTracker


@dataclass(frozen=True)
class SACandidateState:
    component_order: List[int]


@dataclass
class SAPreprocessed:
    components: List[Set[str]]
    node_to_component: Dict[str, int]
    component_root: Dict[int, str]
    component_metrics: Dict[int, Dict[str, Any]]
    bottom_level: Dict[str, int]
    node_duration: Dict[str, int]
    block_criticality: Dict[int, int]
    node_block_criticality: Dict[str, int]
    link_nodes: Set[str] = field(default_factory=set)

    link_route_blocks: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    link_route_couplers: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    link_route_depth: Dict[str, int] = field(default_factory=dict)       # current duration proxy
    link_route_span: Dict[str, int] = field(default_factory=dict)        # number of blocks in route
    link_route_width: Dict[str, int] = field(default_factory=dict)       # number of couplers in route
    link_route_kind: Dict[str, str] = field(default_factory=dict)       # metadata tag


@dataclass
class SimulatedAnnealingScheduler(BaseScheduler):
    name: str = "sa_scheduler"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        pol = get_resource_policy(problem.policy_name)
        dag = problem.dag
        prep = self._preprocess(problem)
        init_state = self._build_initial_state(problem, prep)
        print("Initial candidate component order:", init_state.component_order)

        best_state, best_entries, best_node_to_time, best_decode_meta, sa_meta = self._run_simulated_annealing(
            problem, prep, init_state
        )

        print("Best candidate component order:", best_state.component_order)

        print(f"Initial cost {sa_meta['sa_initial_makespan']}, best cost {sa_meta['sa_best_makespan']} after {sa_meta['sa_iterations']} iterations")

        lb = max(prep.bottom_level[nid] for nid in dag.nodes if not dag.pred.get(nid))
        gap = sa_meta['sa_best_makespan'] - lb
        print(f"SA makespan: {sa_meta['sa_best_makespan']}, CPM lower bound: {lb}, gap: {gap} ({100*gap/lb:.1f}%)")
        starts: Dict[int, List[str]] = {}
        for nid, e in best_entries.items():
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
            node_to_time=best_node_to_time,
            meta={
                "scheduler": self.name,
                "resource_policy": pol.name,
                "entries": {
                    nid: {"start": e.start, "end": e.end}
                    for nid, e in best_entries.items()
                },
                "node_to_component": dict(prep.node_to_component),
                "component_root": dict(prep.component_root),
                "component_order": list(init_state.component_order),
                "block_criticality": dict(prep.block_criticality),
                "node_block_criticality": dict(prep.node_block_criticality),
                "link_route_depth": dict(prep.link_route_depth),
                "link_route_span": dict(prep.link_route_span),
                "link_route_width": dict(prep.link_route_width),
                "link_route_kind": dict(prep.link_route_kind),
                **best_decode_meta,
                **sa_meta,
            },
        )


    def _preprocess(self, problem: SchedulingProblem) -> SAPreprocessed:
        dag = problem.dag

        node_duration: Dict[str, int] = {
            nid: self._node_duration(problem, nid)
            for nid in dag.nodes
        }
        bottom_level = self._compute_bottom_levels(dag, node_duration)
        # print("Bottom levels:", bottom_level)

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

        global_indeg: Dict[str, int] = {
            nid: len(dag.pred.get(nid, set()))
            for nid in dag.nodes
        }

        link_nodes: Set[str] = set()
        link_route_blocks: Dict[str, Tuple[int, ...]] = {}
        link_route_couplers: Dict[str, Tuple[str, ...]] = {}
        link_route_depth: Dict[str, int] = {}
        link_route_span: Dict[str, int] = {}
        link_route_width: Dict[str, int] = {}
        link_route_kind: Dict[str, str] = {}

        for nid, node in dag.nodes.items():
            if self._is_link_node(node):
                link_nodes.add(nid)
                route_info = self._extract_link_route_info(node, node_duration[nid])

                link_route_blocks[nid] = route_info["blocks"]
                link_route_couplers[nid] = route_info["couplers"]
                link_route_depth[nid] = route_info["depth"]
                link_route_span[nid] = route_info["span"]
                link_route_width[nid] = route_info["width"]
                link_route_kind[nid] = route_info["kind"]

        component_root: Dict[int, str] = {}
        component_metrics: Dict[int, Dict[str, Any]] = {}
        block_criticality: Dict[int, int] = {}
        node_block_criticality: Dict[str, int] = {}

        for nid, node in dag.nodes.items():
            blocks = self._node_blocks(node)
            bl = bottom_level[nid]
            for b in blocks:
                block_criticality[b] = block_criticality.get(b, 0) + bl
        
        for nid, node in dag.nodes.items():
            blocks = self._node_blocks(node)
            node_block_criticality[nid] = sum(block_criticality.get(b, 0) for b in blocks)

        for cid, comp in enumerate(components):
            roots = [nid for nid in comp if global_indeg[nid] == 0 and self._is_init_node(dag.nodes[nid], nid)]
            if not roots:
                raise RuntimeError(
                    f"Component {cid} has no independent init root. "
                    f"This violates the expected DAG structure."
                )

            root = sorted(roots)[0]
            component_root[cid] = root

            comp_bottom_max = max(bottom_level[nid] for nid in comp)
            comp_bottom_sum = sum(bottom_level[nid] for nid in comp)
            comp_duration_sum = sum(node_duration[nid] for nid in comp)
            comp_block_crit_sum = sum(node_block_criticality[nid] for nid in comp)
            comp_link_count = sum(1 for nid in comp if nid in link_nodes)
            comp_link_depth = sum(link_route_depth.get(nid, 0) for nid in comp)
            comp_link_width = sum(link_route_width.get(nid, 0) for nid in comp)

            # print(f"Component {cid}: size={len(comp)}, root={root}, "
            #       f"bottom_max={comp_bottom_max}, bottom_sum={comp_bottom_sum}, "
            #       f"duration_sum={comp_duration_sum}, block_crit_sum={comp_block_crit_sum}, "
            #       f"link_count={comp_link_count}")

            component_metrics[cid] = {
                "size": len(comp),
                "bottom_max": comp_bottom_max,
                "bottom_sum": comp_bottom_sum,
                "duration_sum": comp_duration_sum,
                "block_crit_sum": comp_block_crit_sum,
                "link_count": comp_link_count,
                "link_depth_sum": comp_link_depth,
                "link_width_sum": comp_link_width,
            }

        return SAPreprocessed(
            components=components,
            node_to_component=node_to_component,
            component_root=component_root,
            component_metrics=component_metrics,
            bottom_level=bottom_level,
            node_duration=node_duration,
            block_criticality=block_criticality,
            node_block_criticality=node_block_criticality,
            link_nodes=link_nodes,
            link_route_blocks=link_route_blocks,
            link_route_couplers=link_route_couplers,
            link_route_depth=link_route_depth,
            link_route_span=link_route_span,
            link_route_width=link_route_width,
            link_route_kind=link_route_kind,
        )
    
    def _run_simulated_annealing(
        self,
        problem: SchedulingProblem,
        prep: SAPreprocessed,
        init_state: SACandidateState,
    ) -> Tuple[
        SACandidateState,
        Dict[str, ScheduleEntry],
        Dict[str, int],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        seed = int(problem.meta.get("sa_seed", problem.seed))
        rng = random.Random(seed)

        iterations = int(problem.meta.get("sa_iterations", 50))
        initial_temp = float(problem.meta.get("sa_initial_temp", 10.0))
        cooling_rate = float(problem.meta.get("sa_cooling_rate", 0.95))
        neighbor_mode = str(problem.meta.get("sa_neighbor", "mixed"))

        if iterations <= 0:
            entries, node_to_time, decode_meta = self._decode_candidate(problem, prep, init_state)
            return init_state, entries, node_to_time, decode_meta, {
                "sa_iterations": 0,
                "sa_initial_temp": initial_temp,
                "sa_cooling_rate": cooling_rate,
                "sa_neighbor": neighbor_mode,
                "sa_seed": seed,
                "sa_best_makespan": decode_meta["makespan"],
                "sa_initial_makespan": decode_meta["makespan"],
                "sa_accept_count": 0,
                "sa_improve_count": 0,
            }

        current_state = init_state
        current_entries, current_node_to_time, current_decode_meta = self._decode_candidate(problem, prep, current_state)
        current_cost = int(current_decode_meta["makespan"])
        true_initial_cost = current_cost  # captured once, never overwritten

        best_state = current_state
        best_entries = current_entries
        best_node_to_time = current_node_to_time
        best_decode_meta = current_decode_meta
        best_cost = current_cost

        accept_count = 0
        improve_count = 0

        # Auto-calibrate initial temperature unless the user supplied an explicit value.
        # Run n_pilot random neighbors, collect uphill deltas, and choose T so that a
        # typical uphill move is accepted with probability p0 ≈ 0.80 at the start.
        user_supplied_temp = "sa_initial_temp" in problem.meta
        if user_supplied_temp:
            temperature = initial_temp
        else:
            n_pilot = min(20, max(1, iterations // 5))
            pilot_deltas: List[float] = []
            for _ in range(n_pilot):
                ps = self._propose_neighbor(current_state, rng, neighbor_mode)
                _, _, pm = self._decode_candidate(problem, prep, ps)
                d = int(pm["makespan"]) - current_cost
                if d > 0:
                    pilot_deltas.append(float(d))
            if pilot_deltas:
                mean_delta = sum(pilot_deltas) / len(pilot_deltas)
                p0 = 0.80
                temperature = -mean_delta / math.log(p0)
            else:
                # All pilot neighbors were improvements — start temperature low
                temperature = max(1.0, current_cost * 0.01)
            initial_temp = temperature  # keep sa_meta consistent

        for _it in range(iterations):
            candidate_state = self._propose_neighbor(current_state, rng, neighbor_mode)

            candidate_entries, candidate_node_to_time, candidate_decode_meta = self._decode_candidate(
                problem, prep, candidate_state
            )
            candidate_cost = int(candidate_decode_meta["makespan"])

            if self._accept(current_cost, candidate_cost, temperature, rng):
                current_state = candidate_state
                current_entries = candidate_entries
                current_node_to_time = candidate_node_to_time
                current_decode_meta = candidate_decode_meta
                current_cost = candidate_cost
                accept_count += 1

                if candidate_cost < best_cost:
                    best_state = candidate_state
                    best_entries = candidate_entries
                    best_node_to_time = candidate_node_to_time
                    best_decode_meta = candidate_decode_meta
                    best_cost = candidate_cost
                    improve_count += 1

            temperature *= cooling_rate

        sa_meta = {
            "sa_iterations": iterations,
            "sa_initial_temp": initial_temp,
            "sa_cooling_rate": cooling_rate,
            "sa_neighbor": neighbor_mode,
            "sa_seed": seed,
            "sa_initial_makespan": true_initial_cost,
            "sa_best_makespan": best_cost,
            "sa_accept_count": accept_count,
            "sa_improve_count": improve_count,
        }

        return best_state, best_entries, best_node_to_time, best_decode_meta, sa_meta

    def _propose_neighbor(
        self,
        state: SACandidateState,
        rng: random.Random,
        mode: str,
    ) -> SACandidateState:
        order = list(state.component_order)
        n = len(order)

        if n <= 1:
            return SACandidateState(component_order=order)

        chosen_mode = mode
        if mode == "mixed":
            chosen_mode = "swap" if rng.random() < 0.5 else "insert"

        if chosen_mode == "swap":
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
            return SACandidateState(component_order=order)

        if chosen_mode == "insert":
            i, j = rng.sample(range(n), 2)
            cid = order.pop(i)
            order.insert(j, cid)
            return SACandidateState(component_order=order)

        raise ValueError(f"Unknown sa_neighbor mode '{mode}'")

    def _accept(
        self,
        current_cost: int,
        candidate_cost: int,
        temperature: float,
        rng: random.Random,
    ) -> bool:
        if candidate_cost < current_cost:
            return True
        if temperature <= 0:
            return False

        delta = candidate_cost - current_cost
        prob = math.exp(-float(delta) / float(temperature))
        return rng.random() < prob

    def _build_initial_state(self, problem: SchedulingProblem, prep: SAPreprocessed) -> SACandidateState:
        ordered = sorted(
            range(len(prep.components)),
            key=lambda cid: (
                -prep.component_metrics[cid]["bottom_max"],
                -prep.component_metrics[cid]["block_crit_sum"],
                -prep.component_metrics[cid]["bottom_sum"],
                -prep.component_metrics[cid]["duration_sum"],
                prep.component_metrics[cid]["link_depth_sum"],   # heavier link routes first
                prep.component_metrics[cid]["link_width_sum"],
                cid,
            ),
        )
        return SACandidateState(component_order=ordered)


    def _decode_candidate(
        self,
        problem: SchedulingProblem,
        prep: SAPreprocessed,
        state: SACandidateState,
    ) -> Tuple[Dict[str, ScheduleEntry], Dict[str, int], Dict[str, Any]]:
        dag = problem.dag
        hw = problem.hw
        pol = get_resource_policy(problem.policy_name)
        tracker = HardwareTracker(hw=hw, policy=pol)
        dbg = bool(problem.meta.get("debug_decode", False))

        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}

        indeg_left: Dict[str, int] = {
            nid: len(dag.pred.get(nid, set()))
            for nid in dag.nodes
        }
        ready: Set[str] = {nid for nid, d in indeg_left.items() if d == 0}

        active: List[Tuple[int, str]] = []
        t = int(problem.meta.get("start_time", 0))

        if dbg:
            print(f"\n{'='*60}")
            print(f"[decode] START  nodes={len(dag.nodes)}  comp_order={state.component_order}")
            print(f"{'='*60}")

        comp_rank: Dict[int, int] = {
            cid: rank for rank, cid in enumerate(state.component_order)
        }

        tie = problem.meta.get("tie_breaker", "nid")

        def node_priority(nid: str) -> Tuple:
            cid = prep.node_to_component[nid]
            dur = prep.node_duration[nid]
            bottom = prep.bottom_level[nid]
            block_crit = prep.node_block_criticality[nid]
            
            is_link = nid in prep.link_nodes
            link_depth = prep.link_route_depth.get(nid, 0)
            link_width = prep.link_route_width.get(nid, 0)
            link_span = prep.link_route_span.get(nid, 0)


            if tie == "duration":
                return (
                    comp_rank[cid],              # lower-ranked component first
                    -bottom,                     # larger bottom-level first
                    -block_crit,                 # larger block criticality first
                    0 if not is_link else link_depth,   # deeper routes first
                    0 if not is_link else link_width,   # wider routes first
                    0 if not is_link else link_span,    # more blocks first
                    -dur,                        # longer duration first
                    nid,
                )

            return (
                comp_rank[cid],
                -bottom,
                -block_crit,
                0 if not is_link else link_depth,
                0 if not is_link else link_width,
                0 if not is_link else link_span,
                -dur,
                nid,
            )

        def add_children_of(finished_nid: str) -> None:
            for ch in dag.succ.get(finished_nid, set()):
                indeg_left[ch] -= 1
                if indeg_left[ch] == 0 and ch not in entries:
                    ready.add(ch)

        while len(entries) < len(dag.nodes):
            while active and active[0][0] <= t:
                _, finished = heapq.heappop(active)
                add_children_of(finished)
                if dbg:
                    print(f"  [t={t}] finished: {finished}")

            if ready:
                pq: List[Tuple[Tuple, str]] = []
                for nid in ready:
                    heapq.heappush(pq, (node_priority(nid), nid))

                if dbg:
                    print(f"\n[t={t}] ready queue ({len(ready)} nodes):")
                    tmp_pq = list(pq)
                    for pri, nid in sorted(tmp_pq):
                        cid = prep.node_to_component[nid]
                        print(f"  {nid:30s}  dur={prep.node_duration[nid]}  bl={prep.bottom_level[nid]}  cid={cid}  priority={pri}")

                started_any = False
                blocked: List[str] = []

                while pq:
                    _, nid = heapq.heappop(pq)
                    if nid in entries or nid not in ready:
                        continue

                    node = dag.nodes[nid]
                    dur = prep.node_duration[nid]

                    if dur < 0:
                        raise ValueError(f"Node {nid} has non-positive duration {dur}")

                    if dur == 0:
                        entries[nid] = ScheduleEntry(nid=nid, start=t, end=t)
                        node_to_time[nid] = t
                        ready.remove(nid)
                        add_children_of(nid)
                        started_any = True
                        if dbg:
                            print(f"  [t={t}] INSTANT  {nid}")
                        continue

                    end = t + dur
                    if tracker.can_reserve(node, t, end):
                        tracker.reserve(node, t, end)
                        entries[nid] = ScheduleEntry(nid=nid, start=t, end=end)
                        node_to_time[nid] = t
                        heapq.heappush(active, (end, nid))
                        ready.remove(nid)
                        started_any = True
                        if dbg:
                            print(f"  [t={t}] STARTED  {nid}  dur={dur}  end={end}")
                    else:
                        blocked.append(nid)
                        if dbg:
                            print(f"  [t={t}] BLOCKED  {nid}  dur={dur}")

                for nid in blocked:
                    ready.add(nid)

                if started_any:
                    if active:
                        t = min(t + 1, active[0][0])
                    else:
                        t = t + 1
                    continue

            if active:
                t = active[0][0]
                continue

            if ready:
                raise RuntimeError(
                    f"Ready nodes exist at t={t}, but none can be scheduled under "
                    f"policy '{pol.name}'. Likely resource deadlock / inconsistent annotation."
                )

            raise RuntimeError("No active and no ready but schedule incomplete.")

        makespan = 0 if not entries else max(e.end for e in entries.values())

        if dbg:
            print(f"\n[decode] DONE  makespan={makespan}  scheduled={len(entries)}")
            print("  Node schedule:")
            for nid in sorted(entries, key=lambda n: entries[n].start):
                e = entries[nid]
                cid = prep.node_to_component[nid]
                print(f"    {nid:30s}  [{e.start:3d} .. {e.end:3d}]  cid={cid}")
            print(f"{'='*60}\n")

        return entries, node_to_time, {
            "candidate_component_order": list(state.component_order),
            "makespan": makespan,
            "num_components": len(prep.components),
            "num_link_nodes": len(prep.link_nodes),
        }

    def _node_duration(self, problem: SchedulingProblem, nid: str) -> int:
        dag = problem.dag
        if nid in problem.duration_override:
            return int(problem.duration_override[nid])
        d = getattr(dag.nodes[nid], "duration", 1)
        return int(d) if d is not None else 1

    def _is_init_node(self, node: Any, nid: str) -> bool:
        kind = getattr(node, "kind", None)
        if kind is not None:
            return kind == K_INIT_PIVOT
        return nid.startswith("init_")
    
    def _node_blocks(self, node: Any) -> Set[int]:
        """
        Extract touched blocks from the node in a robust way.
        """
        blocks = getattr(node, "blocks", None)
        if blocks is None:
            return set()
        return set(blocks)

    def _is_link_node(self, node: Any) -> bool:
        kind = getattr(node, "kind", None)
        nid = getattr(node, "nid", "")
        if kind is not None and str(kind).startswith("link"):
            return True
        return str(nid).startswith("link_")
    
    def _extract_link_route_info(self, node: Any, fallback_duration: int) -> Dict[str, Any]:
        blocks = tuple(sorted(set(getattr(node, "blocks", []) or [])))
        couplers = tuple(dict.fromkeys(getattr(node, "couplers", []) or []))  # stable dedup
        duration = getattr(node, "duration", None)
        duration = fallback_duration if duration is None else int(duration)

        meta = getattr(node, "meta", None)
        route_kind = "unknown"
        if isinstance(meta, dict):
            route_kind = str(meta.get("routing", "unknown"))

        return {
            "blocks": blocks,
            "couplers": couplers,
            "depth": max(0, duration),
            "span": len(blocks),
            "width": len(couplers),
            "kind": route_kind,
        }
    
    def _compute_bottom_levels(self, dag: Any, node_duration: Dict[str, int]) -> Dict[str, int]:
        """
        Weighted bottom level:
          BL(n) = dur(n) + max(BL(child)) over children
        """
        out: Dict[str, int] = {}
        temp_mark: Set[str] = set()
        perm_mark: Set[str] = set()

        def dfs(nid: str) -> int:
            if nid in perm_mark:
                return out[nid]
            if nid in temp_mark:
                raise ValueError("DAG has a cycle; cannot compute bottom levels.")

            temp_mark.add(nid)
            child_vals = [dfs(ch) for ch in dag.succ.get(nid, set())]
            out[nid] = node_duration[nid] + (max(child_vals) if child_vals else 0)
            temp_mark.remove(nid)
            perm_mark.add(nid)
            return out[nid]

        for nid in dag.nodes:
            if nid not in perm_mark:
                dfs(nid)

        return out