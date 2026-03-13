# modqldpc/scheduling/algos/cp_sat_scheduling.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from modqldpc.lowering.ir import K_INTERBLOCK_LINK
from ortools.sat.python import cp_model

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep, ScheduleEntry


# ---------------------------------------------------------------------------
# Route alternative helpers
# ---------------------------------------------------------------------------

def _all_simple_paths(
    neighbors: Dict[int, Set[int]],
    src: int,
    dst: int,
    max_hops: int,
) -> List[List[int]]:
    """BFS over simple paths from src to dst with at most max_hops edges."""
    results: List[List[int]] = []
    # queue entries: (current_node, path_so_far)
    queue: deque = deque([(src, [src])])
    while queue:
        node, path = queue.popleft()
        if node == dst:
            results.append(path)
            continue
        if len(path) - 1 >= max_hops:
            continue
        for nb in neighbors.get(node, set()):
            if nb not in path:  # simple path – no revisits
                queue.append((nb, path + [nb]))
    return results


@dataclass(frozen=True)
class _RouteOption:
    """One candidate route for a link node: a specific assignment of paths."""
    blocks: Tuple[int, ...]       # all blocks touched (deduped)
    couplers: Tuple[str, ...]     # all couplers used (deduped)
    duration: int                 # max hop-length across all source paths


def _enumerate_route_options(
    node: Any,
    hw: Any,
    max_hops: int,
) -> List[_RouteOption]:
    """
    For a link node with meta containing magic_block, source_blocks, route_paths,
    enumerate all feasible route combinations (one path per source block).

    Each combination is a Cartesian product of per-source alternative paths.
    We cap the total number of combinations at 64 to avoid model explosion.
    """
    meta = getattr(node, "meta", {}) or {}
    magic_block: Optional[int] = meta.get("magic_block")
    source_blocks: Optional[List[int]] = meta.get("source_blocks")

    if magic_block is None or not source_blocks:
        # Fallback: use the single fixed route already encoded in node
        blocks = tuple(sorted(set(getattr(node, "blocks", []))))
        couplers = tuple(dict.fromkeys(getattr(node, "couplers", [])))
        duration = getattr(node, "duration", 1) or 1
        return [_RouteOption(blocks=blocks, couplers=couplers, duration=duration)]

    # Build per-source candidate paths
    per_source: List[List[List[int]]] = []
    for src in source_blocks:
        paths = _all_simple_paths(hw.neighbors, src, magic_block, max_hops)
        if not paths:
            # No alternative — use the original shortest path encoded in the node
            orig_route_paths: List[List[int]] = meta.get("route_paths", [])
            fallback = next((p for p in orig_route_paths if p[0] == src), None)
            paths = [fallback] if fallback else [[src, magic_block]]
        per_source.append(paths)

    # Cartesian product of per-source path choices
    MAX_COMBOS = 64
    combos: List[List[List[int]]] = [[]]
    for paths in per_source:
        new_combos: List[List[List[int]]] = []
        for existing in combos:
            for p in paths:
                new_combos.append(existing + [p])
                if len(new_combos) >= MAX_COMBOS:
                    break
            if len(new_combos) >= MAX_COMBOS:
                break
        combos = new_combos

    options: List[_RouteOption] = []
    seen_sigs: Set[Tuple] = set()

    for combo in combos:
        all_blocks: List[int] = [magic_block]
        all_couplers: List[str] = []
        hop_lengths: List[int] = []

        for path in combo:
            hop_lengths.append(len(path) - 1)
            for b in path:
                if b not in all_blocks:
                    all_blocks.append(b)
            for u, v in zip(path, path[1:]):
                cid = hw.coupler_id(u, v)
                if cid and cid not in all_couplers:
                    all_couplers.append(cid)

        sig = (tuple(sorted(all_blocks)), tuple(all_couplers))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        duration = max(hop_lengths) if hop_lengths else 1
        options.append(_RouteOption(
            blocks=tuple(sorted(all_blocks)),
            couplers=tuple(all_couplers),
            duration=max(duration, 1),
        ))

    return options if options else [_RouteOption(
        blocks=tuple(sorted(set(getattr(node, "blocks", [])))),
        couplers=tuple(dict.fromkeys(getattr(node, "couplers", []))),
        duration=max(getattr(node, "duration", 1) or 1, 1),
    )]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class CPSATScheduler(BaseScheduler):
    name: str = "cp_sat"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        hw = problem.hw

        time_limit: Optional[float] = problem.meta.get("cp_sat_time_limit", None)
        use_routes: bool = bool(problem.meta.get("cp_sat_route_alternatives", True))
        max_hops: int = int(problem.meta.get("cp_sat_max_hops", len(hw.blocks)))
        log_search: bool = bool(problem.meta.get("cp_sat_log", False))

        # ── node durations ───────────────────────────────────────────────────
        node_dur: Dict[str, int] = {}
        for nid in dag.nodes:
            if nid in problem.duration_override:
                node_dur[nid] = int(problem.duration_override[nid])
            else:
                d = getattr(dag.nodes[nid], "duration", 1)
                node_dur[nid] = int(d) if d is not None else 1

        # ── horizon (loose upper bound on makespan) ──────────────────────────
        horizon = sum(node_dur.values()) + 1

        # ── build CP-SAT model ───────────────────────────────────────────────
        model = cp_model.CpModel()

        # For each node: start variable.  end = start + duration (fixed).
        start_vars: Dict[str, cp_model.IntVar] = {}
        end_vars: Dict[str, cp_model.IntVar] = {}

        # Interval variables for resource constraints.
        # Zero-duration nodes are excluded from NoOverlap (they hold no resources).
        # For link nodes with route alternatives, we store optional intervals separately.
        # For all others: one standard interval per node.

        # resource_intervals[resource_key] → list of (interval_var, optional or not)
        # resource_key is either ("block", block_id) or ("coupler", coupler_id)
        block_intervals: Dict[int, List[cp_model.IntervalVar]] = {}
        coupler_intervals: Dict[str, List[cp_model.IntervalVar]] = {}

        # ── Step 1: create variables ─────────────────────────────────────────
        for nid, node in dag.nodes.items():
            dur = node_dur[nid]
            s = model.new_int_var(0, horizon, f"s_{nid}")
            e = model.new_int_var(0, horizon, f"e_{nid}")
            model.add(e == s + dur)
            start_vars[nid] = s
            end_vars[nid] = e

        # ── Step 2: precedence constraints ───────────────────────────────────
        for u in dag.nodes:
            for v in dag.succ.get(u, set()):
                model.add(end_vars[u] <= start_vars[v])

        # ── Step 3: resource constraints ────────────────────────────────────
        for nid, node in dag.nodes.items():
            dur = node_dur[nid]
            if dur == 0:
                continue  # zero-duration nodes hold no resources

            is_link = (getattr(node, "kind", "") == K_INTERBLOCK_LINK)

            if use_routes and is_link:
                # ── route-alternative encoding ───────────────────────────────
                options = _enumerate_route_options(node, hw, max_hops)

                if len(options) == 1:
                    # Only one route — use a standard interval (cheaper model)
                    opt = options[0]
                    iv = model.new_interval_var(
                        start_vars[nid], opt.duration, end_vars[nid], f"iv_{nid}"
                    )
                    model.add(end_vars[nid] == start_vars[nid] + opt.duration)
                    self._register_interval(iv, opt.blocks, opt.couplers,
                                            block_intervals, coupler_intervals)
                else:
                    # Multiple routes: optional intervals, exactly one selected
                    sel_vars = [
                        model.new_bool_var(f"sel_{nid}_r{i}")
                        for i in range(len(options))
                    ]
                    model.add_exactly_one(sel_vars)

                    for i, (opt, sel) in enumerate(zip(options, sel_vars)):
                        opt_s = model.new_int_var(0, horizon, f"s_{nid}_r{i}")
                        opt_e = model.new_int_var(0, horizon, f"e_{nid}_r{i}")
                        iv = model.new_optional_interval_var(
                            opt_s, opt.duration, opt_e, sel, f"iv_{nid}_r{i}"
                        )
                        # Link to node start/end when this route is selected
                        model.add(start_vars[nid] == opt_s).only_enforce_if(sel)
                        model.add(end_vars[nid] == opt_e).only_enforce_if(sel)
                        self._register_interval(iv, opt.blocks, opt.couplers,
                                                block_intervals, coupler_intervals)
            else:
                # ── standard (fixed-route) interval ──────────────────────────
                iv = model.new_interval_var(
                    start_vars[nid], dur, end_vars[nid], f"iv_{nid}"
                )
                # Collect blocks: directly touched + incident via couplers
                raw_blocks = list(getattr(node, "blocks", []) or [])
                raw_couplers = list(getattr(node, "couplers", []) or [])

                all_blocks: Set[int] = set(raw_blocks)
                for cid in raw_couplers:
                    spec = hw.couplers.get(cid)
                    if spec:
                        all_blocks.add(spec.u)
                        all_blocks.add(spec.v)

                self._register_interval(iv, tuple(all_blocks), tuple(raw_couplers),
                                        block_intervals, coupler_intervals)

        # Add NoOverlap for each block and each coupler
        for intervals in block_intervals.values():
            if len(intervals) > 1:
                model.add_no_overlap(intervals)
        for intervals in coupler_intervals.values():
            if len(intervals) > 1:
                model.add_no_overlap(intervals)

        # ── Step 4: objective ────────────────────────────────────────────────
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, list(end_vars.values()))
        model.minimize(makespan)

        # ── Solve ────────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = log_search
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = float(time_limit)

        status = solver.solve(model)

        status_name = solver.status_name(status)
        cp_makespan = int(solver.objective_value) if status in (
            cp_model.OPTIMAL, cp_model.FEASIBLE
        ) else -1
        cp_lb = int(solver.best_objective_bound) if status in (
            cp_model.OPTIMAL, cp_model.FEASIBLE
        ) else -1

        print(f"[CP-SAT] status={status_name}  makespan={cp_makespan}  lb={cp_lb}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"CP-SAT could not find a feasible schedule (status={status_name}). "
                "Check hardware constraints or increase time limit."
            )

        # ── Extract schedule ─────────────────────────────────────────────────
        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}

        for nid in dag.nodes:
            s_val = solver.value(start_vars[nid])
            e_val = solver.value(end_vars[nid])
            entries[nid] = ScheduleEntry(nid=nid, start=s_val, end=e_val)
            node_to_time[nid] = s_val

        starts_grouped: Dict[int, List[str]] = {}
        for nid, e in entries.items():
            starts_grouped.setdefault(e.start, []).append(nid)

        steps: List[ScheduleStep] = [
            ScheduleStep(
                t=tt,
                nodes=sorted(starts_grouped[tt]),
                meta={"algo": self.name},
            )
            for tt in sorted(starts_grouped)
        ]

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={
                "scheduler": self.name,
                "cp_sat_status": status_name,
                "cp_sat_makespan": cp_makespan,
                "cp_sat_lb": cp_lb,
                "cp_sat_gap": cp_makespan - cp_lb,
                "cp_sat_time_limit": time_limit,
                "cp_sat_route_alternatives": use_routes,
                "cp_sat_max_hops": max_hops,
                "entries": {
                    nid: {"start": e.start, "end": e.end}
                    for nid, e in entries.items()
                },
            },
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _register_interval(
        iv: cp_model.IntervalVar,
        blocks: Any,
        couplers: Any,
        block_intervals: Dict[int, List],
        coupler_intervals: Dict[str, List],
    ) -> None:
        for b in blocks:
            block_intervals.setdefault(b, []).append(iv)
        for c in couplers:
            coupler_intervals.setdefault(c, []).append(iv)
