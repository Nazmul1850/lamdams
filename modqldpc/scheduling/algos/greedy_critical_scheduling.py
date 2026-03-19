# modqldpc/scheduling/algos/greedy_critical_scheduling.py
from __future__ import annotations

import contextlib
import heapq
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from modqldpc.lowering.ir import (
    K_FRAME_UPDATE,
    K_INIT_PIVOT,
    K_INTERBLOCK_LINK,
    K_LOCAL_COUPLE,
    K_MEAS_MAGIC_X,
    K_MEAS_PARITY_PZ,
)

from ..base import BaseScheduler
from ..ownership import BlockOwnershipTracker
from ..policy import get_resource_policy
from ..tracker import HardwareTracker
from ..types import Schedule, ScheduleEntry, ScheduleStep, SchedulingProblem


# ============================================================
# Preprocessed state
# ============================================================

@dataclass
class _Preprocessed:
    # ---- component topology ----
    components: List[Set[str]]
    node_to_component: Dict[str, int]
    comp_root: Dict[int, str]
    comp_metrics: Dict[int, Dict[str, Any]]

    # ---- scheduling metrics ----
    bottom_level: Dict[str, int]
    node_duration: Dict[str, int]
    block_criticality: Dict[int, int]
    node_block_criticality: Dict[str, int]

    # ---- link-node metadata (for priority) ----
    link_nodes: Set[str] = field(default_factory=set)
    link_route_depth: Dict[str, int] = field(default_factory=dict)
    link_route_span: Dict[str, int] = field(default_factory=dict)
    link_route_width: Dict[str, int] = field(default_factory=dict)

    # ---- per-component role maps (for ownership enforcement) ----
    comp_init_nodes: Dict[int, List[str]] = field(default_factory=dict)
    comp_link_node: Dict[int, Optional[str]] = field(default_factory=dict)
    comp_pz_node: Dict[int, Optional[str]] = field(default_factory=dict)
    comp_xm_node: Dict[int, Optional[str]] = field(default_factory=dict)
    comp_magic_block: Dict[int, Optional[int]] = field(default_factory=dict)
    comp_participant_blocks: Dict[int, Set[int]] = field(default_factory=dict)
    comp_route_blocks: Dict[int, Set[int]] = field(default_factory=dict)
    comp_route_couplers: Dict[int, List[str]] = field(default_factory=dict)
    comp_intermediate_blocks: Dict[int, Set[int]] = field(default_factory=dict)
    comp_is_multiblock: Dict[int, bool] = field(default_factory=dict)


# ============================================================
# Scheduler
# ============================================================

@dataclass
class GreedyCriticalScheduler(BaseScheduler):
    """
    Component-aware critical-path list scheduler with surgery-correct
    ownership semantics.

    Equivalent to SimulatedAnnealingScheduler with sa_iterations=0:
    runs exactly one decode pass using the same component-priority ordering
    and node-priority tuple as the SA decoder, with no annealing.

    Ownership scopes enforced
    -------------------------
      (1) Preparation lock:  block owned from first init_pivot until
          meas_parity_PZ (non-magic) or meas_magic_X (magic / single-block).
      (2) Route lock:        route blocks + couplers owned from
          interblock_link start to meas_parity_PZ end.
      (3) Magic-block lock:  magic block owned from first init (or link)
          through meas_magic_X end.

    Node priority (lexicographic, lower = higher priority):
      (comp_rank, -bottom_level, -block_criticality,
       link_depth, link_width, link_span, -duration, nid)

    Component ordering (most critical first):
      (-bottom_max, -block_crit_sum, -bottom_sum, -duration_sum,
       link_depth_sum, link_width_sum, cid)

    meta keys (all optional):
      start_time    (int)  : schedule offset, default 0
      tie_breaker   (str)  : "nid" (default) or "duration"
      debug_decode  (bool) : verbose decode trace, default False

    Fields
    ------
      use_safe_fill (bool):
        When True (default), a blocked multiblock component B is still
        admitted if its projected Xm completion falls before the lower-bound
        start of every conflicting in-progress component's link node
        (Phase 2 safe-fill).  Set to False to use strict Phase-1-only
        deadlock prevention (no speculative admission).
    """
    name: str = "greedy_critical"
    use_safe_fill: bool = True

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def solve(self, problem: SchedulingProblem) -> Schedule:
        prep = self._preprocess(problem)
        comp_order = self._build_component_order(prep)
        self.use_safe_fill = problem.meta.get("safe_fill", self.use_safe_fill)

        debug_decode: bool = bool(problem.meta.get("debug_decode", False))
        layer_tag = problem.meta.get("layer_idx", "")
        suffix = f"_L{layer_tag:02d}" if isinstance(layer_tag, int) else (
            f"_{layer_tag}" if layer_tag else ""
        )
        if debug_decode:
            _debug_path = pathlib.Path(__file__).parent / f"greedy_logs/greedy_debug{suffix}.txt"
            with open(_debug_path, "w") as _dbg_file:
                with contextlib.redirect_stdout(_dbg_file):
                    entries, node_to_time, decode_meta, final_ownership = self._decode(
                        problem, prep, comp_order
                    )
        else:
            entries, node_to_time, decode_meta, final_ownership = self._decode(
                problem, prep, comp_order
            )

        lb = max(
            prep.bottom_level[nid]
            for nid in problem.dag.nodes
            if not problem.dag.pred.get(nid)
        )
        makespan = decode_meta["makespan"]

        starts: Dict[int, List[str]] = {}
        for nid, e in entries.items():
            starts.setdefault(e.start, []).append(nid)

        pol = get_resource_policy(problem.policy_name)
        steps: List[ScheduleStep] = [
            ScheduleStep(
                t=tt,
                nodes=sorted(starts[tt]),
                meta={"algo": self.name, "policy": pol.name},
            )
            for tt in sorted(starts)
        ]

        # Build ownership snapshot for external validation
        block_ownership_intervals, coupler_ownership_intervals = (
            final_ownership.get_all_intervals()
        )

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={
                "scheduler": self.name,
                "resource_policy": pol.name,
                "makespan": makespan,
                "cp_lower_bound": lb,
                "gap": makespan - lb,
                "num_components": len(prep.components),
                "num_link_nodes": len(prep.link_nodes),
                "component_order": comp_order,
                "block_criticality": dict(prep.block_criticality),
                "node_block_criticality": dict(prep.node_block_criticality),
                "block_ownership_intervals": block_ownership_intervals,
                "coupler_ownership_intervals": coupler_ownership_intervals,
                "entries": {
                    nid: {"start": e.start, "end": e.end}
                    for nid, e in entries.items()
                },
                **decode_meta,
            },
        )

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, problem: SchedulingProblem) -> _Preprocessed:
        dag = problem.dag

        node_duration: Dict[str, int] = {
            nid: self._node_duration(problem, nid) for nid in dag.nodes
        }
        bottom_level = self._compute_bottom_levels(dag, node_duration)

        # ---- connected-component decomposition (undirected adjacency) ----
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
                for v in set(dag.pred.get(u, set())) | set(dag.succ.get(u, set())):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            cid = len(components)
            components.append(comp)
            for nid in comp:
                node_to_component[nid] = cid

        global_indeg: Dict[str, int] = {
            nid: len(dag.pred.get(nid, set())) for nid in dag.nodes
        }

        # ---- link-node metadata (for scheduling priority) ----
        link_nodes: Set[str] = set()
        link_route_depth: Dict[str, int] = {}
        link_route_span: Dict[str, int] = {}
        link_route_width: Dict[str, int] = {}

        for nid, node in dag.nodes.items():
            if node.kind == K_INTERBLOCK_LINK:
                link_nodes.add(nid)
                info = self._extract_link_route_info(node, node_duration[nid])
                link_route_depth[nid] = info["depth"]
                link_route_span[nid] = info["span"]
                link_route_width[nid] = info["width"]

        # ---- block and node criticality ----
        block_criticality: Dict[int, int] = {}
        node_block_criticality: Dict[str, int] = {}

        for nid, node in dag.nodes.items():
            bl = bottom_level[nid]
            for b in node.blocks:
                block_criticality[b] = block_criticality.get(b, 0) + bl

        for nid, node in dag.nodes.items():
            node_block_criticality[nid] = sum(
                block_criticality.get(b, 0) for b in node.blocks
            )

        # ---- per-component role maps ----
        comp_root: Dict[int, str] = {}
        comp_metrics: Dict[int, Dict[str, Any]] = {}
        comp_init_nodes: Dict[int, List[str]] = {}
        comp_link_node: Dict[int, Optional[str]] = {}
        comp_pz_node: Dict[int, Optional[str]] = {}
        comp_xm_node: Dict[int, Optional[str]] = {}
        comp_magic_block: Dict[int, Optional[int]] = {}
        comp_participant_blocks: Dict[int, Set[int]] = {}
        comp_route_blocks: Dict[int, Set[int]] = {}
        comp_route_couplers: Dict[int, List[str]] = {}
        comp_intermediate_blocks: Dict[int, Set[int]] = {}
        comp_is_multiblock: Dict[int, bool] = {}

        for cid, comp in enumerate(components):
            # component root
            roots = [
                nid for nid in comp
                if global_indeg[nid] == 0 and dag.nodes[nid].kind == K_INIT_PIVOT
            ]
            if not roots:
                raise RuntimeError(
                    f"Component {cid} has no independent init_pivot root. "
                    "This violates the expected DAG structure."
                )
            comp_root[cid] = sorted(roots)[0]

            comp_metrics[cid] = {
                "bottom_max": max(bottom_level[n] for n in comp),
                "bottom_sum": sum(bottom_level[n] for n in comp),
                "duration_sum": sum(node_duration[n] for n in comp),
                "block_crit_sum": sum(node_block_criticality[n] for n in comp),
                "link_depth_sum": sum(link_route_depth.get(n, 0) for n in comp),
                "link_width_sum": sum(link_route_width.get(n, 0) for n in comp),
            }

            # classify nodes by role
            init_nids: List[str] = []
            link_nid: Optional[str] = None
            pz_nid: Optional[str] = None
            xm_nid: Optional[str] = None

            for nid in comp:
                kind = dag.nodes[nid].kind
                if kind == K_INIT_PIVOT:
                    init_nids.append(nid)
                elif kind == K_INTERBLOCK_LINK:
                    link_nid = nid
                elif kind == K_MEAS_PARITY_PZ:
                    pz_nid = nid
                elif kind == K_MEAS_MAGIC_X:
                    xm_nid = nid

            # magic block comes from Xm node
            magic_block: Optional[int] = None
            if xm_nid is not None:
                xm_blocks = dag.nodes[xm_nid].blocks
                magic_block = xm_blocks[0] if xm_blocks else None

            # participant blocks = all blocks from init + lc nodes
            participant_blocks: Set[int] = set()
            for nid in comp:
                kind = dag.nodes[nid].kind
                if kind in (K_INIT_PIVOT, K_LOCAL_COUPLE):
                    participant_blocks.update(dag.nodes[nid].blocks)

            # route info from link node
            is_multiblock = link_nid is not None
            if link_nid is not None:
                ln = dag.nodes[link_nid]
                route_blocks: Set[int] = set(ln.blocks)
                route_couplers: List[str] = list(ln.couplers)
            else:
                route_blocks = set(participant_blocks)
                route_couplers = []

            intermediate_blocks: Set[int] = route_blocks - participant_blocks

            comp_init_nodes[cid] = init_nids
            comp_link_node[cid] = link_nid
            comp_pz_node[cid] = pz_nid
            comp_xm_node[cid] = xm_nid
            comp_magic_block[cid] = magic_block
            comp_participant_blocks[cid] = participant_blocks
            comp_route_blocks[cid] = route_blocks
            comp_route_couplers[cid] = route_couplers
            comp_intermediate_blocks[cid] = intermediate_blocks
            comp_is_multiblock[cid] = is_multiblock

        return _Preprocessed(
            components=components,
            node_to_component=node_to_component,
            comp_root=comp_root,
            comp_metrics=comp_metrics,
            bottom_level=bottom_level,
            node_duration=node_duration,
            block_criticality=block_criticality,
            node_block_criticality=node_block_criticality,
            link_nodes=link_nodes,
            link_route_depth=link_route_depth,
            link_route_span=link_route_span,
            link_route_width=link_route_width,
            comp_init_nodes=comp_init_nodes,
            comp_link_node=comp_link_node,
            comp_pz_node=comp_pz_node,
            comp_xm_node=comp_xm_node,
            comp_magic_block=comp_magic_block,
            comp_participant_blocks=comp_participant_blocks,
            comp_route_blocks=comp_route_blocks,
            comp_route_couplers=comp_route_couplers,
            comp_intermediate_blocks=comp_intermediate_blocks,
            comp_is_multiblock=comp_is_multiblock,
        )

    # ------------------------------------------------------------------ #
    # Component ordering                                                   #
    # ------------------------------------------------------------------ #

    def _build_component_order(self, prep: _Preprocessed) -> List[int]:
        """Sort components by criticality (most critical first)."""
        return sorted(
            range(len(prep.components)),
            key=lambda cid: (
                -prep.comp_metrics[cid]["bottom_max"],
                -prep.comp_metrics[cid]["block_crit_sum"],
                -prep.comp_metrics[cid]["bottom_sum"],
                -prep.comp_metrics[cid]["duration_sum"],
                prep.comp_metrics[cid]["link_depth_sum"],
                prep.comp_metrics[cid]["link_width_sum"],
                cid,
            ),
        )

    # ------------------------------------------------------------------ #
    # Ownership check helpers                                              #
    # ------------------------------------------------------------------ #

    def _link_frontier_lb(
        self,
        a_cid: int,
        prep: _Preprocessed,
        dag: Any,
        entries: Dict[str, "ScheduleEntry"],
        t: int,
    ) -> int:
        """
        Lower bound on when in-progress component *a_cid*'s link node can start.

        For each direct predecessor of the link node:
          - if already scheduled: use entries[pred].end  (exact)
          - if not yet scheduled: use t + bottom_level[pred]  (optimistic lower bound)
        Returns max over all predecessors.

        Complexity: O(|preds of link|) — typically 2-4 nodes.
        Reuses: prep.comp_link_node, prep.bottom_level, existing entries dict.
        """
        link_nid = prep.comp_link_node[a_cid]
        if link_nid is None:
            return t
        lb = t
        for pred in dag.pred.get(link_nid, set()):
            if pred in entries:
                lb = max(lb, entries[pred].end)
            else:
                lb = max(lb, t + prep.bottom_level[pred])
        return lb

    def _ownership_can_start(
        self,
        nid: str,
        node: Any,
        cid: int,
        t: int,
        ownership: BlockOwnershipTracker,
        prep: _Preprocessed,
        dag: Any,
        entries: Dict[str, "ScheduleEntry"],
        in_progress_multiblock: Set[int],
        comp_init_started: Set[int],
        dbg: bool,
        use_phase2: bool = True,
    ) -> bool:
        """
        Returns True iff the ownership tracker allows this node to start.
        Only init_pivot and interblock_link need active ownership checks —
        all other node kinds operate on resources already owned by their
        component (enforced by DAG dependencies and prior claims).

        Phase 1 — Deadlock prevention:
          Before admitting the first init_pivot of multiblock component B,
          check every in-progress multiblock component A for a hold-and-wait
          cycle (R_B ∩ P_A ≠ ∅  AND  R_A ∩ P_B ≠ ∅).

        Phase 2 — Safe-fill exception:
          Even if a hold-and-wait would occur, B may be admitted safely if
          its projected Xm completion (t + bottom_max_B) is at or before the
          lower-bound start of A's link node.  In that case B finishes and
          releases P_B before A ever needs to traverse its route.
          Uses _link_frontier_lb (O(|preds of link|)) and the precomputed
          component_metrics["bottom_max"] — no new stored state.
        """
        kind = node.kind

        if kind == K_INIT_PIVOT:
            # ---- Phase 1 + 2: deadlock prevention with safe-fill exception ----
            is_first_init = cid not in comp_init_started
            if is_first_init and prep.comp_is_multiblock[cid]:
                P_B = prep.comp_participant_blocks[cid]
                R_B = prep.comp_route_blocks[cid]
                # Estimate B's Xm completion from current time.
                # bottom_max = longest path in component = time until B is fully done.
                b_xm_est = t + prep.comp_metrics[cid]["bottom_max"]

                for a_cid in in_progress_multiblock:
                    P_A = prep.comp_participant_blocks[a_cid]
                    R_A = prep.comp_route_blocks[a_cid]
                    if not (R_B & P_A) or not (R_A & P_B):
                        continue  # no hold-and-wait risk with this A

                    # Phase 2: is B guaranteed to finish before A's link starts?
                    # Guard: only applies when P_A and P_B are disjoint AND B's route
                    # does not cross A's participant blocks.
                    #
                    # R_B ∩ P_A ≠ ∅ means B's route passes through blocks A has claimed
                    # with end=INF (until A's PZ). B's link will be blocked waiting for
                    # A to release those blocks, so b_xm_est is not an independent
                    # completion estimate — B cannot finish before A's link in that case.
                    # Since we are always inside R_B ∩ P_A ≠ ∅ here (it is the trigger
                    # for reaching this code), Phase 2 is disabled for all hold-and-wait
                    # cases that involve route/participant overlap, preventing circular
                    # ownership deadlock on sparse (long-route) hardware graphs.
                    if use_phase2 and not (P_A & P_B) and not (R_B & P_A):
                        a_link_lb = self._link_frontier_lb(a_cid, prep, dag, entries, t)
                        if b_xm_est <= a_link_lb:
                            if dbg:
                                print(
                                    f"    [PHASE2_SAFE] {nid} (first-init cid={cid}) "
                                    f"admitted — B finishes by t={b_xm_est} ≤ "
                                    f"A(cid={a_cid}) link lb={a_link_lb}  "
                                    f"R_B∩P_A={sorted(R_B & P_A)}  "
                                    f"R_A∩P_B={sorted(R_A & P_B)}"
                                )
                            continue  # safe: B will release P_B before A needs it

                    # Genuine deadlock risk (or Phase 2 disabled) — block B.
                    if dbg:
                        print(
                            f"    [DEADLOCK_PREVENT] {nid} (first-init cid={cid}) "
                            f"blocked — hold-and-wait with cid={a_cid}  "
                            + (
                                f"b_xm_est={b_xm_est} > "
                                f"a_link_lb={self._link_frontier_lb(a_cid, prep, dag, entries, t)}  "
                                if use_phase2 else "(phase2 disabled)  "
                            )
                            + f"R_B∩P_A={sorted(R_B & P_A)}  "
                            f"R_A∩P_B={sorted(R_A & P_B)}"
                        )
                    return False

            # ---- block ownership check ----
            for b in node.blocks:
                if not ownership.can_claim_block(b, t, cid):
                    if dbg:
                        reason = ownership.block_conflict_info(b, t, cid)
                        print(
                            f"    [OWNERSHIP_BLOCK] {nid} (init) "
                            f"blocked — {reason}"
                        )
                    return False
            return True

        if kind == K_INTERBLOCK_LINK:
            # All route blocks (including intermediates) and couplers must be free.
            for rb in prep.comp_route_blocks[cid]:
                if not ownership.can_claim_block(rb, t, cid):
                    if dbg:
                        reason = ownership.block_conflict_info(rb, t, cid)
                        print(
                            f"    [OWNERSHIP_BLOCK] {nid} (link) "
                            f"route blocked — {reason}"
                        )
                    return False
            for rc in prep.comp_route_couplers[cid]:
                if not ownership.can_claim_coupler(rc, t, cid):
                    if dbg:
                        reason = ownership.coupler_conflict_info(rc, t, cid)
                        print(
                            f"    [OWNERSHIP_COUPLER] {nid} (link) "
                            f"route blocked — {reason}"
                        )
                    return False
            return True

        # K_LOCAL_COUPLE, K_MEAS_PARITY_PZ, K_MEAS_MAGIC_X, K_FRAME_UPDATE:
        # resources already owned by this component via prior claims.
        return True

    def _ownership_apply(
        self,
        nid: str,
        node: Any,
        cid: int,
        t: int,
        end: int,
        ownership: BlockOwnershipTracker,
        prep: _Preprocessed,
        in_progress_multiblock: Set[int],
        comp_init_started: Set[int],
        dbg: bool,
    ) -> None:
        """
        Update the ownership tracker after a node is committed to the schedule.
        Also maintains admission tracking sets for deadlock prevention.
        """
        kind = node.kind

        if kind == K_INIT_PIVOT:
            # Claim each block with an open-ended interval (end set by PZ or Xm later).
            for b in node.blocks:
                ownership.claim_block(b, t, cid)
                if dbg:
                    print(
                        f"    [OWN+] {nid}  init_pivot "
                        f"claimed block {b}  [{t}, INF)  cid={cid}"
                    )
            # Track admission: first init_pivot of this component.
            if cid not in comp_init_started:
                comp_init_started.add(cid)
                if prep.comp_is_multiblock[cid]:
                    in_progress_multiblock.add(cid)
                    if dbg:
                        print(
                            f"    [ADMIT] cid={cid} admitted as in-progress multiblock  "
                            f"P={sorted(prep.comp_participant_blocks[cid])}  "
                            f"R={sorted(prep.comp_route_blocks[cid])}"
                        )

        elif kind == K_INTERBLOCK_LINK:
            # Claim intermediate blocks (terminal blocks already claimed by their inits).
            # Claim all route couplers.
            for rb in prep.comp_intermediate_blocks[cid]:
                ownership.claim_block(rb, t, cid)
                if dbg:
                    print(
                        f"    [OWN+] {nid}  link "
                        f"claimed intermediate block {rb}  [{t}, INF)  cid={cid}"
                    )
            for rc in prep.comp_route_couplers[cid]:
                ownership.claim_coupler(rc, t, cid)
                if dbg:
                    print(
                        f"    [OWN+] {nid}  link "
                        f"claimed coupler {rc!r}  [{t}, INF)  cid={cid}"
                    )

        elif kind == K_MEAS_PARITY_PZ:
            # Finalise non-magic participant blocks and all route items at pz_end.
            magic = prep.comp_magic_block[cid]
            pz_end = end  # = t + duration

            # Non-magic participant blocks: release at pz_end.
            for b in prep.comp_participant_blocks[cid]:
                if b != magic:
                    ownership.update_block_end(b, pz_end, cid)
                    if dbg:
                        print(
                            f"    [OWN=] {nid}  PZ "
                            f"finalized non-magic block {b}  end→{pz_end}  cid={cid}"
                        )

            # Intermediate route blocks: release at pz_end.
            for rb in prep.comp_intermediate_blocks[cid]:
                ownership.update_block_end(rb, pz_end, cid)
                if dbg:
                    print(
                        f"    [OWN=] {nid}  PZ "
                        f"finalized intermediate block {rb}  end→{pz_end}  cid={cid}"
                    )

            # Route couplers: release at pz_end.
            for rc in prep.comp_route_couplers[cid]:
                ownership.update_coupler_end(rc, pz_end, cid)
                if dbg:
                    print(
                        f"    [OWN=] {nid}  PZ "
                        f"finalized coupler {rc!r}  end→{pz_end}  cid={cid}"
                    )

            if dbg and magic is not None:
                print(
                    f"    [OWN~] {nid}  PZ "
                    f"magic block {magic} stays locked (until Xm)  cid={cid}"
                )

        elif kind == K_MEAS_MAGIC_X:
            # Finalise magic block at xm_end.
            magic = prep.comp_magic_block[cid]
            xm_end = end  # = t + duration
            if magic is not None:
                ownership.update_block_end(magic, xm_end, cid)
                if dbg:
                    print(
                        f"    [OWN=] {nid}  Xm "
                        f"finalized magic block {magic}  end→{xm_end}  cid={cid}"
                    )
            # Component is now fully committed; release deadlock-prevention hold.
            if cid in in_progress_multiblock:
                in_progress_multiblock.discard(cid)
                if dbg:
                    print(
                        f"    [COMPLETE] cid={cid} Xm done — "
                        f"removed from in_progress_multiblock"
                    )

        # K_LOCAL_COUPLE and K_FRAME_UPDATE: no ownership updates needed.

    # ------------------------------------------------------------------ #
    # Decode (list-schedule with ownership enforcement)                   #
    # ------------------------------------------------------------------ #

    def _decode(
        self,
        problem: SchedulingProblem,
        prep: _Preprocessed,
        comp_order: List[int],
    ) -> Tuple[
        Dict[str, ScheduleEntry],
        Dict[str, int],
        Dict[str, Any],
        BlockOwnershipTracker,
    ]:
        dag = problem.dag
        hw = problem.hw
        pol = get_resource_policy(problem.policy_name)
        tracker = HardwareTracker(hw=hw, policy=pol)
        ownership = BlockOwnershipTracker()
        dbg: bool = problem.meta.get("debug_decode", False)

        # Deadlock-prevention state:
        #   comp_init_started     — components with at least one init_pivot committed
        #   in_progress_multiblock — multiblock components admitted but Xm not yet done
        comp_init_started: Set[int] = set()
        in_progress_multiblock: Set[int] = set()

        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}

        indeg_left: Dict[str, int] = {
            nid: len(dag.pred.get(nid, set())) for nid in dag.nodes
        }
        ready: Set[str] = {nid for nid, d in indeg_left.items() if d == 0}
        active: List[Tuple[int, str]] = []

        t = int(problem.meta.get("start_time", 0))

        comp_rank: Dict[int, int] = {cid: rank for rank, cid in enumerate(comp_order)}

        def node_priority(nid: str) -> Tuple:
            cid = prep.node_to_component[nid]
            dur = prep.node_duration[nid]
            bottom = prep.bottom_level[nid]
            block_crit = prep.node_block_criticality[nid]
            is_link = nid in prep.link_nodes
            link_depth = prep.link_route_depth.get(nid, 0)
            link_width = prep.link_route_width.get(nid, 0)
            link_span = prep.link_route_span.get(nid, 0)
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

        if dbg:
            print(f"\n{'='*70}")
            print(
                f"[greedy_critical] START  "
                f"nodes={len(dag.nodes)}  components={len(prep.components)}"
            )
            print(f"  comp_order={comp_order}")
            print(
                f"  multiblock_components="
                f"{[c for c in range(len(prep.components)) if prep.comp_is_multiblock[c]]}"
            )
            for cid in range(len(prep.components)):
                mb = prep.comp_is_multiblock[cid]
                print(
                    f"  cid={cid:3d}  multiblock={mb}"
                    f"  magic_block={prep.comp_magic_block[cid]}"
                    f"  participant_blocks={sorted(prep.comp_participant_blocks[cid])}"
                    f"  route_blocks={sorted(prep.comp_route_blocks[cid])}"
                    f"  intermediate={sorted(prep.comp_intermediate_blocks[cid])}"
                    f"  route_couplers={prep.comp_route_couplers[cid]}"
                )
            print(f"{'='*70}")

        # ---- main scheduling loop ----
        while len(entries) < len(dag.nodes):

            # Release finished nodes and unlock their DAG children.
            while active and active[0][0] <= t:
                _, finished = heapq.heappop(active)
                add_children_of(finished)
                if dbg:
                    fin_cid = prep.node_to_component[finished]
                    print(
                        f"  [t={t}] DONE  {finished}"
                        f"  kind={dag.nodes[finished].kind}  cid={fin_cid}"
                    )

            if ready:
                # Build priority queue for this time step.
                pq: List[Tuple[Tuple, str]] = []
                for nid in ready:
                    heapq.heappush(pq, (node_priority(nid), nid))

                if dbg:
                    print(f"\n[t={t}] ready={len(ready)} nodes")
                    ownership.print_state(t)
                    for pri, nid in sorted(pq):
                        cid = prep.node_to_component[nid]
                        print(
                            f"  READY  {nid:40s}"
                            f"  kind={dag.nodes[nid].kind:20s}"
                            f"  cid={cid:3d}  bl={prep.bottom_level[nid]:4d}"
                        )

                started_any = False
                blocked: List[str] = []

                while pq:
                    _, nid = heapq.heappop(pq)
                    if nid in entries or nid not in ready:
                        continue

                    node = dag.nodes[nid]
                    cid = prep.node_to_component[nid]
                    dur = prep.node_duration[nid]

                    if dur < 0:
                        raise ValueError(
                            f"Node {nid} has negative duration {dur}"
                        )

                    # ---- zero-duration (frame update) ----
                    if dur == 0:
                        entries[nid] = ScheduleEntry(nid=nid, start=t, end=t)
                        node_to_time[nid] = t
                        ready.remove(nid)
                        add_children_of(nid)
                        started_any = True
                        if dbg:
                            print(
                                f"  [t={t}] INSTANT  {nid}"
                                f"  kind={node.kind}  cid={cid}"
                            )
                        continue

                    end = t + dur

                    # ---- ownership check ----
                    if not self._ownership_can_start(
                        nid, node, cid, t, ownership, prep,
                        dag, entries,
                        in_progress_multiblock, comp_init_started, dbg,
                        use_phase2=self.use_safe_fill,
                    ):
                        blocked.append(nid)
                        continue

                    # ---- hardware resource check ----
                    if not tracker.can_reserve(node, t, end):
                        if dbg:
                            print(
                                f"    [HW_BLOCK] {nid}  kind={node.kind}  cid={cid}"
                                f"  blocks={node.blocks}  couplers={node.couplers}"
                            )
                        blocked.append(nid)
                        continue

                    # ---- commit ----
                    tracker.reserve(node, t, end)
                    entries[nid] = ScheduleEntry(nid=nid, start=t, end=end)
                    node_to_time[nid] = t
                    heapq.heappush(active, (end, nid))
                    ready.remove(nid)
                    started_any = True

                    if dbg:
                        print(
                            f"  [t={t}] START  {nid}"
                            f"  kind={node.kind}  cid={cid}"
                            f"  dur={dur}  end={end}"
                            f"  blocks={node.blocks}"
                        )

                    # Apply ownership update for this node.
                    self._ownership_apply(
                        nid, node, cid, t, end, ownership, prep,
                        in_progress_multiblock, comp_init_started, dbg,
                    )

                for nid in blocked:
                    ready.add(nid)

                if started_any:
                    t = min(t + 1, active[0][0]) if active else t + 1
                    continue

            if active:
                t = active[0][0]
                continue

            if ready:
                if dbg:
                    print(f"\n[t={t}] DEADLOCK — ready nodes cannot be scheduled:")
                    ownership.print_state(t, label="DEADLOCK")
                    for nid in sorted(ready):
                        cid = prep.node_to_component[nid]
                        node = dag.nodes[nid]
                        print(
                            f"  {nid:40s}  kind={node.kind:20s}"
                            f"  cid={cid:3d}  blocks={node.blocks}"
                            f"  couplers={node.couplers}"
                        )
                raise RuntimeError(
                    f"Ready nodes exist at t={t} but none can be scheduled under "
                    f"policy '{pol.name}'. Possible resource/ownership deadlock."
                )
            raise RuntimeError(
                "No active and no ready but schedule incomplete."
            )

        makespan = 0 if not entries else max(e.end for e in entries.values())

        if dbg:
            print(f"\n[greedy_critical] DONE  makespan={makespan}")
            print("  Final schedule (sorted by start):")
            for nid in sorted(entries, key=lambda n: (entries[n].start, n)):
                e = entries[nid]
                cid = prep.node_to_component[nid]
                print(
                    f"    [{e.start:4d}..{e.end:4d}]  cid={cid:3d}"
                    f"  {dag.nodes[nid].kind:20s}  {nid}"
                )
            ownership.print_state(makespan, label="FINAL")

        return entries, node_to_time, {"makespan": makespan}, ownership

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _node_duration(self, problem: SchedulingProblem, nid: str) -> int:
        if nid in problem.duration_override:
            return int(problem.duration_override[nid])
        d = getattr(problem.dag.nodes[nid], "duration", 1)
        return int(d) if d is not None else 1

    def _extract_link_route_info(self, node: Any, fallback_duration: int) -> Dict[str, Any]:
        blocks = tuple(sorted(set(getattr(node, "blocks", []) or [])))
        couplers = tuple(dict.fromkeys(getattr(node, "couplers", []) or []))
        duration = getattr(node, "duration", None)
        duration = fallback_duration if duration is None else int(duration)
        return {
            "blocks": blocks,
            "couplers": couplers,
            "depth": max(0, duration),
            "span": len(blocks),
            "width": len(couplers),
        }

    def _compute_bottom_levels(
        self, dag: Any, node_duration: Dict[str, int]
    ) -> Dict[str, int]:
        """Weighted bottom level: BL(n) = dur(n) + max(BL(child))."""
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
