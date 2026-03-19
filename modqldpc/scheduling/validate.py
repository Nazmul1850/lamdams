# modqldpc/scheduling/validate.py
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from modqldpc.lowering.ir import (
    K_INIT_PIVOT,
    K_INTERBLOCK_LINK,
    K_LOCAL_COUPLE,
    K_MEAS_MAGIC_X,
    K_MEAS_PARITY_PZ,
)

from .policy import get_resource_policy
from .resources import new_step_state
from .types import Schedule, SchedulingProblem


# ============================================================
# Standard resource + precedence validation
# ============================================================

def validate_schedule(problem: SchedulingProblem, sched: Schedule) -> None:
    """
    Checks:
      (1) Every DAG node appears exactly once.
      (2) Precedence constraints are satisfied.
      (3) No per-step resource violations (via the configured policy).
    """
    dag = problem.dag
    hw = problem.hw
    pol = get_resource_policy(problem.policy_name)

    # ---- (1) node coverage ----
    scheduled: List[str] = []
    for st in sched.steps:
        scheduled.extend(st.nodes)

    if len(scheduled) != len(set(scheduled)):
        raise ValueError("Schedule contains duplicate node IDs.")

    dag_nodes = set(dag.nodes.keys())
    sch_nodes = set(scheduled)
    missing = dag_nodes - sch_nodes
    extra = sch_nodes - dag_nodes
    if missing:
        raise ValueError(f"Schedule missing nodes: {sorted(list(missing))[:20]}")
    if extra:
        raise ValueError(f"Schedule has unknown nodes: {sorted(list(extra))[:20]}")

    # ---- (2) precedence ----
    time: Dict[str, int] = {}
    duration: Dict[str, int] = {nid: dag.nodes[nid].duration for nid in dag.nodes}
    for st in sched.steps:
        for nid in st.nodes:
            time[nid] = st.t

    for u, vs in dag.succ.items():
        tu = time[u]
        for v in vs:
            tv = time[v]
            if not (tu + duration[u] <= tv):
                raise ValueError(
                    f"Precedence violated: {u} → {v}  "
                    f"t[{u}]={tu}+{duration[u]}  t[{v}]={tv}"
                )

    # ---- (3) per-step resource constraints ----
    for st in sched.steps:
        state = new_step_state()
        for nid in st.nodes:
            node = dag.nodes[nid]
            claim = pol.claim_for_node(node, hw)
            if not pol.can_apply(state, claim, hw):
                raise ValueError(
                    f"Resource violation at step t={st.t}: "
                    f"node '{nid}' not placeable under policy '{pol.name}'."
                )
            pol.apply(state, claim, hw)


# ============================================================
# Ownership / surgery-lock validation
# ============================================================

def validate_ownership(problem: SchedulingProblem, sched: Schedule) -> None:
    """
    Checks the three surgery-lock invariants:

      (1) Preparation lock — no two components have overlapping init-through-
          PZ/Xm windows on the same block.
      (2) Route lock — no two components have overlapping link-through-PZ
          windows on the same route block or coupler.
      (3) Magic-block lock — no two components have overlapping link-through-
          Xm windows on the same magic block.

    Requires sched.meta["entries"] = {nid: {start, end}} (produced by
    greedy_critical and sa_scheduler).

    Raises ValueError with a human-readable description on the first conflict.
    """
    dag = problem.dag
    entries_meta: Dict[str, Dict[str, int]] = sched.meta.get("entries", {})
    if not entries_meta:
        raise ValueError(
            "validate_ownership requires sched.meta['entries'] "
            "(start/end per node). Run greedy_critical or sa_scheduler."
        )

    def node_start(nid: str) -> int:
        return entries_meta[nid]["start"]

    def node_end(nid: str) -> int:
        return entries_meta[nid]["end"]

    # ---- reconstruct components ----
    components: List[Set[str]] = []
    node_to_comp: Dict[str, int] = {}
    seen: Set[str] = set()

    for start_nid in dag.nodes:
        if start_nid in seen:
            continue
        comp: Set[str] = set()
        stack = [start_nid]
        seen.add(start_nid)
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
            node_to_comp[nid] = cid

    # ---- collect per-component role nodes ----
    # For each component we need: init_nids, link_nid, pz_nid, xm_nid

    # Intervals to accumulate per block / coupler:
    #   "prep"   block  → (start, end, cid, note)
    #   "route"  block  → (start, end, cid, note)
    #   "route"  coupler→ (start, end, cid, note)

    # block_prep_intervals[b] = [(start, end, cid, desc), ...]
    block_prep_intervals: Dict[int, List[Tuple[int, int, int, str]]] = {}
    # block_route_intervals[b] = [(start, end, cid, desc), ...]
    block_route_intervals: Dict[int, List[Tuple[int, int, int, str]]] = {}
    # coupler_route_intervals[c] = [(start, end, cid, desc), ...]
    coupler_route_intervals: Dict[str, List[Tuple[int, int, int, str]]] = {}

    for cid, comp in enumerate(components):
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

        if xm_nid is None:
            continue  # incomplete / frame-only component — skip

        xm_end = node_end(xm_nid)
        magic_block: Optional[int] = (
            dag.nodes[xm_nid].blocks[0] if dag.nodes[xm_nid].blocks else None
        )

        pz_end = node_end(pz_nid) if pz_nid else xm_end

        is_multiblock = link_nid is not None

        # Participant blocks (from init + lc nodes)
        participant_blocks: Set[int] = set()
        for nid in comp:
            kind = dag.nodes[nid].kind
            if kind in (K_INIT_PIVOT, K_LOCAL_COUPLE):
                participant_blocks.update(dag.nodes[nid].blocks)

        # Route info
        route_blocks: Set[int] = set()
        route_couplers: List[str] = []
        if link_nid:
            route_blocks = set(dag.nodes[link_nid].blocks)
            route_couplers = list(dag.nodes[link_nid].couplers)
        else:
            route_blocks = set(participant_blocks)

        # intermediate_blocks = route_blocks - participant_blocks (used implicitly via route_blocks)

        # ---- (1) preparation lock intervals ----
        # For each participant block, find earliest init that touches it.
        block_first_init: Dict[int, int] = {}
        for nid in init_nids:
            t_start = node_start(nid)
            for b in dag.nodes[nid].blocks:
                if b not in block_first_init or t_start < block_first_init[b]:
                    block_first_init[b] = t_start

        for b, t_init in block_first_init.items():
            if b == magic_block:
                t_lock_end = xm_end
                note = f"magic  cid={cid}"
            elif is_multiblock:
                t_lock_end = pz_end
                note = f"non-magic  cid={cid}"
            else:
                # single-block: the one block is the magic block
                t_lock_end = xm_end
                note = f"single-block  cid={cid}"

            block_prep_intervals.setdefault(b, []).append(
                (t_init, t_lock_end, cid, f"prep({note})")
            )

        # ---- (2) route lock intervals (link → PZ) ----
        if link_nid:
            t_link = node_start(link_nid)
            for rb in route_blocks:
                block_route_intervals.setdefault(rb, []).append(
                    (t_link, pz_end, cid, f"route  cid={cid}")
                )
            for rc in route_couplers:
                coupler_route_intervals.setdefault(rc, []).append(
                    (t_link, pz_end, cid, f"route  cid={cid}")
                )

        # ---- (3) magic-block lock (link → Xm) already covered by prep ----
        # The prep interval for the magic block already runs to xm_end.

    # ---- check for overlaps ----
    errors: List[str] = []

    def _check_intervals(
        name: str,
        resource_id: str,
        intervals: List[Tuple[int, int, int, str]],
    ) -> None:
        sorted_iv = sorted(intervals, key=lambda x: x[0])
        for i in range(len(sorted_iv)):
            s1, e1, cid1, desc1 = sorted_iv[i]
            for j in range(i + 1, len(sorted_iv)):
                s2, e2, cid2, desc2 = sorted_iv[j]
                if cid1 == cid2:
                    continue
                if s2 >= e1:
                    break  # sorted by start; no later pair can overlap
                # overlap: [s1, e1) ∩ [s2, e2) ≠ ∅  AND cid1 ≠ cid2
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                errors.append(
                    f"Ownership conflict on {name} {resource_id!r}:\n"
                    f"  cid={cid1} [{s1},{e1})  {desc1}\n"
                    f"  cid={cid2} [{s2},{e2})  {desc2}\n"
                    f"  overlap [{overlap_start}, {overlap_end})"
                )

    for b, ivs in block_prep_intervals.items():
        _check_intervals("block", str(b), ivs)
    for b, ivs in block_route_intervals.items():
        _check_intervals("route-block", str(b), ivs)
    for c, ivs in coupler_route_intervals.items():
        _check_intervals("coupler", c, ivs)

    if errors:
        header = f"validate_ownership found {len(errors)} conflict(s):\n"
        raise ValueError(header + "\n".join(errors))
