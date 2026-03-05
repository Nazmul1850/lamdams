# modqldpc/scheduling/validate.py
from __future__ import annotations
from typing import Dict

from .types import SchedulingProblem, Schedule
from .resources import new_step_state
from .policy import get_resource_policy


def validate_schedule(problem: SchedulingProblem, sched: Schedule) -> None:
    dag = problem.dag
    hw = problem.hw
    pol = get_resource_policy(problem.policy_name)

    # all nodes scheduled exactly once
    scheduled = []
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

    # precedence constraints
    time: Dict[str, int] = {}
    for st in sched.steps:
        for nid in st.nodes:
            time[nid] = st.t

    for u, vs in dag.succ.items():
        tu = time[u]
        for v in vs:
            tv = time[v]
            if not (tu < tv):
                raise ValueError(f"Precedence violated: {u} -> {v} but t[{u}]={tu}, t[{v}]={tv}")

    # resource constraints step-by-step
    for st in sched.steps:
        state = new_step_state()
        for nid in st.nodes:
            node = dag.nodes[nid]
            claim = pol.claim_for_node(node, hw)
            if not pol.can_apply(state, claim, hw):
                raise ValueError(
                    f"Resource violation at step t={st.t}: node '{nid}' not placeable under policy '{pol.name}'."
                )
            pol.apply(state, claim, hw)