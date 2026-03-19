"""
Minimal reproduction of the CP-SAT infeasibility with two overlapping
multiblock components on a 3x3 grid.

  cid=0: magic_block=2, participant=[2,3,6,7,8,9], intermediate=[5]
  cid=1: magic_block=1, participant=[1,3,4,5,7,8],  intermediate=[2]

Shared participant blocks: {3, 7, 8}
cid=1's route must pass through block 2 (owned by cid=0 as magic block).
cid=0's route must pass through block 5 (owned by cid=1 as participant block).
"""
from __future__ import annotations

from modqldpc.lowering.ir import (
    ExecDAG,
    ExecNode,
    K_INIT_PIVOT, K_LOCAL_COUPLE, K_INTERBLOCK_LINK,
    K_MEAS_PARITY_PZ, K_MEAS_MAGIC_X, K_FRAME_UPDATE,
    node_init_pivot, node_local_couple, node_interblock_link,
    node_meas_parity_PZ, node_meas_magic_X, node_frame_update,
    ClassicalKey,
)
from modqldpc.mapping.model import HardwareGraph, GridTopology, GraphFactory
from modqldpc.core.types import PauliAxis
from modqldpc.scheduling.algos.cp_sat_scheduling import CPSATScheduler
from modqldpc.scheduling.types import SchedulingProblem


# ── Hardware: 3×3 grid, blocks 1-9 ───────────────────────────────────────────
def build_hw() -> HardwareGraph:
    factory = GraphFactory(default_num_logicals=11, default_port_capacity=1)
    hw = factory.build(
        topology=GridTopology(rows=3, cols=3),
        block_ids=list(range(1, 10)),
        coupler_capacity=1,
    )
    return hw


# ── Dummy PauliAxis for meas nodes ────────────────────────────────────────────
_AXIS = PauliAxis(sign=1, tensor="Z")


def _add_component(
    dag: ExecDAG,
    *,
    tag: str,                   # "c0" or "c1"
    participant_blocks: list,
    magic_block: int,
    link_blocks: list,          # all blocks the link node touches
    link_couplers: list,
    route_paths: list,          # per-source path lists (list of lists)
    source_blocks: list,
) -> None:
    """Build: init→lc (per block) → link → PZ → Xm → FCL/FPA."""

    # ── inits + local-couples ──────────────────────────────────────────────
    lc_nids = []
    for b in participant_blocks:
        nid_init = f"init_{tag}_B{b}"
        nid_lc   = f"lc_{tag}_B{b}"
        dag.add_node(node_init_pivot(nid_init, b, duration=1))
        dag.add_node(node_local_couple(nid_lc, b,
                                       local_ops={}, pivot_id=f"piv_{tag}_B{b}",
                                       duration=1))
        dag.add_edge(nid_init, nid_lc)
        lc_nids.append(nid_lc)

    # ── interblock link ────────────────────────────────────────────────────
    nid_link = f"link_{tag}"
    link_dur = max(len(p) - 1 for p in route_paths)  # longest path
    dag.add_node(node_interblock_link(
        nid_link,
        blocks=link_blocks,
        couplers=link_couplers,
        duration=max(link_dur, 1),
        meta={
            "magic_block": magic_block,
            "source_blocks": source_blocks,
            "route_paths": route_paths,
            "participant_blocks": participant_blocks,
        },
    ))
    for lc in lc_nids:
        dag.add_edge(lc, nid_link)

    # ── parity measurement ────────────────────────────────────────────────
    nid_pz = f"PZ_{tag}"
    dag.add_node(node_meas_parity_PZ(
        nid_pz,
        pauli=_AXIS,
        magic_id=f"m_{tag}",
        out_key=f"bPZ_{tag}",
        blocks=participant_blocks,
        couplers=link_couplers,
        duration=1,
    ))
    dag.add_edge(nid_link, nid_pz)

    # ── Clifford frame update (instant, depends on PZ) ─────────────────────
    nid_fcl = f"FCL_{tag}"
    dag.add_node(node_frame_update(
        nid_fcl,
        update_kind="clifford_pi4",
        depends_on=f"bPZ_{tag}",
        axis=_AXIS,
        duration=0,
    ))
    dag.add_edge(nid_pz, nid_fcl)

    # ── magic-X measurement ───────────────────────────────────────────────
    nid_xm = f"Xm_{tag}"
    dag.add_node(node_meas_magic_X(
        nid_xm,
        magic_id=f"m_{tag}",
        out_key=f"bXm_{tag}",
        block=magic_block,
        duration=1,
    ))
    dag.add_edge(nid_pz, nid_xm)

    # ── Pauli frame update (instant, depends on Xm) ────────────────────────
    nid_fpa = f"FPA_{tag}"
    dag.add_node(node_frame_update(
        nid_fpa,
        update_kind="pauli",
        depends_on=f"bXm_{tag}",
        axis=_AXIS,
        duration=0,
    ))
    dag.add_edge(nid_xm, nid_fpa)


def build_dag() -> ExecDAG:
    dag = ExecDAG()

    # cid=0: magic=2, participants=[2,3,6,7,8,9]
    # routes: 3→2, 6→3→2, 7→8→5→2, 8→5→2, 9→8→5→2
    _add_component(
        dag,
        tag="c0",
        participant_blocks=[2, 3, 6, 7, 8, 9],
        magic_block=2,
        link_blocks=[2, 3, 5, 6, 7, 8, 9],
        link_couplers=["c_2_3", "c_3_6", "c_7_8", "c_5_8", "c_2_5", "c_8_9"],
        source_blocks=[3, 6, 7, 8, 9],
        route_paths=[
            [3, 2],
            [6, 3, 2],
            [7, 8, 5, 2],
            [8, 5, 2],
            [9, 8, 5, 2],
        ],
    )

    # cid=1: magic=1, participants=[1,3,4,5,7,8]
    # routes: 3→2→1, 4→1, 5→2→1, 7→4→1, 8→5→2→1
    _add_component(
        dag,
        tag="c1",
        participant_blocks=[1, 3, 4, 5, 7, 8],
        magic_block=1,
        link_blocks=[1, 2, 3, 4, 5, 7, 8],
        link_couplers=["c_2_3", "c_1_2", "c_1_4", "c_2_5", "c_4_7", "c_5_8"],
        source_blocks=[3, 4, 5, 7, 8],
        route_paths=[
            [3, 2, 1],
            [4, 1],
            [5, 2, 1],
            [7, 4, 1],
            [8, 5, 2, 1],
        ],
    )

    return dag


if __name__ == "__main__":
    hw  = build_hw()
    dag = build_dag()

    print(f"Nodes: {len(dag.nodes)}")
    print(f"Edges: {sum(len(v) for v in dag.succ.values())}")

    problem = SchedulingProblem(
        dag=dag,
        hw=hw,
        seed=0,
        policy_name="incident_coupler_blocks_local",
        meta={
            "cp_sat_log": True,
            "cp_sat_route_alternatives": True,
            "cp_sat_time_limit": 30.0,
            "layer_idx": "infeas_test",
        },
    )

    scheduler = CPSATScheduler()
    try:
        schedule = scheduler.solve(problem)
        print(f"\nSchedule found! makespan={schedule.depth()}")
        for step in schedule.steps:
            print(f"  t={step.t:3d}  {step.nodes}")
    except RuntimeError as e:
        print(f"\nCP-SAT FAILED: {e}")
