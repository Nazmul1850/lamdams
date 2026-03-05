# modqldpc/lowering/magic_gadget.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .ir import (
    ExecDAG,
    count_init_pivots,
    node_init_pivot,
    node_local_couple,
    node_interblock_link,
    node_meas_parity_PZ,
    node_meas_magic_X,
    node_frame_update,
)
from .plans import RotationLoweringPlan, LocalMeasurePlan
from .policy import LoweringPolicies


@dataclass(frozen=True)
class GadgetLoweringResult:
    """
    Returned for debugging / later frame rewrite integration.
    """
    parity_node: str
    xm_node: str
    frame_cliff_node: str
    frame_pauli_node: str
    produced_bPZ: str
    produced_bXm: str


def emit_pi8_gadget(
    *,
    dag: ExecDAG,
    plan: RotationLoweringPlan,
    policies: LoweringPolicies,
) -> GadgetLoweringResult:
    """
    Emit nodes/edges for a single π/8 Pauli-product rotation using the standard
    two-measurement gadget:
      1) measure (P ⊗ Z_m) -> bPZ  (parity)
      2) measure X_m -> bXm
    with frame updates:
      - if bPZ=1 => apply P_{π/4} (Clifford)  (represented as frame_update)
      - if bXm=1 => apply P_{π/2}=P (Pauli)   (represented as frame_update)

    Dependencies:
      local coupling + interblock link -> parity
      parity -> (frame_cliff_update) and parity -> X_m
      X_m -> (frame_pauli_update)
    """

    layer = plan.layer
    ridx = plan.ridx
    namer = policies.namer

    # --- 0) (optional) pivot init + local coupling tasks per block ---
    local_tail_nodes: List[str] = []

    for lp in plan.local_plans:
        # pivot init (always for now; can be policy-controlled later)
        piv = namer.pivot_id(layer, ridx, lp.block)
        
        last = None
        # local coupling sequence (native-aware)
        # lp.sequence is a list of native primitives (each primitive is ops dict)
        for k, prim_ops in enumerate(lp.sequence if lp.sequence else [lp.target.ops]):
            count = count_init_pivots(dag, layer=layer, ridx=ridx, block=lp.block)
            nid_init = namer.nid("init", layer, ridx, suffix=f"B{lp.block}_c{count}")
            dag.add_node(node_init_pivot(nid_init, lp.block, duration=1, meta={"pivot_id": piv}))
            if last:
                dag.add_edge(last, nid_init)
            last = nid_init
            nid_lc = namer.nid("lc", layer, ridx, suffix=f"B{lp.block}_c{count}_k{k}")
            dag.add_node(
                node_local_couple(
                    nid_lc,
                    lp.block,
                    local_ops=prim_ops,
                    pivot_id=piv,
                    duration=1,
                    meta={
                        "target_native": lp.native,
                        "requires_gauge_fix": lp.requires_gauge_fix,
                        "combine_rule": lp.combine_rule,
                        "step_in_local_sequence": k,
                    },
                )
            )
            dag.add_edge(last, nid_lc)
            last = nid_lc

        local_tail_nodes.append(last)

    # --- 1) interblock link (if needed) ---
    link_nid: Optional[str] = None
    if plan.interblock is not None:
        link_nid = namer.nid("link", layer, ridx)
        dag.add_node(
            node_interblock_link(
                link_nid,
                blocks=plan.interblock.blocks_involved,
                couplers=plan.interblock.couplers_used,
                duration=plan.interblock.duration,
                meta=plan.interblock.meta,
            )
        )
        # all local tails must precede the link (gather pivots / make them ready)
        for t in local_tail_nodes:
            dag.add_edge(t, link_nid)

    # --- 2) parity measurement: measure (P ⊗ Z_m) -> bPZ ---
    parity_nid = namer.nid("PZ", layer, ridx)
    dag.add_node(
        node_meas_parity_PZ(
            parity_nid,
            pauli=plan.axis,
            magic_id=plan.magic_id,
            out_key=plan.out_bPZ,
            blocks=plan.blocks_involved,
            couplers=(plan.interblock.couplers_used if plan.interblock is not None else None),
            duration=1,
            meta={"angle": plan.angle},
        )
    )

    # inputs to parity:
    if link_nid is not None:
        dag.add_edge(link_nid, parity_nid)
    else:
        # single block: local tail nodes feed directly to parity
        for t in local_tail_nodes:
            dag.add_edge(t, parity_nid)

    # --- 3) Clifford frame update depends on bPZ ---
    cliff_nid = namer.nid("FCL", layer, ridx)
    dag.add_node(
        node_frame_update(
            cliff_nid,
            update_kind="clifford_pi4",
            depends_on=plan.out_bPZ,
            axis=plan.axis,
            duration=0,
            meta={"note": "if bPZ=1 apply P_{pi/4} (Clifford) in frame"},
        )
    )
    dag.add_edge(parity_nid, cliff_nid)

    # --- 4) measure X_m -> bXm (must happen after parity) ---
    xm_nid = namer.nid("Xm", layer, ridx)
    dag.add_node(
        node_meas_magic_X(
            xm_nid,
            magic_id=plan.magic_id,
            out_key=plan.out_bXm,
            block=plan.magic_block,
            duration=1,
            meta={},
        )
    )
    dag.add_edge(parity_nid, xm_nid)

    # --- 5) Pauli frame update depends on bXm ---
    pauli_nid = namer.nid("FPA", layer, ridx)
    dag.add_node(
        node_frame_update(
            pauli_nid,
            update_kind="pauli",
            depends_on=plan.out_bXm,
            axis=plan.axis,
            duration=0,
            meta={"note": "if bXm=1 apply P_{pi/2}=P (Pauli) in frame"},
        )
    )
    dag.add_edge(xm_nid, pauli_nid)

    return GadgetLoweringResult(
        parity_node=parity_nid,
        xm_node=xm_nid,
        frame_cliff_node=cliff_nid,
        frame_pauli_node=pauli_nid,
        produced_bPZ=plan.out_bPZ,
        produced_bXm=plan.out_bXm,
    )