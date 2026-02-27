# modqldpc/pipeline/run_one.py (demo snippet)
from modqldpc.mapping.model import GraphFactory, RingTopology
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.policy import (
    LoweringPolicies,
    ChooseMagicBlockMinId,
    ShortestPathGatherRouting,
    NativeAllPaulisForNow,
    MagicPlacementPolicy,
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.core.types import PauliAxis, PauliRotation  # your dataclasses
from modqldpc.lowering.visualize import dag_to_dot



class ChooseMagicBlockMaxId:
    name = "choose_magic_maxid"
    def choose_magic_block(self, *, blocks_involved, hw):
        return max(blocks_involved)


def demo_lowering_step3():
    # --- hardware: ring of 2 blocks (1-2) ---
    hw = GraphFactory().build(topology=RingTopology(), block_ids=[1, 2])

    # map 2 logicals: q0->B1, q1->B2
    hw.add_mapping(0, 1, 0)
    hw.add_mapping(1, 2, 0)

    # two commuting rotations example:
    # P0 = Z0 Z1 (touches both blocks)
    # P1 = X0 X1 (also touches both; note: ZZ and XX commute on 2 qubits)
    rotations = [
        PauliRotation(axis=PauliAxis(+1, "ZZ"), angle=3.141592653589793/8, source="t q[?]", idx=0),
        PauliRotation(axis=PauliAxis(+1, "XX"), angle=3.141592653589793/8, source="t q[?]", idx=1),
    ]
    layers = [[0, 1]]  # single layer

    base_policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=NativeAllPaulisForNow(),
    )

    res1 = lower_one_layer(
        layer_idx=0,
        rotations=rotations,
        rotation_indices=layers[0],
        hw=hw,
        policies=base_policies,
    )
    print("Policy1 magic=minid; nodes:", len(res1.dag.nodes))
    print("Topo:", res1.dag.topological_order())

    dot_str = dag_to_dot(res1.dag)
    print(dot_str)

    # swap only magic placement policy (Strategy pattern)
    policies2 = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMaxId(),  # different policy
        routing=ShortestPathGatherRouting(),
        native=NativeAllPaulisForNow(),
    )

    res2 = lower_one_layer(
        layer_idx=0,
        rotations=rotations,
        rotation_indices=layers[0],
        hw=hw,
        policies=policies2,
    )
    print("Policy2 magic=maxid; nodes:", len(res2.dag.nodes))
    dot_str = dag_to_dot(res2.dag)
    print(dot_str)
    # validation: same number of nodes, but magic_id / magic_block should differ in meta
    # pick first parity node:
    pz0_1 = [n for n in res1.dag.nodes.values() if n.kind == "meas_parity_PZ"][0]
    pz0_2 = [n for n in res2.dag.nodes.values() if n.kind == "meas_parity_PZ"][0]
    print("PZ meta policy1:", pz0_1.meta.get("magic_id"), "blocks:", pz0_1.blocks)
    print("PZ meta policy2:", pz0_2.meta.get("magic_id"), "blocks:", pz0_2.blocks)

    # sanity: enforce parity -> Xm dependency exists
    # find Xm node and verify predecessor includes PZ
    for gr in res1.gadget_results:
        assert gr.xm_node in res1.dag.succ[gr.parity_node], "Missing parity->Xm edge"


if __name__ == "__main__":
    demo_lowering_step3()