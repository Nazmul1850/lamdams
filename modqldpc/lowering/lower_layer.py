# modqldpc/lowering/lower_layer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .ir import ExecDAG
from .policy import LoweringPolicies, plan_rotation_lowering
from .magic_gadget import emit_pi8_gadget, GadgetLoweringResult
from ..core.types import PauliRotation, PauliAxis  # your dataclasses
from qiskit.quantum_info import Pauli

@dataclass(frozen=True)
class LayerLoweringResult:
    layer: int
    dag: ExecDAG
    gadget_results: List[GadgetLoweringResult]
    meta: Dict[str, Any] = field(default_factory=dict)



def lower_one_layer(
    *,
    layer_idx: int,
    rotations: List[PauliRotation],
    rotation_indices: List[int],   # the indices belonging to this layer
    hw,
    policies: LoweringPolicies,
    logical_ids: Optional[List[int]] = None,
) -> LayerLoweringResult:
    """
    Lower one commuting layer into an ExecDAG, in a deterministic order.
    (Within-layer commutativity assumed from layering pass.)
    """
    dag = ExecDAG()
    gadget_results: List[GadgetLoweringResult] = []

    for ridx in rotation_indices:
        r = rotations[ridx]
        sign = -1 if r.angle < 0 else 1
        tensor = r.axis.to_label()
        if '-' in tensor:
            tensor = tensor[1:]
        axis = PauliAxis(sign=sign, tensor=tensor)
        plan = plan_rotation_lowering(
            layer=layer_idx,
            ridx=ridx,
            axis=axis,
            angle=float(r.angle),
            hw=hw,
            policies=policies,
            logical_ids=logical_ids,
        )
        gr = emit_pi8_gadget(dag=dag, plan=plan, policies=policies)
        gadget_results.append(gr)

    # basic checks
    dag.validate()
    _ = dag.topological_order()  # raises on cycles

    return LayerLoweringResult(layer=layer_idx, dag=dag, gadget_results=gadget_results)