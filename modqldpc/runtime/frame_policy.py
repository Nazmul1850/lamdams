# modqldpc/runtime/frame_policy.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .pauli_ops import PauliAxis, conj_by_pi4_generator, paulis_commute, flip_sign


@dataclass
class FrameState:
    """
    State carried between layers.
    We store byproducts as lists in time order (within a layer execution).
    """
    bits: Dict[str, int] = field(default_factory=dict)

    # Byproducts accumulated so far:
    clifford_pi4_generators: List[PauliAxis] = field(default_factory=list)  # apply conjugation
    pauli_byproducts: List[PauliAxis] = field(default_factory=list)         # apply sign flips on anticommute


@dataclass(frozen=True)
class AxisRewriteLog:
    ridx: int
    before: PauliAxis
    after: PauliAxis
    angle_before: float
    angle_after: float
    changed_support: bool
    reason: str


class FrameUpdatePolicy:
    """
    Implements:
      - if bPZ=1 -> apply Clifford correction P_{pi/4}
      - if bXm=1 -> apply Pauli correction P (pi/2)
    Then provides a method to rewrite next-layer axes.
    """

    def apply_frame_update(self, *, update_kind: str, bit: int, axis: PauliAxis, st: FrameState) -> None:
        if bit not in (0, 1):
            raise ValueError("bit must be 0/1")

        if bit == 0:
            return

        if update_kind == "clifford_pi4":
            # print("Any update!!")
            st.clifford_pi4_generators.append(axis)
            return

        if update_kind == "pauli":
            st.pauli_byproducts.append(axis)
            return

        raise ValueError(f"Unknown update_kind '{update_kind}'")

    def rewrite_axis(self, axis: PauliAxis, angle: float, st: FrameState) -> tuple[PauliAxis, float, str]:
        """
        Apply accumulated byproducts to a future rotation axis/angle.

        Convention:
          - Clifford pi/4 generators conjugate axis: Q -> C Q C†
          - Pauli byproducts flip sign/angle when anticommute: Q -> -Q  (equiv angle -> -angle)
        """
        Q = axis
        ang = float(angle)
        reasons = []

        # apply Clifford corrections in order
        for P in st.clifford_pi4_generators:
            Q2 = conj_by_pi4_generator(P, Q)
            if Q2 != Q:
                reasons.append("conj_pi4")
            Q = Q2

        # apply Pauli corrections in order (sign flip if anticommute)
        for P in st.pauli_byproducts:
            if not paulis_commute(P, Q):
                # Q = flip_sign(Q)
                ang = -ang
                reasons.append("pauli_anticomm_flip")

        return Q, ang, "+".join(reasons) if reasons else "none"