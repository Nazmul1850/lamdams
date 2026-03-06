# modqldpc/runtime/pauli_ops.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from modqldpc.core.types import PauliAxis


def _mul_single(a: str, b: str) -> Tuple[str, complex]:
    """
    Returns (c, phase) such that a*b = phase * c on one qubit.
    phase in {1,-1,1j,-1j}
    """
    if a == "I": return b, 1
    if b == "I": return a, 1
    if a == b:   return "I", 1

    # multiplication table for Pauli matrices
    table = {
        ("X","Y"): ("Z", 1j),
        ("Y","Z"): ("X", 1j),
        ("Z","X"): ("Y", 1j),
        ("Y","X"): ("Z",-1j),
        ("Z","Y"): ("X",-1j),
        ("X","Z"): ("Y",-1j),
    }
    return table[(a,b)]

def pauli_multiply(P: PauliAxis, Q: PauliAxis) -> Tuple[complex, PauliAxis]:
    """
    Returns (phase, R) such that P*Q = phase * R, where R is PauliAxis with sign +/-1 only.
    The complex phase captures leftover global phase in {1,-1,1j,-1j}.
    """
    if len(P.tensor) != len(Q.tensor):
        raise ValueError("Pauli length mismatch")

    phase = 1+0j
    out = []
    for a, b in zip(P.tensor, Q.tensor):
        c, ph = _mul_single(a, b)
        out.append(c)
        phase *= ph

    # include signs as +/-1 (real)
    phase *= (1 if P.sign >= 0 else -1)
    phase *= (1 if Q.sign >= 0 else -1)

    # strip any real +/- into axis.sign; keep phase in {1, -1, 1j, -1j}
    sign = +1
    if phase == -1:
        sign = -1
        phase = 1
    elif phase == -1j:
        phase = -1j  # keep as is
    elif phase == 1j:
        phase = 1j

    return phase, PauliAxis(sign=sign, tensor="".join(out))

def paulis_commute(P: PauliAxis, Q: PauliAxis) -> bool:
    """
    Two Pauli products commute iff number of positions with anti-commuting non-I Paulis is even.
    """
    if len(P.tensor) != len(Q.tensor):
        raise ValueError("Pauli length mismatch")

    anti = 0
    for a, b in zip(P.tensor, Q.tensor):
        if a == "I" or b == "I" or a == b:
            continue
        anti += 1
    return (anti % 2) == 0

def flip_sign(A: PauliAxis) -> PauliAxis:
    return PauliAxis(sign=-A.sign, tensor=A.tensor)

def conj_by_pi4_generator(P: PauliAxis, Q: PauliAxis) -> PauliAxis:
    """
    Implements GoSC commutation rule:
      If [P,Q]=0 then Q unchanged.
      If {P,Q}=0 then Q -> (i P Q)   (Hermitian Pauli product).
    We return a PauliAxis with sign +/-1, no complex phase.
    """
    if paulis_commute(P, Q):
        return Q

    # Compute P*Q = phase * R, where phase is +/- i for anti-commuting case.
    phase, R = pauli_multiply(P, Q)

    # For anti-commuting Paulis, phase should be ±i up to our sign conventions.
    # We need i*(P*Q) = i*phase*R = ±1 * R.
    if phase == 1j:
        # i*(+i)*R = -1*R
        return PauliAxis(sign=-R.sign, tensor=R.tensor)
    if phase == -1j:
        # i*(-i)*R = +1*R
        return PauliAxis(sign=+R.sign, tensor=R.tensor)

    # If something odd happens, fall back to R but warn via sign handling.
    # (Should not happen if commute-test is correct.)
    return R