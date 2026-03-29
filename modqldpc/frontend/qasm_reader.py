from __future__ import annotations

import re
from typing import List, Tuple, Optional

import pyzx as zx
from lsqecc.pauli_rotations.circuit import PauliOpCircuit
from lsqecc.pauli_rotations.rotation import Measurement, PauliOperator


# ── Constants ─────────────────────────────────────────────────────────────────

_QASM_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

# Lines starting with these tokens are not reversible gate instructions.
_SKIP_PREFIXES = ("OPENQASM", "include", "qreg", "creg")


# ── Public API ────────────────────────────────────────────────────────────────

def load_qasm_file(path: str) -> str:
    """Read a QASM file from disk and return its contents as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_qasm(qasm_str: str) -> Tuple[PauliOpCircuit, List[Tuple[int, Optional[int]]]]:
    """
    Parse a QASM 2.0 string into a PauliOpCircuit using PyZX as the backend.

    PyZX cannot handle: barrier, measure, creg, OPENQASM header, include.
    Strategy:
      - Split on barrier / measure lines.
      - Feed each reversible chunk to pyzx.Circuit.from_qasm().
      - Represent each measure as an explicit Z-basis Pauli Measurement.

    Returns
    -------
    circuit : PauliOpCircuit
        The full circuit as Pauli operations.
    meas_map : list of (qbit, cbit | None)
        Original qubit → classical-bit assignments for every measure instruction
        encountered, in program order.  Used downstream to assign qbit/cbit to
        PauliMeasurement objects.
    """
    lines = [ln.strip() for ln in qasm_str.splitlines()]

    # Extract qubit count from qreg declaration.
    num_qubits: Optional[int] = None
    for ln in lines:
        m = re.match(r"qreg\s+\w+\[(\d+)\]\s*;", ln)
        if m:
            num_qubits = int(m.group(1))
            break
    if num_qubits is None:
        raise ValueError("No qreg declaration found in QASM string.")

    # Segment lines into reversible gate blocks and measure events.
    # Each segment is ("reversible", [gate_line, ...]) or ("measure", qbit, cbit).
    segments: List[tuple] = []
    current_gates: List[str] = []

    for ln in lines:
        if not ln:
            continue
        if any(ln.startswith(k) for k in _SKIP_PREFIXES):
            continue

        if ln.startswith("barrier"):
            if current_gates:
                segments.append(("reversible", current_gates))
                current_gates = []

        elif ln.startswith("measure"):
            if current_gates:
                segments.append(("reversible", current_gates))
                current_gates = []
            # Parse:  measure q[i] -> c[j];
            qbit = _parse_bracket_index(ln, 0)
            cbit = _parse_bracket_index(ln, 1)
            segments.append(("measure", qbit, cbit))

        else:
            current_gates.append(ln)

    if current_gates:
        segments.append(("reversible", current_gates))

    # Build sub-circuits and join them.
    qreg_decl = f"qreg q[{num_qubits}];\n"
    sub_circuits: List[PauliOpCircuit] = []
    meas_map: List[Tuple[int, Optional[int]]] = []

    for seg in segments:
        if seg[0] == "reversible":
            gate_lines = seg[1]
            if not gate_lines:
                continue
            seg_qasm = _QASM_HEADER + qreg_decl + "\n".join(gate_lines) + "\n"
            pyzx_circ = zx.Circuit.from_qasm(seg_qasm)
            sub_circuits.append(PauliOpCircuit.load_from_pyzx(pyzx_circ))

        elif seg[0] == "measure":
            _, qbit, cbit = seg
            meas_map.append((qbit, cbit))
            c = PauliOpCircuit(num_qubits)
            ops = [PauliOperator.I] * num_qubits
            ops[qbit] = PauliOperator.Z
            c.add_pauli_block(Measurement.from_list(ops))
            sub_circuits.append(c)

    if not sub_circuits:
        return PauliOpCircuit(num_qubits), meas_map

    result = sub_circuits[0]
    for sc in sub_circuits[1:]:
        result = PauliOpCircuit.join(result, sc)

    return result, meas_map


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_bracket_index(line: str, occurrence: int) -> Optional[int]:
    """
    Return the integer inside the n-th [...] pair in ``line``.
    Returns None if the occurrence does not exist.
    """
    matches = re.findall(r"\[(\d+)\]", line)
    if occurrence < len(matches):
        return int(matches[occurrence])
    return None
