from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

from lsqecc.pauli_rotations.circuit import PauliOpCircuit
from lsqecc.pauli_rotations.rotation import (
    Measurement as LsqMeasurement,
    PauliOperator,
    PauliRotation as LsqRotation,
)

from modqldpc.core.types import PauliAxis, PauliMeasurement, PauliProgram, PauliRotation
from modqldpc.frontend.qasm_reader import load_qasm_file, parse_qasm


# ── Constants ─────────────────────────────────────────────────────────────────

_PI_OVER_2 = math.pi / 2.0
_PI_OVER_8 = math.pi / 8.0


# ── Pauli-word utilities ──────────────────────────────────────────────────────

def word_support(word: str) -> Set[int]:
    """Indices where the Pauli word is not 'I'."""
    return {i for i, ch in enumerate(word) if ch != "I"}


def paulis_commute(p: str, q: str) -> bool:
    """
    True iff two Pauli word strings commute (symplectic check).

    Two Pauli products commute iff the number of qubit positions where
    they each have a non-identity operator AND those operators differ
    (i.e. single-qubit anticommute) is even.
    """
    anti = sum(1 for a, b in zip(p, q) if a != "I" and b != "I" and a != b)
    return anti % 2 == 0


# ── lsqecc ↔ modqldpc conversion helpers ─────────────────────────────────────

def _lsq_word(op: LsqRotation | LsqMeasurement) -> str:
    """
    Build a Pauli word string from an lsqecc ops_list.

    lsqecc stores qubit-0 at index 0 (left to right).
    Qiskit Pauli labels are right-to-left (qubit-0 is the rightmost character).
    We reverse so that Pauli("IXYZ") in Qiskit matches lsqecc qubit ordering.
    """
    return "".join(o.value for o in reversed(op.ops_list))


def _lsq_sign(rot: LsqRotation) -> int:
    """Return +1 if rotation_amount > 0, else -1."""
    return 1 if rot.rotation_amount > 0 else -1


def _lsq_denom(rot: LsqRotation) -> int:
    """
    Return the exact denominator of the rotation_amount Fraction.

    Expected values after Litinski transform:
      2  → π/2  Clifford Pauli rotation (survives transform)
      4  → π/4  should be zero (absorbed by Litinski)
      8  → π/8  T-gate (magic state required)
      ≥16 → high-precision rotation from small-angle gates (e.g. QFT CPhase)

    NOTE: do NOT use limit_denominator — it rounds high-precision fractions
    (e.g. 1/524288) to 0, producing a spurious denom=1 and angle=π.
    """
    return Fraction(abs(rot.rotation_amount)).denominator


def _lsq_rot_to_modqldpc(rot: LsqRotation, idx: int) -> PauliRotation:
    """Convert one lsqecc PauliRotation to a modqldpc PauliRotation."""
    word  = _lsq_word(rot)
    sign  = _lsq_sign(rot)
    denom = _lsq_denom(rot)
    angle = sign * math.pi / denom
    return PauliRotation(axis=word, angle=angle, source="", idx=idx)


def _lsq_meas_to_modqldpc(
    meas: LsqMeasurement,
    idx: int,
    qbit: int,
    cbit: Optional[int],
) -> PauliMeasurement:
    """Convert one lsqecc Measurement to a modqldpc PauliMeasurement."""
    return PauliMeasurement(axis=_lsq_word(meas), cbit=cbit, qbit=qbit, idx=idx)


# ── Main converter ────────────────────────────────────────────────────────────

class GoSCConverter:
    """
    Compiles a QASM 2.0 circuit into Pauli-product rotations via the
    Litinski / lsqecc pipeline:

      1. Parse QASM with PyZX → PauliOpCircuit
      2. Inject Z-basis measurements if absent (required for Litinski)
      3. to_y_free_equivalent()  — replace Y with X + flanking ±π/4 Z
      4. apply_transformation()  — push and remove all ±π/4 (Litinski game)
      5. Classify ops: π/2 and π/8 kept; π/4 warned and dropped (should be 0)
      6. Greedy first-fit layering

    Public interface
    ----------------
    convert_qasm(qasm_str)  → PauliProgram
    greedy_layering()       → List[List[int]]
    to_cache_payload()      → dict   (verbose JSON)
    to_compact_payload()    → dict   (compact JSON v2)
    load_cache_json(path)   → dict   (auto-detects schema v1 / v2)
    """

    # Schema identifiers — v1 kept for reading legacy files.
    CACHE_SCHEMA         = "modqldpc.pauli_program_cache.v1"
    COMPACT_CACHE_SCHEMA = "modqldpc.pauli_program_cache.v2.compact"
    _LEGACY_COMPACT      = "modqldpc.pauli_program_cache.v1.compact"

    def __init__(self, verbose: bool = False):
        self.verbose  = verbose
        self.program:    Optional[PauliProgram]  = None
        self.layers:     Optional[List[List[int]]] = None
        self.num_qubits: int = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Primary entry point ───────────────────────────────────────────────────

    def convert_qasm(self, qasm_str: str) -> PauliProgram:
        """
        Full Litinski compilation pipeline on a raw QASM 2.0 string.

        Populates self.program and self.num_qubits.
        Call greedy_layering() afterwards to populate self.layers.
        """
        # 1. Parse
        circuit, meas_map = parse_qasm(qasm_str)
        self.num_qubits = circuit.qubit_num
        self.log(f"[parse]  qubits={self.num_qubits}  ops={len(circuit.ops)}")

        # 2. Inject synthetic Z measurements if the circuit has none.
        #    Litinski's transform can only absorb ±π/4 when measurements exist.
        if not circuit.circuit_has_measurements():
            self.log("[inject] No measurements found — injecting synthetic Z-basis measurements.")
            for q in range(circuit.qubit_num):
                ops = [PauliOperator.I] * circuit.qubit_num
                ops[q] = PauliOperator.Z
                circuit.add_pauli_block(LsqMeasurement.from_list(ops))
            # Synthetic measurements: qbit=q, cbit=None
            meas_map = [(q, None) for q in range(circuit.qubit_num)]

        # 3. Replace Y operators with X + flanking ±π/4 Z rotations.
        #    MUST happen before Litinski so those π/4s get absorbed too.
        circuit = circuit.to_y_free_equivalent()
        self.log(f"[y_free] ops after to_y_free_equivalent: {len(circuit.ops)}")

        # 4. Litinski transform: push all ±π/4 to end and remove them.
        #    After this: only π/2 and π/8 rotations survive (+ measurements).
        #    NEVER call to_y_free_equivalent() again after this point.
        circuit.apply_transformation()
        self.log(f"[litinski] ops after apply_transformation: {len(circuit.ops)}")

        # 5. Extract rotations and measurements.
        rotations: List[PauliRotation]    = []
        final_meas: List[PauliMeasurement] = []
        meas_iter = iter(meas_map)
        pi4_count = 0

        for op in circuit.ops:
            if isinstance(op, LsqMeasurement):
                qbit, cbit = next(meas_iter, (0, None))
                pm = _lsq_meas_to_modqldpc(op, idx=len(final_meas), qbit=qbit, cbit=cbit)
                final_meas.append(pm)

            elif isinstance(op, LsqRotation):
                denom = _lsq_denom(op)

                if denom == 1:
                    # exp(i*pi*P) = -I  (global phase), physically trivial.
                    continue

                if denom == 4:
                    # Should be zero after correct Litinski — warn and skip.
                    pi4_count += 1
                    continue

                # Keep all other rotations:
                #   denom=2  → π/2  Clifford Pauli (no magic state, but real time step)
                #   denom=8  → π/8  T-gate (magic state required)
                #   denom≥16 → high-precision rotation (QFT small-angle CPhase etc.)
                #              needs T-gate synthesis on fault-tolerant hardware;
                #              contributes to depth; AQFT truncates these below a threshold.
                pr = _lsq_rot_to_modqldpc(op, idx=len(rotations))
                rotations.append(pr)

        if pi4_count:
            print(
                f"[WARNING] {pi4_count} pi/4 rotation(s) survived Litinski transform. "
                "This should not happen — check circuit structure."
            )

        t_count  = sum(1 for r in rotations if abs(abs(r.angle) - _PI_OVER_8) < 1e-9)
        p2_count = sum(1 for r in rotations if abs(abs(r.angle) - _PI_OVER_2) < 1e-9)
        hp_count = len(rotations) - t_count - p2_count
        self.log(
            f"[extract] rotations={len(rotations)} "
            f"(pi8={t_count}, pi2={p2_count}, high-precision={hp_count})"
            f"  measurements={len(final_meas)}"
        )

        self.program = PauliProgram(
            rotations=rotations,
            final_meas=final_meas,
            final_clifford=None,
        )
        return self.program

    # ── Layering ──────────────────────────────────────────────────────────────

    def greedy_layering(self) -> List[List[int]]:
        """
        First-fit greedy layer assignment (program order).

        For each rotation in order: place it in the first existing layer
        where it commutes with every rotation already there.  If none found,
        open a new layer.

        Strictly better than bubble-merge: can skip to an earlier compatible
        layer without being blocked by intermediate anticommuting rotations.

        Returns layer_ids: List[List[rotation_index]].
        """
        if self.program is None:
            raise ValueError("Call convert_qasm() first.")

        rotations = self.program.rotations
        layers: List[List[int]] = []

        for rot in rotations:
            placed = False
            for layer in layers:
                if all(paulis_commute(rot.axis, rotations[j].axis) for j in layer):
                    layer.append(rot.idx)
                    placed = True
                    break
            if not placed:
                layers.append([rot.idx])

        self.layers = layers
        self.log(f"[layers] {len(layers)} layers for {len(rotations)} rotations.")
        return layers

    # ── JSON serialisation ────────────────────────────────────────────────────

    @staticmethod
    def _axis_to_dict(axis: str) -> dict:
        """Split a Pauli word string (no sign prefix expected) into sign+tensor dict."""
        if axis.startswith("-"):
            return {"sign": -1, "tensor": axis[1:]}
        if axis.startswith("+"):
            return {"sign": +1, "tensor": axis[1:]}
        return {"sign": +1, "tensor": axis}

    def to_cache_payload(self) -> Dict[str, Any]:
        """Verbose JSON payload (schema v1)."""
        if self.program is None:
            raise ValueError("self.program is None. Nothing to export.")
        if self.layers is None:
            raise ValueError("Call greedy_layering() before serialising.")

        rotations_view = []
        for r in self.program.rotations:
            d    = self._axis_to_dict(r.axis)
            word = ("" if d["sign"] == 1 else "-") + d["tensor"]
            rotations_view.append({
                "idx":     r.idx,
                "word":    word,
                "support": sorted(word_support(d["tensor"])),
                "angle":   r.angle,
                "source":  r.source,
                "axis":    d,
            })

        meas_view = []
        for m in self.program.final_meas:
            d = self._axis_to_dict(m.axis)
            word = ("" if d["sign"] == 1 else "-") + d["tensor"]
            meas_view.append({
                "idx":     m.idx,
                "word":    word,
                "support": sorted(word_support(d["tensor"])),
                "qbit":    m.qbit,
                "cbit":    m.cbit,
                "axis":    d,
            })

        layers_view = []
        for li, layer in enumerate(self.layers):
            items = [
                {"ridx": ridx, "word": ("" if self._axis_to_dict(self.program.rotations[ridx].axis)["sign"] == 1 else "-") + self._axis_to_dict(self.program.rotations[ridx].axis)["tensor"]}
                for ridx in layer
            ]
            layers_view.append({"layer": li, "size": len(layer), "rotations": items})

        return {
            "schema":             self.CACHE_SCHEMA,
            "rotations":          rotations_view,
            "final_measurements": meas_view,
            "layers":             layers_view,
            "layer_ids":          self.layers,
        }

    def to_compact_payload(self) -> Dict[str, Any]:
        """
        Compact JSON payload (schema v2).

        Rotation entry : [sign, tensor, denominator]
            sign        = ±1 (sign of the rotation angle)
            tensor      = Pauli word without sign prefix
            denominator = 2 | 4 | 8  (π/denom)

        Measurement entry : [sign, tensor, qbit, cbit_or_null]
        """
        if self.program is None:
            raise ValueError("self.program is None. Nothing to export.")
        if self.layers is None:
            raise ValueError("Call greedy_layering() before serialising.")

        rotations_compact = []
        for r in self.program.rotations:
            d    = self._axis_to_dict(r.axis)
            sign = 1 if r.angle >= 0 else -1
            # Reconstruct denominator from angle: angle = sign * π / denom
            denom = round(math.pi / abs(r.angle)) if abs(r.angle) > 1e-12 else 1
            rotations_compact.append([sign, d["tensor"], denom])

        measurements_compact = []
        for m in self.program.final_meas:
            d = self._axis_to_dict(m.axis)
            measurements_compact.append([d["sign"], d["tensor"], m.qbit, m.cbit])

        return {
            "schema":       self.COMPACT_CACHE_SCHEMA,
            "rotations":    rotations_compact,
            "measurements": measurements_compact,
            "layer_ids":    self.layers,
        }

    def to_cache_json_string(self) -> str:
        return json.dumps(self.to_cache_payload(), indent=2, sort_keys=True)

    # ── JSON loading ──────────────────────────────────────────────────────────

    def load_cache_json(self, path: str) -> Dict[str, Any]:
        """
        Load a PBC JSON file.  Auto-detects schema:
          - v1 verbose        → "modqldpc.pauli_program_cache.v1"
          - v1 compact (legacy) → "modqldpc.pauli_program_cache.v1.compact"
          - v2 compact        → "modqldpc.pauli_program_cache.v2.compact"
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        schema = payload.get("schema", "")

        if schema == self.COMPACT_CACHE_SCHEMA:
            self._load_compact_v2(payload)
        elif schema == self._LEGACY_COMPACT:
            self._load_compact_v1(payload)
        elif schema == self.CACHE_SCHEMA:
            self._load_verbose_v1(payload)
        else:
            raise ValueError(f"Unknown PBC cache schema: {schema!r}")

        return payload

    # ── Private load helpers ──────────────────────────────────────────────────

    def _load_compact_v2(self, payload: Dict[str, Any]) -> None:
        """Decode v2 compact: rotation entry = [sign, tensor, denominator]."""
        rotations: List[PauliRotation] = []
        for i, entry in enumerate(payload.get("rotations", [])):
            sign, tensor, denom = entry
            rotations.append(PauliRotation(
                axis=tensor,
                angle=sign * math.pi / denom,
                source="",
                idx=i,
            ))

        final_meas: List[PauliMeasurement] = []
        for i, entry in enumerate(payload.get("measurements", [])):
            _sign, tensor, qbit, cbit = entry
            final_meas.append(PauliMeasurement(
                axis=tensor,
                cbit=None if cbit is None else int(cbit),
                qbit=int(qbit),
                idx=i,
            ))

        self.program = PauliProgram(rotations=rotations, final_meas=final_meas, final_clifford=None)
        self.layers  = [[int(x) for x in layer] for layer in payload.get("layer_ids", [])]

    def _load_compact_v1(self, payload: Dict[str, Any]) -> None:
        """
        Decode legacy v1 compact: rotation entry = [axis_sign, tensor, t_sign]
        where angle = t_sign * π/8.

        The combined sign is axis_sign * t_sign absorbed into angle.
        """
        rotations: List[PauliRotation] = []
        for i, entry in enumerate(payload.get("rotations", [])):
            axis_sign, tensor, t_sign = entry
            angle = axis_sign * t_sign * _PI_OVER_8
            rotations.append(PauliRotation(axis=tensor, angle=angle, source="", idx=i))

        final_meas: List[PauliMeasurement] = []
        for i, entry in enumerate(payload.get("measurements", [])):
            _axis_sign, tensor, qbit, cbit = entry
            final_meas.append(PauliMeasurement(
                axis=tensor,
                cbit=None if cbit is None else int(cbit),
                qbit=int(qbit),
                idx=i,
            ))

        self.program = PauliProgram(rotations=rotations, final_meas=final_meas, final_clifford=None)
        self.layers  = [[int(x) for x in layer] for layer in payload.get("layer_ids", [])]

    def _load_verbose_v1(self, payload: Dict[str, Any]) -> None:
        """Decode verbose v1 format."""
        rotations: List[PauliRotation] = []
        for r in payload.get("rotations", []):
            rotations.append(PauliRotation(
                axis=r["axis"]["tensor"],
                angle=float(r["angle"]),
                source=str(r.get("source", "")),
                idx=int(r["idx"]),
            ))

        final_meas: List[PauliMeasurement] = []
        for m in payload.get("final_measurements", []):
            final_meas.append(PauliMeasurement(
                axis=m["axis"]["tensor"],
                cbit=None if m["cbit"] is None else int(m["cbit"]),
                qbit=int(m["qbit"]),
                idx=int(m["idx"]),
            ))

        self.program = PauliProgram(rotations=rotations, final_meas=final_meas, final_clifford=None)
        self.layers  = [[int(x) for x in layer] for layer in payload.get("layer_ids", [])]
