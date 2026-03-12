
from dataclasses import asdict, is_dataclass
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from modqldpc.core.types import PauliAxis, PauliMeasurement, PauliProgram, PauliRotation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Clifford



# -----------------------------
# Pauli-word utilities
# -----------------------------

def pauli_to_word(p: Pauli) -> str:
    """Return a compact Pauli word like 'IXYZ' (Qiskit uses little-endian internally; we print qubit0 on the right by default)."""
    # Qiskit Pauli str is like 'IXYZ' with leftmost = highest qubit index.
    return p.to_label()

def word_support(word: str) -> Set[int]:
    """Support indices where word != 'I'. Here index 0 is the RIGHTMOST char (qubit 0)."""
    supp = set()
    n = len(word)
    for k, ch in enumerate(reversed(word)):
        if ch != "I" and ch != "-":
            supp.add(k)
    return supp

def paulis_commute(p: Pauli, q: Pauli) -> bool:
    """True if p and q commute."""
    # Qiskit Pauli has commutes() in newer versions; fallback to symplectic check if needed.
    try:
        return p.commutes(q)
    except Exception:
        # symplectic: commute iff (x1·z2 + z1·x2) mod2 ==0
        x1 = p.x.astype(int)
        z1 = p.z.astype(int)
        x2 = q.x.astype(int)
        z2 = q.z.astype(int)
        s = int((x1 @ z2 + z1 @ x2) % 2)
        return s == 0

def pauli_mul_up_to_phase(p: Pauli, q: Pauli) -> Pauli:
    """
    Multiply Paulis and drop the global phase (i, -1, etc.).
    This matches the paper's "i P P'" style axis update: axis is defined up to phase.
    """
    r = p @ q
    # Qiskit Pauli keeps a phase in Pauli.phase sometimes. Converting to label drops it.
    return Pauli(r.to_label())


def clifford_conj_pauli(C: Clifford, P: Pauli) -> Pauli:
    """
    Return C * P * C†  (Heisenberg action).
    Works across Qiskit versions by trying Pauli.evolve frames.
    """
    # Most common API: Pauli.evolve(other, frame=...)
    for frame in ("s", "h"):  # we'll detect which matches C P C†
        try:
            out = P.evolve(C, frame=frame)
            
            # quick self-consistency: if C is identity, output should equal input
            # (this is always true, so it's not a strong check, but keeps us safe)
            # print(f"Conjugation out={out.to_label()}, frame = {frame}")
            return Pauli(out.to_label())  # drop any global phase, keep axis
        except TypeError:
            pass

    # Older API sometimes omits the frame kwarg
    try:
        out = P.evolve(C)
        # print(f"Conjugation out={out.to_label()}, no frame")
        return Pauli(out.to_label())
    except Exception as e:
        raise RuntimeError(
            "Could not conjugate Pauli by Clifford. "
            "Your Qiskit version does not support Pauli.evolve(Clifford)."
        ) from e

def clifford_conj_pauli_adjoint(C: Clifford, P: Pauli) -> Pauli:
    """Return C† * P * C."""
    return clifford_conj_pauli(C.adjoint(), P)



class GoSCConverter:
    """
    Maintains a running Clifford frame C consisting of all Clifford gates encountered so far.
    When a T on qubit i occurs, emit rotation about (C Z_i C†) with angle +/- pi/8.

    This directly implements the paper's idea:
    - commute Cliffords to the end,
    - turn Z_{pi/8} into Pauli product rotations via conjugation by Cliffords,
    - absorb final Clifford into Pauli product measurements.
    """
    CACHE_SCHEMA         = "modqldpc.pauli_program_cache.v1"
    COMPACT_CACHE_SCHEMA = "modqldpc.pauli_program_cache.v1.compact"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.program: PauliProgram = None
        self.layers: List[List[int]] = None

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def gate_is_supported(name: str) -> bool:
        return name in {
            "h", "s", "sdg", "x", "y", "z",
            "cx", "cz", "swap",
            "t", "tdg",
            "barrier",
            "measure",
        }

    @staticmethod
    def is_clifford_gate(name: str) -> bool:
        return name in {"h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"}

    @staticmethod
    def is_t_gate(name: str) -> bool:
        return name in {"t", "tdg"}

    @staticmethod
    def z_on_qubit(n: int, q: int) -> Pauli:
        """
        Return Pauli Z on qubit q in an n-qubit system.
        Qiskit Pauli label leftmost is highest index, so:
          qubit 0 is rightmost char.
        """
        label = ["I"] * n
        label[n - 1 - q] = "Z"
        return Pauli("".join(label))

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {str(k): GoSCConverter._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [GoSCConverter._jsonable(x) for x in obj]
        # NOTE: final_clifford may be a Qiskit object; you likely can't JSON it directly.
        # Store str(obj) as a fallback so the cache is still writable.
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return {"__repr__": repr(obj), "__type__": type(obj).__name__}
    
    @staticmethod
    def _axis_from_word(word: str) -> PauliAxis:
        # word may have leading '-' or '+'
        if word.startswith("-"):
            return PauliAxis(sign=-1, tensor=word[1:])
        if word.startswith("+"):
            return PauliAxis(sign=+1, tensor=word[1:])
        return PauliAxis(sign=+1, tensor=word)
    
    @staticmethod
    def _axis_to_dict(axis) -> dict:
        word = pauli_to_word(axis)   # always returns string like "-XIZI"
        if word.startswith("-"):
            return {"sign": -1, "tensor": word[1:]}
        elif word.startswith("+"):
            return {"sign": +1, "tensor": word[1:]}
        else:
            return {"sign": +1, "tensor": word}

    def clifford_of_instruction(self, n: int, inst_name: str, qargs: List[int]) -> Clifford:
        """Build a Clifford for one supported Clifford instruction."""
        qc = QuantumCircuit(n)
        if inst_name == "h":
            qc.h(qargs[0])
        elif inst_name == "s":
            qc.s(qargs[0])
        elif inst_name == "sdg":
            qc.sdg(qargs[0])
        elif inst_name == "x":
            qc.x(qargs[0])
        elif inst_name == "y":
            qc.y(qargs[0])
        elif inst_name == "z":
            qc.z(qargs[0])
        elif inst_name == "cx":
            qc.cx(qargs[0], qargs[1])
        elif inst_name == "cz":
            qc.cz(qargs[0], qargs[1])
        elif inst_name == "swap":
            qc.swap(qargs[0], qargs[1])
        else:
            raise ValueError(f"Not a Clifford gate: {inst_name}")
        return Clifford(qc)

    def convert(self, qc: QuantumCircuit) -> PauliProgram:
        n = qc.num_qubits
        C = Clifford(QuantumCircuit(n))  # identity frame

        rotations: List[PauliRotation] = []
        final_meas: List[Tuple[int, Optional[int]]] = []  # (qbit, cbit)

        # Validate "final measurements only"
        seen_measure = False

        self.log(f"[init] n={n}, starting Clifford frame C = I")

        for k, ci in enumerate(qc.data):
            inst = ci.operation
            qargs = ci.qubits
            cargs = ci.clbits
            name = inst.name

            if not self.gate_is_supported(name):
                raise ValueError(
                    f"Unsupported gate '{name}' at op#{k}. "
                    f"Transpile to basis {{h,s,sdg,x,y,z,cx,cz,swap,t,tdg,measure}} first."
                )

            if name == "barrier":
                self.log(f"[op#{k}] barrier (ignored)")
                continue

            if name == "measure":
                seen_measure = True
                q = qc.find_bit(qargs[0]).index
                c = qc.find_bit(cargs[0]).index if cargs else None
                final_meas.append((q, c))
                self.log(f"[op#{k}] measure q[{q}] -> c[{c}] (recorded for final absorption)")
                continue

            if seen_measure:
                raise ValueError(
                    f"Found gate '{name}' after a measurement at op#{k}. "
                    f"This script assumes measurements are final only."
                )

            qidx = [qc.find_bit(q).index for q in qargs]

            if self.is_clifford_gate(name):
                G = self.clifford_of_instruction(n, name, qidx)
                # Update running frame: applying gate after current C means new C = G ∘ C
                C = C.compose(G)
                self.log(f"[op#{k}] Clifford {name} {qidx} -> update frame C := G ∘ C")
                continue

            if self.is_t_gate(name):
                q = qidx[0]
                sign = +1.0 if name == "t" else -1.0
                angle = sign * (math.pi / 8.0)

                # The paper: commuting Cliffords to end maps Z_{π/8} -> (C Z C†)_{π/8}
                Zq = self.z_on_qubit(n, q)
                axis = clifford_conj_pauli(C, Zq)

                rotations.append(
                    PauliRotation(axis=axis, angle=angle, source=f"{name} q[{q}]", idx=len(rotations))
                )

                self.log(
                    f"[op#{k}] {name} on q[{q}]: emit rotation axis = C Z[{q}] C† = {pauli_to_word(axis)}, angle={angle:+.6f}"
                )
                continue

            raise RuntimeError(f"Unhandled instruction '{name}' (should not happen).")

        # Absorb final Clifford into measurements: measure C† Z_i C (equivalently conjugate Z by C†).
        # Here we have U = (rotations) * C, then measure Z. Equivalent to measure (C† Z C) after rotations.
        measurements: List[PauliMeasurement] = []

        if not final_meas:
            self.log("[final] No measurements found. (You can still inspect rotations.)")

        for mi, (q, c) in enumerate(final_meas):
            Zq = self.z_on_qubit(n, q)
            axis = clifford_conj_pauli_adjoint(C, Zq)  # C† Z C
            measurements.append(PauliMeasurement(axis=axis, cbit=c, qbit=q, idx=mi))
            self.log(
                f"[final] absorb Clifford into meas q[{q}]: axis = C† Z[{q}] C = {pauli_to_word(axis)} -> c[{c}]"
            )
        program = PauliProgram(rotations=rotations, final_meas=measurements, final_clifford=C)
        self.program = program
        return program
    

    def greedy_layering(self) -> List[List[int]]:
        """
        Make a naive layering then repeatedly move commuting rotations earlier,
        matching the pseudocode near Fig. 6.

        Returns: layers as lists of rotation indices.
        """
        if not self.program:
            raise ValueError(
                f"Convert the circuit first to get greedy layering"
            )
        # start naive: each rotation in its own layer
        rotations = self.program.rotations
        layers: List[List[int]] = [[r.idx] for r in rotations]

        def layer_commutes_with(layer: List[int], ridx: int) -> bool:
            pr = rotations[ridx].axis
            for j in layer:
                if not paulis_commute(pr, rotations[j].axis):
                    return False
            return True

        changed = True
        while changed:
            changed = False
            i = 0
            while i + 1 < len(layers):
                # Try moving elements from layer i+1 to layer i if they commute with all in layer i
                moved_any = False
                j = 0
                while j < len(layers[i + 1]):
                    ridx = layers[i + 1][j]
                    if layer_commutes_with(layers[i], ridx):
                        layers[i].append(ridx)
                        layers[i + 1].pop(j)
                        moved_any = True
                        changed = True
                    else:
                        j += 1

                if len(layers[i + 1]) == 0:
                    layers.pop(i + 1)
                    changed = True
                    continue

                if not moved_any:
                    i += 1
        self.layers = layers
        return layers

    # -----------------------------
    # Visualization helpers (text-based; optional matplotlib for graph)
    # -----------------------------

    def print_rotations(self):
        print("\n=== π/8 Pauli-product rotations (in order) ===")
        for r in self.program.rotations:
            word = pauli_to_word(r.axis)
            supp = sorted(word_support(word))
            print(f" R{r.idx:03d}: ({word})_{r.angle:+.6f}   supp={supp}   src={r.source}")

    def print_measurements(self):
        print("\n=== Final Pauli-product measurements (absorbing final Clifford) ===")
        for m in self.program.final_meas:
            word = pauli_to_word(m.axis)
            supp = sorted(word_support(word))
            print(f"  M{m.idx:03d}: measure {word}   supp={supp}   (from Z on q[{m.qbit}]) -> c[{m.cbit}]")

    def print_layers(self):
        print("\n=== Greedy commuting layers (\"T layers\") ===")
        for i, layer in enumerate(self.layers):
            words = [pauli_to_word(self.program.rotations[j].axis) for j in layer]
            print(f"  L{i:02d} (size={len(layer)}): " + ", ".join(f"R{j:03d}:{w}" for j, w in zip(layer, words)))

    # JSON helpers

    def to_cache_payload(self) -> Dict[str, Any]:
        if self.program is None:
            raise ValueError("self.program is None. Nothing to export.")

        # Store core program in your dataclass form (JSONable via asdict),
        # plus *derived* fields to match the printed view (word/support).
        rotations_view = []
        for r in self.program.rotations:
            w = pauli_to_word(r.axis)
            rotations_view.append(
                {
                    "idx": r.idx,
                    "word": w,
                    "support": sorted(word_support(w)),
                    "angle": r.angle,
                    "source": r.source,
                    # canonical axis (sign,tensor) so load is exact even if word formatting changes
                    "axis": self._axis_to_dict(r.axis),
                }
            )

        meas_view = []
        for m in self.program.final_meas:
            w = pauli_to_word(m.axis)
            meas_view.append(
                {
                    "idx": m.idx,
                    "word": w,
                    "support": sorted(word_support(w)),
                    "qbit": m.qbit,
                    "cbit": m.cbit,
                    "axis": self._axis_to_dict(m.axis),
                }
            )

        layers_view = []
        for li, layer in enumerate(self.layers):
            items = []
            for ridx in layer:
                # layer uses rotation indices; build word lookup (robust)
                # (assumes indices are unique)
                rot = next(rr for rr in self.program.rotations if rr.idx == ridx)
                items.append({"ridx": ridx, "word": pauli_to_word(rot.axis)})
            layers_view.append({"layer": li, "size": len(layer), "rotations": items})

        payload = {
            "schema": self.CACHE_SCHEMA,
            # "program": {
            #     "rotations": GoSCConverter._jsonable(self.program.rotations),
            #     "final_meas": GoSCConverter._jsonable(self.program.final_meas),
            #     "final_clifford": GoSCConverter._jsonable(self.program.final_clifford),
            # },
            # "printed_view": {
            "rotations": rotations_view,
            "final_measurements": meas_view,
            "layers": layers_view,
            "layer_ids": self.layers
            # },
            # "layers": GoSCConverter._jsonable(self.layers),  # canonical layer indices
        }
        return payload

    def to_cache_json_string(self) -> str:
        return json.dumps(self.to_cache_payload(), indent=2, sort_keys=True)

    # def save_cache_json(self, *, compact: bool = False) -> str:
    #     if compact:
    #         payload = self.to_compact_payload()
    #         return payload
    #         # s = json.dumps(payload, separators=(",", ":"))  # no whitespace
    #     else:
    #         payload = self.to_cache_payload()
    #         return payload
            # s = json.dumps(payload, indent=2, sort_keys=True)
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # with open(path, "w", encoding="utf-8") as f:
        #     f.write(s)

    # -------------- Compact format --------------

    def to_compact_payload(self) -> Dict[str, Any]:
        """
        Minimal serialisation of the PauliProgram.

        Compact encoding:
          rotations    : [[axis_sign, tensor, t_sign], ...]
                         axis_sign = ±1 (leading sign of the Pauli word)
                         tensor    = Pauli word without the sign prefix
                         t_sign    = ±1  (+1 → T gate, -1 → Tdg gate)
                         angle is exactly t_sign * π/8 — reconstructed on load.
          measurements : [[axis_sign, tensor, qbit, cbit_or_null], ...]
          layer_ids    : [[r_idx, ...], ...]  — unchanged
        """
        if self.program is None:
            raise ValueError("self.program is None. Nothing to export.")

        rotations_compact = []
        for r in self.program.rotations:
            d = self._axis_to_dict(r.axis)
            t_sign = 1 if r.angle >= 0 else -1
            rotations_compact.append([d["sign"], d["tensor"], t_sign])

        measurements_compact = []
        for m in self.program.final_meas:
            d = self._axis_to_dict(m.axis)
            measurements_compact.append([d["sign"], d["tensor"], m.qbit, m.cbit])

        return {
            "schema": self.COMPACT_CACHE_SCHEMA,
            "rotations": rotations_compact,
            "measurements": measurements_compact,
            "layer_ids": self.layers,
        }

    def _load_compact(self, payload: Dict[str, Any]) -> None:
        """Restore self.program and self.layers from a compact payload."""
        rotations: List[PauliRotation] = []
        for i, entry in enumerate(payload.get("rotations", [])):
            axis_sign, tensor, t_sign = entry
            axis = Pauli(str(axis_sign) + str(tensor))
            rotations.append(
                PauliRotation(
                    axis=axis,
                    angle=t_sign * (math.pi / 8.0),
                    source="",
                    idx=i,
                )
            )

        final_meas: List[PauliMeasurement] = []
        for i, entry in enumerate(payload.get("measurements", [])):
            axis_sign, tensor, qbit, cbit = entry
            axis = Pauli(str(axis_sign) + str(tensor))
            final_meas.append(
                PauliMeasurement(
                    axis=axis,
                    cbit=None if cbit is None else int(cbit),
                    qbit=int(qbit),
                    idx=i,
                )
            )

        self.program = PauliProgram(rotations=rotations, final_meas=final_meas, final_clifford=None)
        self.layers = [[int(x) for x in layer] for layer in payload.get("layer_ids", [])]

    # -------------- Import --------------

    def load_cache_json(self, path: str) -> Dict[str, Any]:
        """
        Load a PBC JSON file.  Automatically detects verbose vs compact format
        from the 'schema' field — no caller changes needed after switching formats.
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        schema = payload.get("schema")

        if schema == self.COMPACT_CACHE_SCHEMA:
            self._load_compact(payload)
            return payload

        if schema != self.CACHE_SCHEMA:
            raise ValueError(f"Unknown cache schema: {schema!r}")

        # Verbose format (original)
        rotations: List[PauliRotation] = []
        for r in payload.get("rotations"):
            axis = Pauli(str(r["axis"]["sign"]) + str(r["axis"]["tensor"]))
            rotations.append(
                PauliRotation(
                    axis=axis,
                    angle=float(r["angle"]),
                    source=str(r["source"]),
                    idx=int(r["idx"]),
                )
            )

        final_meas: List[PauliMeasurement] = []
        for m in payload.get("final_measurements"):
            axis = Pauli(str(m["axis"]["sign"]) + str(m["axis"]["tensor"]))
            final_meas.append(
                PauliMeasurement(
                    axis=axis,
                    cbit=None if m["cbit"] is None else int(m["cbit"]),
                    qbit=int(m["qbit"]),
                    idx=int(m["idx"]),
                )
            )

        self.program = PauliProgram(rotations=rotations, final_meas=final_meas, final_clifford=None)
        self.layers = [[int(x) for x in layer] for layer in payload.get("layer_ids", [])]

        return payload
