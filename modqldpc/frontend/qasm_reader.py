from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Set, Iterable

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


DEFAULT_BASIS: tuple[str, ...] = (
    "h", "s", "sdg", "x", "y", "z",
    "cx", "cz", "swap",
    "t", "tdg",
    "measure",
)


@dataclass(frozen=True)
class TranspileOptions:
    basis_gates: Sequence[str] = DEFAULT_BASIS
    optimization_level: int = 1          # 0..3 (Qiskit preset)
    seed_transpiler: Optional[int] = 0
    # If you want a fixed qubit mapping, set initial_layout=list[int] or Layout
    initial_layout: Optional[object] = None
    # Preserve barriers if you use them for debugging/structure
    preserve_barriers: bool = True


class QiskitCircuitHandler:
    """
    Loads QASM into a QuantumCircuit and transpiles into a target basis gate set.

    Design goals:
      - Deterministic by default (seeded).
      - Validates output basis strictly.
      - Keeps interface small and unit-testable.
    """

    def __init__(self, *, demo_qasm: Optional[str] = None, opts: Optional[TranspileOptions] = None):
        self._demo_qasm = demo_qasm
        self._opts = opts or TranspileOptions()

    # ---------- Loading ----------

    def load_qasm(self, path: Optional[str], demo: bool) -> QuantumCircuit:
        """
        Your requested function: loads from file unless demo=True, in which case loads from demo_qasm.
        """
        if demo:
            if not self._demo_qasm:
                raise ValueError("demo=True but no demo_qasm was provided to QiskitCircuitHandler.")
            return QuantumCircuit.from_qasm_str(self._demo_qasm)
        if not path:
            raise ValueError("Need --qasm PATH or --demo")
        return QuantumCircuit.from_qasm_file(path)

    # ---------- Transpilation ----------

    def transpile_to_basis(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Transpile to the configured basis_gates and validate that the output uses only that basis
        (ignoring barrier if preserve_barriers=True).
        """
        opts = self._opts

        # Prefer preset pass manager (more stable across versions) when available
        try:
            pm: PassManager = generate_preset_pass_manager(
                optimization_level=int(opts.optimization_level),
                basis_gates=list(opts.basis_gates),
                seed_transpiler=opts.seed_transpiler,
                initial_layout=opts.initial_layout,
            )
            out = pm.run(qc)
        except Exception:
            # Fallback to transpile() if preset PM API differs in your Qiskit version
            out = transpile(
                qc,
                basis_gates=list(opts.basis_gates),
                optimization_level=int(opts.optimization_level),
                seed_transpiler=opts.seed_transpiler,
                initial_layout=opts.initial_layout,
            )

        if opts.preserve_barriers:
            # Qiskit may drop barriers depending on passes; if you require them,
            # use an explicit pass manager later. For now, we just allow them.
            pass

        self.assert_in_basis(out, opts.basis_gates, allow_barrier=opts.preserve_barriers)
        return out

    # ---------- Validation / utilities ----------

    @staticmethod
    def assert_in_basis(
        qc: QuantumCircuit,
        basis_gates: Sequence[str],
        *,
        allow_barrier: bool = True,
    ) -> None:
        allowed: Set[str] = set(basis_gates)
        if allow_barrier:
            allowed.add("barrier")

        bad = []
        for inst in qc.data:
            name = inst.operation.name
            if name not in allowed:
                bad.append(name)

        if bad:
            bad_set = sorted(set(bad))
            raise ValueError(
                "Transpiled circuit contains gates outside requested basis.\n"
                f"  bad_gates={bad_set}\n"
                f"  allowed={sorted(allowed)}"
            )

    @staticmethod
    def gate_histogram(qc: QuantumCircuit) -> dict[str, int]:
        """Simple gate count by name."""
        counts: dict[str, int] = {}
        for inst in qc.data:
            name = inst.operation.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def load_and_transpile(self, path: Optional[str], demo: bool) -> QuantumCircuit:
        """Convenience: load QASM then transpile to basis."""
        qc = self.load_qasm(path=path, demo=demo)
        num_logicals = qc.num_qubits
        return self.transpile_to_basis(qc), num_logicals
