"""
Generate benchmark circuits for the PBC compiler experimental evaluation.

Usage:
    python circuits/generate_benchmarks.py [--qft] [--random] [--all]
    python circuits/generate_benchmarks.py --qft-scaling   # scaling sweep sizes
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# Ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "benchmarks")

# ccz gate definition (7-T decomposition) to inject when parsing older QASM files
_CCZ_GATE_DEF = (
    'gate ccz a,b,c {\n'
    '  h c; cx b,c; tdg c; cx a,c; t c; cx b,c; tdg c; cx a,c;'
    '  t b; t c; h c; cx a,b; t a; tdg b; cx a,b;\n'
    '}\n'
)


def transpile_to_clifford_t(src_qasm_path: str, out_path: str) -> str:
    """
    Load a QASM file that may contain higher-level gates (ccz, ccx, etc.),
    transpile to {h, s, sdg, t, tdg, cx}, and save to out_path.
    Returns out_path.
    """
    from qiskit import qasm2
    from qiskit.compiler import transpile

    with open(src_qasm_path) as f:
        qasm_str = f.read()

    # Inject ccz/ccx definitions if present but not defined
    if 'ccz' in qasm_str and 'gate ccz' not in qasm_str:
        qasm_str = qasm_str.replace('include "qelib1.inc";',
                                    'include "qelib1.inc";\n' + _CCZ_GATE_DEF, 1)
    if 'ccx' in qasm_str and 'gate ccx' not in qasm_str:
        # ccx = Toffoli; standard qelib1 defines it, reload via loads workaround
        from qiskit.circuit.library import CCXGate
        ccx_inst = qasm2.CustomInstruction('ccx', 3, 0, CCXGate)
        qc = qasm2.loads(qasm_str, custom_instructions=[ccx_inst])
    else:
        qc = qasm2.loads(qasm_str)

    qc_ct = transpile(qc, basis_gates=['h', 's', 'sdg', 't', 'tdg', 'cx'],
                      optimization_level=0)
    t_count = sum(1 for g in qc_ct.data if g.operation.name in ('t', 'tdg'))
    print(f'  transpile_to_clifford_t: {os.path.basename(src_qasm_path)}'
          f'  qubits={qc_ct.num_qubits}  T-gates={t_count}')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    qasm2.dump(qc_ct, out_path)
    print(f'  -> saved: {out_path}')
    return out_path


# ── QFT circuits ─────────────────────────────────────────────────────────────

def generate_qft(n_qubits: int, approx_degree: int | None = None, *, label: str | None = None) -> str:
    """
    Generate an approximate QFT circuit, transpile to Clifford+T, and save to
    circuits/benchmarks/.  Returns the output QASM path.

    approx_degree: number of approximation levels to drop (default: n_qubits - 2,
    keeping only S and T gates).
    """
    from qiskit.circuit.library import QFT
    from qiskit.compiler import transpile
    from qiskit import qasm2

    if approx_degree is None:
        approx_degree = max(0, n_qubits - 2)

    qc = QFT(n_qubits, approximation_degree=approx_degree, do_swaps=False)
    qc_ct = transpile(
        qc,
        basis_gates=["h", "s", "sdg", "t", "tdg", "cx"],
        optimization_level=0,
    )

    t_count = sum(1 for g in qc_ct.data if g.operation.name in ("t", "tdg"))
    print(
        f"  QFT({n_qubits})  approx_degree={approx_degree}"
        f"  qubits={qc_ct.num_qubits}  T-gates={t_count}"
    )

    if label is None:
        label = f"qft_{n_qubits}_approx"
    out_path = os.path.join(BENCHMARK_DIR, f"{label}.qasm")
    qasm2.dump(qc_ct, out_path)
    print(f"  → saved: {out_path}")
    return out_path


# ── Random Clifford+T circuit ─────────────────────────────────────────────────

def generate_random_ct(
    n_qubits: int,
    target_t_rotations: int,
    seed: int = 42,
    *,
    label: str | None = None,
) -> str:
    """
    Generate a random Clifford+T circuit with approximately target_t_rotations
    T/Tdg gates. Uses only {h, s, sdg, t, tdg, cx} (native Clifford+T gate set).
    Returns the output QASM path.
    """
    import random as _rnd
    from qiskit import QuantumCircuit
    from qiskit import qasm2

    _rnd.seed(seed)
    qc = QuantumCircuit(n_qubits)
    t_count = 0

    # ~30% of gates are T/Tdg; ~25% two-qubit; rest single-qubit Clifford
    estimated_total = int(target_t_rotations / 0.30)
    for _ in range(estimated_total):
        r = _rnd.random()
        if r < 0.25:
            q1, q2 = _rnd.sample(range(n_qubits), 2)
            qc.cx(q1, q2)
        elif r < 0.55:
            q = _rnd.randint(0, n_qubits - 1)
            if _rnd.random() < 0.5:
                qc.t(q)
            else:
                qc.tdg(q)
            t_count += 1
            if t_count >= target_t_rotations:
                break
        else:
            gate = _rnd.choice(["h", "s", "sdg"])
            q = _rnd.randint(0, n_qubits - 1)
            getattr(qc, gate)(q)

    print(
        f"  random_ct({n_qubits}q, target={target_t_rotations:,}T)"
        f"  actual_T={t_count:,}  seed={seed}"
    )

    if label is None:
        label = f"random_ct_{n_qubits}q_{target_t_rotations // 1000}k"
    out_path = os.path.join(BENCHMARK_DIR, f"{label}.qasm")
    qasm2.dump(qc, out_path)
    print(f"  → saved: {out_path}")
    return out_path


# ── Scaling sweep ─────────────────────────────────────────────────────────────

# QFT sizes for Figure 5 scaling sweep.
# n_blocks ≈ ceil(n_qubits / 11). Target block counts: ~2, ~3, ~4, ~6, ~9.
SCALING_SIZES = [22, 33, 44, 66, 99]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PBC compiler benchmark circuits")
    parser.add_argument("--qft",         action="store_true", help="Generate qft_100_approx.qasm")
    parser.add_argument("--random",      action="store_true", help="Generate random_ct_500q_10k_validate.qasm")
    parser.add_argument("--qft-scaling", action="store_true", help="Generate QFT scaling sweep circuits")
    parser.add_argument("--all",         action="store_true", help="Generate all circuits")
    args = parser.parse_args()

    if not any([args.qft, args.random, args.qft_scaling, args.all]):
        parser.print_help()
        sys.exit(0)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    if args.all or args.qft:
        print("Generating qft_100_approx.qasm ...")
        generate_qft(100, approx_degree=98, label="qft_100_approx")

    if args.all or args.qft_scaling:
        print("Generating QFT scaling sweep ...")
        for n in SCALING_SIZES:
            generate_qft(n, label=f"qft_{n}_approx")

    if args.all or args.random:
        print("Generating random_ct_500q_10k_validate.qasm (validation, 10k T) ...")
        generate_random_ct(
            n_qubits=500,
            target_t_rotations=10_000,
            seed=42,
            label="random_ct_500q_10k_validate",
        )
        print()
        print("NOTE: Full 100M-T circuit not generated.")
        print("      Only run after validation circuit passes the full pipeline.")


if __name__ == "__main__":
    main()
