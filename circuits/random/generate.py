"""
circuits/random/generate.py
---------------------------
Generate random Clifford+T circuits in OpenQASM 2.0 format.

Naming convention:
    rand_{n}q_{t}t_s{seed}.qasm
    e.g.  rand_10q_100t_s42.qasm
          rand_10q_1Mt_s0.qasm
          rand_10q_100Mt_s0.qasm

Usage:
    python generate.py --qubits 10 --t-gates 100 --seed 42
    python generate.py --qubits 20 --t-gates 1000000 --seed 0 --out-dir .
    python generate.py --qubits 10 --t-gates 100000000 --seed 0 --dry-run

Gate composition per T gate (on average):
    - 0–3 single-qubit Clifford gates  (h, s, sdg) before each T/Tdg
    - ~30 % of Clifford slots filled by a CX instead
    - Barrier every ~10 T gates
    - Final measurements on all qubits
"""

from __future__ import annotations

import argparse
import os
import random
import sys


# ── naming helpers ────────────────────────────────────────────────────────────

def _t_suffix(n: int) -> str:
    """Human-readable suffix for T-gate count."""
    for value, suffix in [
        (1_000_000_000, "G"),
        (1_000_000,     "M"),
        (1_000,         "k"),
    ]:
        if n >= value and n % value == 0:
            return f"{n // value}{suffix}"
        if n >= value and n % (value // 10) == 0:
            return f"{n // (value // 10)}{suffix[0] if suffix != 'G' else 'G'}"
    return str(n)


def circuit_filename(n_qubits: int, n_t: int, seed: int) -> str:
    return f"rand_{n_qubits}q_{_t_suffix(n_t)}t_s{seed}.qasm"


def estimate_file_size(n_qubits: int, n_t: int) -> str:
    """Rough estimate: ~2.5 gate lines per T gate, ~16 bytes each."""
    lines = n_t * 2.5 + n_qubits  # gates + measurements
    bytes_ = lines * 18
    for unit, thresh in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if bytes_ >= thresh:
            return f"{bytes_ / thresh:.1f} {unit}"
    return f"{bytes_:.0f} B"


# ── gate line generators ───────────────────────────────────────────────────────

def _single_qubit_clifford(rng: random.Random, n: int) -> str:
    gate = rng.choice(["h", "s", "sdg"])
    q = rng.randrange(n)
    return f"{gate} q[{q}];"


def _cx(rng: random.Random, n: int) -> str:
    ctrl = rng.randrange(n)
    tgt = rng.randrange(n - 1)
    if tgt >= ctrl:
        tgt += 1  # ensure ctrl != tgt
    return f"cx q[{ctrl}],q[{tgt}];"


def _t_gate(rng: random.Random, n: int) -> str:
    gate = rng.choice(["t", "tdg"])
    q = rng.randrange(n)
    return f"{gate} q[{q}];"


# ── streaming writer ───────────────────────────────────────────────────────────

def generate(
    n_qubits: int,
    n_t: int,
    seed: int,
    out_path: str,
) -> None:
    rng = random.Random(seed)

    # Parameters controlling density
    # avg Clifford gates between consecutive T gates: Poisson-like via randint
    max_cliffords_per_slot = 4
    cx_probability = 0.25          # chance a Clifford slot becomes a CX
    barrier_every_t = 10           # place barrier every ~N T gates

    with open(out_path, "w", buffering=256 * 1024) as f:
        # Header
        f.write("OPENQASM 2.0;\n")
        f.write('include "qelib1.inc";\n')
        f.write(f"qreg q[{n_qubits}];\n")
        f.write(f"creg c[{n_qubits}];\n")
        f.write("\n")

        for t_idx in range(n_t):
            # Random Clifford gates before this T
            n_cliffords = rng.randint(0, max_cliffords_per_slot)
            for _ in range(n_cliffords):
                if n_qubits >= 2 and rng.random() < cx_probability:
                    f.write(_cx(rng, n_qubits) + "\n")
                else:
                    f.write(_single_qubit_clifford(rng, n_qubits) + "\n")

            # T / Tdg gate
            f.write(_t_gate(rng, n_qubits) + "\n")

            # Barrier every `barrier_every_t` T gates
            if (t_idx + 1) % barrier_every_t == 0:
                f.write("barrier q;\n")

        f.write("\n")

        # Final measurements
        for i in range(n_qubits):
            f.write(f"measure q[{i}] -> c[{i}];\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random Clifford+T circuits as OpenQASM 2.0 files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--qubits",   type=int, required=True, help="Number of qubits")
    parser.add_argument("-t", "--t-gates",  type=int, required=True, help="Number of T/Tdg gates")
    parser.add_argument("-s", "--seed",     type=int, default=0,     help="RNG seed")
    parser.add_argument(
        "-o", "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory (default: same directory as this script)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print filename and estimated size without writing",
    )
    args = parser.parse_args()

    if args.qubits < 1:
        parser.error("--qubits must be ≥ 1")
    if args.t_gates < 1:
        parser.error("--t-gates must be ≥ 1")

    fname = circuit_filename(args.qubits, args.t_gates, args.seed)
    out_path = os.path.join(args.out_dir, fname)
    est_size = estimate_file_size(args.qubits, args.t_gates)

    print(f"Circuit : {fname}")
    print(f"Qubits  : {args.qubits}")
    print(f"T gates : {args.t_gates:,}")
    print(f"Seed    : {args.seed}")
    print(f"Est.size: {est_size}")
    print(f"Output  : {out_path}")

    if args.dry_run:
        print("(dry run — no file written)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print("Generating...", end=" ", flush=True)
    generate(args.qubits, args.t_gates, args.seed, out_path)
    actual_size = os.path.getsize(out_path)
    actual_mb = actual_size / 1e6
    print(f"done.  ({actual_mb:.2f} MB written)")


if __name__ == "__main__":
    main()
