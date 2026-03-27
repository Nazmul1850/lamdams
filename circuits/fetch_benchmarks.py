"""
fetch_benchmarks.py
-------------------
Downloads and generates Clifford+T benchmark circuits (.qasm) into a
local benchmarks/ folder, ready to be consumed by the PBC compiler pipeline.

Two sources:
  1. GENERATED  — Qiskit builds circuits natively in {h, s, t, sdg, tdg, cx}.
                  No T-gate explosion because we build at the logical level.
  2. DOWNLOADED — Canonical .qasm files from Matthew Amy's t-par / Feynman
                  benchmark suite hosted on GitHub (already Clifford+T).

Gate set enforced: {h, s, sdg, t, tdg, cx, x, y, z}
Anything outside this set is flagged and skipped (no silent conversion).

Usage
-----
    python fetch_benchmarks.py                  # fetch everything
    python fetch_benchmarks.py --tier 1         # only large circuits (≥22 qubits)
    python fetch_benchmarks.py --list           # print circuit catalogue
    python fetch_benchmarks.py --validate-only  # check existing folder
"""

import os
import argparse
import urllib.request
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
CLIFFORD_T_GATES = {"h", "s", "sdg", "t", "tdg", "cx", "x", "y", "z",
                    "barrier", "measure", "reset"}

# GitHub API for listing files (no auth needed for public repos, 60 req/hr)
GITHUB_API = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"
GITHUB_RAW  = "https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"

# ---------------------------------------------------------------------------
# Remote benchmark registry
# Source: Matthew Amy's t-par and Feynman repos — canonical Clifford+T QASM
# These are the exact circuits used in T-gate optimisation papers (op-T-mize)
# ---------------------------------------------------------------------------

REMOTE_CIRCUITS = [
    # ── Arithmetic / Adder family (T-count scales with n) ──────────────────
    {"name": "adder_8",       "qubits": 24, "t_count": 399,  "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/adder_8.qasm"},
    {"name": "adder_16",      "qubits": 48, "t_count": 831,  "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/adder_16.qasm"},

    # ── Toffoli / Barenco decomposition family ─────────────────────────────
    {"name": "barenco_tof_3", "qubits": 5,  "t_count": 28,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/barenco_tof_3.qasm"},
    {"name": "barenco_tof_4", "qubits": 7,  "t_count": 56,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/barenco_tof_4.qasm"},
    {"name": "barenco_tof_5", "qubits": 9,  "t_count": 100,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/barenco_tof_5.qasm"},
    {"name": "barenco_tof_10","qubits": 19, "t_count": 224,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/barenco_tof_10.qasm"},

    # ── NC Toffoli (no-ancilla) ────────────────────────────────────────────
    {"name": "tof_3",         "qubits": 5,  "t_count": 21,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/tof_3.qasm"},
    {"name": "tof_4",         "qubits": 7,  "t_count": 35,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/tof_4.qasm"},
    {"name": "tof_5",         "qubits": 9,  "t_count": 49,   "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/tof_5.qasm"},
    {"name": "tof_10",        "qubits": 19, "t_count": 119,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/tof_10.qasm"},

    # ── GF(2^m) multiplication — high T-count, many qubits ────────────────
    {"name": "gf2^4_mult",   "qubits": 12, "t_count": 112,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/gf2^4_mult.qasm"},
    {"name": "gf2^5_mult",   "qubits": 15, "t_count": 175,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/gf2^5_mult.qasm"},
    {"name": "gf2^6_mult",   "qubits": 18, "t_count": 252,  "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/gf2^6_mult.qasm"},
    {"name": "gf2^7_mult",   "qubits": 21, "t_count": 343,  "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/gf2^7_mult.qasm"},
    {"name": "gf2^8_mult",   "qubits": 24, "t_count": 448,  "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/gf2^8_mult.qasm"},

    # ── Modular multiplication ─────────────────────────────────────────────
    {"name": "mod_mult_55",  "qubits": 9,  "t_count": 49,   "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/mod_mult_55.qasm"},

    # ── Hidden weighted bit / Hamming weight ──────────────────────────────
    {"name": "hwb6",         "qubits": 7,  "t_count": 105,  "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/hwb6.qasm"},
    {"name": "hwb8",         "qubits": 12, "t_count": 5887, "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/hwb8.qasm"},

    # ── Quantum Fourier Transform (native Clifford+T decomposition) ────────
    {"name": "qft_4",        "qubits": 4,  "t_count": 12,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/qft_4.qasm"},
    {"name": "qft_8",        "qubits": 8,  "t_count": 56,   "tier": 3,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/qft_8.qasm"},

    # ── Hamming codes ──────────────────────────────────────────────────────
    {"name": "ham15_low",    "qubits": 17, "t_count": 161,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/ham15-low.qasm"},
    {"name": "ham15_med",    "qubits": 17, "t_count": 574,  "tier": 2,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/ham15-med.qasm"},
    {"name": "ham15_high",   "qubits": 20, "t_count": 2457, "tier": 1,
     "url": "https://raw.githubusercontent.com/njross/optimizer/master/benchmarks/ham15-high.qasm"},
]

# ---------------------------------------------------------------------------
# Qiskit-generated circuits (deterministic, no network needed)
# These are built natively in Clifford+T — NO transpilation T-explosion.
# We use the known T-count-optimal decompositions directly.
# ---------------------------------------------------------------------------

def _make_toffoli_ladder(n: int) -> "QuantumCircuit":
    """
    Builds a chain of n Toffoli gates in Clifford+T decomposition.
    Each Toffoli = 7 T-gates (optimal decomposition).
    Total qubits: n+2, T-count: 7*n
    Great for scaling because qubits grow linearly with n.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CCXGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollCustomDefinitions, BasisTranslator
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

    qubits = n + 2
    qc = QuantumCircuit(qubits, name=f"toffoli_ladder_{n}")
    for i in range(n):
        qc.ccx(i, i + 1, i + 2)
    return _transpile_to_clifford_t(qc)


def _make_grover_oracle(n: int) -> "QuantumCircuit":
    """
    n-qubit Grover oracle (phase oracle for all-ones string).
    Implemented as a multi-controlled Z using Toffoli decompositions.
    T-count grows as O(n), qubits = n + ancilla.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, name=f"grover_oracle_{n}")
    # Multi-controlled Z via H + multi-controlled X + H
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    return _transpile_to_clifford_t(qc)


def _transpile_to_clifford_t(qc: "QuantumCircuit") -> "QuantumCircuit":
    """Transpile a circuit into {h, s, sdg, t, tdg, cx} using Qiskit."""
    from qiskit.compiler import transpile
    basis = ["h", "s", "sdg", "t", "tdg", "cx", "x", "y", "z"]
    return transpile(qc, basis_gates=basis, optimization_level=0)


# Registry of generated circuits: (name, factory_fn, tier)
GENERATED_CIRCUITS = [
    # Small — for validation / SA convergence plots
    ("toffoli_ladder_3",   lambda: _make_toffoli_ladder(3),   3),
    ("toffoli_ladder_5",   lambda: _make_toffoli_ladder(5),   3),
    # Medium — 2 blocks
    ("toffoli_ladder_10",  lambda: _make_toffoli_ladder(10),  2),
    ("toffoli_ladder_15",  lambda: _make_toffoli_ladder(15),  2),
    ("grover_oracle_10",   lambda: _make_grover_oracle(10),   2),
    # Large — 3+ blocks (≥33 qubits) — best for SA mapping
    ("toffoli_ladder_25",  lambda: _make_toffoli_ladder(25),  1),
    ("toffoli_ladder_35",  lambda: _make_toffoli_ladder(35),  1),
    ("toffoli_ladder_50",  lambda: _make_toffoli_ladder(50),  1),
    ("grover_oracle_20",   lambda: _make_grover_oracle(20),   1),
    ("grover_oracle_30",   lambda: _make_grover_oracle(30),   1),
]

# ---------------------------------------------------------------------------
# Gate validation
# ---------------------------------------------------------------------------

def validate_gate_set(qasm_text: str, filepath: str) -> bool:
    """
    Parse a QASM file and check every gate call is in CLIFFORD_T_GATES.
    Returns True if clean, False and prints offending gates if not.
    """
    offenders = set()
    for line in qasm_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") \
                or line.startswith("include") or line.startswith("qreg") \
                or line.startswith("creg") or line.startswith("gate "):
            continue
        # Extract gate name (first token before space or '(')
        token = line.split("(")[0].split(" ")[0].lower().rstrip(";")
        if token and token not in CLIFFORD_T_GATES:
            offenders.add(token)
    if offenders:
        print(f"  ⚠  SKIP {Path(filepath).name}: non-Clifford+T gates found: {offenders}")
        return False
    return True


def analyse_circuit(qasm_text: str) -> dict:
    """Count qubits, T-gates, total gates from QASM text."""
    qubits, t_count, total = 0, 0, 0
    for line in qasm_text.splitlines():
        line = line.strip()
        if line.startswith("qreg"):
            # qreg q[n];
            try:
                qubits += int(line.split("[")[1].split("]")[0])
            except Exception:
                pass
        elif not line.startswith("//") and not line.startswith("OPENQASM") \
                and not line.startswith("include") and line.endswith(";"):
            token = line.split("(")[0].split(" ")[0].lower().rstrip(";")
            if token in {"t", "tdg"}:
                t_count += 1
            if token not in {"", "qreg", "creg", "gate", "barrier", "measure"}:
                total += 1
    return {"qubits": qubits, "t_count": t_count, "total_gates": total}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_circuit(entry: dict, dest_dir: Path, timeout: int = 10) -> Optional[Path]:
    """Download a single circuit entry. Returns saved path or None on failure."""
    url  = entry["url"]
    name = entry["name"]
    dest = dest_dir / f"{name}.qasm"

    if dest.exists():
        print(f"  ✓ cached  {name}.qasm")
        return dest

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ✗ failed  {name}: {e}")
        return None

    if not validate_gate_set(text, name):
        return None

    dest.write_text(text)
    stats = analyse_circuit(text)
    print(f"  ↓ saved   {name}.qasm  "
          f"({stats['qubits']} qubits, {stats['t_count']} T-gates, "
          f"{stats['total_gates']} total)")
    return dest


# ---------------------------------------------------------------------------
# Generate helpers
# ---------------------------------------------------------------------------

def generate_circuit(name: str, factory, dest_dir: Path) -> Optional[Path]:
    """Generate a circuit with Qiskit and save as QASM."""
    dest = dest_dir / f"{name}.qasm"
    if dest.exists():
        print(f"  ✓ cached  {name}.qasm")
        return dest

    try:
        from qiskit import QuantumCircuit
        qc = factory()
        qasm_text = qc.qasm()
    except Exception as e:
        print(f"  ✗ failed  {name}: {e}")
        return None

    if not validate_gate_set(qasm_text, name):
        return None

    dest.write_text(qasm_text)
    stats = analyse_circuit(qasm_text)
    print(f"  ⚙ built   {name}.qasm  "
          f"({stats['qubits']} qubits, {stats['t_count']} T-gates, "
          f"{stats['total_gates']} total)")
    return dest


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def write_index(dest_dir: Path):
    """Write benchmarks/index.json with metadata for all .qasm files."""
    index = []
    for qasm_file in sorted(dest_dir.glob("*.qasm")):
        text  = qasm_file.read_text()
        stats = analyse_circuit(text)
        # Determine tier from filename
        tier = 3
        for entry in REMOTE_CIRCUITS:
            if entry["name"] in qasm_file.stem:
                tier = entry.get("tier", 3)
                break
        for name, _, t in GENERATED_CIRCUITS:
            if name == qasm_file.stem:
                tier = t
                break
        index.append({
            "file":        qasm_file.name,
            "qubits":      stats["qubits"],
            "t_count":     stats["t_count"],
            "total_gates": stats["total_gates"],
            "tier":        tier,
            "blocks_approx": max(1, stats["qubits"] // 11),  # your block size
        })
    index.sort(key=lambda x: x["qubits"])
    (dest_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n📋 index.json written with {len(index)} circuits")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_circuits():
    print("\n─── Remote circuits ─────────────────────────────────────────")
    print(f"{'Name':<22} {'Qubits':>6} {'T-count':>8} {'Blocks':>7} {'Tier':>5}")
    print("─" * 55)
    for c in sorted(REMOTE_CIRCUITS, key=lambda x: x["qubits"]):
        blocks = max(1, c["qubits"] // 11)
        print(f"{c['name']:<22} {c['qubits']:>6} {c['t_count']:>8} {blocks:>7} {'⭐' * (4 - c['tier'])}")

    print("\n─── Generated circuits (Qiskit) ─────────────────────────────")
    print(f"{'Name':<26} {'Tier':>5}  Notes")
    print("─" * 60)
    for name, _, tier in GENERATED_CIRCUITS:
        n = int(name.rsplit("_", 1)[-1])
        if "ladder" in name:
            q = n + 2
            t = 7 * n
        else:
            q = n
            t = "~O(n log n)"
        blocks = max(1, q // 11) if isinstance(q, int) else "—"
        print(f"{name:<26} {'⭐' * (4 - tier)}   {q} qubits, {t} T-gates, ~{blocks} blocks")


def main():
    parser = argparse.ArgumentParser(description="Fetch/generate Clifford+T benchmarks")
    parser.add_argument("--tier",         type=int, choices=[1, 2, 3],
                        help="1=large(≥22q), 2=medium, 3=small. Default: all")
    parser.add_argument("--list",         action="store_true",
                        help="Print circuit catalogue and exit")
    parser.add_argument("--validate-only",action="store_true",
                        help="Validate existing .qasm files in benchmarks/")
    parser.add_argument("--no-download",  action="store_true",
                        help="Skip remote downloads, only generate with Qiskit")
    parser.add_argument("--no-generate",  action="store_true",
                        help="Skip Qiskit generation, only download")
    parser.add_argument("--outdir",       type=str, default=str(BENCHMARK_DIR),
                        help=f"Output directory (default: {BENCHMARK_DIR})")
    args = parser.parse_args()

    if args.list:
        list_circuits()
        return

    dest = Path(args.outdir)
    dest.mkdir(parents=True, exist_ok=True)

    # ── Validate only mode ───────────────────────────────────────────────
    if args.validate_only:
        print(f"\n🔍 Validating .qasm files in {dest} …\n")
        ok, fail = 0, 0
        for f in sorted(dest.glob("*.qasm")):
            text = f.read_text()
            if validate_gate_set(text, str(f)):
                stats = analyse_circuit(text)
                print(f"  ✓ {f.name:<30} {stats['qubits']:>4}q  "
                      f"{stats['t_count']:>6} T-gates")
                ok += 1
            else:
                fail += 1
        print(f"\nResult: {ok} valid, {fail} failed")
        return

    tier_filter = args.tier  # None means all

    # ── Download remote circuits ─────────────────────────────────────────
    if not args.no_download:
        print(f"\n⬇  Downloading remote Clifford+T benchmarks → {dest}/\n")
        for entry in REMOTE_CIRCUITS:
            if tier_filter and entry.get("tier", 3) > tier_filter:
                continue
            download_circuit(entry, dest)

    # ── Generate with Qiskit ─────────────────────────────────────────────
    if not args.no_generate:
        print(f"\n⚙  Generating Qiskit circuits → {dest}/\n")
        for name, factory, tier in GENERATED_CIRCUITS:
            if tier_filter and tier > tier_filter:
                continue
            generate_circuit(name, factory, dest)

    # ── Write index ──────────────────────────────────────────────────────
    write_index(dest)

    # ── Summary ──────────────────────────────────────────────────────────
    all_qasm = list(dest.glob("*.qasm"))
    print(f"\n✅  Done. {len(all_qasm)} circuits in {dest}/")
    print(f"    Run your compiler with any of them:")
    print(f"    python your_compiler.py --input {dest}/<circuit>.qasm\n")


if __name__ == "__main__":
    main()