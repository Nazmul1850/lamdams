"""
run_compile.py — Compile and schedule a quantum circuit through the PBC pipeline.

Pipeline stages:
  1. Frontend   : QASM → Pauli-Based Computation (PBC) via Litinski / lsqecc
  2. Hardware   : Build grid or ring topology from Gross code blocks
  3. Mapping    : Assign logical qubits to physical hardware blocks
  4. Lowering   : Decompose Pauli layers into native gate DAGs
  5. Scheduling : Assign time-steps to operations
  6. Execution  : Apply Clifford frame corrections, collect depth metrics

Pre-compiled PBC files are cached in circuits/compiled/ and reused on
subsequent runs, skipping the compilation step.

Each run writes two files to runs/<circuit>__<config>__seed<N>__<ts>/:
  result.json   — circuit stats, hardware config, and depth result
  trace.ndjson  — per-stage event log (newline-delimited JSON)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

from modqldpc.core.trace import Trace
from modqldpc.core.types import PauliRotation
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.mapping.factory import get_mapper
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.policy import (
    HeuristicRepeatNativePolicy, LoweringPolicies,
    ChooseMagicBlockMinId, ShortestPathGatherRouting,
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.mapping.algos.sa_mapping import (
    TUNED_SA_STEPS, TUNED_SA_T0, TUNED_SA_TEND, TUNED_SCORE_KWARGS,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT        = pathlib.Path(__file__).parent
COMPILED_DIR = _ROOT / "circuits" / "compiled"
RUNS_DIR     = _ROOT / "runs"

# ── Preset configurations ──────────────────────────────────────────────────────
#
# LaM = our Lattice Mapping algorithm (simulated annealing, tuned hyperparameters)

CONFIGS: dict[str, tuple[str, str]] = {
    "naive": ("pure_random",         "sequential_scheduler"),
    "A":     ("pure_random",         "greedy_critical"),
    "B":     ("pure_random",         "cp_sat"),
    "C":     ("simulated_annealing", "greedy_critical"),
    "D":     ("simulated_annealing", "cp_sat"),
}

CONFIG_LABELS: dict[str, str] = {
    "naive": "random placement  +  sequential scheduling   [baseline]",
    "A":     "random placement  +  greedy scheduling       [default]",
    "B":     "random placement  +  CP-SAT scheduling",
    "C":     "LaM placement     +  greedy scheduling",
    "D":     "LaM placement     +  CP-SAT scheduling",
}

# ── Family-specific SA scaling factors ────────────────────────────────────────
#
# Each circuit family has tuned SA score weights that improve LaM mapping quality
# (configs C and D). Use --family to activate them; omit for tuned defaults.
#
# Families and their circuits:
#   adder  Adder8, Adder16, Adder32, Adder64, Adder128
#   qft    QFT8, QFT16, QFT32, QFT64, QFT128
#   gf     gf6_mult … gf10_mult       (GF-multiplication; span-suppressed weights)
#   rand   rand_50q_500t_s42 … rand_50q_2kt_s42

FAMILY_SCORE_KWARGS: dict[str, dict[str, float]] = {
    "adder": dict(TUNED_SCORE_KWARGS),
    "qft":   dict(TUNED_SCORE_KWARGS),
    "gf": {
        "W_UNUSED_BLOCKS": 1_000_000.0,
        "W_OCC_RANGE":        40_000.0,
        "W_OCC_STD":          20_000.0,
        "W_MULTI_BLOCK":           0.0,
        "W_SPAN":                800.0,   
        "W_MST":                  50.0,
        "W_SPLIT":                30.0,
        "W_SUPPORT_PEAK":        100.0,
        "W_SUPPORT_RANGE":        20.0,
        "W_SUPPORT_STD":           0.0,
    },
    "rand": {
        "W_UNUSED_BLOCKS": 1_000_000.0,
        "W_OCC_RANGE":        22_000.0,
        "W_OCC_STD":          10_000.0,
        "W_MULTI_BLOCK":           0.0,
        "W_SPAN":              2_000.0,
        "W_MST":                 275.0,
        "W_SPLIT":                10.0,
        "W_SUPPORT_PEAK":        100.0,
        "W_SUPPORT_RANGE":        20.0,
        "W_SUPPORT_STD":           0.0,
    },
}

# Auto-detect family from circuit name prefix
_FAMILY_PREFIXES: list[tuple[str, str]] = [
    ("Adder", "adder"),
    ("QFT",   "qft"),
    ("gf",    "gf"),
    ("rand",  "rand"),
]

def _detect_family(name: str) -> str | None:
    for prefix, family in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family
    return None

# Hardware defaults (Gross code)
_N_DATA           = 11
_COUPLER_CAPACITY = 1

# CP-SAT: fall back to greedy if solver raises
_CPSAT_FALLBACK = "greedy_critical"


# ── Circuit discovery ──────────────────────────────────────────────────────────

def _find_pbc(name: str) -> pathlib.Path | None:
    for candidate in [
        COMPILED_DIR / f"{name}_PBC.json",
        COMPILED_DIR / f"{name}.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _list_circuits() -> dict[str, pathlib.Path]:
    if not COMPILED_DIR.is_dir():
        return {}
    result = {}
    for f in sorted(COMPILED_DIR.glob("*.json")):
        name = f.stem[:-4] if f.stem.endswith("_PBC") else f.stem
        result[name] = f
    return result


# ── Frontend compilation ───────────────────────────────────────────────────────

def _compile_qasm(qasm_path: pathlib.Path) -> pathlib.Path:
    """Compile a QASM file to PBC and cache it in circuits/compiled/."""
    from modqldpc.frontend.qasm_reader import load_qasm_file
    name     = qasm_path.stem
    out_path = COMPILED_DIR / f"{name}_PBC.json"
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[compile]  {qasm_path.name}  →  compiling via lsqecc ...")
    conv = GoSCConverter(verbose=False)
    conv.convert_qasm(load_qasm_file(str(qasm_path)))
    conv.greedy_layering()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conv.to_compact_payload(), f, indent=2)
    print(f"[compile]  cached  →  {out_path.relative_to(_ROOT)}")
    return out_path


# ── PBC loading ────────────────────────────────────────────────────────────────

def _load_pbc(pbc_path: pathlib.Path) -> Tuple[GoSCConverter, int, int, int]:
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(str(pbc_path))
    n_logicals = len(next(iter(conv.program.rotations)).axis.lstrip("+-"))

    # Count T-gates directly from the raw JSON — don't rely on loaded angles.
    # In both v1.compact and v2.compact each rotation entry is [sign, tensor, denom];
    # a T-gate has denom == 8.  (v1 loader computes angle = sign * denom * π/8, which
    # gives ±π for denom=8, making any angle-based formula return 0.)
    raw  = json.loads(pbc_path.read_text(encoding="utf-8"))
    rots = raw.get("rotations", [])
    if raw.get("schema", "").endswith(".compact"):
        t_count = sum(1 for e in rots if e[2] == 8)
    else:
        # verbose v1: angle stored explicitly as float
        _pi8 = math.pi / 8.0
        t_count = sum(
            1 for r in raw.get("rotations", [])
            if abs(abs(float(r.get("angle", 0))) - _pi8) < 1e-9
        )
    # Fallback: all-Clifford circuits (no T-gates) — use total rotation count
    if t_count == 0:
        t_count = len(rots)

    return conv, n_logicals, t_count, len(conv.layers)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _run_mapping(
    mapper_name: str, seed: int, conv: GoSCConverter, hw, n_logicals: int,
    sa_steps: int,
    score_kwargs: dict | None = None,
) -> Tuple[MappingPlan, float]:
    cfg  = MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=TUNED_SA_T0, sa_tend=TUNED_SA_TEND)
    meta: dict = {"rotations": conv.program.rotations, "verbose": False, "debug": False}
    if mapper_name == "simulated_annealing" and score_kwargs:
        meta["score_kwargs"] = score_kwargs
    t0   = time.perf_counter()
    plan = get_mapper(mapper_name).solve(MappingProblem(n_logicals=n_logicals), hw, cfg, meta)
    return plan, round(time.perf_counter() - t0, 3)


def _run_scheduling(
    conv: GoSCConverter, hw, plan: MappingPlan,
    sched_name: str, seed: int, cp_time: float,
) -> Tuple[int, float]:
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=make_gross_actual_cost_fn(plan)),
    )
    effective: Dict[int, PauliRotation] = {r.idx: r for r in conv.program.rotations}
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    total_depth = 0
    t0 = time.perf_counter()
    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id, rotations=effective,
            rotation_indices=layer, hw=hw, policies=policies,
        )
        problem = SchedulingProblem(
            dag=res.dag, hw=hw, seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time": 0, "layer_idx": layer_id,
                "tie_breaker": "duration", "cp_sat_time_limit": cp_time,
                "debug_decode": False, "safe_fill": True, "cp_sat_log": False,
            },
        )
        try:
            S = get_scheduler(sched_name).solve(problem)
        except Exception:
            if sched_name == "cp_sat":
                S = get_scheduler(_CPSAT_FALLBACK).solve(problem)
            else:
                raise
        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        ex = executor.execute_layer(
            layer=layer_id, dag=res.dag, schedule=S, frame_in=frame,
            next_layer_rotations=[effective[i] for i in next_idxs],
        )
        for r in ex.next_rotations_effective:
            effective[r.idx] = r
        frame = ex.frame_after
        total_depth += ex.depth
    return total_depth, round(time.perf_counter() - t0, 3)


# ── Per-circuit runner ─────────────────────────────────────────────────────────

def run_circuit(
    name: str,
    pbc_path: pathlib.Path,
    config: str,
    mapper_name: str,
    sched_name: str,
    topology: str,
    seed: int,
    cp_time: float,
    sa_steps: int,
    n_data: int,
    sparse_pct: float,
    coupler_capacity: int,
    family: str | None = None,
    tag: str | None = None,
) -> dict:
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RUNS_DIR / f"{tag or name}__{config}__seed{seed}__{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace   = Trace(str(run_dir / "trace.ndjson"))

    trace.event("run_start", circuit=name, config=config, topology=topology, seed=seed)

    # Load PBC
    conv, n_logicals, t_count, n_layers = _load_pbc(pbc_path)
    trace.event("frontend_loaded", n_qubits=n_logicals, t_count=t_count, n_layers=n_layers)
    print(f"[frontend]  n_qubits={n_logicals}  t_count={t_count}  n_layers={n_layers}")

    # Build hardware
    hw, hw_spec = make_hardware(
        n_logicals, topology=topology, sparse_pct=sparse_pct,
        n_data=n_data, coupler_capacity=coupler_capacity,
    )
    trace.event("hardware_built", label=hw_spec.label(),
                n_blocks=len(hw.blocks), n_couplers=len(hw.couplers))
    print(f"[hardware]  {hw_spec.label()}  "
          f"blocks={len(hw.blocks)}  couplers={len(hw.couplers)}")

    # Mapping
    score_kwargs = FAMILY_SCORE_KWARGS.get(family) if family else None
    family_note  = f"  [family={family}]" if family and mapper_name == "simulated_annealing" else ""
    print(f"[mapping]   {mapper_name}{family_note} ...", end="", flush=True)
    plan, t_map = _run_mapping(mapper_name, seed, conv, hw, n_logicals, sa_steps, score_kwargs)
    trace.event("mapping_done", mapper=mapper_name, family=family, time_sec=t_map)
    print(f" done ({t_map:.1f}s)")

    # Scheduling
    print(f"[schedule]  {sched_name}  ({n_layers} layers) ...")
    depth, t_sched = _run_scheduling(conv, hw, plan, sched_name, seed, cp_time)
    trace.event("scheduling_done", scheduler=sched_name, logical_depth=depth, time_sec=t_sched)
    print(f"[result]    logical_depth={depth:,}  ({t_sched:.1f}s)")

    # Save result.json
    record = {
        "circuit":   name,
        "config":    config,
        "mapper":    mapper_name,
        "scheduler": sched_name,
        "topology":  topology,
        "family":    family,
        "seed":      seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_qubits":  n_logicals,
        "t_count":   t_count,
        "n_layers":  n_layers,
        "hardware": {
            "label":            hw_spec.label(),
            "n_blocks":         len(hw.blocks),
            "n_couplers":       len(hw.couplers),
            "n_data":           n_data,
            "coupler_capacity": coupler_capacity,
            "sparse_pct":       sparse_pct,
        },
        "results": {
            "logical_depth":       depth,
            "mapping_time_sec":    t_map,
            "scheduling_time_sec": t_sched,
            "total_time_sec":      round(t_map + t_sched, 3),
        },
    }
    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)
    trace.event("run_complete", logical_depth=depth, result=str(result_path))

    sep = "─" * 56
    print(f"\n{'='*60}")
    print(f"  {name}  |  config {config}  |  {topology}")
    print(f"  {sep}")
    print(f"  logical depth  :  {depth:,}")
    print(f"  total time     :  {record['results']['total_time_sec']:.1f}s")
    print(f"  output         :  {run_dir.relative_to(_ROOT)}/")
    print(f"{'='*60}")

    return record


# ── CLI ────────────────────────────────────────────────────────────────────────

_CONFIG_TABLE = "\n".join(f"  {k:<6} {v}" for k, v in CONFIG_LABELS.items())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run_compile.py",
        description=(
            "Compile and schedule a quantum circuit through the PBC pipeline.\n\n"
            "CIRCUIT is a circuit name (e.g. Adder8) or a path to a .qasm file.\n"
            "Named circuits are looked up in circuits/compiled/. QASM files are\n"
            "compiled on first use and cached there automatically.\n\n"
            "configurations (--config):\n"
            + _CONFIG_TABLE + "\n\n"
            "  LaM = our Locality-Aware Mapping algorithm (simulated annealing)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_compile.py --list
  python run_compile.py Adder8                             # config A (default)
  python run_compile.py Adder8 --config C                  # LaM + greedy
  python run_compile.py Adder8 --config D                  # LaM + CP-SAT
  python run_compile.py QFT32  --config C --topology ring

  # family weights are auto-detected from the circuit name for configs C/D:
  python run_compile.py gf8_mult  --config C               # uses gf   weights
  python run_compile.py Adder32   --config C               # uses adder weights
  python run_compile.py QFT64     --config D               # uses qft   weights
  python run_compile.py rand_50q_1kt_s42 --config C        # uses rand  weights

  # override auto-detection explicitly:
  python run_compile.py Adder8 --config C --family gf

  # compile from QASM then run:
  python run_compile.py circuits/Test/example.qasm --config C
""",
    )

    parser.add_argument("circuit", nargs="?", metavar="CIRCUIT",
                        help="Circuit name or path to a .qasm file")
    parser.add_argument("--list", action="store_true",
                        help="List all available pre-compiled circuits and exit")
    parser.add_argument("--config", choices=list(CONFIGS), default="A",
                        metavar="{" + ",".join(CONFIGS) + "}",
                        help="Preset mapper+scheduler configuration (default: A)")
    parser.add_argument("--family", choices=list(FAMILY_SCORE_KWARGS), default=None,
                        metavar="{" + ",".join(FAMILY_SCORE_KWARGS) + "}",
                        help=(
                            "Circuit family — activates tuned SA score weights for configs C/D. "
                            "Auto-detected from circuit name if omitted. "
                            "Families: adder (Adder*), qft (QFT*), gf (gf*_mult), rand (rand_*)"
                        ))
    parser.add_argument("--topology", choices=["grid", "ring"], default="grid",
                        help="Hardware graph topology (default: grid)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Label for the output directory (default: circuit name)")

    adv = parser.add_argument_group(
        "advanced", "Override the config's mapper or scheduler individually.")
    adv.add_argument("--mapper",
                     choices=["pure_random", "simulated_annealing",
                               "auto_round_robin_mapping", "auto_pack", "random_pack_mapping"],
                     default=None, help="Override mapper from config")
    adv.add_argument("--scheduler",
                     choices=["sequential_scheduler", "greedy_critical", "cp_sat"],
                     default=None, help="Override scheduler from config")
    adv.add_argument("--cp-sat-time", type=float, default=120.0, metavar="S",
                     help="CP-SAT per-layer time budget in seconds (default: 120)")
    adv.add_argument("--sa-steps", type=int, default=None, metavar="N",
                     help="SA mapper iterations (default: tuned value)")
    adv.add_argument("--n-data", type=int, default=_N_DATA, metavar="N",
                     help="Data qubit slots per block (default: 11)")
    adv.add_argument("--sparse-pct", type=float, default=0.0, metavar="F",
                     help="Fraction of qubit slots left empty (default: 0.0)")
    adv.add_argument("--coupler-capacity", type=int, default=_COUPLER_CAPACITY, metavar="N",
                     help="Max parallel ops per coupler link (default: 1)")

    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    # ── --list ─────────────────────────────────────────────────────────────────
    if args.list:
        circuits = _list_circuits()
        if not circuits:
            print("No pre-compiled circuits found in circuits/compiled/")
            print("Tip: run a .qasm file to compile and cache it.")
            return
        print(f"Pre-compiled circuits ({len(circuits)}):\n")
        for name in circuits:
            print(f"  {name}")
        print("\nRun with:  python run_compile.py <name> [--config {naive,A,B,C,D}]")
        return

    if not args.circuit:
        parser.print_help()
        return

    # ── Resolve PBC ────────────────────────────────────────────────────────────
    circuit_arg = args.circuit
    if circuit_arg.endswith(".qasm"):
        qasm_path = pathlib.Path(circuit_arg)
        if not qasm_path.exists():
            print(f"Error: file not found: {circuit_arg}")
            return
        name    = qasm_path.stem
        pbc_src = _find_pbc(name)
        if pbc_src is None:
            pbc_src = _compile_qasm(qasm_path)
        else:
            print(f"[compile]  cached PBC found  →  {pbc_src.relative_to(_ROOT)}")
    else:
        name    = circuit_arg
        pbc_src = _find_pbc(name)
        if pbc_src is None:
            available = sorted(_list_circuits())
            print(f"Error: no pre-compiled circuit named '{name}'.")
            if available:
                print(f"Available: {', '.join(available)}")
            print("Tip: pass a .qasm file path to compile a new circuit.")
            return

    # ── Resolve config + optional overrides ────────────────────────────────────
    mapper_name, sched_name = CONFIGS[args.config]
    if args.mapper    is not None:
        mapper_name = args.mapper
    if args.scheduler is not None:
        sched_name  = args.scheduler
    sa_steps = args.sa_steps if args.sa_steps is not None else TUNED_SA_STEPS

    # Family: explicit flag > auto-detect from circuit name
    family = args.family or _detect_family(name)

    config_label  = CONFIG_LABELS.get(args.config, f"{mapper_name} + {sched_name}")
    uses_sa       = mapper_name == "simulated_annealing"
    family_suffix = f"  (family weights: {family})" if family and uses_sa else (
                    "  (auto-detected; no SA weights for this family)" if family and not uses_sa else ""
    )
    print(f"circuit    : {name}")
    print(f"config     : {args.config}  ({config_label.split('[')[0].strip()})")
    print(f"topology   : {args.topology}")
    if family:
        print(f"family     : {family}{family_suffix}")
    print()

    run_circuit(
        name=name,
        pbc_path=pbc_src,
        config=args.config,
        mapper_name=mapper_name,
        sched_name=sched_name,
        topology=args.topology,
        seed=args.seed,
        cp_time=args.cp_sat_time,
        sa_steps=sa_steps,
        n_data=args.n_data,
        sparse_pct=args.sparse_pct,
        coupler_capacity=args.coupler_capacity,
        family=family,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
