"""
Scaling study: does SA mapping advantage grow with circuit size?

Generates random Clifford+T circuits across a range of qubit counts and
measures total circuit depth for three mappers (SA, RoundRobin, PureRandom).

Key insight:
  - Hardware block capacity = n_data (default 11 qubits/block)
  - n_blocks = ceil(n_qubits / n_data)
  - Below the first block boundary (n_qubits ≤ 11): all mappers are equivalent
  - Above it: SA can exploit multi-block placement → depth advantage grows

Expected result:
  SA depth reduction (%) increases with n_qubits once n_blocks ≥ 2.

Usage:
    python test_scaling.py
    python test_scaling.py --n_t 200 --seeds 42,43,44 --topology ring
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.frontend.qasm_reader import QiskitCircuitHandler
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId, HeuristicRepeatNativePolicy,
    LoweringPolicies, ShortestPathGatherRouting,
)
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import (
    MappingConfig, MappingPlan, MappingProblem, get_mapper,
)
from modqldpc.mapping.algos.sa_mapping import _score, ScoreBreakdown
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem

# ── Circuit generator ─────────────────────────────────────────────────────────
_GEN_PATH = os.path.join(_ROOT, "circuits", "random", "generate.py")
_gen_spec = importlib.util.spec_from_file_location("_circuit_gen", _GEN_PATH)
_gen_mod  = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen_mod)
_generate_qasm = _gen_mod.generate  # generate(n_qubits, n_t, seed, out_path)

# ── Synthesis DB singleton ────────────────────────────────────────────────────
_SYNCH_DIR = os.path.join(_ROOT, "modqldpc", "rotation_synch")
_GROSS_CACHE: Optional[Tuple[Any, Any]] = None


def _gross_synth() -> Tuple[Any, Any]:
    global _GROSS_CACHE
    if _GROSS_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "gross_clifford", os.path.join(_SYNCH_DIR, "gross_clifford.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gross_clifford"] = mod
        spec.loader.exec_module(mod)
        synth = mod.GrossCliffordSynth.load_precomputed(_SYNCH_DIR)
        _GROSS_CACHE = (mod, synth)
        print("[synth]  rotation synthesis database loaded")
    return _GROSS_CACHE


def make_cost_fn(plan: MappingPlan, n_data: int = 11):
    mod, synth = _gross_synth()
    def cost_fn(_b, ops: Dict[int, str], _h) -> int:
        del _b, _h
        chars = ["I"] * n_data
        for lid, axis in ops.items():
            local_id = plan.logical_to_local.get(lid)
            if local_id is not None and local_id < n_data:
                chars[local_id] = axis
        if all(c == "I" for c in chars):
            return 1
        return int(synth.rotation_cost(mod.pauli_to_mask("".join(chars))))
    return cost_fn


# ── Sweep configuration ───────────────────────────────────────────────────────
# Qubit counts: chosen to span 1 → 4 blocks (n_data=11)
#   1 block  : q ≤ 11   → baseline, SA ≈ RR
#   2 blocks : 12–22 q
#   3 blocks : 23–33 q
#   4 blocks : 34–44 q
DEFAULT_QUBIT_COUNTS = [8, 11, 15, 22, 33, 44]
DEFAULT_N_T          = 100     # T-gates per circuit (keep small for runtime)
DEFAULT_SEEDS        = [42, 43, 44]
DEFAULT_N_DATA       = 11

# Best SA weights from the hierarchy gap / preset analysis (peak>span>range ordering)
SA_SCORE_KWARGS = dict(
    W_PEAK  = 1e5,
    W_SPAN  = 1e4,
    W_RANGE = 1e3,
    W_SPLIT = 1e2,
    W_MST   = 10.0,
    W_STD   = 0.0,
    W_DISCONNECTED = 0.0,
)
SA_STEPS = 25_000
SA_T0    = 1e5
SA_TEND  = 10.0

OUT_FIG = "test_scaling.png"

# ── Data containers ───────────────────────────────────────────────────────────
@dataclass
class ScalingPoint:
    n_qubits:    int
    n_blocks:    int
    seed:        int
    mapper:      str
    total_depth: int
    elapsed_s:   float
    n_layers:    int


# ── Circuit → rotations/layers ────────────────────────────────────────────────
def load_qasm_to_pbc(qasm_path: str):
    """QASM → (n_logicals, rotations list, layers list-of-lists)."""
    qc_handler = QiskitCircuitHandler()
    qc, n_logicals = qc_handler.load_and_transpile(path=qasm_path, demo=False)
    conv = GoSCConverter(verbose=False)
    conv.convert(qc=qc)
    conv.greedy_layering()
    rotations = list(conv.program.rotations)
    layers    = conv.layers   # List[List[int]]
    return n_logicals, rotations, layers


# ── Depth computation ─────────────────────────────────────────────────────────
def _compute_depth(
    rotations, layers, hw, plan: MappingPlan,
    n_data: int, seed: int,
) -> int:
    policies = LoweringPolicies(
        namer   = KeyNamer(),
        magic   = ChooseMagicBlockMinId(),
        routing = ShortestPathGatherRouting(),
        native  = HeuristicRepeatNativePolicy(cost_fn=make_cost_fn(plan, n_data)),
    )
    effective = {r.idx: r for r in rotations}
    total = 0
    for layer_id, layer_rot_ids in enumerate(layers):
        res = lower_one_layer(
            layer_idx        = layer_id,
            rotations        = effective,
            rotation_indices = layer_rot_ids,
            hw               = hw,
            policies         = policies,
        )
        S = get_scheduler("greedy_critical").solve(SchedulingProblem(
            dag         = res.dag,
            hw          = hw,
            seed        = seed,
            policy_name = "incident_coupler_blocks_local",
            meta        = {
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": 30.0,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        ))
        entries = S.meta.get("entries", {})
        total  += max((se["end"] for se in entries.values()), default=0)
    return total


# ── One run: map + depth ──────────────────────────────────────────────────────
def run_one_config(
    n_logicals: int,
    rotations:  list,
    layers:     list,
    *,
    mapper:           str,
    topology:         str,
    sparse_pct:       float,
    n_data:           int,
    coupler_capacity: int,
    seed:             int,
) -> Tuple[MappingPlan, int]:
    hw, _ = make_hardware(
        n_logicals,
        topology         = topology,
        sparse_pct       = sparse_pct,
        n_data           = n_data,
        coupler_capacity = coupler_capacity,
    )
    map_cfg = MappingConfig(seed=seed, sa_steps=SA_STEPS, sa_t0=SA_T0, sa_tend=SA_TEND)

    if mapper == "simulated_annealing_custom":
        # SA with peak>span>range weights
        from modqldpc.mapping.algos.sa_mapping import _random_move, _undo_move
        get_mapper("auto_round_robin_mapping").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed)
        )
        rng = random.Random(seed)
        cur = _score(rotations, hw, **SA_SCORE_KWARGS)
        best = cur
        best_map = {q: (hw.logical_to_block[q], hw.logical_to_local[q])
                    for q in hw.logical_to_block}
        for it in range(1, SA_STEPS + 1):
            T = SA_T0 * ((SA_TEND / SA_T0) ** ((it - 1) / max(1, SA_STEPS - 1)))
            move = _random_move(hw, rng)
            if move[0] == "noop":
                continue
            nxt = _score(rotations, hw, **SA_SCORE_KWARGS)
            delta = nxt.total - cur.total
            if delta <= 0 or rng.random() < math.exp(-delta / max(1e-12, T)):
                cur = nxt
                if cur.total < best.total:
                    best = cur
                    best_map = {q: (hw.logical_to_block[q], hw.logical_to_local[q])
                                for q in hw.logical_to_block}
            else:
                _undo_move(hw, move)
        hw.logical_to_block.clear()
        hw.logical_to_local.clear()
        for q, (b, l) in best_map.items():
            hw.logical_to_block[q] = b
            hw.logical_to_local[q] = l
        plan = MappingPlan(
            logical_to_block=dict(hw.logical_to_block),
            logical_to_local=dict(hw.logical_to_local),
        )
    else:
        plan = get_mapper(mapper).solve(
            MappingProblem(n_logicals=n_logicals), hw, map_cfg
        )

    depth = _compute_depth(rotations, layers, hw, plan, n_data, seed)
    return plan, depth


# ── Full scaling sweep ────────────────────────────────────────────────────────
def run_scaling_sweep(
    qubit_counts:     List[int],
    n_t:              int,
    seeds:            List[int],
    topology:         str,
    sparse_pct:       float,
    n_data:           int,
    coupler_capacity: int,
    mappers:          List[str],
) -> List[ScalingPoint]:
    results: List[ScalingPoint] = []
    total_runs = len(qubit_counts) * len(seeds) * len(mappers)
    run_idx = 0

    for n_q in qubit_counts:
        n_blocks = math.ceil(n_q / n_data)
        print(f"\n{'='*70}")
        print(f"  n_qubits={n_q}  n_blocks={n_blocks}  n_t={n_t}")
        print(f"{'='*70}")

        for seed in seeds:
            # Generate circuit to a temp file, then load as PBC
            with tempfile.NamedTemporaryFile(suffix=".qasm", delete=False) as tf:
                qasm_path = tf.name
            try:
                _generate_qasm(n_q, n_t, seed, qasm_path)
                n_logicals, rotations, layers = load_qasm_to_pbc(qasm_path)
            finally:
                os.unlink(qasm_path)

            print(f"  [circuit]  seed={seed}  n_logicals={n_logicals}"
                  f"  layers={len(layers)}  rotations={len(rotations)}")

            for mapper in mappers:
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}]  mapper={mapper}  seed={seed}  ...",
                      end=" ", flush=True)
                t0 = time.perf_counter()
                try:
                    _, depth = run_one_config(
                        n_logicals, rotations, layers,
                        mapper           = mapper,
                        topology         = topology,
                        sparse_pct       = sparse_pct,
                        n_data           = n_data,
                        coupler_capacity = coupler_capacity,
                        seed             = seed,
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"depth={depth}  ({elapsed:.1f}s)")
                    results.append(ScalingPoint(
                        n_qubits    = n_q,
                        n_blocks    = n_blocks,
                        seed        = seed,
                        mapper      = mapper,
                        total_depth = depth,
                        elapsed_s   = elapsed,
                        n_layers    = len(layers),
                    ))
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    print(f"ERROR: {e}  ({elapsed:.1f}s)")

    return results


# ── Aggregate helper ──────────────────────────────────────────────────────────
def _agg(results: List[ScalingPoint], n_q: int, mapper: str):
    pts = [r.total_depth for r in results if r.n_qubits == n_q and r.mapper == mapper]
    if not pts:
        return None, None, None
    return np.mean(pts), np.std(pts), len(pts)


# ── Figure ────────────────────────────────────────────────────────────────────
_C = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

MAPPER_LABELS = {
    "simulated_annealing_custom": "SA (peak>span>range)",
    "auto_round_robin_mapping":   "Round-Robin",
    "pure_random":                "Pure Random",
}

def build_scaling_figure(
    results:      List[ScalingPoint],
    qubit_counts: List[int],
    mappers:      List[str],
    n_data:       int,
    n_t:          int,
    out_path:     str,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Scaling Study: SA Mapping Advantage vs Circuit Size\n"
        f"Random Clifford+T circuits, {n_t} T-gates, "
        f"n_data={n_data} qubits/block, greedy scheduler",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.38)

    # ── [0,:]: depth vs n_qubits — main comparison ────────────────────────────
    ax_main = fig.add_subplot(gs[0, :])
    sa_mapper = mappers[0]  # first mapper assumed to be SA
    for i, mapper in enumerate(mappers):
        means, stds, xs_valid = [], [], []
        for n_q in qubit_counts:
            mu, sigma, cnt = _agg(results, n_q, mapper)
            if mu is not None:
                xs_valid.append(n_q)
                means.append(mu)
                stds.append(sigma if sigma else 0.0)
        if not xs_valid:
            continue
        label = MAPPER_LABELS.get(mapper, mapper)
        ax_main.plot(xs_valid, means, "o-", color=_C[i], linewidth=2,
                     markersize=6, label=label, zorder=3)
        ax_main.fill_between(xs_valid,
                              [m - s for m, s in zip(means, stds)],
                              [m + s for m, s in zip(means, stds)],
                              color=_C[i], alpha=0.15)
    # Block boundary lines
    boundary = n_data
    while boundary <= max(qubit_counts):
        ax_main.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_main.text(boundary + 0.3,
                     ax_main.get_ylim()[1] if ax_main.get_ylim()[1] > 0 else 1,
                     f"{boundary//n_data+1}B", fontsize=6, color="gray", va="top")
        boundary += n_data
    ax_main.set_xlabel("n_qubits", fontsize=8)
    ax_main.set_ylabel("total depth  (mean ± 1σ)", fontsize=8)
    ax_main.set_title("Total circuit depth vs qubit count  |  dashed lines = block boundaries",
                      fontsize=10)
    ax_main.legend(fontsize=8)
    ax_main.tick_params(labelsize=8)
    ax_main.set_xticks(qubit_counts)

    # ── [1,0]: SA depth reduction (%) vs n_qubits ────────────────────────────
    ax_adv = fig.add_subplot(gs[1, 0])
    rr_mapper = next((m for m in mappers if "round_robin" in m), None)
    if rr_mapper:
        xs_adv, ys_adv, ys_err = [], [], []
        for n_q in qubit_counts:
            mu_sa,  s_sa,  _ = _agg(results, n_q, sa_mapper)
            mu_rr,  s_rr,  _ = _agg(results, n_q, rr_mapper)
            if mu_sa is not None and mu_rr is not None and mu_rr > 0:
                reduction = (mu_rr - mu_sa) / mu_rr * 100.0
                xs_adv.append(n_q)
                ys_adv.append(reduction)
                ys_err.append(math.sqrt(s_sa**2 + s_rr**2) / mu_rr * 100.0
                              if (s_sa and s_rr) else 0.0)
        ax_adv.bar(xs_adv, ys_adv, yerr=ys_err, color=_C[0], alpha=0.8,
                   capsize=4, width=2.0)
        ax_adv.axhline(0, color="black", linewidth=0.8)
        # Block boundaries
        bnd = n_data
        while bnd <= max(qubit_counts):
            ax_adv.axvline(bnd, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            bnd += n_data
        ax_adv.set_xlabel("n_qubits", fontsize=7)
        ax_adv.set_ylabel("depth reduction vs RR (%)", fontsize=7)
        ax_adv.set_title("SA depth reduction over Round-Robin\n(positive = SA wins)",
                          fontsize=9)
        ax_adv.tick_params(labelsize=7)
        ax_adv.set_xticks(qubit_counts)

    # ── [1,1]: n_blocks vs n_qubits + avg depth/layer ────────────────────────
    ax_blk = fig.add_subplot(gs[1, 1])
    n_blocks_list = [math.ceil(q / n_data) for q in qubit_counts]
    ax_blk.bar(qubit_counts, n_blocks_list, color="#8172B2", alpha=0.75, width=2.5)
    for x, nb in zip(qubit_counts, n_blocks_list):
        ax_blk.text(x, nb + 0.05, str(nb), ha="center", va="bottom", fontsize=7)
    ax_blk.set_xlabel("n_qubits", fontsize=7)
    ax_blk.set_ylabel("# hardware blocks", fontsize=7)
    ax_blk.set_title(f"Hardware blocks per qubit count\n(n_data={n_data} qubits/block)",
                     fontsize=9)
    ax_blk.tick_params(labelsize=7)
    ax_blk.set_xticks(qubit_counts)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[figure]  saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scaling study: SA vs baselines across qubit counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--qubits",    default=",".join(map(str, DEFAULT_QUBIT_COUNTS)),
                        help="Comma-separated qubit counts")
    parser.add_argument("--n_t",       type=int,   default=DEFAULT_N_T,
                        help="T-gates per random circuit")
    parser.add_argument("--seeds",     default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated RNG seeds (one circuit per seed)")
    parser.add_argument("--n_data",    type=int,   default=DEFAULT_N_DATA,
                        help="Data qubit slots per block")
    parser.add_argument("--topology",  default="grid", choices=["grid", "ring"])
    parser.add_argument("--sparse_pct",type=float, default=0.0)
    parser.add_argument("--out",       default=OUT_FIG)
    args = parser.parse_args()

    qubit_counts = [int(x) for x in args.qubits.split(",")]
    seeds        = [int(x) for x in args.seeds.split(",")]
    mappers      = [
        "simulated_annealing_custom",
        "auto_round_robin_mapping",
        "pure_random",
    ]

    print(f"Qubit counts : {qubit_counts}")
    print(f"T-gates/circ : {args.n_t}")
    print(f"Seeds        : {seeds}")
    print(f"Mappers      : {mappers}")
    print(f"n_data       : {args.n_data}  (block capacity)")
    print(f"Topology     : {args.topology}  sparse_pct={args.sparse_pct:.0%}")
    print(f"Block boundaries at qubits: "
          f"{[args.n_data * k for k in range(1, max(qubit_counts)//args.n_data + 2)]}")
    print()

    results = run_scaling_sweep(
        qubit_counts     = qubit_counts,
        n_t              = args.n_t,
        seeds            = seeds,
        topology         = args.topology,
        sparse_pct       = args.sparse_pct,
        n_data           = args.n_data,
        coupler_capacity = 1,
        mappers          = mappers,
    )

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'─'*90}")
    print(f"  {'n_q':>4}  {'blocks':>6}  {'mapper':<35}  {'mean_depth':>10}  {'std':>7}  {'n':>3}")
    print(f"{'─'*90}")
    for n_q in qubit_counts:
        n_blocks = math.ceil(n_q / args.n_data)
        for mapper in mappers:
            mu, sigma, cnt = _agg(results, n_q, mapper)
            if mu is not None:
                label = MAPPER_LABELS.get(mapper, mapper)
                print(f"  {n_q:>4}  {n_blocks:>6}  {label:<35}  "
                      f"{mu:>10.1f}  {sigma:>7.1f}  {cnt:>3}")
        print()
    print(f"{'='*90}")

    build_scaling_figure(
        results, qubit_counts, mappers,
        n_data=args.n_data, n_t=args.n_t, out_path=args.out,
    )


if __name__ == "__main__":
    main()
