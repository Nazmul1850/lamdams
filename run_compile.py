"""
Entry point for the compiled PBC pipeline.

Edit the `META` dict below to control hardware topology, mapper, scheduler,
and experiment settings — no source-code changes needed between runs.

Usage:
    python run_compile.py
"""
import argparse

from modqldpc.core.types import PipelineConfig
from modqldpc.pipeline.run_one import run_one

# ── Default meta knobs ────────────────────────────────────────────────────────
# All keys are optional; unrecognised keys are silently ignored.
META: dict = {
    # ── Hardware ──────────────────────────────────────────────────────────────
    "topology":          "grid",   # "grid" | "ring"
    "sparse_pct":        0.0,      # 0.0 = dense, 0.5 = 50 % slots empty, etc.
    "n_data":            11,       # qubit data slots per block (Gross code = 11)
    "coupler_capacity":  1,        # capacity of each coupler link

    # ── Mapping ───────────────────────────────────────────────────────────────
    "mapper":            "auto_round_robin_mapping",
    #   options: "auto_round_robin_mapping" | "pure_random" |
    #            "simulated_annealing"      | "auto_pack"   | "random_pack_mapping"
    "sa_steps":          10_000,   # SA mapper iterations (only used when mapper=SA)
    "sa_t0":             1e5,      # SA start temperature
    "sa_tend":           1.1,      # SA end temperature

    # ── Scheduling ────────────────────────────────────────────────────────────
    "scheduler":         "greedy_critical",
    #   options: "sequential_scheduler" | "greedy_critical" | "cp_sat"
    "cp_sat_time_limit": 120.0,    # CP-SAT per-layer budget in seconds

    # ── Experiment flags (Fig 8 / 9 / 10) ────────────────────────────────────
    "compiled":          True,      # set False to run from QASM instead
    "run_experiments":   True,     # set False to skip Fig 8/9/10 (faster)
    "exp_sparse_pct":    0.7,      # sparsity used in Fig 9 sparse-vs-dense run
    "exp_mapper":        "simulated_annealing",   # mapper used in Fig 9
    "exp_scheduler":     "greedy_critical",       # scheduler used in Fig 9
}

# ── PBC paths ─────────────────────────────────────────────────────────────────
PBC_PATHS: dict[str, str] = {
    "randon_10_100t": "runs/randon_10_100t_v2__seed42__2026-03-07T08-24-50Z/stage_frontend/PBC.json",
    "rand_50q_1kt":   "runs/rand_50q_1kt__seed42__2026-03-12T06-06-52Z/stage_frontend/PBC.json",
    "test_rotations": "runs/test-rotations/PBC.json",
}

ACTIVE_RUN = "randon_10_100t"   # ← change this to switch circuits


def _parse_args() -> dict:
    """Allow any META key to be overridden from the command line.

    Example:
        python run_compile.py --topology ring --sparse_pct 0.5 --scheduler cp_sat
    """
    parser = argparse.ArgumentParser(
        description="Run the compiled PBC pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pbc",              default=None,
                        help="Override PBC path (overrides ACTIVE_RUN)")
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--tag",              type=str,   default="tag")
    parser.add_argument("--topology",         default=None, choices=["grid", "ring"])
    parser.add_argument("--sparse_pct",       type=float, default=None)
    parser.add_argument("--n_data",           type=int,   default=None)
    parser.add_argument("--coupler_capacity", type=int,   default=None)
    parser.add_argument("--mapper",           default=None)
    parser.add_argument("--scheduler",        default=None)
    parser.add_argument("--cp_sat_time_limit",type=float, default=None)
    parser.add_argument("--sa_steps",         type=int,   default=None)
    parser.add_argument("--sa_t0",            type=float, default=None)
    parser.add_argument("--sa_tend",          type=float, default=None)
    parser.add_argument("--compiled",         type=lambda x: x.lower() != "false",
                        default=None, metavar="true|false")
    parser.add_argument("--run_exp",  type=lambda x: x.lower() != "false",
                        default=False, metavar="true|false")
    parser.add_argument("--exp_sparse_pct",   type=float, default=None)
    parser.add_argument("--exp_mapper",       default=None)
    parser.add_argument("--exp_scheduler",    default=None)

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "pbc"}
    return args.pbc, overrides


if __name__ == "__main__":
    pbc_override, cli_overrides = _parse_args()

    # Merge: META defaults ← CLI overrides
    meta = {**META, **cli_overrides}

    # Resolve PBC path
    pbc_path = pbc_override or PBC_PATHS[ACTIVE_RUN]
    tag = meta.get("tag", ACTIVE_RUN)

    cfg = PipelineConfig(seed=meta.get("seed", 42), run_tag=tag)

    print(f"PBC path   : {pbc_path}")
    print(f"Topology   : {meta['topology']}  sparse_pct={meta['sparse_pct']:.0%}")
    print(f"Mapper     : {meta['mapper']}")
    print(f"Scheduler  : {meta['scheduler']}")
    print(f"Experiments: {meta['run_exp']}")
    print()

    run_one(pbc_path, cfg=cfg, meta=meta)
