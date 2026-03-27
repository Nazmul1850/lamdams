from __future__ import annotations
from collections import Counter
import hashlib
import pathlib
import importlib.util
import os
import random
import sys
from typing import Dict, List
from modqldpc.core.artifacts import ArtifactStore
from modqldpc.core.trace import Trace
from modqldpc.core.types import PipelineConfig
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.frontend.qasm_reader import QiskitCircuitHandler
from modqldpc.mapping.mapper import MappingConfig, MappingProblem, get_mapper
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.policy import (
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ChooseMagicBlockMinId,
    ShortestPathGatherRouting,
    NativeCostFn,
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.core.types import PauliRotation
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem
from modqldpc.pipeline.profiling import (
    CircuitProfile, LayerProfile,
    collect_circuit_profile, collect_layer_profile,
)
from modqldpc.pipeline.viz import (
    plot_circuit_character,
    plot_depth_profile,
    plot_block_utilization,
    plot_routing_distances,
    plot_frame_rewrites,
    plot_parallelism_profile,
)

DEFAULT_BASIS: tuple[str, ...] = (
    "h", "s", "sdg", "x", "y", "z",
    "cx", "cz", "swap",
    "t", "tdg",
    "measure",
)

def _stable_seed(block: int, ops: Dict[int, str]) -> int:
    items = tuple(sorted(ops.items()))
    payload = f"{block}|{items}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:16], 16)

def _round_to_nearest_odd(x: float) -> int:
    k = int(round(x))
    if k % 2 == 0:
        # choose the nearer odd
        if abs((k - 1) - x) <= abs((k + 1) - x):
            k = k - 1
        else:
            k = k + 1
    return max(1, k)

def normal_rotation_cost_fn(
    block: int,
    ops: Dict[int, str],
    hw,  # HardwareGraph
    *,
    base: float = 2.5,
    slope: float = 1.0,
    sigma_base: float = 0.75,
    sigma_slope: float = 0.15,
    min_cost: int = 1,
    max_cost: int = 25,
) -> int:
    w = len(ops)

    if w <= 0:
        return 1

    mu = base + slope * w
    sigma = sigma_base + sigma_slope * w

    rng = random.Random(_stable_seed(block, ops))
    x = rng.gauss(mu, sigma)

    x = max(min_cost, min(max_cost, x))
    c = _round_to_nearest_odd(x)
    c = max(min_cost, min(max_cost, c))
    return c


_ROTATION_SYNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "rotation_synch")


def _load_gross_synth(base_dir: str):
    spec = importlib.util.spec_from_file_location(
        "gross_clifford",
        os.path.join(base_dir, "gross_clifford.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gross_clifford"] = mod  # required so @dataclass can resolve its own module
    spec.loader.exec_module(mod)
    return mod


def make_gross_actual_cost_fn(plan, base_dir: str = _ROTATION_SYNCH_DIR, n_data: int = 11) -> NativeCostFn:
    """
    Returns a cost function backed by the actual gross code BFS closure data.

    Converts ops (global logical IDs -> Pauli char) to the n_data-length Pauli
    string expected by GrossCliffordSynth using plan.logical_to_local as the
    position index.  Logical IDs whose local_id >= n_data are the pivot qubit
    and are intentionally excluded.
    """
    mod = _load_gross_synth(base_dir)
    synth = mod.GrossCliffordSynth.load_precomputed(base_dir)

    def cost_fn(_block: int, ops: Dict[int, str], _hw) -> int:
        chars: List[str] = ["I"] * n_data
        for lid, axis in ops.items():
            local_id = plan.logical_to_local.get(lid)
            if local_id is not None and local_id < n_data:
                chars[local_id] = axis
        if all(c == "I" for c in chars):
            return 1
        mask = mod.pauli_to_mask("".join(chars))
        cost: int = synth.rotation_cost(mask)
        return cost
    return cost_fn


def run_one(path: str, cfg: PipelineConfig, meta: dict | None = None) -> str:
    meta = meta or {}
    if meta.get("compiled", False):
        run_one_compiled(pbc_path=path, cfg=cfg, meta=meta)
        return path

    qasm_path = path

    # ── Setup ─────────────────────────────────────────────────────────────────
    run_dir = ArtifactStore.make_run_dir(tag=f"{cfg.run_tag}__seed{cfg.seed}")
    store   = ArtifactStore(run_dir)
    trace   = Trace(f"{run_dir}/trace.ndjson")

    trace.event("run_start", qasm_path=qasm_path, seed=cfg.seed, run_tag=cfg.run_tag)
    store.put_json("config.json", cfg)
    store.copy_in(qasm_path, "input.qasm")
    trace.event("artifact_written", name="input.qasm")

    # ── Stage 1 : QASM → CircuitIR ───────────────────────────────────────────
    qc_handler     = QiskitCircuitHandler()
    qc, n_logicals = qc_handler.load_and_transpile(path=qasm_path, demo=False)

    cnt = Counter(inst.operation.name for inst in qc.data)
    print(f"[frontend]  qubits={qc.num_qubits}  clbits={qc.num_clbits}"
          f"  T={cnt['t']}  Tdg={cnt['tdg']}  T-like={cnt['t'] + cnt['tdg']}")

    conv    = GoSCConverter(verbose=False)
    program = conv.convert(qc=qc)
    _       = conv.greedy_layering()

    print(f"[frontend]  n_logicals={n_logicals}"
          f"  layers={len(conv.layers)}"
          f"  rotations={len(program.rotations)}")
    # conv.print_layers()

    # ── Stage 2 : Save PBC JSON ───────────────────────────────────────────────
    compact  = bool(meta.get("compact", True))
    payload  = conv.to_compact_payload() if compact else conv.to_cache_payload()
    pbc_rel  = "stage_frontend/PBC.json"
    store.put_json(pbc_rel, payload)
    pbc_path = str(pathlib.Path(run_dir) / pbc_rel)
    trace.event("stage_frontend_done",
                n_qubits=n_logicals,
                n_rots=len(program.rotations),
                compact=compact,
                pbc_path=pbc_path)
    print(f"[frontend]  PBC saved → {pbc_path}")

    # ── Stages 3–8 : delegate to compiled pipeline ───────────────────────────
    run_one_compiled(pbc_path=pbc_path, cfg=cfg, meta=meta)
    return run_dir


def run_one_compiled(
    pbc_path: str,
    cfg: PipelineConfig,
    meta: dict | None = None,
) -> None:
    """
    Run the full compiled PBC pipeline on a pre-converted circuit.

    Parameters
    ----------
    pbc_path : str
        Path to the compact PBC JSON produced by the frontend.
    cfg : PipelineConfig
        Pipeline-level config (run tag, global seed, …).
    meta : dict, optional
        Runtime knobs — all keys are optional.  Recognised keys and defaults::

          # Hardware
          topology          "grid" | "ring"           default: "grid"
          sparse_pct        float  [0, 1)              default: 0.0   (dense)
          n_data            int    qubit slots/block    default: 11
          coupler_capacity  int    capacity per link    default: 1

          # Mapping
          mapper            str    mapper name          default: "auto_round_robin_mapping"
          seed              int    global RNG seed      default: 42
          sa_steps          int    SA mapper iterations default: 10_000
          sa_t0             float  SA start temperature default: 1e5
          sa_tend           float  SA end temperature   default: 1.1

          # Scheduling
          scheduler         str    scheduler name       default: "greedy_critical"
          cp_sat_time_limit float  CP-SAT budget (s)    default: 120.0

          # Experiment flags
          run_experiments   bool   run Fig 8/9/10       default: True
          exp_sparse_pct    float  sparsity for Fig 9   default: 0.7
          exp_mapper        str    mapper  for Fig 9    default: "simulated_annealing"
          exp_scheduler     str    scheduler for Fig 9  default: "greedy_critical"
    """
    meta = meta or {}

    # ── Resolve meta with defaults ────────────────────────────────────────────
    topology          = str(meta.get("topology",          "grid"))
    sparse_pct        = float(meta.get("sparse_pct",      0.0))
    n_data            = int(meta.get("n_data",             11))
    coupler_cap       = int(meta.get("coupler_capacity",   1))
    mapper_name       = str(meta.get("mapper",            "auto_round_robin_mapping"))
    sched_name        = str(meta.get("scheduler",         "greedy_critical"))
    seed              = int(meta.get("seed",               cfg.seed))
    sa_steps          = int(meta.get("sa_steps",           10_000))
    sa_t0             = float(meta.get("sa_t0",            1e5))
    sa_tend           = float(meta.get("sa_tend",          1.1))
    cp_sat_time_limit = float(meta.get("cp_sat_time_limit", 120.0))
    run_experiments   = bool(meta.get("run_experiments",   False))
    exp_sparse_pct    = float(meta.get("exp_sparse_pct",   0.7))
    exp_mapper        = str(meta.get("exp_mapper",        "simulated_annealing"))
    exp_scheduler     = str(meta.get("exp_scheduler",     "greedy_critical"))

    # ── Stage 1 : Load compiled PBC ──────────────────────────────────────────
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)

    # Infer n_logicals from the Pauli string length of the first rotation.
    # The string may carry a sign prefix (+/-) which is stripped before measuring.
    first_rot  = next(iter(conv.program.rotations))
    pauli_str  = first_rot.axis.to_label().lstrip("+-")
    n_logicals = len(pauli_str)

    print(f"[frontend]  n_logicals={n_logicals}"
          f"  layers={len(conv.layers)}"
          f"  rotations={len(conv.program.rotations)}")

    # # ── Stage 2 : Build hardware ──────────────────────────────────────────────
    # # make_hardware handles both grid and ring topologies, choosing the minimum
    # # number of blocks required for the requested fill rate.
    hw, hw_spec = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=sparse_pct,
        n_data=n_data,
        coupler_capacity=coupler_cap,
    )
    print(f"[hardware]  {hw_spec.label()}"
          f"  topology={topology}"
          f"  sparse_pct={sparse_pct:.0%}"
          f"  n_data={n_data}")

    # # ── Stage 3 : Mapping ─────────────────────────────────────────────────────
    map_cfg     = MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend)
    map_problem = MappingProblem(n_logicals=n_logicals)
    mapper      = get_mapper(mapper_name)
    plan        = mapper.solve(map_problem, hw, map_cfg, {
        "rotations": conv.program.rotations,
        "verbose":   False,
        "debug":     False,
    })
    print(f"[mapping]   {mapper_name}  →  cost={plan.meta.get('cost', '?')}")

    # # ── Stage 4 : Lowering policies ───────────────────────────────────────────
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=make_gross_actual_cost_fn(plan)),
    )

    # # ── Stage 5 : Execution state ─────────────────────────────────────────────
    # # effective_rotations tracks frame-corrected rotations across layers.
    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame    = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )

    circuit_profile = collect_circuit_profile(
        n_logicals=n_logicals,
        layers=conv.layers,
        rotations=effective_rotations,
        hw=hw,
        plan=plan,
    )
    layer_profiles: List[LayerProfile] = []
    total_depth: int = 0

    # # ── Stage 6 : Main pipeline loop (lower → schedule → execute) ────────────
    print(f"[pipeline]  scheduler={sched_name}  layers={len(conv.layers)}")

    for layer_id, layer in enumerate(conv.layers):

        # 6a. Lowering: PBC layer → operation DAG
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )

        # 6b. Scheduling: assign time-steps to DAG nodes
        sched        = get_scheduler(sched_name)
        sched_problem = SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time":        0,
                "layer_idx":         layer_id,
                "tie_breaker":       "duration",
                "cp_sat_time_limit": cp_sat_time_limit,
                "debug_decode":      False,
                "safe_fill":         True,
                "cp_sat_log":        False,
            },
        )
        S = sched.solve(sched_problem)

        # 6c. Execution: apply Clifford frame, propagate corrected rotations
        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next  = [effective_rotations[i] for i in next_idxs]
        ex = executor.execute_layer(
            layer=layer_id,
            dag=res.dag,
            schedule=S,
            frame_in=frame,
            next_layer_rotations=rot_next,
        )
        for r in ex.next_rotations_effective:
            effective_rotations[r.idx] = r
        frame = ex.frame_after

        lp = collect_layer_profile(
            layer_id=layer_id,
            rotation_indices=layer,
            effective_rotations=effective_rotations,
            res=res,
            S=S,
            ex=ex,
            hw=hw,
            plan=plan,
        )
        layer_profiles.append(lp)
        total_depth += ex.depth

    # ── Circuit summary ───────────────────────────────────────────────────────
    n_layers = circuit_profile.n_layers
    sep = "─" * 68
    print(f"\n{'='*72}")
    print(f"  Circuit summary"
          f"  [{mapper_name}  +  {sched_name}  |  {hw_spec.label()}]")
    print(f"  {sep}")
    print(f"  Layers            {n_layers}")
    print(f"  Total rotations   {circuit_profile.n_rotations_total}")
    print(f"  Total depth       {total_depth}")
    print(f"  Avg depth/layer   {total_depth / max(n_layers, 1):.1f}")
    print(f"  Total rewrites    {sum(lp.n_rewrites        for lp in layer_profiles)}")
    print(f"  Support changes   {sum(lp.n_support_changes for lp in layer_profiles)}")
    print(f"  Angle flips       {sum(lp.n_angle_flips     for lp in layer_profiles)}")
    print(f"{'='*72}\n")

    # # ── Single-circuit figures (Fig 1–6) ──────────────────────────────────────
    fig_dir = pathlib.Path(pbc_path).parent.parent / "figures"
    print(f"Saving figures to: {fig_dir}/")

    for fig, name in [
        (plot_circuit_character(circuit_profile, layer_profiles,
            save_path=str(fig_dir / "fig_01_circuit_character.png")),
         "fig_01_circuit_character.png"),
        (plot_depth_profile(layer_profiles,
            save_path=str(fig_dir / "fig_02_depth_profile.png")),
         "fig_02_depth_profile.png"),
        (plot_block_utilization(circuit_profile, layer_profiles,
            save_path=str(fig_dir / "fig_03_block_utilization.png")),
         "fig_03_block_utilization.png"),
        (plot_routing_distances(layer_profiles,
            save_path=str(fig_dir / "fig_04_routing_distances.png")),
         "fig_04_routing_distances.png"),
        (plot_frame_rewrites(layer_profiles,
            save_path=str(fig_dir / "fig_05_frame_rewrites.png")),
         "fig_05_frame_rewrites.png"),
        (plot_parallelism_profile(layer_profiles,
            save_path=str(fig_dir / "fig_06_parallelism_profile.png")),
         "fig_06_parallelism_profile.png"),
    ]:
        fig.clf()
        print(f"  [done] {name}")

    if not run_experiments:
        return

    from modqldpc.pipeline.experiment import (
        run_algo_comparison,
        run_sparse_dense_comparison,
        run_topology_gallery,
    )

    # # ── Fig 8 : Algorithm comparison (3 mappers × 3 schedulers) ──────────────
    # print("\nRunning algorithm comparison (9 combos) ...")
    run_algo_comparison(
        pbc_path, str(fig_dir),
        n_logicals=n_logicals,
        seed=seed,
        sparse_pct=0.5,
        verbose=True,
    )

    # ── Fig 9 : Sparse vs dense hardware comparison ───────────────────────────
    print("\nRunning sparse vs dense hardware comparison ...")
    run_sparse_dense_comparison(
        pbc_path, str(fig_dir),
        n_logicals=n_logicals,
        mapper_name=exp_mapper,
        sched_name=exp_scheduler,
        sparse_pct=exp_sparse_pct,
        seed=seed,
        verbose=True,
    )

    # # ── Fig 10 : Hardware topology gallery (grid/ring × dense/sparse) ─────────
    print("\nBuilding hardware topology gallery ...")
    run_topology_gallery(
        str(fig_dir),
        n_logicals=n_logicals,
        sparse_pct=exp_sparse_pct,
        n_data=n_data,
        verbose=True,
    )
