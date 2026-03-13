from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
import os
import random
import sys
from typing import Callable, Dict, List
from modqldpc.core.artifacts import ArtifactStore
from modqldpc.core.trace import Trace
from modqldpc.core.types import PipelineConfig
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.frontend.qasm_reader import QiskitCircuitHandler
from modqldpc.mapping.mapper import MappingConfig, MappingProblem, get_mapper
from modqldpc.mapping.model import GraphFactory, GridTopology, HardwareGraph, RingTopology
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.policy import (
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ChooseMagicBlockMinId,
    ShortestPathGatherRouting,
    NativeAllPaulisForNow,
    MagicPlacementPolicy,
    NativeCostFn,
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.core.types import PauliAxis, PauliRotation  # your dataclasses
from modqldpc.lowering.visualize import dag_to_dot
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem
from modqldpc.scheduling.validate import validate_schedule

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


def run_one(qasm_path: str, cfg: PipelineConfig) -> str:
    run_dir = ArtifactStore.make_run_dir(tag=f"{cfg.run_tag}__seed{cfg.seed}")
    store = ArtifactStore(run_dir)
    trace = Trace(f"{run_dir}/trace.ndjson")

    trace.event("run_start", qasm_path=qasm_path, seed=cfg.seed, run_tag=cfg.run_tag)
    store.put_json("config.json", cfg)

    store.copy_in(qasm_path, "input.qasm")
    trace.event("artifact_written", name="input.qasm")

    # Stage 1: QASM -> CircuitIR
    qc_handler = QiskitCircuitHandler()
    qc, num_logicals = qc_handler.load_and_transpile(path=qasm_path, demo=False)
    cnt = Counter(inst.operation.name for inst in qc.data)
    print("t:", cnt["t"], "tdg:", cnt["tdg"], "total T-like:", cnt["t"]+cnt["tdg"])
    print(cnt)
    # print(qc_handler.gate_histogram(qc))
    conv = GoSCConverter(verbose=False)
    program = conv.convert(qc=qc)
    layers = conv.greedy_layering()
    # conv.print_rotations()
    # conv.print_measurements()
    conv.print_layers()

    compact = True #will come from configs
    if compact == True:
        payload = conv.to_compact_payload()
        store.put_json("stage_frontend/PBC.json", payload)
    else:
        payload = conv.to_cache_payload()
        store.put_json("stage_frontend/PBC.json", payload)
    # conv.save_cache_json(abspath, compact=True)
    # store.put_json("stage_frontend/PBC.json", conv.to_cache_payload())
    trace.event("stage_frontend_done", n_qubits=num_logicals, n_rots=len(program.rotations))

    # hw = GraphFactory().build(topology=GridTopology(2,2), block_ids=[1,2,3,4], coupler_capacity=1)
    hw = GraphFactory().build(topology=RingTopology(), block_ids=[1,2,3,4,5], coupler_capacity=1)
    print(qc.num_qubits)
    print(qc.num_clbits)
    problem = MappingProblem(n_logicals=num_logicals) 
    cfg = MappingConfig(seed=123, sa_steps=10000, sa_t0=1e5, sa_tend=0.05)
    mapper = get_mapper("simulated_annealing")
    plan = mapper.solve(problem, hw, cfg, {
        "rotations": conv.program.rotations,
        "verbose": True, 
        "debug": True,
    })
    # print(plan.meta)

    base_policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=make_gross_actual_cost_fn(plan)),
    )

    total_depth: int = 0
    trace.event("lowering_start")
    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=conv.program.rotations,
            rotation_indices=layer,
            hw=hw,
            policies=base_policies,
        )
        # dot_str = dag_to_dot(res.dag)
        # print(dot_str)
        # sched = get_scheduler("sa_scheduler")
        # sched = get_scheduler("sequential_scheduler")
        sched = get_scheduler("cp_sat")
        problem = SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=0,
            policy_name="incident_coupler_blocks_local",
            meta={"start_time": 0, "tie_breaker": "duration", "sa_iterations": 0, "sa_initial_temp": 10.0, "sa_cooling_rate": 0.95, "sa_neighbor": "mixed"},
        )
        S = sched.solve(problem)

        next_idxs = conv.layers[layer_id+1] if (layer_id+1) in conv.layers else []
        rot_next = [conv.program.rotations[i] for i in next_idxs]
        executor = LayerExecutor(
            outcome_model=RandomOutcomeModel(seed=42),
            frame_policy=FrameUpdatePolicy(),
        )

        frame = FrameState()  # empty at start
        ex = executor.execute_layer(
            layer=layer_id,
            dag=res.dag,
            schedule=S,
            frame_in=frame,
            next_layer_rotations=rot_next,
        )

        # ex0.next_rotations_effective is your "actual next layer" after updates
        # print("changed:", len(ex.rewrite_log))
        # print(ex0.rewrite_log)
        # print(ex0.events)
        # print("depth:", ex0.depth)
        trace.event("layer_executed", layer_id=layer_id, depth=ex.depth, num_rewrites=len(ex.rewrite_log), sched='sequential_scheduler')
        total_depth += ex.depth

        # conv.print_layers()

        # trace.event("run_done")
    trace.event("run_done", total_depth=total_depth)
    return "run_dir"


def run_one_compiled(pbc_path: str, cfg: PipelineConfig):
    # run_dir = ArtifactStore.make_run_dir(tag=f"{cfg.run_tag}__seed{cfg.seed}")
    # store = ArtifactStore(run_dir)
    # trace = Trace(f"{run_dir}/trace.ndjson")

    # trace.event("run_start", pbc_path=pbc_path, seed=cfg.seed, run_tag=cfg.run_tag)
    # store.put_json("config.json", cfg)

    conv = GoSCConverter(verbose=False)
    payload = conv.load_cache_json(pbc_path)

    hw = GraphFactory().build(topology=GridTopology(3,3), block_ids=[i+1 for i in range(9)], coupler_capacity=1)
    # hw = GraphFactory().build(topology=RingTopology(), block_ids=[1,2], coupler_capacity=1)

    problem = MappingProblem(n_logicals=10)   # look at the trace file to get the actual number of logicals for this run
    cfg = MappingConfig(seed=123, sa_steps=10000, sa_t0=1e5, sa_tend=0.05)
    # mapper = get_mapper("auto_round_robin_mapping")
    # mapper = get_mapper("auto_pack")
    # mapper = get_mapper("pure_random")
    # print(type(mapper))
    # plan = mapper.solve(problem=problem, hw=hw, cfg=cfg)
    mapper = get_mapper("simulated_annealing")
    plan = mapper.solve(problem, hw, cfg, {
        "rotations": conv.program.rotations,
        "verbose": True, 
        "debug": True,
    })
    print(plan.meta)

    base_policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=make_gross_actual_cost_fn(plan)),
    )
    total_depth: int = 0

    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=conv.program.rotations,
            rotation_indices=layer,
            hw=hw,
            policies=base_policies,
        )
        # dot_str = dag_to_dot(res.dag)
        # print(dot_str)
        # sched = get_scheduler("sa_scheduler")
        sched = get_scheduler("cp_sat")
        # sched = get_scheduler("sequential_scheduler")
        problem = SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=0,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time": 0, 
                "tie_breaker": "duration", 
                "sa_iterations": 0, 
                "sa_initial_temp": 10.0, 
                "sa_cooling_rate": 0.95, 
                "sa_neighbor": "mixed", 
                "debug_decode": False
            },
        )
        S = sched.solve(problem)

        next_idxs = conv.layers[layer_id+1] if (layer_id+1) in conv.layers else []
        rot_next = [conv.program.rotations[i] for i in next_idxs]
        executor = LayerExecutor(
            outcome_model=RandomOutcomeModel(seed=42),
            frame_policy=FrameUpdatePolicy(),
        )

        frame = FrameState()  # empty at start
        ex = executor.execute_layer(
            layer=layer_id,
            dag=res.dag,
            schedule=S,
            frame_in=frame,
            next_layer_rotations=rot_next,
        )

        # ex0.next_rotations_effective is your "actual next layer" after updates
        # print("changed:", len(ex.rewrite_log))
        # print(ex0.rewrite_log)
        # print(ex0.events)
        # print("depth:", ex0.depth)
        total_depth += ex.depth
        break

        # conv.print_layers()

        # trace.event("run_done")
    print("Total depth:", total_depth)
