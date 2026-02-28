from __future__ import annotations
import os
from typing import Dict
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
)
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.core.types import PauliAxis, PauliRotation  # your dataclasses
from modqldpc.lowering.visualize import dag_to_dot

DEFAULT_BASIS: tuple[str, ...] = (
    "h", "s", "sdg", "x", "y", "z",
    "cx", "cz", "swap",
    "t", "tdg",
    "measure",
)

def example_cost_fn(block: int, ops: Dict[int, str], hw: HardwareGraph) -> int:
    # toy heuristic: larger support costs more
    w = len(ops)
    return 2
    if w <= 1:
        return 1
    if w <= 3:
        return 2
    return 3

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
    qc_handler.assert_in_basis(qc=qc, basis_gates=DEFAULT_BASIS)
    # print(qc_handler.gate_histogram(qc))
    conv = GoSCConverter(verbose=False)
    program = conv.convert(qc=qc)
    layers = conv.greedy_layering()
    # conv.print_rotations()
    # conv.print_measurements()
    conv.print_layers()
    # cir = read_openqasm2(qasm_path)
    abspath = os.path.join(run_dir, "stage_frontend/PBC.json")
    conv.save_cache_json(abspath)
    # store.put_json("stage_frontend/PBC.json", conv.to_cache_json_string())
    trace.event("stage_frontend_done", n_qubits=num_logicals, n_rots=len(program.rotations))

    hw = GraphFactory().build(topology=GridTopology(2,2), block_ids=[1,2,3,4], coupler_capacity=1)
    print(qc.num_qubits)
    print(qc.num_clbits)
    problem = MappingProblem(n_logicals=num_logicals) 
    cfg = MappingConfig(seed=123, pack_fraction=0.6, shuffle_blocks=False)
    mapper = get_mapper("auto_round_robin_mapping")
    print(type(mapper))
    plan = mapper.solve(problem=problem, hw=hw, cfg=cfg)
    print(plan.meta)

    base_policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=example_cost_fn),
    )

    res1 = lower_one_layer(
        layer_idx=0,
        rotations=program.rotations,
        rotation_indices=layers[0],
        hw=hw,
        policies=base_policies,
    )
    print("Policy1 magic=minid; nodes:", len(res1.dag.nodes))
    print("Topo:", res1.dag.topological_order())

    dot_str = dag_to_dot(res1.dag)
    print(dot_str)
    # trace.event("run_done")
    return run_dir


def run_one_compiled(pbc_path: str, cfg: PipelineConfig):
    # run_dir = ArtifactStore.make_run_dir(tag=f"{cfg.run_tag}__seed{cfg.seed}")
    # store = ArtifactStore(run_dir)
    # trace = Trace(f"{run_dir}/trace.ndjson")

    # trace.event("run_start", pbc_path=pbc_path, seed=cfg.seed, run_tag=cfg.run_tag)
    # store.put_json("config.json", cfg)

    conv = GoSCConverter(verbose=False)
    
    program_ret = conv.load_cache_json(pbc_path)
    for r in program_ret["final_measurements"]:
        print(r)

    hw = GraphFactory().build(topology=GridTopology(1,2), block_ids=[1,2], coupler_capacity=1)

    problem = MappingProblem(n_logicals=20)   # logical ids 0..19
    cfg = MappingConfig(seed=123, pack_fraction=0.6, shuffle_blocks=False)
    mapper = get_mapper("auto_round_robin_mapping")
    print(type(mapper))
    plan = mapper.solve(problem=problem, hw=hw, cfg=cfg)
    print(plan.meta)
    for i in range(2):
        print(plan.loc(i))

    # trace.event("run_done")
