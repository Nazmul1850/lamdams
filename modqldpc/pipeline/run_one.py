from __future__ import annotations
import os
from modqldpc.core.artifacts import ArtifactStore
from modqldpc.core.trace import Trace
from modqldpc.core.types import PipelineConfig
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.frontend.qasm_reader import QiskitCircuitHandler

DEFAULT_BASIS: tuple[str, ...] = (
    "h", "s", "sdg", "x", "y", "z",
    "cx", "cz", "swap",
    "t", "tdg",
    "measure",
)

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
    qc = qc_handler.load_and_transpile(path=qasm_path, demo=False)
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
    trace.event("stage_frontend_done", n_qubits=qc.num_qubits, n_rots=len(program.rotations))
    
    program_ret = conv.load_cache_json(abspath)
    for r in program_ret.rotations:
        print(r)

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

    # trace.event("run_done")
