from __future__ import annotations
from modqldpc.core.artifacts import ArtifactStore
from modqldpc.core.trace import Trace
from modqldpc.core.types import PipelineConfig

def run_one(qasm_path: str, cfg: PipelineConfig) -> str:
    run_dir = ArtifactStore.make_run_dir(tag=f"{cfg.run_tag}__seed{cfg.seed}")
    store = ArtifactStore(run_dir)
    trace = Trace(f"{run_dir}/trace.ndjson")

    trace.event("run_start", qasm_path=qasm_path, seed=cfg.seed, run_tag=cfg.run_tag)
    store.put_json("config.json", cfg)

    store.copy_in(qasm_path, "input.qasm")
    trace.event("artifact_written", name="input.qasm")

    # Stage 1: QASM -> CircuitIR
    # cir = read_openqasm2(qasm_path)
    # store.put_json("stage_frontend/circuit_ir.json", cir)
    # trace.event("stage_frontend_done", n_qubits=cir.n_qubits, n_ops=len(cir.ops))

    # trace.event("run_done")
    return run_dir
