from modqldpc.core.types import PipelineConfig
from modqldpc.pipeline.run_one import run_one, run_one_compiled

if __name__ == "__main__":
    cfg = PipelineConfig(seed=42, run_tag="example")
    # run_dir = run_one("circuits/Test/example.qasm", cfg, compiled=True, compiled_dir="2026-02-19T22-06-33Z__example__seed42")
    # print("Run stored at:", run_dir)
    run = run_one_compiled(pbc_path="runs/2026-02-19T22-06-33Z__example__seed42/stage_frontend/PBC.json", cfg=cfg)
