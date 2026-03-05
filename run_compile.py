from modqldpc.core.types import PipelineConfig
from modqldpc.pipeline.run_one import run_one, run_one_compiled

if __name__ == "__main__":
    cfg = PipelineConfig(seed=42, run_tag="randon_10_100t")
    # run_dir = run_one("circuits/Test/rand_10q_100t.qasm", cfg)
    # print("Run stored at:", run_dir)
    run = run_one_compiled(pbc_path="runs/2026-03-01T22-28-52Z__randon_10_100t__seed42/stage_frontend/PBC.json", cfg=cfg)
