from modqldpc.core.types import PipelineConfig
from modqldpc.pipeline.run_one import run_one, run_one_compiled

if __name__ == "__main__":
    cfg = PipelineConfig(seed=42, run_tag="compare-dascot")
    # run_dir = run_one("circuits/Test/qft_20.qasm", cfg)
    # print("Run stored at:", run_dir)test-rotations
    run = run_one_compiled(pbc_path="runs/randon_10_100t_v2__seed42__2026-03-07T08-24-50Z/stage_frontend/PBC.json", cfg=cfg)
    # run = run_one_compiled(pbc_path="runs/test-rotations/PBC.json", cfg=cfg)
