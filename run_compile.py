from modqldpc.core.types import PipelineConfig
from modqldpc.pipeline.run_one import run_one

if __name__ == "__main__":
    cfg = PipelineConfig(seed=42, run_tag="example")
    run_dir = run_one("circuits/Test/example.qasm", cfg)
    print("Run stored at:", run_dir)
