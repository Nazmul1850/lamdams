from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass(frozen=True)
class PipelineConfig:
    seed: int=0
    run_tag: str="dev"


@dataclass(frozen=True)
class PauliAxis:
    sign: int          # +1/-1
    tensor: str        # "IXYZ..." (no leading +/-)

@dataclass(frozen=True)
class PauliRotation:
    rid: str
    axis: PauliAxis
    angle: float       # +/- pi/8
    source: str
    idx: int

@dataclass(frozen=True)
class PauliMeasurement:
    mid: str
    axis: PauliAxis
    cbit: int

@dataclass(frozen=True)
class PauliProgram:
    rotations: List[PauliRotation]
    final_meas: List[PauliMeasurement]
    final_clifford: Any    # store qiskit Clifford or your own encoded form