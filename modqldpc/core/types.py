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
    axis: PauliAxis          # Pauli product P
    angle: float         # rotation angle phi (we use +/- pi/8)
    source: str          # e.g. "t q[3]" or "tdg q[2]"
    idx: int             # sequence index

@dataclass(frozen=True)
class PauliMeasurement:
    axis: PauliAxis          # Pauli product to measure
    cbit: Optional[int]  # classical bit index if known
    qbit: int            # measured qubit (original)
    idx: int

@dataclass(frozen=True)
class PauliProgram:
    rotations: List[PauliRotation]
    final_meas: List[PauliMeasurement]
    final_clifford: Any    # store qiskit Clifford or your own encoded form