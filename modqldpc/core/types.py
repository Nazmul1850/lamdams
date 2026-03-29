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
    axis: str            # Pauli word string, e.g. "IXZI" (qubit-0 is rightmost char)
    angle: float         # rotation angle phi (pi/8 for T-gates, pi/2 for Clifford Pauli rotations)
    source: str          # e.g. "t q[3]" or "tdg q[2]"
    idx: int             # sequence index

@dataclass(frozen=True)
class PauliMeasurement:
    axis: str            # Pauli word string, e.g. "ZZZI"
    cbit: Optional[int]  # classical bit index if known
    qbit: int            # measured qubit (original)
    idx: int

@dataclass(frozen=True)
class PauliProgram:
    rotations: List[PauliRotation]
    final_meas: List[PauliMeasurement]
    final_clifford: Any    # reserved; set to None in lsqecc pipeline