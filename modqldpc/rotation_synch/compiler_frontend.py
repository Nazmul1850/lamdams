from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from gross_clifford import GrossCliffordSynth, N_DATA, pauli_to_mask


PIVOT_SEQUENCE = {
    "X": ("Z", "Y"),  # Z -> X -> Y
    "Y": ("X", "Z"),  # X -> Y -> Z
    "Z": ("Y", "X"),  # Y -> Z -> X
}


def _base_measurement_plan(native: Dict[str, Any]) -> Dict[str, Any]:
    src = native["source_measurements"][0]
    pivot = src["pivot_pauli"]
    init_basis, final_basis = PIVOT_SEQUENCE[pivot]
    return {
        "native_generator_index": native["index"],
        "family": src["family"],
        "automorphism_shift": src["shift"],
        "pivot_support": pivot,
        "data_pauli": native["pauli"],
        "joint_measurement": {
            "pivot_basis": pivot,
            "data_pauli": native["pauli"],
            "full_pauli": f"{pivot}⊗{native['pauli']}",
        },
        "init_measurement": {
            "target": "pivot",
            "basis": init_basis,
        },
        "final_measurement": {
            "target": "pivot",
            "basis": final_basis,
        },
        "paper_equation": 65,
    }


def _signed_measurement_plan(native: Dict[str, Any], sign: int) -> Dict[str, Any]:
    plan = _base_measurement_plan(native)
    plan["rotation_sign"] = int(sign)
    if sign < 0:
        plan["init_measurement"], plan["final_measurement"] = (
            plan["final_measurement"],
            plan["init_measurement"],
        )
    return plan


class GrossCompilerFrontend:
    def __init__(self, synth: GrossCliffordSynth):
        self.synth = synth

    @classmethod
    def load_precomputed(cls, base_dir: str) -> "GrossCompilerFrontend":
        return cls(GrossCliffordSynth.load_precomputed(base_dir))

    def synthesize_rotation(self, pauli: str) -> Dict[str, Any]:
        if len(pauli) != N_DATA:
            raise ValueError(f"expected {N_DATA}-qubit Pauli string")
        seq = self.synth.decomposition_from_pauli(pauli)
        steps: List[Dict[str, Any]] = []
        for order, item in enumerate(seq):
            native = self.synth.native_rotations[item["generator_index"]]
            steps.append(
                {
                    "order": order,
                    "generator_index": item["generator_index"],
                    "rotation_pauli": native["pauli"],
                    "rotation_sign": int(item["sign"]),
                    "measurement_plan": _signed_measurement_plan(native, int(item["sign"])),
                }
            )
        mask = pauli_to_mask(pauli)
        return {
            "target": {
                "pauli": pauli,
                "mask": int(mask),
            },
            "distance": int(self.synth.dist[mask]),
            "rotation_circuit_cost": self.synth.rotation_cost(mask),
            "native_sequence": seq,
            "compiler_steps": steps,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pauli", help=f"target {N_DATA}-qubit Pauli string, e.g. XIXIIIIIIII")
    ap.add_argument("--base-dir", default=os.path.dirname(__file__))
    args = ap.parse_args()

    frontend = GrossCompilerFrontend.load_precomputed(args.base_dir)
    out = frontend.synthesize_rotation(args.pauli)
    print(json.dumps(out, indent=2))
