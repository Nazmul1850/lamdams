# modqldpc/lowering/keys.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class KeyNamer:
    """
    Deterministic naming. Seed/config should be reflected in run folder name,
    not in these keys (keys should be stable across reruns for diffing).
    """

    def rot_tag(self, layer: int, ridx: int) -> str:
        return f"L{layer:02d}_R{ridx:03d}"

    # measurement outcomes for π/8 gadget
    def bPZ(self, layer: int, ridx: int) -> str:
        return f"bPZ_{self.rot_tag(layer, ridx)}"

    def bXm(self, layer: int, ridx: int) -> str:
        return f"bXm_{self.rot_tag(layer, ridx)}"

    # nodes
    def nid(self, kind: str, layer: int, ridx: int, suffix: str = "") -> str:
        base = f"{kind}_{self.rot_tag(layer, ridx)}"
        return base if not suffix else f"{base}_{suffix}"

    # resources
    def magic_id(self, layer: int, ridx: int, block: int) -> str:
        return f"m_{self.rot_tag(layer, ridx)}_B{block}"

    def pivot_id(self, layer: int, ridx: int, block: int) -> str:
        return f"p_{self.rot_tag(layer, ridx)}_B{block}"