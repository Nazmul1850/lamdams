
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np


NX = 12
NY = 6
N_PHYS = 144
N_DATA = 11
X_MASK = (1 << N_DATA) - 1


def gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    A = M.copy().astype(np.uint8)
    rows, cols = A.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i, :] ^= A[r, :]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return A, pivots


def gf2_rank(M: np.ndarray) -> int:
    return len(gf2_rref(M)[1])


def gf2_nullspace(M: np.ndarray) -> list[np.ndarray]:
    A, pivots = gf2_rref(M)
    rows, cols = A.shape
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = []
    pivot_row = {c: i for i, c in enumerate(pivots)}
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for c in pivots[::-1]:
            i = pivot_row[c]
            s = (A[i, :] & v).sum() % 2
            if s:
                v[c] = 1
        basis.append(v)
    return basis


def gf2_row_basis(M: np.ndarray) -> list[np.ndarray]:
    R, _ = gf2_rref(M)
    out = []
    for row in R:
        if row.any():
            out.append(row.copy())
    return out


def gf2_in_span(basis: list[np.ndarray], v: np.ndarray) -> bool:
    if not basis:
        return not v.any()
    M = np.array(basis, dtype=np.uint8)
    return gf2_rank(M) == gf2_rank(np.vstack([M, v]))


def quotient_basis(null_basis: list[np.ndarray], mod_basis: list[np.ndarray]) -> list[np.ndarray]:
    span = [b.copy() for b in mod_basis]
    qb: list[np.ndarray] = []
    for v in null_basis:
        if not gf2_in_span(span, v):
            qb.append(v.copy())
            span.append(v.copy())
    return qb


def gf2_coords_in_extension(v: np.ndarray, mod_basis: list[np.ndarray], basis: list[np.ndarray]) -> np.ndarray:
    all_basis = mod_basis + basis
    M = np.array(all_basis, dtype=np.uint8)
    A = M.T.copy().astype(np.uint8)
    b = v.copy().astype(np.uint8)
    rows, cols = A.shape
    Aug = np.concatenate([A, b[:, None]], axis=1)
    r = 0
    pivot_cols: list[int] = []
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if Aug[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            Aug[[r, pivot]] = Aug[[pivot, r]]
        for i in range(rows):
            if i != r and Aug[i, c]:
                Aug[i, :] ^= Aug[r, :]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if Aug[i, -1]:
            raise ValueError("vector not in affine span")
    x = np.zeros(cols, dtype=np.uint8)
    for i, c in enumerate(pivot_cols):
        x[c] = Aug[i, -1]
    return x[len(mod_basis):]


def group_elements() -> list[tuple[int, int]]:
    return [(a, b) for a in range(NX) for b in range(NY)]


GROUP = group_elements()
G2I = {g: i for i, g in enumerate(GROUP)}


def addg(g: tuple[int, int], h: tuple[int, int]) -> tuple[int, int]:
    return ((g[0] + h[0]) % NX, (g[1] + h[1]) % NY)


def negg(g: tuple[int, int]) -> tuple[int, int]:
    return ((-g[0]) % NX, (-g[1]) % NY)


def shift_set(S: set[tuple[int, int]], g: tuple[int, int]) -> set[tuple[int, int]]:
    return {addg(x, g) for x in S}


def transpose_set(S: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {negg(x) for x in S}


def support_vec(Lset: set[tuple[int, int]], Rset: set[tuple[int, int]]) -> np.ndarray:
    v = np.zeros(N_PHYS, dtype=np.uint8)
    for g in Lset:
        v[G2I[g]] = 1
    for g in Rset:
        v[72 + G2I[g]] = 1
    return v


def symp(x: np.ndarray, z: np.ndarray) -> int:
    return int((x & z).sum() % 2)


def mask_to_pauli(mask: int, n: int = N_DATA) -> str:
    x = mask & ((1 << n) - 1)
    z = mask >> n
    out: list[str] = []
    for i in range(n):
        xb = (x >> i) & 1
        zb = (z >> i) & 1
        if xb == 0 and zb == 0:
            out.append("I")
        elif xb == 1 and zb == 0:
            out.append("X")
        elif xb == 0 and zb == 1:
            out.append("Z")
        else:
            out.append("Y")
    return "".join(out)


def pauli_to_mask(pauli: str) -> int:
    x = 0
    z = 0
    for i, c in enumerate(pauli):
        if c in ("X", "Y"):
            x |= 1 << i
        if c in ("Z", "Y"):
            z |= 1 << i
    return x | (z << len(pauli))


@dataclass
class GrossCliffordSynth:
    native_rotations: list[dict]
    dist: np.ndarray
    parent: np.ndarray
    parent_gen: np.ndarray
    summary: dict

    @classmethod
    def load_precomputed(cls, base_dir: str) -> "GrossCliffordSynth":
        with open(os.path.join(base_dir, "native95.json"), "r", encoding="utf-8") as f:
            native = json.load(f)
        with open(os.path.join(base_dir, "summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
        data = np.load(os.path.join(base_dir, "closure_data.npz"))
        return cls(
            native_rotations=native,
            dist=data["dist"],
            parent=data["parent"],
            parent_gen=data["parent_gen"],
            summary=summary,
        )

    def decomposition_from_mask(self, mask: int) -> list[dict]:
        if mask <= 0 or mask >= (1 << (2 * N_DATA)):
            raise ValueError("mask out of range")
        if self.dist[mask] == np.uint16(65535):
            raise ValueError("rotation not reachable")
        if self.parent[mask] == mask:
            idx = int(self.parent_gen[mask])
            return [{"generator_index": idx, "sign": +1, "pauli": self.native_rotations[idx]["pauli"]}]
        prev = int(self.parent[mask])
        gid = int(self.parent_gen[mask])
        center = self.decomposition_from_mask(prev)
        g = {"generator_index": gid, "sign": +1, "pauli": self.native_rotations[gid]["pauli"]}
        return [{**g, "sign": -1}] + center + [g]

    def decomposition_from_pauli(self, pauli: str) -> list[dict]:
        return self.decomposition_from_mask(pauli_to_mask(pauli))

    def rotation_cost(self, mask: int) -> int:
        return 2 * int(self.dist[mask]) + 1


PIVOT_SEQUENCE = {
    "X": ("Z", "Y"),  # Z -> X -> Y
    "Y": ("X", "Z"),  # X -> Y -> Z
    "Z": ("Y", "X"),  # Y -> Z -> X
}


def native_measurement_plan(native: dict, sign: int = +1) -> dict:
    src = native["source_measurements"][0]
    pivot = src["pivot_pauli"]
    init_basis, final_basis = PIVOT_SEQUENCE[pivot]
    plan = {
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
        "init_measurement": {"target": "pivot", "basis": init_basis},
        "final_measurement": {"target": "pivot", "basis": final_basis},
        "paper_equation": 65,
        "rotation_sign": int(sign),
    }
    if sign < 0:
        plan["init_measurement"], plan["final_measurement"] = (
            plan["final_measurement"],
            plan["init_measurement"],
        )
    return plan


def synthesize_rotation(base_dir: str, pauli: str) -> dict:
    synth = GrossCliffordSynth.load_precomputed(base_dir)
    seq = synth.decomposition_from_pauli(pauli)
    steps = []
    for order, item in enumerate(seq):
        native = synth.native_rotations[item["generator_index"]]
        steps.append({
            "order": order,
            "generator_index": item["generator_index"],
            "rotation_pauli": native["pauli"],
            "rotation_sign": int(item["sign"]),
            "measurement_plan": native_measurement_plan(native, int(item["sign"])),
        })
    mask = pauli_to_mask(pauli)
    return {
        "target": {"pauli": pauli, "mask": int(mask)},
        "distance": int(synth.dist[mask]),
        "rotation_circuit_cost": synth.rotation_cost(mask),
        "native_sequence": seq,
        "compiler_steps": steps,
    }
