"""
Hardware generation helpers.

make_hardware() produces a HardwareGraph from high-level parameters:
  - n_logicals   : number of logical qubits that must fit
  - topology     : "grid" or "ring"
  - sparse_pct   : fraction of total block capacity LEFT EMPTY
                   0.0 = dense (minimum blocks), 0.5 = half slots unused
  - n_data       : data slots per block (Gross code = 11)
  - coupler_capacity : per-coupler capacity (default 1)

Grid shape is chosen to be as square as possible.
Ring always uses a single ring over all blocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .model import GraphFactory, GridTopology, RingTopology, HardwareGraph


N_DATA_DEFAULT = 11  # Gross code data qubits per block


@dataclass(frozen=True)
class HardwareSpec:
    """Summary of what was built, for logging / labelling graphs."""
    topology: str           # "grid" or "ring"
    n_blocks: int
    n_data: int             # data slots per block
    total_capacity: int     # n_blocks * n_data
    n_logicals: int         # logical qubits actually placed
    sparse_pct: float       # requested sparsity (0 = dense)
    actual_fill_rate: float # n_logicals / total_capacity
    grid_rows: Optional[int] = None
    grid_cols: Optional[int] = None

    def label(self) -> str:
        """Short string suitable for plot titles / legend entries."""
        topo = f"{self.grid_rows}×{self.grid_cols}" if self.topology == "grid" else "ring"
        fill = f"{self.actual_fill_rate*100:.0f}%"
        return f"{topo} ({self.n_blocks}B, {fill} fill)"


def _grid_shape(n_blocks: int) -> tuple[int, int]:
    """Return (rows, cols) for the most-square grid that fits n_blocks."""
    cols = math.ceil(math.sqrt(n_blocks))
    rows = math.ceil(n_blocks / cols)
    return rows, cols


def _n_blocks_for(n_logicals: int, n_data: int, sparse_pct: float) -> int:
    """Minimum number of blocks so that (1-sparse_pct) fraction is filled."""
    if not (0.0 <= sparse_pct < 1.0):
        raise ValueError(f"sparse_pct must be in [0, 1), got {sparse_pct}")
    fill_rate = 1.0 - sparse_pct
    # n_logicals <= fill_rate * n_blocks * n_data
    # n_blocks >= n_logicals / (fill_rate * n_data)
    n = math.ceil(n_logicals / (fill_rate * n_data))
    return max(n, 2)  # minimum 2 blocks (ring requires ≥2; grid 1×1 is trivial)


def make_hardware(
    n_logicals: int,
    *,
    topology: str = "grid",
    sparse_pct: float = 0.0,
    n_data: int = N_DATA_DEFAULT,
    coupler_capacity: int = 1,
) -> tuple[HardwareGraph, HardwareSpec]:
    """
    Build a HardwareGraph for the requested logical qubit count and density.

    Parameters
    ----------
    n_logicals     : number of logical qubits that must be mappable
    topology       : "grid" | "ring"
    sparse_pct     : fraction of total capacity left empty (0 = dense)
    n_data         : data qubit slots per block (default 11 for Gross code)
    coupler_capacity : capacity per coupler link (default 1)

    Returns
    -------
    (hw, spec) where hw is the HardwareGraph and spec carries summary info.
    """
    if topology not in ("grid", "ring"):
        raise ValueError(f"topology must be 'grid' or 'ring', got '{topology}'")
    if n_logicals < 1:
        raise ValueError("n_logicals must be ≥ 1")

    n_blocks = _n_blocks_for(n_logicals, n_data, sparse_pct)
    factory = GraphFactory(default_num_logicals=n_data, default_port_capacity=1)

    if topology == "ring":
        block_ids = list(range(1, n_blocks + 1))
        hw = factory.build(
            topology=RingTopology(),
            block_ids=block_ids,
            coupler_capacity=coupler_capacity,
        )
        spec = HardwareSpec(
            topology="ring",
            n_blocks=n_blocks,
            n_data=n_data,
            total_capacity=n_blocks * n_data,
            n_logicals=n_logicals,
            sparse_pct=sparse_pct,
            actual_fill_rate=n_logicals / (n_blocks * n_data),
        )

    else:  # grid
        rows, cols = _grid_shape(n_blocks)
        n_blocks_actual = rows * cols  # may be slightly > n_blocks (rounded up for square)
        block_ids = list(range(1, n_blocks_actual + 1))
        hw = factory.build(
            topology=GridTopology(rows=rows, cols=cols),
            block_ids=block_ids,
            coupler_capacity=coupler_capacity,
        )
        spec = HardwareSpec(
            topology="grid",
            n_blocks=n_blocks_actual,
            n_data=n_data,
            total_capacity=n_blocks_actual * n_data,
            n_logicals=n_logicals,
            sparse_pct=sparse_pct,
            actual_fill_rate=n_logicals / (n_blocks_actual * n_data),
            grid_rows=rows,
            grid_cols=cols,
        )

    return hw, spec
