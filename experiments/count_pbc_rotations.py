"""
experiments/count_pbc_rotations.py

For each PBC JSON in circuits/benchmarks/pbc/, prints:
  circuit_name   n_rotations   n_layers   rotations_per_layer

n_rotations = total number of rotation IDs scheduled across all layers
            = sum(len(layer) for layer in layer_ids)

Run:
  python experiments/count_pbc_rotations.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PBC_DIR = Path(__file__).resolve().parent.parent / "circuits" / "benchmarks" / "pbc"

rows = []
for fname in sorted(os.listdir(PBC_DIR)):
    if not fname.endswith(".json"):
        continue
    stem = fname[:-5]
    # strip _PBC suffix if present
    circ = stem[:-4] if stem.endswith("_PBC") else stem

    data = json.loads((PBC_DIR / fname).read_text())
    layer_ids = data.get("layer_ids", [])

    n_layers     = len(layer_ids)
    n_rotations  = sum(len(layer) for layer in layer_ids)
    avg_per_layer = n_rotations / n_layers if n_layers else 0

    # print any top-level keys other than schema and layer_ids
    extra_keys = [k for k in data if k not in ("schema", "layer_ids")]

    rows.append((circ, n_rotations, n_layers, avg_per_layer, extra_keys))

print(f"{'Circuit':<30s}  {'n_rotations':>12s}  {'n_layers':>10s}  {'avg/layer':>10s}  extra_keys")
print("-" * 80)
for circ, nr, nl, avg, extra in sorted(rows, key=lambda r: r[1]):
    print(f"{circ:<30s}  {nr:>12,d}  {nl:>10,d}  {avg:>10.1f}  {extra}")
