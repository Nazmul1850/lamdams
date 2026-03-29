from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..mapping.model import HardwareGraph
from ..mapping.types import MappingPlan
from ..core.types import PauliRotation


@dataclass
class CircuitProfile:
    """Computed once from frontend output before the main pipeline loop."""
    n_logicals: int
    n_rotations_total: int
    n_layers: int
    layer_sizes: List[int]            # rotations per layer, indexed by layer_id
    all_rotation_weights: List[int]   # support size (# non-I) for every rotation
    all_rotation_angles: List[float]  # angle for every rotation
    block_ids: List[int]              # sorted hardware block IDs
    logical_to_block: Dict[int, int]  # logical qubit -> block id


@dataclass
class LayerProfile:
    """All per-layer stats collected during one pass through the pipeline loop."""
    layer_id: int

    # ---- Circuit structure ----
    n_rotations: int
    rotation_weights: List[int]    # non-I Pauli count per rotation in this layer
    rotation_angles: List[float]   # angle per rotation

    # ---- Lowering ----
    dag_n_nodes: int
    dag_node_counts: Dict[str, int]  # {node_kind: count}

    # ---- Scheduling ----
    depth: int                         # layer execution depth (ex.depth)
    block_busy_slots: Dict[int, int]   # block_id -> total busy time-steps
    parallelism_steps: List[Dict[str, Any]]  # [{t, total_nodes, kind_counts}]

    # ---- Frame execution ----
    n_rewrites: int
    n_support_changes: int
    n_angle_flips: int

    # ---- Routing / Mapping quality ----
    rotation_max_hops: List[int]  # per rotation: max BFS hops between support blocks


def _pauli_label(r: PauliRotation) -> str:
    """Return the bare Pauli tensor string (no sign prefix) from a PauliRotation."""
    return r.axis.lstrip("+-")


def collect_circuit_profile(
    *,
    n_logicals: int,
    layers: Dict[int, List[int]],
    rotations: Dict[int, PauliRotation],
    hw: HardwareGraph,
    plan: MappingPlan,
) -> CircuitProfile:
    """Call once before the main loop, using the original program rotations."""
    layer_sizes = [len(layer) for layer in sorted(layers)]
    all_weights = []
    all_angles = []
    for r in rotations.values():
        label = _pauli_label(r)
        all_weights.append(sum(1 for ch in label if ch != "I"))
        all_angles.append(float(r.angle))

    return CircuitProfile(
        n_logicals=n_logicals,
        n_rotations_total=len(rotations),
        n_layers=len(layers),
        layer_sizes=layer_sizes,
        all_rotation_weights=all_weights,
        all_rotation_angles=all_angles,
        block_ids=sorted(hw.blocks),
        logical_to_block=dict(plan.logical_to_block),
    )


def collect_layer_profile(
    *,
    layer_id: int,
    rotation_indices: List[int],
    effective_rotations: Dict[int, PauliRotation],
    res,    # LayerLoweringResult
    S,      # Schedule
    ex,     # LayerExecutionResult
    hw: HardwareGraph,
    plan: MappingPlan,
) -> LayerProfile:
    """Collect all stats for one layer after scheduling + execution."""

    # ---- Circuit structure ----
    rotation_weights: List[int] = []
    rotation_angles: List[float] = []
    for ridx in rotation_indices:
        r = effective_rotations[ridx]
        label = _pauli_label(r)
        rotation_weights.append(sum(1 for ch in label if ch != "I"))
        rotation_angles.append(float(r.angle))

    # ---- Lowering ----
    dag_node_counts = dict(Counter(n.kind for n in res.dag.nodes.values()))
    dag_n_nodes = len(res.dag.nodes)

    # ---- Scheduling: block busy slots ----
    entries = S.meta.get("entries", {})
    block_busy: Dict[int, int] = defaultdict(int)
    for nid, se in entries.items():
        node = res.dag.nodes[nid]
        duration = se["end"] - se["start"]
        for b in node.blocks:
            block_busy[b] += duration

    # ---- Scheduling: parallelism per timestep (grouped by start time) ----
    parallelism_steps: List[Dict[str, Any]] = []
    for step in S.steps:
        kind_counts = dict(Counter(res.dag.nodes[nid].kind for nid in step.nodes))
        parallelism_steps.append({
            "t": step.t,
            "total_nodes": len(step.nodes),
            "kind_counts": kind_counts,
        })

    # ---- Frame execution ----
    n_rewrites = len(ex.rewrite_log)
    n_support_changes = sum(1 for r in ex.rewrite_log if r.changed_support)
    n_angle_flips = sum(1 for r in ex.rewrite_log if r.angle_before != r.angle_after)

    # ---- Routing: max BFS hops per rotation ----
    rotation_max_hops: List[int] = []
    for ridx in rotation_indices:
        r = effective_rotations[ridx]
        label = _pauli_label(r)
        support_lids = [i for i, ch in enumerate(label) if ch != "I"]
        support_blocks = list({
            plan.logical_to_block[lid]
            for lid in support_lids
            if lid in plan.logical_to_block
        })
        max_hops = 0
        for i in range(len(support_blocks)):
            for j in range(i + 1, len(support_blocks)):
                path = hw.shortest_path(support_blocks[i], support_blocks[j])
                if path is not None:
                    max_hops = max(max_hops, len(path) - 1)
        rotation_max_hops.append(max_hops)

    return LayerProfile(
        layer_id=layer_id,
        n_rotations=len(rotation_indices),
        rotation_weights=rotation_weights,
        rotation_angles=rotation_angles,
        dag_n_nodes=dag_n_nodes,
        dag_node_counts=dag_node_counts,
        depth=ex.depth,
        block_busy_slots=dict(block_busy),
        parallelism_steps=parallelism_steps,
        n_rewrites=n_rewrites,
        n_support_changes=n_support_changes,
        n_angle_flips=n_angle_flips,
        rotation_max_hops=rotation_max_hops,
    )
