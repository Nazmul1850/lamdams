# modqldpc/lowering/visualize.py
from __future__ import annotations

from typing import Dict
from .ir import ExecDAG, ExecNode


# Basic color map by node kind
_NODE_COLORS: Dict[str, str] = {
    "init_pivot": "#8dd3c7",
    "local_couple": "#ffffb3",
    "interblock_link": "#bebada",
    "meas_parity_PZ": "#fb8072",
    "meas_magic_X": "#80b1d3",
    "frame_update": "#fdb462",
}


def _escape(s: str) -> str:
    return s.replace('"', '\\"')


def dag_to_dot(
    dag: ExecDAG,
    *,
    show_blocks: bool = True,
    show_couplers: bool = False,
    show_classical: bool = True,
) -> str:
    """
    Convert ExecDAG into Graphviz DOT format string.
    """

    lines = []
    lines.append("digraph ExecDAG {")
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box, style=filled, fontname=Courier];")

    # ---- nodes ----
    for nid in sorted(dag.nodes):
        node: ExecNode = dag.nodes[nid]

        color = _NODE_COLORS.get(node.kind, "#ffffff")

        label_parts = [nid, node.kind]

        if show_blocks and node.blocks:
            label_parts.append(f"B={node.blocks}")

        if show_couplers and node.couplers:
            label_parts.append(f"C={node.couplers}")

        if show_classical:
            if node.produces:
                label_parts.append("→ " + ",".join(k.name for k in node.produces))
            if node.consumes:
                label_parts.append("← " + ",".join(k.name for k in node.consumes))

        label = "\\n".join(_escape(p) for p in label_parts)

        lines.append(
            f'  "{nid}" [label="{label}", fillcolor="{color}"];'
        )

    # ---- edges ----
    for u in sorted(dag.succ):
        for v in sorted(dag.succ[u]):
            lines.append(f'  "{u}" -> "{v}";')

    lines.append("}")
    return "\n".join(lines)