# modqldpc/lowering/ir.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque


NodeId = str
BlockId = int
CouplerId = str


# -------------------------
# Lightweight data payloads
# -------------------------


@dataclass(frozen=True)
class ClassicalKey:
    """
    Named classical bit/symbol produced by measurements.
    Example: "bPZ_L00_R005", "bXm_L00_R005"
    """
    name: str


# -------------------------
# Node kinds (keep minimal)
# -------------------------

@dataclass
class ExecNode:
    """
    Base class for lowering IR nodes.
    - nid must be unique in the DAG.
    - blocks/couplers are *resources touched* (scheduler uses later).
    - produces/consumes are classical keys (for determinism & rewrite).
    """
    nid: NodeId
    kind: str

    blocks: List[BlockId] = field(default_factory=list)
    couplers: List[CouplerId] = field(default_factory=list)

    duration: int = 1  # abstract time units (scheduler can override/scale)
    meta: Dict[str, Any] = field(default_factory=dict)

    produces: List[ClassicalKey] = field(default_factory=list)
    consumes: List[ClassicalKey] = field(default_factory=list)


# ---- concrete node labels (stringly-typed kind for schema stability) ----
K_INIT_PIVOT = "init_pivot"
K_LOCAL_COUPLE = "local_couple"
K_INTERBLOCK_LINK = "interblock_link"
K_MEAS_PARITY_PZ = "meas_parity_PZ"     # measure (P ⊗ Z_m) -> bPZ
K_MEAS_MAGIC_X = "meas_magic_X"        # measure X_m -> bXm
K_FRAME_UPDATE = "frame_update"        # purely classical; updates pauli/clifford frame


def node_init_pivot(nid: str, block: int, *, duration: int = 1, meta: Optional[Dict[str, Any]] = None) -> ExecNode:
    return ExecNode(
        nid=nid,
        kind=K_INIT_PIVOT,
        blocks=[block],
        duration=duration,
        meta=meta or {},
    )


def node_local_couple(
    nid: str,
    block: int,
    *,
    local_ops: Dict[int, str],
    pivot_id: str,
    duration: int = 1,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecNode:
    # local_ops: {logical_id_in_block_or_global: 'X'/'Y'/'Z'}
    m = dict(meta or {})
    m.update({"pivot_id": pivot_id, "local_ops": local_ops})
    return ExecNode(
        nid=nid,
        kind=K_LOCAL_COUPLE,
        blocks=[block],
        duration=duration,
        meta=m,
    )


def node_interblock_link(
    nid: str,
    *,
    blocks: List[int],
    couplers: List[str],
    duration: int = 1,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecNode:
    return ExecNode(
        nid=nid,
        kind=K_INTERBLOCK_LINK,
        blocks=list(blocks),
        couplers=list(couplers),
        duration=duration,
        meta=meta or {},
    )


def node_meas_parity_PZ(
    nid: str,
    *,
    pauli: PauliString,
    magic_id: str,
    out_key: str,
    blocks: List[int],
    couplers: Optional[List[str]] = None,
    duration: int = 1,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecNode:
    m = dict(meta or {})
    m.update({"pauli": {"sign": pauli.sign, "tensor": pauli.tensor}, "magic_id": magic_id})
    return ExecNode(
        nid=nid,
        kind=K_MEAS_PARITY_PZ,
        blocks=list(blocks),
        couplers=list(couplers or []),
        duration=duration,
        meta=m,
        produces=[ClassicalKey(out_key)],
    )


def node_meas_magic_X(
    nid: str,
    *,
    magic_id: str,
    out_key: str,
    block: int,
    duration: int = 1,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecNode:
    m = dict(meta or {})
    m.update({"magic_id": magic_id})
    return ExecNode(
        nid=nid,
        kind=K_MEAS_MAGIC_X,
        blocks=[block],
        duration=duration,
        meta=m,
        produces=[ClassicalKey(out_key)],
    )


def node_frame_update(
    nid: str,
    *,
    update_kind: str,     # "clifford_pi4" or "pauli"
    depends_on: str,      # classical key name
    axis: PauliString,    # the P whose correction/byproduct this is
    duration: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> ExecNode:
    m = dict(meta or {})
    m.update(
        {
            "update_kind": update_kind,
            "depends_on": depends_on,
            "axis": {"sign": axis.sign, "tensor": axis.tensor},
        }
    )
    return ExecNode(
        nid=nid,
        kind=K_FRAME_UPDATE,
        duration=duration,
        meta=m,
        consumes=[ClassicalKey(depends_on)],
    )


# -------------------------
# DAG container
# -------------------------

@dataclass
class ExecDAG:
    """
    Minimal DAG:
      nodes: nid -> ExecNode
      edges: u -> v means u must complete before v can start
    """
    nodes: Dict[NodeId, ExecNode] = field(default_factory=dict)
    succ: Dict[NodeId, Set[NodeId]] = field(default_factory=dict)
    pred: Dict[NodeId, Set[NodeId]] = field(default_factory=dict)

    def add_node(self, node: ExecNode) -> None:
        if node.nid in self.nodes:
            raise ValueError(f"Duplicate node id: {node.nid}")
        self.nodes[node.nid] = node
        self.succ.setdefault(node.nid, set())
        self.pred.setdefault(node.nid, set())

    def add_edge(self, u: NodeId, v: NodeId) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"add_edge requires existing nodes: {u}->{v}")
        if u == v:
            raise ValueError("Self-loop edge not allowed.")
        self.succ[u].add(v)
        self.pred[v].add(u)

    def indegree(self, nid: NodeId) -> int:
        return len(self.pred.get(nid, set()))

    def topological_order(self) -> List[NodeId]:
        """
        Kahn's algorithm. Raises if cycle exists.
        """
        indeg = {nid: len(self.pred[nid]) for nid in self.nodes}
        q = deque([nid for nid, d in indeg.items() if d == 0])
        order: List[NodeId] = []

        while q:
            u = q.popleft()
            order.append(u)
            for v in list(self.succ[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in ExecDAG.")
        return order

    def validate(self) -> None:
        # basic: edges reference nodes
        for u, vs in self.succ.items():
            if u not in self.nodes:
                raise ValueError(f"succ references missing node {u}")
            for v in vs:
                if v not in self.nodes:
                    raise ValueError(f"edge {u}->{v} references missing node {v}")
        # produce/consume sanity: consumed keys should be produced somewhere (optional strictness later)

    # -------------------------
    # Serialization for artifacts
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        nodes_out = {}
        for nid, n in self.nodes.items():
            nodes_out[nid] = {
                "nid": n.nid,
                "kind": n.kind,
                "blocks": list(n.blocks),
                "couplers": list(n.couplers),
                "duration": int(n.duration),
                "meta": n.meta,
                "produces": [k.name for k in n.produces],
                "consumes": [k.name for k in n.consumes],
            }
        edges_out: List[Tuple[str, str]] = []
        for u, vs in self.succ.items():
            for v in vs:
                edges_out.append((u, v))
        edges_out.sort()
        return {
            "schema": "modqldpc.execdag.v1",
            "nodes": nodes_out,
            "edges": edges_out,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExecDAG":
        if d.get("schema") != "modqldpc.execdag.v1":
            raise ValueError(f"Unknown ExecDAG schema: {d.get('schema')}")
        dag = ExecDAG()
        for nid, nd in d["nodes"].items():
            dag.add_node(
                ExecNode(
                    nid=nd["nid"],
                    kind=nd["kind"],
                    blocks=list(nd.get("blocks", [])),
                    couplers=list(nd.get("couplers", [])),
                    duration=int(nd.get("duration", 1)),
                    meta=dict(nd.get("meta", {})),
                    produces=[ClassicalKey(x) for x in nd.get("produces", [])],
                    consumes=[ClassicalKey(x) for x in nd.get("consumes", [])],
                )
            )
        for u, v in d.get("edges", []):
            dag.add_edge(u, v)
        return dag