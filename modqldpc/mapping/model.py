from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Set, Optional, List, Iterable, Protocol, Callable
from collections import deque, defaultdict


BlockId = int
LogicalId = int
LocalId = int
CouplerId = str


# ============================================================
# Specs (unchanged)
# ============================================================

@dataclass(frozen=True)
class BlockSpec:
    block_id: BlockId
    num_logicals: int = 11
    has_root: bool = True
    name: Optional[str] = None


@dataclass(frozen=True)
class CouplerSpec:
    cid: CouplerId
    u: BlockId
    v: BlockId
    capacity: int = 1
    label: Optional[str] = None


# ============================================================
# Core hardware graph (keep it generic; builders create topology)
# ============================================================

@dataclass
class HardwareGraph:
    blocks: Dict[BlockId, BlockSpec] = field(default_factory=dict)
    couplers: Dict[CouplerId, CouplerSpec] = field(default_factory=dict)
    neighbors: Dict[BlockId, Set[BlockId]] = field(default_factory=dict)
    edge_to_coupler: Dict[Tuple[BlockId, BlockId], CouplerId] = field(default_factory=dict)

    logical_to_block: Dict[LogicalId, BlockId] = field(default_factory=dict)
    logical_to_local: Dict[LogicalId, LocalId] = field(default_factory=dict)

    port_capacity: Dict[BlockId, int] = field(default_factory=dict)
    default_num_logicals: int = 11
    default_block_port_capacity: int = 1

    # ----- Construction API -----

    def add_block(
        self,
        block_id: BlockId,
        *,
        num_logicals: Optional[int] = None,
        has_root: bool = True,
        name: Optional[str] = None,
        port_capacity: Optional[int] = None,
    ) -> None:
        if block_id in self.blocks:
            raise ValueError(f"Block {block_id} already exists.")
        nl = self.default_num_logicals if num_logicals is None else int(num_logicals)
        if nl <= 0:
            raise ValueError("num_logicals must be positive.")
        self.blocks[block_id] = BlockSpec(block_id=block_id, num_logicals=nl, has_root=has_root, name=name)
        self.neighbors.setdefault(block_id, set())
        self.port_capacity[block_id] = self.default_block_port_capacity if port_capacity is None else int(port_capacity)

    def add_coupler(
        self,
        u: BlockId,
        v: BlockId,
        *,
        capacity: int = 1,
        cid: Optional[CouplerId] = None,
        label: Optional[str] = None,
    ) -> CouplerId:
        if u == v:
            raise ValueError("Coupler endpoints must be distinct.")
        if u not in self.blocks or v not in self.blocks:
            raise ValueError("Both endpoints must be existing blocks.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        a, b = (u, v) if u < v else (v, u)
        if (a, b) in self.edge_to_coupler:
            raise ValueError(f"Coupler already exists between blocks {a} and {b}.")

        if cid is None:
            cid = f"c_{a}_{b}"
        if cid in self.couplers:
            raise ValueError(f"Coupler id {cid} already exists.")

        self.couplers[cid] = CouplerSpec(cid=cid, u=a, v=b, capacity=int(capacity), label=label)
        self.edge_to_coupler[(a, b)] = cid
        self.neighbors[a].add(b)
        self.neighbors[b].add(a)
        return cid

    def add_mapping(self, logical: LogicalId, block_id: BlockId, local: LocalId) -> None:
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found.")
        spec = self.blocks[block_id]
        if not (0 <= local < spec.num_logicals):
            raise ValueError(f"local must be in [0, {spec.num_logicals-1}] for block {block_id}.")
        if logical in self.logical_to_block:
            raise ValueError(f"Logical {logical} already mapped.")
        self.logical_to_block[logical] = block_id
        self.logical_to_local[logical] = int(local)

    # ----- Query -----

    def coupler_id(self, u: BlockId, v: BlockId) -> Optional[CouplerId]:
        if u == v:
            return None
        a, b = (u, v) if u < v else (v, u)
        return self.edge_to_coupler.get((a, b))

    def shortest_path(self, src: BlockId, dst: BlockId) -> Optional[List[BlockId]]:
        """Unweighted shortest path (BFS)."""
        if src == dst:
            return [src]
        if src not in self.blocks or dst not in self.blocks:
            return None

        q = deque([src])
        prev: Dict[BlockId, Optional[BlockId]] = {src: None}
        while q:
            u = q.popleft()
            for v in self.neighbors.get(u, set()):
                if v in prev:
                    continue
                prev[v] = u
                if v == dst:
                    q.clear()
                    break
                q.append(v)

        if dst not in prev:
            return None

        path: List[BlockId] = []
        cur: Optional[BlockId] = dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def all_simple_paths(
        self,
        src: BlockId,
        dst: BlockId,
        *,
        max_hops: Optional[int] = None,
    ) -> List[List[BlockId]]:
        """
        Enumerate all simple paths src->dst using DFS.
        WARNING: exponential in worst case. Use max_hops for safety.
        """
        if src not in self.blocks or dst not in self.blocks:
            return []
        if src == dst:
            return [[src]]

        max_hops_eff = max_hops if max_hops is not None else 10_000  # caller should set for big graphs

        out: List[List[BlockId]] = []
        stack: List[Tuple[BlockId, List[BlockId], Set[BlockId]]] = [(src, [src], {src})]

        while stack:
            u, path, seen = stack.pop()
            if len(path) - 1 > max_hops_eff:
                continue
            for v in self.neighbors.get(u, set()):
                if v in seen:
                    continue
                if v == dst:
                    out.append(path + [v])
                else:
                    stack.append((v, path + [v], seen | {v}))
        return out

    def validate(self) -> None:
        for b in self.port_capacity:
            if b not in self.blocks:
                raise ValueError(f"port_capacity specified for missing block {b}.")
        for cid, c in self.couplers.items():
            if c.u not in self.blocks or c.v not in self.blocks:
                raise ValueError(f"Coupler {cid} references missing block.")
            if c.v not in self.neighbors.get(c.u, set()) or c.u not in self.neighbors.get(c.v, set()):
                raise ValueError(f"Coupler {cid} neighbor sets inconsistent.")
            a, b = (c.u, c.v) if c.u < c.v else (c.v, c.u)
            if self.edge_to_coupler.get((a, b)) != cid:
                raise ValueError(f"edge_to_coupler inconsistent for {cid}.")


# ============================================================
# Topology builder pattern
#   - create blocks
#   - create couplers via a topology strategy
# ============================================================

class TopologyBuilder(Protocol):
    def build(
        self,
        *,
        graph: HardwareGraph,
        block_ids: List[BlockId],
        coupler_capacity: int = 1,
        cid_fn: Optional[Callable[[BlockId, BlockId], CouplerId]] = None,
    ) -> None:
        ...


def _default_cid_fn(u: BlockId, v: BlockId) -> CouplerId:
    a, b = (u, v) if u < v else (v, u)
    return f"c_{a}_{b}"


@dataclass(frozen=True)
class RingTopology(TopologyBuilder):
    """
    Ring over N blocks in the given order: ids[0]-ids[1]-...-ids[N-1]-ids[0]
    """
    def build(
        self,
        *,
        graph: HardwareGraph,
        block_ids: List[BlockId],
        coupler_capacity: int = 1,
        cid_fn: Optional[Callable[[BlockId, BlockId], CouplerId]] = None,
    ) -> None:
        if len(block_ids) < 2:
            raise ValueError("Ring requires at least 2 blocks.")
        cid_fn = cid_fn or _default_cid_fn
        n = len(block_ids)
        for i in range(n):
            u = block_ids[i]
            v = block_ids[(i + 1) % n]
            graph.add_coupler(u, v, capacity=coupler_capacity, cid=cid_fn(u, v))


@dataclass(frozen=True)
class GridTopology(TopologyBuilder):
    """
    2D grid topology (rows x cols), 4-neighborhood.
    block_ids length must be rows*cols.
    """
    rows: int
    cols: int
    wrap_rows: bool = False  # torus option
    wrap_cols: bool = False

    def build(
        self,
        *,
        graph: HardwareGraph,
        block_ids: List[BlockId],
        coupler_capacity: int = 1,
        cid_fn: Optional[Callable[[BlockId, BlockId], CouplerId]] = None,
    ) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows/cols must be positive.")
        if len(block_ids) != self.rows * self.cols:
            raise ValueError(f"Need rows*cols={self.rows*self.cols} block_ids, got {len(block_ids)}")
        cid_fn = cid_fn or _default_cid_fn

        def at(r: int, c: int) -> BlockId:
            return block_ids[r * self.cols + c]

        for r in range(self.rows):
            for c in range(self.cols):
                # right neighbor
                if c + 1 < self.cols:
                    graph.add_coupler(at(r, c), at(r, c + 1), capacity=coupler_capacity, cid=cid_fn(at(r, c), at(r, c + 1)))
                elif self.wrap_cols and self.cols > 1:
                    graph.add_coupler(at(r, c), at(r, 0), capacity=coupler_capacity, cid=cid_fn(at(r, c), at(r, 0)))

                # down neighbor
                if r + 1 < self.rows:
                    graph.add_coupler(at(r, c), at(r + 1, c), capacity=coupler_capacity, cid=cid_fn(at(r, c), at(r + 1, c)))
                elif self.wrap_rows and self.rows > 1:
                    graph.add_coupler(at(r, c), at(0, c), capacity=coupler_capacity, cid=cid_fn(at(r, c), at(0, c)))


# ============================================================
# Factory: create a graph with a chosen topology
# ============================================================

@dataclass(frozen=True)
class GraphFactory:
    """
    Creates blocks (with shared defaults) and then applies a topology builder.
    """
    default_num_logicals: int = 11
    default_port_capacity: int = 1
    has_root: bool = True

    def build(
        self,
        *,
        topology: TopologyBuilder,
        block_ids: List[BlockId],
        num_logicals: Optional[int] = None,
        per_block_num_logicals: Optional[Dict[BlockId, int]] = None,
        per_block_port_capacity: Optional[Dict[BlockId, int]] = None,
        coupler_capacity: int = 1,
        cid_fn: Optional[Callable[[BlockId, BlockId], CouplerId]] = None,
    ) -> HardwareGraph:
        g = HardwareGraph(
            default_num_logicals=self.default_num_logicals,
            default_block_port_capacity=self.default_port_capacity,
        )
        per_block_num_logicals = per_block_num_logicals or {}
        per_block_port_capacity = per_block_port_capacity or {}

        for bid in block_ids:
            nl = per_block_num_logicals.get(bid, num_logicals)
            pc = per_block_port_capacity.get(bid, None)
            g.add_block(
                bid,
                num_logicals=nl,
                has_root=self.has_root,
                port_capacity=pc,
            )

        topology.build(
            graph=g,
            block_ids=block_ids,
            coupler_capacity=coupler_capacity,
            cid_fn=cid_fn,
        )

        g.validate()
        return g


# ============================================================
# Example usage (first version)
# ============================================================

if __name__ == "__main__":
    factory = GraphFactory(default_num_logicals=11, default_port_capacity=1)

    # Ring with 6 blocks (ids chosen by you)
    ring = factory.build(
        topology=RingTopology(),
        block_ids=[1, 2, 3, 4, 5, 6],
        coupler_capacity=1,
    )
    print("Ring shortest 1->4:", ring.shortest_path(1, 4))
    print("Ring all paths 1->4 (max_hops=6):", ring.all_simple_paths(1, 4, max_hops=6))

    # 3x3 grid
    grid_ids = list(range(1, 10))
    grid = factory.build(
        topology=GridTopology(rows=3, cols=3),
        block_ids=grid_ids,
        coupler_capacity=1,
    )
    print("Grid shortest 1->9:", grid.shortest_path(1, 9))
    print("Grid all paths 1->9 (max_hops=6):", grid.all_simple_paths(1, 9, max_hops=6))