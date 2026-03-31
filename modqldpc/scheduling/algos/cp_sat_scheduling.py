# modqldpc/scheduling/algos/cp_sat_scheduling.py
from __future__ import annotations

import pathlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from modqldpc.lowering.ir import (
    K_INIT_PIVOT,
    K_INTERBLOCK_LINK,
    K_LOCAL_COUPLE,
    K_MEAS_MAGIC_X,
    K_MEAS_PARITY_PZ,
)
from ortools.sat.python import cp_model

from ..base import BaseScheduler
from ..types import SchedulingProblem, Schedule, ScheduleStep, ScheduleEntry


# ---------------------------------------------------------------------------
# Phase A: Route alternative helpers
# ---------------------------------------------------------------------------

def _all_simple_paths(
    neighbors: Dict[int, Set[int]],
    src: int,
    dst: int,
    max_hops: int,
) -> List[List[int]]:
    """BFS over simple paths from src to dst with at most max_hops edges."""
    results: List[List[int]] = []
    queue: deque = deque([(src, [src])])
    while queue:
        node, path = queue.popleft()
        if node == dst:
            results.append(path)
            continue
        if len(path) - 1 >= max_hops:
            continue
        for nb in neighbors.get(node, set()):
            if nb not in path:
                queue.append((nb, path + [nb]))
    return results


@dataclass(frozen=True)
class _RouteOption:
    """One candidate route for a link node."""
    blocks: Tuple[int, ...]       # all blocks touched (deduped) — for HW interval registration
    couplers: Tuple[str, ...]     # all couplers used (deduped)
    duration: int                 # max hop-length across all source paths
    paths: Tuple[Tuple[int, ...], ...]  # per-source path sequences (block IDs in hop order)


def _enumerate_route_options(
    node: Any,
    hw: Any,
    max_hops: int,
) -> List[_RouteOption]:
    """
    For a link node with meta containing magic_block, source_blocks, route_paths,
    enumerate all feasible route combinations (one path per source block).

    Each combination is a Cartesian product of per-source alternative paths.
    We cap the total number of combinations at 64 to avoid model explosion.

    paths field stores the per-source path lists for per-hop interval generation.
    """
    meta = getattr(node, "meta", {}) or {}
    magic_block: Optional[int] = meta.get("magic_block")
    source_blocks: Optional[List[int]] = meta.get("source_blocks")

    if magic_block is None or not source_blocks:
        # Fallback: use the single fixed route already encoded in node.
        # paths=() signals no per-hop info available.
        blocks = tuple(sorted(set(getattr(node, "blocks", []))))
        couplers = tuple(dict.fromkeys(getattr(node, "couplers", [])))
        duration = getattr(node, "duration", 1) or 1
        return [_RouteOption(blocks=blocks, couplers=couplers, duration=duration, paths=())]

    # Build per-source candidate paths
    per_source: List[List[List[int]]] = []
    for src in source_blocks:
        paths = _all_simple_paths(hw.neighbors, src, magic_block, max_hops)
        if not paths:
            orig_route_paths: List[List[int]] = meta.get("route_paths", [])
            fallback = next((p for p in orig_route_paths if p[0] == src), None)
            paths = [fallback] if fallback else [[src, magic_block]]
        per_source.append(paths)

    # Cartesian product of per-source path choices
    MAX_COMBOS = 64
    combos: List[List[List[int]]] = [[]]
    for paths in per_source:
        new_combos: List[List[List[int]]] = []
        for existing in combos:
            for p in paths:
                new_combos.append(existing + [p])
                if len(new_combos) >= MAX_COMBOS:
                    break
            if len(new_combos) >= MAX_COMBOS:
                break
        combos = new_combos

    options: List[_RouteOption] = []
    seen_sigs: Set[Tuple] = set()

    for combo in combos:
        all_blocks: List[int] = [magic_block]
        all_couplers: List[str] = []
        hop_lengths: List[int] = []

        for path in combo:
            hop_lengths.append(len(path) - 1)
            for b in path:
                if b not in all_blocks:
                    all_blocks.append(b)
            for u, v in zip(path, path[1:]):
                cid = hw.coupler_id(u, v)
                if cid and cid not in all_couplers:
                    all_couplers.append(cid)

        sig = (tuple(sorted(all_blocks)), tuple(all_couplers))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        duration = max(hop_lengths) if hop_lengths else 1
        options.append(_RouteOption(
            blocks=tuple(sorted(all_blocks)),
            couplers=tuple(all_couplers),
            duration=max(duration, 1),
            paths=tuple(tuple(p) for p in combo),
        ))

    return options if options else [_RouteOption(
        blocks=tuple(sorted(set(getattr(node, "blocks", [])))),
        couplers=tuple(dict.fromkeys(getattr(node, "couplers", []))),
        duration=max(getattr(node, "duration", 1) or 1, 1),
        paths=(),
    )]


# ---------------------------------------------------------------------------
# Phase B: Component role extraction
# ---------------------------------------------------------------------------

@dataclass
class _CompRole:
    """
    Per-rotation (component) role summary used to generate ownership constraints.

    init_block_map: maps each participant block → the init_pivot nid that first owns it.
    magic_block:    the block measured by Xm (locks from link_start to Xm.end).
    is_multiblock:  True when a link node exists (multi-block rotation).
    """
    cid: int
    init_block_map: Dict[int, str]   # participant_block → init_pivot nid
    link_nid: Optional[str]
    pz_nid: Optional[str]
    xm_nid: Optional[str]
    magic_block: Optional[int]
    is_multiblock: bool


def _extract_comp_roles(dag: Any) -> Tuple[List[_CompRole], Dict[str, int]]:
    """
    Decompose the DAG into connected components (one per rotation gadget) and
    classify nodes by role (init_pivot, link, PZ, Xm).

    Returns (roles, node_to_cid) where node_to_cid maps every node ID to its
    component index.  Used by solve() to look up participant blocks per node.
    """
    # Connected-component decomposition (undirected adjacency)
    seen: Set[str] = set()
    components: List[Set[str]] = []
    node_to_cid: Dict[str, int] = {}

    for start in dag.nodes:
        if start in seen:
            continue
        comp: Set[str] = set()
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in set(dag.pred.get(u, set())) | set(dag.succ.get(u, set())):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        cid = len(components)
        for nid in comp:
            node_to_cid[nid] = cid
        components.append(comp)

    roles: List[_CompRole] = []
    for cid, comp in enumerate(components):
        init_nids: List[str] = []
        link_nid: Optional[str] = None
        pz_nid: Optional[str] = None
        xm_nid: Optional[str] = None

        for nid in comp:
            kind = dag.nodes[nid].kind
            if kind == K_INIT_PIVOT:
                init_nids.append(nid)
            elif kind == K_INTERBLOCK_LINK:
                link_nid = nid
            elif kind == K_MEAS_PARITY_PZ:
                pz_nid = nid
            elif kind == K_MEAS_MAGIC_X:
                xm_nid = nid

        # Magic block comes from Xm node (first block it touches)
        magic_block: Optional[int] = None
        if xm_nid is not None:
            xm_blocks = dag.nodes[xm_nid].blocks
            magic_block = xm_blocks[0] if xm_blocks else None

        # Map each participant block to the EARLIEST init_pivot in its chain.
        # A block may have multiple sequential init_pivots (c0 → lc0 → c1 → lc1 → c2…).
        # The first in the chain always has global indegree 0; later ones have indegree > 0.
        # We must anchor the ownership interval at c0 (the chain root), otherwise the
        # window [c0.start, c1.start) is unconstrained and another component can claim
        # the same block during that gap.
        init_block_map: Dict[int, str] = {}
        for nid in init_nids:
            for b in dag.nodes[nid].blocks:
                current = init_block_map.get(b)
                if current is None or (
                    len(dag.pred.get(nid, set())) < len(dag.pred.get(current, set()))
                ):
                    init_block_map[b] = nid

        for nid in comp:
            if dag.nodes[nid].kind == K_LOCAL_COUPLE:
                for b in dag.nodes[nid].blocks:
                    if b not in init_block_map:
                        for pred in dag.pred.get(nid, set()):
                            if dag.nodes[pred].kind == K_INIT_PIVOT and b in dag.nodes[pred].blocks:
                                init_block_map[b] = pred
                                break

        roles.append(_CompRole(
            cid=cid,
            init_block_map=init_block_map,
            link_nid=link_nid,
            pz_nid=pz_nid,
            xm_nid=xm_nid,
            magic_block=magic_block,
            is_multiblock=link_nid is not None,
        ))

    return roles, node_to_cid


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class CPSATScheduler(BaseScheduler):
    name: str = "cp_sat"

    def solve(self, problem: SchedulingProblem) -> Schedule:
        dag = problem.dag
        hw = problem.hw

        time_limit: Optional[float] = problem.meta.get("cp_sat_time_limit", None)
        use_routes: bool = bool(problem.meta.get("cp_sat_route_alternatives", True))
        max_hops: int = int(problem.meta.get("cp_sat_max_hops", len(hw.blocks)))
        log_search: bool = bool(problem.meta.get("cp_sat_log", False))

        # ── Debug file setup (only when cp_sat_log=True) ─────────────────────
        layer_tag = problem.meta.get("layer_idx", "")
        suffix = f"_L{layer_tag:02d}" if isinstance(layer_tag, int) else (
            f"_{layer_tag}" if layer_tag else ""
        )
        if log_search:
            dbg_path = pathlib.Path(__file__).parent / f"cpsat_logs/cp_sat_debug{suffix}.txt"
            dbg_file = open(dbg_path, "w")
        else:
            dbg_file = None

        def dbg(msg: str) -> None:
            if dbg_file is not None:
                dbg_file.write(msg + "\n")

        # ── Phase B: Extract component roles ─────────────────────────────────
        comp_roles, node_to_cid = _extract_comp_roles(dag)
        link_nid_to_role: Dict[str, _CompRole] = {
            r.link_nid: r for r in comp_roles if r.link_nid
        }
        # participant blocks per component — used to suppress self-conflicting intervals
        cid_participant_blocks: Dict[int, Set[int]] = {
            r.cid: set(r.init_block_map.keys()) for r in comp_roles
        }
        n_multiblock = sum(1 for r in comp_roles if r.is_multiblock)
        dbg(f"[INIT] nodes={len(dag.nodes)}  components={len(comp_roles)}"
            f"  multiblock={n_multiblock}")

        # ── Node durations ───────────────────────────────────────────────────
        node_dur: Dict[str, int] = {}
        for nid in dag.nodes:
            if nid in problem.duration_override:
                node_dur[nid] = int(problem.duration_override[nid])
            else:
                d = getattr(dag.nodes[nid], "duration", 1)
                node_dur[nid] = int(d) if d is not None else 1

        # ── Horizon (loose upper bound on makespan) ──────────────────────────
        horizon = sum(node_dur.values()) + 1

        # ── Build CP-SAT model ───────────────────────────────────────────────
        model = cp_model.CpModel()

        start_vars: Dict[str, cp_model.IntVar] = {}
        end_vars: Dict[str, cp_model.IntVar] = {}

        # ── Step 1: create start/end variables ──────────────────────────────
        for nid, node in dag.nodes.items():
            dur = node_dur[nid]
            s = model.new_int_var(0, horizon, f"s_{nid}")
            e = model.new_int_var(0, horizon, f"e_{nid}")
            model.add(e == s + dur)
            start_vars[nid] = s
            end_vars[nid] = e

        # ── Step 2: precedence constraints ───────────────────────────────────
        for u in dag.nodes:
            for v in dag.succ.get(u, set()):
                model.add(end_vars[u] <= start_vars[v])

        # ── Phase D: resource tracking — (IntervalVar, demand) tuples ────────
        # Using add_cumulative instead of add_no_overlap to support capacity > 1.
        # All current capacities are 1 so behaviour is identical to NoOverlap.
        block_ivals: Dict[int, List[Tuple[cp_model.IntervalVar, int]]] = {}
        coupler_ivals: Dict[str, List[Tuple[cp_model.IntervalVar, int]]] = {}

        # Saved per link-node for ownership interval construction (Steps 3b/3c)
        link_sel_vars: Dict[str, List[cp_model.BoolVar]] = {}
        link_route_opts: Dict[str, List[_RouteOption]] = {}
        link_opt_start_vars: Dict[str, List[cp_model.IntVar]] = {}

        # ── Step 3: hardware resource constraints (physical operations) ──────
        #
        # Block-registration strategy to avoid self-conflicts:
        #   - Participant blocks are covered by Group 1 ownership intervals.
        #   - Intermediate route blocks are covered by Group 2 per-hop intervals.
        #   - Node execution intervals therefore register COUPLERS only for
        #     non-link nodes, and NOTHING for link nodes (when use_routes=True).
        #   - For link nodes with use_routes=False (no per-hop), we register
        #     intermediate blocks (non-participant) via the standard interval.
        for nid, node in dag.nodes.items():
            dur = node_dur[nid]
            if dur == 0:
                continue  # zero-duration nodes hold no hardware resources

            is_link = (getattr(node, "kind", "") == K_INTERBLOCK_LINK)
            p_blocks = cid_participant_blocks[node_to_cid[nid]]

            if use_routes and is_link:
                # Route-alternative encoding: create variables only.
                # Block and coupler resource constraints are handled entirely
                # by Group 2 per-hop intervals in Step 3c — do NOT register here.
                options = _enumerate_route_options(node, hw, max_hops)

                if len(options) == 1:
                    opt = options[0]
                    model.add(end_vars[nid] == start_vars[nid] + opt.duration)
                    link_sel_vars[nid] = []
                    link_route_opts[nid] = [opt]
                    link_opt_start_vars[nid] = [start_vars[nid]]
                else:
                    sel_vars = [
                        model.new_bool_var(f"sel_{nid}_r{i}")
                        for i in range(len(options))
                    ]
                    model.add_exactly_one(sel_vars)

                    route_opt_starts: List[cp_model.IntVar] = []
                    for i, (opt, sel) in enumerate(zip(options, sel_vars)):
                        opt_s = model.new_int_var(0, horizon, f"s_{nid}_r{i}")
                        opt_e = model.new_int_var(0, horizon, f"e_{nid}_r{i}")
                        model.add(start_vars[nid] == opt_s).only_enforce_if(sel)
                        model.add(end_vars[nid] == opt_e).only_enforce_if(sel)
                        model.add(opt_e == opt_s + opt.duration)
                        route_opt_starts.append(opt_s)

                    link_sel_vars[nid] = sel_vars
                    link_route_opts[nid] = options
                    link_opt_start_vars[nid] = route_opt_starts
            else:
                # Standard interval for non-link nodes or link with routes disabled.
                # BLOCKS: skip participant blocks (covered by Group 1).
                #         For use_routes=False link nodes, keep intermediate blocks.
                # COUPLERS: always register (not covered by ownership intervals).
                raw_blocks = list(getattr(node, "blocks", []) or [])
                raw_couplers = list(getattr(node, "couplers", []) or [])

                # Determine which blocks to register
                if is_link:
                    # use_routes=False: keep only intermediate (non-participant) blocks
                    reg_blocks: Set[int] = {b for b in raw_blocks if b not in p_blocks}
                else:
                    # Non-link: all touched blocks are participant → skip all
                    reg_blocks = set()

                # Incident-coupler block expansion (IncidentCouplerBlocksLocalOpsPolicy):
                # add blocks incident to couplers, but only if not participant-owned
                for c in raw_couplers:
                    spec = hw.couplers.get(c)
                    if spec:
                        for b in (spec.u, spec.v):
                            if b not in p_blocks:
                                reg_blocks.add(b)

                if reg_blocks or raw_couplers:
                    iv = model.new_interval_var(
                        start_vars[nid], dur, end_vars[nid], f"iv_{nid}"
                    )
                    self._register_hw_interval(iv, tuple(reg_blocks), tuple(raw_couplers),
                                               block_ivals, coupler_ivals)

        # ── Step 3b: Ownership intervals — Group 1 (Preparation lock) ────────
        #
        # Each participant block b is locked for its owning component from
        # its first init_pivot until the measurement that releases it:
        #   - magic block or single-block rotation → lock until Xm.end
        #   - non-magic participant block           → lock until PZ.end
        #
        # These are mandatory intervals: a different component scheduling an
        # init_pivot on the same block within this window will conflict.
        dbg(f"\n[GROUP1] Preparation lock intervals  (component count={len(comp_roles)}):")
        for role in comp_roles:
            pz_end_var = end_vars.get(role.pz_nid) if role.pz_nid else None
            xm_end_var = end_vars.get(role.xm_nid) if role.xm_nid else None

            for b, init_nid in role.init_block_map.items():
                is_magic_or_single = (b == role.magic_block) or (not role.is_multiblock)
                if is_magic_or_single:
                    # Prefer Xm end; fall back to PZ end for single-block rotations
                    # that have no meas_magic_X node (only init_pivot → PZ → frame_update).
                    lock_end_var = xm_end_var if xm_end_var is not None else pz_end_var
                else:
                    lock_end_var = pz_end_var

                if lock_end_var is None:
                    continue  # component has no measurement node at all — skip

                lock_start_var = start_vars[init_nid]
                size_var = model.new_int_var(0, horizon, f"prep_sz_{role.cid}_{b}")
                model.add(size_var == lock_end_var - lock_start_var)
                own_iv = model.new_interval_var(
                    lock_start_var, size_var, lock_end_var,
                    f"prep_{role.cid}_{b}"
                )
                block_ivals.setdefault(b, []).append((own_iv, 1))

                lock_type = "Xm" if lock_end_var is xm_end_var else "PZ"
                dbg(f"  cid={role.cid}  block={b}  init={init_nid}"
                    f"  end={lock_type}  multiblock={role.is_multiblock}")

        # ── Step 3c: Route-lock intervals — Group 2 (per-hop, couplers only) ───
        #
        # Intermediate route blocks: per-hop interval [link_s+h, link_s+h+1).
        # Participant blocks (including magic): SKIPPED — Group 1 ownership already
        #   covers them for their full window; adding Group 2 would self-conflict.
        # Couplers: per-hop interval alongside their blocks.
        #
        # Fallback (paths empty): coarse [link_s, link_e) for intermediate blocks
        # and couplers, since no hop-level path info is available.
        dbg(f"\n[GROUP2] Route-lock intervals  (multiblock links={n_multiblock}):")
        for link_nid, role in link_nid_to_role.items():
            if link_nid not in link_opt_start_vars:
                continue  # link node had dur=0 — skipped in Step 3

            p_blocks = cid_participant_blocks[role.cid]
            opts = link_route_opts[link_nid]
            sels = link_sel_vars[link_nid]        # [] when single route
            opt_starts = link_opt_start_vars[link_nid]

            for i, (opt, opt_s) in enumerate(zip(opts, opt_starts)):
                sel: Optional[cp_model.BoolVar] = sels[i] if sels else None

                if not opt.paths:
                    # ── Fallback: coarse lock on intermediate blocks + couplers ──
                    # Used when no path metadata is available (paths=()).
                    link_e = end_vars[link_nid]
                    link_dur = int(opt.duration)
                    for b in opt.blocks:
                        if b in p_blocks:
                            continue  # participant: Group 1 covers it
                        if sel is not None:
                            fb_e = model.new_int_var(0, horizon, f"fbe_{link_nid}_r{i}_{b}")
                            fb_iv = model.new_optional_interval_var(
                                opt_s, link_dur, fb_e, sel, f"fbiv_{link_nid}_r{i}_{b}"
                            )
                            model.add(fb_e == opt_s + link_dur).only_enforce_if(sel)
                        else:
                            fb_iv = model.new_interval_var(
                                opt_s, link_dur, link_e, f"fbiv_{link_nid}_{b}"
                            )
                        block_ivals.setdefault(b, []).append((fb_iv, 1))
                        dbg(f"  [FALLBACK_BLOCK] link={link_nid}  r={i}  block={b}")

                    for c in opt.couplers:
                        if sel is not None:
                            fc_e = model.new_int_var(0, horizon, f"fce_{link_nid}_r{i}_{c}")
                            fc_iv = model.new_optional_interval_var(
                                opt_s, link_dur, fc_e, sel, f"fciv_{link_nid}_r{i}_{c}"
                            )
                            model.add(fc_e == opt_s + link_dur).only_enforce_if(sel)
                        else:
                            fc_iv = model.new_interval_var(
                                opt_s, link_dur, link_e, f"fciv_{link_nid}_{c}"
                            )
                        coupler_ivals.setdefault(c, []).append((fc_iv, 1))
                        dbg(f"  [FALLBACK_COUPLER] link={link_nid}  r={i}  coupler={c}")
                    continue

                # ── Per-hop: intermediate blocks + couplers ───────────────────
                # Deduplicate (h, u, v) across all paths in this route option
                hop_set: Set[Tuple[int, int, int]] = set()
                for path in opt.paths:
                    for h, (u, v) in enumerate(zip(path, path[1:])):
                        hop_set.add((h, u, v))

                # Track (h, block) pairs already registered for this route option
                # to prevent double-counting when two hops at the same time slot
                # both touch the same intermediate block (e.g. (h, 8→5) and (h, 5→2)
                # both contribute block 5 at hop h, causing a self-conflict).
                registered_hb: Set[Tuple[int, int]] = set()

                for h, u, v in sorted(hop_set):
                    cid_str = hw.coupler_id(u, v)
                    tag = f"{link_nid}_r{i}_h{h}_{u}_{v}"

                    hop_s = model.new_int_var(0, horizon, f"hs_{tag}")
                    hop_e = model.new_int_var(0, horizon, f"he_{tag}")

                    if sel is not None:
                        model.add(hop_s == opt_s + h).only_enforce_if(sel)
                        model.add(hop_e == opt_s + h + 1).only_enforce_if(sel)
                        hop_iv = model.new_optional_interval_var(
                            hop_s, 1, hop_e, sel, f"hiv_{tag}"
                        )
                    else:
                        model.add(hop_s == opt_s + h)
                        model.add(hop_e == opt_s + h + 1)
                        hop_iv = model.new_interval_var(hop_s, 1, hop_e, f"hiv_{tag}")

                    # Register only non-participant (intermediate) blocks.
                    # Participant blocks (including endpoint source/dest) are
                    # covered by Group 1 ownership intervals — adding them here
                    # would create self-conflicts within the same component.
                    # Additionally, skip (h, block) pairs already registered for
                    # this route option: multiple paths in the same combo may
                    # both touch the same intermediate block at the same hop
                    # index (e.g. paths [7,8,5,2] and [8,5,2] both produce
                    # block-5 intervals at h=1). Double-registration with
                    # demand=1 each against capacity=1 creates a self-conflict
                    # that makes the route infeasible even when run alone.
                    if u not in p_blocks and (h, u) not in registered_hb:
                        block_ivals.setdefault(u, []).append((hop_iv, 1))
                        registered_hb.add((h, u))
                    if v not in p_blocks and (h, v) not in registered_hb:
                        block_ivals.setdefault(v, []).append((hop_iv, 1))
                        registered_hb.add((h, v))
                    if cid_str:
                        coupler_ivals.setdefault(cid_str, []).append((hop_iv, 1))

                    dbg(f"  [HOP]  link={link_nid}  r={i}  h={h}"
                        f"  ({u}→{v})  coupler={cid_str}"
                        f"  optional={sel is not None}")

        # ── Phase D: add_cumulative for each block and coupler ────────────────
        # Capacity defaults to 1 (current hardware); supports capacity > 1 when needed.
        for b, ivals in block_ivals.items():
            if len(ivals) > 1:
                cap = hw.port_capacity.get(b, 1)
                model.add_cumulative(
                    [iv for iv, _ in ivals],
                    [d for _, d in ivals],
                    cap,
                )
        for c, ivals in coupler_ivals.items():
            if len(ivals) > 1:
                cap = hw.couplers[c].capacity
                model.add_cumulative(
                    [iv for iv, _ in ivals],
                    [d for _, d in ivals],
                    cap,
                )

        # ── Step 4: Objective ─────────────────────────────────────────────────
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, list(end_vars.values()))

        # Primary objective: minimize makespan.
        # Secondary soft objective (ACTIVE): minimize total link→PZ delay.
        #   Weighted combination ensures any 1-step makespan reduction
        #   outweighs the maximum achievable link→PZ delay saving.
        #
        # To activate additional secondary objectives, uncomment and combine:
        #   PZ→Xm delay:       start_vars[role.xm_nid] - end_vars[role.pz_nid]
        #   Component span:    end_vars[role.xm_nid] - start_vars[first_init_nid]
        #   Route length:      sum opt.duration for chosen route per link node
        link_to_pz_delays: List[cp_model.IntVar] = []
        for role in comp_roles:
            if role.link_nid and role.pz_nid:
                dv = model.new_int_var(0, horizon, f"lp_delay_{role.cid}")
                model.add(dv == start_vars[role.pz_nid] - end_vars[role.link_nid])
                link_to_pz_delays.append(dv)

        if link_to_pz_delays:
            total_lp = model.new_int_var(
                0, horizon * len(link_to_pz_delays), "total_lp_delay"
            )
            model.add(total_lp == sum(link_to_pz_delays))
            # BIG_M: ensures any 1-step makespan improvement dominates full lp_delay savings
            BIG_M = len(link_to_pz_delays) * horizon + 1
            model.minimize(makespan * BIG_M + total_lp)
            dbg(f"\n[OBJ] weighted: makespan*{BIG_M} + total_lp_delay"
                f"  (link→PZ pairs={len(link_to_pz_delays)})")
        else:
            model.minimize(makespan)
            dbg(f"\n[OBJ] minimize makespan only (no link→PZ pairs found)")

        # ── Solve ─────────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = log_search
        solver.parameters.log_to_stdout = log_search  # suppress CpSolverResponse summary
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = float(time_limit)

        status = solver.solve(model)

        status_name = solver.status_name(status)
        solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        cp_makespan = solver.value(makespan) if solved else -1
        cp_obj = int(solver.objective_value) if solved else -1
        cp_obj_lb = int(solver.best_objective_bound) if solved else -1

        dbg(f"\n[SOLVE] status={status_name}  makespan={cp_makespan}"
            f"  obj={cp_obj}  obj_lb={cp_obj_lb}")

        if not solved:
            if dbg_file is not None: dbg_file.close()
            raise RuntimeError(
                f"CP-SAT could not find a feasible schedule (status={status_name}). "
                "Check hardware constraints or increase time limit."
            )

        # ── Extract schedule ──────────────────────────────────────────────────
        entries: Dict[str, ScheduleEntry] = {}
        node_to_time: Dict[str, int] = {}

        for nid in dag.nodes:
            s_val = solver.value(start_vars[nid])
            e_val = solver.value(end_vars[nid])
            entries[nid] = ScheduleEntry(nid=nid, start=s_val, end=e_val)
            node_to_time[nid] = s_val

        starts_grouped: Dict[int, List[str]] = {}
        for nid, e in entries.items():
            starts_grouped.setdefault(e.start, []).append(nid)

        steps: List[ScheduleStep] = [
            ScheduleStep(
                t=tt,
                nodes=sorted(starts_grouped[tt]),
                meta={"algo": self.name},
            )
            for tt in sorted(starts_grouped)
        ]

        # ── Write per-step schedule to debug file ─────────────────────────────
        dbg(f"\n[SCHEDULE] makespan={cp_makespan}  steps={len(steps)}")
        for step in steps:
            node_details = []
            for nid in step.nodes:
                e = entries[nid]
                kind = getattr(dag.nodes[nid], "kind", "?")
                node_details.append(f"{nid}({kind},{e.start}-{e.end})")
            dbg(f"  t={step.t:4d}  [{', '.join(node_details)}]")
        if dbg_file is not None:
            dbg_file.close()

        return Schedule(
            steps=steps,
            node_to_time=node_to_time,
            meta={
                "scheduler": self.name,
                "cp_sat_status": status_name,
                "cp_sat_makespan": cp_makespan,
                "cp_sat_obj": cp_obj,
                "cp_sat_obj_lb": cp_obj_lb,
                "cp_sat_time_limit": time_limit,
                "cp_sat_route_alternatives": use_routes,
                "cp_sat_max_hops": max_hops,
                "entries": {
                    nid: {"start": e.start, "end": e.end}
                    for nid, e in entries.items()
                },
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _register_hw_interval(
        iv: cp_model.IntervalVar,
        blocks: Any,
        couplers: Any,
        block_ivals: Dict[int, List],
        coupler_ivals: Dict[str, List],
    ) -> None:
        """Register a hardware (physical operation) interval with demand=1."""
        for b in blocks:
            block_ivals.setdefault(b, []).append((iv, 1))
        for c in couplers:
            coupler_ivals.setdefault(c, []).append((iv, 1))
