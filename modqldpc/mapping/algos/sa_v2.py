from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..base import BaseMapper
from ..model import BlockId, HardwareGraph, LocalId, LogicalId
from ..types import MappingConfig, MappingPlan, MappingProblem
from .sa_mapping import _blocks_touched, _mst_len, _random_move, _rotation_support, _undo_move


DEFAULT_SCORE_KWARGS_V2: Dict[str, float] = {
    "W_UNUSED_BLOCKS": 1e6,
    "W_OCC_RANGE": 1e4,
    "W_OCC_STD": 1e3,
    "W_MULTI_BLOCK": 2e5,
    "W_SPAN": 5e4,
    "W_MST": 5e2,
    "W_SPLIT": 10.0,
    "W_SUPPORT_PEAK": 1e2,
    "W_SUPPORT_RANGE": 20.0,
    "W_SUPPORT_STD": 0.0,
}


@dataclass
class ScoreBreakdownV2:
    total: float

    active_blocks: int
    unused_blocks: int

    occupancy_range: float
    occupancy_std: float

    support_peak: float
    support_range: float
    support_std: float

    num_multiblock: int
    span_total: float
    mean_blocks_touched: float
    max_blocks_touched: int

    mst_total: float
    mean_mst: float
    max_mst: float

    split_total: float
    mean_split: float
    max_split: float

    unused_block_pen: float
    occupancy_range_pen: float
    occupancy_std_pen: float
    multiblock_pen: float
    span_pen: float
    mst_pen: float
    split_pen: float
    support_peak_pen: float
    support_range_pen: float
    support_std_pen: float

    block_occupancies: Dict[BlockId, int] = field(default_factory=dict)
    block_support_loads: Dict[BlockId, int] = field(default_factory=dict)
    blocks_touched_per_rotation: List[int] = field(default_factory=list)
    mst_per_multiblock: List[float] = field(default_factory=list)
    split_per_multiblock: List[float] = field(default_factory=list)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _occupancy_counts(hw: HardwareGraph) -> Dict[BlockId, int]:
    counts: Dict[BlockId, int] = {b: 0 for b in hw.blocks}
    for b in hw.logical_to_block.values():
        counts[b] += 1
    return counts


def _support_split(
    rot,
    hw: HardwareGraph,
    blocks: List[BlockId],
    *,
    split_mode: str = "l2",
) -> float:
    per_block_counts: Dict[BlockId, int] = {b: 0 for b in blocks}
    for q in _rotation_support(rot):
        b = hw.logical_to_block.get(q)
        if b in per_block_counts:
            per_block_counts[b] += 1
    counts = list(per_block_counts.values())
    if not counts:
        return 0.0
    if split_mode == "l1":
        return float(max(counts) - min(counts))
    if split_mode == "pairwise":
        if len(counts) <= 1:
            return 0.0
        return float(
            max(
                abs(counts[i] - counts[j])
                for i in range(len(counts))
                for j in range(i + 1, len(counts))
            )
        )
    return _std([float(c) for c in counts])


def score_mapping_v2(
    rotations: list,
    hw: HardwareGraph,
    *,
    W_UNUSED_BLOCKS: float = DEFAULT_SCORE_KWARGS_V2["W_UNUSED_BLOCKS"],
    W_OCC_RANGE: float = DEFAULT_SCORE_KWARGS_V2["W_OCC_RANGE"],
    W_OCC_STD: float = DEFAULT_SCORE_KWARGS_V2["W_OCC_STD"],
    W_MULTI_BLOCK: float = DEFAULT_SCORE_KWARGS_V2["W_MULTI_BLOCK"],
    W_SPAN: float = DEFAULT_SCORE_KWARGS_V2["W_SPAN"],
    W_MST: float = DEFAULT_SCORE_KWARGS_V2["W_MST"],
    W_SPLIT: float = DEFAULT_SCORE_KWARGS_V2["W_SPLIT"],
    W_SUPPORT_PEAK: float = DEFAULT_SCORE_KWARGS_V2["W_SUPPORT_PEAK"],
    W_SUPPORT_RANGE: float = DEFAULT_SCORE_KWARGS_V2["W_SUPPORT_RANGE"],
    W_SUPPORT_STD: float = DEFAULT_SCORE_KWARGS_V2["W_SUPPORT_STD"],
    SPLIT_MODE: str = "l2",
) -> ScoreBreakdownV2:
    """
    V2 score for the mapper:
      1. Encourage using all available hardware blocks.
      2. Reduce how often rotations become multi-block and how far they spread.
      3. Prefer compact multi-block spans.
      4. Balance support traffic across blocks.

    This is still a weighted sum, but unlike v1 it explicitly scores occupancy
    over all hardware blocks rather than only over active blocks.
    """
    occupancies = _occupancy_counts(hw)
    support_loads: Dict[BlockId, int] = {b: 0 for b in hw.blocks}

    blocks_touched_per_rotation: List[int] = []
    mst_per_multiblock: List[float] = []
    split_per_multiblock: List[float] = []

    num_multiblock = 0
    span_total = 0.0
    mst_total = 0.0
    split_total = 0.0

    for rot in rotations:
        touched = sorted(_blocks_touched(rot, hw))
        k = len(touched)
        blocks_touched_per_rotation.append(k)
        for b in touched:
            support_loads[b] += 1

        if k > 1:
            num_multiblock += 1
            span_total += float(k - 1)
            mst_val = float(_mst_len(set(touched), hw))
            split_val = _support_split(rot, hw, touched, split_mode=SPLIT_MODE)
            mst_total += mst_val
            split_total += split_val
            mst_per_multiblock.append(mst_val)
            split_per_multiblock.append(split_val)

    occ_values = [float(v) for _, v in sorted(occupancies.items())]
    support_values = [float(v) for _, v in sorted(support_loads.items())]

    active_blocks = sum(1 for v in occupancies.values() if v > 0)
    unused_blocks = len(occupancies) - active_blocks

    occupancy_range = float(max(occ_values) - min(occ_values)) if occ_values else 0.0
    occupancy_std = _std(occ_values)

    support_peak = float(max(support_values)) if support_values else 0.0
    support_range = float(max(support_values) - min(support_values)) if support_values else 0.0
    support_std = _std(support_values)

    mean_blocks_touched = _mean([float(k) for k in blocks_touched_per_rotation])
    max_blocks_touched = max(blocks_touched_per_rotation) if blocks_touched_per_rotation else 0

    mean_mst = _mean(mst_per_multiblock)
    max_mst = max(mst_per_multiblock) if mst_per_multiblock else 0.0

    mean_split = _mean(split_per_multiblock)
    max_split = max(split_per_multiblock) if split_per_multiblock else 0.0

    unused_block_pen = W_UNUSED_BLOCKS * float(unused_blocks)
    occupancy_range_pen = W_OCC_RANGE * occupancy_range
    occupancy_std_pen = W_OCC_STD * occupancy_std
    multiblock_pen = W_MULTI_BLOCK * float(num_multiblock)
    span_pen = W_SPAN * span_total
    mst_pen = W_MST * mst_total
    split_pen = W_SPLIT * split_total
    support_peak_pen = W_SUPPORT_PEAK * support_peak
    support_range_pen = W_SUPPORT_RANGE * support_range
    support_std_pen = W_SUPPORT_STD * support_std

    total = (
        unused_block_pen
        + occupancy_range_pen
        + occupancy_std_pen
        + multiblock_pen
        + span_pen
        + mst_pen
        + split_pen
        + support_peak_pen
        + support_range_pen
        + support_std_pen
    )

    return ScoreBreakdownV2(
        total=total,
        active_blocks=active_blocks,
        unused_blocks=unused_blocks,
        occupancy_range=occupancy_range,
        occupancy_std=occupancy_std,
        support_peak=support_peak,
        support_range=support_range,
        support_std=support_std,
        num_multiblock=num_multiblock,
        span_total=span_total,
        mean_blocks_touched=mean_blocks_touched,
        max_blocks_touched=max_blocks_touched,
        mst_total=mst_total,
        mean_mst=mean_mst,
        max_mst=max_mst,
        split_total=split_total,
        mean_split=mean_split,
        max_split=max_split,
        unused_block_pen=unused_block_pen,
        occupancy_range_pen=occupancy_range_pen,
        occupancy_std_pen=occupancy_std_pen,
        multiblock_pen=multiblock_pen,
        span_pen=span_pen,
        mst_pen=mst_pen,
        split_pen=split_pen,
        support_peak_pen=support_peak_pen,
        support_range_pen=support_range_pen,
        support_std_pen=support_std_pen,
        block_occupancies=occupancies,
        block_support_loads=support_loads,
        blocks_touched_per_rotation=blocks_touched_per_rotation,
        mst_per_multiblock=mst_per_multiblock,
        split_per_multiblock=split_per_multiblock,
    )


def _anneal_core_v2(
    rotations: list,
    hw: HardwareGraph,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    report_every: int = 2_000,
    n_check: int = 0,
) -> Tuple[
    ScoreBreakdownV2,
    Dict[LogicalId, Tuple[BlockId, LocalId]],
    List[Dict[str, float]],
]:
    rng = random.Random(seed)
    score_kwargs = score_kwargs or {}

    def _snapshot() -> Dict[LogicalId, Tuple[BlockId, LocalId]]:
        return {
            q: (hw.logical_to_block[q], hw.logical_to_local[q])
            for q in hw.logical_to_block
        }

    best_map = _snapshot()
    cur = score_mapping_v2(rotations, hw, **score_kwargs)
    best = cur

    checkpoints: List[Dict[str, float]] = []
    check_at = set()
    if n_check and n_check > 1:
        check_at = {int(i * (steps - 1) / (n_check - 1)) for i in range(n_check)}

    n_noop = 0
    n_accept = 0
    n_reject = 0

    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = t0 * ((t_end / t0) ** frac)

        if it in check_at:
            checkpoints.append({
                "pct": 100.0 * frac,
                "temperature": T,
                "best_total": best.total,
                "current_total": cur.total,
                "best_active_blocks": float(best.active_blocks),
                "best_unused_blocks": float(best.unused_blocks),
                "best_num_multiblock": float(best.num_multiblock),
                "best_span_total": best.span_total,
                "best_mean_mst": best.mean_mst,
                "best_support_peak": best.support_peak,
                "best_support_range": best.support_range,
                "accept": float(n_accept),
                "reject": float(n_reject),
                "noop": float(n_noop),
            })
            n_noop = n_accept = n_reject = 0

        move = _random_move(hw, rng)
        if move[0] == "noop":
            n_noop += 1
            continue

        nxt = score_mapping_v2(rotations, hw, **score_kwargs)
        delta = nxt.total - cur.total
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(1e-12, T))

        if accept:
            n_accept += 1
            cur = nxt
            if cur.total < best.total:
                best = cur
                best_map = _snapshot()
        else:
            n_reject += 1
            _undo_move(hw, move)

        if verbose and report_every and ((it + 1) % report_every == 0):
            print(
                f"[SA-v2] it={it + 1:6d}  T={T:.4f}  "
                f"cur={cur.total:.1f}  best={best.total:.1f}  "
                f"(unused={cur.unused_blocks} multi={cur.num_multiblock} "
                f"span={cur.span_total:.1f} mst={cur.mean_mst:.2f} "
                f"peak={cur.support_peak:.1f})  "
                f"accept={n_accept}  reject={n_reject}  noop={n_noop}"
            )
            n_noop = n_accept = n_reject = 0

    hw.logical_to_block.clear()
    hw.logical_to_local.clear()
    for q, (b, l) in best_map.items():
        hw.logical_to_block[q] = b
        hw.logical_to_local[q] = l

    return best, best_map, checkpoints


def anneal_with_checkpoints_v2(
    rotations: list,
    hw: HardwareGraph,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Optional[Dict[str, float]] = None,
    n_check: int = 11,
) -> Tuple[
    ScoreBreakdownV2,
    List[Dict[str, float]],
    Dict[LogicalId, Tuple[BlockId, LocalId]],
]:
    best, best_map, checkpoints = _anneal_core_v2(
        rotations,
        hw,
        steps=steps,
        t0=t0,
        t_end=t_end,
        seed=seed,
        score_kwargs=score_kwargs,
        n_check=n_check,
    )
    return best, checkpoints, best_map


def _anneal_v2(
    rotations: list,
    hw: HardwareGraph,
    *,
    steps: int,
    t0: float,
    t_end: float,
    seed: int,
    score_kwargs: Optional[Dict[str, float]] = None,
    verbose: bool = False,
    report_every: int = 2_000,
) -> Tuple[ScoreBreakdownV2, Dict[LogicalId, Tuple[BlockId, LocalId]]]:
    best, best_map, _ = _anneal_core_v2(
        rotations,
        hw,
        steps=steps,
        t0=t0,
        t_end=t_end,
        seed=seed,
        score_kwargs=score_kwargs,
        verbose=verbose,
        report_every=report_every,
    )
    return best, best_map


@dataclass(frozen=True)
class SimulatedAnnealingV2Mapper(BaseMapper):
    """
    V2 SA mapper with explicit block-utilization and locality-aware scoring.
    """
    name: str = "sa_v2"
    init_mapper_name: str = "auto_round_robin_mapping"

    def solve(
        self,
        problem: MappingProblem,
        hw: HardwareGraph,
        cfg: MappingConfig,
        meta: Optional[dict] = None,
    ) -> MappingPlan:
        meta = meta or {}
        rotations = meta.get("rotations")
        if rotations is None:
            raise ValueError(
                "SimulatedAnnealingV2Mapper requires meta['rotations'] "
                "(list of PauliRotation from GoSCConverter.program.rotations)."
            )

        verbose = bool(meta.get("verbose", False))
        report_every = int(meta.get("report_every", 2_000))
        score_kwargs = dict(DEFAULT_SCORE_KWARGS_V2)
        score_kwargs.update(meta.get("score_kwargs", {}) or {})

        from ..factory import get_mapper

        init_mapper = get_mapper(self.init_mapper_name)
        init_mapper.solve(problem, hw, cfg)

        best_score, _ = _anneal_v2(
            rotations,
            hw,
            steps=cfg.sa_steps,
            t0=cfg.sa_t0,
            t_end=cfg.sa_tend,
            seed=cfg.seed,
            score_kwargs=score_kwargs,
            verbose=verbose,
            report_every=report_every,
        )

        out_b = dict(hw.logical_to_block)
        out_l = dict(hw.logical_to_local)
        return MappingPlan(
            out_b,
            out_l,
            meta={
                "mapper": self.name,
                "init_mapper": self.init_mapper_name,
                "sa_steps": cfg.sa_steps,
                "sa_t0": cfg.sa_t0,
                "sa_t_end": cfg.sa_tend,
                "score_kwargs": score_kwargs,
                "best_score_total": best_score.total,
                "active_blocks": best_score.active_blocks,
                "unused_blocks": best_score.unused_blocks,
                "occupancy_range": best_score.occupancy_range,
                "occupancy_std": best_score.occupancy_std,
                "support_peak": best_score.support_peak,
                "support_range": best_score.support_range,
                "support_std": best_score.support_std,
                "num_multiblock": best_score.num_multiblock,
                "span_total": best_score.span_total,
                "mean_blocks_touched": best_score.mean_blocks_touched,
                "max_blocks_touched": best_score.max_blocks_touched,
                "mst_total": best_score.mst_total,
                "mean_mst": best_score.mean_mst,
                "max_mst": best_score.max_mst,
                "split_total": best_score.split_total,
                "mean_split": best_score.mean_split,
                "max_split": best_score.max_split,
                "unused_block_pen": best_score.unused_block_pen,
                "occupancy_range_pen": best_score.occupancy_range_pen,
                "occupancy_std_pen": best_score.occupancy_std_pen,
                "multiblock_pen": best_score.multiblock_pen,
                "span_pen": best_score.span_pen,
                "mst_pen": best_score.mst_pen,
                "split_pen": best_score.split_pen,
                "support_peak_pen": best_score.support_peak_pen,
                "support_range_pen": best_score.support_range_pen,
                "support_std_pen": best_score.support_std_pen,
                "n_rotations": len(rotations),
            },
        )
