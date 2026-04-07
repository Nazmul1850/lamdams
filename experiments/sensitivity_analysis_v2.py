"""
experiments/sensitivity_analysis_v2.py

Separate evaluation path for the new SA-v2 mapper.

Goals
-----
1. Keep the existing implementation untouched.
2. Evaluate SA-v2 on gf2_16_mult with greedy depth for now.
3. Save enough raw metrics to support paper plots later without rerunning.

Phases
------
0  Baselines     - random / round-robin / SA-v1 / SA-v2(default), greedy depth.
1  Weight sweep  - sweep SA-v2 objective weights and compare preset hierarchies.
2  Hyperparams   - sweep SA-v2 annealing params for one chosen preset.
3  CPSAT replay  - rerun saved top SA-v2 configs with CP-SAT.

Outputs
-------
results/sensitivity_v2/
  raw/       one JSON per run with mapping, score, depth, and trajectory data
  summary/   per-phase JSON summaries
  figures/   reviewer-facing PNGs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.lowering.keys import KeyNamer
from modqldpc.lowering.lower_layer import lower_one_layer
from modqldpc.lowering.policy import (
    ChooseMagicBlockMinId,
    HeuristicRepeatNativePolicy,
    LoweringPolicies,
    ShortestPathGatherRouting,
)
from modqldpc.mapping.algos.sa_v2 import (
    DEFAULT_SCORE_KWARGS_V2,
    anneal_with_checkpoints_v2,
    score_mapping_v2,
)
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem, get_mapper
from modqldpc.pipeline.run_one import make_gross_actual_cost_fn
from modqldpc.runtime.frame_policy import FrameState, FrameUpdatePolicy
from modqldpc.runtime.layer_exec import LayerExecutor
from modqldpc.runtime.outcomes import RandomOutcomeModel
from modqldpc.scheduling.factory import get_scheduler
from modqldpc.scheduling.types import SchedulingProblem


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
RESULTS_ROOT = os.path.join(_ROOT, "results", "sensitivity_v2")
RAW_DIR = os.path.join(RESULTS_ROOT, "raw")
SUMMARY_DIR = os.path.join(RESULTS_ROOT, "summary")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")

GF2_CIRCUIT = "gf2_16_mult"
N_DATA = 11

BASELINES = {
    "random_greedy": 24_893,
    "random_cpsat": 18_936,
    "sa_v1_greedy": 23_689,
    "sa_v1_cpsat": 19_352,
}

SA_V2_EXPLORE = {
    "steps": 25_000,
    "t0": 1e5,
    "t_end": 0.05,
}

WEIGHT_SWEEPS: Dict[str, List[float]] = {
    "W_UNUSED_BLOCKS": [0.0, 1e4, 1e5, 1e6, 1e7],
    "W_MULTI_BLOCK": [0.0, 1e4, 1e5, 2e5, 1e6],
    "W_SPAN": [0.0, 1e3, 1e4, 5e4, 2e5],
    "W_OCC_RANGE": [0.0, 1e2, 1e3, 1e4, 1e5],
    "W_MST": [0.0, 10.0, 1e2, 1e3, 1e4],
    "W_SUPPORT_PEAK": [0.0, 10.0, 1e2, 1e3, 1e4],
}

HYPER_SWEEPS = {
    # Local sweeps centered around the promising neighborhood.
    "steps": [20_000, 22_500, 25_000, 27_500, 30_000, 35_000, 40_000],
    "t0": [7.5e4, 9e4, 1e5, 1.1e5, 1.25e5, 1.5e5, 2e5],
    "t_end": [0.02, 0.035, 0.05, 0.065, 0.08, 0.1],
}

V2_PRESETS: Dict[str, Dict[str, float]] = {
    "default_v2": dict(DEFAULT_SCORE_KWARGS_V2),
    "utilization_first": {
        **DEFAULT_SCORE_KWARGS_V2,
        "W_UNUSED_BLOCKS": 1e7,
        "W_OCC_RANGE": 1e5,
        "W_MULTI_BLOCK": 1e5,
        "W_SPAN": 1e4,
    },
    "interblock_first": {
        **DEFAULT_SCORE_KWARGS_V2,
        "W_UNUSED_BLOCKS": 5e5,
        "W_MULTI_BLOCK": 1e6,
        "W_SPAN": 2e5,
        "W_OCC_RANGE": 1e4,
    },
    "locality_first": {
        **DEFAULT_SCORE_KWARGS_V2,
        "W_MST": 1e4,
        "W_MULTI_BLOCK": 5e5,
        "W_SPAN": 1e5,
    },
    "support_first": {
        **DEFAULT_SCORE_KWARGS_V2,
        "W_SUPPORT_PEAK": 1e4,
        "W_SUPPORT_RANGE": 1e3,
        "W_MULTI_BLOCK": 1e5,
    },
    "balanced_tradeoff": {
        "W_UNUSED_BLOCKS": 1e6,
        "W_OCC_RANGE": 5e4,
        "W_OCC_STD": 5e3,
        "W_MULTI_BLOCK": 5e5,
        "W_SPAN": 1e5,
        "W_MST": 1e3,
        "W_SPLIT": 50.0,
        "W_SUPPORT_PEAK": 5e2,
        "W_SUPPORT_RANGE": 50.0,
        "W_SUPPORT_STD": 0.0,
    },
}


def _ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def _slug(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("=", "-")
        .replace(".", "p")
    )


def _sci(v: float) -> str:
    return f"{v:.1e}".replace("+", "p").replace("-", "m")


def _result_path(
    phase: str,
    mapper_name: str,
    scheduler: str,
    seed: int,
    *,
    label: str,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
) -> str:
    _ensure_dirs()
    fname = (
        f"p{phase}_{GF2_CIRCUIT}_{mapper_name}_{scheduler}"
        f"_steps={sa_steps}_t0={_sci(sa_t0)}_tend={_sci(sa_tend)}"
        f"_seed{seed}_{_slug(label)}.json"
    )
    return os.path.join(RAW_DIR, fname)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_pbc(circuit_name: str = GF2_CIRCUIT):
    path = os.path.join(PBC_DIR, f"{circuit_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PBC not found for '{circuit_name}' at {path}\n"
            "Run: python experiments/run_experiment.py --build-pbc"
        )
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(path)
    first_rot = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    rotations = list(conv.program.rotations)
    return n_logicals, rotations, conv


def _plan_from_hw(hw) -> MappingPlan:
    return MappingPlan(
        logical_to_block=dict(hw.logical_to_block),
        logical_to_local=dict(hw.logical_to_local),
        meta={},
    )


def _count_inter_block(rotations: list, plan: MappingPlan, *, t_only: bool = False) -> int:
    count = 0
    for rot in rotations:
        if t_only and not (abs(rot.angle) < math.pi / 2 - 1e-9):
            continue
        axis = rot.axis.lstrip("+-")
        n = len(axis)
        blocks = set()
        for qi in range(n):
            if axis[n - 1 - qi] != "I":
                b = plan.logical_to_block.get(qi)
                if b is not None:
                    blocks.add(b)
        if len(blocks) >= 2:
            count += 1
    return count


def _score_to_json(score) -> Dict[str, Any]:
    return {
        "total": score.total,
        "active_blocks": score.active_blocks,
        "unused_blocks": score.unused_blocks,
        "occupancy_range": score.occupancy_range,
        "occupancy_std": score.occupancy_std,
        "support_peak": score.support_peak,
        "support_range": score.support_range,
        "support_std": score.support_std,
        "num_multiblock": score.num_multiblock,
        "span_total": score.span_total,
        "mean_blocks_touched": score.mean_blocks_touched,
        "max_blocks_touched": score.max_blocks_touched,
        "mst_total": score.mst_total,
        "mean_mst": score.mean_mst,
        "max_mst": score.max_mst,
        "split_total": score.split_total,
        "mean_split": score.mean_split,
        "max_split": score.max_split,
        "weighted": {
            "unused_block_pen": score.unused_block_pen,
            "occupancy_range_pen": score.occupancy_range_pen,
            "occupancy_std_pen": score.occupancy_std_pen,
            "multiblock_pen": score.multiblock_pen,
            "span_pen": score.span_pen,
            "mst_pen": score.mst_pen,
            "split_pen": score.split_pen,
            "support_peak_pen": score.support_peak_pen,
            "support_range_pen": score.support_range_pen,
            "support_std_pen": score.support_std_pen,
        },
        "block_occupancies": score.block_occupancies,
        "block_support_loads": score.block_support_loads,
        "blocks_touched_per_rotation": score.blocks_touched_per_rotation,
        "mst_per_multiblock": score.mst_per_multiblock,
        "split_per_multiblock": score.split_per_multiblock,
    }


def _compute_greedy_depth(
    conv,
    hw,
    plan: MappingPlan,
    *,
    seed: int,
    n_data: int,
    scheduler_name: str = "greedy_critical",
    cp_sat_time_limit: float = 120.0,
) -> tuple[int, List[int]]:
    cost_fn = make_gross_actual_cost_fn(plan, n_data=n_data)
    policies = LoweringPolicies(
        namer=KeyNamer(),
        magic=ChooseMagicBlockMinId(),
        routing=ShortestPathGatherRouting(),
        native=HeuristicRepeatNativePolicy(cost_fn=cost_fn),
    )

    from modqldpc.core.types import PauliRotation

    effective_rotations: Dict[int, PauliRotation] = {
        r.idx: r for r in conv.program.rotations
    }
    frame = FrameState()
    executor = LayerExecutor(
        outcome_model=RandomOutcomeModel(seed=seed),
        frame_policy=FrameUpdatePolicy(),
    )
    sched_obj = get_scheduler(scheduler_name)

    total_depth = 0
    layer_depths: List[int] = []
    for layer_id, layer in enumerate(conv.layers):
        res = lower_one_layer(
            layer_idx=layer_id,
            rotations=effective_rotations,
            rotation_indices=layer,
            hw=hw,
            policies=policies,
        )
        S = sched_obj.solve(SchedulingProblem(
            dag=res.dag,
            hw=hw,
            seed=seed,
            policy_name="incident_coupler_blocks_local",
            meta={
                "start_time": 0,
                "layer_idx": layer_id,
                "tie_breaker": "duration",
                "cp_sat_time_limit": cp_sat_time_limit,
                "debug_decode": False,
                "safe_fill": True,
                "cp_sat_log": False,
            },
        ))

        next_idxs = conv.layers[layer_id + 1] if (layer_id + 1) in conv.layers else []
        rot_next = [effective_rotations[i] for i in next_idxs]
        ex = executor.execute_layer(
            layer=layer_id,
            dag=res.dag,
            schedule=S,
            frame_in=frame,
            next_layer_rotations=rot_next,
        )
        for r in ex.next_rotations_effective:
            effective_rotations[r.idx] = r
        frame = ex.frame_after
        total_depth += ex.depth
        layer_depths.append(ex.depth)

    return total_depth, layer_depths


def _run_mapping(
    mapper_name: str,
    *,
    phase: str,
    label: str,
    score_kwargs: Optional[Dict[str, float]] = None,
    seed: int = 42,
    sa_steps: int = SA_V2_EXPLORE["steps"],
    sa_t0: float = SA_V2_EXPLORE["t0"],
    sa_tend: float = SA_V2_EXPLORE["t_end"],
    force: bool = False,
    trajectory: bool = False,
    scheduler_name: str = "greedy_critical",
    cp_sat_time_limit: float = 120.0,
) -> Dict[str, Any]:
    rpath = _result_path(
        phase,
        mapper_name,
        scheduler_name,
        seed,
        label=label,
        sa_steps=sa_steps,
        sa_t0=sa_t0,
        sa_tend=sa_tend,
    )
    if not force and os.path.exists(rpath):
        with open(rpath, "r", encoding="utf-8") as f:
            return json.load(f)

    n_logicals, rotations, conv = _load_pbc(GF2_CIRCUIT)
    hw, hw_spec = make_hardware(
        n_logicals,
        topology="grid",
        sparse_pct=0.0,
        n_data=N_DATA,
        coupler_capacity=1,
    )

    t_start = time.perf_counter()
    map_cfg = MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend)
    plan: MappingPlan
    trace: List[Dict[str, float]] = []

    if mapper_name == "random":
        plan = get_mapper("pure_random").solve(
            MappingProblem(n_logicals=n_logicals),
            hw,
            MappingConfig(seed=seed),
        )
    elif mapper_name == "round_robin":
        plan = get_mapper("auto_round_robin_mapping").solve(
            MappingProblem(n_logicals=n_logicals),
            hw,
            MappingConfig(seed=seed),
        )
    elif mapper_name == "sa_v1":
        plan = get_mapper("simulated_annealing").solve(
            MappingProblem(n_logicals=n_logicals),
            hw,
            map_cfg,
            {"rotations": rotations, "verbose": False},
        )
    elif mapper_name == "sa_v2":
        start = get_mapper("auto_round_robin_mapping")
        start.solve(MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed))
        merged_score_kwargs = dict(DEFAULT_SCORE_KWARGS_V2)
        merged_score_kwargs.update(score_kwargs or {})
        if trajectory:
            _, trace, _ = anneal_with_checkpoints_v2(
                rotations,
                hw,
                steps=sa_steps,
                t0=sa_t0,
                t_end=sa_tend,
                seed=seed,
                score_kwargs=merged_score_kwargs,
                n_check=11,
            )
            plan = MappingPlan(
                logical_to_block=dict(hw.logical_to_block),
                logical_to_local=dict(hw.logical_to_local),
                meta={"mapper": "sa_v2", "score_kwargs": merged_score_kwargs},
            )
        else:
            plan = get_mapper("sa_v2").solve(
                MappingProblem(n_logicals=n_logicals),
                hw,
                map_cfg,
                {
                    "rotations": rotations,
                    "verbose": False,
                    "score_kwargs": merged_score_kwargs,
                },
            )
    else:
        raise ValueError(f"Unknown mapper_name: {mapper_name!r}")

    map_time = time.perf_counter() - t_start

    effective_score_kwargs = dict(DEFAULT_SCORE_KWARGS_V2)
    effective_score_kwargs.update(score_kwargs or {})
    score = score_mapping_v2(rotations, hw, **effective_score_kwargs)

    inter_block_all = _count_inter_block(rotations, plan, t_only=False)
    inter_block_t_only = _count_inter_block(rotations, plan, t_only=True)

    t_depth = time.perf_counter()
    total_depth, layer_depths = _compute_greedy_depth(
        conv,
        hw,
        plan,
        seed=seed,
        n_data=N_DATA,
        scheduler_name=scheduler_name,
        cp_sat_time_limit=cp_sat_time_limit,
    )
    depth_time = time.perf_counter() - t_depth

    record = {
        "phase": phase,
        "label": label,
        "circuit": GF2_CIRCUIT,
        "mapper_name": mapper_name,
        "scheduler": scheduler_name,
        "seed": seed,
        "hardware": {
            "label": hw_spec.label(),
            "n_blocks": len(hw.blocks),
            "n_logicals": n_logicals,
            "n_data": N_DATA,
        },
        "sa_params": {
            "steps": sa_steps,
            "t0": sa_t0,
            "t_end": sa_tend,
        },
        "cp_sat_time_limit": cp_sat_time_limit,
        "score_kwargs_v2": effective_score_kwargs,
        "mapping_meta": plan.meta,
        "mapping_metrics_v2": _score_to_json(score),
        "inter_block_all": inter_block_all,
        "inter_block_t_only": inter_block_t_only,
        "logical_depth": total_depth,
        "layer_depths": layer_depths,
        "timing_sec": {
            "map": round(map_time, 3),
            "greedy_depth": round(depth_time, 3),
            "total": round(map_time + depth_time, 3),
        },
        "trajectory": trace,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(rpath, record)
    return record


def _top_entry(rec: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "label": rec["label"],
        "mapper_name": rec["mapper_name"],
        "scheduler": rec["scheduler"],
        "logical_depth": rec["logical_depth"],
        "inter_block_all": rec["inter_block_all"],
        "inter_block_t_only": rec["inter_block_t_only"],
        "active_blocks": rec["mapping_metrics_v2"]["active_blocks"],
        "unused_blocks": rec["mapping_metrics_v2"]["unused_blocks"],
        "num_multiblock": rec["mapping_metrics_v2"]["num_multiblock"],
        "mean_mst": rec["mapping_metrics_v2"]["mean_mst"],
        "support_peak": rec["mapping_metrics_v2"]["support_peak"],
        "score_total": rec["mapping_metrics_v2"]["total"],
        "score_kwargs_v2": rec["score_kwargs_v2"],
        "sa_params": rec["sa_params"],
        "cp_sat_time_limit": rec.get("cp_sat_time_limit", 120.0),
    }


def _select_top_configs(
    records: List[Dict[str, Any]],
    *,
    top_k: int = 8,
    source: str,
) -> List[Dict[str, Any]]:
    ranked = sorted(records, key=lambda r: (r["logical_depth"], r["mapping_metrics_v2"]["total"]))
    return [_top_entry(rec, source=source) for rec in ranked[:top_k]]


def _save_top_configs(name: str, entries: List[Dict[str, Any]]) -> str:
    payload = {
        "name": name,
        "circuit": GF2_CIRCUIT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    _save_json(path, payload)
    return path


def _load_top_configs(name: str) -> List[Dict[str, Any]]:
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Top-config summary not found: {path}\n"
            "Run phase 1 and/or phase 2 first."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("entries", []))


def _save_summary(name: str, payload: Dict[str, Any]) -> str:
    _ensure_dirs()
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    _save_json(path, payload)
    return path


def _add_bar_labels(ax, bars) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _annotate_points(ax, records: List[Dict[str, Any]], x_key: str, y_key: str) -> None:
    for rec in records:
        ax.annotate(
            rec["label"],
            (
                rec["mapping_metrics_v2"][x_key],
                rec[y_key] if y_key in rec else rec["mapping_metrics_v2"][y_key],
            ),
            fontsize=6,
            alpha=0.8,
        )


def _plot_baselines(records: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in records]
    depths = [r["logical_depth"] for r in records]
    active = [r["mapping_metrics_v2"]["active_blocks"] for r in records]
    multi = [r["mapping_metrics_v2"]["num_multiblock"] for r in records]
    mst = [r["mapping_metrics_v2"]["mean_mst"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("GF2 Baseline Comparison  (greedy depth, scored under SA-v2 metrics)", fontsize=12)

    bars = axes[0, 0].bar(labels, depths, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    _add_bar_labels(axes[0, 0], bars)
    axes[0, 0].axhline(BASELINES["random_greedy"], color="red", linestyle="--", linewidth=1.2, label="random+greedy")
    axes[0, 0].axhline(BASELINES["random_cpsat"], color="black", linestyle=":", linewidth=1.2, label="random+cpsat")
    axes[0, 0].set_title("Total depth")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(fontsize=7)

    bars = axes[0, 1].bar(labels, active, color="#55A868")
    _add_bar_labels(axes[0, 1], bars)
    axes[0, 1].set_title("Active blocks")
    axes[0, 1].tick_params(axis="x", rotation=20)

    bars = axes[1, 0].bar(labels, multi, color="#DD8452")
    _add_bar_labels(axes[1, 0], bars)
    axes[1, 0].set_title("Multi-block rotations")
    axes[1, 0].tick_params(axis="x", rotation=20)

    bars = axes[1, 1].bar(labels, mst, color="#937860")
    _add_bar_labels(axes[1, 1], bars)
    axes[1, 1].set_title("Mean MST of multi-block rotations")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory(record: Dict[str, Any], out_path: str) -> None:
    trace = record.get("trajectory") or []
    if not trace:
        return

    pcts = [t["pct"] for t in trace]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"SA-v2 Annealing Trajectory  ({record['label']})", fontsize=12)

    axes[0, 0].plot(pcts, [t["best_total"] for t in trace], "o-", label="best")
    axes[0, 0].plot(pcts, [t["current_total"] for t in trace], "s--", label="current")
    axes[0, 0].set_title("Score trajectory")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(pcts, [t["best_active_blocks"] for t in trace], "o-", label="active")
    axes[0, 1].plot(pcts, [t["best_unused_blocks"] for t in trace], "s--", label="unused")
    axes[0, 1].set_title("Block utilization")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(pcts, [t["best_num_multiblock"] for t in trace], "o-", color="#DD8452")
    axes[1, 0].set_title("Multi-block rotations")

    axes[1, 1].plot(pcts, [t["best_span_total"] for t in trace], "o-", color="#4C72B0")
    axes[1, 1].set_title("Span total")

    axes[2, 0].plot(pcts, [t["best_mean_mst"] for t in trace], "o-", color="#55A868")
    axes[2, 0].set_title("Mean MST")

    axes[2, 1].plot(pcts, [t["best_support_peak"] for t in trace], "o-", label="peak")
    axes[2, 1].plot(pcts, [t["best_support_range"] for t in trace], "s--", label="range")
    axes[2, 1].set_title("Support load shape")
    axes[2, 1].legend(fontsize=7)

    for ax in axes.ravel():
        ax.set_xlabel("anneal %")
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_sweeps(results_by_weight: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    weights = list(results_by_weight.keys())
    ncols = 3
    nrows = math.ceil(len(weights) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.ravel()
    fig.suptitle("SA-v2 Weight Sweeps on GF2  (greedy depth)", fontsize=12)

    for i, weight in enumerate(weights):
        recs = results_by_weight[weight]
        xs = [r["score_kwargs_v2"][weight] for r in recs]
        ys = [r["logical_depth"] for r in recs]
        axes[i].plot(xs, ys, "o-", color="#4C72B0")
        axes[i].axhline(BASELINES["random_greedy"], color="red", linestyle="--", linewidth=1.0)
        axes[i].axhline(BASELINES["random_cpsat"], color="black", linestyle=":", linewidth=1.0)
        if min(xs) > 0:
            axes[i].set_xscale("log")
        axes[i].set_title(f"Depth vs {weight}")
        axes[i].set_xlabel(weight)
        axes[i].set_ylabel("greedy depth")
        axes[i].tick_params(labelsize=7)

    for j in range(len(weights), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_preset_tradeoffs(records: List[Dict[str, Any]], out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("SA-v2 Preset Tradeoffs on GF2  (greedy depth)", fontsize=12)

    metrics = [
        ("active_blocks", "Active blocks"),
        ("num_multiblock", "Multi-block rotations"),
        ("mean_mst", "Mean MST"),
        ("support_peak", "Support peak"),
    ]
    for ax, (key, title) in zip(axes.ravel(), metrics):
        xs = [r["mapping_metrics_v2"][key] for r in records]
        ys = [r["logical_depth"] for r in records]
        ax.scatter(xs, ys, s=50, color="#4C72B0")
        for rec in records:
            ax.annotate(
                rec["label"],
                (rec["mapping_metrics_v2"][key], rec["logical_depth"]),
                fontsize=6,
            )
        ax.axhline(BASELINES["random_greedy"], color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel(title)
        ax.set_ylabel("greedy depth")
        ax.set_title(f"Depth vs {title.lower()}")
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_hyperparams(results_by_param: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("SA-v2 Hyperparameter Sweeps on GF2", fontsize=12)

    params = ["steps", "t0", "t_end"]
    for col, param in enumerate(params):
        recs = results_by_param[param]
        xs = [r["sa_params"][param] for r in recs]
        depth = [r["logical_depth"] for r in recs]
        map_t = [r["timing_sec"]["map"] for r in recs]

        axes[0, col].plot(xs, depth, "o-", color="#4C72B0")
        axes[0, col].axhline(BASELINES["random_greedy"], color="red", linestyle="--", linewidth=1.0)
        axes[0, col].axhline(BASELINES["random_cpsat"], color="black", linestyle=":", linewidth=1.0)
        axes[0, col].set_title(f"Depth vs {param}")
        axes[0, col].set_ylabel("greedy depth")
        axes[0, col].tick_params(labelsize=7)
        if param != "steps":
            axes[0, col].set_xscale("log")

        axes[1, col].plot(xs, map_t, "o-", color="#55A868")
        axes[1, col].set_title(f"Map time vs {param}")
        axes[1, col].set_ylabel("map time (s)")
        axes[1, col].set_xlabel(param)
        axes[1, col].tick_params(labelsize=7)
        if param != "steps":
            axes[1, col].set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def phase0_baselines(*, seed: int, force: bool) -> Dict[str, Any]:
    records = [
        _run_mapping("random", phase="0", label="random", seed=seed, force=force),
        _run_mapping("round_robin", phase="0", label="round_robin", seed=seed, force=force),
        _run_mapping("sa_v1", phase="0", label="sa_v1", seed=seed, force=force),
        _run_mapping("sa_v2", phase="0", label="sa_v2_default", seed=seed, force=force, trajectory=True),
    ]

    best = min(records, key=lambda r: r["logical_depth"])
    _plot_baselines(records, os.path.join(FIG_DIR, "fig_01_baselines.png"))
    sa_v2_record = next(r for r in records if r["mapper_name"] == "sa_v2")
    _plot_trajectory(sa_v2_record, os.path.join(FIG_DIR, "fig_02_sa_v2_trajectory.png"))

    payload = {
        "phase": 0,
        "seed": seed,
        "best_label": best["label"],
        "best_depth": best["logical_depth"],
        "records": records,
    }
    _save_summary("phase0_baselines", payload)
    return payload


def phase1_weight_sweep(*, seed: int, force: bool) -> Dict[str, Any]:
    results_by_weight: Dict[str, List[Dict[str, Any]]] = {}
    for weight_name, values in WEIGHT_SWEEPS.items():
        records: List[Dict[str, Any]] = []
        for value in values:
            kw = dict(DEFAULT_SCORE_KWARGS_V2)
            kw[weight_name] = value
            records.append(_run_mapping(
                "sa_v2",
                phase="1w",
                label=f"{weight_name}={value:.0e}",
                seed=seed,
                force=force,
                score_kwargs=kw,
            ))
        results_by_weight[weight_name] = records

    preset_records: List[Dict[str, Any]] = []
    for preset_name, kw in V2_PRESETS.items():
        preset_records.append(_run_mapping(
            "sa_v2",
            phase="1p",
            label=preset_name,
            seed=seed,
            force=force,
            score_kwargs=kw,
        ))

    all_records = [r for rs in results_by_weight.values() for r in rs] + preset_records
    best = min(all_records, key=lambda r: r["logical_depth"])
    top_configs = _select_top_configs(all_records, top_k=10, source="phase1_weight_sweep")

    _plot_weight_sweeps(results_by_weight, os.path.join(FIG_DIR, "fig_03_weight_sweeps.png"))
    _plot_preset_tradeoffs(preset_records, os.path.join(FIG_DIR, "fig_04_preset_tradeoffs.png"))
    _save_top_configs("top_configs_phase1", top_configs)

    payload = {
        "phase": 1,
        "seed": seed,
        "weight_sweeps": results_by_weight,
        "presets": preset_records,
        "best_label": best["label"],
        "best_depth": best["logical_depth"],
        "best_score_kwargs": best["score_kwargs_v2"],
        "top_configs": top_configs,
    }
    _save_summary("phase1_weight_sweep", payload)
    return payload


def phase2_hyperparams(*, seed: int, preset: str, force: bool) -> Dict[str, Any]:
    if preset not in V2_PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Available: {sorted(V2_PRESETS)}")

    base_kw = deepcopy(V2_PRESETS[preset])
    results_by_param: Dict[str, List[Dict[str, Any]]] = {
        "steps": [],
        "t0": [],
        "t_end": [],
    }

    for steps in HYPER_SWEEPS["steps"]:
        results_by_param["steps"].append(_run_mapping(
            "sa_v2",
            phase="2s",
            label=f"{preset}_steps={steps}",
            seed=seed,
            force=force,
            score_kwargs=base_kw,
            sa_steps=steps,
            sa_t0=SA_V2_EXPLORE["t0"],
            sa_tend=SA_V2_EXPLORE["t_end"],
        ))
    for t0 in HYPER_SWEEPS["t0"]:
        results_by_param["t0"].append(_run_mapping(
            "sa_v2",
            phase="2t0",
            label=f"{preset}_t0={t0:.0e}",
            seed=seed,
            force=force,
            score_kwargs=base_kw,
            sa_steps=SA_V2_EXPLORE["steps"],
            sa_t0=t0,
            sa_tend=SA_V2_EXPLORE["t_end"],
        ))
    for t_end in HYPER_SWEEPS["t_end"]:
        results_by_param["t_end"].append(_run_mapping(
            "sa_v2",
            phase="2te",
            label=f"{preset}_tend={t_end:.0e}",
            seed=seed,
            force=force,
            score_kwargs=base_kw,
            sa_steps=SA_V2_EXPLORE["steps"],
            sa_t0=SA_V2_EXPLORE["t0"],
            sa_tend=t_end,
        ))

    all_records = [r for rs in results_by_param.values() for r in rs]
    best = min(all_records, key=lambda r: r["logical_depth"])
    top_configs = _select_top_configs(all_records, top_k=10, source=f"phase2_hyperparams:{preset}")
    _plot_hyperparams(results_by_param, os.path.join(FIG_DIR, "fig_05_hyperparams.png"))
    _save_top_configs("top_configs_phase2", top_configs)

    payload = {
        "phase": 2,
        "seed": seed,
        "preset": preset,
        "base_score_kwargs": base_kw,
        "results_by_param": results_by_param,
        "best_label": best["label"],
        "best_depth": best["logical_depth"],
        "best_sa_params": best["sa_params"],
        "top_configs": top_configs,
    }
    _save_summary("phase2_hyperparams", payload)
    return payload


def phase3_cpsat_replay(
    *,
    seed: int,
    top_source: str,
    top_k: int,
    cp_time: float,
    force: bool,
) -> Dict[str, Any]:
    name_map = {
        "phase1": "top_configs_phase1",
        "phase2": "top_configs_phase2",
    }
    summary_name = name_map.get(top_source, top_source)
    entries = _load_top_configs(summary_name)[:top_k]

    results: List[Dict[str, Any]] = []
    for entry in entries:
        results.append(_run_mapping(
            "sa_v2",
            phase="3cpsat",
            label=f"cpsat_{entry['label']}",
            seed=seed,
            force=force,
            score_kwargs=entry["score_kwargs_v2"],
            sa_steps=entry["sa_params"]["steps"],
            sa_t0=entry["sa_params"]["t0"],
            sa_tend=entry["sa_params"]["t_end"],
            scheduler_name="cp_sat",
            cp_sat_time_limit=cp_time,
        ))

    ranked = sorted(results, key=lambda r: r["logical_depth"])
    payload = {
        "phase": 3,
        "seed": seed,
        "top_source": top_source,
        "cp_sat_time_limit": cp_time,
        "top_k": top_k,
        "results": ranked,
        "best_label": ranked[0]["label"] if ranked else None,
        "best_depth": ranked[0]["logical_depth"] if ranked else None,
    }
    _save_summary("phase3_cpsat_replay", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for the isolated SA-v2 mapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", choices=["0", "1", "2", "3", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="balanced_tradeoff", choices=sorted(V2_PRESETS))
    parser.add_argument("--top-source", default="phase2", choices=["phase1", "phase2"],
                        help="Which saved top-config set to replay with CP-SAT in phase 3")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many saved top configs to replay with CP-SAT")
    parser.add_argument("--cp-time", type=float, default=120.0,
                        help="CP-SAT time limit per layer for phase 3")
    parser.add_argument("--force", action="store_true", help="Re-run even if raw JSON exists")
    args = parser.parse_args()

    _ensure_dirs()
    print(f"\n[sensitivity_analysis_v2] phase={args.phase} seed={args.seed} preset={args.preset}")
    print(f"[results] {RESULTS_ROOT}")
    print(f"[circuit] {GF2_CIRCUIT}")
    print(f"[greedy baseline] random={BASELINES['random_greedy']}  random+cpsat={BASELINES['random_cpsat']}")
    print(f"[sa_v2 default] steps={SA_V2_EXPLORE['steps']} t0={SA_V2_EXPLORE['t0']:.1e} t_end={SA_V2_EXPLORE['t_end']}")
    print(f"[hyper sweep center] steps≈25000  t0≈1e5  t_end≈0.05")
    print(f"[phase3] top_source={args.top_source} top_k={args.top_k} cp_time={args.cp_time}\n")

    if args.phase in ("0", "all"):
        phase0_baselines(seed=args.seed, force=args.force)
    if args.phase in ("1", "all"):
        phase1_weight_sweep(seed=args.seed, force=args.force)
    if args.phase in ("2", "all"):
        phase2_hyperparams(seed=args.seed, preset=args.preset, force=args.force)
    if args.phase in ("3", "all"):
        phase3_cpsat_replay(
            seed=args.seed,
            top_source=args.top_source,
            top_k=args.top_k,
            cp_time=args.cp_time,
            force=args.force,
        )


if __name__ == "__main__":
    main()
