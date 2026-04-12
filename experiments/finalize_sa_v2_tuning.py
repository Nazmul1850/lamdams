"""
Global SA-v2 tuning workflow over cached PBC circuits.

This script keeps the existing experiment runners intact and adds a separate,
staged workflow for selecting:
1. impactful SA-v2 score weights ("scaling factors")
2. SA hyperparameters
3. top configurations to replay with CP-SAT
4. a final stress-test configuration

Design goals
------------
- Start from one base point and change one thing at a time.
- Average decisions across the cached PBC circuits being tested.
- Save enough raw and aggregated data to justify the final chosen values.
- Explicitly identify low-impact variables so they can be ignored in figures.

Outputs
-------
results/sa_v2_final_tuning/
  raw/       one JSON per run
  summary/   phase summaries and final choice
  figures/   paper-facing sensitivity plots
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.sensitivity_analysis_v2 import (
    _compute_greedy_depth,
    _count_inter_block,
    _score_to_json,
)
from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.mapping.algos.sa_v2 import DEFAULT_SCORE_KWARGS_V2, score_mapping_v2
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingPlan, MappingProblem, get_mapper


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBC_DIR = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
BENCH_DIR = os.path.join(_ROOT, "circuits", "benchmarks")

RESULTS_ROOT = os.path.join(_ROOT, "results", "sa_v2_final_tuning")
RAW_DIR = os.path.join(RESULTS_ROOT, "raw")
SUMMARY_DIR = os.path.join(RESULTS_ROOT, "summary")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures")
SC_BASELINE_PATH = os.path.join(_ROOT, "results", "sc_baseline", "sc_baseline.json")

DEFAULT_SEED = 42
N_DATA = 11
DEFAULT_TOPOLOGIES = ["grid", "ring"]

BASE_SCORE_KWARGS = dict(DEFAULT_SCORE_KWARGS_V2)
BASE_SA_PARAMS = {
    "steps": 25_000,
    "t0": 1e5,
    "t_end": 0.05,
}

LOW_IMPACT_DEPTH_THRESHOLD = 0.01
LOW_IMPACT_INTERBLOCK_THRESHOLD = 0.01
MARGINAL_GAIN_THRESHOLD = 0.002

WEIGHT_CANDIDATES: Dict[str, List[float]] = {
    "W_UNUSED_BLOCKS": [0.0, 1e5, 5e5, 1e6, 5e6, 1e7],
    "W_OCC_RANGE": [0.0, 1e3, 1e4, 5e4, 1e5],
    "W_OCC_STD": [0.0, 1e2, 1e3, 5e3, 1e4],
    "W_MULTI_BLOCK": [0.0, 5e4, 2e5, 5e5, 1e6],
    "W_SPAN": [0.0, 1e4, 5e4, 1e5, 2e5],
    "W_MST": [0.0, 1e2, 5e2, 1e3, 5e3],
    "W_SPLIT": [0.0, 5.0, 10.0, 50.0, 100.0],
    "W_SUPPORT_PEAK": [0.0, 10.0, 1e2, 5e2, 1e3],
    "W_SUPPORT_RANGE": [0.0, 5.0, 20.0, 50.0, 100.0],
    "W_SUPPORT_STD": [0.0, 1.0, 10.0, 50.0],
}

WEIGHT_DESCRIPTIONS: Dict[str, str] = {
    "W_UNUSED_BLOCKS": "Penalty for leaving hardware blocks unused; drives block utilization.",
    "W_OCC_RANGE": "Penalty on occupancy max-min across blocks; evens out logical assignment.",
    "W_OCC_STD": "Penalty on occupancy standard deviation; smoother occupancy balancing.",
    "W_MULTI_BLOCK": "Penalty on the number of rotations that touch multiple blocks.",
    "W_SPAN": "Penalty on the number of extra blocks touched by each rotation (k-1 span).",
    "W_MST": "Penalty on the graph distance among touched blocks; encourages spatial locality.",
    "W_SPLIT": "Penalty on uneven support split inside a multi-block rotation.",
    "W_SUPPORT_PEAK": "Penalty on the busiest block under rotation-touch load.",
    "W_SUPPORT_RANGE": "Penalty on max-min support load across blocks.",
    "W_SUPPORT_STD": "Penalty on support-load standard deviation across blocks.",
}

HYPER_CANDIDATES: Dict[str, List[float]] = {
    "steps": [20_000, 22_500, 25_000, 27_500, 30_000, 35_000, 40_000],
    "t0": [7.5e4, 9e4, 1e5, 1.1e5, 1.25e5, 1.5e5, 2e5],
    "t_end": [0.02, 0.035, 0.05, 0.065, 0.08, 0.1],
}


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[sa_v2_tuning {now}] {message}", flush=True)


def _fmt_case(circuit: str, topology: str) -> str:
    return f"{circuit} [{topology}]"


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


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_summary(name: str, payload: Dict[str, Any]) -> str:
    _ensure_dirs()
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    _save_json(path, payload)
    return path


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_surface_code_baselines() -> Dict[str, float]:
    if not os.path.exists(SC_BASELINE_PATH):
        return {}
    payload = _load_json(SC_BASELINE_PATH)
    baselines: Dict[str, float] = {}
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("rows", [])
    for row in rows:
        circuit = row.get("circuit")
        depth = row.get("sc_logical_depth") or row.get("sc_depth")
        if circuit and depth is not None:
            baselines[circuit] = float(depth)
    return baselines


def _candidate_pbc_paths(circuit_name: str) -> List[str]:
    return [
        os.path.join(PBC_DIR, f"{circuit_name}.json"),
        os.path.join(PBC_DIR, f"{circuit_name}_PBC.json"),
        os.path.join(BENCH_DIR, f"{circuit_name}.json"),
        os.path.join(BENCH_DIR, f"{circuit_name}_PBC.json"),
    ]


def _circuit_name_from_pbc_filename(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    if name.endswith("_PBC"):
        return name[:-4]
    return name


def _resolve_pbc_path(circuit_name: str) -> str:
    for path in _candidate_pbc_paths(circuit_name):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No cached PBC found for circuit '{circuit_name}'. Checked:\n"
        + "\n".join(_candidate_pbc_paths(circuit_name))
    )


def _load_pbc(circuit_name: str):
    path = _resolve_pbc_path(circuit_name)
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(path)
    first_rot = next(iter(conv.program.rotations))
    n_logicals = len(first_rot.axis.lstrip("+-"))
    rotations = list(conv.program.rotations)
    return path, n_logicals, rotations, conv


def _config_signature(
    *,
    mapper_name: str,
    scheduler_name: str,
    seed: int,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    score_kwargs: Optional[Dict[str, float]],
) -> str:
    payload = {
        "mapper": mapper_name,
        "scheduler": scheduler_name,
        "seed": seed,
        "steps": sa_steps,
        "t0": sa_t0,
        "tend": sa_tend,
        "score_kwargs": score_kwargs or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def _legacy_result_path(
    phase: str,
    circuit: str,
    topology: str,
    mapper_name: str,
    scheduler_name: str,
    seed: int,
    label: str,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
) -> str:
    fname = (
        f"{circuit}_p{phase}_{topology}_{mapper_name}_{scheduler_name}"
        f"_steps={sa_steps}_t0={_sci(sa_t0)}_tend={_sci(sa_tend)}"
        f"_seed{seed}_{_slug(label)}.json"
    )
    return os.path.join(RAW_DIR, fname)


def _result_path(
    phase: str,
    circuit: str,
    topology: str,
    mapper_name: str,
    scheduler_name: str,
    seed: int,
    label: str,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    score_kwargs: Optional[Dict[str, float]] = None,
) -> str:
    _ensure_dirs()
    sig = _config_signature(
        mapper_name=mapper_name,
        scheduler_name=scheduler_name,
        seed=seed,
        sa_steps=sa_steps,
        sa_t0=sa_t0,
        sa_tend=sa_tend,
        score_kwargs=score_kwargs,
    )
    fname = (
        f"{circuit}_p{phase}_{topology}_{scheduler_name}"
        f"_s{seed}_{_slug(label)}_{sig}.json"
    )
    return os.path.join(RAW_DIR, fname)


def _merge_score_kwargs(score_kwargs: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(BASE_SCORE_KWARGS)
    merged.update(score_kwargs or {})
    return merged


def _mapping_plan_for(
    mapper_name: str,
    n_logicals: int,
    hw,
    rotations: list,
    *,
    seed: int,
    sa_steps: int,
    sa_t0: float,
    sa_tend: float,
    score_kwargs: Optional[Dict[str, float]] = None,
) -> MappingPlan:
    problem = MappingProblem(n_logicals=n_logicals)
    if mapper_name == "random":
        return get_mapper("pure_random").solve(problem, hw, MappingConfig(seed=seed))
    if mapper_name == "round_robin":
        return get_mapper("auto_round_robin_mapping").solve(problem, hw, MappingConfig(seed=seed))
    if mapper_name == "sa_v1":
        return get_mapper("simulated_annealing").solve(
            problem,
            hw,
            MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend),
            {"rotations": rotations, "verbose": False},
        )
    if mapper_name == "sa_v2":
        return get_mapper("sa_v2").solve(
            problem,
            hw,
            MappingConfig(seed=seed, sa_steps=sa_steps, sa_t0=sa_t0, sa_tend=sa_tend),
            {"rotations": rotations, "verbose": False, "score_kwargs": _merge_score_kwargs(score_kwargs)},
        )
    raise ValueError(f"Unknown mapper_name: {mapper_name!r}")


def _run_one(
    circuit: str,
    mapper_name: str,
    *,
    phase: str,
    label: str,
    topology: str = "grid",
    scheduler_name: str = "greedy_critical",
    seed: int = DEFAULT_SEED,
    sa_steps: int = BASE_SA_PARAMS["steps"],
    sa_t0: float = BASE_SA_PARAMS["t0"],
    sa_tend: float = BASE_SA_PARAMS["t_end"],
    score_kwargs: Optional[Dict[str, float]] = None,
    cp_sat_time_limit: Optional[float] = None,
    force: bool = False,
) -> Dict[str, Any]:
    path = _result_path(
        phase,
        circuit,
        topology,
        mapper_name,
        scheduler_name,
        seed,
        label,
        sa_steps,
        sa_t0,
        sa_tend,
        score_kwargs=score_kwargs,
    )
    legacy_path = _legacy_result_path(
        phase,
        circuit,
        topology,
        mapper_name,
        scheduler_name,
        seed,
        label,
        sa_steps,
        sa_t0,
        sa_tend,
    )
    if not force and os.path.exists(path):
        _log(f"cache hit: {_fmt_case(circuit, topology)} | {mapper_name}/{scheduler_name} | label={label}")
        return _load_json(path)
    if not force and os.path.exists(legacy_path):
        _log(
            f"legacy cache hit: {_fmt_case(circuit, topology)} | {mapper_name}/{scheduler_name} | "
            f"label={label}"
        )
        return _load_json(legacy_path)

    _log(
        "run start: "
        f"{_fmt_case(circuit, topology)} | mapper={mapper_name} | scheduler={scheduler_name} | "
        f"label={label} | steps={sa_steps} t0={sa_t0:.2g} tend={sa_tend:.3g} | "
        f"cp_limit={'none' if cp_sat_time_limit is None else cp_sat_time_limit}"
    )

    pbc_path, n_logicals, rotations, conv = _load_pbc(circuit)
    hw, hw_spec = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=0.0,
        n_data=N_DATA,
        coupler_capacity=1,
    )
    sc_baselines = _load_surface_code_baselines()
    sc_depth = sc_baselines.get(
        circuit,
        float(sum(1 for r in rotations if abs(r.angle) < math.pi / 2 - 1e-9)),
    )

    t0 = time.perf_counter()
    plan = _mapping_plan_for(
        mapper_name,
        n_logicals,
        hw,
        rotations,
        seed=seed,
        sa_steps=sa_steps,
        sa_t0=sa_t0,
        sa_tend=sa_tend,
        score_kwargs=score_kwargs,
    )
    map_time = time.perf_counter() - t0

    effective_score_kwargs = _merge_score_kwargs(score_kwargs)
    score = score_mapping_v2(rotations, hw, **effective_score_kwargs)
    inter_block_all = _count_inter_block(rotations, plan, t_only=False)
    inter_block_t_only = _count_inter_block(rotations, plan, t_only=True)

    t1 = time.perf_counter()
    logical_depth, layer_depths = _compute_greedy_depth(
        conv,
        hw,
        plan,
        seed=seed,
        n_data=N_DATA,
        scheduler_name=scheduler_name,
        cp_sat_time_limit=cp_sat_time_limit,
    )
    depth_time = time.perf_counter() - t1

    record = {
        "phase": phase,
        "label": label,
        "circuit": circuit,
        "topology": topology,
        "pbc_path": pbc_path,
        "mapper_name": mapper_name,
        "scheduler": scheduler_name,
        "seed": seed,
        "hardware": {
            "label": hw_spec.label(),
            "topology": topology,
            "n_blocks": len(hw.blocks),
            "n_logicals": n_logicals,
            "n_data": N_DATA,
        },
        "surface_code_baseline": {
            "logical_depth": sc_depth,
            "depth_over_sc": (logical_depth / sc_depth) if sc_depth else None,
        },
        "sa_params": {
            "steps": sa_steps,
            "t0": sa_t0,
            "t_end": sa_tend,
        },
        "score_kwargs_v2": effective_score_kwargs,
        "mapping_meta": plan.meta,
        "mapping_metrics_v2": _score_to_json(score),
        "inter_block_all": inter_block_all,
        "inter_block_t_only": inter_block_t_only,
        "logical_depth": logical_depth,
        "layer_depths": layer_depths,
        "timing_sec": {
            "map": round(map_time, 3),
            "schedule_eval": round(depth_time, 3),
            "total": round(map_time + depth_time, 3),
        },
        "cp_sat_time_limit": cp_sat_time_limit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(path, record)
    _log(
        "run done: "
        f"{_fmt_case(circuit, topology)} | mapper={mapper_name} | scheduler={scheduler_name} | "
        f"depth={logical_depth} | depth/sc={record['surface_code_baseline']['depth_over_sc']:.3f} | "
        f"inter={inter_block_all} | map={record['timing_sec']['map']}s"
    )
    return record


def _safe_ratio(numer: float, denom: float) -> float:
    if abs(denom) < 1e-12:
        return 0.0
    return numer / denom


def _find_record(records: List[Dict[str, Any]], circuit: str, label: str, topology: Optional[str] = None) -> Dict[str, Any]:
    for rec in records:
        if rec["circuit"] == circuit and rec["label"] == label and (topology is None or rec.get("topology") == topology):
            return rec
    raise KeyError(f"Record not found for circuit={circuit!r}, label={label!r}, topology={topology!r}")


def _aggregate_candidate(
    candidate_label: str,
    records: List[Dict[str, Any]],
    baselines: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    per_circuit: List[Dict[str, Any]] = []
    for rec in sorted(records, key=lambda r: (r.get("topology", "grid"), r["circuit"])):
        circuit = rec["circuit"]
        topology = rec.get("topology", "grid")
        baseline = baselines[(circuit, topology)]
        depth_gain = _safe_ratio(
            baseline["logical_depth"] - rec["logical_depth"],
            baseline["logical_depth"],
        )
        inter_gain = _safe_ratio(
            baseline["inter_block_all"] - rec["inter_block_all"],
            baseline["inter_block_all"],
        )
        peak_gain = _safe_ratio(
            baseline["mapping_metrics_v2"]["support_peak"] - rec["mapping_metrics_v2"]["support_peak"],
            baseline["mapping_metrics_v2"]["support_peak"],
        )
        range_gain = _safe_ratio(
            baseline["mapping_metrics_v2"]["support_range"] - rec["mapping_metrics_v2"]["support_range"],
            baseline["mapping_metrics_v2"]["support_range"],
        )
        active_frac = _safe_ratio(
            rec["mapping_metrics_v2"]["active_blocks"],
            rec["hardware"]["n_blocks"],
        )
        per_circuit.append({
            "circuit": circuit,
            "topology": topology,
            "logical_depth": rec["logical_depth"],
            "inter_block_all": rec["inter_block_all"],
            "map_time_sec": rec["timing_sec"]["map"],
            "active_blocks": rec["mapping_metrics_v2"]["active_blocks"],
            "unused_blocks": rec["mapping_metrics_v2"]["unused_blocks"],
            "support_peak": rec["mapping_metrics_v2"]["support_peak"],
            "support_range": rec["mapping_metrics_v2"]["support_range"],
            "depth_gain_vs_random_greedy": depth_gain,
            "interblock_gain_vs_random_greedy": inter_gain,
            "support_peak_gain_vs_random_greedy": peak_gain,
            "support_range_gain_vs_random_greedy": range_gain,
            "active_block_fraction": active_frac,
            "score_total": rec["mapping_metrics_v2"]["total"],
            "depth_over_sc": rec.get("surface_code_baseline", {}).get("depth_over_sc"),
        })

    mean_depth_gain = _safe_ratio(
        sum(x["depth_gain_vs_random_greedy"] for x in per_circuit),
        len(per_circuit),
    )
    mean_inter_gain = _safe_ratio(
        sum(x["interblock_gain_vs_random_greedy"] for x in per_circuit),
        len(per_circuit),
    )
    mean_support_peak_gain = _safe_ratio(
        sum(x["support_peak_gain_vs_random_greedy"] for x in per_circuit),
        len(per_circuit),
    )
    mean_support_range_gain = _safe_ratio(
        sum(x["support_range_gain_vs_random_greedy"] for x in per_circuit),
        len(per_circuit),
    )
    mean_active_frac = _safe_ratio(
        sum(x["active_block_fraction"] for x in per_circuit),
        len(per_circuit),
    )
    mean_map_time = _safe_ratio(sum(x["map_time_sec"] for x in per_circuit), len(per_circuit))
    mean_score_total = _safe_ratio(sum(x["score_total"] for x in per_circuit), len(per_circuit))

    return {
        "candidate_label": candidate_label,
        "per_circuit": per_circuit,
        "mean_depth_gain_vs_random_greedy": mean_depth_gain,
        "mean_interblock_gain_vs_random_greedy": mean_inter_gain,
        "mean_support_peak_gain_vs_random_greedy": mean_support_peak_gain,
        "mean_support_range_gain_vs_random_greedy": mean_support_range_gain,
        "mean_active_block_fraction": mean_active_frac,
        "mean_map_time_sec": mean_map_time,
        "mean_score_total": mean_score_total,
    }


def _candidate_sort_key(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        -summary["mean_depth_gain_vs_random_greedy"],
        -summary["mean_interblock_gain_vs_random_greedy"],
        -summary["mean_support_range_gain_vs_random_greedy"],
        summary["mean_map_time_sec"],
    )


def _best_summary(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(summaries, key=_candidate_sort_key)[0]


def _top_summaries(summaries: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
    return sorted(summaries, key=_candidate_sort_key)[:top_k]


def _final_config_entry(
    label: str,
    score_kwargs: Dict[str, float],
    sa_params: Dict[str, float],
    aggregate: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    return {
        "label": label,
        "source": source,
        "score_kwargs_v2": dict(score_kwargs),
        "sa_params": dict(sa_params),
        "aggregate": aggregate,
    }


def _discover_default_circuits() -> Tuple[List[str], Dict[str, str]]:
    if not os.path.isdir(PBC_DIR):
        raise FileNotFoundError(f"PBC directory not found: {PBC_DIR}")

    usable: List[str] = []
    notes: Dict[str, str] = {}
    seen = set()
    for filename in sorted(os.listdir(PBC_DIR)):
        if not filename.endswith(".json"):
            continue
        circuit = _circuit_name_from_pbc_filename(filename)
        if circuit in seen:
            continue
        seen.add(circuit)
        path = os.path.join(PBC_DIR, filename)
        usable.append(circuit)
        notes[circuit] = path
    if not usable:
        raise FileNotFoundError(f"No cached PBC JSON files were found in {PBC_DIR}")
    return usable, notes


def _cases(circuits: List[str], topologies: List[str]) -> List[Tuple[str, str]]:
    return [(circuit, topology) for topology in topologies for circuit in circuits]


def _plot_phase0_baselines(records: List[Dict[str, Any]], out_path: str) -> None:
    circuits = sorted({r["circuit"] for r in records})
    topologies = sorted({r.get("topology", "grid") for r in records})
    labels = ["random", "round_robin", "sa_v1", "sa_v2_default"]
    colors = {
        "random": "#C44E52",
        "round_robin": "#8172B2",
        "sa_v1": "#55A868",
        "sa_v2_default": "#4C72B0",
    }
    x = list(range(len(circuits)))
    width = 0.18

    fig, axes = plt.subplots(len(topologies), 2, figsize=(13, 5 * len(topologies)))
    if len(topologies) == 1:
        axes = [axes]
    for row_idx, topology in enumerate(topologies):
        for j, label in enumerate(labels):
            xs = [i + (j - 1.5) * width for i in x]
            depth_ratios = [
                _find_record(records, c, label, topology)["surface_code_baseline"]["depth_over_sc"]
                for c in circuits
            ]
            inters = [_find_record(records, c, label, topology)["inter_block_all"] for c in circuits]
            axes[row_idx][0].bar(xs, depth_ratios, width=width, label=label, color=colors[label])
            axes[row_idx][1].bar(xs, inters, width=width, label=label, color=colors[label])

        axes[row_idx][0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axes[row_idx][0].set_xticks(x)
        axes[row_idx][0].set_xticklabels(circuits, rotation=15)
        axes[row_idx][0].set_title(f"{topology}: normalized depth vs surface code")
        axes[row_idx][0].set_ylabel("qLDPC depth / SC depth")
        axes[row_idx][0].legend(fontsize=8)

        axes[row_idx][1].set_xticks(x)
        axes[row_idx][1].set_xticklabels(circuits, rotation=15)
        axes[row_idx][1].set_title(f"{topology}: inter-block rotations")
        axes[row_idx][1].set_ylabel("inter-block count")
        axes[row_idx][1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_impacts(weight_rows: List[Dict[str, Any]], out_path: str) -> None:
    rows = sorted(weight_rows, key=lambda r: r["depth_impact_span"], reverse=True)
    names = [r["weight_name"] for r in rows]
    depth_impacts = [r["depth_impact_span"] * 100.0 for r in rows]
    inter_impacts = [r["interblock_impact_span"] * 100.0 for r in rows]
    colors = ["#4C72B0" if r["selected_for_phase2"] else "#BBBBBB" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(names, depth_impacts, color=colors)
    axes[0].axhline(LOW_IMPACT_DEPTH_THRESHOLD * 100.0, color="red", linestyle="--", linewidth=1.0)
    axes[0].set_title("Depth sensitivity span")
    axes[0].set_ylabel("span in mean depth gain (%)")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(names, inter_impacts, color=colors)
    axes[1].axhline(LOW_IMPACT_INTERBLOCK_THRESHOLD * 100.0, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Inter-block sensitivity span")
    axes[1].set_ylabel("span in mean inter-block gain (%)")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_sweeps(weight_results: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    names = list(weight_results.keys())
    ncols = 3
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = axes.ravel()
    for i, name in enumerate(names):
        rows = sorted(weight_results[name], key=lambda r: float(r["value"]))
        xs = [r["value"] for r in rows]
        ys = [r["aggregate"]["mean_depth_gain_vs_random_greedy"] * 100.0 for r in rows]
        axes[i].plot(xs, ys, "o-", color="#4C72B0")
        if xs and min(xs) > 0:
            axes[i].set_xscale("log")
        axes[i].set_title(name)
        axes[i].set_xlabel("candidate value")
        axes[i].set_ylabel("mean depth gain (%)")
        axes[i].tick_params(labelsize=7)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_trajectory(history: List[Dict[str, Any]], out_path: str) -> None:
    xs = list(range(len(history)))
    ys = [step["aggregate"]["mean_depth_gain_vs_random_greedy"] * 100.0 for step in history]
    labels = [step["label"] for step in history]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, "o-", color="#55A868")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("mean depth gain (%)")
    ax.set_title("Cumulative weight-selection trajectory")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_hyper_sweeps(hyper_results: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    params = list(hyper_results.keys())
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4.5))
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        rows = hyper_results[param]
        xs = [r["value"] for r in rows]
        ys = [r["aggregate"]["mean_depth_gain_vs_random_greedy"] * 100.0 for r in rows]
        ax.plot(xs, ys, "o-", color="#4C72B0")
        if param != "steps":
            ax.set_xscale("log")
        ax.set_title(param)
        ax.set_xlabel("candidate value")
        ax.set_ylabel("mean depth gain (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_stress(records: List[Dict[str, Any]], out_path: str) -> None:
    grouped: Dict[str, List[int]] = {}
    for rec in records:
        label = f"{rec['circuit']}:{rec.get('topology', 'grid')}"
        grouped.setdefault(label, []).append(rec["logical_depth"])
    circuits = sorted(grouped)
    values = [grouped[c] for c in circuits]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=circuits, showmeans=True)
    ax.set_title("Stress test over seeds")
    ax.set_ylabel("logical depth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def phase0_baselines(*, circuits: List[str], topologies: List[str], seed: int, force: bool) -> Dict[str, Any]:
    _log(
        f"phase 0 start: baselines over {len(circuits)} circuits x {len(topologies)} topologies "
        f"(seed={seed})"
    )
    labels_and_mappers = [
        ("random", "random"),
        ("round_robin", "round_robin"),
        ("sa_v1", "sa_v1"),
        ("sa_v2_default", "sa_v2"),
    ]
    records: List[Dict[str, Any]] = []
    for topology in topologies:
        for circuit in circuits:
            for label, mapper_name in labels_and_mappers:
                records.append(_run_one(
                    circuit,
                    mapper_name,
                    phase="0",
                    label=label,
                    topology=topology,
                    seed=seed,
                    force=force,
                ))

    baselines = {
        (circuit, topology): _find_record(records, circuit, "random", topology)
        for topology in topologies
        for circuit in circuits
    }
    sa_v2_default = [
        _find_record(records, circuit, "sa_v2_default", topology)
        for topology in topologies
        for circuit in circuits
    ]
    aggregate = _aggregate_candidate("sa_v2_default", sa_v2_default, baselines)
    _plot_phase0_baselines(records, os.path.join(FIG_DIR, "fig_01_phase0_baselines.png"))
    payload = {
        "phase": 0,
        "seed": seed,
        "circuits": circuits,
        "topologies": topologies,
        "baselines": {
            topology: {circuit: baselines[(circuit, topology)] for circuit in circuits}
            for topology in topologies
        },
        "sa_v2_default_aggregate": aggregate,
        "records": records,
    }
    _save_summary("phase0_baselines", payload)
    _log(
        "phase 0 done: "
        f"sa_v2 default mean depth gain vs random+greedy = "
        f"{aggregate['mean_depth_gain_vs_random_greedy']*100:.2f}%"
    )
    return payload


def phase1_screen_weights(
    *,
    circuits: List[str],
    topologies: List[str],
    seed: int,
    force: bool,
    baselines: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    _log(
        f"phase 1 start: screening {len(WEIGHT_CANDIDATES)} score variables over "
        f"{len(circuits)} circuits x {len(topologies)} topologies"
    )
    base_records = [
        _run_one(
            circuit,
            "sa_v2",
            phase="1base",
            label="base_weights",
            topology=topology,
            seed=seed,
            force=force,
            score_kwargs=BASE_SCORE_KWARGS,
        )
        for circuit, topology in _cases(circuits, topologies)
    ]
    base_aggregate = _aggregate_candidate("base_weights", base_records, baselines)

    weight_results: Dict[str, List[Dict[str, Any]]] = {}
    weight_rows: List[Dict[str, Any]] = []

    for weight_name, values in WEIGHT_CANDIDATES.items():
        _log(f"phase 1: sweeping {weight_name} over {len(values)} candidate values")
        rows: List[Dict[str, Any]] = []
        summaries: List[Dict[str, Any]] = []
        for value in values:
            candidate_records: List[Dict[str, Any]] = []
            score_kwargs = dict(BASE_SCORE_KWARGS)
            score_kwargs[weight_name] = value
            for circuit, topology in _cases(circuits, topologies):
                candidate_records.append(_run_one(
                    circuit,
                    "sa_v2",
                    phase="1w",
                    label=f"{weight_name}={value}",
                    topology=topology,
                    seed=seed,
                    force=force,
                    score_kwargs=score_kwargs,
                ))
            aggregate = _aggregate_candidate(f"{weight_name}={value}", candidate_records, baselines)
            summaries.append(aggregate)
            rows.append({
                "weight_name": weight_name,
                "value": value,
                "score_kwargs_v2": score_kwargs,
                "aggregate": aggregate,
            })

        best = _best_summary(summaries)
        depth_vals = [s["mean_depth_gain_vs_random_greedy"] for s in summaries]
        inter_vals = [s["mean_interblock_gain_vs_random_greedy"] for s in summaries]
        depth_impact_span = max(depth_vals) - min(depth_vals)
        inter_impact_span = max(inter_vals) - min(inter_vals)
        selected = (
            depth_impact_span >= LOW_IMPACT_DEPTH_THRESHOLD
            or inter_impact_span >= LOW_IMPACT_INTERBLOCK_THRESHOLD
        )
        weight_results[weight_name] = rows
        weight_rows.append({
            "weight_name": weight_name,
            "best_value": rows[[r["aggregate"]["candidate_label"] for r in rows].index(best["candidate_label"])]["value"],
            "base_value": BASE_SCORE_KWARGS[weight_name],
            "best_aggregate": best,
            "depth_impact_span": depth_impact_span,
            "interblock_impact_span": inter_impact_span,
            "selected_for_phase2": selected,
        })
        _log(
            f"phase 1: {weight_name} best={rows[[r['aggregate']['candidate_label'] for r in rows].index(best['candidate_label'])]['value']} "
            f"| depth-span={depth_impact_span*100:.2f}% | inter-span={inter_impact_span*100:.2f}% | "
            f"{'keep' if selected else 'ignore'}"
        )

    selected_variables = [r["weight_name"] for r in weight_rows if r["selected_for_phase2"]]
    ignored_variables = [r["weight_name"] for r in weight_rows if not r["selected_for_phase2"]]

    _plot_weight_sweeps(weight_results, os.path.join(FIG_DIR, "fig_02_phase1_weight_sweeps.png"))
    _plot_weight_impacts(weight_rows, os.path.join(FIG_DIR, "fig_03_phase1_weight_impacts.png"))

    payload = {
        "phase": 1,
        "circuits": circuits,
        "topologies": topologies,
        "seed": seed,
        "weight_descriptions": WEIGHT_DESCRIPTIONS,
        "base_aggregate": base_aggregate,
        "weight_results": weight_results,
        "weight_rows": sorted(weight_rows, key=lambda r: r["depth_impact_span"], reverse=True),
        "selected_variables": selected_variables,
        "ignored_variables": ignored_variables,
    }
    _save_summary("phase1_screen_weights", payload)
    _log(
        f"phase 1 done: selected {len(selected_variables)} impactful variables, "
        f"ignored {len(ignored_variables)} low-impact variables"
    )
    return payload


def phase2_select_weights(
    *,
    circuits: List[str],
    topologies: List[str],
    seed: int,
    force: bool,
    baselines: Dict[Tuple[str, str], Dict[str, Any]],
    screening: Dict[str, Any],
) -> Dict[str, Any]:
    _log("phase 2 start: cumulative weight selection from the base SA-v2 score")
    ordered = list(screening["weight_rows"])
    current_weights = dict(BASE_SCORE_KWARGS)
    current_records = [
        _run_one(
            circuit,
            "sa_v2",
            phase="2sel",
            label="current_base",
            topology=topology,
            seed=seed,
            force=force,
            score_kwargs=current_weights,
        )
        for circuit, topology in _cases(circuits, topologies)
    ]
    current_aggregate = _aggregate_candidate("current_base", current_records, baselines)
    history = [{
        "label": "base_weights",
        "score_kwargs_v2": dict(current_weights),
        "aggregate": current_aggregate,
        "accepted_change": None,
    }]
    decision_rows: List[Dict[str, Any]] = []

    for row in ordered:
        weight_name = row["weight_name"]
        if not row["selected_for_phase2"]:
            _log(f"phase 2: skip {weight_name} (low impact in phase 1)")
            decision_rows.append({
                "weight_name": weight_name,
                "status": "ignored_low_impact",
                "kept_value": current_weights[weight_name],
                "best_tested_value": row["best_value"],
            })
            continue

        candidates = []
        _log(f"phase 2: refining {weight_name}")
        for value in WEIGHT_CANDIDATES[weight_name]:
            trial_weights = dict(current_weights)
            trial_weights[weight_name] = value
            trial_records = [
                _run_one(
                    circuit,
                    "sa_v2",
                    phase="2sel",
                    label=f"{weight_name}={value}",
                    topology=topology,
                    seed=seed,
                    force=force,
                    score_kwargs=trial_weights,
                )
                for circuit, topology in _cases(circuits, topologies)
            ]
            trial_aggregate = _aggregate_candidate(f"{weight_name}={value}", trial_records, baselines)
            candidates.append({
                "weight_name": weight_name,
                "value": value,
                "score_kwargs_v2": trial_weights,
                "aggregate": trial_aggregate,
            })

        best_candidate = sorted(candidates, key=lambda c: _candidate_sort_key(c["aggregate"]))[0]
        gain_delta = (
            best_candidate["aggregate"]["mean_depth_gain_vs_random_greedy"]
            - current_aggregate["mean_depth_gain_vs_random_greedy"]
        )
        if gain_delta >= MARGINAL_GAIN_THRESHOLD:
            current_weights = dict(best_candidate["score_kwargs_v2"])
            current_aggregate = best_candidate["aggregate"]
            history.append({
                "label": f"{weight_name}={best_candidate['value']}",
                "score_kwargs_v2": dict(current_weights),
                "aggregate": current_aggregate,
                "accepted_change": weight_name,
            })
            decision_rows.append({
                "weight_name": weight_name,
                "status": "accepted",
                "old_value": history[-2]["score_kwargs_v2"][weight_name],
                "new_value": best_candidate["value"],
                "gain_delta": gain_delta,
                "aggregate": best_candidate["aggregate"],
            })
            _log(
                f"phase 2: accept {weight_name}={best_candidate['value']} "
                f"(delta gain={gain_delta*100:.2f}%, new mean gain="
                f"{current_aggregate['mean_depth_gain_vs_random_greedy']*100:.2f}%)"
            )
        else:
            decision_rows.append({
                "weight_name": weight_name,
                "status": "rejected_marginal",
                "kept_value": current_weights[weight_name],
                "best_tested_value": best_candidate["value"],
                "gain_delta": gain_delta,
                "aggregate": best_candidate["aggregate"],
            })
            _log(
                f"phase 2: keep {weight_name}={current_weights[weight_name]} "
                f"(best tested {best_candidate['value']} only changed mean gain by {gain_delta*100:.2f}%)"
            )

    _plot_selection_trajectory(history, os.path.join(FIG_DIR, "fig_04_phase2_weight_selection.png"))
    payload = {
        "phase": 2,
        "circuits": circuits,
        "topologies": topologies,
        "seed": seed,
        "selection_history": history,
        "decision_rows": decision_rows,
        "final_score_kwargs_v2": current_weights,
        "final_aggregate": current_aggregate,
    }
    _save_summary("phase2_select_weights", payload)
    _log("phase 2 done: finalized score weights for the next hyperparameter stage")
    return payload


def phase3_tune_hyperparams(
    *,
    circuits: List[str],
    topologies: List[str],
    seed: int,
    force: bool,
    baselines: Dict[Tuple[str, str], Dict[str, Any]],
    score_kwargs: Dict[str, float],
) -> Dict[str, Any]:
    _log("phase 3 start: tuning SA hyperparameters around the selected score weights")
    current_params = dict(BASE_SA_PARAMS)
    current_records = [
        _run_one(
            circuit,
            "sa_v2",
            phase="3hyp",
            label="base_hyper",
            topology=topology,
            seed=seed,
            force=force,
            score_kwargs=score_kwargs,
            sa_steps=int(current_params["steps"]),
            sa_t0=current_params["t0"],
            sa_tend=current_params["t_end"],
        )
        for circuit, topology in _cases(circuits, topologies)
    ]
    current_aggregate = _aggregate_candidate("base_hyper", current_records, baselines)
    history = [{
        "label": "base_hyper",
        "sa_params": dict(current_params),
        "aggregate": current_aggregate,
    }]
    hyper_results: Dict[str, List[Dict[str, Any]]] = {}
    top_candidates: List[Dict[str, Any]] = []

    for param_name, values in HYPER_CANDIDATES.items():
        _log(f"phase 3: sweeping {param_name} over {len(values)} candidate values")
        rows: List[Dict[str, Any]] = []
        for value in values:
            trial_params = dict(current_params)
            trial_params[param_name] = value
            records = [
                _run_one(
                    circuit,
                    "sa_v2",
                    phase=f"3{param_name}",
                    label=f"{param_name}={value}",
                    topology=topology,
                    seed=seed,
                    force=force,
                    score_kwargs=score_kwargs,
                    sa_steps=int(trial_params["steps"]),
                    sa_t0=trial_params["t0"],
                    sa_tend=trial_params["t_end"],
                )
                for circuit, topology in _cases(circuits, topologies)
            ]
            aggregate = _aggregate_candidate(f"{param_name}={value}", records, baselines)
            row = {
                "param_name": param_name,
                "value": value,
                "sa_params": dict(trial_params),
                "aggregate": aggregate,
            }
            rows.append(row)
            top_candidates.append(row)
        hyper_results[param_name] = rows

        best_row = sorted(rows, key=lambda r: _candidate_sort_key(r["aggregate"]))[0]
        gain_delta = (
            best_row["aggregate"]["mean_depth_gain_vs_random_greedy"]
            - current_aggregate["mean_depth_gain_vs_random_greedy"]
        )
        if gain_delta >= MARGINAL_GAIN_THRESHOLD:
            current_params = dict(best_row["sa_params"])
            current_aggregate = best_row["aggregate"]
            history.append({
                "label": f"{param_name}={best_row['value']}",
                "sa_params": dict(current_params),
                "aggregate": current_aggregate,
            })
            _log(
                f"phase 3: accept {param_name}={best_row['value']} "
                f"(delta gain={gain_delta*100:.2f}%, new mean gain="
                f"{current_aggregate['mean_depth_gain_vs_random_greedy']*100:.2f}%)"
            )
        else:
            _log(
                f"phase 3: keep current {param_name}={current_params[param_name]} "
                f"(best tested {best_row['value']} only changed mean gain by {gain_delta*100:.2f}%)"
            )

    _plot_hyper_sweeps(hyper_results, os.path.join(FIG_DIR, "fig_05_phase3_hyperparams.png"))

    ranked_top = []
    ranked_top.append(_final_config_entry(
        label="final_greedy_choice",
        score_kwargs=score_kwargs,
        sa_params=current_params,
        aggregate=current_aggregate,
        source="phase3_final_choice",
    ))
    for row in sorted(top_candidates, key=lambda r: _candidate_sort_key(r["aggregate"]))[:8]:
        entry = _final_config_entry(
            label=f"{row['param_name']}={row['value']}",
            score_kwargs=score_kwargs,
            sa_params=row["sa_params"],
            aggregate=row["aggregate"],
            source="phase3_hyperparams",
        )
        if not any(
            existing["sa_params"] == entry["sa_params"] and existing["score_kwargs_v2"] == entry["score_kwargs_v2"]
            for existing in ranked_top
        ):
            ranked_top.append(entry)

    payload = {
        "phase": 3,
        "circuits": circuits,
        "topologies": topologies,
        "seed": seed,
        "history": history,
        "hyper_results": hyper_results,
        "final_sa_params": current_params,
        "final_aggregate": current_aggregate,
        "top_configs": ranked_top,
    }
    _save_summary("phase3_tune_hyperparams", payload)
    _log(
        "phase 3 done: selected SA params "
        f"steps={current_params['steps']}, t0={current_params['t0']}, t_end={current_params['t_end']}"
    )
    return payload


def phase4_cpsat_replay(
    *,
    circuits: List[str],
    topologies: List[str],
    seed: int,
    force: bool,
    top_configs: List[Dict[str, Any]],
    cp_time: Optional[float],
) -> Dict[str, Any]:
    _log(
        f"phase 4 start: replaying top {len(top_configs)} greedy configs with CP-SAT "
        f"over {len(circuits)} circuits x {len(topologies)} topologies "
        f"(time limit: {'none' if cp_time is None else cp_time})"
    )
    results: List[Dict[str, Any]] = []
    for entry in top_configs:
        for circuit, topology in _cases(circuits, topologies):
            rec = _run_one(
                circuit,
                "sa_v2",
                phase="4cpsat",
                label=entry["label"],
                topology=topology,
                seed=seed,
                force=force,
                score_kwargs=entry["score_kwargs_v2"],
                sa_steps=int(entry["sa_params"]["steps"]),
                sa_t0=entry["sa_params"]["t0"],
                sa_tend=entry["sa_params"]["t_end"],
                scheduler_name="cp_sat",
                cp_sat_time_limit=cp_time,
            )
            results.append(rec)

    random_cpsat = [
        _run_one(
            circuit,
            "random",
            phase="4base",
            label="random_cpsat",
            topology=topology,
            seed=seed,
            force=force,
            scheduler_name="cp_sat",
            cp_sat_time_limit=cp_time,
        )
        for circuit, topology in _cases(circuits, topologies)
    ]
    baselines = {(rec["circuit"], rec["topology"]): rec for rec in random_cpsat}

    aggregates = []
    for entry in top_configs:
        entry_records = [r for r in results if r["label"] == entry["label"]]
        aggregate = _aggregate_candidate(entry["label"], entry_records, baselines)
        aggregates.append({
            "label": entry["label"],
            "score_kwargs_v2": entry["score_kwargs_v2"],
            "sa_params": entry["sa_params"],
            "aggregate_vs_random_cpsat": aggregate,
        })

    best = sorted(
        aggregates,
        key=lambda x: _candidate_sort_key({
            "mean_depth_gain_vs_random_greedy": x["aggregate_vs_random_cpsat"]["mean_depth_gain_vs_random_greedy"],
            "mean_interblock_gain_vs_random_greedy": x["aggregate_vs_random_cpsat"]["mean_interblock_gain_vs_random_greedy"],
            "mean_support_range_gain_vs_random_greedy": x["aggregate_vs_random_cpsat"]["mean_support_range_gain_vs_random_greedy"],
            "mean_map_time_sec": x["aggregate_vs_random_cpsat"]["mean_map_time_sec"],
        }),
    )[0]

    payload = {
        "phase": 4,
        "circuits": circuits,
        "topologies": topologies,
        "seed": seed,
        "cp_time": cp_time,
        "random_cpsat_baselines": random_cpsat,
        "aggregates": aggregates,
        "best_cpsat_config": best,
    }
    _save_summary("phase4_cpsat_replay", payload)
    _log(f"phase 4 done: best CP-SAT config = {best['label']}")
    return payload


def phase5_stress_test(
    *,
    circuits: List[str],
    topologies: List[str],
    seeds: List[int],
    force: bool,
    score_kwargs: Dict[str, float],
    sa_params: Dict[str, float],
) -> Dict[str, Any]:
    _log(
        f"phase 5 start: stress-testing final greedy config over {len(seeds)} seeds, "
        f"{len(circuits)} circuits x {len(topologies)} topologies"
    )
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        for circuit, topology in _cases(circuits, topologies):
            records.append(_run_one(
                circuit,
                "sa_v2",
                phase="5stress",
                label=f"stress_seed{seed}",
                topology=topology,
                seed=seed,
                force=force,
                score_kwargs=score_kwargs,
                sa_steps=int(sa_params["steps"]),
                sa_t0=sa_params["t0"],
                sa_tend=sa_params["t_end"],
            ))
    _plot_stress(records, os.path.join(FIG_DIR, "fig_06_phase5_stress.png"))
    payload = {
        "phase": 5,
        "circuits": circuits,
        "topologies": topologies,
        "seeds": seeds,
        "score_kwargs_v2": score_kwargs,
        "sa_params": sa_params,
        "records": records,
    }
    _save_summary("phase5_stress_test", payload)
    _log("phase 5 done: stress test summary saved")
    return payload


def _summary_path(name: str) -> str:
    return os.path.join(SUMMARY_DIR, f"{name}.json")


def _load_summary(name: str) -> Dict[str, Any]:
    path = _summary_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary not found: {path}")
    return _load_json(path)


def _coerce_circuits(
    arg_circuits: Optional[List[str]],
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    exclude_set = set(exclude or [])
    if arg_circuits:
        notes = {}
        usable = []
        for circuit in arg_circuits:
            if circuit in exclude_set:
                _log(f"[exclude] skipping {circuit} (--exclude)")
                continue
            try:
                notes[circuit] = _resolve_pbc_path(circuit)
                usable.append(circuit)
            except FileNotFoundError as exc:
                notes[circuit] = str(exc)
        if not usable:
            raise FileNotFoundError("None of the requested circuits have cached PBC files.")
        return usable, notes
    usable, notes = _discover_default_circuits()
    if exclude_set:
        removed = [c for c in usable if c in exclude_set]
        usable = [c for c in usable if c not in exclude_set]
        for c in removed:
            _log(f"[exclude] skipping {c} (--exclude)")
    return usable, notes


def _coerce_topologies(arg_topologies: Optional[List[str]]) -> List[str]:
    topologies = arg_topologies or list(DEFAULT_TOPOLOGIES)
    normalized = []
    for topology in topologies:
        topo = topology.strip().lower()
        if topo not in ("grid", "ring"):
            raise ValueError(f"Unsupported topology: {topology!r}. Expected 'grid' or 'ring'.")
        if topo not in normalized:
            normalized.append(topo)
    return normalized


def _baseline_lookup_from_summary(phase0: Dict[str, Any], circuits: List[str], topologies: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stored = phase0["baselines"]
    for topology in topologies:
        topo_rows = stored[topology]
        for circuit in circuits:
            lookup[(circuit, topology)] = topo_rows[circuit]
    return lookup


def run_all(
    *,
    circuits: List[str],
    topologies: List[str],
    circuit_notes: Dict[str, str],
    seed: int,
    stress_seeds: List[int],
    cp_time: Optional[float],
    top_k: int,
    force: bool,
) -> Dict[str, Any]:
    p0 = phase0_baselines(circuits=circuits, topologies=topologies, seed=seed, force=force)
    baselines = _baseline_lookup_from_summary(p0, circuits, topologies)

    p1 = phase1_screen_weights(
        circuits=circuits,
        topologies=topologies,
        seed=seed,
        force=force,
        baselines=baselines,
    )
    p2 = phase2_select_weights(
        circuits=circuits,
        topologies=topologies,
        seed=seed,
        force=force,
        baselines=baselines,
        screening=p1,
    )
    p3 = phase3_tune_hyperparams(
        circuits=circuits,
        topologies=topologies,
        seed=seed,
        force=force,
        baselines=baselines,
        score_kwargs=p2["final_score_kwargs_v2"],
    )
    p4 = phase4_cpsat_replay(
        circuits=circuits,
        topologies=topologies,
        seed=seed,
        force=force,
        top_configs=p3["top_configs"][:top_k],
        cp_time=cp_time,
    )
    p5 = phase5_stress_test(
        circuits=circuits,
        topologies=topologies,
        seeds=stress_seeds,
        force=force,
        score_kwargs=p2["final_score_kwargs_v2"],
        sa_params=p3["final_sa_params"],
    )

    final_choice = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "circuits": circuits,
        "topologies": topologies,
        "circuit_pbc_notes": circuit_notes,
        "weight_descriptions": WEIGHT_DESCRIPTIONS,
        "selected_score_kwargs_v2": p2["final_score_kwargs_v2"],
        "selected_sa_params": p3["final_sa_params"],
        "ignored_low_impact_variables": p1["ignored_variables"],
        "weight_screening_order": [r["weight_name"] for r in p1["weight_rows"]],
        "greedy_final_aggregate": p3["final_aggregate"],
        "cpsat_best_config": p4["best_cpsat_config"],
        "cp_sat_time_limit": cp_time,
        "stress_seed_count": len(stress_seeds),
        "supporting_summaries": {
            "phase0": _summary_path("phase0_baselines"),
            "phase1": _summary_path("phase1_screen_weights"),
            "phase2": _summary_path("phase2_select_weights"),
            "phase3": _summary_path("phase3_tune_hyperparams"),
            "phase4": _summary_path("phase4_cpsat_replay"),
            "phase5": _summary_path("phase5_stress_test"),
        },
    }
    _save_summary("final_choice", final_choice)
    return final_choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Global SA-v2 tuning over cached PBC circuits")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "0", "1", "2", "3", "4", "5"],
        help="Run one phase or the full workflow",
    )
    parser.add_argument(
        "--circuits",
        nargs="*",
        help="Optional circuit list. Defaults to all cached PBC circuits found in circuits/benchmarks/pbc/.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        metavar="CIRCUIT",
        help="Circuit names to exclude from the run (e.g. --exclude rand_100_10k). "
             "Existing cached raw results for excluded circuits are preserved and still "
             "used by analysis phases that load from raw/.",
    )
    parser.add_argument(
        "--topologies",
        nargs="*",
        help="Topology list to evaluate. Defaults to: grid ring",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Primary seed for tuning phases")
    parser.add_argument(
        "--stress-seeds",
        default="40,41,42,43,44",
        help="Comma-separated seed list for the stress phase",
    )
    parser.add_argument(
        "--cp-time",
        type=float,
        default=None,
        help="Optional CP-SAT time limit in seconds. Default: no limit.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many top greedy configs to replay with CP-SAT")
    parser.add_argument("--force", action="store_true", help="Recompute runs even if cached JSONs exist")
    args = parser.parse_args()

    _ensure_dirs()
    circuits, circuit_notes = _coerce_circuits(args.circuits, exclude=args.exclude)
    topologies = _coerce_topologies(args.topologies)
    stress_seeds = [int(x.strip()) for x in args.stress_seeds.split(",") if x.strip()]
    _log(
        f"setup: circuits={circuits} | topologies={topologies} | "
        f"seed={args.seed} | stress_seeds={stress_seeds}"
    )

    if args.phase == "all":
        _log("workflow start: running all phases to select SA-v2 score weights and hyperparameters")
        final_choice = run_all(
            circuits=circuits,
            topologies=topologies,
            circuit_notes=circuit_notes,
            seed=args.seed,
            stress_seeds=stress_seeds,
            cp_time=args.cp_time,
            top_k=args.top_k,
            force=args.force,
        )
        print("Final SA-v2 selection")
        print(json.dumps(final_choice, indent=2))
        return

    if args.phase == "0":
        _log("workflow start: phase 0 only")
        payload = phase0_baselines(circuits=circuits, topologies=topologies, seed=args.seed, force=args.force)
        print(json.dumps({
            "phase": 0,
            "circuits": circuits,
            "topologies": topologies,
            "sa_v2_default_aggregate": payload["sa_v2_default_aggregate"],
        }, indent=2))
        return

    phase0 = _load_summary("phase0_baselines")
    baselines = _baseline_lookup_from_summary(phase0, circuits, topologies)

    if args.phase == "1":
        _log("workflow start: phase 1 only")
        payload = phase1_screen_weights(
            circuits=circuits,
            topologies=topologies,
            seed=args.seed,
            force=args.force,
            baselines=baselines,
        )
        print(json.dumps({
            "phase": 1,
            "selected_variables": payload["selected_variables"],
            "ignored_variables": payload["ignored_variables"],
        }, indent=2))
        return

    phase1 = _load_summary("phase1_screen_weights")

    if args.phase == "2":
        _log("workflow start: phase 2 only")
        payload = phase2_select_weights(
            circuits=circuits,
            topologies=topologies,
            seed=args.seed,
            force=args.force,
            baselines=baselines,
            screening=phase1,
        )
        print(json.dumps({
            "phase": 2,
            "final_score_kwargs_v2": payload["final_score_kwargs_v2"],
            "final_aggregate": payload["final_aggregate"],
        }, indent=2))
        return

    phase2 = _load_summary("phase2_select_weights")

    if args.phase == "3":
        _log("workflow start: phase 3 only")
        payload = phase3_tune_hyperparams(
            circuits=circuits,
            topologies=topologies,
            seed=args.seed,
            force=args.force,
            baselines=baselines,
            score_kwargs=phase2["final_score_kwargs_v2"],
        )
        print(json.dumps({
            "phase": 3,
            "final_sa_params": payload["final_sa_params"],
            "final_aggregate": payload["final_aggregate"],
        }, indent=2))
        return

    phase3 = _load_summary("phase3_tune_hyperparams")

    if args.phase == "4":
        _log("workflow start: phase 4 only")
        payload = phase4_cpsat_replay(
            circuits=circuits,
            topologies=topologies,
            seed=args.seed,
            force=args.force,
            top_configs=phase3["top_configs"][:args.top_k],
            cp_time=args.cp_time,
        )
        print(json.dumps({
            "phase": 4,
            "best_cpsat_config": payload["best_cpsat_config"],
        }, indent=2))
        return

    if args.phase == "5":
        _log("workflow start: phase 5 only")
        payload = phase5_stress_test(
            circuits=circuits,
            topologies=topologies,
            seeds=stress_seeds,
            force=args.force,
            score_kwargs=phase2["final_score_kwargs_v2"],
            sa_params=phase3["final_sa_params"],
        )
        print(json.dumps({
            "phase": 5,
            "circuits": circuits,
            "seed_count": len(stress_seeds),
            "summary_path": _summary_path("phase5_stress_test"),
        }, indent=2))
        return


if __name__ == "__main__":
    main()
