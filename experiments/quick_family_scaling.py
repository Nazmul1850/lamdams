"""
Fast family-level SA scaling tuner for small random/gf circuit sets.

Goal
----
Find practical SA score scaling factors with a small run budget by using:
  - random + greedy_critical as baseline
  - SA(v1) + greedy_critical for candidate presets
  - two-stage search: pilot (representatives) -> full (family circuits)

Default families
----------------
random_small:
  rand_50q_500t_s42, rand_50q_1kt_s42, rand_50q_1500t_s42
gf_small:
  gf6_mult, gf7_mult, gf8_mult, gf9_mult

Outputs
-------
results/family_scaling/
  raw/      one json per run
  summary/  pilot and final family recommendations

Usage
-----
  # Fast grid-only run (recommended first)
  ./.venv/bin/python experiments/quick_family_scaling.py --topologies grid

  # Grid + ring
  ./.venv/bin/python experiments/quick_family_scaling.py --topologies grid ring
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_LSQECC_SRC = os.path.join(_ROOT, "third_party", "lsqecc", "src")
if os.path.isdir(_LSQECC_SRC):
    sys.path.insert(0, _LSQECC_SRC)

from modqldpc.frontend.extract_pauli import GoSCConverter
from modqldpc.mapping.algos.sa_mapping import TUNED_SCORE_KWARGS, _score
from modqldpc.mapping.hardware_gen import make_hardware
from modqldpc.mapping.mapper import MappingConfig, MappingProblem, get_mapper

from experiments.sensitivity_analysis_v2 import _compute_greedy_depth, _count_inter_block


PBC_DIR = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
OUT_ROOT = os.path.join(_ROOT, "results", "family_scaling")
RAW_DIR = os.path.join(OUT_ROOT, "raw")
SUMMARY_DIR = os.path.join(OUT_ROOT, "summary")

N_DATA = 11
DEFAULT_SEED = 42

RANDOM_SMALL = [
    "rand_50q_500t_s42",
    "rand_50q_1kt_s42",
    "rand_50q_1500t_s42",
]
GF_SMALL = [
    "gf6_mult",
    "gf7_mult",
    "gf8_mult",
    "gf9_mult",
]

FAMILIES: Dict[str, List[str]] = {
    "random_small": RANDOM_SMALL,
    "gf_small": GF_SMALL,
}

PILOT_REP: Dict[str, str] = {
    "random_small": "rand_50q_1kt_s42",
    "gf_small": "gf8_mult",
}


BASE_KW = dict(TUNED_SCORE_KWARGS)

# Small, intuition-driven candidate presets.
# Multipliers apply to BASE_KW values.
PROFILE_SCALES: Dict[str, Dict[str, float]] = {
    "base": {},
    "balance_mild": {
        "W_OCC_RANGE": 1.8,
        "W_OCC_STD": 2.0,
        "W_SUPPORT_PEAK": 1.8,
        "W_SUPPORT_RANGE": 6.0,
        "W_SPLIT": 1.5,
        "W_SPAN": 0.85,
        "W_MST": 0.85,
    },
    "balance_strong": {
        "W_OCC_RANGE": 2.5,
        "W_OCC_STD": 3.0,
        "W_SUPPORT_PEAK": 2.5,
        "W_SUPPORT_RANGE": 12.0,
        "W_SPLIT": 2.0,
        "W_SPAN": 0.65,
        "W_MST": 0.65,
    },
    "locality_mild": {
        "W_SPAN": 1.3,
        "W_MST": 1.4,
        "W_OCC_RANGE": 0.9,
        "W_OCC_STD": 0.9,
        "W_SUPPORT_RANGE": 0.9,
    },
    "locality_light_balance": {
        "W_SPAN": 0.45,
        "W_MST": 0.55,
        "W_OCC_RANGE": 2.2,
        "W_OCC_STD": 2.2,
        "W_SUPPORT_PEAK": 2.0,
        "W_SUPPORT_RANGE": 8.0,
        "W_SPLIT": 1.5,
    },
}

FAMILY_PROFILES: Dict[str, List[str]] = {
    "random_small": ["base", "balance_mild", "balance_strong", "locality_light_balance"],
    "gf_small": ["base", "balance_mild", "locality_light_balance", "locality_mild"],
}


@dataclass(frozen=True)
class SAParams:
    steps: int
    t0: float
    tend: float


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[family_scaling {now}] {msg}", flush=True)


def _ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def _slug(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("=", "-")
        .replace(".", "p")
    )


def _config_sig(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _resolve_pbc(circuit: str) -> str:
    candidates = [
        os.path.join(PBC_DIR, f"{circuit}.json"),
        os.path.join(PBC_DIR, f"{circuit}_PBC.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No cached PBC found for {circuit!r}. Checked: {candidates}"
    )


def _load_pbc(circuit: str) -> Tuple[str, int, list, Any]:
    pbc_path = _resolve_pbc(circuit)
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(pbc_path)
    rots = list(conv.program.rotations)
    n_logicals = len(rots[0].axis.lstrip("+-"))
    return pbc_path, n_logicals, rots, conv


def _scaled_kwargs(scale_map: Dict[str, float]) -> Dict[str, float]:
    out = dict(BASE_KW)
    for key, mult in scale_map.items():
        out[key] = BASE_KW[key] * mult
    return out


def _case_path(
    *,
    circuit: str,
    topology: str,
    mapper: str,
    profile: str,
    seed: int,
    sa: SAParams,
    score_kwargs: Optional[Dict[str, float]],
) -> str:
    payload = {
        "circuit": circuit,
        "topology": topology,
        "mapper": mapper,
        "profile": profile,
        "seed": seed,
        "sa_steps": sa.steps,
        "sa_t0": sa.t0,
        "sa_tend": sa.tend,
        "score_kwargs": score_kwargs or {},
    }
    sig = _config_sig(payload)
    name = f"{circuit}_{topology}_{mapper}_{_slug(profile)}_{sig}.json"
    return os.path.join(RAW_DIR, name)


def _record_metrics(
    *,
    score,
    hw,
) -> Dict[str, Any]:
    return {
        "score_total": score.total,
        "active_blocks": score.active_blocks,
        "unused_blocks": score.unused_blocks,
        "occupancy_range": score.occupancy_range,
        "occupancy_std": score.occupancy_std,
        "num_multiblock": score.num_multiblock,
        "span_total": score.span_total,
        "mst_total": score.mst_total,
        "split_total": score.split_total,
        "support_peak": score.support_peak,
        "support_range": score.support_range,
        "n_blocks": len(hw.blocks),
    }


def _run_case(
    *,
    circuit: str,
    topology: str,
    mapper: str,
    profile: str,
    seed: int,
    sa: SAParams,
    score_kwargs: Optional[Dict[str, float]],
    force: bool,
) -> Dict[str, Any]:
    path = _case_path(
        circuit=circuit,
        topology=topology,
        mapper=mapper,
        profile=profile,
        seed=seed,
        sa=sa,
        score_kwargs=score_kwargs,
    )
    if not force and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    pbc_path, n_logicals, rotations, conv = _load_pbc(circuit)
    hw, hw_spec = make_hardware(
        n_logicals,
        topology=topology,
        sparse_pct=0.0,
        n_data=N_DATA,
        coupler_capacity=1,
    )

    t_map_0 = time.perf_counter()
    if mapper == "random":
        plan = get_mapper("pure_random").solve(
            MappingProblem(n_logicals=n_logicals),
            hw,
            MappingConfig(seed=seed),
        )
    elif mapper == "sa":
        plan = get_mapper("simulated_annealing").solve(
            MappingProblem(n_logicals=n_logicals),
            hw,
            MappingConfig(seed=seed, sa_steps=sa.steps, sa_t0=sa.t0, sa_tend=sa.tend),
            {
                "rotations": rotations,
                "verbose": False,
                "score_kwargs": score_kwargs or {},
            },
        )
    else:
        raise ValueError(f"Unknown mapper {mapper!r}")
    map_time = time.perf_counter() - t_map_0

    # Metrics (computed with candidate score kwargs to match SA objective view).
    metrics_score = _score(rotations, hw, **(score_kwargs or {}))
    inter_block_all = _count_inter_block(rotations, plan, t_only=False)
    inter_block_t = _count_inter_block(rotations, plan, t_only=True)

    t_sched_0 = time.perf_counter()
    logical_depth, layer_depths = _compute_greedy_depth(
        conv,
        hw,
        plan,
        seed=seed,
        n_data=N_DATA,
        scheduler_name="greedy_critical",
        cp_sat_time_limit=120.0,
    )
    sched_time = time.perf_counter() - t_sched_0

    rec = {
        "circuit": circuit,
        "topology": topology,
        "mapper": mapper,
        "profile": profile,
        "seed": seed,
        "pbc_path": pbc_path,
        "hardware_label": hw_spec.label(),
        "n_logicals": n_logicals,
        "logical_depth": logical_depth,
        "layer_depths": layer_depths,
        "inter_block_all": inter_block_all,
        "inter_block_t_only": inter_block_t,
        "metrics": _record_metrics(score=metrics_score, hw=hw),
        "timing_sec": {
            "map": round(map_time, 3),
            "schedule": round(sched_time, 3),
            "total": round(map_time + sched_time, 3),
        },
        "sa_params": {"steps": sa.steps, "t0": sa.t0, "tend": sa.tend},
        "score_kwargs": score_kwargs or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def _safe_ratio(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def _pair_gain(sa_rec: Dict[str, Any], rnd_rec: Dict[str, Any]) -> Dict[str, float]:
    depth_gain = _safe_ratio(
        rnd_rec["logical_depth"] - sa_rec["logical_depth"],
        rnd_rec["logical_depth"],
    )
    inter_gain = _safe_ratio(
        rnd_rec["inter_block_all"] - sa_rec["inter_block_all"],
        rnd_rec["inter_block_all"],
    )
    support_range_gain = _safe_ratio(
        rnd_rec["metrics"]["support_range"] - sa_rec["metrics"]["support_range"],
        rnd_rec["metrics"]["support_range"],
    )
    # Depth dominates. Inter-block and support-range are tie-breakers.
    rank_score = depth_gain + 0.15 * inter_gain + 0.10 * support_range_gain
    return {
        "depth_gain_vs_random": depth_gain,
        "inter_gain_vs_random": inter_gain,
        "support_range_gain_vs_random": support_range_gain,
        "rank_score": rank_score,
    }


def _aggregate_candidate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_cases": 0,
            "mean_depth_gain_vs_random": 0.0,
            "mean_inter_gain_vs_random": 0.0,
            "mean_support_range_gain_vs_random": 0.0,
            "mean_rank_score": 0.0,
            "mean_depth": 0.0,
            "mean_random_depth": 0.0,
            "mean_map_time_sec": 0.0,
        }
    n = len(rows)
    return {
        "n_cases": n,
        "mean_depth_gain_vs_random": sum(r["depth_gain_vs_random"] for r in rows) / n,
        "mean_inter_gain_vs_random": sum(r["inter_gain_vs_random"] for r in rows) / n,
        "mean_support_range_gain_vs_random": sum(r["support_range_gain_vs_random"] for r in rows) / n,
        "mean_rank_score": sum(r["rank_score"] for r in rows) / n,
        "mean_depth": sum(r["sa_depth"] for r in rows) / n,
        "mean_random_depth": sum(r["random_depth"] for r in rows) / n,
        "mean_map_time_sec": sum(r["sa_map_time_sec"] for r in rows) / n,
    }


def _rank_profiles(profile_rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for profile, rows in profile_rows.items():
        out.append({
            "profile": profile,
            "aggregate": _aggregate_candidate(rows),
        })
    return sorted(
        out,
        key=lambda x: (
            -x["aggregate"]["mean_rank_score"],
            -x["aggregate"]["mean_depth_gain_vs_random"],
            -x["aggregate"]["mean_inter_gain_vs_random"],
            x["aggregate"]["mean_map_time_sec"],
        ),
    )


def _save_summary(name: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_family_stage(
    *,
    stage_name: str,
    family: str,
    circuits: List[str],
    profiles: List[str],
    topologies: List[str],
    seed: int,
    sa: SAParams,
    force: bool,
) -> Dict[str, Any]:
    _log(
        f"{stage_name}: family={family} circuits={circuits} profiles={profiles} "
        f"topologies={topologies}"
    )
    by_profile: Dict[str, List[Dict[str, Any]]] = {p: [] for p in profiles}
    raw_rows: List[Dict[str, Any]] = []

    for topology in topologies:
        for circuit in circuits:
            rnd = _run_case(
                circuit=circuit,
                topology=topology,
                mapper="random",
                profile="random_baseline",
                seed=seed,
                sa=sa,
                score_kwargs=None,
                force=force,
            )
            for profile in profiles:
                score_kwargs = _scaled_kwargs(PROFILE_SCALES[profile])
                sa_rec = _run_case(
                    circuit=circuit,
                    topology=topology,
                    mapper="sa",
                    profile=profile,
                    seed=seed,
                    sa=sa,
                    score_kwargs=score_kwargs,
                    force=force,
                )
                gain = _pair_gain(sa_rec, rnd)
                row = {
                    "family": family,
                    "stage": stage_name,
                    "profile": profile,
                    "circuit": circuit,
                    "topology": topology,
                    "random_depth": rnd["logical_depth"],
                    "sa_depth": sa_rec["logical_depth"],
                    "random_inter": rnd["inter_block_all"],
                    "sa_inter": sa_rec["inter_block_all"],
                    "sa_map_time_sec": sa_rec["timing_sec"]["map"],
                    "score_kwargs": score_kwargs,
                    **gain,
                }
                by_profile[profile].append(row)
                raw_rows.append(row)
                _log(
                    f"{stage_name} {family} {circuit}[{topology}] {profile}: "
                    f"depth {rnd['logical_depth']} -> {sa_rec['logical_depth']} "
                    f"({gain['depth_gain_vs_random']*100:+.2f}%)"
                )

    ranked = _rank_profiles(by_profile)
    return {
        "stage": stage_name,
        "family": family,
        "circuits": circuits,
        "profiles": profiles,
        "topologies": topologies,
        "ranked_profiles": ranked,
        "rows": raw_rows,
    }


def run_quick_tuning(
    *,
    families: List[str],
    topologies: List[str],
    seed: int,
    sa: SAParams,
    top_k: int,
    force: bool,
    out_name: str = "_auto",
) -> Tuple[Dict[str, Any], str]:
    _ensure_dirs()
    _log(
        f"start: families={families} topologies={topologies} seed={seed} "
        f"sa=({sa.steps},{sa.t0},{sa.tend})"
    )

    pilot_results: Dict[str, Dict[str, Any]] = {}
    full_results: Dict[str, Dict[str, Any]] = {}
    recommendations: Dict[str, Dict[str, Any]] = {}

    for family in families:
        rep = PILOT_REP[family]
        pilot = _run_family_stage(
            stage_name="pilot",
            family=family,
            circuits=[rep],
            profiles=FAMILY_PROFILES[family],
            topologies=topologies,
            seed=seed,
            sa=sa,
            force=force,
        )
        pilot_results[family] = pilot

        top_profiles = [x["profile"] for x in pilot["ranked_profiles"][:top_k]]
        if "base" not in top_profiles:
            top_profiles = ["base"] + top_profiles
        top_profiles = list(dict.fromkeys(top_profiles))

        full = _run_family_stage(
            stage_name="full",
            family=family,
            circuits=FAMILIES[family],
            profiles=top_profiles,
            topologies=topologies,
            seed=seed,
            sa=sa,
            force=force,
        )
        full_results[family] = full

        best = full["ranked_profiles"][0]
        profile_name = best["profile"]
        recommendations[family] = {
            "best_profile": profile_name,
            "best_profile_scales": PROFILE_SCALES[profile_name],
            "best_score_kwargs": _scaled_kwargs(PROFILE_SCALES[profile_name]),
            "aggregate": best["aggregate"],
            "tested_profiles": top_profiles,
            "pilot_top_profiles": [x["profile"] for x in pilot["ranked_profiles"][:top_k]],
        }
        _log(
            f"family={family} best={profile_name} "
            f"mean_depth_gain={best['aggregate']['mean_depth_gain_vs_random']*100:.2f}%"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "topologies": topologies,
        "families": families,
        "sa_params": {"steps": sa.steps, "t0": sa.t0, "tend": sa.tend},
        "base_kwargs_v1": BASE_KW,
        "profile_scales": PROFILE_SCALES,
        "pilot_results": pilot_results,
        "full_results": full_results,
        "recommendations": recommendations,
        "notes": (
            "Ranking score = depth_gain + 0.15*inter_gain + 0.10*support_range_gain; "
            "depth gain is primary."
        ),
    }
    # Auto-name summary: single family → "<family>_scaling", multiple → "family_scaling"
    if out_name == "_auto":
        out_name = f"{families[0]}_scaling" if len(families) == 1 else "family_scaling"
    _save_summary(out_name, payload)
    return payload, out_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Family-level SA scaling tuner")
    parser.add_argument(
        "--families",
        nargs="*",
        default=["random_small", "gf_small"],
        choices=["random_small", "gf_small"],
        help="Circuit families to tune",
    )
    parser.add_argument(
        "--topologies",
        nargs="*",
        default=["grid"],
        choices=["grid", "ring"],
        help="Topologies to evaluate (default grid for speed)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sa-steps", type=int, default=12_000, help="SA steps per run")
    parser.add_argument("--sa-t0", type=float, default=1e5)
    parser.add_argument("--sa-tend", type=float, default=0.05)
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top pilot profiles per family to promote into full stage",
    )
    parser.add_argument("--out-name", type=str, default="_auto",
                        help="Summary filename stem (auto: '<family>_scaling')")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sa = SAParams(steps=args.sa_steps, t0=args.sa_t0, tend=args.sa_tend)
    payload, out_name = run_quick_tuning(
        families=args.families,
        topologies=args.topologies,
        seed=args.seed,
        sa=sa,
        top_k=args.top_k,
        force=args.force,
        out_name=args.out_name,
    )

    print(json.dumps({
        "summary": os.path.join(SUMMARY_DIR, f"{out_name}.json"),
        "recommendations": payload["recommendations"],
    }, indent=2))


if __name__ == "__main__":
    main()
