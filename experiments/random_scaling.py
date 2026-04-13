"""
Random-family SA scaling tuner  (rand_50q_500t, rand_50q_1kt, rand_50q_1500t)

Background
----------
Random 50q circuits have sparse, uniformly distributed Pauli support.
Each rotation typically touches only 1-2 blocks so W_SPAN is less pathologically
dominant than in GF circuits (span_total is lower per rotation).  SA can
genuinely benefit from locality pressure here — clustering related qubits
reduces multi-block spans that do exist.

Score dominance (random circuits, rough estimate):
  span_pen   ≈ 10,000 × span_total  (dominant, but span_total is smaller)
  mst_pen    ≈    500 × mst_total
  occ terms  ≈ 65,000               (meaningful — balance matters for scheduling)

Strategy
--------
Four profiles spanning the balance ↔ locality trade-off:

  base                   — unchanged (reference)
  balance_mild           — gentle occupancy push, slight locality reduction
  locality_light_balance — span-halved, moderate balance (best of both)
  locality_mild          — span/MST boosted, slight balance reduction

Pilot circuit: rand_50q_1kt_s42 (mid-size representative)
Full family  : rand_50q_500t_s42, rand_50q_1kt_s42, rand_50q_1500t_s42

Usage
-----
  ./.venv/bin/python experiments/random_scaling.py
  ./.venv/bin/python experiments/random_scaling.py --sa-steps 18000
  ./.venv/bin/python experiments/random_scaling.py --topologies grid ring

Outputs → results/family_scaling/summary/random_small_scaling.json
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


PBC_DIR     = os.path.join(_ROOT, "circuits", "benchmarks", "pbc")
OUT_ROOT    = os.path.join(_ROOT, "results", "family_scaling")
RAW_DIR     = os.path.join(OUT_ROOT, "raw")
SUMMARY_DIR = os.path.join(OUT_ROOT, "summary")

N_DATA       = 11
DEFAULT_SEED = 42
FAMILY       = "random_small"
OUT_NAME     = "random_small_scaling"
PILOT_REP    = "rand_50q_1kt_s42"

CIRCUITS = [
    "rand_50q_500t_s42",
    "rand_50q_1kt_s42",
    "rand_50q_1500t_s42",
]

BASE_KW = dict(TUNED_SCORE_KWARGS)

# ------------------------------------------------------------------ #
# Profiles — informed by random_sensitivity.py sweep on               #
# rand_50q_500t_s42 (both topologies).                                #
#                                                                      #
# Key finding: span dominates BASE_KW for random circuits too.        #
# locality_light_balance (W_SPAN×0.45, W_MST×0.55) was best at -2.2% #
# Sensitivity sweep probes MST boost, split boost, and combos.        #
# PILOT_PROFILES lists the top candidates from that sweep.            #
# Update PILOT_PROFILES once random_sensitivity.py results are known. #
# ------------------------------------------------------------------ #
PROFILES: Dict[str, Dict[str, float]] = {
    # ── Reference ──────────────────────────────────────────────────
    "base": {},

    # ── Prior best (before sensitivity sweep) ──────────────────────
    "locality_light_balance": {          # -2.2% on 1kt/grid
        "W_SPAN":  0.45, "W_MST":  0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.2,
        "W_SUPPORT_PEAK": 2.0, "W_SUPPORT_RANGE": 8.0, "W_SPLIT": 1.5,
    },

    # ── Sensitivity sweep candidates (from random_sensitivity.py) ──

    # Group A — span screening (W_MST×0.55, mild balance)
    "span_x010": {
        "W_SPAN": 0.10, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x020": {
        "W_SPAN": 0.20, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x035": {
        "W_SPAN": 0.35, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },

    # Group B — MST boost at W_SPAN×0.45
    "mst_2x": {
        "W_SPAN": 0.45, "W_MST": 2.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "mst_4x": {
        "W_SPAN": 0.45, "W_MST": 4.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "mst_7x": {
        "W_SPAN": 0.45, "W_MST": 7.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },

    # Group C — split boost at (W_SPAN×0.45, W_MST×0.55)
    "split_x10": {
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0, "W_SPLIT": 10.0,
    },
    "split_x20": {
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0, "W_SPLIT": 20.0,
    },

    # Group D — combined best-guess combos
    "locality_strong": {
        "W_SPAN": 0.25, "W_MST": 3.0,
        "W_OCC_RANGE": 2.0, "W_OCC_STD": 1.8, "W_SPLIT": 5.0,
    },
    "locality_balanced": {
        "W_SPAN": 0.35, "W_MST": 3.5,
        "W_OCC_RANGE": 2.5, "W_OCC_STD": 2.0, "W_SPLIT": 8.0,
    },
    "all_locality": {
        "W_SPAN": 0.20, "W_MST": 5.0,
        "W_OCC_RANGE": 2.0, "W_OCC_STD": 1.8, "W_SPLIT": 10.0,
    },
    "all_in": {
        "W_SPAN": 0.15, "W_MST": 4.0,
        "W_OCC_RANGE": 3.0, "W_OCC_STD": 2.5, "W_SPLIT": 12.0,
    },
}

# Update PILOT_PROFILES after running random_sensitivity.py —
# replace with top-4 from that sweep's ranking.
PILOT_PROFILES = [
    "base",
    "locality_light_balance",  # current best
    "locality_balanced",       # combo candidate
    "all_locality",            # strong locality + split
]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class SAParams:
    steps: int
    t0: float
    tend: float


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[random_scaling {now}] {msg}", flush=True)


def _ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def _slug(text: str) -> str:
    return (
        text.replace(" ", "_").replace("/", "_").replace("\\", "_")
            .replace(":", "_").replace("=", "-").replace(".", "p")
    )


def _config_sig(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _resolve_pbc(circuit: str) -> str:
    candidates = [
        os.path.join(PBC_DIR, f"{circuit}.json"),
        os.path.join(PBC_DIR, f"{circuit}_PBC.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No PBC found for {circuit!r}. Checked: {candidates}")


def _load_pbc(circuit: str) -> Tuple[str, int, list, Any]:
    path = _resolve_pbc(circuit)
    conv = GoSCConverter(verbose=False)
    conv.load_cache_json(path)
    rots = list(conv.program.rotations)
    n_logicals = len(rots[0].axis.lstrip("+-"))
    return path, n_logicals, rots, conv


def _scaled_kwargs(scale_map: Dict[str, float]) -> Dict[str, float]:
    out = dict(BASE_KW)
    for key, mult in scale_map.items():
        out[key] = BASE_KW[key] * mult
    return out


def _case_path(
    *, circuit: str, topology: str, mapper: str, profile: str,
    seed: int, sa: SAParams, score_kwargs: Optional[Dict[str, float]],
) -> str:
    sig = _config_sig({
        "circuit": circuit, "topology": topology, "mapper": mapper,
        "profile": profile, "seed": seed, "sa_steps": sa.steps,
        "sa_t0": sa.t0, "sa_tend": sa.tend, "score_kwargs": score_kwargs or {},
    })
    return os.path.join(RAW_DIR, f"{circuit}_{topology}_{mapper}_{_slug(profile)}_{sig}.json")


def _record_metrics(*, score, hw) -> Dict[str, Any]:
    return {
        "score_total":      score.total,
        "active_blocks":    score.active_blocks,
        "unused_blocks":    score.unused_blocks,
        "occupancy_range":  score.occupancy_range,
        "occupancy_std":    score.occupancy_std,
        "num_multiblock":   score.num_multiblock,
        "span_total":       score.span_total,
        "mst_total":        score.mst_total,
        "split_total":      score.split_total,
        "support_peak":     score.support_peak,
        "support_range":    score.support_range,
        "n_blocks":         len(hw.blocks),
    }


def _find_cached(circuit: str, topology: str, mapper: str, profile: str) -> Optional[str]:
    """Return path of any existing raw file for this (circuit, topology, mapper, profile)."""
    prefix = f"{circuit}_{topology}_{mapper}_{_slug(profile)}_"
    for fname in os.listdir(RAW_DIR):
        if fname.startswith(prefix) and fname.endswith(".json"):
            return os.path.join(RAW_DIR, fname)
    return None


def _run_case(
    *, circuit: str, topology: str, mapper: str, profile: str,
    seed: int, sa: SAParams, score_kwargs: Optional[Dict[str, float]], force: bool,
) -> Dict[str, Any]:
    path = _case_path(
        circuit=circuit, topology=topology, mapper=mapper, profile=profile,
        seed=seed, sa=sa, score_kwargs=score_kwargs,
    )
    if not force:
        # Exact hash match first, then prefix scan (handles sa_steps changes)
        hit = path if os.path.exists(path) else _find_cached(circuit, topology, mapper, profile)
        if hit:
            _log(f"  [skip] {os.path.basename(hit)}")
            with open(hit, "r", encoding="utf-8") as f:
                return json.load(f)

    pbc_path, n_logicals, rotations, conv = _load_pbc(circuit)
    hw, hw_spec = make_hardware(
        n_logicals, topology=topology, sparse_pct=0.0, n_data=N_DATA, coupler_capacity=1,
    )

    t0 = time.perf_counter()
    if mapper == "random":
        plan = get_mapper("pure_random").solve(
            MappingProblem(n_logicals=n_logicals), hw, MappingConfig(seed=seed),
        )
    elif mapper == "sa":
        plan = get_mapper("simulated_annealing").solve(
            MappingProblem(n_logicals=n_logicals), hw,
            MappingConfig(seed=seed, sa_steps=sa.steps, sa_t0=sa.t0, sa_tend=sa.tend),
            {"rotations": rotations, "verbose": False, "score_kwargs": score_kwargs or {}},
        )
    else:
        raise ValueError(f"Unknown mapper {mapper!r}")
    map_time = time.perf_counter() - t0

    metrics_score = _score(rotations, hw, **(score_kwargs or {}))
    inter_all = _count_inter_block(rotations, plan, t_only=False)
    inter_t   = _count_inter_block(rotations, plan, t_only=True)

    t1 = time.perf_counter()
    logical_depth, layer_depths = _compute_greedy_depth(
        conv, hw, plan, seed=seed, n_data=N_DATA,
        scheduler_name="greedy_critical", cp_sat_time_limit=120.0,
    )
    sched_time = time.perf_counter() - t1

    rec = {
        "circuit": circuit, "topology": topology, "mapper": mapper,
        "profile": profile, "seed": seed, "pbc_path": pbc_path,
        "hardware_label": hw_spec.label(), "n_logicals": n_logicals,
        "logical_depth": logical_depth, "layer_depths": layer_depths,
        "inter_block_all": inter_all, "inter_block_t_only": inter_t,
        "metrics": _record_metrics(score=metrics_score, hw=hw),
        "timing_sec": {
            "map": round(map_time, 3), "schedule": round(sched_time, 3),
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
    return a / b if abs(b) > 1e-12 else 0.0


def _pair_gain(sa_rec: Dict, rnd_rec: Dict) -> Dict[str, float]:
    depth_gain = _safe_ratio(rnd_rec["logical_depth"] - sa_rec["logical_depth"], rnd_rec["logical_depth"])
    inter_gain = _safe_ratio(rnd_rec["inter_block_all"] - sa_rec["inter_block_all"], rnd_rec["inter_block_all"])
    supr_gain  = _safe_ratio(
        rnd_rec["metrics"]["support_range"] - sa_rec["metrics"]["support_range"],
        rnd_rec["metrics"]["support_range"],
    )
    return {
        "depth_gain_vs_random":         depth_gain,
        "inter_gain_vs_random":         inter_gain,
        "support_range_gain_vs_random": supr_gain,
        "rank_score": depth_gain + 0.15 * inter_gain + 0.10 * supr_gain,
    }


def _aggregate(rows: List[Dict]) -> Dict[str, Any]:
    if not rows:
        return {"n_cases": 0, "mean_depth_gain_vs_random": 0.0, "mean_rank_score": 0.0,
                "mean_depth": 0.0, "mean_random_depth": 0.0, "mean_map_time_sec": 0.0}
    n = len(rows)
    return {
        "n_cases": n,
        "mean_depth_gain_vs_random":         sum(r["depth_gain_vs_random"]         for r in rows) / n,
        "mean_inter_gain_vs_random":         sum(r["inter_gain_vs_random"]         for r in rows) / n,
        "mean_support_range_gain_vs_random": sum(r["support_range_gain_vs_random"] for r in rows) / n,
        "mean_rank_score":                   sum(r["rank_score"]                   for r in rows) / n,
        "mean_depth":                        sum(r["sa_depth"]                     for r in rows) / n,
        "mean_random_depth":                 sum(r["random_depth"]                 for r in rows) / n,
        "mean_map_time_sec":                 sum(r["sa_map_time_sec"]              for r in rows) / n,
    }


def _rank_profiles(by_profile: Dict[str, List[Dict]]) -> List[Dict]:
    out = [{"profile": p, "aggregate": _aggregate(rows)} for p, rows in by_profile.items()]
    return sorted(out, key=lambda x: (
        -x["aggregate"]["mean_rank_score"],
        -x["aggregate"]["mean_depth_gain_vs_random"],
        x["aggregate"]["mean_map_time_sec"],
    ))


def _save_summary(name: str, payload: Dict) -> str:
    path = os.path.join(SUMMARY_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_stage(
    *, stage: str, circuits: List[str], profiles: List[str],
    topologies: List[str], seed: int, sa: SAParams, force: bool,
) -> Dict[str, Any]:
    _log(f"{stage}: circuits={circuits}  profiles={profiles}  topologies={topologies}")
    by_profile: Dict[str, List[Dict]] = {p: [] for p in profiles}
    rows: List[Dict] = []

    for topology in topologies:
        for circuit in circuits:
            rnd = _run_case(
                circuit=circuit, topology=topology, mapper="random",
                profile="random_baseline", seed=seed, sa=sa,
                score_kwargs=None, force=force,
            )
            for profile in profiles:
                skw = _scaled_kwargs(PROFILES[profile])
                sa_rec = _run_case(
                    circuit=circuit, topology=topology, mapper="sa",
                    profile=profile, seed=seed, sa=sa,
                    score_kwargs=skw, force=force,
                )
                gain = _pair_gain(sa_rec, rnd)
                row = {
                    "stage": stage, "profile": profile,
                    "circuit": circuit, "topology": topology,
                    "random_depth": rnd["logical_depth"],
                    "sa_depth":     sa_rec["logical_depth"],
                    "random_inter": rnd["inter_block_all"],
                    "sa_inter":     sa_rec["inter_block_all"],
                    "sa_map_time_sec": sa_rec["timing_sec"]["map"],
                    "score_kwargs": skw,
                    **gain,
                }
                by_profile[profile].append(row)
                rows.append(row)
                _log(
                    f"  [{stage}] {circuit}[{topology}] {profile}: "
                    f"depth {rnd['logical_depth']} → {sa_rec['logical_depth']} "
                    f"({gain['depth_gain_vs_random']*100:+.2f}%)"
                )

    return {"stage": stage, "circuits": circuits, "profiles": profiles,
            "ranked_profiles": _rank_profiles(by_profile), "rows": rows}


# ------------------------------------------------------------------ #
# Main tuning loop                                                     #
# ------------------------------------------------------------------ #

def run(*, topologies: List[str], seed: int, sa: SAParams, top_k: int, force: bool) -> Dict:
    _ensure_dirs()
    _log(f"Random scaling: circuits={CIRCUITS}  topologies={topologies}  sa=({sa.steps},{sa.t0},{sa.tend})")

    # Stage 1: pilot on representative (rand_50q_1kt_s42)
    pilot = _run_stage(
        stage="pilot", circuits=[PILOT_REP], profiles=PILOT_PROFILES,
        topologies=topologies, seed=seed, sa=sa, force=force,
    )
    _log("Pilot done. Rankings:")
    for r in pilot["ranked_profiles"]:
        _log(f"  {r['profile']:28s}  depth_gain={r['aggregate']['mean_depth_gain_vs_random']*100:+.2f}%  rank={r['aggregate']['mean_rank_score']:+.4f}")

    # Promote top-k + always keep base
    top_profiles = [x["profile"] for x in pilot["ranked_profiles"][:top_k]]
    if "base" not in top_profiles:
        top_profiles = ["base"] + top_profiles
    top_profiles = list(dict.fromkeys(top_profiles))
    _log(f"Promoted to full stage: {top_profiles}")

    # Stage 2: full family with promoted profiles
    full = _run_stage(
        stage="full", circuits=CIRCUITS, profiles=top_profiles,
        topologies=topologies, seed=seed, sa=sa, force=force,
    )

    best = full["ranked_profiles"][0]
    profile_name = best["profile"]
    recommendation = {
        "best_profile":        profile_name,
        "best_profile_scales": PROFILES[profile_name],
        "best_score_kwargs":   _scaled_kwargs(PROFILES[profile_name]),
        "aggregate":           best["aggregate"],
        "tested_profiles":     top_profiles,
        "pilot_top_profiles":  [x["profile"] for x in pilot["ranked_profiles"][:top_k]],
    }
    _log(
        f"BEST: {profile_name}  "
        f"mean_depth_gain={best['aggregate']['mean_depth_gain_vs_random']*100:.2f}%"
    )

    payload = {
        "family":        FAMILY,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "seed":          seed,
        "topologies":    topologies,
        "circuits":      CIRCUITS,
        "pilot_rep":     PILOT_REP,
        "sa_params":     {"steps": sa.steps, "t0": sa.t0, "tend": sa.tend},
        "base_kwargs":   BASE_KW,
        "profiles":      PROFILES,
        "pilot":         pilot,
        "full":          full,
        "recommendation": recommendation,
        "notes": (
            "Random 50q circuits: sparser support than GF, SA can benefit from "
            "locality pressure. locality_light_balance tests the balance/locality "
            "trade-off; locality_mild tests pure span clustering. "
            "Ranking: depth_gain + 0.15*inter_gain + 0.10*support_range_gain."
        ),
    }
    _save_summary(OUT_NAME, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-family SA scaling tuner")
    parser.add_argument("--topologies", nargs="*", default=["grid", "ring"], choices=["grid", "ring"])
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    parser.add_argument("--sa-steps", type=int,   default=12_000)
    parser.add_argument("--sa-t0",    type=float, default=1e5)
    parser.add_argument("--sa-tend",  type=float, default=0.05)
    parser.add_argument("--top-k",    type=int,   default=2,
                        help="Pilot profiles to promote to full stage")
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    sa = SAParams(steps=args.sa_steps, t0=args.sa_t0, tend=args.sa_tend)
    payload = run(topologies=args.topologies, seed=args.seed, sa=sa,
                  top_k=args.top_k, force=args.force)

    print(json.dumps({
        "summary":        os.path.join(SUMMARY_DIR, f"{OUT_NAME}.json"),
        "recommendation": payload["recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
