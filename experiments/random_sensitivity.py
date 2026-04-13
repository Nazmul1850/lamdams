"""
Focused sensitivity analysis for random circuits (rand_50q_500t_s42).

Problem
-------
SA+greedy with BASE_KW is WORSE than random+greedy for random circuits:
  rand_50q_1kt grid:  base=27,604 vs random=26,837 (+2.8% worse)
Best so far: locality_light_balance (W_SPAN×0.45, W_MST×0.55) → -2.2% gain.

Span dominance is real but less severe than GF. Key axes to explore:
  1. W_SPAN reduction — how far to push (×0.1 to ×0.7)?
  2. W_MST direction — currently reduced (×0.55); boosting may help locality
  3. W_SPLIT boost — support balance within touched blocks (base=10, very weak)
  4. W_OCC_RANGE — occupancy balance strength

Strategy
--------
Three targeted sweep groups on rand_50q_500t_s42 (both topologies):

  Group A — Span screening (W_MST fixed at ×0.55, mild balance):
    Explore W_SPAN from ×0.10 down to ×0.70 with fixed occupancy boost.

  Group B — MST direction (at best W_SPAN from A):
    Test W_MST from ×0.55 (current) up to ×7.0 — does spatial
    compactness help random circuits beyond just reducing span?

  Group C — Split + combined best guesses:
    W_SPLIT boosts and combo profiles targeting all three improvements.

Circuit:   rand_50q_500t_s42 (smallest/fastest random circuit)
Topologies: grid + ring
SA steps:   12,000 (fast screening; promote winner to 22,500 in random_scaling)

Outputs
-------
  results/family_scaling/raw/    — individual cached run JSONs
  results/family_scaling/summary/random_sensitivity.json — full ranking

Usage
-----
  ./.venv/bin/python experiments/random_sensitivity.py
  ./.venv/bin/python experiments/random_sensitivity.py --sa-steps 22500
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
CIRCUIT      = "rand_50q_500t_s42"
OUT_NAME     = "random_sensitivity"

BASE_KW = dict(TUNED_SCORE_KWARGS)
# BASE_KW reference (for comments below):
#   W_SPAN=10000, W_MST=500, W_OCC_RANGE=10000, W_OCC_STD=5000, W_SPLIT=10

# ------------------------------------------------------------------ #
# Sweep profiles                                                       #
#                                                                      #
# All values are MULTIPLIERS on BASE_KW.                               #
# Existing best: locality_light_balance at -2.2% depth gain.          #
# ------------------------------------------------------------------ #
PROFILES: Dict[str, Dict[str, float]] = {

    # ── Reference ──────────────────────────────────────────────────
    "base": {},

    # ── Group A: Span screening ─────────────────────────────────────
    # W_MST fixed ×0.55, W_OCC_RANGE×2.2 (mild balance), W_OCC_STD×2.0
    # Question: how low does W_SPAN need to go for random circuits?

    "span_x010": {                    # W_SPAN = 1,000
        "W_SPAN": 0.10, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x020": {                    # W_SPAN = 2,000
        "W_SPAN": 0.20, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x035": {                    # W_SPAN = 3,500
        "W_SPAN": 0.35, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x045": {                    # W_SPAN = 4,500 (current LLB)
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "span_x070": {                    # W_SPAN = 7,000
        "W_SPAN": 0.70, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },

    # ── Group B: MST direction at W_SPAN×0.45 ───────────────────────
    # Current LLB reduces MST (×0.55=275). Does boosting MST help?
    # Higher W_MST → SA prefers spatially compact multi-block spans.

    "mst_2x": {                       # W_MST = 1,000
        "W_SPAN": 0.45, "W_MST": 2.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "mst_4x": {                       # W_MST = 2,000
        "W_SPAN": 0.45, "W_MST": 4.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },
    "mst_7x": {                       # W_MST = 3,500
        "W_SPAN": 0.45, "W_MST": 7.0,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
    },

    # ── Group C: Split boost ────────────────────────────────────────
    # W_SPLIT base=10. Boosting drives SA to spread rotation support
    # evenly across touched blocks — improves parallelism.

    "split_x05": {                    # W_SPLIT = 50
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
        "W_SPLIT": 5.0,
    },
    "split_x10": {                    # W_SPLIT = 100
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
        "W_SPLIT": 10.0,
    },
    "split_x20": {                    # W_SPLIT = 200
        "W_SPAN": 0.45, "W_MST": 0.55,
        "W_OCC_RANGE": 2.2, "W_OCC_STD": 2.0,
        "W_SPLIT": 20.0,
    },

    # ── Group D: Combined best-guess combos ─────────────────────────
    # Low span + high MST + split boost

    "locality_strong": {              # span low, MST high, split moderate
        "W_SPAN": 0.25, "W_MST": 3.0,
        "W_OCC_RANGE": 2.0, "W_OCC_STD": 1.8,
        "W_SPLIT": 5.0,
    },
    "locality_balanced": {            # span moderate, MST high, split strong
        "W_SPAN": 0.35, "W_MST": 3.5,
        "W_OCC_RANGE": 2.5, "W_OCC_STD": 2.0,
        "W_SPLIT": 8.0,
    },
    "all_locality": {                 # max locality push, strong split
        "W_SPAN": 0.20, "W_MST": 5.0,
        "W_OCC_RANGE": 2.0, "W_OCC_STD": 1.8,
        "W_SPLIT": 10.0,
    },
    "all_in": {                       # strong everywhere
        "W_SPAN": 0.15, "W_MST": 4.0,
        "W_OCC_RANGE": 3.0, "W_OCC_STD": 2.5,
        "W_SPLIT": 12.0,
    },
}


# ------------------------------------------------------------------ #
# Helpers (identical pattern to gf_scaling / random_scaling)          #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class SAParams:
    steps: int
    t0: float
    tend: float


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[rand_sens {now}] {msg}", flush=True)


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
    for p in [os.path.join(PBC_DIR, f"{circuit}.json"),
              os.path.join(PBC_DIR, f"{circuit}_PBC.json")]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No PBC for {circuit!r}")


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


def _find_cached(circuit: str, topology: str, mapper: str, profile: str) -> Optional[str]:
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
        "metrics": {
            "score_total":     metrics_score.total,
            "active_blocks":   metrics_score.active_blocks,
            "unused_blocks":   metrics_score.unused_blocks,
            "occupancy_range": metrics_score.occupancy_range,
            "occupancy_std":   metrics_score.occupancy_std,
            "num_multiblock":  metrics_score.num_multiblock,
            "span_total":      metrics_score.span_total,
            "mst_total":       metrics_score.mst_total,
            "split_total":     metrics_score.split_total,
            "support_peak":    metrics_score.support_peak,
            "support_range":   metrics_score.support_range,
            "n_blocks":        len(hw.blocks),
        },
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


# ------------------------------------------------------------------ #
# Main sweep                                                           #
# ------------------------------------------------------------------ #

def run(*, topologies: List[str], seed: int, sa: SAParams, force: bool) -> Dict:
    _ensure_dirs()
    n_profiles = len(PROFILES)
    n_runs = (1 + n_profiles) * len(topologies)
    _log(
        f"rand sensitivity: circuit={CIRCUIT}  topologies={topologies}  "
        f"profiles={n_profiles}  total_runs={n_runs}  "
        f"sa=({sa.steps},{sa.t0},{sa.tend})"
    )

    baselines: Dict[str, Dict] = {}
    results: List[Dict] = []

    for topology in topologies:
        # Random baseline
        rnd = _run_case(
            circuit=CIRCUIT, topology=topology, mapper="random",
            profile="random_baseline", seed=seed, sa=sa,
            score_kwargs=None, force=force,
        )
        baselines[topology] = rnd
        _log(f"  baseline [{topology}] depth={rnd['logical_depth']}")

        for profile, scale_map in PROFILES.items():
            skw = _scaled_kwargs(scale_map)
            sa_rec = _run_case(
                circuit=CIRCUIT, topology=topology, mapper="sa",
                profile=profile, seed=seed, sa=sa,
                score_kwargs=skw, force=force,
            )
            rnd_depth = rnd["logical_depth"]
            sa_depth  = sa_rec["logical_depth"]
            depth_gain = _safe_ratio(rnd_depth - sa_depth, rnd_depth)
            inter_gain = _safe_ratio(
                rnd["inter_block_all"] - sa_rec["inter_block_all"],
                rnd["inter_block_all"],
            )
            rank_score = depth_gain + 0.15 * inter_gain

            results.append({
                "profile":          profile,
                "topology":         topology,
                "random_depth":     rnd_depth,
                "sa_depth":         sa_depth,
                "depth_gain":       depth_gain,
                "inter_gain":       inter_gain,
                "rank_score":       rank_score,
                "sa_map_time_sec":  sa_rec["timing_sec"]["map"],
                "score_kwargs":     skw,
                "scale_map":        scale_map,
                "span_total":       sa_rec["metrics"]["span_total"],
                "mst_total":        sa_rec["metrics"]["mst_total"],
                "split_total":      sa_rec["metrics"]["split_total"],
                "occ_range":        sa_rec["metrics"]["occupancy_range"],
            })
            _log(
                f"  [{topology}] {profile:28s}  "
                f"depth {rnd_depth} → {sa_depth} ({depth_gain*100:+.2f}%)"
            )

    # Aggregate across topologies
    by_profile: Dict[str, List[Dict]] = {}
    for r in results:
        by_profile.setdefault(r["profile"], []).append(r)

    ranked: List[Dict] = []
    for profile, rows in by_profile.items():
        n = len(rows)
        ranked.append({
            "profile": profile,
            "scale_map": rows[0]["scale_map"],
            "best_score_kwargs": rows[0]["score_kwargs"],
            "mean_depth_gain":   sum(r["depth_gain"] for r in rows) / n,
            "mean_inter_gain":   sum(r["inter_gain"] for r in rows) / n,
            "mean_rank_score":   sum(r["rank_score"] for r in rows) / n,
            "mean_sa_depth":     sum(r["sa_depth"] for r in rows) / n,
            "per_topology": {r["topology"]: {
                "random_depth": r["random_depth"],
                "sa_depth":     r["sa_depth"],
                "depth_gain":   r["depth_gain"],
            } for r in rows},
        })
    ranked.sort(key=lambda x: (-x["mean_rank_score"], -x["mean_depth_gain"]))

    # Print top-5 summary
    _log("\n=== TOP-5 PROFILES ===")
    _log(f"  {'Profile':28s}  {'grid gain':>10}  {'ring gain':>10}  {'mean gain':>10}")
    _log(f"  {'-'*64}")
    for r in ranked[:5]:
        g = r["per_topology"].get("grid", {}).get("depth_gain", 0) * 100
        ring = r["per_topology"].get("ring", {}).get("depth_gain", 0) * 100
        mean = r["mean_depth_gain"] * 100
        _log(f"  {r['profile']:28s}  {g:>+9.2f}%  {ring:>+9.2f}%  {mean:>+9.2f}%")

    best = ranked[0]
    _log(f"\nBEST: {best['profile']}  mean_depth_gain={best['mean_depth_gain']*100:.2f}%")
    _log(f"  score_kwargs: {best['best_score_kwargs']}")

    payload = {
        "circuit":        CIRCUIT,
        "topologies":     topologies,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "seed":           seed,
        "sa_params":      {"steps": sa.steps, "t0": sa.t0, "tend": sa.tend},
        "base_kwargs":    BASE_KW,
        "n_profiles":     n_profiles,
        "ranked":         ranked,
        "best_profile":   best["profile"],
        "best_score_kwargs": best["best_score_kwargs"],
        "baselines":      {t: {"depth": baselines[t]["logical_depth"]} for t in topologies},
        "notes": (
            "Sweep for rand_50q_500t_s42. Groups: A=span screening, "
            "B=MST direction, C=split boost, D=combined combos. "
            "Ranking: depth_gain + 0.15*inter_gain."
        ),
    }
    path = os.path.join(SUMMARY_DIR, f"{OUT_NAME}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved → {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Random circuit sensitivity sweep")
    parser.add_argument("--topologies", nargs="*", default=["grid", "ring"],
                        choices=["grid", "ring"])
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    parser.add_argument("--sa-steps", type=int,   default=12_000,
                        help="12k for screening; use 22500 for confirmation")
    parser.add_argument("--sa-t0",    type=float, default=1e5)
    parser.add_argument("--sa-tend",  type=float, default=0.05)
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    sa = SAParams(steps=args.sa_steps, t0=args.sa_t0, tend=args.sa_tend)
    payload = run(topologies=args.topologies, seed=args.seed, sa=sa, force=args.force)

    print(json.dumps({
        "summary":           os.path.join(SUMMARY_DIR, f"{OUT_NAME}.json"),
        "best_profile":      payload["best_profile"],
        "best_score_kwargs": payload["best_score_kwargs"],
        "top3": [
            {"profile": r["profile"], "mean_depth_gain_pct": round(r["mean_depth_gain"] * 100, 2)}
            for r in payload["ranked"][:3]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
