"""
Create a mapping-quality tradeoff panel from LaM raw sensitivity runs.

Behavior
--------
- For one selected circuit: plot all matching raw runs directly.
- For multiple selected circuits: average runs that share the same context
  (phase/label/mapper/scheduler/SA params/score kwargs) and plot the averaged
  point once.

The default figure is a 2x2 panel with:
  1. active block fraction vs logical depth
  2. inter-block rotation count vs logical depth
  3. mean MST vs logical depth
  4. mean split vs logical depth

By default, the script scans results/sensitivity_v2/raw and uses phase-1 weight
and preset runs only, since that is the main tuning data currently available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_ROOT, "results", "sensitivity_v2", "raw")
OUT_DIR = os.path.join(_ROOT, "results", "sensitivity_v2", "figures")
SUMMARY_DIR = os.path.join(_ROOT, "results", "sensitivity_v2", "summary")

# Edit this list as needed. With one circuit, the script plots raw points.
# With multiple circuits, it averages points in the same run context.
DEFAULT_CIRCUITS = ["gf2_16_mult"]
DEFAULT_PHASES = {"1w", "1p"}
ALGORITHM_NAME = "LaM"

PRETTY_LABELS = {
    "default_v2": "reference objective",
    "balanced_tradeoff": "balanced objective",
    "utilization_first": "utilization-focused",
    "interblock_first": "inter-block-focused",
    "locality_first": "locality-focused",
    "support_first": "support-focused",
}


@dataclass(frozen=True)
class PlotPoint:
    context_key: str
    label: str
    phase: str
    mapper_name: str
    scheduler: str
    logical_depth: float
    active_block_fraction: float
    inter_block_all: float
    mean_mst: float
    mean_split: float
    circuits: Tuple[str, ...]
    count: int


def _ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _iter_raw_records(raw_dir: str) -> Iterable[Dict[str, Any]]:
    if not os.path.isdir(raw_dir):
        return
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(raw_dir, name)
        try:
            rec = _load_json(path)
        except Exception:
            continue
        rec["_path"] = path
        yield rec


def _record_matches(rec: Dict[str, Any], circuits: List[str], phases: set[str]) -> bool:
    if rec.get("circuit") not in circuits:
        return False
    if phases and rec.get("phase") not in phases:
        return False
    return True


def _context_key(rec: Dict[str, Any]) -> str:
    payload = {
        "phase": rec.get("phase"),
        "label": rec.get("label"),
        "mapper_name": rec.get("mapper_name"),
        "scheduler": rec.get("scheduler"),
        "sa_params": rec.get("sa_params"),
        "score_kwargs_v2": rec.get("score_kwargs_v2"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _point_from_record(rec: Dict[str, Any]) -> PlotPoint:
    metrics = rec["mapping_metrics_v2"]
    n_blocks = float(rec["hardware"]["n_blocks"])
    return PlotPoint(
        context_key=_context_key(rec),
        label=str(rec["label"]),
        phase=str(rec["phase"]),
        mapper_name=str(rec["mapper_name"]),
        scheduler=str(rec["scheduler"]),
        logical_depth=float(rec["logical_depth"]),
        active_block_fraction=float(metrics["active_blocks"]) / n_blocks if n_blocks else 0.0,
        inter_block_all=float(rec["inter_block_all"]),
        mean_mst=float(metrics["mean_mst"]),
        mean_split=float(metrics["mean_split"]),
        circuits=(str(rec["circuit"]),),
        count=1,
    )


def _average_points(points: List[PlotPoint]) -> PlotPoint:
    if len(points) == 1:
        return points[0]
    p0 = points[0]
    return PlotPoint(
        context_key=p0.context_key,
        label=p0.label,
        phase=p0.phase,
        mapper_name=p0.mapper_name,
        scheduler=p0.scheduler,
        logical_depth=sum(p.logical_depth for p in points) / len(points),
        active_block_fraction=sum(p.active_block_fraction for p in points) / len(points),
        inter_block_all=sum(p.inter_block_all for p in points) / len(points),
        mean_mst=sum(p.mean_mst for p in points) / len(points),
        mean_split=sum(p.mean_split for p in points) / len(points),
        circuits=tuple(sorted({c for p in points for c in p.circuits})),
        count=len(points),
    )


def collect_points(
    *,
    raw_dir: str,
    circuits: List[str],
    phases: set[str],
) -> List[PlotPoint]:
    matched = [
        _point_from_record(rec)
        for rec in _iter_raw_records(raw_dir)
        if _record_matches(rec, circuits, phases)
    ]
    if not matched:
        raise FileNotFoundError(
            f"No raw SA-v2 records found in {raw_dir} for circuits={circuits} and phases={sorted(phases)}"
        )
    if len(circuits) == 1:
        return matched

    grouped: Dict[str, List[PlotPoint]] = {}
    for pt in matched:
        grouped.setdefault(pt.context_key, []).append(pt)
    return [_average_points(grouped[k]) for k in sorted(grouped)]


def _style_for(point: PlotPoint) -> Tuple[str, float, int]:
    if point.phase == "1p":
        return "#C44E52", 0.95, 85
    return "#4C72B0", 0.55, 35


def _pretty_label(label: str) -> str:
    if label in PRETTY_LABELS:
        return PRETTY_LABELS[label]
    return label.replace("_", " ")


def _annotate(ax, points: List[PlotPoint], x_attr: str, y_attr: str) -> None:
    for pt in points:
        if pt.phase != "1p":
            continue
        ax.annotate(
            _pretty_label(pt.label),
            (getattr(pt, x_attr), getattr(pt, y_attr)),
            fontsize=7,
            alpha=0.9,
            xytext=(4, 4),
            textcoords="offset points",
        )


def plot_panel(points: List[PlotPoint], out_path: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    specs = [
        (
            "active_block_fraction",
            "logical_depth",
            "Active block fraction (higher is better)",
            "Greedy depth (lower is better)",
        ),
        (
            "inter_block_all",
            "logical_depth",
            "Inter-block rotation count (lower is better)",
            "Greedy depth (lower is better)",
        ),
        (
            "mean_mst",
            "logical_depth",
            "Mean inter-block locality cost / MST (lower is better)",
            "Greedy depth (lower is better)",
        ),
        (
            "mean_split",
            "logical_depth",
            "Mean support split imbalance (lower is better)",
            "Greedy depth (lower is better)",
        ),
    ]

    preset_points = [p for p in points if p.phase == "1p"]
    sweep_points = [p for p in points if p.phase != "1p"]

    for ax, (x_attr, y_attr, xlabel, ylabel) in zip(axes.ravel(), specs):
        for bucket in (sweep_points, preset_points):
            for pt in bucket:
                color, alpha, size = _style_for(pt)
                ax.scatter(
                    getattr(pt, x_attr),
                    getattr(pt, y_attr),
                    s=size,
                    color=color,
                    alpha=alpha,
                    edgecolors="none",
                )
        _annotate(ax, preset_points, x_attr, y_attr)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle=":")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="single-weight sweep",
            markerfacecolor="#4C72B0",
            markersize=7,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="named LaM objective",
            markerfacecolor="#C44E52",
            markersize=9,
            alpha=0.95,
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.text(
        0.5,
        0.92,
        "Red annotated points are named LaM objective presets. Blue points are one-weight sensitivity runs.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_summary(points: List[PlotPoint], out_path: str, *, circuits: List[str], phases: set[str]) -> None:
    payload = {
        "circuits": circuits,
        "phases": sorted(phases),
        "num_points": len(points),
        "points": [
            {
                "context_key": p.context_key,
                "label": p.label,
                "pretty_label": _pretty_label(p.label),
                "phase": p.phase,
                "mapper_name": p.mapper_name,
                "scheduler": p.scheduler,
                "logical_depth": p.logical_depth,
                "active_block_fraction": p.active_block_fraction,
                "inter_block_all": p.inter_block_all,
                "mean_mst": p.mean_mst,
                "mean_split": p.mean_split,
                "circuits": list(p.circuits),
                "count": p.count,
            }
            for p in points
        ],
    }
    _save_json(out_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LaM mapping-quality tradeoff panel")
    parser.add_argument("--circuits", nargs="*", default=DEFAULT_CIRCUITS, help="Circuit list to include")
    parser.add_argument("--phases", nargs="*", default=sorted(DEFAULT_PHASES), help="Raw phases to include")
    parser.add_argument("--raw-dir", default=RAW_DIR, help="Directory containing raw SA-v2 JSON runs")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--summary-out", default=None, help="Optional JSON summary output path")
    args = parser.parse_args()

    _ensure_dirs()
    circuits = list(args.circuits)
    phases = set(args.phases)
    points = collect_points(raw_dir=args.raw_dir, circuits=circuits, phases=phases)

    tag = circuits[0] if len(circuits) == 1 else f"{len(circuits)}circuits_avg"
    out_path = args.out or os.path.join(OUT_DIR, f"mapping_tradeoff_panel_{tag}.png")
    summary_path = args.summary_out or os.path.join(SUMMARY_DIR, f"mapping_tradeoff_panel_{tag}.json")

    title = (
        f"{ALGORITHM_NAME} mapping Quality vs Greedy Depth ({circuits[0]})"
        if len(circuits) == 1
        else f"{ALGORITHM_NAME} mapping Quality vs Greedy Depth ({len(circuits)}-circuit average)"
    )
    plot_panel(points, out_path, title)
    save_summary(points, summary_path, circuits=circuits, phases=phases)

    print(f"[mapping-panel] saved figure: {out_path}")
    print(f"[mapping-panel] saved summary: {summary_path}")
    print(f"[mapping-panel] plotted {len(points)} point(s)")


if __name__ == "__main__":
    main()
