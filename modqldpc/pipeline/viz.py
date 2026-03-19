"""
Visualization module for single-circuit pipeline analysis.

Each plot_* function accepts collected profiles and saves a PNG figure.
All functions return the matplotlib Figure so callers can close/reuse them.
"""
from __future__ import annotations

import math
import pathlib
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe in scripts with no display
import matplotlib.pyplot as plt
import numpy as np

from .profiling import CircuitProfile, LayerProfile

# ── Angle thresholds ─────────────────────────────────────────────────────────
_T_ANGLE   = math.pi / 8   # T-type magic rotation
_CLF_ANGLE = math.pi / 4   # Clifford rotation
_ATOL      = 1e-4

# ── DAG node kind display order + colours ────────────────────────────────────
_KIND_ORDER = [
    "init_pivot",
    "local_couple",
    "interblock_link",
    "meas_parity_PZ",
    "meas_magic_X",
    "frame_update",
]
_KIND_COLORS = {
    "init_pivot":      "#4e79a7",
    "local_couple":    "#f28e2b",
    "interblock_link": "#e15759",
    "meas_parity_PZ":  "#76b7b2",
    "meas_magic_X":    "#59a14f",
    "frame_update":    "#b07aa1",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path is not None:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")


def _layer_x(layer_profiles: List[LayerProfile]):
    """x-axis positions and tick labels for per-layer plots."""
    ids = [lp.layer_id for lp in layer_profiles]
    x   = np.arange(len(ids))
    return x, ids


def _set_layer_ticks(ax, x, ids):
    if len(x) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(ids, fontsize=7)
    else:
        # sparse ticks for long circuits
        step = max(1, len(x) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(ids[::step], fontsize=7)


# ── Fig 1 ─────────────────────────────────────────────────────────────────────

def plot_circuit_character(
    circuit_profile: CircuitProfile,
    layer_profiles: List[LayerProfile],
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2×2 circuit character panel.

    (a) Rotation weight histogram   — distribution of Pauli support sizes
    (b) Rotations per layer         — bar per layer_id
    (c) Angle type per layer        — stacked bar: T (π/8) vs Clifford (π/4)
    (d) DAG node counts per layer   — stacked bar by node kind
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cp = circuit_profile
    fig.suptitle(
        f"Circuit character  |  {cp.n_rotations_total} rotations  ·  "
        f"{cp.n_layers} layers  ·  {cp.n_logicals} logical qubits",
        fontsize=12, fontweight="bold",
    )

    x, ids = _layer_x(layer_profiles)

    # ── (a) Rotation weight histogram ────────────────────────────────────────
    ax = axes[0, 0]
    weights = cp.all_rotation_weights
    if weights:
        max_w = max(weights)
        bins  = np.arange(-0.5, max_w + 1.5, 1.0)
        ax.hist(weights, bins=bins, color="#4e79a7", edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(0, max_w + 1))
    ax.set_xlabel("Support size (# non-I Paulis)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Rotation weight distribution")

    t_count = sum(1 for a in cp.all_rotation_angles if abs(abs(a) - _T_ANGLE) < _ATOL)
    total   = len(cp.all_rotation_angles)
    ax.text(
        0.97, 0.97,
        f"T-type: {t_count}/{total}  ({100 * t_count / max(total, 1):.0f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # ── (b) Rotations per layer ───────────────────────────────────────────────
    ax = axes[0, 1]
    n_rots = [lp.n_rotations for lp in layer_profiles]
    ax.bar(x, n_rots, color="#4e79a7", width=0.7)
    mean_rots = sum(n_rots) / max(len(n_rots), 1)
    ax.axhline(mean_rots, color="#e15759", linestyle="--", linewidth=1.2,
               label=f"mean = {mean_rots:.1f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotations")
    ax.set_title("(b) Rotations per layer")
    _set_layer_ticks(ax, x, ids)
    ax.legend(fontsize=8)

    # ── (c) Angle type per layer ──────────────────────────────────────────────
    ax = axes[1, 0]
    t_counts, clf_counts, other_counts = [], [], []
    for lp in layer_profiles:
        t = sum(1 for a in lp.rotation_angles if abs(abs(a) - _T_ANGLE)   < _ATOL)
        c = sum(1 for a in lp.rotation_angles if abs(abs(a) - _CLF_ANGLE) < _ATOL)
        o = len(lp.rotation_angles) - t - c
        t_counts.append(t)
        clf_counts.append(c)
        other_counts.append(o)

    t_arr   = np.array(t_counts,   dtype=float)
    clf_arr = np.array(clf_counts, dtype=float)
    oth_arr = np.array(other_counts, dtype=float)

    ax.bar(x, t_arr,   width=0.7, color="#f28e2b", label="T  (π/8)")
    ax.bar(x, clf_arr, width=0.7, color="#4e79a7", bottom=t_arr, label="Clifford  (π/4)")
    if oth_arr.sum() > 0:
        ax.bar(x, oth_arr, width=0.7, color="#bab0ac",
               bottom=t_arr + clf_arr, label="other")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotations")
    ax.set_title("(c) Angle type per layer")
    _set_layer_ticks(ax, x, ids)
    ax.legend(fontsize=8)

    # ── (d) DAG node counts per layer ─────────────────────────────────────────
    ax = axes[1, 1]
    # include any kinds not in the standard order
    extra_kinds = sorted({
        k for lp in layer_profiles
        for k in lp.dag_node_counts
        if k not in _KIND_ORDER
    })
    all_kinds = _KIND_ORDER + extra_kinds

    bottom = np.zeros(len(layer_profiles))
    legend_patches = []
    for kind in all_kinds:
        vals = np.array([lp.dag_node_counts.get(kind, 0) for lp in layer_profiles], dtype=float)
        if vals.sum() == 0:
            continue
        color = _KIND_COLORS.get(kind, "#aaaaaa")
        ax.bar(x, vals, width=0.7, bottom=bottom, color=color)
        legend_patches.append(matplotlib.patches.Patch(color=color, label=kind))
        bottom += vals

    ax.set_xlabel("Layer")
    ax.set_ylabel("DAG nodes")
    ax.set_title("(d) DAG node counts per layer")
    _set_layer_ticks(ax, x, ids)
    ax.legend(handles=legend_patches, fontsize=7, ncol=2, loc="upper right")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 2 ─────────────────────────────────────────────────────────────────────

def plot_depth_profile(
    layer_profiles: List[LayerProfile],
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 2: Scheduling depth profile.

    Left y-axis  — bar: execution depth per layer (from ex.depth)
    Right y-axis — line: rotation count per layer
    Annotated with total depth, average depth/layer, and depth-per-rotation ratio.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    x, ids = _layer_x(layer_profiles)
    depths  = np.array([lp.depth      for lp in layer_profiles], dtype=float)
    n_rots  = np.array([lp.n_rotations for lp in layer_profiles], dtype=float)

    total_depth = int(depths.sum())
    avg_depth   = depths.mean() if len(depths) else 0.0
    avg_rots    = n_rots.mean() if len(n_rots) else 0.0

    # ── bars: depth ──────────────────────────────────────────────────────────
    # colour bars by relative depth so outliers stand out
    norm   = plt.Normalize(vmin=depths.min(), vmax=max(depths.max(), 1))
    colors = plt.cm.Blues(norm(depths) * 0.7 + 0.3)   # keep in 0.3–1.0 range

    bars = ax1.bar(x, depths, width=0.65, color=colors, zorder=2)
    ax1.axhline(avg_depth, color="#e15759", linestyle="--", linewidth=1.4,
                label=f"mean depth = {avg_depth:.1f}", zorder=3)
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Execution depth (time steps)", fontsize=11)
    ax1.set_title(
        f"Scheduling depth per layer  |  total = {total_depth}  ·  "
        f"avg = {avg_depth:.1f}  ·  {len(layer_profiles)} layers",
        fontsize=12, fontweight="bold",
    )
    _set_layer_ticks(ax1, x, ids)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(0, depths.max() * 1.15 if depths.max() > 0 else 1)

    # ── right axis: rotation count overlay ───────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, n_rots, color="#f28e2b", linewidth=1.8, marker="o",
             markersize=4, label=f"rotations (mean={avg_rots:.1f})", zorder=4)
    ax2.set_ylabel("Rotations in layer", fontsize=11, color="#f28e2b")
    ax2.tick_params(axis="y", labelcolor="#f28e2b")
    ax2.set_ylim(0, n_rots.max() * 1.3 if n_rots.max() > 0 else 1)
    ax2.legend(loc="upper right", fontsize=9)

    # ── depth-per-rotation annotation ────────────────────────────────────────
    if avg_rots > 0:
        dpr = avg_depth / avg_rots
        fig.text(0.5, 0.01,
                 f"avg depth / rotation = {dpr:.1f}",
                 ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 3 ─────────────────────────────────────────────────────────────────────

def plot_block_utilization(
    circuit_profile: CircuitProfile,
    layer_profiles: List[LayerProfile],
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 3: Block utilization heatmap.

    Rows = hardware blocks, Columns = layers.
    Colour = occupancy: fraction of layer depth during which the block is busy.
    High occupancy (→1.0) = throughput bottleneck.
    Low occupancy = spare capacity / possible mapping imbalance.
    """
    block_ids = circuit_profile.block_ids
    n_blocks  = len(block_ids)
    n_layers  = len(layer_profiles)

    matrix = np.zeros((n_blocks, n_layers))
    for col, lp in enumerate(layer_profiles):
        depth = max(lp.depth, 1)
        for row, bid in enumerate(block_ids):
            busy = lp.block_busy_slots.get(bid, 0)
            matrix[row, col] = min(busy / depth, 1.0)

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(14, max(4, n_blocks * 0.6 + 2)),
        gridspec_kw={"width_ratios": [5, 1]},
    )
    fig.suptitle(
        f"Block utilization heatmap  |  {n_blocks} blocks · {n_layers} layers",
        fontsize=12, fontweight="bold",
    )

    # ── heatmap ───────────────────────────────────────────────────────────────
    im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd",
                        vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_heat.set_xlabel("Layer", fontsize=10)
    ax_heat.set_ylabel("Block", fontsize=10)
    ax_heat.set_title("Occupancy  (block busy time / layer depth)", fontsize=10)

    x, ids = _layer_x(layer_profiles)
    if n_layers <= 40:
        ax_heat.set_xticks(np.arange(n_layers))
        ax_heat.set_xticklabels(ids, fontsize=7, rotation=90)
    else:
        step = max(1, n_layers // 20)
        ax_heat.set_xticks(np.arange(0, n_layers, step))
        ax_heat.set_xticklabels(ids[::step], fontsize=7, rotation=90)

    ax_heat.set_yticks(np.arange(n_blocks))
    ax_heat.set_yticklabels([f"B{bid}" for bid in block_ids], fontsize=9)

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cb.set_label("Occupancy", fontsize=9)

    # ── right bar: mean occupancy per block ───────────────────────────────────
    mean_occ = matrix.mean(axis=1)
    y_pos = np.arange(n_blocks)
    bar_colors = plt.cm.YlOrRd(mean_occ)
    ax_bar.barh(y_pos, mean_occ, color=bar_colors, height=0.6)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Mean\noccupancy", fontsize=9)
    ax_bar.set_title("Avg", fontsize=10)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"B{bid}" for bid in block_ids], fontsize=9)
    ax_bar.axvline(mean_occ.mean(), color="#e15759", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_occ.mean():.2f}")
    ax_bar.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 4 ─────────────────────────────────────────────────────────────────────

def plot_routing_distances(
    layer_profiles: List[LayerProfile],
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 4: Routing distance distribution.

    Left  — histogram of per-rotation max-hop distance, split by weight class.
    Right — per-layer mean and max hop distance as line plots.
    High distances indicate mapping spread support qubits far apart.
    """
    all_weights: List[int] = []
    all_hops:    List[int] = []
    for lp in layer_profiles:
        all_weights.extend(lp.rotation_weights)
        all_hops.extend(lp.rotation_max_hops)

    fig, (ax_hist, ax_line) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Routing distance distribution  (max BFS hops between support blocks)",
        fontsize=12, fontweight="bold",
    )

    # ── left: histogram split by weight class ────────────────────────────────
    ax = ax_hist
    if all_hops:
        max_h = max(all_hops)
        w_arr = np.array(all_weights)
        h_arr = np.array(all_hops)
        bins  = np.arange(-0.5, max_h + 1.5, 1.0)
        x_pos = np.arange(max_h + 1)

        weight_classes = {
            "weight 1":   w_arr == 1,
            "weight 2–3": (w_arr >= 2) & (w_arr <= 3),
            "weight 4+":  w_arr >= 4,
        }
        cls_colors = {
            "weight 1":   "#76b7b2",
            "weight 2–3": "#f28e2b",
            "weight 4+":  "#e15759",
        }
        bottom = np.zeros(max_h + 1)
        for label, mask in weight_classes.items():
            if mask.sum() == 0:
                continue
            counts, _ = np.histogram(h_arr[mask], bins=bins)
            ax.bar(x_pos, counts, width=0.8, bottom=bottom,
                   color=cls_colors[label], label=label,
                   edgecolor="white", linewidth=0.5)
            bottom += counts

        ax.set_xticks(x_pos)

    ax.set_xlabel("Max BFS hops between support blocks")
    ax.set_ylabel("Count (rotations)")
    ax.set_title("(a) Hop distance histogram by weight class")
    ax.legend(fontsize=9)

    # ── right: per-layer mean / max hops ─────────────────────────────────────
    ax = ax_line
    x, ids = _layer_x(layer_profiles)
    mean_hops_arr = np.array([
        float(np.mean(lp.rotation_max_hops)) if lp.rotation_max_hops else 0.0
        for lp in layer_profiles
    ])
    max_hops_arr = np.array([
        max(lp.rotation_max_hops) if lp.rotation_max_hops else 0
        for lp in layer_profiles
    ], dtype=float)

    ax.fill_between(x, mean_hops_arr, max_hops_arr,
                    alpha=0.2, color="#e15759", label="mean → max range")
    ax.plot(x, max_hops_arr, color="#e15759", linewidth=1.5,
            marker="^", markersize=4, label="max hops")
    ax.plot(x, mean_hops_arr, color="#4e79a7", linewidth=1.8,
            marker="o", markersize=4, label="mean hops")
    ax.set_xlabel("Layer")
    ax.set_ylabel("BFS hops")
    ax.set_title("(b) Max & mean hops per layer")
    _set_layer_ticks(ax, x, ids)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 5 ─────────────────────────────────────────────────────────────────────

def plot_frame_rewrites(
    layer_profiles: List[LayerProfile],
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 5: Clifford frame rewrite profile.

    Top    — grouped bar per layer: total / support-changing / angle-flipping rewrites.
    Bottom — cumulative rewrite lines across layers.
    """
    fig, (ax_bar, ax_cum) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Clifford frame rewrite profile", fontsize=12, fontweight="bold")

    x, ids = _layer_x(layer_profiles)
    total_rw   = np.array([lp.n_rewrites       for lp in layer_profiles], dtype=float)
    support_rw = np.array([lp.n_support_changes for lp in layer_profiles], dtype=float)
    angle_rw   = np.array([lp.n_angle_flips     for lp in layer_profiles], dtype=float)

    # ── top: grouped bar ──────────────────────────────────────────────────────
    w = 0.27
    ax_bar.bar(x - w, total_rw,   width=w, color="#4e79a7", label="total rewrites")
    ax_bar.bar(x,     support_rw, width=w, color="#e15759", label="support changed")
    ax_bar.bar(x + w, angle_rw,   width=w, color="#f28e2b", label="angle flipped")
    ax_bar.set_ylabel("Rewrites", fontsize=10)
    ax_bar.set_title("(a) Rewrites per layer", fontsize=10)
    ax_bar.legend(fontsize=9)
    ax_bar.text(
        0.99, 0.97,
        f"totals  →  all: {int(total_rw.sum())}  |  "
        f"support Δ: {int(support_rw.sum())}  |  angle Δ: {int(angle_rw.sum())}",
        transform=ax_bar.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )

    # ── bottom: cumulative lines ──────────────────────────────────────────────
    ax_cum.fill_between(x, np.cumsum(total_rw),   alpha=0.20, color="#4e79a7")
    ax_cum.fill_between(x, np.cumsum(support_rw), alpha=0.30, color="#e15759")
    ax_cum.plot(x, np.cumsum(total_rw),   color="#4e79a7", linewidth=2.0,
                label="cumulative total")
    ax_cum.plot(x, np.cumsum(support_rw), color="#e15759", linewidth=2.0,
                label="cumulative support Δ")
    ax_cum.plot(x, np.cumsum(angle_rw),   color="#f28e2b", linewidth=1.5,
                linestyle="--", label="cumulative angle Δ")
    ax_cum.set_xlabel("Layer", fontsize=10)
    ax_cum.set_ylabel("Cumulative rewrites", fontsize=10)
    ax_cum.set_title("(b) Cumulative rewrite accumulation", fontsize=10)
    ax_cum.legend(fontsize=9)
    _set_layer_ticks(ax_cum, x, ids)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 6 ─────────────────────────────────────────────────────────────────────

def plot_parallelism_profile(
    layer_profiles: List[LayerProfile],
    *,
    target_layer_id: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 6: Parallelism profile for one layer.

    Stacked bar per timestep coloured by node kind.
    Defaults to the layer with the maximum execution depth.
    Shows how densely the scheduler packs operations over time.
    """
    if not layer_profiles:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No layer profiles", ha="center", va="center")
        _save(fig, save_path)
        return fig

    if target_layer_id is not None:
        lp = next((p for p in layer_profiles if p.layer_id == target_layer_id), None)
        if lp is None:
            lp = layer_profiles[0]
    else:
        lp = max(layer_profiles, key=lambda p: p.depth)

    steps = lp.parallelism_steps

    if not steps:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Layer {lp.layer_id}: no schedule steps recorded",
                ha="center", va="center")
        _save(fig, save_path)
        return fig

    times = np.array([s["t"] for s in steps])

    all_step_kinds = list(dict.fromkeys(k for s in steps for k in s["kind_counts"]))
    ordered_kinds  = [k for k in _KIND_ORDER if k in all_step_kinds]
    ordered_kinds += [k for k in all_step_kinds if k not in _KIND_ORDER]

    fig, (ax_stack, ax_total) = plt.subplots(
        2, 1,
        figsize=(max(10, len(times) * 0.4 + 3), 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"Parallelism profile — layer {lp.layer_id}  "
        f"(depth={lp.depth}, rotations={lp.n_rotations})",
        fontsize=12, fontweight="bold",
    )

    # ── top: stacked bar by kind ──────────────────────────────────────────────
    bottom = np.zeros(len(times))
    for kind in ordered_kinds:
        vals = np.array([s["kind_counts"].get(kind, 0) for s in steps], dtype=float)
        if vals.sum() == 0:
            continue
        color = _KIND_COLORS.get(kind, "#aaaaaa")
        ax_stack.bar(times, vals, width=0.75, bottom=bottom, color=color, label=kind)
        bottom += vals

    ax_stack.set_ylabel("Nodes starting at timestep", fontsize=10)
    ax_stack.set_title("(a) Operations starting per timestep (by kind)", fontsize=10)
    ax_stack.legend(fontsize=8, ncol=3, loc="upper right")

    # ── bottom: total per step ────────────────────────────────────────────────
    totals = np.array([s["total_nodes"] for s in steps], dtype=float)
    ax_total.bar(times, totals, width=0.75, color="#4e79a7", alpha=0.8)
    ax_total.axhline(totals.mean(), color="#e15759", linestyle="--", linewidth=1.2,
                     label=f"mean = {totals.mean():.1f}")
    ax_total.set_xlabel("Timestep", fontsize=10)
    ax_total.set_ylabel("Total ops", fontsize=10)
    ax_total.set_title("(b) Total operations per timestep", fontsize=10)
    ax_total.legend(fontsize=8)

    if len(times) <= 80:
        ax_total.set_xticks(times)
        ax_total.set_xticklabels([str(t) for t in times], fontsize=7, rotation=45)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 8 ─────────────────────────────────────────────────────────────────────

def plot_algo_comparison(
    results,                          # List[ComboResult] from experiment.py
    *,
    mapper_labels: Optional[dict] = None,
    sched_labels: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 8: Algorithm comparison (3 mappers × 3 schedulers).

    Top    — grouped bar: total execution depth per (mapper, scheduler) combo.
             Groups = mappers, bars within group = schedulers.
    Bottom — heatmap: mean block utilisation for each combo.
    """
    mapper_names = list(dict.fromkeys(r.mapper_name for r in results))
    sched_names  = list(dict.fromkeys(r.sched_name  for r in results))

    mlabels = mapper_labels or {m: m for m in mapper_names}
    slabels = sched_labels  or {s: s for s in sched_names}

    depth_mat = np.zeros((len(mapper_names), len(sched_names)))
    util_mat  = np.zeros((len(mapper_names), len(sched_names)))

    for r in results:
        mi = mapper_names.index(r.mapper_name)
        si = sched_names.index(r.sched_name)
        depth_mat[mi, si] = r.total_depth
        all_occ = []
        for lp in r.layer_profiles:
            d = max(lp.depth, 1)
            for busy in lp.block_busy_slots.values():
                all_occ.append(min(busy / d, 1.0))
        util_mat[mi, si] = float(np.mean(all_occ)) if all_occ else 0.0

    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(
        "Algorithm comparison  (3 mappers × 3 schedulers)",
        fontsize=13, fontweight="bold",
    )

    # ── top: grouped bar ──────────────────────────────────────────────────────
    n_mappers = len(mapper_names)
    n_scheds  = len(sched_names)
    bar_w     = 0.22
    x_center  = np.arange(n_mappers)
    sched_colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    offsets = np.linspace(-(n_scheds - 1) / 2, (n_scheds - 1) / 2, n_scheds) * bar_w

    for si, sched_name in enumerate(sched_names):
        vals = depth_mat[:, si]
        ax_bar.bar(
            x_center + offsets[si], vals, width=bar_w,
            color=sched_colors[si % len(sched_colors)],
            label=slabels[sched_name],
            edgecolor="white", linewidth=0.5,
        )
        for xi, v in zip(x_center + offsets[si], vals):
            if v > 0:
                ax_bar.text(xi, v + depth_mat.max() * 0.01, str(int(v)),
                            ha="center", va="bottom", fontsize=7)

    ax_bar.set_xticks(x_center)
    ax_bar.set_xticklabels([mlabels[m] for m in mapper_names], fontsize=11)
    ax_bar.set_ylabel("Total execution depth", fontsize=10)
    ax_bar.set_title("(a) Total depth by mapper + scheduler combo", fontsize=10)
    ax_bar.legend(title="Scheduler", fontsize=9, title_fontsize=9)
    ax_bar.set_ylim(0, depth_mat.max() * 1.18 if depth_mat.max() > 0 else 1)

    best_idx = np.unravel_index(np.argmin(depth_mat), depth_mat.shape)
    best_mi, best_si = best_idx
    ax_bar.annotate(
        "best",
        xy=(x_center[best_mi] + offsets[best_si], depth_mat[best_mi, best_si]),
        xytext=(0, 14), textcoords="offset points",
        ha="center", fontsize=8, color="#2ca02c",
        arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=1.2),
    )

    # ── bottom: utilisation heatmap ───────────────────────────────────────────
    im = ax_heat.imshow(
        util_mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    ax_heat.set_xticks(np.arange(n_scheds))
    ax_heat.set_xticklabels([slabels[s] for s in sched_names], fontsize=10)
    ax_heat.set_yticks(np.arange(n_mappers))
    ax_heat.set_yticklabels([mlabels[m] for m in mapper_names], fontsize=10)
    ax_heat.set_xlabel("Scheduler", fontsize=10)
    ax_heat.set_ylabel("Mapper", fontsize=10)
    ax_heat.set_title("(b) Mean block utilisation per combo", fontsize=10)

    for mi in range(n_mappers):
        for si in range(n_scheds):
            ax_heat.text(si, mi, f"{util_mat[mi, si]:.2f}",
                         ha="center", va="center",
                         fontsize=9, color="black" if util_mat[mi, si] < 0.6 else "white")

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cb.set_label("Mean occupancy", fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 9 ─────────────────────────────────────────────────────────────────────

def plot_sparse_dense_comparison(
    dense_result,                      # ComboResult
    sparse_result,                     # ComboResult
    *,
    dense_label: str = "Dense",
    sparse_label: str = "Sparse",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 9: Sparse vs dense hardware comparison.

    Left  — overlay of per-layer execution depth for both hardware configs.
    Right — side-by-side mean block utilisation bar per block.
    """
    fig, (ax_depth, ax_util) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Hardware density comparison:  {dense_label}  vs  {sparse_label}",
        fontsize=12, fontweight="bold",
    )

    # ── left: per-layer depth overlay ────────────────────────────────────────
    d_depths = np.array([lp.depth for lp in dense_result.layer_profiles], dtype=float)
    s_depths = np.array([lp.depth for lp in sparse_result.layer_profiles], dtype=float)
    x_d = np.arange(len(d_depths))
    x_s = np.arange(len(s_depths))

    ax_depth.fill_between(x_d, d_depths, alpha=0.15, color="#4e79a7")
    ax_depth.plot(x_d, d_depths, color="#4e79a7", linewidth=2.0,
                  marker="o", markersize=3,
                  label=f"{dense_label}  (total={int(d_depths.sum())})")
    ax_depth.fill_between(x_s, s_depths, alpha=0.15, color="#e15759")
    ax_depth.plot(x_s, s_depths, color="#e15759", linewidth=2.0,
                  marker="s", markersize=3,
                  label=f"{sparse_label}  (total={int(s_depths.sum())})")

    ax_depth.axhline(d_depths.mean(), color="#4e79a7", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_depth.axhline(s_depths.mean(), color="#e15759", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_depth.set_xlabel("Layer", fontsize=10)
    ax_depth.set_ylabel("Execution depth (time steps)", fontsize=10)
    ax_depth.set_title("(a) Per-layer execution depth", fontsize=10)
    ax_depth.legend(fontsize=9)
    ax_depth.set_ylim(bottom=0)

    # ── right: mean block utilisation per block ───────────────────────────────
    def _mean_util_per_block(result):
        block_ids = result.circuit_profile.block_ids
        totals = {bid: 0.0 for bid in block_ids}
        counts = {bid: 0   for bid in block_ids}
        for lp in result.layer_profiles:
            d = max(lp.depth, 1)
            for bid in block_ids:
                busy = lp.block_busy_slots.get(bid, 0)
                totals[bid] += min(busy / d, 1.0)
                counts[bid] += 1
        return {bid: (totals[bid] / counts[bid] if counts[bid] else 0.0)
                for bid in block_ids}

    d_util = _mean_util_per_block(dense_result)
    s_util = _mean_util_per_block(sparse_result)
    all_blocks = sorted(set(d_util) | set(s_util))
    y_pos = np.arange(len(all_blocks))
    w = 0.35

    d_vals = np.array([d_util.get(b, 0.0) for b in all_blocks])
    s_vals = np.array([s_util.get(b, 0.0) for b in all_blocks])

    ax_util.barh(y_pos - w / 2, d_vals, height=w, color="#4e79a7",
                 alpha=0.85, label=dense_label)
    ax_util.barh(y_pos + w / 2, s_vals, height=w, color="#e15759",
                 alpha=0.85, label=sparse_label)
    ax_util.set_yticks(y_pos)
    ax_util.set_yticklabels([f"B{b}" for b in all_blocks], fontsize=9)
    ax_util.set_xlabel("Mean occupancy", fontsize=10)
    ax_util.set_title("(b) Mean block utilisation", fontsize=10)
    ax_util.set_xlim(0, 1.05)
    ax_util.axvline(d_vals.mean(), color="#4e79a7", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_util.axvline(s_vals.mean(), color="#e15759", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_util.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── Fig 10 ────────────────────────────────────────────────────────────────────

def _hw_layout(hw, spec) -> dict:
    """Return {block_id: (x, y)} drawing positions for the hardware graph."""
    block_ids = sorted(hw.blocks)
    if spec.topology == "grid":
        rows, cols = spec.grid_rows, spec.grid_cols
        return {bid: (float(i % cols), float(-(i // cols)))
                for i, bid in enumerate(block_ids)}
    else:  # ring
        n = len(block_ids)
        return {
            bid: (math.cos(2 * math.pi * i / n - math.pi / 2),
                  math.sin(2 * math.pi * i / n - math.pi / 2))
            for i, bid in enumerate(block_ids)
        }


def _draw_hw_on_ax(ax, hw, spec, *, logical_to_block=None, title=""):
    """Draw one hardware graph topology onto ax (no axes frame)."""
    pos = _hw_layout(hw, spec)
    block_ids = sorted(hw.blocks)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    radius = span * 0.10          # adaptive node radius
    font_size = max(6, min(10, int(radius * 30)))

    # logical count per block
    lcount = {bid: 0 for bid in block_ids}
    if logical_to_block:
        for lid, bid in logical_to_block.items():
            if bid in lcount:
                lcount[bid] += 1

    # ── edges ─────────────────────────────────────────────────────────────────
    for c in hw.couplers.values():
        x0, y0 = pos[c.u]
        x1, y1 = pos[c.v]
        ax.plot([x0, x1], [y0, y1], "-", color="#999999",
                linewidth=2.5, zorder=1, solid_capstyle="round")
        if c.capacity > 1:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, str(c.capacity), ha="center", va="center",
                    fontsize=6, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.8))

    # ── nodes ─────────────────────────────────────────────────────────────────
    for bid in block_ids:
        x, y = pos[bid]
        n_cap = hw.blocks[bid].num_logicals
        n_assigned = lcount[bid]
        fill_frac = (n_assigned / n_cap) if logical_to_block else 0.35
        node_color = plt.cm.YlOrRd(0.15 + fill_frac * 0.75)

        circle = plt.Circle((x, y), radius, color=node_color,
                             ec="#333333", linewidth=1.5, zorder=3)
        ax.add_patch(circle)

        # block ID
        ax.text(x, y + radius * 0.15, f"B{bid}",
                ha="center", va="center", fontsize=font_size,
                fontweight="bold", zorder=4)

        # capacity / assigned label below the node
        sub = (f"{n_assigned}/{n_cap} lq" if logical_to_block
               else f"{n_cap} slots")
        ax.text(x, y - radius * 1.55, sub,
                ha="center", va="top",
                fontsize=max(5, font_size - 2), color="#444444", zorder=4)

    # ── corner annotation ─────────────────────────────────────────────────────
    n_edges = len(hw.couplers)
    ax.text(0.03, 0.03,
            f"{len(block_ids)} blocks · {n_edges} couplers",
            transform=ax.transAxes, fontsize=7, color="#777777", va="bottom")

    margin = radius * 3.2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.axis("off")


def plot_hardware_gallery(
    configs,          # list of (hw, spec, logical_to_block_or_None, label)
    *,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fig 10: Hardware topology gallery.

    Draws each hardware configuration as a node-edge graph.
    Pass logical_to_block (dict or None) per config to colour nodes
    by qubit load; omit (None) to show just the topology structure.

    Layouts:
      Grid topology  — blocks positioned on a 2-D grid matching HardwareSpec.
      Ring topology  — blocks on a circle, equal angular spacing.
    """
    n = len(configs)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)

    fig, raw_axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.8, nrows * 4.8),
        squeeze=False,
    )

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        raw_axes[r][c].axis("off")

    for idx, (hw, spec, l2b, label) in enumerate(configs):
        r, c = divmod(idx, ncols)
        _draw_hw_on_ax(raw_axes[r][c], hw, spec,
                       logical_to_block=l2b, title=label)

    fig.suptitle("Hardware topology diagrams", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
