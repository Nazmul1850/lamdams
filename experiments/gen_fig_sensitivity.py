"""
experiments/gen_fig_sensitivity.py

SA hyperparameter tuning + family sensitivity — combined single figure.

Two stories, one graph
----------------------
LEFT  (positions 1-3): How the default SA configuration was chosen.
      We compare four mapping strategies on Adder and QFT families.
      SA v2 (default) achieves best or near-best improvement on both.

PIVOT (position 3 — "SA v2 Default"):
      All three families converge here.  Adder and QFT bars are high;
      the GF bar is near zero — revealing the family-specific gap.

RIGHT (positions 3-5): GF sensitivity to objective weights.
      Progressively suppressing the span penalty (span_drop) recovers
      competitive depth gains for GF multiplication circuits.

Layout
------
X-axis  : Round Robin → SA v1 → SA v2 Default → Locality+Balance → Span Drop
Y-axis  : % depth gain vs random mapping baseline  (higher = better)
Series  : Adder (blue)  |  QFT (teal)  |  GF mult (orange)
          Each series plotted only where experiments were run.
Overlay : individual (circuit × topology) dots for each bar
Error   : ±1 std dev across circuits
Divider : vertical dashed separator between "tuning" and "GF sensitivity"

Data
----
Adder + QFT:  results/sa_v2_final_tuning/raw/
              round_robin, sa_v1, sa_v2_default vs random baseline
GF mult:      results/family_scaling/summary/gf_small_scaling.json
              base(=default), locality_light_balance, span_drop vs random

Output
------
  results/figures/fig_sensitivity.pdf
  results/figures/fig_sensitivity.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
TUNE_DIR = _ROOT / "results" / "sa_v2_final_tuning" / "raw"
GF_SUM   = _ROOT / "results" / "family_scaling" / "summary" / "gf_small_scaling.json"
OUT_DIR  = _ROOT / "results" / "figures"

SUFFIX = "steps=25000_t0=1.0ep05_tend=5.0em02_seed42"

# ── Family and circuit definitions ────────────────────────────────────────────
ADDER_CIRCS = ["Adder8", "Adder16", "Adder32", "Adder64"]
QFT_CIRCS   = ["QFT8",   "QFT16",   "QFT32",   "QFT64"]
GF_CIRCS    = ["gf6_mult", "gf7_mult", "gf8_mult", "gf9_mult", "gf10_mult"]
TOPOLOGIES  = ["grid", "ring"]

# ── X-axis profiles ───────────────────────────────────────────────────────────
# Labels describe the dominant objective, not internal version names.
X_PROFILES = [
    ("sa_v1",                  "Inter-block\nMinimization"),
    ("default",                "Balanced\nObjectives"),      # pivot — all families
    ("locality_light_balance", "Locality\n+ Balance"),
    ("span_drop",              "Block Util.\nDominated"),
]
N_PROF = len(X_PROFILES)

# Which profiles each family has data for
HAS_DATA = {
    "adder": {"sa_v1", "default"},
    "qft":   {"sa_v1", "default"},
    "gf":    {"default", "locality_light_balance", "span_drop"},
}

# ── Visual style (grayscale, matching gen_fig_families theme) ─────────────────
COLOR_ADDER = "#999999"   # light-mid gray
COLOR_QFT   = "#555555"   # mid-dark gray
COLOR_GF    = "#111111"   # near-black

MARKER_ADDER = "o"
MARKER_QFT   = "^"
MARKER_GF    = "s"

BAR_W    = 0.22
OFFSETS  = {"adder": -0.24, "qft": 0.0, "gf": +0.24}
JITTER   = 0.038


# ── File I/O helpers ──────────────────────────────────────────────────────────
def _top_depth(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    v = d.get("logical_depth")
    return float(v) if v is not None else None


def _tune_file(circuit: str, topo: str, label: str) -> Path:
    """Build path to a sa_v2_final_tuning raw file."""
    return TUNE_DIR / f"p0_{circuit}_{topo}_{label}_greedy_critical_{SUFFIX}_{label}.json"


# ── Collect Adder / QFT gains ─────────────────────────────────────────────────
_LABEL_MAP = {
    "round_robin": "round_robin",
    "sa_v1":       "sa_v1",
    "default":     "sa_v2",           # file uses "sa_v2_default" suffix
}
_FILE_SUFFIX = {
    "round_robin": "round_robin",
    "sa_v1":       "sa_v1",
    "default":     "sa_v2_default",
}

def collect_aq(circuits: list[str]) -> dict[str, list[float]]:
    """
    Returns {profile_key: [gain_pct, ...]} for the given circuits.
    gain = (random_depth - mapped_depth) / random_depth  × 100
    """
    gains: dict[str, list[float]] = {pk: [] for pk in HAS_DATA["adder"]}
    for circ in circuits:
        for topo in TOPOLOGIES:
            rand_path = _tune_file(circ, topo, "random")
            rand_d = _top_depth(rand_path)
            if rand_d is None or rand_d == 0:
                print(f"  [skip] {circ}/{topo} — no random depth")
                continue
            for prof_key in HAS_DATA["adder"]:
                fsuf  = _FILE_SUFFIX[prof_key]
                flabel = _LABEL_MAP.get(prof_key, fsuf)
                fpath = TUNE_DIR / (
                    f"p0_{circ}_{topo}_{flabel}_greedy_critical_{SUFFIX}_{fsuf}.json"
                )
                d = _top_depth(fpath)
                if d is None:
                    continue
                gains[prof_key].append((rand_d - d) / rand_d * 100)
    return gains


# ── Collect GF gains ──────────────────────────────────────────────────────────
def collect_gf() -> dict[str, list[float]]:
    """
    Returns {profile_key: [gain_pct, ...]} for GF circuits.
    'base' in summary → our 'default' profile.
    """
    d = json.loads(GF_SUM.read_text())
    label_map = {
        "base":                   "default",
        "locality_light_balance": "locality_light_balance",
        "span_drop":              "span_drop",
    }
    gains: dict[str, list[float]] = {pk: [] for pk in HAS_DATA["gf"]}
    for row in d["full"]["rows"]:
        pk = label_map.get(row["profile"])
        if pk is None:
            continue
        r, s = row["random_depth"], row["sa_depth"]
        if r and r > 0:
            gains[pk].append((r - s) / r * 100)
    return gains


# ── Build figure ──────────────────────────────────────────────────────────────
def build_figure() -> None:
    adder_gains = collect_aq(ADDER_CIRCS)
    qft_gains   = collect_aq(QFT_CIRCS)
    gf_gains    = collect_gf()

    series_data = {
        "adder": adder_gains,
        "qft":   qft_gains,
        "gf":    gf_gains,
    }
    series_meta = {
        "adder": ("Adder",   COLOR_ADDER),
        "qft":   ("QFT",     COLOR_QFT),
        "gf":    ("GF mult", COLOR_GF),
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    rng = np.random.default_rng(0)

    x = np.arange(N_PROF)   # one position per profile

    # track winner position per family for annotation
    best = {}   # family_key → (x_center, mean_val)

    FAM_MARKERS = {"adder": MARKER_ADDER, "qft": MARKER_QFT, "gf": MARKER_GF}

    for fam_key, (fam_label, color) in series_meta.items():
        offset  = OFFSETS[fam_key]
        pf_data = series_data[fam_key]
        mk      = FAM_MARKERS[fam_key]

        for i, (prof_key, _) in enumerate(X_PROFILES):
            if prof_key not in HAS_DATA[fam_key]:
                continue
            vals = pf_data.get(prof_key, [])
            if not vals:
                continue

            xc  = x[i] + offset
            mu  = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

            # bar
            ax.bar(xc, mu, width=BAR_W,
                   color=color, alpha=0.95, zorder=3,
                   edgecolor="white", linewidth=0.6)

            # error bar
            ax.errorbar(xc, mu, yerr=std,
                        fmt="none", color="#222222",
                        linewidth=1.1, capsize=3.5, capthick=1.0, zorder=5)

            # individual dots with family-specific marker
            jit = rng.uniform(-JITTER, JITTER, len(vals))
            ax.scatter(xc + jit, vals,
                       color=color, marker=mk,
                       edgecolors="white", linewidths=0.5,
                       s=42, zorder=6, alpha=0.92)

            # track best for family
            if fam_key not in best or mu > best[fam_key][1]:
                best[fam_key] = (xc, mu, std)

    # ── Random baseline reference ─────────────────────────────────────────────
    ax.axhline(0, color="#333333", linewidth=1.4, linestyle="--",
               zorder=7, label="Random mapping baseline  (0 %)")

    # ── Divider between default config and GF-sensitivity profiles ───────────
    # Between position 1 (default, idx=1) and position 2 (locality, idx=2)
    divider_x = x[1] + 0.55
    ax.axvline(divider_x, color="#aaaaaa", linewidth=1.0, linestyle=":",
               zorder=4, alpha=0.85)

    # ── Winner annotations ────────────────────────────────────────────────────
    # Adder and QFT both peak at "default" (pivot); annotate once, offset text
    # so the two labels don't collide.  GF peaks at span_drop — annotate separately.
    _aq_mu  = max(best.get("adder", (0, 0, 0))[1],
                  best.get("qft",   (0, 0, 0))[1])
    _aq_std = max(best.get("adder", (0, 0, 0))[2],
                  best.get("qft",   (0, 0, 0))[2])
    _aq_xc  = x[1]   # pivot position

    ax.annotate(
        "★ Balanced objectives:\nbest for Adder & QFT",
        xy=(_aq_xc, _aq_mu + _aq_std + 0.8),
        xytext=(_aq_xc - 0.45, _aq_mu + _aq_std + 9),
        fontsize=10, color="#222222", fontweight="extra bold",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.4),
    )

    if "gf" in best:
        gf_xc, gf_mu, gf_std = best["gf"]
        ax.annotate(
            "★ Block util. dominated:\nbest for GF",
            xy=(gf_xc, gf_mu + gf_std + 0.8),
            xytext=(gf_xc + 0.3, gf_mu + gf_std + 9),
            fontsize=10, color="#000000", fontweight="extra bold",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.4),
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in X_PROFILES], fontsize=12, fontweight="bold")
    ax.set_xlim(x[0] - 0.52, x[-1] + 0.52)
    ax.set_ylabel("Depth gain over random mapping  (%)", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.4, alpha=0.4, zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=11, length=3)

    # weight-space gradient annotation at bottom
    for side, txt, ha in [
        (0.01, "← inter-block communication cost dominant",    "left"),
        (0.99, "block utilization & locality dominant →", "right"),
    ]:
        ax.text(side, -0.10, txt,
                transform=ax.transAxes,
                fontsize=10, color="#333333", style="italic",
                ha=ha, va="top")

    # ── Legend ────────────────────────────────────────────────────────────────
    adder_p = mlines.Line2D([], [], color=COLOR_ADDER, marker=MARKER_ADDER,
                             markersize=8, linestyle="none",
                             markeredgecolor="white", markeredgewidth=0.4,
                             label="① Adder  (A8–A64, grid + ring)")
    qft_p   = mlines.Line2D([], [], color=COLOR_QFT, marker=MARKER_QFT,
                             markersize=8, linestyle="none",
                             markeredgecolor="white", markeredgewidth=0.4,
                             label="② QFT  (Q8–Q64, grid + ring)")
    gf_p    = mlines.Line2D([], [], color=COLOR_GF, marker=MARKER_GF,
                             markersize=8, linestyle="none",
                             markeredgecolor="white", markeredgewidth=0.4,
                             label="③ GF mult  (GF6–GF10, grid + ring)")
    ref_l   = mlines.Line2D([], [], color="#333333", linestyle="--", linewidth=1.4,
                             label="Random baseline  (0 %)")
    ax.legend(
        handles=[adder_p, qft_p, gf_p, ref_l],
        loc="lower right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=1.8,
        labelspacing=0.5,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1], pad=0.4)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_sensitivity.pdf", dpi=300,
                bbox_inches="tight", format="pdf")
    fig.savefig(OUT_DIR / "fig_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_DIR / 'fig_sensitivity.pdf'}")
    print(f"Saved → {OUT_DIR / 'fig_sensitivity.png'}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\nData summary:")
    for fam_key, pf_data in [("Adder", adder_gains),
                               ("QFT",   qft_gains),
                               ("GF",    gf_gains)]:
        for pk, vals in pf_data.items():
            if vals:
                print(f"  {fam_key:<6s}  {pk:<25s}  "
                      f"{np.mean(vals):+.1f}% ± {np.std(vals,ddof=1):.1f}%  "
                      f"(n={len(vals)})")


if __name__ == "__main__":
    build_figure()
