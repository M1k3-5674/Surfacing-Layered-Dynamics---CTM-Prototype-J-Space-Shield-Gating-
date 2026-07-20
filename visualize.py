"""
visualize.py
============
Visualization scripts for CalibratedTransitionModule outputs.

Plots produced
--------------
1. C heatmap         : pairwise affinity matrix with masked diagonal marker
2. A bar chart       : per-signal A_i weights
3. q histogram       : distribution of uncertainty scores with threshold bands
4. C off-diagonal histogram : distribution of off-diagonal affinity values
5. Routing Sankey    : flow of signals → admit / quarantine / reject
6. Threshold sweep   : quarantine_rate vs theta_low (from harness data)

Usage
-----
    # Quick demo with synthetic data:
    python visualize.py

    # Save all figures to a directory:
    python visualize.py --out-dir ./plots

    # Use harness JSON results for the threshold sweep:
    python visualize.py --harness-json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; change to "TkAgg" for interactive use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from calibrated_transition import CalibratedTransitionModule, RoutingResult

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
ADMIT_COLOR      = "#2ECC71"   # green
QUARANTINE_COLOR = "#F39C12"   # amber
REJECT_COLOR     = "#E74C3C"   # red
BG_COLOR         = "#F8F9FA"
GRID_COLOR       = "#DDE1E7"


# ---------------------------------------------------------------------------
# 1. C heatmap
# ---------------------------------------------------------------------------

def plot_C_heatmap(
    result: RoutingResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Affinity Matrix C",
) -> plt.Figure:
    """
    Heatmap of the n×n affinity matrix C.
    Diagonal cells are hatched to indicate masking in A_i computation.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    C = result.C
    n = C.shape[0]
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Affinity")

    # Hatch diagonal to show it's excluded from C_max
    for i in range(n):
        ax.add_patch(
            mpatches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, hatch="////",
                edgecolor="#999999", linewidth=0.5,
            )
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Signal index j")
    ax.set_ylabel("Signal index i")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Annotate cells for small matrices
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{C[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if C[i, j] > 0.6 else "black",
                )

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# 2. A bar chart
# ---------------------------------------------------------------------------

def plot_A_bar(
    result: RoutingResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Per-Signal Affinity Weight A_i",
) -> plt.Figure:
    """Bar chart of A_i coloured by routing decision."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    n = len(result.A)
    colors = []
    for d in result.decisions:
        if d == "admit":
            colors.append(ADMIT_COLOR)
        elif d == "quarantine":
            colors.append(QUARANTINE_COLOR)
        else:
            colors.append(REJECT_COLOR)

    bars = ax.bar(range(n), result.A, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1 / n, color="#555", linestyle="--", linewidth=0.8, label=f"Uniform (1/n={1/n:.3f})")
    ax.set_xlabel("Signal index")
    ax.set_ylabel("A_i weight")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

    legend_patches = [
        mpatches.Patch(color=ADMIT_COLOR,      label="Admit"),
        mpatches.Patch(color=QUARANTINE_COLOR, label="Quarantine"),
        mpatches.Patch(color=REJECT_COLOR,     label="Reject"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# 3. q histogram
# ---------------------------------------------------------------------------

def plot_q_histogram(
    result: RoutingResult,
    theta_low:  float = 0.3,
    theta_high: float = 0.7,
    ax: Optional[plt.Axes] = None,
    title: str = "Uncertainty Score Distribution (q)",
) -> plt.Figure:
    """
    Histogram of q values with shaded bands for each routing region.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    q = result.q
    bins = min(30, max(10, len(q) // 5))

    # Shaded threshold bands
    ax.axvspan(0,          theta_low,  alpha=0.12, color=ADMIT_COLOR,      label="Admit zone")
    ax.axvspan(theta_low,  theta_high, alpha=0.12, color=QUARANTINE_COLOR, label="Quarantine zone")
    ax.axvspan(theta_high, 1.0,        alpha=0.12, color=REJECT_COLOR,     label="Reject zone")

    # Threshold lines
    ax.axvline(theta_low,  color=ADMIT_COLOR,      linestyle="--", linewidth=1.2)
    ax.axvline(theta_high, color=REJECT_COLOR,      linestyle="--", linewidth=1.2)

    ax.hist(q, bins=bins, color="#4A90D9", edgecolor="white", zorder=3, alpha=0.85)
    ax.set_xlabel("q (uncertainty score)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)
    ax.legend(fontsize=8, loc="upper right")

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# 4. C off-diagonal histogram
# ---------------------------------------------------------------------------

def plot_C_histogram(
    result: RoutingResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Off-Diagonal Affinity Distribution (C)",
) -> plt.Figure:
    """Histogram of all off-diagonal C values (diagonal excluded)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    C = result.C
    n = C.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = C[mask].ravel()

    c_max = CalibratedTransitionModule.C_max_masked(C)
    ax.hist(off_diag, bins=30, color="#9B59B6", edgecolor="white", alpha=0.85)
    ax.axvline(c_max, color="#E74C3C", linestyle="--", linewidth=1.2,
               label=f"C_max (off-diag) = {c_max:.3f}")
    ax.axvline(off_diag.mean(), color="#3498DB", linestyle=":", linewidth=1.2,
               label=f"Mean = {off_diag.mean():.3f}")

    ax.set_xlabel("Affinity value")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# 5. Routing Sankey
# ---------------------------------------------------------------------------

def plot_sankey(
    result: RoutingResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Signal Routing Flow",
) -> plt.Figure:
    """
    Simplified Sankey-style flow diagram built with matplotlib rectangles
    and bezier arrows (no external sankey library required).
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    n     = len(result.signals)
    n_adm = len(result.admitted_idx)
    n_qua = len(result.quarantine_idx)
    n_rej = len(result.rejected_idx)

    fracs = [n_adm / n, n_qua / n, n_rej / n]
    labels   = ["Admit",       "Quarantine",       "Reject"]
    colors   = [ADMIT_COLOR, QUARANTINE_COLOR, REJECT_COLOR]
    counts   = [n_adm,      n_qua,             n_rej]

    # ---- source box (left)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.15), 0.18, 0.70,
        boxstyle="round,pad=0.02",
        facecolor="#4A90D9", edgecolor="white", linewidth=1.5, zorder=3,
    ))
    ax.text(0.14, 0.50, f"Signals\nn={n}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=4)

    # ---- destination boxes (right) with proportional height
    y_positions = [0.70, 0.40, 0.10]   # admit, quarantine, reject (top to bottom)
    box_h = 0.22

    for i, (label, color, count, frac, y) in enumerate(
        zip(labels, colors, counts, fracs, y_positions)
    ):
        h = max(frac * 0.70, 0.04)     # proportional height, min 4%
        y_center = y + box_h / 2

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.72, y), 0.23, box_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
        ))
        ax.text(0.835, y + box_h / 2, f"{label}\n{count} ({frac*100:.0f}%)",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=4)

        # Arrow from source to destination
        ax.annotate(
            "",
            xy=(0.72, y + box_h / 2),
            xytext=(0.23, 0.50),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=max(frac * 8, 0.5),
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=2,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", pad=10)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# 6. Threshold sweep
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    harness_results: Optional[List[dict]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Quarantine Rate vs. θ_low",
) -> plt.Figure:
    """
    Line chart of quarantine_rate vs theta_low from harness data.
    If no harness_results are provided, generates fresh data inline.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    if harness_results is None:
        # Generate inline
        sys.path.insert(0, os.path.dirname(__file__))
        from harness import run_threshold_sweep
        sweep = run_threshold_sweep()
        harness_results = [
            {"suite": r.suite, "quarantine_rate": r.quarantine_rate,
             "admit_rate": r.admit_rate, "reject_rate": r.reject_rate}
            for r in sweep
        ]

    # Filter to threshold_sweep rows
    rows = [r for r in harness_results if r.get("suite", "").startswith("threshold")]
    if not rows:
        ax.text(0.5, 0.5, "No threshold sweep data found",
                ha="center", va="center", transform=ax.transAxes)
        if fig:
            fig.tight_layout()
        return fig or ax.figure

    # Extract theta_low from suite name "threshold_sweep/tl=X.XX"
    theta_vals = []
    for r in rows:
        try:
            tl = float(r["suite"].split("tl=")[1])
            theta_vals.append(tl)
        except (IndexError, ValueError):
            theta_vals.append(float("nan"))

    theta_arr = np.array(theta_vals)
    q_rate    = np.array([r["quarantine_rate"] for r in rows])
    a_rate    = np.array([r["admit_rate"]       for r in rows])
    r_rate    = np.array([r["reject_rate"]      for r in rows])

    order = np.argsort(theta_arr)
    theta_arr, q_rate, a_rate, r_rate = (
        theta_arr[order], q_rate[order], a_rate[order], r_rate[order]
    )

    ax.plot(theta_arr, q_rate, "o-", color=QUARANTINE_COLOR, linewidth=2,
            label="Quarantine rate")
    ax.plot(theta_arr, a_rate, "s--", color=ADMIT_COLOR,      linewidth=1.5,
            label="Admit rate")
    ax.plot(theta_arr, r_rate, "^--", color=REJECT_COLOR,     linewidth=1.5,
            label="Reject rate")

    ax.set_xlabel("θ_low (theta_low)")
    ax.set_ylabel("Fraction of signals")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.set_facecolor(BG_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR)

    if fig:
        fig.tight_layout()
    return fig or ax.figure


# ---------------------------------------------------------------------------
# Dashboard: all plots on one figure
# ---------------------------------------------------------------------------

def plot_dashboard(
    result: RoutingResult,
    theta_low:  float = 0.3,
    theta_high: float = 0.7,
    harness_results: Optional[List[dict]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a 2×3 dashboard with all six visualisations.

    Parameters
    ----------
    result : RoutingResult
        Output from CalibratedTransitionModule.route().
    theta_low, theta_high : float
        Threshold values used — passed to q histogram for band shading.
    harness_results : list of dicts, optional
        JSON-loaded harness output for the threshold sweep panel.
    save_path : str, optional
        If given, save the figure to this path (PNG / PDF etc.).
    """
    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_C_heatmap(result,    ax=ax1)
    plot_A_bar(result,        ax=ax2)
    plot_sankey(result,       ax=ax3)
    plot_q_histogram(result,  theta_low=theta_low, theta_high=theta_high, ax=ax4)
    plot_C_histogram(result,  ax=ax5)
    plot_threshold_sweep(harness_results=harness_results, ax=ax6)

    fig.suptitle(
        "CalibratedTransitionModule — Analysis Dashboard",
        fontsize=14, fontweight="bold", y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"Dashboard saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_demo_result(n: int = 30, seed: int = 0) -> RoutingResult:
    rng = np.random.default_rng(seed)
    signals = rng.uniform(0, 1, n)
    ctm = CalibratedTransitionModule(theta_low=0.3, theta_high=0.7)
    return ctm.route(signals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTM Visualization scripts")
    parser.add_argument("--out-dir",     default=None,
                        help="Directory to save individual plot PNGs.")
    parser.add_argument("--dashboard",   default=None,
                        help="Path to save the combined dashboard PNG.")
    parser.add_argument("--harness-json", default=None,
                        help="Path to harness JSON results for threshold sweep.")
    parser.add_argument("--n", type=int, default=30, help="Demo signal count.")
    args = parser.parse_args()

    harness_results = None
    if args.harness_json and os.path.exists(args.harness_json):
        with open(args.harness_json) as fh:
            harness_results = json.load(fh)

    result = _make_demo_result(n=args.n)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        plots = {
            "C_heatmap.png":    lambda: plot_C_heatmap(result),
            "A_bar.png":        lambda: plot_A_bar(result),
            "q_histogram.png":  lambda: plot_q_histogram(result, 0.3, 0.7),
            "C_histogram.png":  lambda: plot_C_histogram(result),
            "sankey.png":       lambda: plot_sankey(result),
            "threshold_sweep.png": lambda: plot_threshold_sweep(harness_results),
        }
        for fname, fn in plots.items():
            fig = fn()
            path = os.path.join(args.out_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")
    elif args.dashboard:
        plot_dashboard(result, harness_results=harness_results, save_path=args.dashboard)
    else:
        # Default: save dashboard to cwd
        plot_dashboard(
            result,
            harness_results=harness_results,
            save_path="ctm_dashboard.png",
        )
