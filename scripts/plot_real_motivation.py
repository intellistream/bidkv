#!/usr/bin/env python3
"""Plot motivation figure from REAL preemption event data.

Reads the JSONL output of preemption_logger.py (captured during live vLLM serving)
and generates motivation figures.

The reclamation opportunity score R_i = B_i * (1 - p_i) is used as an
analysis-only, method-agnostic proxy:
  B_i = tokens_freed  (KV tokens currently held; freed upon eviction)
  p_i = completion_ratio = num_output_tokens / max_output_tokens
        (online-observable; not a BidKV-specific parameter)

The spread metric (max R / min R) expresses how widely candidate opportunities
vary at each KV-pressure event across both capacity AND progress dimensions.

Modes:
  --panels 1  Single panel: CDF of eviction-opportunity spread (recommended)
  --panels 2  Two panels: (a) spread CDF + (b) per-event LIFO vs BidKV scatter
  --panels 3  Three panels: (a) CDF + (b) cumulative freed + (c) per-eviction CDF

Usage:
    python scripts/plot_real_motivation.py \\
        --log-file results/motivation_real_rate3.8/preemption_events.jsonl \\
        --output paper/figures/fig_motivation_1panel.pdf \\
        --rate 3.8 --panels 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib + numpy required.  pip install matplotlib numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Reclamation opportunity score R (analysis-only, method-agnostic)
# ---------------------------------------------------------------------------

def compute_R(candidate: dict, completions: dict[str, int] | None = None) -> float:
    """Compute reclamation opportunity score R = B * (1 - p).

    B = tokens_freed: KV tokens currently held by the candidate; all freed
        upon preemption under recompute-fallback semantics.
    p = completion fraction:
        If `completions` dict is provided, p = g_i / G_i where G_i is the
        request's actual final output length (ground-truth denominator).
        Otherwise falls back to completion_ratio = g / max_output_tokens
        (generation budget estimate).

    Interpretation: R is high when a candidate occupies large KV footprint
    AND has not yet invested much decode work---the classic "high reward, low
    waste" eviction target.  R does not reference any scheduling formula
    (no w_c, w_P, delta, or epsilon); it is used only to characterise the
    spread of reclamation opportunities at each KV-pressure event.
    """
    B = float(candidate["tokens_freed"])
    if completions is not None:
        rid = candidate["request_id"]
        G_true = completions.get(rid, 0)
        if G_true > 0:
            g = float(candidate["num_output_tokens"])
            p = g / G_true
        else:
            p = float(candidate["completion_ratio"])
    else:
        p = float(candidate["completion_ratio"])   # = g / max_output_tokens
    return B * (1.0 - p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list[dict]:
    """Load preemption events from JSONL file."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def load_completions(path: Path) -> dict[str, int]:
    """Load ground-truth final output lengths from completions.jsonl.

    Returns a dict mapping request_id -> final_output_tokens.
    The completions file is produced by the same experiment run as the
    preemption events log, so request IDs match exactly.
    """
    completions: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            completions[r["request_id"]] = int(r["final_output_tokens"])
    return completions


def analyze_events(events: list[dict], completions: dict[str, int] | None = None) -> dict:
    """Analyze events and compute per-strategy metrics.

    Args:
        events: Preemption events loaded from JSONL.
        completions: Optional dict mapping request_id -> final_output_tokens.
            When provided, R uses the ground-truth actual output length as the
            completion-fraction denominator instead of max_output_tokens.

    Returns dict with:
      - utility_ratios: list of max_U/min_U per event (for Panel A)
      - per_strategy: dict[strategy_name] -> {ts, freed, deltas, utilities, ...}
    """
    spread_ratios: list[float] = []  # max S / min S per event
    n_candidates_list: list[int] = []

    strategy_names = ["pe-lifo", "largest-first", "pe-sjf", "bidkv"]
    per_strategy: dict[str, dict] = {
        name: {"ts": [], "freed": [], "completion_ratios": []}
        for name in strategy_names
    }

    # Also collect per-event paired freed (for scatter plot: Panel B in 2-panel mode)
    per_event_paired: list[dict] = []

    t0 = events[0]["ts"] if events else 0

    for ev in events:
        candidates = ev["candidates"]
        if len(candidates) < 2:
            continue

        # Compute method-agnostic reclamation opportunity R = B*(1-p)
        # Skip events where any candidate is missing from completions (safety)
        if completions is not None:
            if any(c["request_id"] not in completions for c in candidates):
                continue
        scores = [compute_R(c, completions) for c in candidates]
        min_r = min(scores)
        max_r = max(scores)
        # Floor denominator at 1.0 to avoid division by near-zero
        # (can occur when a newly-started request has p≈1 due to short max_tokens)
        ratio = max_r / max(min_r, 1.0)
        spread_ratios.append(ratio)
        n_candidates_list.append(len(candidates))

        # Build request_id → candidate lookup
        cand_map = {c["request_id"]: c for c in candidates}

        # For each strategy, record the victim's characteristics
        choices = ev.get("strategy_choices", {})
        ts_rel = ev["ts"] - t0  # seconds from start

        lifo_freed = None
        bidkv_freed = None
        for sname in strategy_names:
            victim_id = choices.get(sname)
            if victim_id and victim_id in cand_map:
                victim = cand_map[victim_id]
                per_strategy[sname]["ts"].append(ts_rel)
                per_strategy[sname]["freed"].append(victim["tokens_freed"])
                per_strategy[sname]["completion_ratios"].append(
                    victim["completion_ratio"])
                if sname == "pe-lifo":
                    lifo_freed = victim["tokens_freed"]
                elif sname == "bidkv":
                    bidkv_freed = victim["tokens_freed"]

        if lifo_freed is not None and bidkv_freed is not None:
            per_event_paired.append({
                "lifo_freed": lifo_freed,
                "bidkv_freed": bidkv_freed,
            })

    return {
        "spread_ratios": spread_ratios,   # max R / min R per event
        "n_candidates": n_candidates_list,
        "per_strategy": per_strategy,
        "per_event_paired": per_event_paired,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "pe-lifo": "LIFO (vLLM default)",
    "largest-first": "Largest-First",
    "pe-sjf": "SJF",
    "bidkv": "BidKV (ours)",
}

COLORS = {
    "pe-lifo": "#D63027",
    "largest-first": "#f39c12",
    "pe-sjf": "#8e44ad",
    "bidkv": "#2980b9",
}

LINE_STYLES = {
    "pe-lifo": "-",
    "largest-first": "--",
    "pe-sjf": ":",
    "bidkv": "-",
}

# Only show these 3 strategies (main comparison in paper)
PLOT_STRATEGIES = ["pe-lifo", "largest-first", "bidkv"]


def _setup_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "figure.dpi": 200,
    })


def _plot_panel_a(ax: plt.Axes, analysis: dict, rate: float, n_events: int) -> None:
    """Panel A: CDF of reclamation-opportunity spread (max R / min R).

    R = tokens_freed * (1 - completion_ratio) is the method-agnostic proxy.
    """
    sr = np.array(analysis["spread_ratios"])
    if len(sr) == 0:
        return

    # Colour gradient: blue (low spread) → orange (high spread)
    CURVE_COLOR = "#2471a3"   # steel blue — main CDF line
    FILL_COLOR  = "#aed6f1"   # light blue — fill under curve
    VLINE_COLOR = "#e67e22"   # orange     — threshold reference line
    MEDIAN_COLOR = "#1a5276"  # dark blue  — median marker

    sorted_r = np.sort(sr)
    cdf_y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, cdf_y, color=CURVE_COLOR, linewidth=1.8)
    ax.fill_between(sorted_r, 0, cdf_y, alpha=0.22, color=FILL_COLOR)

    # Annotate 10× threshold (100% >= 5x so use 10x as the annotation anchor)
    thresh = 10.0
    if thresh <= sorted_r[-1]:
        pct = np.mean(sr >= thresh) * 100
        yp = np.searchsorted(sorted_r, thresh) / len(sorted_r)
        ax.axvline(thresh, color=VLINE_COLOR, linestyle="-.", linewidth=1.0)
        ax.annotate(
            f"{pct:.1f}% of events have\n$\\geq${thresh:.0f}$\\times$ spread",
            xy=(thresh, yp),
            xytext=(thresh + max(1.5, (sorted_r[-1] - sorted_r[0]) * 0.08), 0.30),
            fontsize=6, color=VLINE_COLOR, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=VLINE_COLOR, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9,
                      ec=VLINE_COLOR, lw=0.7),
        )

    median_r = np.median(sr)
    ax.axvline(median_r, color=MEDIAN_COLOR, linestyle="--", linewidth=0.9, alpha=0.8)
    ax.text(
        median_r + 0.3, 0.95, f"median={median_r:.1f}$\\times$",
        fontsize=5.5, color=MEDIAN_COLOR, alpha=0.9,
        transform=ax.get_xaxis_transform(), va="top",
    )

    avg_cand = np.mean(analysis["n_candidates"]) if analysis["n_candidates"] else 0
    ax.set_xlabel("Reclamation opportunity spread ($\\max R$ / $\\min R$)")
    ax.set_ylabel("CDF")
    ax.set_title(
        f"(a) Victim heterogeneity under KV pressure\n"
        f"({n_events} events, avg {avg_cand:.0f} candidates, rate={rate} req/s)",
        pad=3,
    )
    ax.set_xlim(left=1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)


def _plot_panel_b_scatter(ax: plt.Axes, analysis: dict) -> None:
    """Panel B (2-panel mode): Per-event scatter — LIFO freed vs BidKV freed."""
    paired = analysis["per_event_paired"]
    if not paired:
        return

    lifo_f = np.array([p["lifo_freed"] for p in paired])
    bidkv_f = np.array([p["bidkv_freed"] for p in paired])

    # Scatter: each dot = one KV pressure event
    ax.scatter(lifo_f, bidkv_f, s=12, alpha=0.45, color=COLORS["bidkv"],
               edgecolors="none", zorder=3)

    # y=x reference line
    max_val = max(lifo_f.max(), bidkv_f.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], color="#999", linestyle="--",
            linewidth=0.7, alpha=0.6, zorder=1)

    # Annotate: points above line = BidKV frees more
    above = np.sum(bidkv_f > lifo_f)
    pct_above = above / len(paired) * 100
    ax.text(
        0.97, 0.05,
        f"{pct_above:.0f}% events:\nBidKV frees more",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color=COLORS["bidkv"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"),
    )

    # Median lines
    med_l = np.median(lifo_f)
    med_b = np.median(bidkv_f)
    ax.axvline(med_l, color=COLORS["pe-lifo"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax.axhline(med_b, color=COLORS["bidkv"], linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(med_l, max_val * 0.97, f"LIFO med={int(med_l)}",
            fontsize=5, color=COLORS["pe-lifo"], ha="left", va="top", rotation=90)
    ax.text(max_val * 0.02, med_b + max_val * 0.02, f"BidKV med={int(med_b)}",
            fontsize=5, color=COLORS["bidkv"], ha="left", va="bottom")

    ax.set_xlabel("LIFO victim: KV tokens freed")
    ax.set_ylabel("BidKV victim: KV tokens freed")
    ax.set_title("(b) Per-event victim comparison", pad=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)


def _plot_panel_b_cumulative(ax: plt.Axes, analysis: dict) -> None:
    """Panel B (3-panel mode): Cumulative KV tokens freed."""
    ps = analysis["per_strategy"]
    for sname in PLOT_STRATEGIES:
        s = ps[sname]
        if s["ts"]:
            ts_min = np.array(s["ts"]) / 60
            cum_freed = np.cumsum(s["freed"])
            ax.plot(ts_min, cum_freed / 1000,
                    color=COLORS[sname], linestyle=LINE_STYLES[sname],
                    linewidth=1.3, label=DISPLAY_NAMES[sname])

    s_l, s_b = ps["pe-lifo"], ps["bidkv"]
    if s_l["ts"] and s_b["ts"]:
        cum_fl = np.cumsum(s_l["freed"])
        cum_fb = np.cumsum(s_b["freed"])
        if cum_fl[-1] > 0:
            ratio = cum_fb[-1] / cum_fl[-1]
            t_max = min(s_l["ts"][-1], s_b["ts"][-1])
            tg = np.linspace(0, t_max, 300) / 60
            li = np.interp(tg * 60, s_l["ts"], cum_fl) / 1000
            bi = np.interp(tg * 60, s_b["ts"], cum_fb) / 1000
            ax.fill_between(tg, li, bi, alpha=0.08, color=COLORS["bidkv"])
            mid_idx = int(len(tg) * 0.55)
            mid_y = (bi[mid_idx] + li[mid_idx]) / 2
            ax.annotate(
                f"{ratio:.0f}$\\times$",
                xy=(tg[mid_idx], mid_y),
                fontsize=8, color=COLORS["bidkv"], fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9,
                          ec=COLORS["bidkv"], lw=0.5),
            )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Cumulative KV tokens freed (×1000)")
    ax.set_title("(b) KV space recovered per strategy", pad=3)
    ax.legend(loc="upper left", framealpha=0.9, handletextpad=0.3, borderpad=0.4)
    ax.grid(True, linestyle=":", alpha=0.4)


def _plot_panel_c_cdf(ax: plt.Axes, analysis: dict) -> None:
    """Panel C (3-panel mode): CDF of per-eviction freed tokens."""
    ps = analysis["per_strategy"]
    for sname in PLOT_STRATEGIES:
        s = ps[sname]
        if s["freed"]:
            sorted_f = np.sort(s["freed"])
            cdf = np.arange(1, len(sorted_f) + 1) / len(sorted_f)
            ax.plot(sorted_f, cdf,
                    color=COLORS[sname], linestyle=LINE_STYLES[sname],
                    linewidth=1.3, label=DISPLAY_NAMES[sname])

    for sname in PLOT_STRATEGIES:
        s = ps[sname]
        if s["freed"]:
            med = np.median(s["freed"])
            ax.axvline(med, color=COLORS[sname], linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("KV tokens freed per eviction")
    ax.set_ylabel("CDF")
    ax.set_title("(c) Per-eviction KV reclamation (victim quality)", pad=3)
    ax.legend(loc="lower right", framealpha=0.9, handletextpad=0.3, borderpad=0.4)
    ax.grid(True, linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Top-level plot drivers
# ---------------------------------------------------------------------------

def plot_1panel(analysis: dict, output_path: Path, rate: float, n_events: int) -> None:
    _setup_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(3.33, 2.2))
    _plot_panel_a(ax, analysis, rate, n_events)
    # Remove (a) prefix for standalone figure
    title = ax.get_title().replace("(a) ", "")
    ax.set_title(title, pad=3)
    _save_fig(fig, output_path)


def plot_2panel(analysis: dict, output_path: Path, rate: float, n_events: int) -> None:
    _setup_rcparams()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.8, 2.6),
        gridspec_kw={"wspace": 0.45},
    )
    _plot_panel_a(ax_a, analysis, rate, n_events)
    _plot_panel_b_scatter(ax_b, analysis)
    _save_fig(fig, output_path)


def plot_3panel(analysis: dict, output_path: Path, rate: float, n_events: int) -> None:
    _setup_rcparams()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(3.33, 5.8),
        gridspec_kw={"hspace": 0.65},
    )
    _plot_panel_a(ax_a, analysis, rate, n_events)
    _plot_panel_b_cumulative(ax_b, analysis)
    _plot_panel_c_cdf(ax_c, analysis)
    fig.align_ylabels([ax_a, ax_b, ax_c])
    _save_fig(fig, output_path)


def _save_fig(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"  PDF → {output_path}")
    png = output_path.with_suffix(".png")
    plt.savefig(png, bbox_inches="tight", format="png", dpi=300)
    print(f"  PNG → {png}")


def print_summary(analysis: dict) -> None:
    sr = np.array(analysis["spread_ratios"])
    ps = analysis["per_strategy"]

    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS (for paper §2)")
    print("=" * 65)

    print(f"\nPanel A — Reclamation-Opportunity Spread ({len(sr)} events):")
    print("  (R = tokens_freed * (1 - completion_ratio), method-agnostic proxy)")
    if len(sr) > 0:
        print(f"  Median spread:    {np.median(sr):.1f}×")
        print(f"  Mean spread:      {np.mean(sr):.1f}×")
        for th in [3, 5, 10, 20]:
            print(f"  Events ≥ {th:>2d}×:    {np.mean(sr >= th) * 100:.1f}%")

    print(f"\n{'Strategy':<22} {'ΣFreed':>10} {'MedianFreed':>12} {'AvgComp':>8}")
    print("-" * 56)
    for sname in PLOT_STRATEGIES:
        s = ps[sname]
        sf = sum(s["freed"])
        med_f = int(np.median(s["freed"])) if s["freed"] else 0
        avg_c = np.mean(s["completion_ratios"]) if s["completion_ratios"] else 0
        print(f"{DISPLAY_NAMES[sname]:<22} {sf:>10,} {med_f:>12,} {avg_c:>8.3f}")

    # Cross-strategy ratios
    if ps["bidkv"]["freed"] and ps["pe-lifo"]["freed"]:
        bidkv_f = sum(ps["bidkv"]["freed"])
        lifo_f = sum(ps["pe-lifo"]["freed"])
        print(f"\n  BidKV/LIFO freed ratio:    {bidkv_f / lifo_f:.1f}×")
        print(f"  BidKV median freed:        {int(np.median(ps['bidkv']['freed'])):,}")
        print(f"  LIFO median freed:         {int(np.median(ps['pe-lifo']['freed'])):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real-data motivation figure")
    parser.add_argument("--log-file", required=True, type=Path,
                        help="Path to preemption_events.jsonl")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PDF path")
    parser.add_argument("--rate", required=True, type=float,
                        help="Request rate (for title)")
    parser.add_argument("--panels", type=int, default=2, choices=[1, 2, 3],
                        help="Number of panels: 1=CDF only, 2=CDF+scatter, 3=full")
    parser.add_argument("--completions-file", type=Path, default=None,
                        help="Path to completions.jsonl with ground-truth final_output_tokens")
    args = parser.parse_args()

    print(f"Loading events from {args.log_file} ...")
    events = load_events(args.log_file)
    print(f"  Loaded {len(events)} events")

    completions: dict[str, int] | None = None
    if args.completions_file is not None:
        print(f"Loading ground-truth completions from {args.completions_file} ...")
        completions = load_completions(args.completions_file)
        print(f"  Loaded {len(completions)} completion records")

    print("Analyzing events ...")
    analysis = analyze_events(events, completions)
    print(f"  spread ratios: {len(analysis['spread_ratios'])} events")

    print_summary(analysis)

    print(f"\nGenerating {args.panels}-panel figure → {args.output} ...")
    if args.panels == 1:
        plot_1panel(analysis, args.output, args.rate, len(events))
    elif args.panels == 2:
        plot_2panel(analysis, args.output, args.rate, len(events))
    else:
        plot_3panel(analysis, args.output, args.rate, len(events))
    print("Done.")


if __name__ == "__main__":
    main()
