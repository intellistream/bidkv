#!/usr/bin/env python3
"""Generate 3-panel preliminary motivation figure for BidKV SC 2026 §2.

Reads victim_heterogeneity.jsonl captured by preemption_logger.py and
produces a 3-panel column-width figure:

  Panel A — Reclamation Cost Heterogeneity (CDF of max_δ/min_δ per event)
  Panel B — Policy Divergence (δ_LIFO / δ_BidKV per event, sorted)
  Panel C — Cumulative Recompute Waste (two curves over time)

Usage:
    python scripts/plot_motivation_3panel.py \
        --log-file results/preliminary_motivation/victim_heterogeneity.jsonl \
        --output paper/figures/fig_preliminary_motivation.pdf

    # Fallback: synthetic data for development (no GPU needed)
    python scripts/plot_motivation_3panel.py --synthetic \
        --output paper/figures/fig_preliminary_motivation.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib + numpy required.  pip install matplotlib numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(log_path: Path) -> list[dict]:
    """Load and validate events from victim_heterogeneity.jsonl."""
    events: list[dict] = []
    with open(log_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARN: skipping line {line_no}: {exc}")
                continue
            cands = ev.get("candidates", [])
            choices = ev.get("strategy_choices", {})
            if len(cands) >= 2 and "bidkv" in choices and "pe-lifo" in choices:
                events.append(ev)
    print(f"  Loaded {len(events)} valid events from {log_path}")
    return events


def compute_delta(c: dict) -> float:
    """Compute disruption estimate δ for a candidate (same formula as BidKV)."""
    output_tokens = c.get("num_output_tokens", 0)
    num_prompt = c.get("num_prompt_tokens", 0)
    num_preemptions = c.get("num_preemptions", 0)
    completion_ratio = c.get("completion_ratio", 0.0)

    if output_tokens > 2:
        recompute_norm = max(0.5, num_prompt / 256.0)
        late_penalty = completion_ratio * completion_ratio * 2.0
        starvation = num_preemptions * 0.5
        return max(0.1, recompute_norm + late_penalty + starvation)
    else:
        return max(0.1, 1.0 + num_preemptions * 0.5)


def compute_utility(c: dict) -> float:
    """Compute utility U = tokens_freed / (δ + ε) for a candidate."""
    output_tokens = c.get("num_output_tokens", 0)
    tokens_freed = c.get("tokens_freed", c.get("num_computed_tokens", 0))
    delta = compute_delta(c)
    eps = 1e-3
    if output_tokens > 2:
        return output_tokens / (delta + eps)
    else:
        return tokens_freed / (delta + eps)


def get_candidate_by_id(cands: list[dict], rid: str | None) -> dict | None:
    """Find candidate dict by request_id."""
    if rid is None:
        return None
    for c in cands:
        if c.get("request_id") == rid:
            return c
    return None


# ---------------------------------------------------------------------------
# Analysis: extract per-event metrics
# ---------------------------------------------------------------------------

def analyze_events(events: list[dict]) -> dict:
    """Compute per-event metrics for the 3 panels."""

    # Panel A: heterogeneity ratio per event
    delta_ratios: list[float] = []    # max_δ / min_δ
    utility_ratios: list[float] = []  # max_U / min_U

    # Panel B: cost divergence per event
    cost_ratios: list[float] = []     # δ_LIFO / δ_BidKV

    # Panel C: cumulative recompute waste
    cum_lifo: list[float] = []
    cum_bidkv: list[float] = []
    cum_largest: list[float] = []
    timestamps: list[float] = []

    running_lifo = 0.0
    running_bidkv = 0.0
    running_largest = 0.0
    t0 = events[0]["ts"] if events else 0.0

    for ev in events:
        cands = ev["candidates"]
        choices = ev.get("strategy_choices", {})

        # Compute δ for all candidates
        deltas = [compute_delta(c) for c in cands]
        utilities = [compute_utility(c) for c in cands]

        # Panel A
        d_min, d_max = min(deltas), max(deltas)
        if d_min > 0:
            delta_ratios.append(d_max / d_min)
        u_min, u_max = min(utilities), max(utilities)
        if u_min > 0:
            utility_ratios.append(u_max / u_min)

        # Panel B: compare LIFO vs BidKV choices
        lifo_rid = choices.get("pe-lifo")
        bidkv_rid = choices.get("bidkv")
        lifo_cand = get_candidate_by_id(cands, lifo_rid)
        bidkv_cand = get_candidate_by_id(cands, bidkv_rid)

        if lifo_cand is not None and bidkv_cand is not None:
            d_lifo = compute_delta(lifo_cand)
            d_bidkv = compute_delta(bidkv_cand)
            if d_bidkv > 0:
                cost_ratios.append(d_lifo / d_bidkv)

        # Panel C: cumulative waste
        # "Recompute waste" ≈ number of output tokens that must be recomputed
        # after preemption = output_tokens_of_victim
        largest_rid = choices.get("largest-first")
        largest_cand = get_candidate_by_id(cands, largest_rid)

        if lifo_cand is not None:
            running_lifo += lifo_cand.get("num_output_tokens", 0)
        if bidkv_cand is not None:
            running_bidkv += bidkv_cand.get("num_output_tokens", 0)
        if largest_cand is not None:
            running_largest += largest_cand.get("num_output_tokens", 0)

        cum_lifo.append(running_lifo)
        cum_bidkv.append(running_bidkv)
        cum_largest.append(running_largest)
        timestamps.append(ev["ts"] - t0)

    return {
        "delta_ratios": delta_ratios,
        "utility_ratios": utility_ratios,
        "cost_ratios": cost_ratios,
        "cum_lifo": cum_lifo,
        "cum_bidkv": cum_bidkv,
        "cum_largest": cum_largest,
        "timestamps": timestamps,
        "n_events": len(events),
    }


# ---------------------------------------------------------------------------
# Synthetic data (for development without GPU)
# ---------------------------------------------------------------------------

def generate_synthetic_data() -> dict:
    """Generate plausible synthetic data matching expected distributions."""
    rng = np.random.default_rng(42)
    n_events = 200

    # Panel A: heterogeneity ratios — log-normal distributed
    delta_ratios = list(rng.lognormal(mean=1.5, sigma=0.8, size=n_events))
    delta_ratios = [max(1.0, r) for r in delta_ratios]  # floor at 1.0
    utility_ratios = list(rng.lognormal(mean=2.0, sigma=0.9, size=n_events))
    utility_ratios = [max(1.0, r) for r in utility_ratios]

    # Panel B: cost ratios — LIFO tends to pick ~2-5x more expensive
    cost_ratios = list(rng.lognormal(mean=0.7, sigma=0.6, size=n_events))
    cost_ratios = [max(0.3, r) for r in cost_ratios]

    # Panel C: cumulative waste — LIFO accumulates faster
    cum_lifo, cum_bidkv, cum_largest = [], [], []
    r_lifo = r_bidkv = r_largest = 0.0
    timestamps = []
    for i in range(n_events):
        t = i * 1.5  # ~1.5s between events
        r_lifo += rng.exponential(80)
        r_bidkv += rng.exponential(30)
        r_largest += rng.exponential(55)
        cum_lifo.append(r_lifo)
        cum_bidkv.append(r_bidkv)
        cum_largest.append(r_largest)
        timestamps.append(t)

    return {
        "delta_ratios": delta_ratios,
        "utility_ratios": utility_ratios,
        "cost_ratios": cost_ratios,
        "cum_lifo": cum_lifo,
        "cum_bidkv": cum_bidkv,
        "cum_largest": cum_largest,
        "timestamps": timestamps,
        "n_events": n_events,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3panel(data: dict, output_path: Path, is_synthetic: bool = False) -> None:
    """Render the 3-panel motivation figure."""
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

    FIG_W = 3.33  # ACM sigconf single column
    FIG_H = 5.8
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H),
        gridspec_kw={"hspace": 0.65}
    )

    # Colors
    COL_LIFO = "#D63027"
    COL_BIDKV = "#2980b9"
    COL_LARGEST = "#f39c12"
    COL_LIGHT = "#999999"

    delta_ratios = np.array(data["delta_ratios"])
    cost_ratios = np.array(data["cost_ratios"])
    cum_lifo = np.array(data["cum_lifo"])
    cum_bidkv = np.array(data["cum_bidkv"])
    cum_largest = np.array(data["cum_largest"])
    timestamps = np.array(data["timestamps"])
    n_events = data["n_events"]

    # ── Panel A: CDF of δ heterogeneity ratio ──────────────────────────
    sorted_ratios = np.sort(delta_ratios)
    cdf_y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

    ax_a.plot(sorted_ratios, cdf_y, color=COL_BIDKV, linewidth=1.5)
    ax_a.fill_between(sorted_ratios, 0, cdf_y, alpha=0.15, color=COL_BIDKV)

    # Reference lines
    for thresh, ls in [(3.0, "--"), (5.0, ":"), (10.0, "-.")]:
        if thresh <= sorted_ratios[-1]:
            pct_above = np.mean(delta_ratios >= thresh) * 100
            ax_a.axvline(thresh, color=COL_LIGHT, linestyle=ls, linewidth=0.6)
            y_pos = np.searchsorted(sorted_ratios, thresh) / len(sorted_ratios)
            ax_a.annotate(
                f"{pct_above:.0f}% $\\geq$ {thresh:.0f}$\\times$",
                xy=(thresh, y_pos), xytext=(thresh + 0.5, y_pos - 0.08),
                fontsize=5.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4),
            )

    ax_a.set_xlabel("Disruption cost ratio (max $\\delta$ / min $\\delta$) per event")
    ax_a.set_ylabel("CDF")
    ax_a.set_title(
        f"(a) Reclamation cost heterogeneity\n({n_events} KV-pressure events)",
        pad=3
    )
    ax_a.set_xlim(left=1.0)
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, axis="both", linestyle=":", alpha=0.4)

    # ── Panel B: Per-event cost ratio (LIFO / BidKV) ──────────────────
    sorted_cr = np.sort(cost_ratios)
    n_cr = len(sorted_cr)

    # Bar-like visualization: sorted ratios as a step function
    x_idx = np.arange(n_cr)
    colors_b = [COL_LIFO if r > 1.0 else COL_BIDKV for r in sorted_cr]

    ax_b.bar(x_idx, sorted_cr, width=1.0, color=colors_b, alpha=0.7,
             edgecolor="none")
    ax_b.axhline(1.0, color="black", linewidth=0.8, linestyle="-",
                 label="Equal cost (ratio = 1)")

    # Statistics
    pct_lifo_worse = np.mean(cost_ratios > 1.0) * 100
    median_ratio = float(np.median(cost_ratios))
    ax_b.text(
        0.98, 0.95,
        f"LIFO costlier in {pct_lifo_worse:.0f}% of events\n"
        f"Median ratio: {median_ratio:.1f}$\\times$",
        transform=ax_b.transAxes, fontsize=5.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"),
    )

    ax_b.set_xlabel("Pressure events (sorted by cost ratio)")
    ax_b.set_ylabel("$\\delta_{\\mathrm{LIFO}}\\, /\\, \\delta_{\\mathrm{BidKV}}$")
    ax_b.set_title("(b) Per-event victim cost: LIFO vs. BidKV", pad=3)
    ax_b.set_xlim(-0.5, n_cr - 0.5)
    ax_b.set_xticks([])
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.4)

    # ── Panel C: Cumulative recompute waste ────────────────────────────
    ts_min = timestamps / 60.0  # convert to minutes

    ax_c.plot(ts_min, cum_lifo / 1000, color=COL_LIFO, linewidth=1.3,
              label="LIFO (vLLM default)")
    ax_c.plot(ts_min, cum_largest / 1000, color=COL_LARGEST, linewidth=1.1,
              linestyle="--", label="Largest-First")
    ax_c.plot(ts_min, cum_bidkv / 1000, color=COL_BIDKV, linewidth=1.3,
              label="BidKV (ours)")
    ax_c.fill_between(ts_min, cum_bidkv / 1000, cum_lifo / 1000,
                       alpha=0.08, color=COL_LIFO)

    # Final ratio annotation
    if cum_bidkv[-1] > 0:
        ratio_final = cum_lifo[-1] / cum_bidkv[-1]
        ax_c.annotate(
            f"{ratio_final:.1f}$\\times$ more\nwasted recompute",
            xy=(ts_min[-1], cum_lifo[-1] / 1000),
            xytext=(ts_min[-1] * 0.65, cum_lifo[-1] / 1000 * 0.85),
            fontsize=5.5, color=COL_LIFO, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL_LIFO, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"),
        )

    ax_c.set_xlabel("Time (minutes)")
    ax_c.set_ylabel("Cumulative recompute waste ($\\times$1K tokens)")
    ax_c.set_title("(c) Accumulated victim recompute cost over time", pad=3)
    ax_c.legend(loc="upper left", framealpha=0.9, handletextpad=0.3,
                borderpad=0.4, ncol=1)
    ax_c.grid(True, axis="both", linestyle=":", alpha=0.4)

    # ── Save ──────────────────────────────────────────────────────────
    fig.align_ylabels([ax_a, ax_b, ax_c])

    if is_synthetic:
        fig.text(0.5, 0.01, "[SYNTHETIC DATA — for layout preview only]",
                 ha="center", fontsize=6, color="red", alpha=0.7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"\nFigure saved → {output_path}")

    # Also save PNG for preview
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", format="png", dpi=300)
    print(f"PNG preview → {png_path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(data: dict) -> None:
    """Print key statistics for paper text."""
    dr = np.array(data["delta_ratios"])
    cr = np.array(data["cost_ratios"])

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (for paper §2 text)")
    print("=" * 60)
    print(f"Total pressure events analyzed: {data['n_events']}")

    print(f"\nPanel A — Heterogeneity:")
    print(f"  Median δ ratio (max/min per event): {np.median(dr):.1f}×")
    print(f"  Events with ratio ≥ 3×: {np.mean(dr >= 3.0) * 100:.0f}%")
    print(f"  Events with ratio ≥ 5×: {np.mean(dr >= 5.0) * 100:.0f}%")
    print(f"  Events with ratio ≥ 10×: {np.mean(dr >= 10.0) * 100:.0f}%")

    print(f"\nPanel B — Policy divergence:")
    print(f"  Events where LIFO picks costlier victim: {np.mean(cr > 1.0) * 100:.0f}%")
    print(f"  Median cost ratio (LIFO/BidKV): {np.median(cr):.1f}×")
    print(f"  P90 cost ratio: {np.percentile(cr, 90):.1f}×")

    cum_lifo = data["cum_lifo"][-1] if data["cum_lifo"] else 0
    cum_bidkv = data["cum_bidkv"][-1] if data["cum_bidkv"] else 0
    cum_largest = data["cum_largest"][-1] if data["cum_largest"] else 0

    print(f"\nPanel C — Cumulative waste:")
    print(f"  LIFO total recompute waste: {cum_lifo:,.0f} tokens")
    print(f"  Largest-First total: {cum_largest:,.0f} tokens")
    print(f"  BidKV total: {cum_bidkv:,.0f} tokens")
    if cum_bidkv > 0:
        print(f"  LIFO / BidKV ratio: {cum_lifo / cum_bidkv:.1f}×")
        print(f"  Largest-First / BidKV ratio: {cum_largest / cum_bidkv:.1f}×")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3-panel motivation figure for BidKV §2"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to victim_heterogeneity.jsonl"
    )
    parser.add_argument(
        "--output", type=str,
        default="paper/figures/fig_preliminary_motivation.pdf",
        help="Output PDF path"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (for layout preview without real data)"
    )
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic motivation data (layout preview)...")
        data = generate_synthetic_data()
        is_synthetic = True
    elif args.log_file:
        log_path = Path(args.log_file)
        if not log_path.exists():
            print(f"ERROR: log file not found: {log_path}")
            sys.exit(1)
        events = load_events(log_path)
        if len(events) < 10:
            print(f"ERROR: too few events ({len(events)}). Need ≥10 for meaningful figure.")
            sys.exit(1)
        data = analyze_events(events)
        is_synthetic = False
    else:
        # Try default path
        default = Path("results/preliminary_motivation/victim_heterogeneity.jsonl")
        if default.exists():
            events = load_events(default)
            data = analyze_events(events)
            is_synthetic = False
        else:
            print("No log file found; using synthetic data for preview.")
            print(f"  (Expected: {default})")
            print("  Run data capture first, or use --synthetic flag.")
            data = generate_synthetic_data()
            is_synthetic = True

    print_summary(data)
    plot_3panel(data, Path(args.output), is_synthetic=is_synthetic)


if __name__ == "__main__":
    main()
