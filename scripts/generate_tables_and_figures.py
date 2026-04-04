#!/usr/bin/env python3
"""Generate experiment tables and figures from vllm_24run_full results.

Produces:
  - paper/tables/table1_main.tex            (Table 1: main comparison)
  - paper/tables/table2_rate_full.tex       (Table 2: full rate sweep)
  - paper/figures/fig1_main_comparison.pdf   (Figure 1: visual main comparison)
  - paper/figures/fig3_rate_sensitivity.pdf  (Figure 3: rate sweep)
  - paper/figures/fig3b_slo_attainment.pdf   (Figure 3b: SLO attainment)
  - paper/figures/fig5_compress_coverage.pdf  (Figure 5: compression coverage)
  - paper/figures/fig6_ttft_grouped_bar.pdf  (Figure 6: TTFT P95 grouped bar)

Metrics:
  Throughput (req/s), TTFT P50/P95 (ms), TPOT P50/P95 (ms),
  SLO Attainment (TTFT < 2s %), Eviction Count, Completion Rate (%)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "vllm_24run_full"
PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"
TABLE_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"

STRATEGIES_ORDER = [
    "preempt-evict",
    "static-random",
    "h2o-style",  # legacy name in result files; displays as Largest-First
    "uniform",
    "global-nobid",
    "slack-aware",
    "bidkv",
]

STRATEGY_DISPLAY = {
    "preempt-evict": "Preempt-Evict",
    "static-random": "Static-Random",
    "h2o-style": "Largest-First",
    "uniform": "Uniform",
    "global-nobid": "Global-NoBid",
    "slack-aware": "Slack-Aware",
    "bidkv": r"\textbf{BidKV}",
}

RATES = [0.35, 0.5, 0.7]
SLO_TTFT_MS = 2000  # TTFT < 2s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) using nearest-rank."""
    if not values:
        return 0.0
    s = sorted(values)
    k = int(len(s) * p / 100)
    k = min(k, len(s) - 1)
    return s[k]


def load_run(filepath: Path) -> dict:
    """Load a single run JSON and compute all metrics."""
    with open(filepath) as f:
        d = json.load(f)

    rr = d.get("request_results", [])
    summary = d.get("summary", {})
    am = d.get("adapter_metrics", {})

    # Per-request TTFT
    ttfts = [r["ttft_ms"] for r in rr if r.get("ttft_ms") is not None]

    # Per-request TPOT: (total_latency - ttft) / (completion_tokens - 1)
    tpots = []
    for r in rr:
        ct = r.get("completion_tokens", 0)
        ttft = r.get("ttft_ms", 0)
        tot = r.get("total_latency_ms", 0)
        if ct > 1 and tot > ttft:
            tpots.append((tot - ttft) / (ct - 1))

    total = summary.get("total_requests", len(rr))
    success = summary.get("successful_requests", total)
    failed = summary.get("failed_requests", 0)

    slo_ok = sum(1 for t in ttfts if t < SLO_TTFT_MS)
    slo_pct = (slo_ok / len(ttfts) * 100) if ttfts else 0.0

    return {
        "strategy": d.get("strategy", ""),
        "rate": d.get("request_rate", 0),
        "throughput": summary.get("throughput_rps", 0),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p95": percentile(tpots, 95),
        "tpot_p99": percentile(tpots, 99),
        "slo_attainment": slo_pct,
        "eviction_count": am.get("total_evictions", am.get("total_compressions", 0)),
        "tokens_freed": am.get("total_tokens_freed", 0),
        "total_requests": total,
        "successful_requests": success,
        "failed_requests": failed,
        "completion_rate": (success / total * 100) if total > 0 else 0.0,
        "duration_s": d.get("duration_s", 0),
    }


def load_all() -> list[dict]:
    """Load all experiment JSON files."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if "consistency" in f.name:
            continue
        row = load_run(f)
        if row["strategy"] in STRATEGIES_ORDER:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------
def fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def generate_table1(rows: list[dict]) -> str:
    """Table 1: Main comparison at rate=0.5 (long_context workload)."""
    rate = 0.5
    rate_rows = {r["strategy"]: r for r in rows if r["rate"] == rate}

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Main comparison on vLLM (Llama-3.1-8B-Instruct, A6000 48\,GiB,"
        r" long-context workload, rate=0.5\,req/s).  Best result in"
        r" \textbf{bold}.}"
    )
    lines.append(r"\label{tab:main}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l c c c c c c c c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Strategy} & \textbf{Thpt} & \textbf{TTFT P50}"
        r" & \textbf{TTFT P95} & \textbf{TPOT P50} & \textbf{TPOT P95}"
        r" & \textbf{SLO Att.} & \textbf{Evict} & \textbf{Compl.} \\"
    )
    lines.append(r" & (req/s) & (ms) & (ms) & (ms) & (ms) & (\%) & (cnt) & (\%) \\")
    lines.append(r"\midrule")

    # Find best for bolding
    best_thpt = max(rate_rows[s]["throughput"] for s in STRATEGIES_ORDER if s in rate_rows)
    best_slo = max(rate_rows[s]["slo_attainment"] for s in STRATEGIES_ORDER if s in rate_rows)
    best_ttft95 = min(rate_rows[s]["ttft_p95"] for s in STRATEGIES_ORDER if s in rate_rows)
    best_tpot95 = min(rate_rows[s]["tpot_p95"] for s in STRATEGIES_ORDER if s in rate_rows)

    for strat in STRATEGIES_ORDER:
        if strat not in rate_rows:
            continue
        r = rate_rows[strat]
        name = STRATEGY_DISPLAY[strat]

        def bold_if_best(val, best, fmt_fn, lower_better=False):
            s = fmt_fn(val)
            if lower_better:
                return rf"\textbf{{{s}}}" if val <= best else s
            else:
                return rf"\textbf{{{s}}}" if val >= best else s

        thpt = bold_if_best(r["throughput"], best_thpt, lambda v: f"{v:.3f}")
        ttft50 = fmt_ms(r["ttft_p50"])
        ttft95 = bold_if_best(r["ttft_p95"], best_ttft95, fmt_ms, lower_better=True)
        tpot50 = f"{r['tpot_p50']:.1f}"
        tpot95 = bold_if_best(r["tpot_p95"], best_tpot95, lambda v: f"{v:.1f}", lower_better=True)
        slo = bold_if_best(r["slo_attainment"], best_slo, lambda v: f"{v:.1f}")
        evict = str(r["eviction_count"])
        compl = f"{r['completion_rate']:.0f}"

        lines.append(
            f"{name:<20} & {thpt} & {ttft50} & {ttft95}"
            f" & {tpot50} & {tpot95} & {slo} & {evict} & {compl} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_rate_table(rows: list[dict]) -> str:
    """Supplementary: full rate × strategy table."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full results across request rates"
        r" (long-context workload, vLLM).}"
    )
    lines.append(r"\label{tab:rate_full}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{@{}l c c c c c c c c c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Strategy} & \textbf{Rate} & \textbf{Thpt}"
        r" & \textbf{TTFT P50} & \textbf{TTFT P95}"
        r" & \textbf{TPOT P50} & \textbf{TPOT P95}"
        r" & \textbf{SLO\%} & \textbf{Evict} & \textbf{Compl\%} \\"
    )
    lines.append(r"\midrule")

    for strat in STRATEGIES_ORDER:
        first = True
        for rate in RATES:
            r = next(
                (x for x in rows if x["strategy"] == strat and x["rate"] == rate),
                None,
            )
            if r is None:
                continue
            name = STRATEGY_DISPLAY[strat] if first else ""
            first = False
            lines.append(
                f"{name:<20} & {rate} & {r['throughput']:.3f}"
                f" & {fmt_ms(r['ttft_p50'])} & {fmt_ms(r['ttft_p95'])}"
                f" & {r['tpot_p50']:.1f} & {r['tpot_p95']:.1f}"
                f" & {r['slo_attainment']:.1f} & {r['eviction_count']}"
                f" & {r['completion_rate']:.0f} \\\\"
            )
        if strat != STRATEGIES_ORDER[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def _bidkv_advantage_text(
    rows: list[dict], rate: float, metric: str, lower_better: bool = True
) -> str:
    """Compute BidKV advantage vs worst baseline for annotation."""
    rate_rows = {r["strategy"]: r for r in rows if r["rate"] == rate}
    if "bidkv" not in rate_rows:
        return ""
    bv = rate_rows["bidkv"][metric]
    others = {s: rate_rows[s][metric] for s in STRATEGIES_ORDER if s in rate_rows and s != "bidkv"}
    if not others:
        return ""
    if lower_better:
        worst_name = max(others, key=lambda s: others[s])
        worst_val = others[worst_name]
        if worst_val == 0:
            return ""
        pct = (worst_val - bv) / worst_val * 100
        return f"BidKV vs {STRATEGY_DISPLAY.get(worst_name, worst_name)}: -{pct:.0f}%"
    else:
        worst_name = min(others, key=lambda s: others[s])
        worst_val = others[worst_name]
        if worst_val == 0:
            return ""
        pct = (bv - worst_val) / worst_val * 100
        return f"BidKV vs {STRATEGY_DISPLAY.get(worst_name, worst_name)}: +{pct:.0f}%"


def generate_fig1(rows: list[dict]) -> None:
    """Figure 1: Visual main comparison at rate=0.5 (multi-panel bar chart)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available. Skipping fig1.", file=sys.stderr)
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rate = 0.5
    rate_rows = {r["strategy"]: r for r in rows if r["rate"] == rate}
    strats = [s for s in STRATEGIES_ORDER if s in rate_rows]
    labels = [STRATEGY_DISPLAY.get(s, s).replace(r"\textbf{", "").replace("}", "") for s in strats]
    n = len(strats)

    bar_colors = ["#d62728" if s == "bidkv" else "#4c72b0" for s in strats]
    edge_colors = ["#a01c1c" if s == "bidkv" else "#2f4f6f" for s in strats]

    metrics = [
        ("SLO Attainment (%)", "slo_attainment", False),
        ("TTFT P95 (s)", "ttft_p95", True),
        ("TPOT P95 (ms)", "tpot_p95", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (ylabel, key, lower_better) in zip(axes, metrics, strict=False):
        vals = []
        for s in strats:
            v = rate_rows[s][key]
            if key == "ttft_p95":
                v = v / 1000  # convert ms → s
            vals.append(v)

        ax.bar(range(n), vals, color=bar_colors, edgecolor=edge_colors, linewidth=0.8, alpha=0.9)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        # Highlight BidKV bar value
        bidkv_idx = strats.index("bidkv") if "bidkv" in strats else None
        if bidkv_idx is not None:
            bv = vals[bidkv_idx]
            ax.annotate(
                f"{bv:.1f}",
                xy=(bidkv_idx, bv),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#d62728",
            )

        # Compute BidKV advantage annotation
        if bidkv_idx is not None:
            bv = vals[bidkv_idx]
            other_vals = {strats[i]: vals[i] for i in range(n) if i != bidkv_idx}
            if lower_better:
                worst_s = max(other_vals, key=lambda s: other_vals[s])
                wv = other_vals[worst_s]
                if wv > 0:
                    pct = (wv - bv) / wv * 100
                    wname = (
                        STRATEGY_DISPLAY.get(worst_s, worst_s)
                        .replace(r"\textbf{", "")
                        .replace("}", "")
                    )
                    ax.set_title(f"{ylabel}\nBidKV vs {wname}: -{pct:.0f}%", fontsize=9)
                else:
                    ax.set_title(ylabel, fontsize=9)
            else:
                worst_s = min(other_vals, key=lambda s: other_vals[s])
                wv = other_vals[worst_s]
                if wv > 0:
                    pct = (bv - wv) / wv * 100
                    wname = (
                        STRATEGY_DISPLAY.get(worst_s, worst_s)
                        .replace(r"\textbf{", "")
                        .replace("}", "")
                    )
                    ax.set_title(f"{ylabel}\nBidKV vs {wname}: +{pct:.0f}%", fontsize=9)
                else:
                    ax.set_title(ylabel, fontsize=9)

    # Legend
    bidkv_patch = mpatches.Patch(color="#d62728", label="BidKV")
    other_patch = mpatches.Patch(color="#4c72b0", label="Baselines")
    fig.legend(
        handles=[bidkv_patch, other_patch], loc="upper center", ncol=2, fontsize=9, frameon=False
    )

    fig.suptitle("Main Comparison (rate=0.5, long-context, vLLM)", fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig1_main_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig1_main_comparison.pdf'}")


def generate_figures(rows: list[dict]) -> None:
    """Generate all PDF figures using matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available. Skipping figures.", file=sys.stderr)
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Color scheme
    colors = {
        "preempt-evict": "#1f77b4",
        "static-random": "#aec7e8",
        "h2o-style": "#ff7f0e",
        "uniform": "#c7c7c7",
        "global-nobid": "#98df8a",
        "slack-aware": "#9467bd",
        "bidkv": "#d62728",
    }
    markers = {
        "preempt-evict": "s",
        "static-random": "v",
        "h2o-style": "D",
        "uniform": "^",
        "global-nobid": "<",
        "slack-aware": ">",
        "bidkv": "o",
    }
    display = {
        "preempt-evict": "Preempt-Evict",
        "static-random": "Static-Random",
        "h2o-style": "Largest-First",
        "uniform": "Uniform",
        "global-nobid": "Global-NoBid",
        "slack-aware": "Slack-Aware",
        "bidkv": "BidKV",
    }

    # ---- Figure 3: Rate Sensitivity (TTFT P95 + Throughput) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for strat in STRATEGIES_ORDER:
        strat_rows = sorted([r for r in rows if r["strategy"] == strat], key=lambda x: x["rate"])
        if not strat_rows:
            continue
        rates = [r["rate"] for r in strat_rows]
        ttft95 = [r["ttft_p95"] for r in strat_rows]
        thpt = [r["throughput"] for r in strat_rows]
        lw = 2.5 if strat in ("bidkv", "preempt-evict") else 1.2
        alpha = 1.0 if strat in ("bidkv", "preempt-evict", "h2o-style") else 0.5

        ax1.plot(
            rates,
            ttft95,
            color=colors[strat],
            marker=markers[strat],
            label=display[strat],
            linewidth=lw,
            alpha=alpha,
            markersize=7,
        )
        ax2.plot(
            rates,
            thpt,
            color=colors[strat],
            marker=markers[strat],
            label=display[strat],
            linewidth=lw,
            alpha=alpha,
            markersize=7,
        )

    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("TTFT P95 (ms)")
    ax1.set_yscale("log")
    ax1.set_xticks(RATES)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc="upper left")
    # BidKV advantage annotation at highest rate
    adv = _bidkv_advantage_text(rows, 0.7, "ttft_p95", lower_better=True)
    ax1.set_title(f"(a) TTFT P95 vs Rate\n{adv}" if adv else "(a) TTFT P95 vs Rate", fontsize=9)

    ax2.set_xlabel("Request Rate (req/s)")
    ax2.set_ylabel("Throughput (req/s)")
    ax2.set_xticks(RATES)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("(b) Throughput vs Rate")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rate_sensitivity.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig3_rate_sensitivity.pdf'}")

    # ---- Figure 3b: SLO Attainment vs Rate ----
    fig, ax = plt.subplots(figsize=(5, 4))
    for strat in STRATEGIES_ORDER:
        strat_rows = sorted([r for r in rows if r["strategy"] == strat], key=lambda x: x["rate"])
        if not strat_rows:
            continue
        rates = [r["rate"] for r in strat_rows]
        slo = [r["slo_attainment"] for r in strat_rows]
        lw = 2.5 if strat in ("bidkv", "preempt-evict") else 1.2
        alpha = 1.0 if strat in ("bidkv", "preempt-evict", "h2o-style") else 0.5
        ax.plot(
            rates,
            slo,
            color=colors[strat],
            marker=markers[strat],
            label=display[strat],
            linewidth=lw,
            alpha=alpha,
            markersize=7,
        )
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("SLO Attainment (TTFT < 2s) %")
    ax.set_xticks(RATES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    adv = _bidkv_advantage_text(rows, 0.5, "slo_attainment", lower_better=False)
    ax.set_title(f"SLO Attainment vs Rate\n{adv}" if adv else "SLO Attainment vs Rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3b_slo_attainment.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig3b_slo_attainment.pdf'}")

    # ---- Figure 5: Compression / Eviction Coverage ----
    fig, ax = plt.subplots(figsize=(8, 4))
    rate = 0.7  # High pressure shows most differentiation
    rate_rows = sorted(
        [r for r in rows if r["rate"] == rate],
        key=lambda x: STRATEGIES_ORDER.index(x["strategy"])
        if x["strategy"] in STRATEGIES_ORDER
        else 99,
    )
    strats = [display[r["strategy"]] for r in rate_rows]
    evicts = [r["eviction_count"] for r in rate_rows]
    freed = [r["tokens_freed"] / 1000 for r in rate_rows]  # in thousands

    x = range(len(strats))
    bar_w = 0.35
    ax.bar(
        [i - bar_w / 2 for i in x],
        evicts,
        bar_w,
        label="Eviction Count",
        color="#d62728",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2 = ax.twinx()
    ax2.bar(
        [i + bar_w / 2 for i in x],
        freed,
        bar_w,
        label="Tokens Freed (K)",
        color="#2ca02c",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Eviction Count")
    ax2.set_ylabel("Tokens Freed (×1000)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(strats, rotation=30, ha="right", fontsize=8)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    # Annotation: BidKV eviction count vs Preempt-Evict
    bidkv_row = next((r for r in rate_rows if r["strategy"] == "bidkv"), None)
    pe_row = next((r for r in rate_rows if r["strategy"] == "preempt-evict"), None)
    subtitle = f"Compression Coverage (rate={rate})"
    if bidkv_row and pe_row:
        subtitle += f"\nBidKV: {bidkv_row['eviction_count']} evictions"
        subtitle += f", {bidkv_row['tokens_freed']} tokens freed"
    ax.set_title(subtitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_compress_coverage.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig5_compress_coverage.pdf'}")

    # ---- Figure 6: Normalized Latency comparison (grouped bar) ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n_strats = len(STRATEGIES_ORDER)
    n_rates = len(RATES)
    bar_w = 0.8 / n_rates
    for i, rate in enumerate(RATES):
        vals = []
        for strat in STRATEGIES_ORDER:
            r = next(
                (x for x in rows if x["strategy"] == strat and x["rate"] == rate),
                None,
            )
            vals.append(r["ttft_p95"] / 1000 if r else 0)  # seconds
        positions = [j + (i - n_rates / 2 + 0.5) * bar_w for j in range(n_strats)]
        ax.bar(
            positions,
            vals,
            bar_w,
            label=f"rate={rate}",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(range(n_strats))
    ax.set_xticklabels(
        [display[s] for s in STRATEGIES_ORDER],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("TTFT P95 (seconds)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("TTFT P95 across Strategies and Rates")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_ttft_grouped_bar.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig6_ttft_grouped_bar.pdf'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = load_all()
    print(f"Loaded {len(rows)} experiment runs.\n")

    # Print summary to console
    for rate in RATES:
        print(f"=== Rate {rate} ===")
        print(
            f"{'Strategy':<16} {'Thpt':>6} {'TTFT50':>7} {'TTFT95':>7}"
            f" {'TPOT50':>7} {'TPOT95':>7} {'SLO%':>6} {'Evict':>5}"
            f" {'Freed':>8} {'Compl%':>6}"
        )
        rate_rows = sorted(
            [r for r in rows if r["rate"] == rate],
            key=lambda x: STRATEGIES_ORDER.index(x["strategy"])
            if x["strategy"] in STRATEGIES_ORDER
            else 99,
        )
        for r in rate_rows:
            print(
                f"{r['strategy']:<16} {r['throughput']:>6.3f}"
                f" {r['ttft_p50']:>7.0f} {r['ttft_p95']:>7.0f}"
                f" {r['tpot_p50']:>7.1f} {r['tpot_p95']:>7.1f}"
                f" {r['slo_attainment']:>6.1f} {r['eviction_count']:>5}"
                f" {r['tokens_freed']:>8} {r['completion_rate']:>6.0f}"
            )
        print()

    # Generate tables
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    t1 = generate_table1(rows)
    (TABLE_DIR / "table1_main.tex").write_text(t1)
    print(f"Saved {TABLE_DIR / 'table1_main.tex'}")

    t2 = generate_rate_table(rows)
    (TABLE_DIR / "table2_rate_full.tex").write_text(t2)
    print(f"Saved {TABLE_DIR / 'table2_rate_full.tex'}")

    # Generate figures
    generate_fig1(rows)
    generate_figures(rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
