#!/usr/bin/env python3
"""Generate 3-panel motivation figure with per-strategy separate simulation.

Each strategy runs a SEPARATE simulation path: victim selection affects
downstream state (which requests survive, KV occupancy, future pressure),
producing realistic divergence across strategies.

Panels:
  (a) CDF of δ heterogeneity across KV pressure events (workload-intrinsic)
  (b) Cumulative disruption cost (Σδ of evicted victims) per strategy over time
  (c) Cumulative recompute waste (Σ output tokens lost) per strategy over time

Data source: frozen ShareGPT trace (mixed, rate=5.7, 1000 requests)
Parameters: match frozen env (600 KV blocks, max-num-seqs=32, etc.)

Usage:
    python scripts/generate_real_motivation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
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
# Simulation parameters (matching frozen experiment env)
# ---------------------------------------------------------------------------
KV_CAPACITY = 9600      # tokens (600 blocks × 16)
MAX_CONCURRENT = 32     # max-num-seqs
PREFILL_TPS = 3000.0    # tokens/s prefill throughput (A6000, 8B, bf16)
DECODE_TPS = 40.0       # tokens/s per-request decode throughput
STEP_MS = 25.0          # simulation timestep
EPSILON = 1e-3


@dataclass
class SimRequest:
    """A simulated request."""
    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_ms: float
    # Runtime state
    state: str = "waiting"  # waiting | prefilling | decoding | done
    prefill_start_ms: float = 0.0
    decode_start_ms: float = 0.0
    output_tokens: int = 0
    kv_tokens: int = 0
    num_preemptions: int = 0
    prefill_progress: int = 0


def load_trace(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    requests = []
    for r in data["requests"]:
        meta = r.get("metadata", {})
        pt = int(meta.get("actual_prompt_tokens", 64))
        requests.append({
            "request_id": r["request_id"],
            "prompt_tokens": max(pt, 4),
            "max_tokens": max(r["max_tokens"], 1),
            "arrival_ms": r["arrival_time_ms"],
        })
    return requests


# ---------------------------------------------------------------------------
# Disruption cost δ (exact match: bidkv_strategy.py)
# ---------------------------------------------------------------------------

def compute_delta(r: SimRequest) -> float:
    ot = r.output_tokens
    if ot > 2:
        recompute_norm = max(0.5, r.prompt_tokens / 256.0)
        completion = min(1.0, ot / r.max_output_tokens) if r.max_output_tokens > 0 else 0.0
        late_penalty = completion * completion * 2.0
        starvation = r.num_preemptions * 0.5
        return max(0.1, recompute_norm + late_penalty + starvation)
    else:
        return max(0.1, 1.0 + r.num_preemptions * 0.5)


def compute_utility(r: SimRequest) -> float:
    delta = compute_delta(r)
    freed = r.output_tokens if r.output_tokens > 2 else r.kv_tokens
    return freed / (delta + EPSILON)


# ---------------------------------------------------------------------------
# Victim selection strategies
# ---------------------------------------------------------------------------

def select_victims_lifo(running: list[SimRequest], needed: int) -> list[SimRequest]:
    """vLLM default: evict most recently admitted."""
    candidates = sorted(running, key=lambda r: r.prefill_start_ms, reverse=True)
    victims, freed = [], 0
    for r in candidates:
        if r.kv_tokens <= 0:
            continue
        victims.append(r)
        freed += r.kv_tokens
        if freed >= needed:
            break
    return victims


def select_victims_largest(running: list[SimRequest], needed: int) -> list[SimRequest]:
    """Largest-First: evict by descending KV occupancy."""
    candidates = sorted(running, key=lambda r: r.kv_tokens, reverse=True)
    victims, freed = [], 0
    for r in candidates:
        if r.kv_tokens <= 0:
            continue
        victims.append(r)
        freed += r.kv_tokens
        if freed >= needed:
            break
    return victims


def select_victims_bidkv(running: list[SimRequest], needed: int) -> list[SimRequest]:
    """BidKV: evict by descending utility U = freed / (δ + ε)."""
    candidates = sorted(running, key=compute_utility, reverse=True)
    victims, freed = [], 0
    for r in candidates:
        if r.kv_tokens <= 0:
            continue
        victims.append(r)
        freed += r.kv_tokens
        if freed >= needed:
            break
    return victims


STRATEGY_FNS = {
    "LIFO (vLLM default)": select_victims_lifo,
    "Largest-First": select_victims_largest,
    "BidKV (ours)": select_victims_bidkv,
}


# ---------------------------------------------------------------------------
# Simulation statistics
# ---------------------------------------------------------------------------

@dataclass
class SimStats:
    preemption_count: int = 0
    total_delta: float = 0.0
    total_output_lost: int = 0
    total_prefill_recompute: int = 0  # Σ prompt_tokens of evicted requests
    total_kv_freed: int = 0
    completed: int = 0
    # Timeseries
    ts_seconds: list[float] = field(default_factory=list)
    ts_cum_delta: list[float] = field(default_factory=list)
    ts_cum_prefill_recompute: list[float] = field(default_factory=list)
    ts_preempt_count: list[int] = field(default_factory=list)
    # Panel A data (workload-intrinsic, recorded once)
    delta_ratios: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def make_requests(trace: list[dict]) -> list[SimRequest]:
    return [
        SimRequest(
            request_id=r["request_id"],
            prompt_tokens=r["prompt_tokens"],
            max_output_tokens=r["max_tokens"],
            arrival_ms=r["arrival_ms"],
        )
        for r in trace
    ]


def simulate(
    trace: list[dict],
    victim_fn,
    record_heterogeneity: bool = False,
) -> SimStats:
    """Run full simulation for one strategy."""
    all_reqs = make_requests(trace)
    waiting: list[SimRequest] = []
    running: list[SimRequest] = []
    stats = SimStats()

    next_idx = 0
    t_ms = 0.0
    max_t_ms = max(r.arrival_ms for r in all_reqs) + 600_000
    sample_interval = 500.0
    last_sample = -sample_interval

    while t_ms < max_t_ms:
        # 1. Admit arrivals
        while next_idx < len(all_reqs):
            r = all_reqs[next_idx]
            if r.arrival_ms <= t_ms:
                r.state = "waiting"
                waiting.append(r)
                next_idx += 1
            else:
                break

        # 2. Advance running requests
        for r in running:
            if r.state == "prefilling":
                step_tok = int(PREFILL_TPS * STEP_MS / 1000)
                r.prefill_progress += step_tok
                r.kv_tokens = min(r.prompt_tokens, r.prefill_progress)
                if r.prefill_progress >= r.prompt_tokens:
                    r.state = "decoding"
                    r.decode_start_ms = t_ms
                    r.kv_tokens = r.prompt_tokens
            elif r.state == "decoding":
                step_tok = max(1, int(DECODE_TPS * STEP_MS / 1000))
                r.output_tokens += step_tok
                r.kv_tokens = r.prompt_tokens + r.output_tokens
                if r.output_tokens >= r.max_output_tokens:
                    r.state = "done"

        # 3. Complete finished
        still_running = []
        for r in running:
            if r.state == "done":
                stats.completed += 1
            else:
                still_running.append(r)
        running = still_running

        # 4. Admit from waiting queue
        total_kv = sum(r.kv_tokens for r in running)
        while waiting and len(running) < MAX_CONCURRENT:
            cand = waiting[0]
            if total_kv + cand.prompt_tokens <= KV_CAPACITY:
                req = waiting.pop(0)
                req.state = "prefilling"
                req.prefill_start_ms = t_ms
                req.prefill_progress = 0
                req.kv_tokens = 0
                running.append(req)
                total_kv = sum(r.kv_tokens for r in running)
            else:
                break

        # 5. Preemption if over capacity
        total_kv = sum(r.kv_tokens for r in running)
        rounds = 0
        while total_kv > KV_CAPACITY and running and rounds < 10:
            rounds += 1
            overflow = total_kv - KV_CAPACITY
            eligible = [r for r in running if r.kv_tokens > 0]

            # Record heterogeneity (Panel A)
            if record_heterogeneity and len(eligible) >= 2:
                deltas = [compute_delta(r) for r in eligible]
                d_min, d_max = min(deltas), max(deltas)
                if d_min > 0:
                    stats.delta_ratios.append(d_max / d_min)

            victims = victim_fn(eligible, overflow)
            if not victims:
                break

            for v in victims:
                stats.preemption_count += 1
                stats.total_delta += compute_delta(v)
                stats.total_output_lost += v.output_tokens
                stats.total_prefill_recompute += v.prompt_tokens
                stats.total_kv_freed += v.kv_tokens
                running.remove(v)
                v.state = "waiting"
                v.kv_tokens = 0
                v.output_tokens = 0
                v.prefill_progress = 0
                v.num_preemptions += 1
                waiting.insert(0, v)

            total_kv = sum(r.kv_tokens for r in running)

        # 6. Timeseries sample
        if t_ms - last_sample >= sample_interval:
            last_sample = t_ms
            stats.ts_seconds.append(t_ms / 1000)
            stats.ts_cum_delta.append(stats.total_delta)
            stats.ts_cum_prefill_recompute.append(stats.total_prefill_recompute)
            stats.ts_preempt_count.append(stats.preemption_count)

        # 7. Termination
        if next_idx >= len(all_reqs) and not waiting and not running:
            break

        t_ms += STEP_MS

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_3panel(all_stats: dict[str, SimStats], output_path: Path, n_reqs: int) -> None:
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

    FIG_W, FIG_H = 3.33, 5.8
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H),
        gridspec_kw={"hspace": 0.65},
    )

    COL = {
        "LIFO (vLLM default)": "#D63027",
        "Largest-First": "#f39c12",
        "BidKV (ours)": "#2980b9",
    }
    LS = {
        "LIFO (vLLM default)": "-",
        "Largest-First": "--",
        "BidKV (ours)": "-",
    }

    # ── Panel A: CDF of δ heterogeneity ──────────────────────────
    bidkv_stats = all_stats["BidKV (ours)"]
    dr = np.array(bidkv_stats.delta_ratios)
    n_events = len(dr)

    if n_events > 0:
        sorted_r = np.sort(dr)
        cdf_y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax_a.plot(sorted_r, cdf_y, color=COL["BidKV (ours)"], linewidth=1.5)
        ax_a.fill_between(sorted_r, 0, cdf_y, alpha=0.15, color=COL["BidKV (ours)"])

        # Only annotate 10× threshold to avoid clutter
        th10 = 10.0
        if th10 <= sorted_r[-1]:
            pct10 = np.mean(dr >= th10) * 100
            ax_a.axvline(th10, color="#999", linestyle="-.", linewidth=0.8)
            yp10 = np.searchsorted(sorted_r, th10) / len(sorted_r)
            ax_a.annotate(
                f"{pct10:.0f}% of events have $\\geq$10$\\times$ cost spread",
                xy=(th10, yp10),
                xytext=(th10 + 2.0, 0.45),
                fontsize=6, color="#333", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"),
            )

    ax_a.set_xlabel("Disruption cost ratio (max $\\delta$ / min $\\delta$)")
    ax_a.set_ylabel("CDF")
    ax_a.set_title(
        f"(a) Victim heterogeneity under KV pressure\n"
        f"({n_events} events, {n_reqs} ShareGPT reqs, rate=5.7 req/s)",
        pad=3,
    )
    ax_a.set_xlim(left=1.0)
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, linestyle=":", alpha=0.4)

    # ── Panel B: Cumulative disruption Σδ ─────────────────────────
    for name in ["LIFO (vLLM default)", "Largest-First", "BidKV (ours)"]:
        s = all_stats[name]
        if s.ts_seconds:
            ts_min = np.array(s.ts_seconds) / 60
            ax_b.plot(ts_min, s.ts_cum_delta, color=COL[name],
                      linestyle=LS[name], linewidth=1.3, label=name)

    # Shading + ratio annotation
    s_l, s_b = all_stats["LIFO (vLLM default)"], all_stats["BidKV (ours)"]
    if s_l.ts_seconds and s_b.ts_seconds:
        t_max = min(s_l.ts_seconds[-1], s_b.ts_seconds[-1])
        tg = np.linspace(0, t_max, 300) / 60
        li = np.interp(tg * 60, s_l.ts_seconds, s_l.ts_cum_delta)
        bi = np.interp(tg * 60, s_b.ts_seconds, s_b.ts_cum_delta)
        ax_b.fill_between(tg, bi, li, alpha=0.08, color=COL["LIFO (vLLM default)"])
        if bi[-1] > 0:
            ratio = li[-1] / bi[-1]
            ax_b.annotate(
                f"{ratio:.1f}$\\times$",
                xy=(tg[-1], li[-1]),
                xytext=(tg[-1] * 0.6, li[-1] * 0.9),
                fontsize=6, color=COL["LIFO (vLLM default)"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL["LIFO (vLLM default)"], lw=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"),
            )

    ax_b.set_xlabel("Time (minutes)")
    ax_b.set_ylabel("Cumulative disruption ($\\Sigma\\delta$)")
    ax_b.set_title("(b) Accumulated eviction disruption cost", pad=3)
    ax_b.legend(loc="upper left", framealpha=0.9, handletextpad=0.3, borderpad=0.4)
    ax_b.grid(True, linestyle=":", alpha=0.4)

    # ── Panel C: Cumulative preemption count (scheduling thrash) ──
    for name in ["LIFO (vLLM default)", "Largest-First", "BidKV (ours)"]:
        s = all_stats[name]
        if s.ts_seconds:
            ts_min = np.array(s.ts_seconds) / 60
            ax_c.plot(ts_min, np.array(s.ts_preempt_count),
                      color=COL[name], linestyle=LS[name], linewidth=1.3, label=name)

    if s_l.ts_seconds and s_b.ts_seconds:
        lo_i = np.interp(tg * 60, s_l.ts_seconds, s_l.ts_preempt_count)
        bo_i = np.interp(tg * 60, s_b.ts_seconds, s_b.ts_preempt_count)
        ax_c.fill_between(tg, bo_i, lo_i,
                          alpha=0.08, color=COL["LIFO (vLLM default)"])
        if bo_i[-1] > 0:
            ro = lo_i[-1] / bo_i[-1]
            # Position annotation at right side, below the line
            ax_c.annotate(
                f"{ro:.1f}$\\times$ more preemptions",
                xy=(tg[int(len(tg)*0.8)], lo_i[int(len(lo_i)*0.8)]),
                xytext=(tg[int(len(tg)*0.35)], lo_i[-1] * 0.55),
                fontsize=5.5, color=COL["LIFO (vLLM default)"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL["LIFO (vLLM default)"], lw=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="none"),
            )

    ax_c.set_xlabel("Time (minutes)")
    ax_c.set_ylabel("Cumulative preemption count")
    ax_c.set_title("(c) Scheduling disruption from preemption events", pad=3)
    ax_c.legend(loc="lower right", framealpha=0.9, handletextpad=0.3, borderpad=0.4)
    ax_c.grid(True, linestyle=":", alpha=0.4)

    fig.align_ylabels([ax_a, ax_b, ax_c])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"\n  PDF → {output_path}")
    png = output_path.with_suffix(".png")
    plt.savefig(png, bbox_inches="tight", format="png", dpi=300)
    print(f"  PNG → {png}")


def print_summary(all_stats: dict[str, SimStats]) -> None:
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS (for paper §2)")
    print("=" * 65)

    bs = all_stats["BidKV (ours)"]
    dr = np.array(bs.delta_ratios)
    print(f"\nPanel A — Heterogeneity ({len(dr)} events):")
    if len(dr) > 0:
        print(f"  Median δ ratio:  {np.median(dr):.1f}×")
        for th in [3, 5, 10]:
            print(f"  Events ≥ {th}×:    {np.mean(dr >= th) * 100:.0f}%")

    print(f"\n{'Strategy':<22} {'#Preempt':>8} {'Σδ':>10} {'RePrefill':>10} {'#Done':>6}")
    print("-" * 60)
    for name in ["LIFO (vLLM default)", "Largest-First", "BidKV (ours)"]:
        s = all_stats[name]
        print(f"{name:<22} {s.preemption_count:>8} {s.total_delta:>10.1f} "
              f"{s.total_prefill_recompute:>10,} {s.completed:>6}")

    ls, bs = all_stats["LIFO (vLLM default)"], all_stats["BidKV (ours)"]
    if bs.total_delta > 0:
        print(f"\nLIFO/BidKV Σδ:         {ls.total_delta / bs.total_delta:.2f}×")
    if bs.total_prefill_recompute > 0:
        print(f"LIFO/BidKV re-prefill: {ls.total_prefill_recompute / bs.total_prefill_recompute:.2f}×")
    if bs.preemption_count > 0:
        print(f"LIFO/BidKV #preempt:   {ls.preemption_count / bs.preemption_count:.2f}×")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="experiments/vllm/traces/mixed_rate5.7.json")
    parser.add_argument("--output", default="paper/figures/fig_preliminary_motivation.pdf")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"ERROR: trace not found: {trace_path}")
        sys.exit(1)

    print(f"Loading trace: {trace_path}")
    trace = load_trace(trace_path)
    print(f"  {len(trace)} requests, KV={KV_CAPACITY} tokens, max_seq={MAX_CONCURRENT}")

    all_stats: dict[str, SimStats] = {}
    for name, fn in STRATEGY_FNS.items():
        print(f"\nSimulating [{name}] ...")
        stats = simulate(trace, fn, record_heterogeneity=(name == "BidKV (ours)"))
        all_stats[name] = stats
        print(f"  → Preemptions={stats.preemption_count}, Σδ={stats.total_delta:.1f}, "
              f"OutLost={stats.total_output_lost:,}, Done={stats.completed}")

    # Save summary
    out_dir = Path("results/preliminary_motivation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {n: {"preemptions": s.preemption_count, "sigma_delta": round(s.total_delta, 2),
                    "prefill_recompute": s.total_prefill_recompute,
                    "output_lost": s.total_output_lost, "completed": s.completed}
               for n, s in all_stats.items()}
    with open(out_dir / "simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_stats)
    plot_3panel(all_stats, Path(args.output), len(trace))


if __name__ == "__main__":
    main()
