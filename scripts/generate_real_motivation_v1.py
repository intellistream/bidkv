#!/usr/bin/env python3
"""Generate 3-panel motivation figure from frozen trace simulation.

Uses the frozen ShareGPT trace (mixed, rate=5.7) + realistic vLLM serving
parameters to simulate KV pressure events across the ENTIRE workload, then
runs all 4 strategies' victim selection on each simulated pressure snapshot.

This produces the same 3-panel figure as plot_motivation_3panel.py but does NOT
require GPU access — it uses trace-derived simulation that accurately reflects
heterogeneous request lifecycle dynamics.

Simulation approach:
  1. Load frozen trace (1000 requests, Poisson arrivals, ShareGPT prompts)
  2. Simulate a vLLM-like scheduler: FCFS admission, decode at measured speeds
  3. At each step where KV usage > 80%, snapshot the running batch
  4. On each snapshot, run 4 strategies' victim selection (BidKV, LIFO, LF, SJF)
  5. Aggregate results into 3-panel figure

Key parameters match frozen env:
  - KV capacity: 9600 tokens (600 blocks × 16 tokens/block)
  - Max concurrent: 32 (max-num-seqs)
  - Prefill speed: ~3000 tokens/s, Decode speed: ~40 tokens/s (A6000 8B bf16)

Usage:
    python scripts/generate_real_motivation.py
    python scripts/generate_real_motivation.py --trace experiments/vllm/traces/mixed_rate5.7.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
# Simulation parameters (matching frozen env)
# ---------------------------------------------------------------------------
KV_CAPACITY = 9600      # tokens (600 blocks × 16)
MAX_CONCURRENT = 32     # max-num-seqs
PREFILL_TPS = 3000      # tokens/s prefill throughput (A6000, 8B, bf16, eager)
DECODE_TPS = 40         # tokens/s decode throughput
KV_PRESSURE_THRESHOLD = 0.80  # log events above this
STEP_MS = 25.0          # simulation step size in ms


@dataclass
class SimRequest:
    """A simulated request in the serving pipeline."""
    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_ms: float
    # Runtime state
    state: str = "waiting"  # waiting, prefilling, decoding, done
    prefill_start_ms: float = 0.0
    decode_start_ms: float = 0.0
    output_tokens: int = 0
    num_preemptions: int = 0
    kv_tokens: int = 0  # current KV occupancy


def load_trace(path: Path) -> list[dict]:
    """Load frozen trace and return request dicts."""
    with open(path) as f:
        data = json.load(f)
    requests = []
    for r in data["requests"]:
        meta = r.get("metadata", {})
        prompt_tokens = int(meta.get("actual_prompt_tokens", 64))
        requests.append({
            "request_id": r["request_id"],
            "prompt_tokens": prompt_tokens,
            "max_tokens": r["max_tokens"],
            "arrival_ms": r["arrival_time_ms"],
        })
    return requests


# ---------------------------------------------------------------------------
# BidKV δ and U computation (exact match with bidkv_strategy.py)
# ---------------------------------------------------------------------------

def compute_delta(prompt_tokens: int, output_tokens: int,
                  max_output_tokens: int, num_preemptions: int) -> float:
    """Compute disruption estimate δ (Eq. 4 in paper)."""
    if output_tokens > 2:
        recompute_norm = max(0.5, prompt_tokens / 256.0)
        completion = min(1.0, output_tokens / max_output_tokens) if max_output_tokens > 0 else 0.0
        late_penalty = completion * completion * 2.0
        starvation = num_preemptions * 0.5
        return max(0.1, recompute_norm + late_penalty + starvation)
    else:
        return max(0.1, 1.0 + num_preemptions * 0.5)


def compute_utility(prompt_tokens: int, output_tokens: int,
                    max_output_tokens: int, num_preemptions: int,
                    kv_tokens: int) -> float:
    """Compute utility U = tokens_freed / (δ + ε)."""
    delta = compute_delta(prompt_tokens, output_tokens, max_output_tokens, num_preemptions)
    eps = 1e-3
    if output_tokens > 2:
        return output_tokens / (delta + eps)
    else:
        return kv_tokens / (delta + eps)


# ---------------------------------------------------------------------------
# Strategy victim selection
# ---------------------------------------------------------------------------

def select_victim_lifo(candidates: list[SimRequest]) -> SimRequest | None:
    """LIFO: most recently admitted (highest arrival time in running)."""
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.prefill_start_ms)


def select_victim_largest_first(candidates: list[SimRequest]) -> SimRequest | None:
    """Largest-First: highest KV occupancy."""
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.kv_tokens)


def select_victim_bidkv(candidates: list[SimRequest]) -> SimRequest | None:
    """BidKV: highest utility U = tokens_freed / (δ + ε)."""
    if not candidates:
        return None
    def u(r: SimRequest) -> float:
        return compute_utility(r.prompt_tokens, r.output_tokens,
                               r.max_output_tokens, r.num_preemptions, r.kv_tokens)
    return max(candidates, key=u)


def select_victim_pe_sjf(candidates: list[SimRequest]) -> SimRequest | None:
    """PE-SJF: LIFO (same as PE for running queue victim selection)."""
    return select_victim_lifo(candidates)


STRATEGIES = {
    "pe-lifo": select_victim_lifo,
    "largest-first": select_victim_largest_first,
    "pe-sjf": select_victim_pe_sjf,
    "bidkv": select_victim_bidkv,
}


# ---------------------------------------------------------------------------
# Discrete-event simulation
# ---------------------------------------------------------------------------

def simulate_workload(trace_requests: list[dict]) -> list[dict]:
    """Simulate vLLM-like scheduling and record KV pressure events.

    Returns a list of event dicts, each containing:
      - candidates: list of candidate dicts with all fields needed
      - strategy_choices: {strategy_name: chosen_request_id}
      - kv_usage: float
      - ts: simulation time in seconds
    """
    # Create SimRequest objects
    all_requests: list[SimRequest] = []
    for r in trace_requests:
        all_requests.append(SimRequest(
            request_id=r["request_id"],
            prompt_tokens=r["prompt_tokens"],
            max_output_tokens=r["max_tokens"],
            arrival_ms=r["arrival_ms"],
        ))

    waiting: list[SimRequest] = []
    running: list[SimRequest] = []
    done: list[SimRequest] = []
    events: list[dict] = []

    next_arrival_idx = 0
    t_ms = 0.0
    max_t_ms = max(r.arrival_ms for r in all_requests) + 600_000  # +10 min safety

    last_event_t = -2000.0  # ensure first event is logged

    while t_ms < max_t_ms:
        # 1. Admit new arrivals
        while next_arrival_idx < len(all_requests):
            r = all_requests[next_arrival_idx]
            if r.arrival_ms <= t_ms:
                r.state = "waiting"
                waiting.append(r)
                next_arrival_idx += 1
            else:
                break

        # 2. Advance running requests
        for r in running:
            if r.state == "prefilling":
                # Prefill advances by PREFILL_TPS * step_size
                tokens_this_step = int(PREFILL_TPS * STEP_MS / 1000)
                r.kv_tokens = min(r.prompt_tokens, r.kv_tokens + tokens_this_step)
                if r.kv_tokens >= r.prompt_tokens:
                    r.state = "decoding"
                    r.decode_start_ms = t_ms
                    r.kv_tokens = r.prompt_tokens
            elif r.state == "decoding":
                tokens_this_step = max(1, int(DECODE_TPS * STEP_MS / 1000))
                r.output_tokens += tokens_this_step
                r.kv_tokens = r.prompt_tokens + r.output_tokens
                if r.output_tokens >= r.max_output_tokens:
                    r.state = "done"

        # 3. Move completed requests out
        still_running = []
        for r in running:
            if r.state == "done":
                done.append(r)
            else:
                still_running.append(r)
        running = still_running

        # 4. Compute KV usage
        total_kv = sum(r.kv_tokens for r in running)
        kv_usage = total_kv / KV_CAPACITY if KV_CAPACITY > 0 else 0.0

        # 5. Admit waiting → running (if capacity allows)
        while waiting and len(running) < MAX_CONCURRENT:
            # Check if admitting next request would fit
            next_req = waiting[0]
            # Estimate: admit if there's at least prompt_tokens of capacity
            if total_kv + next_req.prompt_tokens <= KV_CAPACITY:
                req = waiting.pop(0)
                req.state = "prefilling"
                req.prefill_start_ms = t_ms
                req.kv_tokens = 0
                running.append(req)
                total_kv += 0  # will grow during prefill
            else:
                break

        # 6. Record pressure event if KV > threshold
        if kv_usage >= KV_PRESSURE_THRESHOLD and len(running) >= 2:
            if t_ms - last_event_t >= 1000.0:  # min 1s between events
                last_event_t = t_ms

                # Build candidate snapshot
                candidates_data = []
                for r in running:
                    if r.kv_tokens <= 1:
                        continue
                    completion = min(1.0, r.output_tokens / r.max_output_tokens) if r.max_output_tokens > 0 else 0.0
                    delta = compute_delta(r.prompt_tokens, r.output_tokens,
                                          r.max_output_tokens, r.num_preemptions)
                    utility = compute_utility(r.prompt_tokens, r.output_tokens,
                                              r.max_output_tokens, r.num_preemptions, r.kv_tokens)
                    candidates_data.append({
                        "request_id": r.request_id,
                        "prompt_tokens": r.prompt_tokens,
                        "output_tokens": r.output_tokens,
                        "max_output_tokens": r.max_output_tokens,
                        "kv_tokens": r.kv_tokens,
                        "completion_ratio": round(completion, 4),
                        "num_preemptions": r.num_preemptions,
                        "delta": round(delta, 4),
                        "utility": round(utility, 2),
                    })

                if len(candidates_data) >= 2:
                    # Run all strategies
                    eligible = [r for r in running if r.kv_tokens > 1]
                    strategy_choices = {}
                    for name, fn in STRATEGIES.items():
                        victim = fn(eligible)
                        strategy_choices[name] = victim.request_id if victim else None

                    events.append({
                        "ts": round(t_ms / 1000, 3),
                        "kv_usage": round(kv_usage, 4),
                        "num_candidates": len(candidates_data),
                        "candidates": candidates_data,
                        "strategy_choices": strategy_choices,
                    })

        # 7. Simple preemption if over capacity (to keep simulation running)
        if total_kv > KV_CAPACITY and running:
            # Preempt the last admitted (LIFO) to simulate native behavior
            victim = max(running, key=lambda r: r.prefill_start_ms)
            running.remove(victim)
            victim.state = "waiting"
            victim.kv_tokens = 0
            victim.output_tokens = 0
            victim.num_preemptions += 1
            waiting.insert(0, victim)  # re-queue at front

        # 8. Check if done
        if next_arrival_idx >= len(all_requests) and not waiting and not running:
            break

        t_ms += STEP_MS

    return events


# ---------------------------------------------------------------------------
# Analysis (reused from plot_motivation_3panel.py)
# ---------------------------------------------------------------------------

def analyze_events(events: list[dict]) -> dict:
    """Compute per-event metrics for the 3 panels."""
    delta_ratios = []
    cost_ratios = []
    cum_lifo = []
    cum_bidkv = []
    cum_largest = []
    timestamps = []

    running_lifo = 0.0
    running_bidkv = 0.0
    running_largest = 0.0
    t0 = events[0]["ts"] if events else 0.0

    for ev in events:
        cands = ev["candidates"]
        choices = ev.get("strategy_choices", {})

        # Panel A: heterogeneity
        deltas = [c["delta"] for c in cands]
        d_min, d_max = min(deltas), max(deltas)
        if d_min > 0:
            delta_ratios.append(d_max / d_min)

        # Panel B: cost divergence
        lifo_rid = choices.get("pe-lifo")
        bidkv_rid = choices.get("bidkv")
        lifo_cand = next((c for c in cands if c["request_id"] == lifo_rid), None)
        bidkv_cand = next((c for c in cands if c["request_id"] == bidkv_rid), None)

        if lifo_cand and bidkv_cand:
            d_lifo = lifo_cand["delta"]
            d_bidkv = bidkv_cand["delta"]
            if d_bidkv > 0:
                cost_ratios.append(d_lifo / d_bidkv)

        # Panel C: cumulative waste
        largest_rid = choices.get("largest-first")
        largest_cand = next((c for c in cands if c["request_id"] == largest_rid), None)

        if lifo_cand:
            running_lifo += lifo_cand.get("output_tokens", 0)
        if bidkv_cand:
            running_bidkv += bidkv_cand.get("output_tokens", 0)
        if largest_cand:
            running_largest += largest_cand.get("output_tokens", 0)

        cum_lifo.append(running_lifo)
        cum_bidkv.append(running_bidkv)
        cum_largest.append(running_largest)
        timestamps.append(ev["ts"] - t0)

    return {
        "delta_ratios": delta_ratios,
        "cost_ratios": cost_ratios,
        "cum_lifo": cum_lifo,
        "cum_bidkv": cum_bidkv,
        "cum_largest": cum_largest,
        "timestamps": timestamps,
        "n_events": len(events),
    }


# ---------------------------------------------------------------------------
# Plotting (same style as plot_motivation_3panel.py)
# ---------------------------------------------------------------------------

def plot_3panel(data: dict, output_path: Path) -> None:
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

    FIG_W = 3.33
    FIG_H = 5.8
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1, figsize=(FIG_W, FIG_H),
        gridspec_kw={"hspace": 0.65}
    )

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

    # ── Panel A: CDF of δ heterogeneity ──────────────────────────
    sorted_ratios = np.sort(delta_ratios)
    cdf_y = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

    ax_a.plot(sorted_ratios, cdf_y, color=COL_BIDKV, linewidth=1.5)
    ax_a.fill_between(sorted_ratios, 0, cdf_y, alpha=0.15, color=COL_BIDKV)

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
        f"(a) Reclamation cost heterogeneity\n({n_events} KV-pressure events, ShareGPT mixed, rate=5.7)",
        pad=3
    )
    ax_a.set_xlim(left=1.0)
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, axis="both", linestyle=":", alpha=0.4)

    # ── Panel B: Per-event cost ratio ──────────────────────────────
    sorted_cr = np.sort(cost_ratios)
    n_cr = len(sorted_cr)
    x_idx = np.arange(n_cr)
    colors_b = [COL_LIFO if r > 1.0 else COL_BIDKV for r in sorted_cr]

    ax_b.bar(x_idx, sorted_cr, width=1.0, color=colors_b, alpha=0.7, edgecolor="none")
    ax_b.axhline(1.0, color="black", linewidth=0.8, linestyle="-")

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

    # ── Panel C: Cumulative recompute waste ────────────────────────
    ts_min = timestamps / 60.0

    ax_c.plot(ts_min, cum_lifo / 1000, color=COL_LIFO, linewidth=1.3,
              label="LIFO (vLLM default)")
    ax_c.plot(ts_min, cum_largest / 1000, color=COL_LARGEST, linewidth=1.1,
              linestyle="--", label="Largest-First")
    ax_c.plot(ts_min, cum_bidkv / 1000, color=COL_BIDKV, linewidth=1.3,
              label="BidKV (ours)")
    ax_c.fill_between(ts_min, cum_bidkv / 1000, cum_lifo / 1000,
                       alpha=0.08, color=COL_LIFO)

    if cum_bidkv[-1] > 0:
        ratio_final = cum_lifo[-1] / cum_bidkv[-1]
        ax_c.annotate(
            f"{ratio_final:.1f}$\\times$ more\nwasted recompute",
            xy=(ts_min[-1], cum_lifo[-1] / 1000),
            xytext=(ts_min[-1] * 0.60, cum_lifo[-1] / 1000 * 0.85),
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

    fig.align_ylabels([ax_a, ax_b, ax_c])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"\nFigure saved → {output_path}")

    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, bbox_inches="tight", format="png", dpi=300)
    print(f"PNG preview → {png_path}")


def print_summary(data: dict) -> None:
    dr = np.array(data["delta_ratios"])
    cr = np.array(data["cost_ratios"])

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (for paper §2 text)")
    print("=" * 60)
    print(f"Total pressure events: {data['n_events']}")

    print(f"\nPanel A — Heterogeneity:")
    print(f"  Median δ ratio: {np.median(dr):.1f}×")
    print(f"  Events ≥ 3×: {np.mean(dr >= 3.0) * 100:.0f}%")
    print(f"  Events ≥ 5×: {np.mean(dr >= 5.0) * 100:.0f}%")
    print(f"  Events ≥ 10×: {np.mean(dr >= 10.0) * 100:.0f}%")

    print(f"\nPanel B — Policy divergence:")
    print(f"  LIFO costlier: {np.mean(cr > 1.0) * 100:.0f}%")
    print(f"  Median ratio: {np.median(cr):.1f}×")
    print(f"  P90 ratio: {np.percentile(cr, 90):.1f}×")

    cl = data["cum_lifo"][-1] if data["cum_lifo"] else 0
    cb = data["cum_bidkv"][-1] if data["cum_bidkv"] else 0
    clf = data["cum_largest"][-1] if data["cum_largest"] else 0
    print(f"\nPanel C — Cumulative waste:")
    print(f"  LIFO: {cl:,.0f} tokens")
    print(f"  Largest-First: {clf:,.0f} tokens")
    print(f"  BidKV: {cb:,.0f} tokens")
    if cb > 0:
        print(f"  LIFO / BidKV: {cl / cb:.1f}×")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate motivation figure from frozen trace simulation"
    )
    parser.add_argument(
        "--trace", type=str,
        default="experiments/vllm/traces/mixed_rate5.7.json",
        help="Path to frozen trace JSON"
    )
    parser.add_argument(
        "--output", type=str,
        default="paper/figures/fig_preliminary_motivation.pdf",
        help="Output PDF path"
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"ERROR: trace file not found: {trace_path}")
        sys.exit(1)

    print(f"Loading trace: {trace_path}")
    trace_requests = load_trace(trace_path)
    print(f"  {len(trace_requests)} requests")

    print("Simulating vLLM-like scheduling...")
    events = simulate_workload(trace_requests)
    print(f"  {len(events)} KV-pressure events captured")

    if len(events) < 5:
        print(f"WARNING: Only {len(events)} events — figure may be sparse.")
        print("  Consider using long_context trace for more pressure events.")

    # Save events for reproducibility
    events_path = Path("results/preliminary_motivation/simulated_events.jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"  Events saved → {events_path}")

    data = analyze_events(events)
    print_summary(data)
    plot_3panel(data, Path(args.output))


if __name__ == "__main__":
    main()
