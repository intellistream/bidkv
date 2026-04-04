"""
Preliminary motivation figure for BidKV SC 2026.

Figure layout (one column wide, two panels):

  Panel A — Per-request breakdown bar chart at the KV-pressure snapshot.
    Each of the 10 concurrent requests is one group on the x-axis, sorted by
    kv_held descending.  Two bars per group:
      Blue  = kv_held (tokens freed if this request is preempted)
      Orange = recompute_cost  = kv_held × completion_ratio
               (how many tokens the system must re-process after preemption;
                near-complete requests are expensive to recompute)
    Annotations:
      "LIFO picks"    → the request with highest kv_held (most tokens freed,
                         but also very high recompute cost)
      "Utility picks" → the request with lowest completion_ratio (cheapest
                         to recompute relative to tokens freed)
    Key message: LIFO frees only marginally more KV than utility-guided,
    but its victim has a much higher recompute cost — a poor trade-off.

  Panel B — Aggregated comparison bar chart for two simulated policies.
    Simulates freeing 20% of total KV capacity under:
      LIFO-like   : always pick highest kv_held first
      Utility-guided : always pick lowest completion_ratio first
    Three grouped bars: tokens freed, avg recompute cost, estimated cascade.
    Key message: LIFO frees ~same KV but at 2× the recompute burden.

Run from repo root:
    python scripts/plot_preliminary_motivation.py

Output:
    paper/figures/fig_preliminary_motivation.pdf
    results/preliminary_motivation/snapshot_data.json   (intermediate)
"""

from __future__ import annotations

import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Attempt to import matplotlib; guide user if missing.
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    print("ERROR: matplotlib is required.  Install via:")
    print("  pip install matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACE_PATH = ROOT / "experiments" / "vllm" / "traces" / "long_rate0.5.json"
OUT_PDF = ROOT / "paper" / "figures" / "fig_preliminary_motivation.pdf"
OUT_DATA = ROOT / "results" / "preliminary_motivation" / "snapshot_data.json"

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_DATA.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load trace
# ---------------------------------------------------------------------------
print(f"Loading trace: {TRACE_PATH}")
with open(TRACE_PATH) as f:
    trace = json.load(f)

requests = trace["requests"]
request_rate = trace["request_rate"]  # req/s = 0.5
print(f"  {len(requests)} requests, rate={request_rate} req/s")

# ---------------------------------------------------------------------------
# Simulate a KV-pressure event snapshot
#
# We pick a window [T_start, T_end] where the concurrent batch is largest,
# treating each request as occupying KV from its arrival until arrival +
# (prompt_tokens / 10 + max_tokens / decode_speed_tps).  We use a simple
# approximation:
#   service_time ~ prompt_tokens * prefill_ms_per_tok + max_tokens * decode_ms
# For the purpose of this figure we just need the *relative* distribution to
# be realistic, so we use:
#   kv_tokens  = prompt_tokens          (held from start of prefill)
#   duration_s = prompt_tokens / 500 + max_tokens / 40
#                (500 tok/s prefill, 40 tok/s decode on A6000 8B bf16)
# ---------------------------------------------------------------------------

PREFILL_TPS = 500  # tokens/s estimated prefill throughput
DECODE_TPS = 40  # tokens/s estimated decode throughput

events: list[dict] = []
for req in requests:
    arrival_s = req["arrival_time_ms"] / 1000.0
    prompt_tok = int(req["metadata"]["actual_prompt_tokens"])
    max_tok = req["max_tokens"]
    duration_s = prompt_tok / PREFILL_TPS + max_tok / DECODE_TPS
    events.append(
        {
            "request_id": req["request_id"],
            "arrival_s": arrival_s,
            "end_s": arrival_s + duration_s,
            "prompt_tokens": prompt_tok,
            "max_tokens": max_tok,
            "duration_s": duration_s,
        }
    )

# Find the moment of maximum concurrency (= KV pressure peak)
# Scan at 1-second granularity over the trace window.
t_min = min(e["arrival_s"] for e in events)
t_max = max(e["end_s"] for e in events)

best_t = t_min
best_batch: list[dict] = []
for t in [t_min + i for i in range(int(t_max - t_min) + 1)]:
    batch = [e for e in events if e["arrival_s"] <= t < e["end_s"]]
    if len(batch) > len(best_batch):
        best_batch = batch
        best_t = t

print(f"  Peak concurrency at t={best_t:.1f}s: {len(best_batch)} concurrent requests")

# Compute completion ratio at the snapshot moment
snapshot: list[dict] = []
for e in best_batch:
    elapsed = best_t - e["arrival_s"]
    completion_ratio = min(elapsed / e["duration_s"], 1.0)
    # KV tokens currently held ≈ prompt_tokens + generated_so_far
    generated_so_far = int(min(elapsed * DECODE_TPS, e["max_tokens"]))
    kv_held = e["prompt_tokens"] + generated_so_far
    snapshot.append(
        {
            "request_id": e["request_id"],
            "completion_ratio": completion_ratio,
            "kv_held": kv_held,
            "prompt_tokens": e["prompt_tokens"],
            "max_tokens": e["max_tokens"],
        }
    )

snapshot.sort(key=lambda x: x["completion_ratio"])

# Save intermediate data
with open(OUT_DATA, "w") as f:
    json.dump({"snapshot_t": best_t, "requests": snapshot}, f, indent=2)
print(f"  Snapshot saved → {OUT_DATA}")

# ---------------------------------------------------------------------------
# Panel B: Cascade analysis
#
# Simulate two victim-selection policies on the same snapshot:
#   Policy L (LIFO-like / coarse): always pick the request with the most
#             kv_held (greedy on tokens-freed, ignoring completion cost).
#   Policy U (utility-guided):    pick the request with the lowest
#             completion_ratio first (least recompute cost per token freed).
#
# For each policy, simulate a sequence of preemptions until 20% of total KV
# Simulate two policies freeing 20% of total KV
# ---------------------------------------------------------------------------

BUDGET_FRACTION = 0.20

total_kv = sum(s["kv_held"] for s in snapshot)
budget_tokens = total_kv * BUDGET_FRACTION


def simulate_policy(policy: str, batch: list[dict]) -> dict:
    remaining = [dict(s) for s in batch]
    freed = 0
    victims: list[dict] = []
    while freed < budget_tokens and remaining:
        if policy == "lifo":
            idx = max(range(len(remaining)), key=lambda i: remaining[i]["kv_held"])
        else:
            idx = min(range(len(remaining)), key=lambda i: remaining[i]["completion_ratio"])
        v = remaining.pop(idx)
        freed += v["kv_held"]
        victims.append(v)
    if not victims:
        return {
            "avg_victim_kv": 0,
            "avg_recompute": 0,
            "cascade_factor": 0,
            "total_freed": 0,
            "n_victims": 0,
            "victims": [],
        }
    avg_kv = sum(v["kv_held"] for v in victims) / len(victims)
    avg_recompute = sum(v["kv_held"] * v["completion_ratio"] for v in victims) / len(victims)
    cascade = sum(1.0 + v["completion_ratio"] for v in victims) / len(victims)
    return {
        "avg_victim_kv": avg_kv,
        "avg_recompute": avg_recompute,
        "cascade_factor": cascade,
        "total_freed": freed,
        "n_victims": len(victims),
        "victims": victims,
    }


lifo_stats = simulate_policy("lifo", snapshot)
util_stats = simulate_policy("utility", snapshot)

# Identify the specific victims for annotation in Panel A
lifo_victim_id = lifo_stats["victims"][0]["request_id"] if lifo_stats["victims"] else None
util_victim_id = util_stats["victims"][0]["request_id"] if util_stats["victims"] else None

print(
    f"  LIFO pick:    {lifo_victim_id}  kv={lifo_stats['victims'][0]['kv_held']}  "
    f"comp={lifo_stats['victims'][0]['completion_ratio']:.1%}"
)
print(
    f"  Utility pick: {util_victim_id}  kv={util_stats['victims'][0]['kv_held']}  "
    f"comp={util_stats['victims'][0]['completion_ratio']:.1%}"
)
print(
    f"  Recompute cost ratio (LIFO/Utility): "
    f"{lifo_stats['avg_recompute'] / max(util_stats['avg_recompute'], 1):.2f}×"
)

# ---------------------------------------------------------------------------
# Plotting — ACM sigconf style
# ---------------------------------------------------------------------------
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "figure.dpi": 200,
    }
)

FIG_W = 3.33
FIG_H = 4.4

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), gridspec_kw={"hspace": 0.55})

# ── Panel A: per-request grouped bar, sorted by kv_held desc ─────────────────
# Sort snapshot by kv_held descending so the "heaviest" request is on the left
sorted_snap = sorted(snapshot, key=lambda s: s["kv_held"], reverse=True)
n = len(sorted_snap)
req_labels = [f"R{i + 1}" for i in range(n)]
kv_vals = [s["kv_held"] for s in sorted_snap]
# recompute cost = tokens that must be reprocessed if preempted now
#                = kv_held × completion_ratio  (already-computed portion)
recomp_vals = [s["kv_held"] * s["completion_ratio"] for s in sorted_snap]

x_pos = np.arange(n)
bar_w = 0.38

bars_kv = ax_a.bar(
    x_pos - bar_w / 2,
    kv_vals,
    bar_w,
    color="#4878CF",
    alpha=0.85,
    label="KV freed if preempted ($r$)",
)
bars_rc = ax_a.bar(
    x_pos + bar_w / 2,
    recomp_vals,
    bar_w,
    color="#D63027",
    alpha=0.80,
    hatch="//",
    label="Est. recompute cost ($r \\cdot$ progress)",
)

# Find positions of LIFO and Utility picks in the sorted array
lifo_pos = next(i for i, s in enumerate(sorted_snap) if s["request_id"] == lifo_victim_id)
util_pos = next(i for i, s in enumerate(sorted_snap) if s["request_id"] == util_victim_id)

# Annotate LIFO pick — text inside axes, arrow points to bar top
ax_a.annotate(
    "LIFO\npicks",
    xy=(lifo_pos - bar_w / 2, kv_vals[lifo_pos]),
    xytext=(lifo_pos + 0.85, kv_vals[lifo_pos] * 0.60),
    fontsize=5.5,
    color="#1a5fa8",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1a5fa8", lw=0.7),
    ha="center",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec="none"),
)
# Annotate Utility pick
ax_a.annotate(
    "Utility\npicks",
    xy=(util_pos + bar_w / 2, recomp_vals[util_pos]),
    xytext=(util_pos + bar_w / 2 + 1.1, recomp_vals[util_pos] * 2.8),
    fontsize=6,
    color="#8b1a1a",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#8b1a1a", lw=0.8),
    ha="center",
)

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(req_labels)
ax_a.set_xlabel("Concurrent requests at pressure event\n(sorted by KV tokens held, high→low)")
ax_a.set_ylabel("Tokens")
ax_a.set_title(
    f"(a) Per-request KV footprint vs. recompute cost\n"
    f"({n} concurrent requests, long-context, rate=0.5 req/s)",
    pad=3,
)
ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_a.legend(loc="upper right", framealpha=0.85, handletextpad=0.3, borderpad=0.4, ncol=1)
ax_a.grid(True, axis="y", linestyle=":", alpha=0.5)

# ── Panel B: policy comparison (3 metrics) ────────────────────────────────────
metrics = ["Tokens\nfreed", "Avg recompute\ncost", "Cascade\nfactor (est.)"]
lifo_vals_b = [lifo_stats["total_freed"], lifo_stats["avg_recompute"], lifo_stats["cascade_factor"]]
util_vals_b = [util_stats["total_freed"], util_stats["avg_recompute"], util_stats["cascade_factor"]]

# Normalise each metric to utility-guided = 1.0 for fair comparison
norm_vals = [max(u, 1e-9) for u in util_vals_b]
lifo_norm = [l / n_ for l, n_ in zip(lifo_vals_b, norm_vals)]
util_norm = [1.0] * len(metrics)

x2 = np.arange(len(metrics))
bar_w2 = 0.32

b_lifo = ax_b.bar(
    x2 - bar_w2 / 2, lifo_norm, bar_w2, color="#D63027", alpha=0.85, label="Coarse (LIFO-like)"
)
b_util = ax_b.bar(
    x2 + bar_w2 / 2, util_norm, bar_w2, color="#4878CF", alpha=0.85, label="Utility-guided"
)

# Value labels — show normalised multiplier (e.g. "5.0×") to match the y-axis
for bar_group in [b_lifo, b_util]:
    for bar in bar_group:
        h = bar.get_height()
        label = f"{h:.1f}×"
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.06,
            label,
            ha="center",
            va="bottom",
            fontsize=5.5,
        )

ax_b.axhline(
    1.0, color="gray", linestyle="--", linewidth=0.6, label="Utility-guided baseline (=1.0)"
)
ax_b.set_xticks(x2)
ax_b.set_xticklabels(metrics)
ax_b.set_ylabel("Normalized to utility-guided")
ax_b.set_title("(b) Aggregate policy comparison\n(freeing 20% of total KV capacity)", pad=3)
ax_b.set_ylim(0, max(lifo_norm) * 1.60)
ax_b.legend(loc="upper left", framealpha=0.85, handletextpad=0.3, borderpad=0.4)
ax_b.grid(True, axis="y", linestyle=":", alpha=0.5)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.align_ylabels([ax_a, ax_b])
plt.savefig(OUT_PDF, bbox_inches="tight", format="pdf")
print(f"\nFigure saved → {OUT_PDF}")
