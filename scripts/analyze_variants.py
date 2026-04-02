#!/usr/bin/env python3
"""Analyze v10 variant results vs v8 baselines."""
from __future__ import annotations

import json
import glob
import statistics
import sys
from pathlib import Path


def compute_metrics(d: dict) -> dict | None:
    reqs = d.get("request_results", [])
    ok = [
        r
        for r in reqs
        if r.get("completion_tokens", 0) > 0 and r.get("ttft_ms", 0) > 0
    ]
    if not ok:
        return None
    ttfts = sorted([r["ttft_ms"] for r in ok])
    tpots = sorted(
        [
            (r["total_latency_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
            for r in ok
            if r["completion_tokens"] > 1
        ]
    )
    dur = d.get("duration_s", 1)
    n, nt = len(ttfts), len(tpots)
    return {
        "goodput": sum(1 for r in ok if r["ttft_ms"] <= 500) / dur,
        "slo300": sum(1 for r in ok if r["ttft_ms"] <= 300) / len(ok) * 100,
        "ttft_p95": ttfts[int(n * 0.95)],
        "tpot_p95": tpots[int(nt * 0.95)] if tpots else 0,
        "normlat": statistics.mean(
            [
                r["total_latency_ms"] / r["completion_tokens"]
                for r in ok
                if r["completion_tokens"] > 0
            ]
        ),
    }


def main() -> None:
    base = Path("results/vllm_v10_variants")
    if not base.exists():
        print("No variant results found")
        sys.exit(1)

    # Load variant results (rate=5.7 only)
    variants: dict[str, dict] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        jsons = list(d.glob("*.json"))
        for f in jsons:
            data = json.loads(f.read_text())
            if "strategy" not in data:
                continue
            m = compute_metrics(data)
            if m:
                variants[d.name] = m

    # Load v8 reference data at rate=5.7 (3-run avg)
    v8_ref: dict[str, list[dict]] = {}
    for f in glob.glob("results/vllm_v8_full_validation/*.json"):
        data = json.loads(Path(f).read_text())
        if "strategy" not in data:
            continue
        if data.get("request_rate") != 5.7:
            continue
        m = compute_metrics(data)
        if m:
            s = data["strategy"]
            if s not in v8_ref:
                v8_ref[s] = []
            v8_ref[s].append(m)

    v8_avg: dict[str, dict] = {}
    for s, runs in v8_ref.items():
        v8_avg[s] = {
            k: statistics.mean([r[k] for r in runs])
            for k in ["goodput", "slo300", "ttft_p95", "tpot_p95", "normlat"]
        }

    # Print comparison
    header = f"{'Label':<28} {'Goodput':>8} {'SLO300':>8} {'TTFT95':>8} {'TPOT95':>8} {'NrmLat':>8}"
    sep = "─" * len(header)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   v10 Variant Comparison (rate=5.7) vs v8 All Strategies       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(header)
    print(sep)

    # Print v10 variants first
    for name in sorted(variants):
        m = variants[name]
        marker = " ★" if name == "v8_baseline" else ""
        print(
            f"{'v10-' + name:<28} {m['goodput']:>8.2f} {m['slo300']:>7.1f}%"
            f" {m['ttft_p95']:>8.0f} {m['tpot_p95']:>8.1f} {m['normlat']:>8.1f}{marker}"
        )

    print(sep)

    # Print v8 references
    for s in [
        "bidkv",
        "static-random",
        "uniform",
        "h2o-style",
        "preempt-evict-sjf",
        "slack-aware",
        "preempt-evict",
    ]:
        if s not in v8_avg:
            continue
        m = v8_avg[s]
        print(
            f"{'v8-' + s:<28} {m['goodput']:>8.2f} {m['slo300']:>7.1f}%"
            f" {m['ttft_p95']:>8.0f} {m['tpot_p95']:>8.1f} {m['normlat']:>8.1f}"
        )

    # Ranking at rate=5.7 (all entries)
    all_entries = {}
    for name, m in variants.items():
        all_entries[f"v10-{name}"] = m
    for s, m in v8_avg.items():
        all_entries[f"v8-{s}"] = m

    metrics_spec = [
        ("goodput", True),
        ("slo300", True),
        ("ttft_p95", False),
        ("tpot_p95", False),
        ("normlat", False),
    ]

    print()
    print("Rate=5.7 Ranking:")
    print(
        f"{'Label':<28} {'Goodput':>8} {'SLO300':>8} {'TTFT95':>8}"
        f" {'TPOT95':>8} {'NrmLat':>8} {'Sum':>5} {'Wins':>5}"
    )
    print(sep)

    ranks: dict[str, dict[str, int]] = {t: {} for t in all_entries}
    for mk, higher in metrics_spec:
        items = sorted(
            [(all_entries[t][mk], t) for t in all_entries], reverse=higher
        )
        for i, (_, t) in enumerate(items):
            ranks[t][mk] = i + 1

    for t in sorted(all_entries, key=lambda t: sum(ranks[t].values())):
        r = ranks[t]
        s = sum(r.values())
        w = sum(1 for v in r.values() if v == 1)
        is_v10 = t.startswith("v10-")
        marker = " ◀" if is_v10 else ""
        print(
            f"{t:<28} {r['goodput']:>8} {r['slo300']:>8} {r['ttft_p95']:>8}"
            f" {r['tpot_p95']:>8} {r['normlat']:>8} {s:>5} {w:>5}{marker}"
        )


if __name__ == "__main__":
    main()
