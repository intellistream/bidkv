#!/usr/bin/env python3
"""Analyze v8 validation experiment results across all available data."""

from __future__ import annotations

import json
from pathlib import Path


def load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def analyze_run(data: dict) -> dict:
    rr = data["request_results"]
    # Success = has completion_tokens > 0 and no error
    ttfts = []
    tpots = []
    success = 0
    for r in rr:
        ct = r.get("completion_tokens", 0) or 0
        err = r.get("error", "")
        ttft = r.get("ttft_ms")
        total = r.get("total_latency_ms", 0) or 0
        if ct > 0 and not err:
            success += 1
            if ttft and ttft > 0:
                ttfts.append(ttft)
            if ct > 1 and ttft and total > ttft:
                tpot = (total - ttft) / (ct - 1)
                tpots.append(tpot)

    duration = data.get("duration_s", 0)
    tput = success / duration if duration > 0 else 0
    slo1s = sum(1 for t in ttfts if t <= 1000) / len(ttfts) * 100 if ttfts else 0

    return {
        "success": success,
        "total": len(rr),
        "tput": tput,
        "slo1s": slo1s,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p95": percentile(tpots, 95),
    }


def main():
    base = Path("/home/cyb/bidkv")
    # Scan all result directories
    search_dirs = [
        ("v8-new", base / "results" / "vllm_v8_analysis"),
        ("v8-old", base / "results" / "vllm_validation_v8"),
        ("v7-bl", base / "results" / "vllm_validation_v7"),
    ]

    # Group: (strategy, rate) -> list of metrics
    groups: dict[tuple[str, str], list[dict]] = {}
    for tag, d in search_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if "consistency" in f.name or "report" in f.name or "server" in f.name:
                continue
            parts = f.stem.split("__")
            if len(parts) < 4:
                continue
            strat, _wl, rate_str, _run = parts[0], parts[1], parts[2], parts[3]
            rate = rate_str.replace("rate", "")
            # Skip v7 bidkv (it's v7 code, not v8)
            if tag == "v7-bl" and strat == "bidkv":
                continue
            key = (strat, rate)
            if key not in groups:
                groups[key] = []
            data = load_run(f)
            m = analyze_run(data)
            m["file"] = f"{tag}/{f.name}"
            groups[key].append(m)

    # Print header
    hdr = f"{'Strategy':25s} {'Rate':5s} {'#':3s} {'Tput':6s} {'SLO1s':6s} {'p50':7s} {'p95':7s} {'p99':7s} {'TPOT95':7s}"
    print(hdr)
    print("=" * len(hdr))

    for key in sorted(groups.keys()):
        strat, rate = key
        runs = groups[key]
        n = len(runs)

        def avg(field):
            return sum(r[field] for r in runs) / n

        slo_str = f"{avg('slo1s'):.1f}%"
        print(
            f"{strat:25s} {rate:5s} {n:3d} "
            f"{avg('tput'):6.2f} {slo_str:6s} "
            f"{avg('ttft_p50'):7.0f} {avg('ttft_p95'):7.0f} {avg('ttft_p99'):7.0f} "
            f"{avg('tpot_p95'):7.1f}"
        )

    # Per-run detail for rate=5.7
    print("\n\n=== Per-run detail (rate=5.7) ===")
    print(f"{'File':55s} {'Tput':6s} {'SLO1s':6s} {'p50':7s} {'p95':7s} {'p99':7s} {'TPOT95':7s}")
    print("-" * 100)
    for key in sorted(groups.keys()):
        strat, rate = key
        if rate != "5.7":
            continue
        for r in groups[key]:
            slo_str = f"{r['slo1s']:.1f}%"
            print(
                f"{r['file']:55s} {r['tput']:6.2f} {slo_str:6s} "
                f"{r['ttft_p50']:7.0f} {r['ttft_p95']:7.0f} {r['ttft_p99']:7.0f} "
                f"{r['tpot_p95']:7.1f}"
            )

    # Per-run detail for rate=2.0
    print("\n\n=== Per-run detail (rate=2.0) ===")
    print(f"{'File':55s} {'Tput':6s} {'SLO1s':6s} {'p50':7s} {'p95':7s} {'p99':7s} {'TPOT95':7s}")
    print("-" * 100)
    for key in sorted(groups.keys()):
        strat, rate = key
        if rate != "2.0":
            continue
        for r in groups[key]:
            slo_str = f"{r['slo1s']:.1f}%"
            print(
                f"{r['file']:55s} {r['tput']:6.2f} {slo_str:6s} "
                f"{r['ttft_p50']:7.0f} {r['ttft_p95']:7.0f} {r['ttft_p99']:7.0f} "
                f"{r['tpot_p95']:7.1f}"
            )


if __name__ == "__main__":
    main()
