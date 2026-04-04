"""Full 7-strategy analysis for mixed workload experiments."""

from __future__ import annotations

import json
import math
import os
import statistics

strategies = [
    "bidkv",
    "preempt-evict",
    "preempt-evict-sjf",
    "h2o-style",
    "static-random",
    "uniform",
    "slack-aware",
]
rates = ["2.0", "3.8", "5.7"]
DATA_DIR = "/home/cyb/bidkv/results/vllm_v8_full_validation"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def extract_metrics(filepath: str) -> dict:
    with open(filepath) as f:
        d = json.load(f)
    summary = d["summary"]
    results = d["request_results"]

    ttfts = []
    tpots = []
    for r in results:
        if r.get("error"):
            continue
        ttft = r.get("ttft_ms", 0)
        total = r.get("total_latency_ms", 0)
        comp = r.get("completion_tokens", 1)
        if ttft > 0:
            ttfts.append(ttft)
        if total > 0 and comp > 1:
            tpot = (total - ttft) / (comp - 1)
            tpots.append(tpot)

    successful = [r for r in results if not r.get("error")]
    total_reqs = len(results)
    success_count = len(successful)

    slo_300 = sum(1 for t in ttfts if t <= 300) / max(len(ttfts), 1) * 100
    slo_500 = sum(1 for t in ttfts if t <= 500) / max(len(ttfts), 1) * 100
    slo_1000 = sum(1 for t in ttfts if t <= 1000) / max(len(ttfts), 1) * 100

    dur = d.get("duration_s", 1)
    good_reqs = sum(1 for t in ttfts if t <= 500)
    goodput_500 = good_reqs / dur

    return {
        "slo_300ms": slo_300,
        "slo_500ms": slo_500,
        "slo_1s": slo_1000,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p95": percentile(tpots, 95),
        "tpot_p99": percentile(tpots, 99),
        "norm_lat": summary.get("normalized_latency_ms_per_token", 0),
        "goodput_500ms": goodput_500,
        "throughput": summary.get("throughput_rps", 0),
        "success_rate": success_count / max(total_reqs, 1) * 100,
    }


def main() -> None:
    os.chdir(DATA_DIR)

    all_data: dict = {}
    for strat in strategies:
        all_data[strat] = {}
        for rate in rates:
            runs = []
            for r in range(3):
                fname = f"{strat}__mixed__rate{rate}__r{r}.json"
                if os.path.exists(fname):
                    try:
                        runs.append(extract_metrics(fname))
                    except Exception as e:
                        print(f"ERROR: {fname}: {e}")
            if runs:
                avg = {}
                for k in runs[0]:
                    vals = [run[k] for run in runs]
                    avg[k] = statistics.mean(vals)
                all_data[strat][rate] = avg

    main_metrics = [
        ("goodput_500ms", "Goodput(500ms)", "rps", "{:.2f}"),
        ("slo_300ms", "SLO(300ms)", "%", "{:.1f}%"),
        ("ttft_p95", "TTFT p95", "ms", "{:.0f}"),
        ("tpot_p95", "TPOT p95", "ms", "{:.1f}"),
        ("norm_lat", "Norm Lat", "ms/tok", "{:.1f}"),
    ]

    print("=" * 100)
    print("TABLE 1: MAIN 7-STRATEGY COMPARISON (Mixed workload, 3-run avg)")
    print("=" * 100)

    for mk, mn, unit, fmt in main_metrics:
        print(f"\n--- {mn} ({unit}) ---")
        hdr = f"{'Strategy':<20} | {'r=2.0':>10} | {'r=3.8':>10} | {'r=5.7':>10} | {'Avg':>10}"
        print(hdr)
        print("-" * len(hdr))

        strat_avgs = []
        for strat in strategies:
            vals = [all_data[strat].get(rate, {}).get(mk, 0) for rate in rates]
            avg = statistics.mean(vals)
            strat_avgs.append((strat, avg))

        if mk in ("ttft_p95", "tpot_p95", "norm_lat"):
            strat_avgs.sort(key=lambda x: x[1])
        else:
            strat_avgs.sort(key=lambda x: -x[1])

        best_strat = strat_avgs[0][0]

        for strat in strategies:
            vals = [all_data[strat].get(rate, {}).get(mk, 0) for rate in rates]
            avg = statistics.mean(vals)
            parts = [fmt.format(v) for v in vals]
            avg_s = fmt.format(avg)
            marker = (
                " ★"
                if strat == "bidkv"
                else (" ←BEST" if strat == best_strat and strat != "bidkv" else "")
            )
            print(
                f"{strat:<20} | {parts[0]:>10} | {parts[1]:>10} | "
                f"{parts[2]:>10} | {avg_s:>10}{marker}"
            )

    # Head-to-head
    print("\n" + "=" * 100)
    print("BIDKV HEAD-TO-HEAD (5 main metrics × 3 rates = 15 cells each)")
    print("=" * 100)

    bidkv_data = all_data["bidkv"]
    total_wins = 0
    total_cells = 0
    for opponent in strategies:
        if opponent == "bidkv":
            continue
        opp_data = all_data[opponent]
        wins = losses = ties = 0
        cells = 0
        for mk, _mn, _unit, _fmt in main_metrics:
            for rate in rates:
                bv = bidkv_data.get(rate, {}).get(mk, 0)
                ov = opp_data.get(rate, {}).get(mk, 0)
                cells += 1
                if mk in ("ttft_p95", "tpot_p95", "norm_lat"):
                    if bv < ov * 0.99:
                        wins += 1
                    elif bv > ov * 1.01:
                        losses += 1
                    else:
                        ties += 1
                else:
                    if bv > ov * 1.01:
                        wins += 1
                    elif bv < ov * 0.99:
                        losses += 1
                    else:
                        ties += 1
        total_wins += wins
        total_cells += cells
        pct = wins / cells * 100
        print(f"  vs {opponent:<20}: {wins}W/{losses}L/{ties}T ({pct:.0f}%)")

    overall_pct = total_wins / total_cells * 100
    print(f"\n  OVERALL: {total_wins}/{total_cells} wins ({overall_pct:.0f}%)")

    # Ranking
    print("\n" + "=" * 100)
    print("BIDKV RANKING PER METRIC (avg across rates)")
    print("=" * 100)

    for mk, mn, _unit, fmt in main_metrics:
        strat_avgs = []
        for strat in strategies:
            vals = [all_data[strat].get(rate, {}).get(mk, 0) for rate in rates]
            avg = statistics.mean(vals)
            strat_avgs.append((strat, avg))
        if mk in ("ttft_p95", "tpot_p95", "norm_lat"):
            strat_avgs.sort(key=lambda x: x[1])
        else:
            strat_avgs.sort(key=lambda x: -x[1])
        bidkv_rank = next(i + 1 for i, (s, _) in enumerate(strat_avgs) if s == "bidkv")
        bidkv_val = next(v for s, v in strat_avgs if s == "bidkv")
        best_s, best_val = strat_avgs[0]
        delta = ""
        if best_s != "bidkv" and best_val != 0:
            if mk in ("ttft_p95", "tpot_p95", "norm_lat"):
                delta = f"  (vs {best_s}: +{(bidkv_val / best_val - 1) * 100:.1f}%)"
            else:
                delta = f"  (vs {best_s}: {(bidkv_val / best_val - 1) * 100:.1f}%)"
        print(f"  {mn:<15}: rank {bidkv_rank}/7 (avg={fmt.format(bidkv_val)}){delta}")

    # Supplementary tables
    supp = [
        ("ttft_p50", "TTFT p50 (ms)", "{:.0f}"),
        ("tpot_p99", "TPOT p99 (ms)", "{:.1f}"),
        ("throughput", "Throughput (rps)", "{:.2f}"),
        ("success_rate", "Success Rate (%)", "{:.1f}%"),
        ("slo_500ms", "SLO(500ms) (%)", "{:.1f}%"),
        ("slo_1s", "SLO(1s) (%)", "{:.1f}%"),
    ]

    for mk, mn, fmt in supp:
        print(f"\n--- {mn} ---")
        hdr = f"{'Strategy':<20} | {'r=2.0':>10} | {'r=3.8':>10} | {'r=5.7':>10}"
        print(hdr)
        print("-" * len(hdr))
        for strat in strategies:
            vals = [all_data[strat].get(rate, {}).get(mk, 0) for rate in rates]
            parts = [fmt.format(v) for v in vals]
            marker = " ★" if strat == "bidkv" else ""
            print(f"{strat:<20} | {parts[0]:>10} | {parts[1]:>10} | {parts[2]:>10}{marker}")


if __name__ == "__main__":
    main()
