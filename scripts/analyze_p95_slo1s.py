"""Analyze with SLO=1s for Mixed and P95 metrics (no P50)."""

from __future__ import annotations

import glob
import json
import statistics


def load(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    s = d.get("summary", d)
    total = s.get("total_requests", 0)
    success = s.get("successful_requests", s.get("success_count", 0))

    reqs = d.get("request_results", d.get("requests", []))
    ttft_list = []
    tpot_list = []
    for r in reqs:
        ttft = r.get("ttft_ms", 0)
        ok = r.get("success", r.get("error", "") == "")
        if ok and ttft > 0:
            ttft_list.append(ttft)
        # Compute TPOT from (total_latency - ttft) / (completion_tokens - 1)
        total_lat = r.get("total_latency_ms", 0)
        comp_tok = r.get("completion_tokens", 0)
        if ok and total_lat > 0 and ttft > 0 and comp_tok > 1:
            decode_ms = total_lat - ttft
            tpot = decode_ms / (comp_tok - 1)
            tpot_list.append(tpot)

    def p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        vals_s = sorted(vals)
        idx = int(len(vals_s) * 0.95)
        idx = min(idx, len(vals_s) - 1)
        return vals_s[idx]

    return {
        "total": total,
        "success": success,
        "success_pct": success / total * 100 if total else 0,
        "tput": s.get("throughput_rps", 0),
        "ttft95": p95(ttft_list),
        "ttft99": s.get("ttft_ms_p99", s.get("ttft_p99_ms", 0)),
        "tpot95": p95(tpot_list),
        "tpot99": s.get("tpot_ms_p99", s.get("tpot_p99_ms", 0)),
        "normlat": s.get("normalized_latency_ms_per_token", 0),
        "ttft_list": ttft_list,
    }


def slo_pct(ttft_list: list[float], threshold: float) -> float:
    if not ttft_list:
        return 0.0
    return sum(1 for t in ttft_list if t <= threshold) / len(ttft_list) * 100


def avg_runs(paths: list[str], slo_thresh: float) -> dict | None:
    runs = [load(p) for p in sorted(paths)]
    if not runs:
        return None
    keys = [
        "success_pct",
        "tput",
        "ttft95",
        "ttft99",
        "tpot95",
        "tpot99",
        "normlat",
    ]
    avg: dict = {}
    for k in keys:
        avg[k] = statistics.mean(r[k] for r in runs)
    avg["n"] = len(runs)
    slo_vals = [slo_pct(r["ttft_list"], slo_thresh) for r in runs]
    avg["slo"] = statistics.mean(slo_vals) if slo_vals else 0.0
    return avg


def single_run(path: str, slo_thresh: float) -> dict | None:
    try:
        m = load(path)
    except FileNotFoundError:
        return None
    m["slo"] = slo_pct(m["ttft_list"], slo_thresh)
    m["n"] = 1
    return m


def print_table(
    title: str,
    workload: str,
    rates: list,
    slo_thresh: float,
    strats: list[str],
    rate_fmt: str,
) -> None:
    header = (
        f"{'Strategy':<22} {'Rate':>5} {'Succ%':>6} {'SLO%':>6} "
        f"{'Tput':>6} {'TTFT p95':>9} {'TTFT p99':>9} "
        f"{'TPOT p95':>8} {'TPOT p99':>8} "
        f"{'NormLat':>8}"
    )
    sep = "=" * 110

    print(sep)
    print(f"{title}  (SLO = TTFT ≤ {slo_thresh:.0f}ms)")
    print(sep)
    print(header)
    print("-" * 110)

    for rate in rates:
        rate_s = f"{rate:{rate_fmt}}"
        for strat in strats:
            files = glob.glob(f"results/vllm_full_v1/{strat}__{workload}__rate{rate}__r*.json")
            if not files:
                continue
            m = avg_runs(files, slo_thresh)
            if m is None:
                continue
            print(
                f"{strat:<22} {rate_s:>5} "
                f"{m['success_pct']:>5.1f}% {m['slo']:>5.1f}% "
                f"{m['tput']:>6.2f} {m['ttft95']:>9.0f} "
                f"{m['ttft99']:>9.0f} "
                f"{m['tpot95']:>8.1f} {m['tpot99']:>8.1f} "
                f"{m['normlat']:>8.1f}"
            )

        # v5b bidkv
        v5b_sfx = "_mixed" if workload == "mixed" else ""
        v5b_path = f"results/vllm_validation_v5b{v5b_sfx}/bidkv__{workload}__rate{rate}__r0.json"
        v5 = single_run(v5b_path, slo_thresh)
        if v5:
            print(
                f"{'bidkv (v5b)':<22} {rate_s:>5} "
                f"{v5['success_pct']:>5.1f}% {v5['slo']:>5.1f}% "
                f"{v5['tput']:>6.2f} {v5['ttft95']:>9.0f} "
                f"{v5['ttft99']:>9.0f} "
                f"{v5['tpot95']:>8.1f} {v5['tpot99']:>8.1f} "
                f"{v5['normlat']:>8.1f}"
            )

        print()


def main() -> None:
    strats = [
        "preempt-evict",
        "preempt-evict-sjf",
        "static-random",
        "h2o-style",
        "uniform",
        "slack-aware",
        "bidkv",
    ]

    # Mixed: SLO = 1s (tighter threshold for differentiation)
    print_table(
        "MIXED WORKLOAD — SLO=1000ms",
        "mixed",
        [2.0, 3.8, 5.7],
        1000.0,
        strats,
        ".1f",
    )

    print()

    # Long-Context: SLO = 5s (unchanged)
    print_table(
        "LONG-CONTEXT WORKLOAD — SLO=5000ms",
        "long_context",
        [0.35, 0.5, 0.7],
        5000.0,
        strats,
        ".2f",
    )

    # Also show Mixed with SLO=2s for reference
    print()
    print_table(
        "MIXED WORKLOAD — SLO=2000ms (reference)",
        "mixed",
        [3.8, 5.7],
        2000.0,
        strats,
        ".1f",
    )


if __name__ == "__main__":
    main()
