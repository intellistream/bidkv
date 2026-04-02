"""Show full data tables for all strategies."""

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
    return {
        "total": total,
        "success": success,
        "success_pct": success / total * 100 if total else 0,
        "tput": s.get("throughput_rps", 0),
        "ttft50": s.get("ttft_ms_p50", s.get("ttft_p50_ms", 0)),
        "ttft99": s.get("ttft_ms_p99", s.get("ttft_p99_ms", 0)),
        "tpot50": s.get("tpot_ms_p50", s.get("tpot_p50_ms", 0)),
        "tpot99": s.get("tpot_ms_p99", s.get("tpot_p99_ms", 0)),
        "e2e50": s.get("e2e_latency_ms_p50", s.get("e2e_p50_ms", 0)),
        "e2e99": s.get("e2e_latency_ms_p99", s.get("e2e_p99_ms", 0)),
        "normlat": s.get("normalized_latency_ms_per_token", 0),
    }


def avg_runs(paths: list[str]) -> dict | None:
    runs = [load(p) for p in sorted(paths)]
    if not runs:
        return None
    keys = [
        "success_pct",
        "tput",
        "ttft50",
        "ttft99",
        "tpot50",
        "tpot99",
        "e2e50",
        "e2e99",
        "normlat",
    ]
    avg = {}
    for k in keys:
        avg[k] = statistics.mean(r[k] for r in runs)
    avg["success"] = f"{statistics.mean(r['success'] for r in runs):.0f}/{runs[0]['total']}"
    avg["n"] = len(runs)
    return avg


def fmt_row(name: str, rate: str, m: dict) -> str:
    return (
        f"{name:<22} {rate:>5} {m['success']:>10} "
        f"{m['tput']:>7.2f} {m['ttft50']:>9.0f} {m['ttft99']:>9.0f} "
        f"{m['tpot50']:>9.1f} {m['e2e50']:>9.0f} {m['e2e99']:>10.0f} "
        f"{m['normlat']:>8.1f}"
    )


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

    header = (
        f"{'Strategy':<22} {'Rate':>5} {'Success':>10} "
        f"{'Tput':>7} {'TTFT p50':>9} {'TTFT p99':>9} "
        f"{'TPOT p50':>9} {'E2E p50':>9} {'E2E p99':>10} "
        f"{'NormLat':>8}"
    )
    sep = "=" * 130

    # ===== TABLE 1: Mixed =====
    print(sep)
    print("TABLE 1: Mixed Workload (SLO = 2000ms TTFT)")
    print(sep)
    print(header)
    print("-" * 130)

    for rate in [2.0, 3.8, 5.7]:
        rate_s = f"{rate:.1f}"
        for strat in strats:
            files = glob.glob(f"results/vllm_full_v1/{strat}__mixed__rate{rate}__r*.json")
            if not files:
                continue
            m = avg_runs(files)
            if m is None:
                continue
            tag = " (OLD)" if strat == "bidkv" else ""
            print(fmt_row(strat + tag, rate_s, m))

        # v5b
        try:
            v5 = load(f"results/vllm_validation_v5b_mixed/bidkv__mixed__rate{rate}__r0.json")
            v5["success"] = f"{v5['success']}/{v5['total']}"
            print(fmt_row("bidkv (v5b)", rate_s, v5))
        except FileNotFoundError:
            pass
        print()

    # ===== TABLE 2: Long-Context =====
    print()
    print(sep)
    print("TABLE 2: Long-Context Workload (SLO = 5000ms TTFT)")
    print(sep)
    print(header)
    print("-" * 130)

    for rate in [0.35, 0.5, 0.7]:
        rate_s = f"{rate:.2f}"
        for strat in strats:
            files = glob.glob(f"results/vllm_full_v1/{strat}__long_context__rate{rate}__r*.json")
            if not files:
                continue
            m = avg_runs(files)
            if m is None:
                continue
            tag = " (OLD)" if strat == "bidkv" else ""
            print(fmt_row(strat + tag, rate_s, m))

        # v5b
        try:
            v5 = load(f"results/vllm_validation_v5b/bidkv__long_context__rate{rate}__r0.json")
            v5["success"] = f"{v5['success']}/{v5['total']}"
            print(fmt_row("bidkv (v5b)", rate_s, v5))
        except FileNotFoundError:
            pass
        print()


if __name__ == "__main__":
    main()
