"""Show full data tables with SLO attainment for all strategies."""

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

    # Per-request data for SLO calculation
    reqs = d.get("request_results", d.get("requests", []))
    ttft_list = []
    for r in reqs:
        ttft = r.get("ttft_ms", 0)
        ok = r.get("success", True)
        if ok and ttft > 0:
            ttft_list.append(ttft)

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
    # SLO: average across runs
    slo_vals = [slo_pct(r["ttft_list"], slo_thresh) for r in runs]
    avg["slo"] = statistics.mean(slo_vals) if slo_vals else 0.0
    return avg


def single_run(path: str, slo_thresh: float) -> dict | None:
    try:
        m = load(path)
    except FileNotFoundError:
        return None
    m["success"] = f"{m['success']}/{m['total']}"
    m["slo"] = slo_pct(m["ttft_list"], slo_thresh)
    m["n"] = 1
    return m


def print_table(
    title: str, workload: str, rates: list, slo_thresh: float, strats: list[str], rate_fmt: str
) -> None:
    header = (
        f"{'Strategy':<22} {'Rate':>5} {'Succ%':>6} {'SLO%':>6} "
        f"{'Tput':>6} {'TTFT p50':>9} {'TTFT p99':>9} "
        f"{'TPOT p50':>8} {'TPOT p99':>8} "
        f"{'E2E p50':>9} {'E2E p99':>10} "
        f"{'NormLat':>8}"
    )
    sep = "=" * 140

    print(sep)
    print(f"{title}  (SLO threshold = {slo_thresh:.0f}ms TTFT)")
    print(sep)
    print(header)
    print("-" * 140)

    for rate in rates:
        rate_s = f"{rate:{rate_fmt}}"
        for strat in strats:
            files = glob.glob(f"results/vllm_full_v1/{strat}__{workload}__rate{rate}__r*.json")
            if not files:
                continue
            m = avg_runs(files, slo_thresh)
            if m is None:
                continue
            tag = " (OLD)" if strat == "bidkv" else ""
            name = strat + tag
            print(
                f"{name:<22} {rate_s:>5} {m['success_pct']:>5.1f}% {m['slo']:>5.1f}% "
                f"{m['tput']:>6.2f} {m['ttft50']:>9.0f} {m['ttft99']:>9.0f} "
                f"{m['tpot50']:>8.1f} {m['tpot99']:>8.1f} "
                f"{m['e2e50']:>9.0f} {m['e2e99']:>10.0f} "
                f"{m['normlat']:>8.1f}"
            )

        # v5b bidkv
        v5b_sfx = "_mixed" if workload == "mixed" else ""
        v5b_path = f"results/vllm_validation_v5b{v5b_sfx}/bidkv__{workload}__rate{rate}__r0.json"
        v5 = single_run(v5b_path, slo_thresh)
        if v5:
            print(
                f"{'bidkv (v5b)':<22} {rate_s:>5} {v5['success_pct']:>5.1f}% {v5['slo']:>5.1f}% "
                f"{v5['tput']:>6.2f} {v5['ttft50']:>9.0f} {v5['ttft99']:>9.0f} "
                f"{v5['tpot50']:>8.1f} {v5['tpot99']:>8.1f} "
                f"{v5['e2e50']:>9.0f} {v5['e2e99']:>10.0f} "
                f"{v5['normlat']:>8.1f}"
            )

        # old bidkv backup (for comparison at rate=0.7)
        old_path = f"results/vllm_full_v1/backup_old_bidkv/bidkv__{workload}__rate{rate}__r*.json"
        old_files = glob.glob(old_path)
        if old_files:
            m = avg_runs(old_files, slo_thresh)
            if m:
                bk = "bidkv (backup-OLD)"
                print(
                    f"{bk:<22} {rate_s:>5} {m['success_pct']:>5.1f}% {m['slo']:>5.1f}% "
                    f"{m['tput']:>6.2f} {m['ttft50']:>9.0f} {m['ttft99']:>9.0f} "
                    f"{m['tpot50']:>8.1f} {m['tpot99']:>8.1f} "
                    f"{m['e2e50']:>9.0f} {m['e2e99']:>10.0f} "
                    f"{m['normlat']:>8.1f}"
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

    print_table(
        "TABLE 1: Mixed Workload",
        "mixed",
        [2.0, 3.8, 5.7],
        2000.0,
        strats,
        ".1f",
    )

    print()

    print_table(
        "TABLE 2: Long-Context Workload",
        "long_context",
        [0.35, 0.5, 0.7],
        5000.0,
        strats,
        ".2f",
    )

    # ===== Summary: Best strategy per scenario =====
    print()
    print("=" * 100)
    print("SUMMARY: Best strategy per scenario (by SLO attainment, then throughput)")
    print("=" * 100)
    for wl, rates, slo_t, rfmt in [
        ("mixed", [2.0, 3.8, 5.7], 2000.0, ".1f"),
        ("long_context", [0.35, 0.5, 0.7], 5000.0, ".2f"),
    ]:
        for rate in rates:
            best_name = ""
            best_slo = -1.0
            best_tput = -1.0
            for strat in strats:
                files = glob.glob(f"results/vllm_full_v1/{strat}__{wl}__rate{rate}__r*.json")
                if not files:
                    continue
                m = avg_runs(files, slo_t)
                if m and (m["slo"] > best_slo or (m["slo"] == best_slo and m["tput"] > best_tput)):
                    best_slo = m["slo"]
                    best_tput = m["tput"]
                    best_name = strat

            # Check v5b
            v5b_sfx = "_mixed" if wl == "mixed" else ""
            v5b_path = f"results/vllm_validation_v5b{v5b_sfx}/bidkv__{wl}__rate{rate}__r0.json"
            v5 = single_run(v5b_path, slo_t)
            v5_note = ""
            if v5:
                v5_note = f"  |  bidkv(v5b): SLO={v5['slo']:.1f}% tput={v5['tput']:.2f}"

            print(
                f"  {wl:15s} rate={rate:{rfmt}}  BEST={best_name:<22s} "
                f"SLO={best_slo:.1f}%  tput={best_tput:.2f}{v5_note}"
            )


if __name__ == "__main__":
    main()
