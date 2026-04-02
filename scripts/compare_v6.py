"""Compare v6 BidKV with baselines at mixed rate=5.7."""

from __future__ import annotations

import glob
import json
import statistics


def compute_metrics(files: list[str], label: str) -> dict | None:
    all_ttft: list[float] = []
    all_tpot: list[float] = []
    total_slo_pass = 0
    tput_per_run: list[float] = []
    per_run_stats: list[dict] = []

    for f in sorted(files):
        data = json.load(open(f))  # noqa: SIM115
        reqs = data.get("request_results", data.get("results", []))
        run_ttft: list[float] = []
        run_tpot: list[float] = []
        for r in reqs:
            ttft = r.get("ttft_ms")
            if ttft is not None and not r.get("error"):
                run_ttft.append(ttft)
                all_ttft.append(ttft)
                if ttft <= 1000:
                    total_slo_pass += 1
                total_lat = r.get("total_latency_ms", 0)
                comp_tok = r.get("completion_tokens", 1)
                if comp_tok > 1 and total_lat > ttft:
                    tpot = (total_lat - ttft) / (comp_tok - 1)
                    run_tpot.append(tpot)
                    all_tpot.append(tpot)

        dur = data.get("end_time", 0) - data.get("start_time", 0)
        succ = len(run_ttft)
        if dur > 0:
            tput_per_run.append(succ / dur)

        if run_ttft:
            s = sorted(run_ttft)
            per_run_stats.append(
                {
                    "p95": s[int(len(s) * 0.95)],
                    "p99": s[int(len(s) * 0.99)],
                }
            )

    if not all_ttft:
        return None

    all_ttft_s = sorted(all_ttft)
    all_tpot_s = sorted(all_tpot) if all_tpot else [0]
    n = len(all_ttft_s)
    nt = len(all_tpot_s)

    return {
        "label": label,
        "ttft_p50": all_ttft_s[n // 2],
        "ttft_p95": all_ttft_s[int(n * 0.95)],
        "ttft_p99": all_ttft_s[int(n * 0.99)],
        "tpot_p95": all_tpot_s[int(nt * 0.95)],
        "tput": statistics.mean(tput_per_run) if tput_per_run else 0,
        "slo_1s": total_slo_pass / len(all_ttft) * 100,
        "n": len(all_ttft),
        "per_run": per_run_stats,
    }


def main() -> None:
    base_dir = "results/vllm_full_v1"
    hdr = f"{'Strategy':<20} {'Tput':>5} {'SLO1s%':>7} {'TTFT p50':>9} {'TTFT p95':>9} {'TTFT p99':>9} {'TPOT p95':>9}"
    print(hdr)
    print("-" * len(hdr))

    # --- BidKV variants ---
    groups = [
        (
            "bidkv(v9)",
            sorted(glob.glob("results/vllm_validation_v9/bidkv__mixed__rate5.7__r*.json")),
        ),
        (
            "bidkv(v8)",
            sorted(glob.glob("results/vllm_validation_v8/bidkv__mixed__rate5.7__r*.json")),
        ),
        (
            "bidkv(v7)",
            sorted(glob.glob("results/vllm_validation_v7/bidkv__mixed__rate5.7__r*.json")),
        ),
        ("bidkv(OLD)", sorted(glob.glob(f"{base_dir}/bidkv__mixed__rate5.7__r*.json"))),
    ]

    for label, files in groups:
        m = compute_metrics(files, label)
        if m:
            print(
                f"{m['label']:<20} {m['tput']:>5.2f} {m['slo_1s']:>6.1f}%"
                f" {m['ttft_p50']:>8.0f} {m['ttft_p95']:>8.0f}"
                f" {m['ttft_p99']:>8.0f} {m['tpot_p95']:>8.1f}"
            )
            for i, r in enumerate(m["per_run"]):
                print(f"  run{i}: p95={r['p95']:.0f}  p99={r['p99']:.0f}")

    print()

    # --- Baselines (fresh runs in v7 dir, then old data) ---
    strategies = [
        "preempt-evict-sjf",
        "h2o-style",
    ]
    print("--- Fresh baselines (same day) ---")
    for strat in strategies:
        files = sorted(glob.glob(f"results/vllm_validation_v7/{strat}__mixed__rate5.7__r*.json"))
        m = compute_metrics(files, f"{strat}(new)")
        if m:
            print(
                f"{m['label']:<20} {m['tput']:>5.2f} {m['slo_1s']:>6.1f}%"
                f" {m['ttft_p50']:>8.0f} {m['ttft_p95']:>8.0f}"
                f" {m['ttft_p99']:>8.0f} {m['tpot_p95']:>8.1f}"
            )
            for i, r in enumerate(m["per_run"]):
                print(f"  run{i}: p95={r['p95']:.0f}  p99={r['p99']:.0f}")

    print("\n--- Old baselines (weeks ago) ---")
    all_strategies = [
        "preempt-evict",
        "preempt-evict-sjf",
        "h2o-style",
        "static-random",
        "uniform",
        "slack-aware",
    ]
    for strat in all_strategies:
        files = sorted(glob.glob(f"{base_dir}/{strat}__mixed__rate5.7__r*.json"))
        m = compute_metrics(files, strat)
        if m:
            print(
                f"{m['label']:<20} {m['tput']:>5.2f} {m['slo_1s']:>6.1f}%"
                f" {m['ttft_p50']:>8.0f} {m['ttft_p95']:>8.0f}"
                f" {m['ttft_p99']:>8.0f} {m['tpot_p95']:>8.1f}"
            )


if __name__ == "__main__":
    main()
