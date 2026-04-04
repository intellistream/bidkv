"""Compare v5 BidKV results with baselines."""

from __future__ import annotations

import glob
import json


def extract_metrics(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)

    # Handle both flat and nested formats
    if "total_requests" in d:
        s = d
    elif "summary" in d:
        s = d["summary"]
    else:
        s = d

    total = s.get("total_requests", 0)
    success = s.get("successful_requests", s.get("success_count", 0))
    tput = s.get("throughput_rps", 0)
    ttft_p50 = s.get("ttft_ms_p50", s.get("ttft_p50_ms", 0))
    ttft_p99 = s.get("ttft_ms_p99", s.get("ttft_p99_ms", 0))
    e2e_p50 = s.get("e2e_latency_ms_p50", s.get("e2e_p50_ms", 0))
    e2e_p99 = s.get("e2e_latency_ms_p99", s.get("e2e_p99_ms", 0))
    norm_lat = s.get("normalized_latency_ms_per_token", 0)

    am = d.get("adapter_metrics", {})
    evictions = am.get("compressions_executed", am.get("eviction_count", "N/A"))

    return {
        "success": f"{success}/{total}",
        "success_pct": success / total * 100 if total else 0,
        "tput": tput,
        "ttft_p50": ttft_p50,
        "ttft_p99": ttft_p99,
        "e2e_p50": e2e_p50,
        "e2e_p99": e2e_p99,
        "norm_lat": norm_lat,
        "evictions": evictions,
    }


def main():
    # v5
    m = extract_metrics("results/vllm_validation_v5/bidkv__long_context__rate0.7__r0.json")
    print("=== v5 BidKV (long_context rate=0.7) ===")
    print(f"  success={m['success']} ({m['success_pct']:.1f}%)")
    print(f"  throughput={m['tput']:.3f}")
    print(f"  TTFT p50={m['ttft_p50']:.0f} p99={m['ttft_p99']:.0f}")
    print(f"  E2E p50={m['e2e_p50']:.0f} p99={m['e2e_p99']:.0f}")
    print(f"  NormLat={m['norm_lat']:.1f} ms/tok")
    print(f"  evictions={m['evictions']}")

    # pe-sjf
    print("\n=== pe-sjf (long_context rate=0.7, 3 runs) ===")
    for fn in sorted(
        glob.glob("results/vllm_full_v1/preempt-evict-sjf__long_context__rate0.7__r*.json")
    ):
        m2 = extract_metrics(fn)
        name = fn.split("/")[-1].replace(".json", "")
        print(
            f"  {name}: succ={m2['success']} tput={m2['tput']:.3f} "
            f"TTFT {m2['ttft_p50']:.0f}/{m2['ttft_p99']:.0f} "
            f"E2E p99={m2['e2e_p99']:.0f} evict={m2['evictions']}"
        )

    # old bidkv
    print("\n=== OLD BidKV (long_context rate=0.7, 3 runs) ===")
    for fn in sorted(
        glob.glob("results/vllm_full_v1/backup_old_bidkv/bidkv__long_context__rate0.7__r*.json")
    ):
        m2 = extract_metrics(fn)
        name = fn.split("/")[-1].replace(".json", "")
        print(
            f"  {name}: succ={m2['success']} tput={m2['tput']:.3f} "
            f"TTFT {m2['ttft_p50']:.0f}/{m2['ttft_p99']:.0f} "
            f"E2E p99={m2['e2e_p99']:.0f} evict={m2['evictions']}"
        )

    # v4
    print("\n=== v4 BidKV (long_context rate=0.7) ===")
    m2 = extract_metrics("results/vllm_validation_v4/bidkv__long_context__rate0.7__r0.json")
    print(
        f"  succ={m2['success']} tput={m2['tput']:.3f} "
        f"TTFT {m2['ttft_p50']:.0f}/{m2['ttft_p99']:.0f} "
        f"E2E p99={m2['e2e_p99']:.0f} evict={m2['evictions']}"
    )

    # All strategies at rate=0.7
    print("\n=== ALL strategies (long_context rate=0.7, avg) ===")
    strats: dict[str, list] = {}
    for fn in sorted(glob.glob("results/vllm_full_v1/*__long_context__rate0.7__r*.json")):
        name_parts = fn.split("/")[-1].replace(".json", "").rsplit("__r", 1)
        strat = name_parts[0].replace("__long_context__rate0.7", "")
        m2 = extract_metrics(fn)
        if strat not in strats:
            strats[strat] = []
        strats[strat].append(m2)

    for strat, runs in sorted(strats.items()):
        n = len(runs)
        avg_tput = sum(r["tput"] for r in runs) / n
        avg_succ = sum(r["success_pct"] for r in runs) / n
        avg_ttft50 = sum(r["ttft_p50"] for r in runs) / n
        avg_ttftp99 = sum(r["ttft_p99"] for r in runs) / n
        avg_e2ep99 = sum(r["e2e_p99"] for r in runs) / n
        print(
            f"  {strat:22s}: succ={avg_succ:5.1f}% tput={avg_tput:.3f} "
            f"TTFT {avg_ttft50:.0f}/{avg_ttftp99:.0f} E2E p99={avg_e2ep99:.0f}"
        )


if __name__ == "__main__":
    main()
