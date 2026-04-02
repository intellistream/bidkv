"""Attribution analysis: SJF contribution vs BidKV-unique contribution."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

DATA_DIR = "results/vllm_v8_full_validation"


def main() -> None:
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    data: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for f in sorted(files):
        with open(f) as fh:
            d = json.load(fh)
        if "strategy" not in d:
            continue
        strat = d["strategy"]
        rate = d["request_rate"]
        reqs = [r for r in d["request_results"] if not r.get("error")]
        if not reqs:
            continue
        ttfts = [r["ttft_ms"] for r in reqs if r.get("ttft_ms")]
        tpots = []
        for r in reqs:
            ct = r.get("completion_tokens", 0)
            lat = r.get("total_latency_ms")
            ttft = r.get("ttft_ms")
            if ct > 1 and lat and ttft:
                tpots.append((lat - ttft) / max(ct - 1, 1))
        normlats = []
        for r in reqs:
            ct = r.get("completion_tokens", 0)
            lat = r.get("total_latency_ms")
            if ct > 0 and lat:
                normlats.append(lat / ct)

        duration = d.get("duration_s", 1)
        slo300 = sum(1 for t in ttfts if t <= 300) / len(ttfts) * 100 if ttfts else 0
        goodput500 = sum(1 for t in ttfts if t <= 500) / duration if ttfts else 0

        data[strat][rate].append(
            {
                "goodput": goodput500,
                "slo300": slo300,
                "ttft_p95": sorted(ttfts)[int(len(ttfts) * 0.95)] if ttfts else 0,
                "tpot_p95": sorted(tpots)[int(len(tpots) * 0.95)] if tpots else 0,
                "normlat": sum(normlats) / len(normlats) if normlats else 0,
            }
        )

    rates = [2.0, 3.8, 5.7]
    metrics = ["goodput", "slo300", "ttft_p95", "tpot_p95", "normlat"]
    labels = {
        "goodput": "Goodput(500ms)",
        "slo300": "SLO(300ms)",
        "ttft_p95": "TTFT p95",
        "tpot_p95": "TPOT p95",
        "normlat": "NormLat",
    }
    higher_better = {
        "goodput": True,
        "slo300": True,
        "ttft_p95": False,
        "tpot_p95": False,
        "normlat": False,
    }

    def cross_avg(strat: str, metric: str) -> float:
        vals = []
        for rate in rates:
            for r in data[strat][rate]:
                vals.append(r[metric])
        return sum(vals) / len(vals) if vals else 0

    def rate_avg(strat: str, rate: float, metric: str) -> float:
        runs = data[strat][rate]
        return sum(r[metric] for r in runs) / len(runs) if runs else 0

    # ===== Section 1: Per-Rate PE -> PE-SJF -> BidKV =====
    print("=" * 130)
    print("ATTRIBUTION ANALYSIS: SJF contribution vs BidKV-unique contribution")
    print("=" * 130)

    print("\n[1] Per-Rate Breakdown: PE -> PE-SJF -> BidKV")
    for rate in rates:
        print(f"\n  Rate = {rate} req/s:")
        hdr = (
            f"  {'Strategy':<22} {'Goodput':>8} {'SLO300':>8}"
            f" {'TTFT95':>8} {'TPOT95':>8} {'NormLat':>8}"
        )
        print(hdr)
        for s in ["preempt-evict", "preempt-evict-sjf", "bidkv"]:
            vals = [rate_avg(s, rate, m) for m in metrics]
            print(
                f"  {s:<22} {vals[0]:>8.2f} {vals[1]:>7.1f}% "
                f"{vals[2]:>8.0f} {vals[3]:>8.1f} {vals[4]:>8.1f}"
            )

    # ===== Section 2: Cross-Rate Attribution =====
    print("\n" + "=" * 130)
    print("[2] Cross-Rate Attribution Decomposition")
    print("=" * 130)

    pe = {m: cross_avg("preempt-evict", m) for m in metrics}
    pesj = {m: cross_avg("preempt-evict-sjf", m) for m in metrics}
    bkv = {m: cross_avg("bidkv", m) for m in metrics}

    hdr = (
        f"\n  {'Metric':<16} {'PE':>10} {'PE-SJF':>10} {'BidKV':>10}"
        f" | {'SJF delta':>12} {'SJF%':>6} {'BKV delta':>12} {'BKV%':>6}"
    )
    print(hdr)
    for m in metrics:
        total = bkv[m] - pe[m]
        sjf_d = pesj[m] - pe[m]
        bkv_d = bkv[m] - pesj[m]
        pct_sjf = sjf_d / abs(total) * 100 if total != 0 else 0
        pct_bkv = bkv_d / abs(total) * 100 if total != 0 else 0
        fmt = (
            ".2f" if m == "goodput" else ".1f" if m in ("slo300", "tpot_p95", "normlat") else ".0f"
        )
        pv = format(pe[m], fmt)
        psv = format(pesj[m], fmt)
        bv = format(bkv[m], fmt)
        sd = format(sjf_d, "+" + fmt)
        bd = format(bkv_d, "+" + fmt)
        u = "%" if m == "slo300" else ""
        print(
            f"  {labels[m]:<16} {pv:>10}{u} {psv:>10}{u} {bv:>10}{u}"
            f" | {sd:>12} {pct_sjf:>5.0f}% {bd:>12} {pct_bkv:>5.0f}%"
        )

    # ===== Section 3: Same-Infrastructure Comparison (isolates victim selection) =====
    print("\n" + "=" * 130)
    print("[3] Same-Infrastructure Strategies (SJF + proactive preempt + running reorder)")
    print("    Isolates victim selection mechanism only")
    print("=" * 130)

    same_infra = ["h2o-style", "static-random", "uniform", "bidkv"]
    notes = {
        "h2o-style": "(attn heuristic, +SRPT)",
        "static-random": "(random, +SRPT)",
        "uniform": "(uniform, +SRPT)",
        "bidkv": "(U-score, no SRPT)",
    }
    hdr = (
        f"\n  {'Strategy':<20} {'Goodput':>10} {'SLO(300)':>10}"
        f" {'TTFT p95':>10} {'TPOT p95':>10} {'NormLat':>10}  Note"
    )
    print(hdr)
    for s in same_infra:
        vals = [cross_avg(s, m) for m in metrics]
        print(
            f"  {s:<20} {vals[0]:>10.2f} {vals[1]:>9.1f}% "
            f"{vals[2]:>10.0f} {vals[3]:>10.1f} {vals[4]:>10.1f}  {notes[s]}"
        )

    # BidKV vs H2O delta
    print("\n  BidKV vs H2O (isolates U-score vs attention heuristic):")
    h2o = {m: cross_avg("h2o-style", m) for m in metrics}
    for m in metrics:
        d = bkv[m] - h2o[m]
        pct = d / abs(h2o[m]) * 100 if h2o[m] != 0 else 0
        better = (d > 0) == higher_better[m]
        mark = "WIN" if better else "LOSS"
        print(f"    {labels[m]:<16}: {d:+.1f} ({pct:+.1f}%) [{mark}]")

    # ===== Section 4: Head-to-Head BidKV vs PE-SJF per rate =====
    print("\n" + "=" * 130)
    print("[4] BidKV vs PE-SJF Head-to-Head per rate")
    print("=" * 130)

    total_wins = 0
    total_comparisons = 0
    for rate in rates:
        bkv_runs = data["bidkv"][rate]
        pesj_runs = data["preempt-evict-sjf"][rate]
        if not bkv_runs or not pesj_runs:
            continue
        bkv_a = {m: sum(r[m] for r in bkv_runs) / len(bkv_runs) for m in metrics}
        pesj_a = {m: sum(r[m] for r in pesj_runs) / len(pesj_runs) for m in metrics}
        wins = 0
        details = []
        for m in metrics:
            d = bkv_a[m] - pesj_a[m]
            win = (d > 0) == higher_better[m]
            if win:
                wins += 1
            total_wins += int(win)
            total_comparisons += 1
            pct = d / abs(pesj_a[m]) * 100 if pesj_a[m] != 0 else 0
            details.append(f"{labels[m]}={'W' if win else 'L'}({pct:+.1f}%)")
        print(f"  Rate {rate}: {wins}/5 wins  [{', '.join(details)}]")

    pct = total_wins / total_comparisons * 100
    print(f"\n  Overall: {total_wins}/{total_comparisons} wins ({pct:.0f}%)")

    # ===== Section 5: High Load Focus =====
    print("\n" + "=" * 130)
    print("[5] HIGH LOAD FOCUS (rate=5.7) - KV pressure most intense")
    print("=" * 130)
    r = 5.7
    hdr = (
        f"\n  {'Strategy':<22} {'Goodput':>8} {'SLO300':>8}"
        f" {'TTFT95':>8} {'TPOT95':>8} {'NormLat':>8}"
    )
    print(hdr)
    all_strats = [
        "preempt-evict",
        "preempt-evict-sjf",
        "h2o-style",
        "static-random",
        "uniform",
        "slack-aware",
        "bidkv",
    ]
    for s in all_strats:
        vals = [rate_avg(s, r, m) for m in metrics]
        print(
            f"  {s:<22} {vals[0]:>8.2f} {vals[1]:>7.1f}% "
            f"{vals[2]:>8.0f} {vals[3]:>8.1f} {vals[4]:>8.1f}"
        )

    # PE-SJF vs BidKV delta at 5.7
    bkv57 = {m: rate_avg("bidkv", 5.7, m) for m in metrics}
    pesj57 = {m: rate_avg("preempt-evict-sjf", 5.7, m) for m in metrics}
    print("\n  BidKV vs PE-SJF at rate=5.7:")
    for m in metrics:
        d = bkv57[m] - pesj57[m]
        pct = d / abs(pesj57[m]) * 100 if pesj57[m] != 0 else 0
        better = (d > 0) == higher_better[m]
        mark = "WIN" if better else "LOSS"
        print(f"    {labels[m]:<16}: {d:+.1f} ({pct:+.1f}%) [{mark}]")


if __name__ == "__main__":
    main()
