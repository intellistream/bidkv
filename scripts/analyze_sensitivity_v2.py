#!/usr/bin/env python3
"""Sensitivity analysis for BidKV v8 formula parameters.

Reads results from results/vllm_sensitivity_v2/ and reports:
  - Per-variant: TTFT p95, SLO(300ms), Throughput, TPOT p95
  - Per-axis: span (max-min) and robustness classification

Usage:
    conda run -n sagellm python scripts/analyze_sensitivity_v2.py
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results/vllm_sensitivity_v2")

# Axis groupings: each axis has (default_label, [non-default labels])
AXES = {
    "completion_weight": ("default", ["cw_0.25", "cw_1.0", "cw_2.0"]),
    "starvation_weight": ("default", ["sw_0.1", "sw_0.6", "sw_1.0"]),
    "kv_gate":           ("default", ["gate_0.85", "gate_0.90", "gate_0.98"]),
}

# Parameter values for display
VARIANT_PARAMS: dict[str, dict[str, str]] = {
    "default":   {"cw": "0.5",  "sw": "0.3",  "gate": "0.95"},
    "cw_0.25":   {"cw": "0.25", "sw": "0.3",  "gate": "0.95"},
    "cw_1.0":    {"cw": "1.0",  "sw": "0.3",  "gate": "0.95"},
    "cw_2.0":    {"cw": "2.0",  "sw": "0.3",  "gate": "0.95"},
    "sw_0.1":    {"cw": "0.5",  "sw": "0.1",  "gate": "0.95"},
    "sw_0.6":    {"cw": "0.5",  "sw": "0.6",  "gate": "0.95"},
    "sw_1.0":    {"cw": "0.5",  "sw": "1.0",  "gate": "0.95"},
    "gate_0.85": {"cw": "0.5",  "sw": "0.3",  "gate": "0.85"},
    "gate_0.90": {"cw": "0.5",  "sw": "0.3",  "gate": "0.90"},
    "gate_0.98": {"cw": "0.5",  "sw": "0.3",  "gate": "0.98"},
}


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def load_variant(label: str) -> dict[str, float] | None:
    """Load all runs for a variant and return averaged metrics."""
    variant_dir = RESULTS_DIR / label
    if not variant_dir.exists():
        return None

    ttft_all: list[float] = []
    tpot_all: list[float] = []
    throughputs: list[float] = []

    json_files = sorted(variant_dir.glob("bidkv__mixed__rate3.8__r*.json"))
    if not json_files:
        return None

    for fp in json_files:
        data = json.loads(fp.read_text())
        ok = [r for r in data["request_results"] if not r.get("error")]

        ttft_run = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
        ttft_all.extend(ttft_run)

        for r in ok:
            ct = r.get("completion_tokens", 0)
            ttft = r.get("ttft_ms")
            lat = r.get("total_latency_ms")
            if ct and ct > 1 and ttft is not None and lat is not None:
                tpot_all.append((lat - ttft) / (ct - 1))

        throughputs.append(data["summary"]["throughput_rps"])

    if not ttft_all:
        return None

    slo_count = sum(1 for t in ttft_all if t <= 300.0)
    return {
        "ttft_p50":   percentile(ttft_all, 50),
        "ttft_p95":   percentile(ttft_all, 95),
        "ttft_p99":   percentile(ttft_all, 99),
        "tpot_p95":   percentile(tpot_all, 95),
        "slo_pct":    slo_count / len(ttft_all) * 100,
        "throughput": statistics.mean(throughputs),
        "n_runs":     len(json_files),
    }


def classify_robustness(slo_span: float, ttft_span_pct: float) -> str:
    if slo_span < 5.0 and ttft_span_pct < 20.0:
        return "ROBUST"
    if slo_span < 10.0 and ttft_span_pct < 40.0:
        return "MODERATE"
    return "SENSITIVE"


def main() -> None:  # noqa: C901
    print("=" * 70)
    print("BidKV v8 Sensitivity Analysis — rate=3.8 req/s, workload=mixed")
    print("=" * 70)

    # Load all variants
    metrics: dict[str, dict[str, float]] = {}
    for label in VARIANT_PARAMS:
        m = load_variant(label)
        if m is None:
            print(f"  [{label}] NOT YET AVAILABLE (skipping)")
        else:
            metrics[label] = m
            print(f"  [{label}] loaded ({m['n_runs']:.0f} runs)")

    print()

    if not metrics:
        print("No results available yet. Re-run after experiments complete.")
        return

    # ── Full table ─────────────────────────────────────────────────────────
    COL = 14
    print(f"{'Variant':<14} {'CW':>4} {'SW':>4} {'Gate':>5} | "
          f"{'Thru':>6} {'SLO%':>6} {'TTFT95':>7} {'TPOT95':>7} | runs")
    print("-" * 70)
    for label, params in VARIANT_PARAMS.items():
        if label not in metrics:
            print(f"{label:<14} {params['cw']:>4} {params['sw']:>4} {params['gate']:>5} | "
                  f"  (not ready)")
            continue
        m = metrics[label]
        marker = " *" if label == "default" else "  "
        print(
            f"{label:<14}{marker}{params['cw']:>4} {params['sw']:>4} {params['gate']:>5} | "
            f"{m['throughput']:>6.2f} {m['slo_pct']:>6.1f} {m['ttft_p95']:>7.0f} "
            f"{m['tpot_p95']:>7.1f} | {m['n_runs']:.0f}"
        )

    print()
    print("* = default configuration")
    print()

    # ── Robustness per axis ─────────────────────────────────────────────────
    if "default" not in metrics:
        print("Cannot compute robustness: default variant not available.")
        return

    def_m = metrics["default"]

    print("Robustness Analysis by Axis")
    print("-" * 70)
    print(f"  Default: SLO={def_m['slo_pct']:.1f}%  TTFT_P95={def_m['ttft_p95']:.0f}ms")
    print()

    for axis_name, (def_label, test_labels) in AXES.items():
        available = [lbl for lbl in test_labels if lbl in metrics]
        if not available:
            print(f"  [{axis_name}] No data yet")
            continue

        slo_vals = [def_m["slo_pct"]] + [metrics[l]["slo_pct"] for l in available]
        ttft_vals = [def_m["ttft_p95"]] + [metrics[l]["ttft_p95"] for l in available]

        slo_span = max(slo_vals) - min(slo_vals)
        ttft_span_pct = (max(ttft_vals) - min(ttft_vals)) / def_m["ttft_p95"] * 100

        robustness = classify_robustness(slo_span, ttft_span_pct)

        print(f"  [{axis_name}]  n_variants={len(available)+1}")
        slo_str = "  ".join(f"{metrics[l]['slo_pct']:.1f}" for l in available)
        ttft_str = "  ".join(f"{metrics[l]['ttft_p95']:.0f}" for l in available)
        print(f"    SLO(%):    default={def_m['slo_pct']:.1f}  variants=[{slo_str}]")
        print(f"    TTFT_P95:  default={def_m['ttft_p95']:.0f}  variants=[{ttft_str}]")
        print(f"    SLO span: {slo_span:.1f}pp   TTFT span: {ttft_span_pct:.1f}%")
        print(f"    → {robustness}")
        print()

    # ── TTFT P95 validation ─────────────────────────────────────────────────
    print("Default TTFT P95 sanity check:")
    in_range = 550 <= def_m["ttft_p95"] <= 750
    print(f"  TTFT P95 = {def_m['ttft_p95']:.0f}ms  (expected 550–750ms)  → {'PASS' if in_range else 'FAIL'}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("Summary for paper §7.1 / Appendix:")
    all_slo = [m["slo_pct"] for m in metrics.values()]
    all_ttft = [m["ttft_p95"] for m in metrics.values()]
    overall_slo_span = max(all_slo) - min(all_slo)
    overall_ttft_span_pct = (max(all_ttft) - min(all_ttft)) / def_m["ttft_p95"] * 100
    overall = classify_robustness(overall_slo_span, overall_ttft_span_pct)
    print(f"  Overall SLO span: {overall_slo_span:.1f}pp")
    print(f"  Overall TTFT P95 span: {overall_ttft_span_pct:.1f}%")
    print(f"  Overall classification: {overall}")


if __name__ == "__main__":
    main()
