"""Sensitivity analysis for BidKV δ parameters, KV gate, and weight robustness.

Reads experiment results from results/vllm_sensitivity/ and produces:
1. Summary table (console + JSON)
2. Axis-specific analysis (ablation, gate, weights)
3. Relative change from default baseline

Usage:
    cd /home/bidkv
    python3 scripts/analyze_sensitivity.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading (reuses copilot-instructions.md template)
# ---------------------------------------------------------------------------

def load_run(filepath: str) -> dict:
    """Load a single experiment result file, return standardized metrics."""
    d = json.load(open(filepath))

    ok = [r for r in d["request_results"] if not r.get("error")]

    ttft_list = sorted(r["ttft_ms"] for r in ok if r["ttft_ms"] is not None)

    tpot_list = []
    for r in ok:
        ct = r.get("completion_tokens", 0)
        ttft = r.get("ttft_ms")
        total = r.get("total_latency_ms")
        if ct > 1 and ttft is not None and total is not None:
            tpot_list.append((total - ttft) / (ct - 1))
    tpot_list.sort()

    def pct(data: list[float], p: int) -> float:
        if not data:
            return float("nan")
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    slo_threshold = 2000.0 if "long" in d.get("workload", "") else 300.0
    slo_count = sum(1 for t in ttft_list if t <= slo_threshold)
    slo_pct = slo_count / len(ttft_list) * 100 if ttft_list else 0

    am = d.get("adapter_metrics", {})
    evictions = am.get("total_evictions", am.get("total_compressions", 0))
    freed = am.get("total_tokens_freed", 0)

    return {
        "throughput": d["summary"]["throughput_rps"],
        "slo_pct": slo_pct,
        "ttft_p50": pct(ttft_list, 50),
        "ttft_p95": pct(ttft_list, 95),
        "ttft_p99": pct(ttft_list, 99),
        "tpot_p50": pct(tpot_list, 50),
        "tpot_p95": pct(tpot_list, 95),
        "tpot_p99": pct(tpot_list, 99),
        "ok_count": len(ok),
        "total_count": len(d["request_results"]),
        "evictions": evictions,
        "tokens_freed": freed,
    }


def load_variant(result_dir: str) -> list[dict]:
    """Load all runs from a variant directory."""
    runs = []
    for fn in sorted(os.listdir(result_dir)):
        if fn.endswith(".json") and not fn.startswith("candidate"):
            runs.append(load_run(os.path.join(result_dir, fn)))
    return runs


def avg_metrics(runs: list[dict]) -> dict:
    """Average metrics across runs."""
    if not runs:
        return {}
    keys = ["throughput", "slo_pct", "ttft_p50", "ttft_p95", "ttft_p99",
            "tpot_p50", "tpot_p95", "tpot_p99", "evictions", "tokens_freed",
            "ok_count"]
    result = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r]
        result[k] = statistics.mean(vals) if vals else float("nan")
        if len(vals) >= 2:
            result[f"{k}_std"] = statistics.stdev(vals)
        else:
            result[f"{k}_std"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANT_META = {
    # Axis 1: Ablation
    "default":        {"axis": "ablation", "label": "Default (Full δ)", "params": "w_c=2.0, w_s=0.5, div=256, gate=0.95"},
    "no-completion":  {"axis": "ablation", "label": "No Completion", "params": "w_c=0"},
    "no-starvation":  {"axis": "ablation", "label": "No Starvation", "params": "w_s=0"},
    "no-recompute":   {"axis": "ablation", "label": "No Recompute Norm", "params": "recompute=1.0"},
    "freed-only":     {"axis": "ablation", "label": "Freed-Only (δ=1)", "params": "δ=const"},
    # Axis 2: KV Gate
    "gate-85":        {"axis": "gate", "label": "Gate 85%", "params": "gate=0.85"},
    "gate-90":        {"axis": "gate", "label": "Gate 90%", "params": "gate=0.90"},
    "gate-98":        {"axis": "gate", "label": "Gate 98%", "params": "gate=0.98"},
    # Axis 3: Weights
    "wc-05":          {"axis": "weight", "label": "w_c=0.5", "params": "w_c=0.5"},
    "wc-10":          {"axis": "weight", "label": "w_c=1.0", "params": "w_c=1.0"},
    "wc-40":          {"axis": "weight", "label": "w_c=4.0", "params": "w_c=4.0"},
    "ws-01":          {"axis": "weight", "label": "w_s=0.1", "params": "w_s=0.1"},
    "ws-025":         {"axis": "weight", "label": "w_s=0.25", "params": "w_s=0.25"},
    "ws-10":          {"axis": "weight", "label": "w_s=1.0", "params": "w_s=1.0"},
}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def print_table(title: str, variants: list[str], all_data: dict[str, dict],
                default_data: dict | None = None) -> None:
    """Print a formatted table for a group of variants."""
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

    header = f"  {'Variant':<25} {'Thru':>7} {'SLO%':>7} {'TTFT p95':>10} {'TPOT p95':>10} {'Evict':>7} {'Freed':>10}"
    if default_data:
        header += f"  {'ΔThru%':>7} {'ΔSLO':>7} {'ΔTTFT%':>8} {'ΔTPOT%':>8}"
    print(header)
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*10}", end="")
    if default_data:
        print(f"  {'-'*7} {'-'*7} {'-'*8} {'-'*8}", end="")
    print()

    for v in variants:
        d = all_data.get(v)
        if d is None:
            print(f"  {v:<25} {'(no data)':>7}")
            continue

        meta = VARIANT_META.get(v, {})
        label = meta.get("label", v)

        row = (f"  {label:<25} {d['throughput']:>7.2f} {d['slo_pct']:>7.1f} "
               f"{d['ttft_p95']:>10.0f} {d['tpot_p95']:>10.1f} "
               f"{d['evictions']:>7.0f} {d['tokens_freed']:>10.0f}")

        if default_data:
            dt = (d["throughput"] - default_data["throughput"]) / default_data["throughput"] * 100
            ds = d["slo_pct"] - default_data["slo_pct"]
            dttft = (d["ttft_p95"] - default_data["ttft_p95"]) / max(1, default_data["ttft_p95"]) * 100
            dtpot = (d["tpot_p95"] - default_data["tpot_p95"]) / max(1, default_data["tpot_p95"]) * 100
            row += f"  {dt:>+7.1f} {ds:>+7.1f} {dttft:>+8.1f} {dtpot:>+8.1f}"

        print(row)


def main() -> None:
    base_dir = Path("/home/bidkv/results/vllm_sensitivity")

    if not base_dir.exists():
        print(f"ERROR: Results directory not found: {base_dir}")
        sys.exit(1)

    # Load all variants
    all_data: dict[str, dict] = {}
    all_runs: dict[str, list[dict]] = {}

    for variant_name in VARIANT_META:
        variant_dir = base_dir / variant_name
        if variant_dir.exists():
            runs = load_variant(str(variant_dir))
            if runs:
                all_runs[variant_name] = runs
                all_data[variant_name] = avg_metrics(runs)

    if not all_data:
        print("ERROR: No sensitivity data found. Run scripts/run_sensitivity.sh first.")
        sys.exit(1)

    print(f"\nLoaded {len(all_data)} variants with data")
    for v, d in sorted(all_data.items()):
        n = len(all_runs.get(v, []))
        print(f"  {v}: {n} runs, thru={d['throughput']:.2f}, SLO={d['slo_pct']:.1f}%")

    default_data = all_data.get("default")

    # === Axis 1: Ablation ===
    ablation_variants = ["default", "no-completion", "no-starvation", "no-recompute", "freed-only"]
    print_table("Axis 1: δ Component Ablation", ablation_variants, all_data, default_data)

    # === Axis 2: KV Gate Threshold ===
    gate_variants = ["gate-85", "gate-90", "default", "gate-98"]
    print_table("Axis 2: KV Gate Threshold", gate_variants, all_data, default_data)

    # === Axis 3: Weight Robustness ===
    wc_variants = ["wc-05", "wc-10", "default", "wc-40"]
    print_table("Axis 3a: Completion Weight (w_c)", wc_variants, all_data, default_data)

    ws_variants = ["ws-01", "ws-025", "default", "ws-10"]
    print_table("Axis 3b: Starvation Weight (w_s)", ws_variants, all_data, default_data)

    # === Cross-Axis Summary ===
    print(f"\n{'='*100}")
    print("  Cross-Axis Summary: Ranking by SLO Attainment")
    print(f"{'='*100}")

    ranked = sorted(all_data.items(), key=lambda x: -x[1]["slo_pct"])
    for rank, (v, d) in enumerate(ranked, 1):
        meta = VARIANT_META.get(v, {})
        label = meta.get("label", v)
        axis = meta.get("axis", "?")
        marker = " ★" if v == "default" else ""
        print(f"  #{rank:<2} {label:<25} [{axis:<8}] SLO={d['slo_pct']:>5.1f}%  "
              f"TTFT95={d['ttft_p95']:>6.0f}ms  Thru={d['throughput']:.2f}{marker}")

    # === Key Findings ===
    print(f"\n{'='*100}")
    print("  Key Findings")
    print(f"{'='*100}")

    if default_data:
        # Find worst ablation
        ablation_data = {v: all_data[v] for v in ablation_variants if v in all_data and v != "default"}
        if ablation_data:
            worst_abl = min(ablation_data.items(), key=lambda x: x[1]["slo_pct"])
            best_abl = max(ablation_data.items(), key=lambda x: x[1]["slo_pct"])
            print(f"\n  Ablation:")
            print(f"    Default SLO: {default_data['slo_pct']:.1f}%")
            print(f"    Worst ablation: {VARIANT_META[worst_abl[0]]['label']} → SLO={worst_abl[1]['slo_pct']:.1f}% "
                  f"(Δ={worst_abl[1]['slo_pct']-default_data['slo_pct']:+.1f}pp)")
            print(f"    Best ablation:  {VARIANT_META[best_abl[0]]['label']} → SLO={best_abl[1]['slo_pct']:.1f}% "
                  f"(Δ={best_abl[1]['slo_pct']-default_data['slo_pct']:+.1f}pp)")

        # Gate sensitivity range
        gate_data = {v: all_data[v] for v in ["gate-85", "gate-90", "default", "gate-98"] if v in all_data}
        if len(gate_data) > 1:
            slo_range = [d["slo_pct"] for d in gate_data.values()]
            print(f"\n  KV Gate Threshold:")
            print(f"    SLO range: {min(slo_range):.1f}% – {max(slo_range):.1f}% "
                  f"(spread={max(slo_range)-min(slo_range):.1f}pp)")

        # Weight robustness
        wc_data = {v: all_data[v] for v in ["wc-05", "wc-10", "default", "wc-40"] if v in all_data}
        ws_data = {v: all_data[v] for v in ["ws-01", "ws-025", "default", "ws-10"] if v in all_data}
        if wc_data:
            wc_slo = [d["slo_pct"] for d in wc_data.values()]
            print(f"\n  Weight Robustness (w_c):")
            print(f"    SLO range: {min(wc_slo):.1f}% – {max(wc_slo):.1f}% "
                  f"(spread={max(wc_slo)-min(wc_slo):.1f}pp)")
        if ws_data:
            ws_slo = [d["slo_pct"] for d in ws_data.values()]
            print(f"  Weight Robustness (w_s):")
            print(f"    SLO range: {min(ws_slo):.1f}% – {max(ws_slo):.1f}% "
                  f"(spread={max(ws_slo)-min(ws_slo):.1f}pp)")

    # Save JSON report
    report = {
        "variants": {},
        "meta": {
            "rate": 5.7,
            "workload": "mixed",
            "num_variants": len(all_data),
        },
    }
    for v, d in all_data.items():
        report["variants"][v] = {
            "metrics": d,
            "num_runs": len(all_runs.get(v, [])),
            **VARIANT_META.get(v, {}),
        }
    report_path = base_dir / "sensitivity_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  JSON report saved: {report_path}")


if __name__ == "__main__":
    main()
