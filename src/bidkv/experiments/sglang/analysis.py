"""跨框架分析 — Figure 7 + Table 2 生成器。

**Figure 7 [Category D: Portability]**：
并排展示 vLLM 和 SGLang 上的 SLO Attainment Rate 趋势。
> BidKV achieves directionally consistent improvements on SGLang
> (RadixAttention, tree-based KV) and vLLM (PagedAttention, flat block),
> demonstrating the portability of compression scheduling primitives.

**Table 2**：跨框架性能汇总。

**v7.1 澄清**：within-platform consistency 对 SGLang 和 vLLM 各自独立要求。
不意味着两个平台之间的 candidate universe 必须完全一致（架构差异使得不可能），
而是说两个平台各自内部的 baseline 比较都必须公平。
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bidkv.experiments.common.report import ExperimentReport
from bidkv.experiments.common.report import RunResult as ReportRunResult
from bidkv.experiments.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


def _compute_tpot(r: dict[str, Any]) -> float:
    """Compute TPOT (ms) from a request result dict."""
    ct = r.get("completion_tokens", 0)
    if ct <= 1:
        return 0.0
    ttft = r.get("ttft_ms", 0.0)
    tl = r.get("total_latency_ms", 0.0)
    decode = tl - ttft
    return decode / (ct - 1) if decode > 0 else 0.0


def _load_collector_results_as_report(
    results_dir: Path,
    framework: str,
    slo_ttft_target_ms: float = 2000.0,
    slo_tpot_target_ms: float = 100.0,
) -> ExperimentReport:
    """从 collector RunResult JSON 文件目录构建 ExperimentReport。

    collector 的 JSON 格式（vllm/sglang runner 输出）与
    common.report.ExperimentReport 格式不同。此函数做转换桥接。

    Parameters
    ----------
    results_dir:
        包含 collector RunResult JSON 文件的目录。
    framework:
        框架名称（"vllm" / "sglang"）。
    slo_ttft_target_ms:
        TTFT SLO 阈值（ms），用于计算 slo_attainment_rate。

    Returns
    -------
    ExperimentReport
        统一格式的报告。
    """
    report = ExperimentReport(framework=framework)

    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping malformed file %s: %s", path, e)
            continue

        # 跳过非 RunResult 文件（如 manifest、analysis 输出等）
        if "strategy" not in data or "request_results" not in data:
            continue

        summary = data.get("summary", {})
        throughput = summary.get("throughput_rps", 0.0)
        p99_ttft = summary.get("p99_ttft_ms", 0.0)

        # 从 request_results 计算 slo_attainment_rate
        request_results = data.get("request_results", [])
        successful = [r for r in request_results if not r.get("error", "")]
        if successful:
            check_tpot = slo_tpot_target_ms > 0
            attained = sum(
                1
                for r in successful
                if r.get("ttft_ms", 0) <= slo_ttft_target_ms
                and (not check_tpot or _compute_tpot(r) <= slo_tpot_target_ms)
            )
            slo_attainment = attained / len(successful)
        else:
            slo_attainment = 0.0

        # adapter_metrics 中的驱逐数据
        adapter_metrics = data.get("adapter_metrics", {})
        total_evictions = adapter_metrics.get(
            "total_evictions", adapter_metrics.get("total_compressions", 0),
        )
        total_requests_seen = adapter_metrics.get("total_requests", len(request_results))
        coverage = total_evictions / total_requests_seen if total_requests_seen > 0 else 0.0

        metrics = ExperimentMetrics(
            slo_attainment_rate=slo_attainment,
            p99_ttft_ms=p99_ttft,
            throughput_rps=throughput,
            eviction_coverage=coverage,
            adapter_metrics=adapter_metrics,
        )

        run_result = ReportRunResult(
            framework=framework,
            strategy=data["strategy"],
            workload=data.get("workload", "unknown"),
            concurrency=int(data.get("request_rate", 0)),
            run_id=f"run_{data.get('run_index', 0)}",
            metrics=metrics,
            trace_hash="",
            audit_event_count=0,
            metadata={
                "request_rate": data.get("request_rate", 0),
                "run_label": data.get("run_label", ""),
            },
        )
        report.add_run(run_result)

    logger.info("Loaded %d runs from %s as %s report", len(report.runs), results_dir, framework)
    return report


@dataclass(frozen=True)
class StrategyAggregation:
    """单个策略在某框架/工作负载上的聚合指标。

    业界标准指标体系（vLLM SOSP'23, DistServe OSDI'24, SGLang）。
    """

    strategy: str
    mean_throughput: float
    mean_p50_ttft: float
    mean_p99_ttft: float
    mean_p50_tpot: float
    mean_p99_tpot: float
    mean_p50_e2e_latency: float
    mean_p99_e2e_latency: float
    mean_normalized_latency: float
    # SLO attainment（可选辅助）
    mean_slo_attainment: float
    ci95_slo_attainment: float
    num_runs: int


@dataclass
class CrossFrameworkComparison:
    """跨框架比较结果。

    Attributes
    ----------
    vllm_results:
        vLLM 各策略的聚合结果。
    sglang_results:
        SGLang 各策略的聚合结果。
    directional_consistency:
        方向一致性检查结果。
    """

    vllm_results: dict[str, StrategyAggregation] = field(default_factory=dict)
    sglang_results: dict[str, StrategyAggregation] = field(default_factory=dict)
    directional_consistency: bool = False
    consistency_details: dict[str, Any] = field(default_factory=dict)


class CrossFrameworkAnalyzer:
    """跨框架分析器 — 生成 Figure 7 和 Table 2 数据。

    Parameters
    ----------
    vllm_report:
        vLLM 实验报告（来自 #047）。可为 None（仅分析 SGLang）。
    sglang_report:
        SGLang 实验报告（来自 #048）。
    """

    def __init__(
        self,
        sglang_report: ExperimentReport,
        vllm_report: ExperimentReport | None = None,
    ) -> None:
        self._vllm_report = vllm_report
        self._sglang_report = sglang_report

    def aggregate_by_strategy(
        self,
        report: ExperimentReport,
        *,
        workload: str | None = None,
        concurrency: int | None = None,
    ) -> dict[str, StrategyAggregation]:
        """按策略聚合结果。

        Parameters
        ----------
        report:
            实验报告。
        workload:
            过滤工作负载（None = 所有）。
        concurrency:
            过滤并发度（None = 所有）。

        Returns
        -------
        dict[str, StrategyAggregation]
            策略名称 → 聚合结果。
        """
        runs = report.filter_runs(workload=workload, concurrency=concurrency)

        # Group by strategy
        by_strategy: dict[str, list[ReportRunResult]] = {}
        for run in runs:
            by_strategy.setdefault(run.strategy, []).append(run)

        aggregations: dict[str, StrategyAggregation] = {}
        for strategy, strategy_runs in by_strategy.items():
            slo_rates = [r.metrics.slo_attainment_rate for r in strategy_runs]
            p99_ttfts = [r.metrics.p99_ttft_ms for r in strategy_runs]
            throughputs = [r.metrics.throughput_rps for r in strategy_runs]

            mean_slo = statistics.mean(slo_rates)
            ci95_slo = _compute_ci95(slo_rates)

            aggregations[strategy] = StrategyAggregation(
                strategy=strategy,
                mean_throughput=statistics.mean(throughputs),
                mean_p50_ttft=0.0,
                mean_p99_ttft=statistics.mean(p99_ttfts),
                mean_p50_tpot=0.0,
                mean_p99_tpot=0.0,
                mean_p50_e2e_latency=0.0,
                mean_p99_e2e_latency=0.0,
                mean_normalized_latency=0.0,
                mean_slo_attainment=mean_slo,
                ci95_slo_attainment=ci95_slo,
                num_runs=len(strategy_runs),
            )

        return aggregations

    def check_directional_consistency(
        self,
        *,
        workload: str | None = None,
        concurrency: int | None = None,
    ) -> CrossFrameworkComparison:
        """检查跨框架方向一致性。

        **Directional Consistency 定义（v2.3）**：
        SGLang 上 BidKV 的改进趋势：
        - DC-1a: BidKV ≥ Preempt-Evict (sglang_default)
        - DC-1b: BidKV ≥ Slack-Aware
        不要求 numerical equivalence。

        Returns
        -------
        CrossFrameworkComparison
            比较结果。
        """
        comparison = CrossFrameworkComparison()

        # SGLang 聚合
        sglang_agg = self.aggregate_by_strategy(
            self._sglang_report,
            workload=workload,
            concurrency=concurrency,
        )
        comparison.sglang_results = sglang_agg

        # vLLM 聚合（如果可用）
        if self._vllm_report is not None:
            vllm_agg = self.aggregate_by_strategy(
                self._vllm_report,
                workload=workload,
                concurrency=concurrency,
            )
            comparison.vllm_results = vllm_agg

        # 方向一致性检查
        details: dict[str, Any] = {}

        # SGLang 内部一致性
        sglang_bidkv = sglang_agg.get("bidkv")
        sglang_default = sglang_agg.get("sglang_default")
        sglang_slack = sglang_agg.get("slack_aware")

        sglang_consistent = True
        # DC-1a: BidKV ≥ Preempt-Evict (sglang_default)
        if sglang_bidkv and sglang_default:
            bidkv_vs_default = (
                sglang_bidkv.mean_slo_attainment >= sglang_default.mean_slo_attainment
            )
            details["sglang_bidkv_vs_default"] = {
                "bidkv_slo": sglang_bidkv.mean_slo_attainment,
                "default_slo": sglang_default.mean_slo_attainment,
                "bidkv_better_or_equal": bidkv_vs_default,
            }
            if not bidkv_vs_default:
                sglang_consistent = False

        # DC-1b: BidKV ≥ Slack-Aware
        if sglang_bidkv and sglang_slack:
            bidkv_vs_slack = sglang_bidkv.mean_slo_attainment >= sglang_slack.mean_slo_attainment
            details["sglang_bidkv_vs_slack_aware"] = {
                "bidkv_slo": sglang_bidkv.mean_slo_attainment,
                "slack_aware_slo": sglang_slack.mean_slo_attainment,
                "bidkv_better_or_equal": bidkv_vs_slack,
            }
            if not bidkv_vs_slack:
                sglang_consistent = False

        details["sglang_internally_consistent"] = sglang_consistent

        # 跨框架一致性
        if self._vllm_report is not None and comparison.vllm_results:
            vllm_bidkv = comparison.vllm_results.get("bidkv")
            vllm_default = comparison.vllm_results.get(
                "preempt_evict"
            ) or comparison.vllm_results.get("preempt-evict")

            cross_consistent = True
            if vllm_bidkv and vllm_default:
                vllm_improvement = vllm_bidkv.mean_slo_attainment - vllm_default.mean_slo_attainment
                details["vllm_bidkv_improvement"] = vllm_improvement
                details["vllm_bidkv_better_than_default"] = vllm_improvement > 0

            if sglang_bidkv and sglang_default:
                sglang_improvement = (
                    sglang_bidkv.mean_slo_attainment - sglang_default.mean_slo_attainment
                )
                details["sglang_bidkv_improvement"] = sglang_improvement
                details["sglang_bidkv_better_than_default"] = sglang_improvement > 0

                # 方向一致：两个框架上 BidKV 都优于 Default
                if "vllm_bidkv_improvement" in details:
                    cross_consistent = details.get(
                        "vllm_bidkv_better_than_default", False
                    ) and details.get("sglang_bidkv_better_than_default", False)
                    details["cross_framework_directional_consistent"] = cross_consistent
            else:
                cross_consistent = False

            comparison.directional_consistency = sglang_consistent and cross_consistent
        else:
            comparison.directional_consistency = sglang_consistent

        comparison.consistency_details = details
        return comparison

    def generate_table2_data(
        self,
        *,
        workload: str | None = None,
        concurrency: int | None = None,
    ) -> list[dict[str, object]]:
        """生成 Table 2 数据：跨框架性能汇总。

        Returns
        -------
        list[dict[str, object]]
            每行一个 dict: framework, strategy, slo_attainment, improvement。

        Table 2 format:
        | Framework | Metric | Default | Slack-Aware | BidKV | Improvement |
        """
        rows: list[dict[str, object]] = []

        # SGLang rows
        sglang_agg = self.aggregate_by_strategy(
            self._sglang_report,
            workload=workload,
            concurrency=concurrency,
        )
        sglang_default_slo = 0.0
        if "sglang_default" in sglang_agg:
            sglang_default_slo = sglang_agg["sglang_default"].mean_slo_attainment

        for strategy, agg in sglang_agg.items():
            improvement = agg.mean_slo_attainment - sglang_default_slo
            rows.append(
                {
                    "framework": "SGLang",
                    "strategy": strategy,
                    "slo_attainment_pct": agg.mean_slo_attainment * 100,
                    "ci95_pct": agg.ci95_slo_attainment * 100,
                    "p99_ttft_ms": agg.mean_p99_ttft,
                    "throughput_rps": agg.mean_throughput,
                    "improvement_pct": improvement * 100,
                    "num_runs": agg.num_runs,
                }
            )

        # vLLM rows (if available)
        if self._vllm_report is not None:
            vllm_agg = self.aggregate_by_strategy(
                self._vllm_report,
                workload=workload,
                concurrency=concurrency,
            )
            vllm_default_slo = 0.0
            for name in ("preempt_evict", "preempt-evict"):
                if name in vllm_agg:
                    vllm_default_slo = vllm_agg[name].mean_slo_attainment
                    break

            for strategy, agg in vllm_agg.items():
                improvement = agg.mean_slo_attainment - vllm_default_slo
                rows.append(
                    {
                        "framework": "vLLM",
                        "strategy": strategy,
                        "slo_attainment_pct": agg.mean_slo_attainment * 100,
                        "ci95_pct": agg.ci95_slo_attainment * 100,
                        "p99_ttft_ms": agg.mean_p99_ttft,
                        "throughput_rps": agg.mean_throughput,
                        "improvement_pct": improvement * 100,
                        "num_runs": agg.num_runs,
                    }
                )

        return rows

    def save_analysis(self, output_dir: str | Path) -> dict[str, Path]:
        """保存完整分析结果（Table 2 数据 + consistency 报告）。

        Returns
        -------
        dict[str, Path]
            保存的文件路径 dict（table2, consistency）。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        # Table 2 数据
        table2 = self.generate_table2_data()
        table2_path = out / "table2_cross_framework.json"
        table2_path.write_text(json.dumps(table2, indent=2, default=str))
        saved["table2"] = table2_path

        # Directional consistency
        comparison = self.check_directional_consistency()
        consistency_path = out / "directional_consistency.json"
        consistency_data = {
            "directional_consistency": comparison.directional_consistency,
            "details": comparison.consistency_details,
            "sglang_strategies": {
                k: {
                    "mean_slo_attainment": v.mean_slo_attainment,
                    "ci95_slo_attainment": v.ci95_slo_attainment,
                    "mean_p99_ttft": v.mean_p99_ttft,
                    "mean_throughput": v.mean_throughput,
                    "num_runs": v.num_runs,
                }
                for k, v in comparison.sglang_results.items()
            },
        }
        if comparison.vllm_results:
            consistency_data["vllm_strategies"] = {
                k: {
                    "mean_slo_attainment": v.mean_slo_attainment,
                    "ci95_slo_attainment": v.ci95_slo_attainment,
                    "mean_p99_ttft": v.mean_p99_ttft,
                    "mean_throughput": v.mean_throughput,
                    "num_runs": v.num_runs,
                }
                for k, v in comparison.vllm_results.items()
            }
        consistency_path.write_text(json.dumps(consistency_data, indent=2, default=str))
        saved["consistency"] = consistency_path

        logger.info("Analysis saved to %s", out)
        return saved


def _compute_ci95(values: list[float]) -> float:
    """计算 95% 置信区间半宽。"""
    if len(values) < 2:
        return 0.0
    n = len(values)
    sd = statistics.stdev(values)
    # t 分布近似（n < 30 时）
    # 简化为 z=1.96 近似
    return 1.96 * sd / (n**0.5)


def _plot_figure7(
    comparison: CrossFrameworkComparison,
    output_dir: Path,
) -> list[Path]:
    """Figure 7 [Category D: Portability] — 跨框架 SLO Attainment 并排柱状图。

    左：vLLM 上各策略的 mean SLO attainment
    右：SGLang 上各策略的 mean SLO attainment
    用于展示 BidKV 在两个框架上方向一致的改进。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping Figure 7 plot")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # SGLang 子图（左）
    sglang_agg = comparison.sglang_results
    if sglang_agg:
        strategies = sorted(sglang_agg.keys())
        slo_vals = [sglang_agg[s].mean_slo_attainment * 100 for s in strategies]
        ci_vals = [sglang_agg[s].ci95_slo_attainment * 100 for s in strategies]

        colors = []
        for s in strategies:
            if s == "bidkv":
                colors.append("#2196F3")
            else:
                colors.append("#9E9E9E")

        axes[0].bar(
            range(len(strategies)),
            slo_vals,
            yerr=ci_vals,
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_xticks(range(len(strategies)))
        axes[0].set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
        axes[0].set_ylabel("SLO Attainment Rate (%)")
        axes[0].set_title("SGLang (RadixAttention)")
        axes[0].grid(axis="y", alpha=0.3)

    # vLLM 子图（右）
    vllm_agg = comparison.vllm_results
    if vllm_agg:
        strategies = sorted(vllm_agg.keys())
        slo_vals = [vllm_agg[s].mean_slo_attainment * 100 for s in strategies]
        ci_vals = [vllm_agg[s].ci95_slo_attainment * 100 for s in strategies]

        colors = []
        for s in strategies:
            if s == "bidkv":
                colors.append("#2196F3")
            elif "preempt" in s:
                colors.append("#FF9800")
            else:
                colors.append("#9E9E9E")

        axes[1].bar(
            range(len(strategies)),
            slo_vals,
            yerr=ci_vals,
            color=colors,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].set_xticks(range(len(strategies)))
        axes[1].set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
        axes[1].set_title("vLLM (PagedAttention)")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "vLLM data not available", ha="center", va="center")
        axes[1].set_title("vLLM (PagedAttention)")

    # DC 标注
    dc_text = "DC: " + ("PASS" if comparison.directional_consistency else "FAIL")
    fig.suptitle(
        f"[Category D] Cross-Framework Portability — {dc_text}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "fig7_cross_framework_portability.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    paths.append(path)
    logger.info("Figure 7 saved to %s", path)

    return paths


def run_sglang_analysis(
    sglang_results_dir: Path,
    output_dir: Path,
    vllm_results_dir: Path | None = None,
) -> None:
    """运行完整 SGLang 分析流程。

    Parameters
    ----------
    sglang_results_dir:
        SGLang RunResult JSON 文件所在目录。
    output_dir:
        分析结果输出目录。
    vllm_results_dir:
        vLLM RunResult JSON 文件所在目录（可选，用于跨框架对比）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 SGLang 结果
    sglang_report = _load_collector_results_as_report(sglang_results_dir, "sglang")
    if not sglang_report.runs:
        logger.warning("No SGLang results found in %s", sglang_results_dir)
        return

    # 可选加载 vLLM 结果
    vllm_report = None
    if vllm_results_dir and vllm_results_dir.exists():
        vllm_report = _load_collector_results_as_report(vllm_results_dir, "vllm")
        if not vllm_report.runs:
            logger.warning(
                "No vLLM results found in %s, skipping cross-framework",
                vllm_results_dir,
            )
            vllm_report = None

    # 构建分析器
    analyzer = CrossFrameworkAnalyzer(
        sglang_report=sglang_report,
        vllm_report=vllm_report,
    )

    # 保存 Table 2 + DC 检查
    saved = analyzer.save_analysis(output_dir)
    for name, path in saved.items():
        logger.info("Saved %s → %s", name, path)

    # Figure 7
    comparison = analyzer.check_directional_consistency()
    fig7_paths = _plot_figure7(comparison, output_dir / "figures")
    for p in fig7_paths:
        logger.info("Figure 7 → %s", p)

    # 打印 DC 摘要
    dc = comparison.consistency_details
    logger.info("=== Directional Consistency Summary ===")
    logger.info("Overall DC: %s", "PASS" if comparison.directional_consistency else "FAIL")
    for key, val in dc.items():
        logger.info("  %s: %s", key, val)

    logger.info("SGLang analysis complete.")


def main(argv: list[str] | None = None) -> None:
    """CLI 入口：SGLang 分析脚本。"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="BidKV SGLang Result Analysis")
    parser.add_argument(
        "--sglang-results-dir",
        type=str,
        required=True,
        help="Directory containing SGLang RunResult JSON files.",
    )
    parser.add_argument(
        "--vllm-results-dir",
        type=str,
        default=None,
        help="Directory containing vLLM RunResult JSON files (for cross-framework comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: sglang-results-dir/analysis).",
    )
    args = parser.parse_args(argv)

    sglang_dir = Path(args.sglang_results_dir)
    vllm_dir = Path(args.vllm_results_dir) if args.vllm_results_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else sglang_dir / "analysis"

    run_sglang_analysis(sglang_dir, output_dir, vllm_dir)


if __name__ == "__main__":
    main()
