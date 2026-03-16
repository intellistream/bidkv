"""Result analysis and paper figure generation for vLLM experiments.

生成论文 §6 需要的 6 张 Figure + 2 张 Table 数据：

Figure 1 [Cat A]: SLO Violation Rate vs KV 压力
Figure 2 [Cat A]: P99 TTFT vs 吞吐量（Pareto 前沿）
Figure 3a/3b [Cat C]: 质量退化对比
Figure 4 [Cat B]: Oracle Gap 可视化
Figure 5 [Cat A]: 覆盖率与降级频率
Figure 6 [Cat B]: Δ budget 敏感性

依赖 matplotlib（可选）— 纯数据汇总不需要 matplotlib。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from bidkv.experiments.vllm.collector import RunResult, load_all_run_results
from bidkv.experiments.vllm.config import (
    STRATEGY_BIDKV,
    STRATEGY_ORACLE_DP,
    SLOConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyAggregation:
    """单个策略在特定 (workload, concurrency) 下的聚合指标。

    多次运行的均值和置信区间。
    """

    strategy: str
    workload: str
    concurrency: int
    runs: int = 0
    throughput_rps_mean: float = 0.0
    throughput_rps_ci95: float = 0.0
    p99_ttft_ms_mean: float = 0.0
    p99_ttft_ms_ci95: float = 0.0
    slo_violation_rate_mean: float = 0.0
    slo_violation_rate_ci95: float = 0.0
    compression_coverage_mean: float = 0.0
    total_compressions_mean: float = 0.0
    total_tokens_freed_mean: float = 0.0
    pressure_events_mean: float = 0.0


def compute_ci95(values: list[float]) -> tuple[float, float]:
    """计算均值 ± 95% CI。

    使用 t 分布近似（n < 30 时 t_{n-1, 0.975}）。

    Parameters
    ----------
    values:
        数据点列表。

    Returns
    -------
    tuple[float, float]
        (mean, half_width_of_95_CI)
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], 0.0)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    # t 分布临界值（近似，n=3 → t=4.303, n=5 → t=2.776, n=10 → t=2.262）
    # 使用保守的查表值
    t_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
    t_crit = t_values.get(n, 1.96)  # n >= 30 时近似 1.96

    ci_half = t_crit * std / math.sqrt(n)
    return (mean, ci_half)


def aggregate_results(
    results: list[RunResult],
    slo_config: SLOConfig | None = None,
) -> list[StrategyAggregation]:
    """将多次运行结果聚合为策略级统计。

    Parameters
    ----------
    results:
        所有运行结果。
    slo_config:
        SLO 配置（用于计算 violation rate）。

    Returns
    -------
    list[StrategyAggregation]
        按策略聚合的统计列表。
    """
    slo = slo_config or SLOConfig()

    # 按 (strategy, workload, concurrency) 分组
    groups: dict[tuple[str, str, int], list[RunResult]] = {}
    for r in results:
        key = (r.strategy, r.workload, r.concurrency)
        groups.setdefault(key, []).append(r)

    aggregations: list[StrategyAggregation] = []
    for (strategy, workload, concurrency), group in sorted(groups.items()):
        throughputs = [r.compute_throughput_rps() for r in group]
        p99_ttfts = [r.compute_p99_ttft_ms() for r in group]
        slo_violations = [r.compute_slo_violation_rate(slo.ttft_target_ms) for r in group]

        # adapter metrics
        total_comps = [float(r.adapter_metrics.get("total_compressions", 0)) for r in group]
        total_freed = [float(r.adapter_metrics.get("total_tokens_freed", 0)) for r in group]
        pressure_events = [float(r.adapter_metrics.get("total_pressure_events", 0)) for r in group]

        tp_mean, tp_ci = compute_ci95(throughputs)
        ttft_mean, ttft_ci = compute_ci95(p99_ttfts)
        slo_mean, slo_ci = compute_ci95(slo_violations)

        # Compression coverage = requests with compression / total requests
        coverages = []
        for r in group:
            total_reqs = len(r.successful_requests)
            # 简化：从 adapter metrics 推断
            comps = r.adapter_metrics.get("total_compressions", 0)
            cov = comps / total_reqs if total_reqs > 0 else 0.0
            coverages.append(min(1.0, cov))

        cov_mean, _ = compute_ci95(coverages)
        comp_mean, _ = compute_ci95(total_comps)
        freed_mean, _ = compute_ci95(total_freed)
        pe_mean, _ = compute_ci95(pressure_events)

        aggregations.append(
            StrategyAggregation(
                strategy=strategy,
                workload=workload,
                concurrency=concurrency,
                runs=len(group),
                throughput_rps_mean=tp_mean,
                throughput_rps_ci95=tp_ci,
                p99_ttft_ms_mean=ttft_mean,
                p99_ttft_ms_ci95=ttft_ci,
                slo_violation_rate_mean=slo_mean,
                slo_violation_rate_ci95=slo_ci,
                compression_coverage_mean=cov_mean,
                total_compressions_mean=comp_mean,
                total_tokens_freed_mean=freed_mean,
                pressure_events_mean=pe_mean,
            )
        )

    return aggregations


def compute_oracle_gap(
    aggregations: list[StrategyAggregation],
) -> dict[str, float]:
    """计算 BidKV 与 Oracle 之间的 gap。

    Oracle Gap = (BidKV_slo_violation - Oracle_slo_violation) / Oracle_slo_violation

    Parameters
    ----------
    aggregations:
        聚合统计。

    Returns
    -------
    dict[str, float]
        {workload__concurrency: oracle_gap}
    """
    # 索引 BidKV 和 Oracle 结果
    bidkv_results: dict[tuple[str, int], StrategyAggregation] = {}
    oracle_results: dict[tuple[str, int], StrategyAggregation] = {}

    for agg in aggregations:
        key = (agg.workload, agg.concurrency)
        if agg.strategy == STRATEGY_BIDKV:
            bidkv_results[key] = agg
        elif agg.strategy == STRATEGY_ORACLE_DP:
            oracle_results[key] = agg

    gaps: dict[str, float] = {}
    for key, bidkv_agg in bidkv_results.items():
        oracle_agg = oracle_results.get(key)
        if oracle_agg is None:
            continue
        oracle_slo = oracle_agg.slo_violation_rate_mean
        bidkv_slo = bidkv_agg.slo_violation_rate_mean
        gap = (bidkv_slo - oracle_slo) / oracle_slo if oracle_slo > 0 else bidkv_slo
        label = f"{key[0]}__c{key[1]}"
        gaps[label] = gap

    return gaps


def export_summary_json(
    aggregations: list[StrategyAggregation],
    oracle_gaps: dict[str, float],
    output_dir: Path,
) -> Path:
    """导出聚合摘要为 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"

    data = {
        "aggregations": [
            {
                "strategy": a.strategy,
                "workload": a.workload,
                "concurrency": a.concurrency,
                "runs": a.runs,
                "throughput_rps": {
                    "mean": a.throughput_rps_mean,
                    "ci95": a.throughput_rps_ci95,
                },
                "p99_ttft_ms": {
                    "mean": a.p99_ttft_ms_mean,
                    "ci95": a.p99_ttft_ms_ci95,
                },
                "slo_violation_rate": {
                    "mean": a.slo_violation_rate_mean,
                    "ci95": a.slo_violation_rate_ci95,
                },
                "compression_coverage": a.compression_coverage_mean,
                "total_compressions": a.total_compressions_mean,
                "total_tokens_freed": a.total_tokens_freed_mean,
                "pressure_events": a.pressure_events_mean,
            }
            for a in aggregations
        ],
        "oracle_gaps": oracle_gaps,
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported summary to %s", path)
    return path


def generate_figures(
    aggregations: list[StrategyAggregation],
    oracle_gaps: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    """生成论文 Figure PDF 文件。

    需要 matplotlib。如果未安装，仅导出数据不生成图片。

    Parameters
    ----------
    aggregations:
        聚合统计。
    oracle_gaps:
        Oracle gap 数据。
    output_dir:
        图片输出目录。

    Returns
    -------
    list[Path]
        生成的 PDF 文件路径列表。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无头模式
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.warning(
            "matplotlib not installed. Exporting data only, skipping figure generation. "
            "Install with: pip install matplotlib"
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    generated: list[Path] = []

    # Figure 1 [Cat A]: SLO Violation Rate — 按策略分组柱状图
    generated.extend(_plot_slo_violation_bar(aggregations, figures_dir))

    # Figure 2 [Cat A]: P99 TTFT vs Throughput — Pareto 散点图
    generated.extend(_plot_pareto_front(aggregations, figures_dir))

    # Figure 4 [Cat B]: Oracle Gap — 箱线图
    generated.extend(_plot_oracle_gap(oracle_gaps, figures_dir))

    # Figure 5 [Cat A]: Compression coverage
    generated.extend(_plot_compression_coverage(aggregations, figures_dir))

    logger.info("Generated %d figures in %s", len(generated), figures_dir)
    return generated


def _plot_slo_violation_bar(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 1: SLO Violation Rate bar chart."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    # 按 workload 分别画图
    workloads = sorted({a.workload for a in aggregations})
    for workload in workloads:
        fig, ax = plt.subplots(figsize=(12, 6))
        wl_data = [a for a in aggregations if a.workload == workload]

        strategies = sorted({a.strategy for a in wl_data})
        concurrencies = sorted({a.concurrency for a in wl_data})

        x_positions = range(len(concurrencies))
        bar_width = 0.8 / max(1, len(strategies))

        for i, strategy in enumerate(strategies):
            means = []
            errors = []
            for c in concurrencies:
                match = [a for a in wl_data if a.strategy == strategy and a.concurrency == c]
                if match:
                    means.append(match[0].slo_violation_rate_mean * 100)
                    errors.append(match[0].slo_violation_rate_ci95 * 100)
                else:
                    means.append(0)
                    errors.append(0)

            positions = [x + i * bar_width for x in x_positions]
            ax.bar(
                positions,
                means,
                bar_width,
                yerr=errors,
                label=strategy,
                capsize=3,
            )

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("SLO Violation Rate (%)")
        ax.set_title(f"[Category A] SLO Violation Rate — {workload}")
        ax.set_xticks([x + bar_width * len(strategies) / 2 for x in x_positions])
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        path = output_dir / f"fig1_slo_violation_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_pareto_front(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 2: P99 TTFT vs Throughput Pareto front."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    workloads = sorted({a.workload for a in aggregations})
    for workload in workloads:
        fig, ax = plt.subplots(figsize=(10, 8))
        wl_data = [a for a in aggregations if a.workload == workload]

        strategies = sorted({a.strategy for a in wl_data})

        for strategy in strategies:
            s_data = [a for a in wl_data if a.strategy == strategy]
            throughputs = [a.throughput_rps_mean for a in s_data]
            ttfts = [a.p99_ttft_ms_mean for a in s_data]
            ax.scatter(throughputs, ttfts, label=strategy, s=60, zorder=5)
            # 连接同一策略不同并发度的点
            sorted_pts = sorted(zip(throughputs, ttfts, strict=False))
            if len(sorted_pts) > 1:
                ax.plot(
                    [p[0] for p in sorted_pts],
                    [p[1] for p in sorted_pts],
                    alpha=0.4,
                    linestyle="--",
                )

        ax.set_xlabel("Throughput (requests/sec)")
        ax.set_ylabel("P99 TTFT (ms)")
        ax.set_title(f"[Category A] Throughput vs Latency Pareto — {workload}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        path = output_dir / f"fig2_pareto_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_oracle_gap(
    oracle_gaps: dict[str, float],
    output_dir: Path,
) -> list[Path]:
    """Figure 4: Oracle Gap box plot."""
    import matplotlib.pyplot as plt

    if not oracle_gaps:
        return []

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = sorted(oracle_gaps.keys())
    values = [oracle_gaps[k] * 100 for k in labels]

    ax.bar(range(len(labels)), values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Workload × Concurrency")
    ax.set_ylabel("Oracle Gap (%)")
    ax.set_title("[Category B] BidKV Oracle Gap")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "fig4_oracle_gap.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [path]


def _plot_compression_coverage(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 5: Compression coverage comparison."""
    import matplotlib.pyplot as plt

    # 仅包含有压缩能力的策略（排除 preempt-evict）
    compress_strategies = [a for a in aggregations if a.strategy != "preempt-evict"]
    if not compress_strategies:
        return []

    paths: list[Path] = []
    workloads = sorted({a.workload for a in compress_strategies})

    for workload in workloads:
        fig, ax = plt.subplots(figsize=(10, 6))
        wl_data = [a for a in compress_strategies if a.workload == workload]

        strategies = sorted({a.strategy for a in wl_data})
        concurrencies = sorted({a.concurrency for a in wl_data})

        bar_width = 0.8 / max(1, len(strategies))
        x_positions = range(len(concurrencies))

        for i, strategy in enumerate(strategies):
            coverages = []
            for c in concurrencies:
                match = [a for a in wl_data if a.strategy == strategy and a.concurrency == c]
                coverages.append(match[0].compression_coverage_mean * 100 if match else 0)
            positions = [x + i * bar_width for x in x_positions]
            ax.bar(positions, coverages, bar_width, label=strategy)

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Compression Coverage (%)")
        ax.set_title(f"[Category A] Compression Coverage — {workload}")
        ax.set_xticks([x + bar_width * len(strategies) / 2 for x in x_positions])
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        path = output_dir / f"fig5_coverage_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def generate_table1_data(aggregations: list[StrategyAggregation]) -> list[dict[str, object]]:
    """生成 Table 1 数据：7 baseline 描述 + 特性矩阵。

    Returns
    -------
    list[dict]
        每个策略一行数据。
    """
    strategy_descriptions = {
        "preempt-evict": {
            "description": "Framework default preemption (no compression)",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Lower bound — zero compression baseline",
        },
        "static-random": {
            "description": "Random victim selection with fixed compression ratio",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Control group — isolates information value",
        },
        "h2o-style": {
            "description": "Token-level importance scoring, no bid mechanism",
            "has_scoring": True,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Isolates bid mechanism contribution from scoring",
        },
        "uniform": {
            "description": "Equal compression ratio across all requests",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Isolates differentiated compression value",
        },
        "global-nobid": {
            "description": "System-inferred utility (no user bid interface)",
            "has_scoring": True,
            "has_bid": False,
            "has_solver": True,
            "design_rationale": "Key bid attribution — scoring + solver without bid",
        },
        "slack-aware": {
            "description": "SLO deadline-aware victim selection",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Quality-unaware scheduling baseline",
        },
        "bidkv": {
            "description": "Full BidKV pipeline (H2O scoring + bid + solver)",
            "has_scoring": True,
            "has_bid": True,
            "has_solver": True,
            "design_rationale": "Complete system with user-explicit quality preference",
        },
        "oracle-dp": {
            "description": "Offline optimal solution via dynamic programming",
            "has_scoring": True,
            "has_bid": True,
            "has_solver": True,
            "design_rationale": "Upper bound reference (not online-feasible)",
        },
    }

    table_rows: list[dict[str, object]] = []
    for strategy, desc in strategy_descriptions.items():
        row: dict[str, object] = {
            "strategy": strategy,
            **desc,
        }
        # 附加聚合指标（如果存在）
        strategy_aggs = [a for a in aggregations if a.strategy == strategy]
        if strategy_aggs:
            avg_slo = sum(a.slo_violation_rate_mean for a in strategy_aggs) / len(strategy_aggs)
            avg_throughput = sum(a.throughput_rps_mean for a in strategy_aggs) / len(strategy_aggs)
            row["avg_slo_violation_rate"] = avg_slo
            row["avg_throughput_rps"] = avg_throughput

        table_rows.append(row)

    return table_rows


def run_analysis(results_dir: Path, output_dir: Path | None = None) -> None:
    """运行完整分析流程。

    Parameters
    ----------
    results_dir:
        RunResult JSON 文件所在目录。
    output_dir:
        分析结果输出目录。默认与 results_dir 相同。
    """
    if output_dir is None:
        output_dir = results_dir

    results = load_all_run_results(results_dir)
    if not results:
        logger.warning("No results found in %s", results_dir)
        return

    logger.info("Loaded %d run results", len(results))

    aggregations = aggregate_results(results)
    oracle_gaps = compute_oracle_gap(aggregations)

    # 导出 JSON 摘要
    export_summary_json(aggregations, oracle_gaps, output_dir)

    # 生成 Table 1 数据
    table1 = generate_table1_data(aggregations)
    table1_path = output_dir / "table1_baselines.json"
    table1_path.write_text(json.dumps(table1, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Table 1 data exported to %s", table1_path)

    # 生成 Figure PDF
    generated = generate_figures(aggregations, oracle_gaps, output_dir)
    logger.info("Analysis complete. %d figures generated.", len(generated))
