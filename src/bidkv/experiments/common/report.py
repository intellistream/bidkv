"""实验结果报告数据结构和 I/O。

统一 vLLM (#047) / SGLang (#048) 结果格式，便于跨框架分析。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from bidkv.experiments.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """单次实验 run 的结果。

    Attributes
    ----------
    framework:
        框架名称（"vllm" / "sglang"）。
    strategy:
        策略名称。
    workload:
        工作负载类型。
    concurrency:
        并发度。
    run_id:
        Run 标识（如 "run_0"）。
    metrics:
        实验指标。
    trace_hash:
        使用的 frozen trace 内容哈希。
    audit_event_count:
        审计日志中的 pressure event 数量。
    metadata:
        附加元数据。
    """

    framework: str
    strategy: str
    workload: str
    concurrency: int
    run_id: str
    metrics: ExperimentMetrics
    trace_hash: str
    audit_event_count: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result = {
            "framework": self.framework,
            "strategy": self.strategy,
            "workload": self.workload,
            "concurrency": self.concurrency,
            "run_id": self.run_id,
            "metrics": self.metrics.to_dict(),
            "trace_hash": self.trace_hash,
            "audit_event_count": self.audit_event_count,
            "metadata": self.metadata,
        }
        return result


@dataclass
class ExperimentReport:
    """完整实验报告（多次 run 的结果集合）。

    Attributes
    ----------
    framework:
        框架名称。
    config_summary:
        实验配置摘要。
    runs:
        所有 run 的结果列表。
    """

    framework: str
    config_summary: dict[str, object] = field(default_factory=dict)
    runs: list[RunResult] = field(default_factory=list)

    def add_run(self, result: RunResult) -> None:
        self.runs.append(result)

    def save(self, output_dir: str | Path) -> Path:
        """保存完整报告为 JSON 文件。

        Returns
        -------
        Path
            保存的文件路径。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"report_{self.framework}.json"

        data = {
            "framework": self.framework,
            "config_summary": self.config_summary,
            "total_runs": len(self.runs),
            "runs": [r.to_dict() for r in self.runs],
        }

        filepath.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Report saved: %s (%d runs)", filepath, len(self.runs))
        return filepath

    @classmethod
    def load(cls, path: str | Path) -> ExperimentReport:
        """从 JSON 文件加载报告。

        Returns
        -------
        ExperimentReport
            已加载的报告。
        """
        p = Path(path)
        data = json.loads(p.read_text())

        report = cls(
            framework=data["framework"],
            config_summary=data.get("config_summary", {}),
        )

        for run_data in data.get("runs", []):
            metrics_data = run_data["metrics"]
            metrics = ExperimentMetrics(
                slo_attainment_rate=metrics_data.get(
                    "slo_attainment_rate", metrics_data.get("slo_violation_rate", 0.0)
                ),
                p99_ttft_ms=metrics_data["p99_ttft_ms"],
                throughput_rps=metrics_data["throughput_rps"],
                compression_coverage=metrics_data["compression_coverage"],
                quality_rouge1=metrics_data.get("quality_rouge1"),
                quality_em=metrics_data.get("quality_em"),
                adapter_metrics=metrics_data.get("adapter_metrics", {}),
            )
            result = RunResult(
                framework=run_data["framework"],
                strategy=run_data["strategy"],
                workload=run_data["workload"],
                concurrency=run_data["concurrency"],
                run_id=run_data["run_id"],
                metrics=metrics,
                trace_hash=run_data["trace_hash"],
                audit_event_count=run_data.get("audit_event_count", 0),
                metadata=run_data.get("metadata", {}),
            )
            report.add_run(result)

        logger.info("Report loaded: %s (%d runs)", p, len(report.runs))
        return report

    def filter_runs(
        self,
        *,
        strategy: str | None = None,
        workload: str | None = None,
        concurrency: int | None = None,
    ) -> list[RunResult]:
        """过滤 runs by 条件。"""
        filtered = self.runs
        if strategy is not None:
            filtered = [r for r in filtered if r.strategy == strategy]
        if workload is not None:
            filtered = [r for r in filtered if r.workload == workload]
        if concurrency is not None:
            filtered = [r for r in filtered if r.concurrency == concurrency]
        return filtered
