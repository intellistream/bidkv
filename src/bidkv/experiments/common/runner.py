"""实验运行器基础设施。

提供跨框架共享的实验配置和运行器基类。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bidkv.experiments.common.audit import AuditLogger
from bidkv.experiments.common.report import RunResult
from bidkv.experiments.common.trace import FrozenTrace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """实验配置。

    Attributes
    ----------
    framework:
        框架名称（"vllm" / "sglang"）。
    strategies:
        要运行的策略名称列表。
    workloads:
        工作负载类型列表（与 trace 文件关联）。
    num_runs:
        每个 (strategy, workload, concurrency) 组合的重复次数（≥ 3）。
    concurrency_levels:
        并发度列表（如 [8, 16, 32]）。
    output_dir:
        实验结果输出目录。
    model_name:
        模型名称/路径。
    max_total_tokens:
        最大 KV token 容量。
    slo_ttft_ms:
        SLO TTFT 上限（毫秒），默认 500。
    trace_dir:
        Frozen trace 文件目录。
    """

    framework: str
    strategies: tuple[str, ...]
    workloads: tuple[str, ...]
    num_runs: int = 3
    concurrency_levels: tuple[int, ...] = (8, 16, 32)
    output_dir: str = "results/"
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    max_total_tokens: int = 16384
    slo_ttft_ms: float = 500.0
    trace_dir: str = "traces/"

    def total_runs(self) -> int:
        """总实验 run 次数。"""
        return (
            len(self.strategies)
            * len(self.workloads)
            * len(self.concurrency_levels)
            * self.num_runs
        )


class BaseExperimentRunner(ABC):
    """实验运行器基类。

    子类实现 ``_run_single()`` 执行单次实验。

    Parameters
    ----------
    config:
        实验配置。
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._results: list[RunResult] = []

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    @property
    def results(self) -> list[RunResult]:
        return list(self._results)

    def plan(self) -> list[dict[str, object]]:
        """生成实验计划（所有 run 的参数列表）。

        Returns
        -------
        list[dict[str, object]]
            每个 dict 包含 strategy, workload, concurrency, run_id。
        """
        runs: list[dict[str, object]] = []
        for strategy in self._config.strategies:
            for workload in self._config.workloads:
                for concurrency in self._config.concurrency_levels:
                    for run_idx in range(self._config.num_runs):
                        runs.append(
                            {
                                "strategy": strategy,
                                "workload": workload,
                                "concurrency": concurrency,
                                "run_id": f"run_{run_idx}",
                                "run_index": run_idx,
                            }
                        )
        logger.info(
            "Experiment plan: %d total runs (%d strategies × %d workloads "
            "× %d concurrency × %d repetitions)",
            len(runs),
            len(self._config.strategies),
            len(self._config.workloads),
            len(self._config.concurrency_levels),
            self._config.num_runs,
        )
        return runs

    @abstractmethod
    def _run_single(
        self,
        *,
        strategy: str,
        workload: str,
        concurrency: int,
        run_id: str,
        trace: FrozenTrace,
        audit_logger: AuditLogger,
    ) -> RunResult:
        """执行单次实验 run。

        Parameters
        ----------
        strategy:
            策略名称。
        workload:
            工作负载类型。
        concurrency:
            并发度。
        run_id:
            Run 标识。
        trace:
            Frozen trace。
        audit_logger:
            审计日志记录器。

        Returns
        -------
        RunResult
            本次 run 的结果。
        """
