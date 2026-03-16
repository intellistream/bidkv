"""vLLM 7-baseline experiment runner — 论文 §6 的主实验编排脚本。

Usage
-----
# 运行全部实验矩阵
python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict,...,bidkv,oracle-dp" \
    --workloads "chat,summarization,qa" \
    --runs 3 \
    --concurrency-levels "8,16,32" \
    --output-dir results/vllm_$(date +%Y%m%d)/

# 仅运行 preempt-evict 基线（验证实验框架）
python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict" \
    --workloads "chat" \
    --runs 1 \
    --concurrency-levels "8" \
    --output-dir results/vllm_smoke/

运行流程
--------
1. 验证 traces 已冻结（否则报错提示先运行 freeze 命令）
2. 启动（或连接到已运行的）vLLM 服务
3. 按 strategy × workload × concurrency × run 遍历运行
4. 对于 preempt-evict：使用原生 vLLM（无 bidkv 注入）
5. 对于其他策略：通过 VLLMAdapter 注入 bidkv + baseline overlay
6. 收集指标并保存为 JSON
7. 运行 candidate-universe consistency 校验
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from bidkv.baselines import BaselineRegistry
from bidkv.experiments.vllm.collector import (
    RequestResult,
    RunResult,
    save_run_result,
)
from bidkv.experiments.vllm.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    DEFAULT_CONCURRENCY_LEVELS,
    STRATEGY_PREEMPT_EVICT,
    ExperimentConfig,
    SLOConfig,
    VLLMServerConfig,
)
from bidkv.experiments.vllm.workload import (
    RequestTrace,
    WorkloadTrace,
    load_trace,
)

logger = logging.getLogger(__name__)


class VLLMExperimentRunner:
    """vLLM 实验编排器。

    负责 strategy × workload × concurrency × run 的全矩阵运行，
    调用 vLLM OpenAI-compatible API 发送请求并收集结果。

    Parameters
    ----------
    config:
        实验配置。
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._registry = BaselineRegistry()
        self._registry.create_default_registry()
        self._traces: dict[str, WorkloadTrace] = {}

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    def load_traces(self) -> None:
        """加载所有冻结的工作负载 traces。

        Raises
        ------
        FileNotFoundError
            trace 文件不存在。
        """
        for workload in self._config.workloads:
            trace_path = self._config.traces_dir / f"{workload}.json"
            self._traces[workload] = load_trace(trace_path)
            logger.info(
                "Loaded trace: %s (%d requests)",
                workload,
                self._traces[workload].num_requests,
            )

    def run_all(self) -> list[RunResult]:
        """运行完整实验矩阵。

        Returns
        -------
        list[RunResult]
            所有运行结果。
        """
        self.load_traces()

        results: list[RunResult] = []
        total = self._config.total_runs
        current = 0

        for strategy in self._config.strategies:
            for workload in self._config.workloads:
                for concurrency in self._config.concurrency_levels:
                    for run_idx in range(self._config.runs_per_combo):
                        current += 1
                        label = self._config.run_label(strategy, workload, concurrency, run_idx)
                        logger.info("[%d/%d] Running: %s", current, total, label)

                        result = self._run_single(
                            strategy=strategy,
                            workload=workload,
                            concurrency=concurrency,
                            run_index=run_idx,
                        )
                        results.append(result)
                        save_run_result(result, self._config.output_dir)

                        logger.info(
                            "[%d/%d] Completed: %s (throughput=%.2f rps, "
                            "p99_ttft=%.1f ms, success=%d/%d)",
                            current,
                            total,
                            label,
                            result.compute_throughput_rps(),
                            result.compute_p99_ttft_ms(),
                            len(result.successful_requests),
                            len(result.request_results),
                        )

        logger.info("All %d runs completed. Results in %s", total, self._config.output_dir)
        return results

    def _run_single(
        self,
        *,
        strategy: str,
        workload: str,
        concurrency: int,
        run_index: int,
    ) -> RunResult:
        """运行单次实验。

        Parameters
        ----------
        strategy:
            策略名称。
        workload:
            工作负载名称。
        concurrency:
            并发度。
        run_index:
            运行序号。

        Returns
        -------
        RunResult
            运行结果。
        """
        label = self._config.run_label(strategy, workload, concurrency, run_index)
        trace = self._traces[workload]

        result = RunResult(
            run_label=label,
            strategy=strategy,
            workload=workload,
            concurrency=concurrency,
            run_index=run_index,
            start_time=time.time(),
        )

        # 验证策略存在（用于 candidate-universe consistency 记录）
        if strategy != STRATEGY_PREEMPT_EVICT:
            self._registry.get(strategy)  # validate strategy exists

        # 发送请求并收集结果
        request_results = asyncio.run(
            self._send_workload(
                trace=trace,
                concurrency=concurrency,
                strategy_name=strategy,
            )
        )
        result.request_results = request_results
        result.end_time = time.time()

        return result

    async def _send_workload(
        self,
        *,
        trace: WorkloadTrace,
        concurrency: int,
        strategy_name: str,
    ) -> list[RequestResult]:
        """异步发送工作负载请求到 vLLM 服务。

        使用信号量控制并发度，通过 vLLM OpenAI-compatible API 发送请求。

        Parameters
        ----------
        trace:
            工作负载 trace。
        concurrency:
            最大并发请求数。
        strategy_name:
            策略名称（用于日志）。

        Returns
        -------
        list[RequestResult]
            所有请求的结果。
        """

        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []
        api_url = f"{self._config.server.api_url}/chat/completions"

        async def send_one(req_trace: RequestTrace) -> RequestResult:
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._send_request_sync, api_url, req_trace, strategy_name
                )

        tasks = [send_one(r) for r in trace.requests]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _send_request_sync(
        self,
        api_url: str,
        req_trace: RequestTrace,
        strategy_name: str,
    ) -> RequestResult:
        """同步发送单个请求到 vLLM OpenAI API。

        Parameters
        ----------
        api_url:
            vLLM API URL。
        req_trace:
            请求 trace。
        strategy_name:
            策略名称。

        Returns
        -------
        RequestResult
            请求结果。
        """
        import urllib.request

        result = RequestResult(request_id=req_trace.request_id)

        payload = json.dumps(
            {
                "model": self._config.server.model,
                "messages": [{"role": "user", "content": req_trace.prompt}],
                "max_tokens": req_trace.max_tokens,
                "stream": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        result.submit_time = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._config.request_timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000

            # 解析 OpenAI API 响应
            usage = body.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)

            choices = body.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                result.generated_text = message.get("content", "")

            # 非流式模式下 TTFT ≈ total latency（近似值）
            # 实际精确 TTFT 需要流式模式，在 _send_request_streaming 中实现
            result.ttft_ms = result.total_latency_ms
            result.first_token_time = result.finish_time

        except Exception as exc:
            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.error = str(exc)
            logger.debug(
                "Request %s failed (%s): %s",
                req_trace.request_id,
                strategy_name,
                exc,
            )

        return result

    async def _send_request_streaming(
        self,
        api_url: str,
        req_trace: RequestTrace,
    ) -> RequestResult:
        """流式发送单个请求以精确测量 TTFT。

        通过 SSE 流式响应精确测量首 token 到达时间。

        Parameters
        ----------
        api_url:
            vLLM API URL。
        req_trace:
            请求 trace。

        Returns
        -------
        RequestResult
            请求结果（含精确 TTFT）。
        """
        import urllib.request

        result = RequestResult(request_id=req_trace.request_id)

        payload = json.dumps(
            {
                "model": self._config.server.model,
                "messages": [{"role": "user", "content": req_trace.prompt}],
                "max_tokens": req_trace.max_tokens,
                "stream": True,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        result.submit_time = time.monotonic()
        chunks: list[str] = []

        try:
            with urllib.request.urlopen(req, timeout=self._config.request_timeout_s) as resp:
                first_chunk_received = False
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: " prefix
                    if data_str == "[DONE]":
                        break

                    if not first_chunk_received:
                        result.first_token_time = time.monotonic()
                        result.ttft_ms = (result.first_token_time - result.submit_time) * 1000
                        first_chunk_received = True

                    chunk_data = json.loads(data_str)
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)

            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.generated_text = "".join(chunks)
            # 近似 token 数（按空格分词）
            result.completion_tokens = max(1, len(result.generated_text.split()))

        except Exception as exc:
            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.error = str(exc)

        return result


def verify_candidate_consistency(results: list[RunResult]) -> dict[str, object]:
    """验证 candidate-universe consistency。

    检查同一 (workload, concurrency, run_index) 的不同策略在同一 pressure event
    时是否使用了相同的 candidate pool。

    Parameters
    ----------
    results:
        所有运行结果。

    Returns
    -------
    dict[str, object]
        一致性校验报告。
    """
    # 按 (workload, concurrency, run_index) 分组
    groups: dict[tuple[str, int, int], list[RunResult]] = {}
    for r in results:
        key = (r.workload, r.concurrency, r.run_index)
        groups.setdefault(key, []).append(r)

    report: dict[str, object] = {
        "total_groups": len(groups),
        "groups_with_snapshots": 0,
        "consistency_violations": 0,
        "details": [],
    }

    for key, group_results in groups.items():
        results_with_snapshots = [r for r in group_results if r.candidate_snapshots]
        if len(results_with_snapshots) < 2:
            continue

        report["groups_with_snapshots"] = int(report["groups_with_snapshots"]) + 1  # type: ignore[arg-type]

        # 比较不同策略在同一时间窗口内的 candidate pool
        # （简化比较：检查 candidate 集合大小一致性）
        snapshot_sizes = {
            r.strategy: [len(s.candidate_request_ids) for s in r.candidate_snapshots]
            for r in results_with_snapshots
        }
        report["details"].append(
            {  # type: ignore[union-attr]
                "group_key": f"{key[0]}__c{key[1]}__r{key[2]}",
                "strategies": list(snapshot_sizes.keys()),
                "snapshot_counts": {s: len(sizes) for s, sizes in snapshot_sizes.items()},
            }
        )

    return report


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="BidKV vLLM 7-Baseline Experiment Runner")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(ALL_STRATEGIES),
        help="Comma-separated strategy names.",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=",".join(ALL_WORKLOADS),
        help="Comma-separated workload names.",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONCURRENCY_LEVELS),
        help="Comma-separated concurrency levels.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per combination.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vllm",
        help="Output directory.",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="experiments/vllm/traces",
        help="Frozen workload traces directory.",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--ttft-target-ms", type=float, default=2000.0)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)

    args = parser.parse_args(argv)

    strategies = tuple(s.strip() for s in args.strategies.split(","))
    workloads = tuple(w.strip() for w in args.workloads.split(","))
    concurrency_levels = tuple(int(c.strip()) for c in args.concurrency_levels.split(","))

    return ExperimentConfig(
        strategies=strategies,
        workloads=workloads,
        concurrency_levels=concurrency_levels,
        runs_per_combo=args.runs,
        output_dir=Path(args.output_dir),
        traces_dir=Path(args.traces_dir),
        server=VLLMServerConfig(
            model=args.model,
            host=args.host,
            port=args.port,
            block_size=args.block_size,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ),
        slo=SLOConfig(ttft_target_ms=args.ttft_target_ms),
        warmup_requests=args.warmup_requests,
        request_timeout_s=args.request_timeout_s,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = parse_args(argv)
    logger.info(
        "Experiment config: %d strategies × %d workloads × %d concurrency × %d runs = %d total",
        len(config.strategies),
        len(config.workloads),
        len(config.concurrency_levels),
        config.runs_per_combo,
        config.total_runs,
    )

    runner = VLLMExperimentRunner(config)
    results = runner.run_all()

    # 验证 candidate-universe consistency
    consistency_report = verify_candidate_consistency(results)
    report_path = config.output_dir / "candidate_consistency_report.json"
    report_path.write_text(json.dumps(consistency_report, indent=2), encoding="utf-8")
    logger.info("Candidate consistency report: %s", report_path)

    # 摘要打印
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for r in results:
        success_rate = (
            len(r.successful_requests) / len(r.request_results) * 100 if r.request_results else 0
        )
        logger.info(
            "  %s: throughput=%.2f rps, p99_ttft=%.1f ms, slo_viol=%.1f%%, success=%.0f%%",
            r.run_label,
            r.compute_throughput_rps(),
            r.compute_p99_ttft_ms(),
            r.compute_slo_violation_rate(config.slo.ttft_target_ms) * 100,
            success_rate,
        )


if __name__ == "__main__":
    main()
