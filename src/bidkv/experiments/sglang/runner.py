"""SGLang 可移植性验证实验运行器 — 真实 HTTP + Poisson 开环。

Usage
-----
python -m bidkv.experiments.sglang.runner \\
    --strategies "sglang_default,slack_aware,bidkv" \\
    --workloads "mixed,long_context" \\
    --runs 3 \\
    --request-rates "1.0,2.0,4.0" \\
    --model /home/cyb/Llama-3.1-8B-Instruct \\
    --output-dir results/sglang_$(date +%%Y%%m%%d)/

运行流程
--------
1. 验证 traces 已冻结（直接复用 Issue-047 的 vLLM traces）
2. 按 strategy 分组：每个 strategy 启动一次 SGLang server
   - sglang_default：原生 SGLang（无 BidKV 注入）
   - 其他策略：通过 BIDKV_STRATEGY 环境变量 + serve_entry.py 注入
3. 对当前 strategy，遍历 workload × rate × run
4. 按 Poisson 开环到达模型发送请求（无 max_inflight 限制）
5. 通过 /v1/chat/completions (streaming) 采集真实 TTFT
6. 保存 RunResult JSON
7. 停止 server，进入下一个 strategy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from bidkv.experiments.sglang.collector import (
    RequestResult,
    RunResult,
    save_run_result,
)
from bidkv.experiments.sglang.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    DEFAULT_REQUEST_RATES,
    WORKLOAD_REQUEST_RATES,
    SGLangExperimentConfig,
    SGLangServerConfig,
)
from bidkv.experiments.sglang.server import SGLangServer
from bidkv.experiments.vllm.workload import (
    RequestTrace,
    WorkloadTrace,
    load_trace,
)

logger = logging.getLogger(__name__)


class SGLangExperimentRunner:
    """SGLang 可移植性实验编排器 — Poisson 开环到达模型。

    负责 strategy × workload × rate × run 的实验矩阵运行。
    每个 strategy 启动一次 SGLang server，运行完所有组合后停止。

    Parameters
    ----------
    config:
        实验配置。
    """

    def __init__(self, config: SGLangExperimentConfig, *, resume: bool = False) -> None:
        self._config = config
        self._resume = resume
        self._server = SGLangServer(config.server)
        self._traces: dict[str, WorkloadTrace] = {}

    @property
    def config(self) -> SGLangExperimentConfig:
        return self._config

    def load_traces(self) -> None:
        """加载 Issue-047 已冻结的工作负载 traces。

        直接复用 vLLM traces，以 Issue-047 实际落盘命名为准。
        命名约定：{prefix}_rate{X.X}.json
        """
        for workload in self._config.workloads:
            rates = self._config.get_rates_for_workload(workload)
            for rate in rates:
                prefix = "mixed" if workload == "mixed" else "long"
                trace_file = f"{prefix}_rate{rate}.json"
                trace_path = self._config.traces_dir / trace_file
                key = f"{workload}__{rate}"
                self._traces[key] = load_trace(trace_path)
                logger.info(
                    "Loaded trace: %s (%d requests, rate=%.1f)",
                    trace_file,
                    self._traces[key].num_requests,
                    rate,
                )

    def run_all(self) -> list[RunResult]:
        """运行完整实验矩阵。"""
        self.load_traces()

        results: list[RunResult] = []
        total = self._config.total_runs
        current = 0

        for strategy in self._config.strategies:
            logger.info(
                "=" * 60 + "\nStarting strategy: %s\n" + "=" * 60,
                strategy,
            )
            for workload in self._config.workloads:
                rates = self._config.get_rates_for_workload(workload)
                for rate in rates:
                    for run_idx in range(self._config.runs_per_combo):
                        current += 1
                        label = self._config.run_label(
                            strategy,
                            workload,
                            rate,
                            run_idx,
                        )

                        # Resume: skip if result file already exists
                        result_path = self._config.output_dir / f"{label}.json"
                        if self._resume and result_path.exists():
                            logger.info(
                                "[%d/%d] SKIP (resume): %s",
                                current,
                                total,
                                label,
                            )
                            continue

                        # Fresh server for EVERY run to avoid KV cache
                        # exhaustion across repeats.
                        try:
                            self._server.start(
                                strategy,
                                audit_dir=str(self._config.output_dir),
                            )
                            self._server.wait_ready(self._config.server_startup_timeout_s)
                            self._warmup(strategy)
                        except RuntimeError as exc:
                            logger.error(
                                "Server error for strategy=%s run=%s: %s — skipping",
                                strategy,
                                label,
                                exc,
                            )
                            continue

                        try:
                            logger.info(
                                "[%d/%d] Running: %s",
                                current,
                                total,
                                label,
                            )

                            result = self._run_single(
                                strategy=strategy,
                                workload=workload,
                                request_rate=rate,
                                run_index=run_idx,
                            )
                            results.append(result)
                            save_run_result(result, self._config.output_dir)

                            run_status = result.server_config.get(
                                "run_status",
                                "completed",
                            )
                            logger.info(
                                "[%d/%d] Completed: %s "
                                "(throughput=%.2f rps, p95_ttft=%.1f ms,"
                                " success=%d/%d, status=%s)",
                                current,
                                total,
                                label,
                                result.compute_throughput_rps(),
                                result.compute_p95_ttft_ms(),
                                len(result.successful_requests),
                                len(result.request_results),
                                run_status,
                            )
                        finally:
                            self._server.stop()

                        # Brief cooldown between runs
                        time.sleep(3)

        logger.info(
            "All %d runs completed. Results in %s",
            len(results),
            self._config.output_dir,
        )
        return results

    def _warmup(self, strategy: str) -> None:
        """发送预热请求，不计入实验统计。"""
        import urllib.request

        api_url = f"{self._config.server.api_url}/chat/completions"
        logger.info("Sending %d warmup requests...", self._config.warmup_requests)

        for i in range(self._config.warmup_requests):
            payload = json.dumps(
                {
                    "model": self._config.server.model,
                    "messages": [{"role": "user", "content": f"Warmup {i}. Say hello."}],
                    "max_tokens": 16,
                    "stream": False,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp.read()
            except Exception as exc:
                logger.warning("Warmup request %d failed: %s", i, exc)

        logger.info("Warmup completed (strategy=%s)", strategy)

    def _run_single(
        self,
        *,
        strategy: str,
        workload: str,
        request_rate: float,
        run_index: int,
    ) -> RunResult:
        """运行单次实验 — Poisson 开环到达。"""
        label = self._config.run_label(strategy, workload, request_rate, run_index)
        trace_key = f"{workload}__{request_rate}"
        trace = self._traces[trace_key]

        result = RunResult(
            run_label=label,
            strategy=strategy,
            workload=workload,
            request_rate=request_rate,
            run_index=run_index,
            start_time=time.time(),
        )

        request_results, run_status = asyncio.run(
            self._send_workload_open_loop(trace=trace, strategy_name=strategy)
        )
        result.request_results = request_results
        result.end_time = time.time()
        result.server_config["run_status"] = run_status

        return result

    async def _send_workload_open_loop(
        self,
        *,
        trace: WorkloadTrace,
        strategy_name: str,
    ) -> tuple[list[RequestResult], str]:
        """Poisson 开环发送 — 按 arrival_time_ms 调度请求发出。

        与 vLLM runner 完全对齐：
        - 按 trace 中冻结的 arrival_time_ms 调度
        - 使用 /v1/chat/completions (streaming=True)
        - 不设 max_inflight（纯开环）
        - 失败记录到结果，不静默重试
        """
        api_url = f"{self._config.server.api_url}/chat/completions"
        loop = asyncio.get_event_loop()

        results: list[RequestResult] = []
        pending_tasks: list[asyncio.Future[RequestResult]] = []
        epoch_start = time.monotonic()
        consecutive_timeouts = 0
        aborted = False

        for req_trace in trace.requests:
            target_time = epoch_start + req_trace.arrival_time_ms / 1000.0
            now = time.monotonic()
            if target_time > now:
                await asyncio.sleep(target_time - now)

            if consecutive_timeouts >= self._config.consecutive_timeout_abort:
                logger.warning(
                    "ABORT: %d consecutive timeouts for strategy=%s, stopping run",
                    consecutive_timeouts,
                    strategy_name,
                )
                aborted = True
                break

            future = loop.run_in_executor(
                None,
                self._send_request_streaming_sync,
                api_url,
                req_trace,
                strategy_name,
            )
            pending_tasks.append(future)

        if pending_tasks:
            done_results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for r in done_results:
                if isinstance(r, RequestResult):
                    results.append(r)
                    if r.error and "timeout" in r.error.lower():
                        consecutive_timeouts += 1
                    else:
                        consecutive_timeouts = 0
                elif isinstance(r, BaseException):
                    logger.error("Unexpected exception in request task: %s", r)

        run_status = "aborted" if aborted else "completed"
        return results, run_status

    def _send_request_streaming_sync(
        self,
        api_url: str,
        req_trace: RequestTrace,
        _strategy_name: str,
    ) -> RequestResult:
        """发送单个 streaming 请求并采集精确 TTFT。

        使用 /v1/chat/completions (streaming=True)，解析 SSE 事件流，
        记录首 token 到达时间作为真实 TTFT。
        """
        import http.client
        import urllib.parse

        result = RequestResult(
            request_id=req_trace.request_id,
            submit_time=time.monotonic(),
        )

        payload = json.dumps(
            {
                "model": self._config.server.model,
                "messages": [{"role": "user", "content": req_trace.prompt}],
                "max_tokens": req_trace.max_tokens,
                "stream": True,
            }
        ).encode("utf-8")

        parsed = urllib.parse.urlparse(api_url)
        timeout = self._config.request_timeout_s

        try:
            conn = http.client.HTTPConnection(
                parsed.hostname or "127.0.0.1",
                parsed.port or 80,
                timeout=timeout,
            )
            conn.request(
                "POST",
                parsed.path,
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()

            if resp.status != 200:
                body = resp.read().decode("utf-8", errors="replace")
                result.error = f"HTTP {resp.status}: {body[:500]}"
                result.finish_time = time.monotonic()
                result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
                return result

            # Parse SSE stream for TTFT
            first_token_received = False
            generated_tokens = 0
            generated_text_parts: list[str] = []

            while True:
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()

                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content and not first_token_received:
                    result.first_token_time = time.monotonic()
                    result.ttft_ms = (result.first_token_time - result.submit_time) * 1000
                    first_token_received = True

                if content:
                    generated_tokens += 1
                    generated_text_parts.append(content)

            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.completion_tokens = generated_tokens
            result.generated_text = "".join(generated_text_parts)

            conn.close()

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000

        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGLang portability experiment runner",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(ALL_STRATEGIES),
        help="Comma-separated strategy names",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=",".join(ALL_WORKLOADS),
        help="Comma-separated workload names",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per (strategy, workload, rate) combo",
    )
    parser.add_argument(
        "--request-rates",
        type=str,
        default=",".join(str(r) for r in DEFAULT_REQUEST_RATES),
        help="Comma-separated fallback rates (req/s). "
        "Use --mixed-rates / --long-rates for per-workload.",
    )
    parser.add_argument(
        "--mixed-rates",
        type=str,
        default=None,
        help="Comma-separated rates for mixed workload (overrides --request-rates).",
    )
    parser.add_argument(
        "--long-rates",
        type=str,
        default=None,
        help="Comma-separated rates for long_context workload (overrides --request-rates).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/cyb/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sglang",
        help="Output directory for results",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="experiments/vllm/traces",
        help="Directory containing frozen traces (shared with vLLM)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="SGLang server port",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip runs whose result file already exists (resume after crash).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(","))
    workloads = tuple(w.strip() for w in args.workloads.split(","))
    rates = tuple(float(r) for r in args.request_rates.split(","))

    # Build per-workload rates: CLI overrides > WORKLOAD_REQUEST_RATES > fallback
    workload_rates = dict(WORKLOAD_REQUEST_RATES)
    if args.mixed_rates is not None:
        workload_rates["mixed"] = tuple(float(r) for r in args.mixed_rates.split(","))
    if args.long_rates is not None:
        workload_rates["long_context"] = tuple(float(r) for r in args.long_rates.split(","))

    config = SGLangExperimentConfig(
        strategies=strategies,
        workloads=workloads,
        request_rates=rates,
        workload_rates=workload_rates,
        runs_per_combo=args.runs,
        output_dir=Path(args.output_dir),
        server=SGLangServerConfig(
            model=args.model,
            port=args.port,
        ),
        traces_dir=Path(args.traces_dir),
    )

    logger.info(
        "SGLang experiment: %d strategies × %d workloads × %d rates × %d runs = %d total",
        len(strategies),
        len(workloads),
        len(rates),
        args.runs,
        config.total_runs,
    )

    runner = SGLangExperimentRunner(config, resume=getattr(args, "resume", False))
    results = runner.run_all()

    logger.info("Experiment complete: %d runs", len(results))


if __name__ == "__main__":
    main()
