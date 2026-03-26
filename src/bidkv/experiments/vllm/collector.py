"""Runtime metrics collector for vLLM experiments.

从 vLLM Prometheus /metrics endpoint 和 BidKV adapter 内部状态收集指标。
所有数据输出为 structured records 便于后续 analysis 模块消费。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """单个推理请求的完整指标记录。

    Attributes
    ----------
    request_id:
        请求标识。
    prompt_tokens:
        输入 token 数。
    completion_tokens:
        输出 token 数。
    ttft_ms:
        Time-To-First-Token（毫秒）。
    total_latency_ms:
        端到端延迟（毫秒）。
    generated_text:
        生成的文本内容。
    error:
        错误描述（成功时为空字符串）。
    submit_time:
        请求提交时间戳（单调 monotonic, 秒）。
    first_token_time:
        首 token 到达时间戳。
    finish_time:
        请求完成时间戳。
    """

    request_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    generated_text: str = ""
    error: str = ""
    submit_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0

    @property
    def success(self) -> bool:
        return not self.error

    @property
    def tpot_ms(self) -> float:
        """Time-Per-Output-Token（毫秒）。"""
        if self.completion_tokens <= 1:
            return 0.0
        decode_time = self.total_latency_ms - self.ttft_ms
        return decode_time / (self.completion_tokens - 1) if decode_time > 0 else 0.0


@dataclass
class CandidateSnapshot:
    """单次 pressure event 的 candidate pool 快照。

    用于 candidate-universe consistency 验证。

    Attributes
    ----------
    timestamp:
        事件时间戳。
    pressure_ratio:
        当前 KV 利用率。
    candidate_request_ids:
        候选请求 ID 列表（排序后）。
    needed_tokens:
        需要释放的 token 数。
    strategy_name:
        当前使用的策略名称。
    """

    timestamp: float
    pressure_ratio: float
    candidate_request_ids: list[str]
    needed_tokens: int
    strategy_name: str


@dataclass
class RunResult:
    """单次实验运行的完整结果。

    Attributes
    ----------
    run_label:
        运行标识符（strategy__workload__rateX__rM）。
    strategy:
        策略名称。
    workload:
        工作负载名称。
    request_rate:
        请求到达速率 (req/s)。
    run_index:
        运行序号。
    request_results:
        所有请求的指标。
    candidate_snapshots:
        pressure event 的 candidate pool 快照。
    adapter_metrics:
        BidKV adapter 的内部指标。
    start_time:
        实验开始时间（epoch 秒）。
    end_time:
        实验结束时间（epoch 秒）。
    server_config:
        vLLM 服务端配置摘要。
    """

    run_label: str
    strategy: str
    workload: str
    request_rate: float
    run_index: int
    request_results: list[RequestResult] = field(default_factory=list)
    candidate_snapshots: list[CandidateSnapshot] = field(default_factory=list)
    adapter_metrics: dict[str, int] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    server_config: dict[str, object] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0

    @property
    def successful_requests(self) -> list[RequestResult]:
        return [r for r in self.request_results if r.success]

    @property
    def failed_requests(self) -> list[RequestResult]:
        return [r for r in self.request_results if not r.success]

    def compute_throughput_rps(self) -> float:
        """计算吞吐量 (successful requests / second)。"""
        if self.duration_s <= 0:
            return 0.0
        return len(self.successful_requests) / self.duration_s

    def _percentile(self, values: list[float], pct: float) -> float:
        """计算百分位数。"""
        if not values:
            return 0.0
        s = sorted(values)
        idx = max(0, int(len(s) * pct) - 1)
        return s[idx]

    def _ttft_values(self) -> list[float]:
        return [r.ttft_ms for r in self.successful_requests if r.ttft_ms > 0]

    def _tpot_values(self) -> list[float]:
        return [r.tpot_ms for r in self.successful_requests if r.tpot_ms > 0]

    def _e2e_latency_values(self) -> list[float]:
        return [r.total_latency_ms for r in self.successful_requests if r.total_latency_ms > 0]

    def compute_p50_ttft_ms(self) -> float:
        """计算 P50 TTFT（毫秒）。"""
        return self._percentile(self._ttft_values(), 0.50)

    def compute_p99_ttft_ms(self) -> float:
        """计算 P99 TTFT（毫秒）。"""
        return self._percentile(self._ttft_values(), 0.99)

    def compute_p50_tpot_ms(self) -> float:
        """计算 P50 TPOT（毫秒）。"""
        return self._percentile(self._tpot_values(), 0.50)

    def compute_p99_tpot_ms(self) -> float:
        """计算 P99 TPOT（毫秒）。"""
        return self._percentile(self._tpot_values(), 0.99)

    def compute_p50_e2e_latency_ms(self) -> float:
        """计算 P50 端到端延迟（毫秒）。"""
        return self._percentile(self._e2e_latency_values(), 0.50)

    def compute_p99_e2e_latency_ms(self) -> float:
        """计算 P99 端到端延迟（毫秒）。"""
        return self._percentile(self._e2e_latency_values(), 0.99)

    def compute_normalized_latency_ms(self) -> float:
        """计算 Normalized Latency (ms/token) = mean(E2E) / mean(output_tokens)。"""
        reqs = [
            r
            for r in self.successful_requests
            if r.total_latency_ms > 0 and r.completion_tokens > 0
        ]
        if not reqs:
            return 0.0
        mean_e2e = sum(r.total_latency_ms for r in reqs) / len(reqs)
        mean_tokens = sum(r.completion_tokens for r in reqs) / len(reqs)
        return mean_e2e / mean_tokens if mean_tokens > 0 else 0.0

    def compute_slo_attainment_rate(
        self, ttft_target_ms: float, tpot_target_ms: float = 0.0
    ) -> float:
        """计算 SLO attainment rate。

        SLO 定义：TTFT < ttft_target AND TPOT < tpot_target（当 tpot_target > 0 时）。

        Parameters
        ----------
        ttft_target_ms:
            TTFT SLO 阈值（ms）。
        tpot_target_ms:
            TPOT SLO 阈值（ms）。0 表示不检查 TPOT。

        Returns
        -------
        float
            满足 SLO 的请求占成功请求的比例 [0, 1]（越高越好）。
        """
        successful = self.successful_requests
        if not successful:
            return 0.0
        check_tpot = tpot_target_ms > 0
        attained = sum(
            1
            for r in successful
            if r.ttft_ms <= ttft_target_ms and (not check_tpot or r.tpot_ms <= tpot_target_ms)
        )
        return attained / len(successful)


def save_run_result(result: RunResult, output_dir: Path) -> Path:
    """将单次运行结果保存为 JSON。

    Parameters
    ----------
    result:
        运行结果。
    output_dir:
        输出目录。

    Returns
    -------
    Path
        输出文件路径。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.run_label}.json"
    path = output_dir / filename

    data = {
        "run_label": result.run_label,
        "strategy": result.strategy,
        "workload": result.workload,
        "request_rate": result.request_rate,
        "run_index": result.run_index,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "duration_s": result.duration_s,
        "server_config": result.server_config,
        "adapter_metrics": result.adapter_metrics,
        "summary": {
            "total_requests": len(result.request_results),
            "successful_requests": len(result.successful_requests),
            "failed_requests": len(result.failed_requests),
            "throughput_rps": result.compute_throughput_rps(),
            "ttft_ms_p50": result.compute_p50_ttft_ms(),
            "ttft_ms_p99": result.compute_p99_ttft_ms(),
            "tpot_ms_p50": result.compute_p50_tpot_ms(),
            "tpot_ms_p99": result.compute_p99_tpot_ms(),
            "e2e_latency_ms_p50": result.compute_p50_e2e_latency_ms(),
            "e2e_latency_ms_p99": result.compute_p99_e2e_latency_ms(),
            "normalized_latency_ms_per_token": result.compute_normalized_latency_ms(),
        },
        "request_results": [asdict(r) for r in result.request_results],
        "candidate_snapshots": [asdict(s) for s in result.candidate_snapshots],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved run result: %s -> %s", result.run_label, path)
    return path


def load_run_result(path: Path) -> RunResult:
    """从 JSON 文件加载运行结果。

    Parameters
    ----------
    path:
        JSON 文件路径。

    Returns
    -------
    RunResult
        加载的运行结果。
    """
    if not path.exists():
        raise FileNotFoundError(f"Run result not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    request_results = [RequestResult(**r) for r in data.get("request_results", [])]
    candidate_snapshots = [CandidateSnapshot(**s) for s in data.get("candidate_snapshots", [])]

    return RunResult(
        run_label=data["run_label"],
        strategy=data["strategy"],
        workload=data["workload"],
        request_rate=data.get("request_rate", data.get("concurrency", 0.0)),
        run_index=data["run_index"],
        request_results=request_results,
        candidate_snapshots=candidate_snapshots,
        adapter_metrics=data.get("adapter_metrics", {}),
        start_time=data.get("start_time", 0.0),
        end_time=data.get("end_time", 0.0),
        server_config=data.get("server_config", {}),
    )


def load_all_run_results(output_dir: Path) -> list[RunResult]:
    """加载目录下所有运行结果。

    Parameters
    ----------
    output_dir:
        包含 JSON 结果文件的目录。

    Returns
    -------
    list[RunResult]
        所有运行结果列表。
    """
    if not output_dir.exists():
        return []

    results: list[RunResult] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            results.append(load_run_result(path))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Skipping malformed result file %s: %s", path, e)
    return results


class MetricsCollector:
    """vLLM 运行时指标收集器。

    在实验运行期间收集 vLLM /metrics endpoint 和 adapter 内部指标。
    支持定期快照和 candidate pool snapshot。
    """

    def __init__(self, base_url: str, *, poll_interval_s: float = 1.0) -> None:
        self._base_url = base_url
        self._poll_interval_s = poll_interval_s
        self._snapshots: list[dict[str, object]] = []

    @property
    def metrics_url(self) -> str:
        return f"{self._base_url}/metrics"

    def collect_prometheus_metrics(self) -> dict[str, float]:
        """从 vLLM /metrics endpoint 获取 Prometheus 指标。

        Returns
        -------
        dict[str, float]
            指标名称 → 值的字典。

        Raises
        ------
        ConnectionError
            无法连接到 vLLM 服务。
        """
        import urllib.request

        try:
            with urllib.request.urlopen(self.metrics_url, timeout=5) as resp:
                body = resp.read().decode("utf-8")
        except Exception as exc:
            raise ConnectionError(f"Failed to fetch metrics from {self.metrics_url}") from exc

        return self._parse_prometheus_text(body)

    @staticmethod
    def _parse_prometheus_text(text: str) -> dict[str, float]:
        """解析 Prometheus text 格式的指标。"""
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    continue
        return metrics

    def take_snapshot(self) -> dict[str, object]:
        """采集一次快照并保存。"""
        snapshot: dict[str, object] = {"timestamp": time.time()}
        try:
            prom_metrics = self.collect_prometheus_metrics()
            snapshot["prometheus"] = prom_metrics
        except ConnectionError:
            snapshot["prometheus"] = {}
            snapshot["error"] = "metrics_endpoint_unavailable"
        self._snapshots.append(snapshot)
        return snapshot

    @property
    def snapshots(self) -> list[dict[str, object]]:
        return list(self._snapshots)

    def reset(self) -> None:
        self._snapshots.clear()
