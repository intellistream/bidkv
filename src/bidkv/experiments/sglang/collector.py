"""SGLang 请求发送与指标收集。

Poisson 开环到达模型 + /v1/chat/completions (streaming) 指标采集。
与 Issue-047 vLLM runner 完全对齐：
- 相同 Poisson 到达机制（按 arrival_time_ms 调度）
- 相同 streaming TTFT 采集（first SSE chunk timestamp）
- 相同 open-loop 语义（不设 max_inflight）
- 失败记入结果、不静默重试
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """单个请求的完整指标。"""

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
        if self.completion_tokens <= 1:
            return 0.0
        decode_time = self.total_latency_ms - self.ttft_ms
        return decode_time / (self.completion_tokens - 1) if decode_time > 0 else 0.0


@dataclass
class RunResult:
    """单次实验 run 的完整结果。"""

    run_label: str
    strategy: str
    workload: str
    request_rate: float
    run_index: int
    request_results: list[RequestResult] = field(default_factory=list)
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
        if self.duration_s <= 0:
            return 0.0
        return len(self.successful_requests) / self.duration_s

    def _percentile(self, values: list[float], pct: float) -> float:
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
        return self._percentile(self._ttft_values(), 0.50)

    def compute_p95_ttft_ms(self) -> float:
        return self._percentile(self._ttft_values(), 0.95)

    def compute_p99_ttft_ms(self) -> float:
        return self._percentile(self._ttft_values(), 0.99)

    def compute_p50_tpot_ms(self) -> float:
        return self._percentile(self._tpot_values(), 0.50)

    def compute_p99_tpot_ms(self) -> float:
        return self._percentile(self._tpot_values(), 0.99)

    def compute_p50_e2e_latency_ms(self) -> float:
        return self._percentile(self._e2e_latency_values(), 0.50)

    def compute_p99_e2e_latency_ms(self) -> float:
        return self._percentile(self._e2e_latency_values(), 0.99)

    def compute_normalized_latency_ms(self) -> float:
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

    def compute_completion_rate(self) -> float:
        if not self.request_results:
            return 0.0
        return len(self.successful_requests) / len(self.request_results)


def save_run_result(result: RunResult, output_dir: Path) -> Path:
    """将 RunResult 保存为 JSON。"""
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
        "summary": {
            "total_requests": len(result.request_results),
            "successful_requests": len(result.successful_requests),
            "failed_requests": len(result.failed_requests),
            "completion_rate": result.compute_completion_rate(),
            "throughput_rps": result.compute_throughput_rps(),
            "ttft_ms_p50": result.compute_p50_ttft_ms(),
            "ttft_ms_p95": result.compute_p95_ttft_ms(),
            "ttft_ms_p99": result.compute_p99_ttft_ms(),
            "tpot_ms_p50": result.compute_p50_tpot_ms(),
            "tpot_ms_p99": result.compute_p99_tpot_ms(),
            "e2e_latency_ms_p50": result.compute_p50_e2e_latency_ms(),
            "e2e_latency_ms_p99": result.compute_p99_e2e_latency_ms(),
            "normalized_latency_ms_per_token": result.compute_normalized_latency_ms(),
        },
        "request_results": [asdict(r) for r in result.request_results],
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved run result: %s -> %s", result.run_label, path)
    return path


def write_audit_entry(
    audit_path: Path,
    *,
    candidate_count: int,
    candidate_request_ids: list[str],
    strategy: str,
    kv_usage_pct: float,
) -> None:
    """写入一条 fairness audit 日志。

    Args:
        audit_path: JSONL 审计日志文件路径。
        candidate_count: 候选请求数量。
        candidate_request_ids: 候选请求 ID 列表。
        strategy: 当前策略名称。
        kv_usage_pct: KV 缓存使用率（0.0 ~ 1.0）。
    """
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "event": "pressure_trigger",
        "candidate_count": candidate_count,
        "candidate_list_hash": hashlib.md5(  # noqa: S324
            ",".join(sorted(candidate_request_ids)).encode()
        ).hexdigest(),
        "strategy": strategy,
        "kv_usage_pct": kv_usage_pct,
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
