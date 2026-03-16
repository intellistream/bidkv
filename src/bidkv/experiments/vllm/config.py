"""Experiment configuration for vLLM 7-baseline experiment.

所有实验参数集中管理，确保 reproducibility。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 策略名称常量 — 与 BaselineRegistry.name 对齐
STRATEGY_PREEMPT_EVICT = "preempt-evict"
STRATEGY_STATIC_RANDOM = "static-random"
STRATEGY_H2O_STYLE = "h2o-style"
STRATEGY_UNIFORM = "uniform"
STRATEGY_GLOBAL_NOBID = "global-nobid"
STRATEGY_SLACK_AWARE = "slack-aware"
STRATEGY_BIDKV = "bidkv"
STRATEGY_ORACLE_DP = "oracle-dp"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_PREEMPT_EVICT,
    STRATEGY_STATIC_RANDOM,
    STRATEGY_H2O_STYLE,
    STRATEGY_UNIFORM,
    STRATEGY_GLOBAL_NOBID,
    STRATEGY_SLACK_AWARE,
    STRATEGY_BIDKV,
    STRATEGY_ORACLE_DP,
)

# 工作负载常量
WORKLOAD_CHAT = "chat"
WORKLOAD_SUMMARIZATION = "summarization"
WORKLOAD_QA = "qa"

ALL_WORKLOADS: tuple[str, ...] = (
    WORKLOAD_CHAT,
    WORKLOAD_SUMMARIZATION,
    WORKLOAD_QA,
)

DEFAULT_CONCURRENCY_LEVELS: tuple[int, ...] = (8, 16, 32)
DEFAULT_RUNS_PER_COMBO = 3


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM 服务端配置。

    Attributes
    ----------
    model:
        HuggingFace 模型名称或路径。
    block_size:
        KV cache block size（token 数/block）。
    max_num_seqs:
        最大并发请求数。
    gpu_memory_utilization:
        GPU 显存利用率上限。
    enforce_eager:
        是否禁用 CUDA graph（便于 H2O hook 工作）。
    host:
        服务监听地址。
    port:
        服务监听端口。
    """

    model: str = "meta-llama/Llama-2-7b-chat-hf"
    block_size: int = 16
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    def to_cli_args(self) -> list[str]:
        """生成 vllm serve 的命令行参数。"""
        args = [
            "--model",
            self.model,
            "--block-size",
            str(self.block_size),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        return args


@dataclass(frozen=True)
class SLOConfig:
    """SLO 配置。

    Attributes
    ----------
    ttft_target_ms:
        Time-To-First-Token SLO 目标（ms）。
    tpot_target_ms:
        Time-Per-Output-Token SLO 目标（ms）。
    """

    ttft_target_ms: float = 2000.0
    tpot_target_ms: float = 100.0


@dataclass(frozen=True)
class ExperimentConfig:
    """完整实验配置。

    Attributes
    ----------
    strategies:
        要运行的策略名称列表。
    workloads:
        要运行的工作负载列表。
    concurrency_levels:
        并发度列表。
    runs_per_combo:
        每个 (策略, 工作负载, 并发度) 组合的独立运行次数。
    output_dir:
        结果输出目录。
    server:
        vLLM 服务端配置。
    slo:
        SLO 配置。
    traces_dir:
        冻结的工作负载 trace 目录。
    warmup_requests:
        预热请求数（不计入统计）。
    request_timeout_s:
        单个请求超时时间（秒）。
    server_startup_timeout_s:
        vLLM 服务启动超时（秒）。
    collect_candidate_snapshots:
        是否收集每个 pressure event 的 candidate pool snapshot
        （用于 candidate-universe consistency 验证）。
    """

    strategies: tuple[str, ...] = ALL_STRATEGIES
    workloads: tuple[str, ...] = ALL_WORKLOADS
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY_LEVELS
    runs_per_combo: int = DEFAULT_RUNS_PER_COMBO
    output_dir: Path = Path("results/vllm")
    server: VLLMServerConfig = field(default_factory=VLLMServerConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    traces_dir: Path = Path("experiments/vllm/traces")
    warmup_requests: int = 5
    request_timeout_s: float = 120.0
    server_startup_timeout_s: float = 300.0
    collect_candidate_snapshots: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.strategies) - set(ALL_STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}. Valid: {ALL_STRATEGIES}")
        unknown_wl = set(self.workloads) - set(ALL_WORKLOADS)
        if unknown_wl:
            raise ValueError(f"Unknown workloads: {unknown_wl}. Valid: {ALL_WORKLOADS}")
        if self.runs_per_combo < 1:
            raise ValueError(f"runs_per_combo must be >= 1, got {self.runs_per_combo}")

    @property
    def total_runs(self) -> int:
        """总实验运行次数。"""
        return (
            len(self.strategies)
            * len(self.workloads)
            * len(self.concurrency_levels)
            * self.runs_per_combo
        )

    def run_label(self, strategy: str, workload: str, concurrency: int, run_idx: int) -> str:
        """生成单次实验运行的标识符。"""
        return f"{strategy}__{workload}__c{concurrency}__r{run_idx}"
