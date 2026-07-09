"""bidkv.experiments.sglang — SGLang 可移植性验证实验（#048）。

证明 BidKV 的 framework-portability：
SGLang (RadixAttention, tree-based KV) 上 BidKV 的改进趋势
与 vLLM (PagedAttention, flat block) 定性一致。

v2.3 SGLang 使用 3 个策略：
- SGLang-Default (= Preempt-Evict)
- Slack-Aware（强无-bid 系统对手）
- BidKV

bid 归因已由 vLLM 主实验承担，仅需证明 directional consistency。
"""

from __future__ import annotations

from bidkv.experiments.sglang.analysis import CrossFrameworkAnalyzer
from bidkv.experiments.sglang.collector import RequestResult, RunResult, save_run_result
from bidkv.experiments.sglang.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    SGLangExperimentConfig,
    SGLangServerConfig,
    SLOConfig,
)
from bidkv.experiments.sglang.runner import SGLangExperimentRunner
from bidkv.experiments.sglang.server import SGLangServer
from bidkv.experiments.sglang.strategies import (
    SGLANG_STRATEGIES,
    SGLangStrategyConfig,
    create_sglang_strategy_configs,
)

__all__ = [
    "ALL_STRATEGIES",
    "ALL_WORKLOADS",
    "CrossFrameworkAnalyzer",
    "RequestResult",
    "RunResult",
    "SGLangExperimentConfig",
    "SGLangExperimentRunner",
    "SGLangServer",
    "SGLangServerConfig",
    "SGLangStrategyConfig",
    "SGLANG_STRATEGIES",
    "SLOConfig",
    "create_sglang_strategy_configs",
    "save_run_result",
]
