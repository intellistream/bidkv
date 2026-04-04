"""bidkv baselines — 7 个 baseline 策略。

用于论文消融实验，验证 BidKV bid 机制的增量价值。
所有 baseline 通过 BaselineRegistry 注册和获取。

归因关系（论文消融核心）：
  Preempt-Evict → Preempt-Evict-SJF :  SJF admission ordering 的收益
  Preempt-Evict-SJF → Largest-First :  容量贪心驱逐的收益（vs LIFO）
  Largest-First → BidKV             :  U-score 多级贪心的收益
  Uniform → BidKV                    :  差异化压缩 vs 均等压缩
  Slack-Aware → BidKV                :  quality-aware vs SLO-only
"""

from bidkv.baselines.base import (
    BaselineContext,
    BaselineStrategy,
    CompressionAction,
    RequestState,
)
from bidkv.baselines.bidkv_strategy import BidKVStrategy
from bidkv.baselines.largest_first import LargestFirstStrategy
from bidkv.baselines.preempt_evict import PreemptEvictStrategy
from bidkv.baselines.preempt_evict_sjf import PreemptEvictSJFStrategy
from bidkv.baselines.registry import BaselineRegistry
from bidkv.baselines.slack_aware import SlackAwareStrategy
from bidkv.baselines.static_random import StaticRandomStrategy
from bidkv.baselines.uniform import UniformStrategy

# Backward compatibility alias
H2OStyleStrategy = LargestFirstStrategy

__all__ = [
    # Types
    "BaselineContext",
    "BaselineStrategy",
    "CompressionAction",
    "RequestState",
    # Registry
    "BaselineRegistry",
    # Strategies
    "BidKVStrategy",
    "H2OStyleStrategy",  # backward compat alias
    "LargestFirstStrategy",
    "PreemptEvictStrategy",
    "PreemptEvictSJFStrategy",
    "SlackAwareStrategy",
    "StaticRandomStrategy",
    "UniformStrategy",
]
