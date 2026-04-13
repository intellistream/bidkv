"""bidkv baselines — vLLM 主评估 5 个策略，SGLang 可移植性验评1 3 个策略。

用于论文消融实验，验评 BidKV bid 机制的增量价値。
所有 baseline 通过 BaselineRegistry 注册和获取。

归因链（论文 vLLM 消融主链，参见 §6）：
  Preempt-Evict → Preempt-Evict-SJF : SJF admission ordering 的收益
  Preempt-Evict-SJF → Largest-First  : 容量贪心驱逐的收益（vs LIFO）
  Largest-First → BidKV             : U-score 多级贪心的收益

SGLang 可移植性验评（sglang_default / slack_aware / bidkv）：
  sglang_default → slack_aware → BidKV

Uniform 和 SlackAware 保留于代码库作为辅助策略，未进入 vLLM 主评估矩阵。
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
    "LargestFirstStrategy",
    "PreemptEvictStrategy",
    "PreemptEvictSJFStrategy",
    "SlackAwareStrategy",
    "StaticRandomStrategy",
    "UniformStrategy",
]
