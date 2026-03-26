"""bidkv baselines — 7 个 baseline 策略。

用于论文消融实验，验证 BidKV bid 机制的增量价值。
所有 baseline 通过 BaselineRegistry 注册和获取。

归因关系（论文消融核心）：
  Preempt-Evict → H2O-Style  :  token-level scoring 的收益
  H2O-Style → Global-NoBid   :  request-level utility 推断的收益
  Global-NoBid → BidKV        :  bid 接口（用户显式偏好）的收益
  Uniform → BidKV             :  差异化压缩 vs 均等压缩
  Slack-Aware → BidKV         :  quality-aware vs SLO-only
"""

from bidkv.baselines.base import (
    BaselineContext,
    BaselineStrategy,
    CompressionAction,
    RequestState,
)
from bidkv.baselines.bidkv_strategy import BidKVStrategy
from bidkv.baselines.global_nobid import GlobalNoBidStrategy
from bidkv.baselines.h2o_style import H2OStyleStrategy
from bidkv.baselines.preempt_evict import PreemptEvictStrategy
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
    "GlobalNoBidStrategy",
    "H2OStyleStrategy",
    "PreemptEvictStrategy",
    "SlackAwareStrategy",
    "StaticRandomStrategy",
    "UniformStrategy",
]
