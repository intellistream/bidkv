"""bidkv.scoring — Token 重要度评分策略层。

评分三级分类
------------
- **Practical Scoring**：H2OScoring — 生产部署中实际使用的评分代理
- **Reference Scoring**：AttentionWeightScoring — 精度上界参考
- **Auxiliary Scoring**：UniformScoring / RandomScoring — 消融实验用基线
"""

from __future__ import annotations

from bidkv.scoring.attention import AttentionWeightScoring
from bidkv.scoring.base import ScoringStrategy
from bidkv.scoring.bid_builder import build_bids
from bidkv.scoring.h2o import H2OScoring
from bidkv.scoring.random_score import RandomScoring
from bidkv.scoring.uniform import UniformScoring

__all__ = [
    "AttentionWeightScoring",
    "H2OScoring",
    "RandomScoring",
    "ScoringStrategy",
    "UniformScoring",
    "build_bids",
]
