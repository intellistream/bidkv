"""ScoringStrategy — Token 重要度评分策略 Protocol。

定义 BidKV scoring 层的核心抽象接口。所有评分策略必须满足此 Protocol，
生产部署、参考基线和消融实验均通过可插拔的 ScoringStrategy 实现区分。

评分三级分类
------------
- **Practical Scoring**：H2OScoring — 生产部署中实际使用的评分代理
- **Reference Scoring**：AttentionWeightScoring — 精度上界参考（需 output_attentions）
- **Auxiliary Scoring**：UniformScoring / RandomScoring — 消融实验用基线

score-only 契约
---------------
新增评分策略只需实现 ``score(token_ids, **context) -> list[float]``。
统一的 bids 生成由 ``bidkv.scoring.bid_builder.build_bids()`` 负责，
评分策略不再需要实现 ``generate_bids``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScoringStrategy(Protocol):
    """Token 重要度评分策略（score-only 契约）。

    任何实现 ``score()`` 方法签名的对象，
    都可作为 ScoringStrategy 使用（structural subtyping，无需继承）。

    bids 生成统一由 ``bidkv.scoring.bid_builder.build_bids()`` 负责，
    scorer 无需实现 ``generate_bids``。
    """

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。
        **context:
            策略特定的上下文信息（例如 attention 权重、decode step 统计等）。

        Returns
        -------
        list[float]
            长度与 ``token_ids`` 相同的分数列表，值域 [0, 1]，越高越重要。
        """
        ...
