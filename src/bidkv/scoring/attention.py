"""Full Attention Weight Aggregate 评分策略。

**Reference Scoring** — 精度上界参考（需框架支持 ``output_attentions=True``）。

原理
----
利用完整的注意力权重矩阵计算每个 KV token 的重要度。对多层、多头的
注意力权重做聚合（mean 或 weighted mean），得到每个 token 的全局重要度分数。

这是最准确的评分方式，但在生产环境中不可用（FlashAttention 不提供
``output_attentions``）。主要用于：
1. 作为 H2OScoring 的精度上界参考
2. Calibration 实验的 reference baseline
3. 论文中的 oracle/reference 对比
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.bid_builder import build_bids


class AttentionWeightScoring:
    """基于完整注意力权重的 token 重要度评分。

    Parameters
    ----------
    aggregation:
        多层注意力聚合方式：
        - ``"mean"``：所有层和头的平均值（默认）
        - ``"max"``：所有层和头的最大值
        - ``"last_layer"``：仅使用最后一层的平均值
    algorithm_id:
        算法标识符，用于 CompressionBid 中。默认 "attention_weight"。

    Notes
    -----
    此策略仅在支持 ``output_attentions=True`` 的框架中可用
    （如 HuggingFace eager attention）。FlashAttention 不支持此功能。
    """

    def __init__(
        self,
        *,
        aggregation: str = "mean",
        algorithm_id: str = "attention_weight",
    ) -> None:
        valid_aggregations = {"mean", "max", "last_layer"}
        if aggregation not in valid_aggregations:
            raise ValueError(
                f"aggregation must be one of {valid_aggregations}, got {aggregation!r}"
            )
        self._aggregation = aggregation
        self._algorithm_id = algorithm_id
        # 存储最新的注意力权重聚合结果（per-token 分数）
        self._latest_scores: list[float] = []

    @property
    def aggregation(self) -> str:
        """聚合方式。"""
        return self._aggregation

    def update_attention_weights(
        self,
        attention_weights: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        """更新完整注意力权重。

        Parameters
        ----------
        attention_weights:
            三维结构：``[layer][head][token_position]``。
            每个值表示对应 KV token position 从所有 query position 聚合后的注意力权重。

            具体含义：对于每一层的每个头，提供长度为 seq_len 的注意力分布。
            这些通常是从 attention matrix 的列（KV 维）做 row-wise mean 得到的。
        """
        if not attention_weights or not attention_weights[0]:
            self._latest_scores = []
            return

        num_layers = len(attention_weights)
        num_heads = len(attention_weights[0])
        seq_len = len(attention_weights[0][0])

        if self._aggregation == "last_layer":
            # 仅使用最后一层
            last_layer = attention_weights[-1]
            aggregated = [0.0] * seq_len
            for head in last_layer:
                for i, w in enumerate(head):
                    aggregated[i] += w
            # 平均所有 head
            aggregated = [a / num_heads for a in aggregated]
        elif self._aggregation == "max":
            # 所有层和头的最大值
            aggregated = [0.0] * seq_len
            for layer in attention_weights:
                for head in layer:
                    for i, w in enumerate(head):
                        aggregated[i] = max(aggregated[i], w)
        else:
            # mean: 所有层和头的平均值
            aggregated = [0.0] * seq_len
            total_count = num_layers * num_heads
            for layer in attention_weights:
                for head in layer:
                    for i, w in enumerate(head):
                        aggregated[i] += w
            aggregated = [a / total_count for a in aggregated]

        # 归一化到 [0, 1]
        max_val = max(aggregated) if aggregated else 0.0
        if max_val > 0:
            self._latest_scores = [a / max_val for a in aggregated]
        else:
            self._latest_scores = [0.0] * seq_len

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
            可选上下文。支持：
            - ``attention_weights``: Sequence[Sequence[Sequence[float]]] —
              如果提供，先调用 ``update_attention_weights()`` 再评分。

        Returns
        -------
        list[float]
            每个 token 的重要度，值域 [0, 1]。
        """
        n = len(token_ids)
        if n == 0:
            return []

        if "attention_weights" in context:
            self.update_attention_weights(context["attention_weights"])

        if self._latest_scores and len(self._latest_scores) >= n:
            return list(self._latest_scores[:n])

        # 无注意力数据时，返回均匀分数（不做位置启发式，与 H2O 区分）
        return [0.5] * n

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于注意力权重评分，生成 CompressionBid。

        Parameters
        ----------
        request_id:
            推理请求 ID。
        token_ids:
            Token ID 序列。
        compression_levels:
            压缩比例列表（0~1）。
        **context:
            传递给 ``score()`` 的上下文。
        """
        n = len(token_ids)
        if n == 0:
            return []

        scores = self.score(token_ids, **context)

        has_real_weights = self._latest_scores and len(self._latest_scores) >= n

        return build_bids(
            request_id=request_id,
            token_ids=token_ids,
            scores=scores,
            compression_levels=compression_levels,
            algorithm_id=self._algorithm_id,
            confidence_fn=lambda: 0.9 if has_real_weights else 0.3,
            extra_metadata={
                "aggregation": self._aggregation,
                "has_real_weights": has_real_weights,
                "scoring_method": "full_attention_aggregate",
            },
        )

    def reset(self) -> None:
        """重置存储的注意力权重。"""
        self._latest_scores.clear()
