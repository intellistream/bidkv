"""位置启发式评分策略。

在没有真实 attention 数据的情况下（vLLM/SGLang 的 FlashAttention
不暴露 output_attentions），使用位置启发式作为 token 重要度代理：

- **Attention Sink**：序列开头 token 有较高重要度（Xiao et al., 2023）
- **Recency**：序列末尾 token 有较高重要度

如果外部通过 ``update_from_decode_step()`` 提供了注意力数据，
则切换为 heavy-hitter + recent 评分。
但在当前 Mode A（请求级调度）下，此路径不会被触发。

重命名记录
----------
原名 ``H2OScoring``（h2o.py），因实际行为是位置启发式
而非 attention-based 评分，重命名为 ``PositionalScoring``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.bid_builder import build_bids


class PositionalScoring:
    """位置启发式 token 重要度评分。

    默认使用 attention-sink + recency 位置启发式。
    如果通过 ``update_from_decode_step()`` 提供了累积 attention 数据，
    则切换为 heavy-hitter + recent 评分。

    Parameters
    ----------
    heavy_ratio:
        Heavy hitter 比例（0~1）。默认 0.2。
    recent_ratio:
        Recent token 比例（0~1）。默认 0.2。
    algorithm_id:
        算法标识符。默认 "positional"。
    """

    def __init__(
        self,
        *,
        heavy_ratio: float = 0.2,
        recent_ratio: float = 0.2,
        algorithm_id: str = "positional",
    ) -> None:
        if not (0.0 <= heavy_ratio <= 1.0):
            raise ValueError(f"heavy_ratio must be in [0, 1], got {heavy_ratio}")
        if not (0.0 <= recent_ratio <= 1.0):
            raise ValueError(f"recent_ratio must be in [0, 1], got {recent_ratio}")
        if heavy_ratio + recent_ratio > 1.0:
            raise ValueError(
                f"heavy_ratio + recent_ratio must be <= 1.0, "
                f"got {heavy_ratio} + {recent_ratio} = {heavy_ratio + recent_ratio}"
            )
        self._heavy_ratio = heavy_ratio
        self._recent_ratio = recent_ratio
        self._algorithm_id = algorithm_id
        # 累积注意力分数：token position -> cumulative attention score
        self._cumulative_attention: list[float] = []
        # decode step 计数
        self._decode_steps: int = 0

    @property
    def heavy_ratio(self) -> float:
        """Heavy hitter 比例。"""
        return self._heavy_ratio

    @property
    def recent_ratio(self) -> float:
        """Recent token 比例。"""
        return self._recent_ratio

    @property
    def decode_steps(self) -> int:
        """已更新的 decode step 数。"""
        return self._decode_steps

    def update_from_decode_step(self, attention_pattern: Sequence[float]) -> None:
        """从一个 decode step 的注意力模式更新累积统计。

        Parameters
        ----------
        attention_pattern:
            当前 decode step 中，query token 对所有 KV token 的注意力权重。
            长度等于当前序列长度。每个值 >= 0，表示对应 KV position 的注意力权重。

            在实际部署中，这可以是：
            - FlashAttention 的 per-head 平均注意力估计
            - 或直接从 PagedAttention sparse pattern 获取的访问频率

            注意：不要求 attention_pattern 归一化到 [0, 1] 或 sum=1，
            因为累积统计在 score() 中会归一化。
        """
        if not attention_pattern:
            return

        # 扩展累积数组以容纳新增 token position
        if len(attention_pattern) > len(self._cumulative_attention):
            self._cumulative_attention.extend(
                0.0 for _ in range(len(attention_pattern) - len(self._cumulative_attention))
            )

        # 累加注意力分数
        for i, attn in enumerate(attention_pattern):
            self._cumulative_attention[i] += attn

        self._decode_steps += 1

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的重要度分数（0~1，越高越重要）。

        评分逻辑：
        1. 如果有累积注意力数据，按 heavy hitter + recent 策略评分。
        2. 如果没有累积注意力数据（尚未 decode），按位置启发式评分：
           - 开头和末尾 token 更重要（attention sink + recency）。

        Parameters
        ----------
        token_ids:
            Token ID 序列。
        **context:
            可选上下文。支持：
            - ``attention_pattern``: Sequence[float] — 如果提供，先调用
              ``update_from_decode_step()`` 再评分。

        Returns
        -------
        list[float]
            每个 token 的重要度，值域 [0, 1]。
        """
        n = len(token_ids)
        if n == 0:
            return []

        # 如果 context 中提供了 attention_pattern，先更新
        if "attention_pattern" in context:
            self.update_from_decode_step(context["attention_pattern"])

        if self._cumulative_attention and len(self._cumulative_attention) >= n:
            return self._score_from_cumulative(n)
        else:
            return self._score_positional(n)

    def _score_from_cumulative(self, n: int) -> list[float]:
        """基于累积注意力的 heavy hitter + recent 评分。"""
        cumulative = self._cumulative_attention[:n]

        # 归一化累积注意力到 [0, 1]
        max_attn = max(cumulative) if cumulative else 0.0
        normalized = [a / max_attn for a in cumulative] if max_attn > 0 else [0.0] * n

        # Heavy hitter：取累积注意力 top-k
        heavy_count = max(1, int(n * self._heavy_ratio))
        # Recent：取最后 k 个
        recent_count = max(1, int(n * self._recent_ratio))

        # 找出 heavy hitter 的索引
        indexed = sorted(enumerate(normalized), key=lambda x: x[1], reverse=True)
        heavy_indices = {idx for idx, _ in indexed[:heavy_count]}

        # Recent 索引
        recent_indices = set(range(max(0, n - recent_count), n))

        # 计算 final score
        scores = []
        for i in range(n):
            base_score = normalized[i]
            if i in heavy_indices:
                # Heavy hitter 获得 bonus
                base_score = max(base_score, 0.7)
            if i in recent_indices:
                # Recent token 获得 bonus
                base_score = max(base_score, 0.6)
            scores.append(min(1.0, base_score))

        return scores

    def _score_positional(self, n: int) -> list[float]:
        """无累积注意力时的位置启发式评分。

        遵循 attention sink (Xiao et al., 2023) 观察：
        - 序列开头 token（position 0~few）有很高的注意力
        - 序列末尾 token（recent）有较高注意力
        - 中间 token 注意力逐渐衰减
        """
        scores = []
        for i in range(n):
            # Attention sink: 前几个 token 重要（衰减到 0.5）
            sink_score = 0.8 * math.exp(-i / max(1, n * 0.05))
            # Recency: 越近的 token 越重要
            recency_score = 0.6 * (i / max(1, n - 1)) if n > 1 else 0.6
            # 取 max（两者代表不同的重要度来源）
            score = max(sink_score, recency_score)
            scores.append(min(1.0, max(0.0, score)))
        return scores

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于 positional 评分，针对多个压缩级别生成 CompressionBid。

        Parameters
        ----------
        request_id:
            推理请求 ID。
        token_ids:
            Token ID 序列。
        compression_levels:
            压缩比例列表（0~1），例如 [0.2, 0.4, 0.6, 0.8]。
            0.2 表示压缩掉 20% 的 token（保留 80%）。
        **context:
            传递给 ``score()`` 的上下文。

        Returns
        -------
        list[CompressionBid]
            每个压缩级别对应一个 bid。
        """
        scores = self.score(token_ids, **context)

        def _confidence() -> float:
            if self._decode_steps > 0:
                return min(1.0, 0.5 + 0.1 * math.log1p(self._decode_steps))
            return 0.3

        return build_bids(
            request_id=request_id,
            token_ids=token_ids,
            scores=scores,
            compression_levels=compression_levels,
            algorithm_id=self._algorithm_id,
            confidence_fn=_confidence,
            extra_metadata={
                "heavy_ratio": self._heavy_ratio,
                "recent_ratio": self._recent_ratio,
                "decode_steps": self._decode_steps,
                "scoring_method": "positional",
            },
        )

    def reset(self) -> None:
        """重置累积注意力统计。"""
        self._cumulative_attention.clear()
        self._decode_steps = 0
