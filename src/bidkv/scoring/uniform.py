"""Uniform 评分策略 — 消融实验用基线。

所有 token 赋予相同的重要度分数，表示"无差别压缩"。
用于消融实验中验证评分策略的价值——如果 H2O/Attention 策略
无法显著优于 Uniform，则说明评分信号没有携带有效信息。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.bid_builder import build_bids


class UniformScoring:
    """Uniform scoring：所有 token 等权（baseline）。

    Parameters
    ----------
    uniform_score:
        所有 token 的固定分数。默认 0.5。
    algorithm_id:
        算法标识符。默认 "uniform"。
    """

    def __init__(
        self,
        *,
        uniform_score: float = 0.5,
        algorithm_id: str = "uniform",
    ) -> None:
        if not (0.0 <= uniform_score <= 1.0):
            raise ValueError(f"uniform_score must be in [0, 1], got {uniform_score}")
        self._uniform_score = uniform_score
        self._algorithm_id = algorithm_id

    @property
    def uniform_score(self) -> float:
        """固定分数值。"""
        return self._uniform_score

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的等权重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。

        Returns
        -------
        list[float]
            所有值均为 ``uniform_score``。
        """
        return [self._uniform_score] * len(token_ids)

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于均匀评分生成 CompressionBid。"""
        n = len(token_ids)
        if n == 0:
            return []

        scores = self.score(token_ids, **context)

        return build_bids(
            request_id=request_id,
            token_ids=token_ids,
            scores=scores,
            compression_levels=compression_levels,
            algorithm_id=self._algorithm_id,
            confidence_fn=lambda: 1.0,
            extra_metadata={
                "uniform_score": self._uniform_score,
                "scoring_method": "uniform",
            },
        )
