"""Random 评分策略 — 消融实验用基线。

为每个 token 随机赋予重要度分数。用于消融实验中验证：
有信息量的评分策略（H2O/Attention）vs 随机猜测的差距。

注意：使用固定 seed 时可复现。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.bid_builder import build_bids


class RandomScoring:
    """Random scoring：随机分数（baseline）。

    Parameters
    ----------
    seed:
        随机种子。若为 None，则不设置 seed（不可复现）。
        默认 None。
    algorithm_id:
        算法标识符。默认 "random"。
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        algorithm_id: str = "random",
    ) -> None:
        self._seed = seed
        self._algorithm_id = algorithm_id
        self._rng = random.Random(seed)

    @property
    def seed(self) -> int | None:
        """随机种子。"""
        return self._seed

    def score(
        self,
        token_ids: Sequence[int],
        **context: Any,
    ) -> list[float]:
        """返回每个 token 的随机重要度分数。

        Parameters
        ----------
        token_ids:
            Token ID 序列。

        Returns
        -------
        list[float]
            每个值在 [0, 1] 内随机生成。
        """
        return [self._rng.random() for _ in token_ids]

    def generate_bids(
        self,
        request_id: str,
        token_ids: Sequence[int],
        compression_levels: Sequence[float],
        **context: Any,
    ) -> list[CompressionBid]:
        """基于随机评分生成 CompressionBid。"""
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
            confidence_fn=lambda: 0.0,
            extra_metadata={
                "seed": self._seed,
                "scoring_method": "random",
            },
        )
