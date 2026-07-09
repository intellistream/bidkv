"""Static-Random baseline — 固定压缩率 + 随机受害者选择。

设计理由：控制变量基线。固定 50% 压缩率 + 随机选择请求，
隔离"信息化选择"（scoring/bid）对结果的贡献。
如果任何策略不能显著优于 Static-Random，说明该策略的
信息利用可能不如随机。

选择公式：victim = random.choice(active), ratio = 0.5
"""

from __future__ import annotations

import random
from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class StaticRandomStrategy(BaselineStrategy):
    """Static-Random：固定压缩比 + 随机受害者选择。

    Parameters
    ----------
    compression_ratio:
        固定压缩比例（0~1）。默认 0.5（压缩掉 50% 的 token）。
    seed:
        随机种子，用于可复现的随机选择。None 则不设种子。
    """

    def __init__(
        self,
        *,
        compression_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        self._compression_ratio = compression_ratio
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "static-random"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """随机选择请求，以固定比率压缩，直到释放目标 token 数。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        # 随机打乱候选列表
        shuffled = list(candidates)
        self._rng.shuffle(shuffled)

        actions: list[CompressionAction] = []
        freed = 0
        for req in shuffled:
            if freed >= needed_tokens:
                break
            # 按固定比率压缩
            tokens_to_free = max(1, int(req.current_tokens * self._compression_ratio))
            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="compress",
                    target_tokens=tokens_to_free,
                    metadata={
                        "strategy": "static-random",
                        "compression_ratio": self._compression_ratio,
                    },
                )
            )
            freed += tokens_to_free

        return actions
