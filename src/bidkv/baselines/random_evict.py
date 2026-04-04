"""Random-Evict baseline — 随机受害者选择基线。

设计理由：作为"随机打破原生顺序"的消融基线，与 BidKV 的质量感知
驱逐排序形成对比。三策略消融链：

    vanilla_sglang → random_evict → bidkv
    （SGLang原生）   （随机干预）    （质量感知干预）

random_evict 与 vanilla_sglang 的唯一区别是：
在 proactive preempt 阶段，victim 通过随机排列选取，而不是由
SGLang native eviction 自行决定。若 BidKV 显著优于 random_evict，
说明 utility scoring 本身（而非简单打破默认顺序）在起作用。

选择公式：victim = random.shuffle(candidates)[0]
"""

from __future__ import annotations

import random
from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class RandomEvictStrategy(BaselineStrategy):
    """Random-Evict：随机选择 running 请求驱逐。

    每次 select_victims() 返回所有候选者的随机排列；
    scheduler_hook 的 proactive preempt 取第一个作为驱逐目标。
    running queue 也按随机顺序排列，影响 SGLang native eviction 路径。

    使用固定种子保证单次实验内可复现（不跨 run）。

    Parameters
    ----------
    seed:
        随机种子，默认 0。
    """

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random-evict"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[CompressionAction]:
        """随机排列所有候选请求并全部返回。

        返回所有候选者（而非只返回 1 个）使得 scheduler_hook 的
        priority cache 为每个请求分配一个随机优先级整数（0, 1, 2, ...），
        从而实现 running queue 的随机排序和随机 proactive preempt。

        Parameters
        ----------
        candidates:
            候选请求列表（当前 running requests）。
        needed_tokens:
            需要释放的 token 数（本策略不使用此参数，只决定顺序）。

        Returns
        -------
        list[CompressionAction]
            所有候选者的随机排列，action_type="evict"。
        """
        if not candidates:
            return []

        shuffled = list(candidates)
        self._rng.shuffle(shuffled)

        return [
            CompressionAction(
                request_id=r.request_id,
                action_type="evict",
                target_tokens=r.current_tokens,
                metadata={
                    "strategy": "random-evict",
                    "random_rank": i,
                },
            )
            for i, r in enumerate(shuffled)
        ]
