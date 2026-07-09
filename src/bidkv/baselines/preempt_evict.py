"""Preempt-Evict baseline — 框架默认行为，零压缩基线。

设计理由：模拟 LLM serving 框架的默认驱逐策略（如 vLLM preemption）。
不执行任何 KV 压缩，直接驱逐最低优先级的请求。
这是信息量最低的 baseline，作为所有其他策略的下界。

选择公式：victim = argmin(priority)
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class PreemptEvictStrategy(BaselineStrategy):
    """Preempt-Evict：不压缩，直接驱逐最低优先级请求。

    与 vLLM/SGLang 的默认 preemption 行为等价：
    当 KV 内存不足时，按优先级从低到高驱逐请求，
    直到释放足够 token。

    这是消融实验的基准下界。
    """

    @property
    def name(self) -> str:
        return "preempt-evict"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """按优先级从低到高驱逐请求，直到释放目标 token 数。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。

        Returns
        -------
        list[CompressionAction]
            驱逐操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        # 按优先级升序排序（最低优先级先驱逐）
        sorted_candidates = sorted(candidates, key=lambda r: r.priority)

        actions: list[CompressionAction] = []
        freed = 0
        for req in sorted_candidates:
            if freed >= needed_tokens:
                break
            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="evict",
                    target_tokens=req.current_tokens,
                    metadata={"strategy": "preempt-evict", "priority": req.priority},
                )
            )
            freed += req.current_tokens

        return actions
