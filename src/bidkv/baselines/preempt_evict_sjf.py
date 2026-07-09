"""Preempt-Evict-SJF ablation baseline — SJF admission + LIFO eviction.

**归因 ablation**：isolate SJF admission 的贡献。
  preempt-evict (FCFS+LIFO) → preempt-evict-sjf (SJF+LIFO) → proactive (SJF+质量感知)
                Δ₁ = SJF admission 增益           Δ₂ = eviction 策略增益

设计：与 preempt-evict 完全相同的 eviction 逻辑（按优先级驱逐），
唯一差异是 scheduler_hook 中 admission ordering 使用 SJF。
无 proactive preempt，无 SRPT，无 priority cache。
"""

from __future__ import annotations

from typing import Any  # noqa: F401

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState


class PreemptEvictSJFStrategy(BaselineStrategy):
    """Preempt-Evict-SJF：SJF admission + LIFO eviction。

    与 PreemptEvictStrategy 相同的 victim selection（按优先级驱逐），
    但在 scheduler_hook 中 waiting queue 按 prompt_tokens 排序。

    用于消融分析：分离 SJF admission ordering 与质量感知驱逐的各自贡献。
    """

    @property
    def name(self) -> str:
        return "preempt-evict-sjf"

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **_kwargs: Any,
    ) -> list[CompressionAction]:
        """按优先级从低到高驱逐请求（与 preempt-evict 完全相同）。

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
                    metadata={
                        "strategy": "preempt-evict-sjf",
                        "priority": req.priority,
                    },
                )
            )
            freed += req.current_tokens

        return actions
