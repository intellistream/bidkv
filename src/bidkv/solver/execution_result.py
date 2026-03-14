"""ExecutionResult — bid 执行结果（含 actual_freed 字段）。

原 S04 #021 修正：bid acceptance 后必须记录实际释放量而非预估值，
以便 Solver/调度器判断是否需要触发 fallback eviction。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    """单个 bid 的执行结果。

    Attributes
    ----------
    bid_id:
        被执行的 bid ID。
    estimated_freed:
        bid 声明的预估释放 token 数（来自 CompressionBid.tokens_freed）。
    actual_freed:
        CompressionExecutor 返回的实际释放 token 数。
        可能与 estimated_freed 不同（块对齐、最小保留等约束）。
    success:
        执行是否成功。False 时 actual_freed 应为 0。
    """

    bid_id: str
    estimated_freed: int
    actual_freed: int
    success: bool

    @property
    def shortfall(self) -> int:
        """estimated 与 actual 的差值（>= 0 表示不足）。"""
        return max(0, self.estimated_freed - self.actual_freed)
