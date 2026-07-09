"""SolverConfig — GreedyBidSolver 配置参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolverConfig:
    """GreedyBidSolver 配置参数。

    Attributes
    ----------
    enabled:
        Feature gate 主开关，默认 False（OFF）。
    delta_budget:
        调度器可接受的全局质量损失上限（Σδ ≤ delta_budget）。
        默认 0.15，即最多允许 15% 的质量退化。
    max_bids_per_solve:
        单次求解最多选取的 bid 数量（防止过度压缩）。默认 20。
    kill_switch:
        Kill switch，True 时 ``solve()`` 立即返回空 acceptance，不执行任何逻辑。
        优先级高于 ``enabled``。默认 False。
    decision_reason:
        BidAcceptance 中记录的触发原因前缀。
        默认 ``"kv_pool_pressure_threshold_exceeded"``。
    """

    enabled: bool = False
    delta_budget: float = 0.15
    max_bids_per_solve: int = 20
    kill_switch: bool = False
    decision_reason: str = "kv_pool_pressure_threshold_exceeded"

    def __post_init__(self) -> None:
        if not (0.0 <= self.delta_budget <= 1.0):
            raise ValueError(f"delta_budget must be in [0.0, 1.0], got {self.delta_budget}")
        if self.max_bids_per_solve < 1:
            raise ValueError(f"max_bids_per_solve must be >= 1, got {self.max_bids_per_solve}")

    @property
    def is_active(self) -> bool:
        """是否实际激活（enabled=True 且 kill_switch=False）。"""
        return self.enabled and not self.kill_switch
