"""BidKV 全局配置：feature gate + kill switch。

设计原则
--------
- **默认 OFF**：``BidKVConfig()`` 创建的实例 ``enabled=False``。
- **Kill switch**：``BidKVConfig(kill_switch=True)`` 立即绕过所有 bid 逻辑。
- **零外部依赖**：仅使用 Python stdlib。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BidKVConfig:
    """BidKV 功能配置。

    Attributes
    ----------
    enabled:
        是否启用 BidKV 功能。默认 OFF。
    kill_switch:
        紧急关闭开关。设为 True 时，无论 ``enabled`` 值如何，
        所有 bid 操作立即变为 no-op。
    delta_budget:
        单次 solve 周期允许的最大累积 quality_delta 预算。
        默认 1.0（不限制）。
    max_bids_per_solve:
        单次 solve 周期最多接受的 bid 数量。
        默认 0 表示不限制。
    """

    enabled: bool = False
    kill_switch: bool = False
    delta_budget: float = 1.0
    max_bids_per_solve: int = 0

    def __post_init__(self) -> None:
        if self.delta_budget < 0.0:
            raise ValueError(f"delta_budget must be >= 0.0, got {self.delta_budget}")
        if self.max_bids_per_solve < 0:
            raise ValueError(f"max_bids_per_solve must be >= 0, got {self.max_bids_per_solve}")

    @property
    def is_active(self) -> bool:
        """是否实际激活（enabled=True 且 kill_switch=False）。"""
        return self.enabled and not self.kill_switch
