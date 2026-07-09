"""PressureConfig — PressureDetector 配置参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PressureConfig:
    """PressureDetector 配置参数。

    Attributes
    ----------
    threshold_pct:
        KV 内存占用率阈值（0.0 ~ 1.0）。超过此值时触发压力检测。默认 0.85。
    min_free_tokens:
        当高优先级请求等待时，可用 KV token 的最小值。低于此值且队列非空时触发。
        默认 512。
    enabled:
        Feature gate 开关，默认 False（OFF）。
    """

    threshold_pct: float = 0.85
    min_free_tokens: int = 512
    enabled: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold_pct <= 1.0):
            raise ValueError(f"threshold_pct must be in (0.0, 1.0], got {self.threshold_pct}")
        if self.min_free_tokens < 0:
            raise ValueError(f"min_free_tokens must be >= 0, got {self.min_free_tokens}")
