"""PressureDetector — KV 内存压力感知组件（通用版）。

从 ``sagellm-control-plane`` 的 ``pressure_detector.py`` 提取并通用化。
原始版本绑定 ``KVCacheManager.get_stats()``；通用版改为纯数值输入，
由 FrameworkAdapter 定期推送 KV 统计（pressure interception boundary）。

Feature Gate: ``compress.scheduling_primitive.v1``（默认 OFF）

设计保证（Fix S01 #018）
-----------------------
- ``update_stats()`` 存储**瞬时值**，``is_under_pressure()`` 直接使用瞬时值。
- **禁止** rolling window / 指数平滑 — 压力判断必须基于当前最新快照。

KV 统计唯一来源（Fix S07 #024）
------------------------------
- ``PressureDetector`` 是 Solver 获取 KV 状态的**唯一入口**。
- Solver 通过 ``detector.needed_tokens()`` 获取需要释放的 token 数，不独立计算。
- 通过 ``detector.get_kv_stats()`` 获取当前 KV 统计快照。
"""

from __future__ import annotations

import logging

from bidkv.pressure.config import PressureConfig

logger = logging.getLogger(__name__)


class PressureDetector:
    """监控 KV 内存占用率，判断系统是否处于 KV 内存压力态。

    与原始 ``sagellm-control-plane`` 版本的区别：
    - **通用化**：不再依赖 ``KVCacheManager``，由外部通过 ``update_stats()`` 推送数据
    - **零 sagellm 依赖**：仅使用 ``bidkv`` 内部类型

    触发条件（任一满足即认为处于压力态）：

    1. KV 内存占用率 >= ``threshold_pct``（默认 85%）
    2. 高优先级请求等待队列 > 0，且可用 KV token < ``min_free_tokens``

    Feature Gate: ``compress.scheduling_primitive.v1``（默认 OFF）。
    Feature OFF 时 :meth:`is_under_pressure` 始终返回 ``False``。

    Parameters
    ----------
    config:
        PressureDetector 配置。若为 None 则使用默认配置（feature OFF）。
    """

    def __init__(self, config: PressureConfig | None = None) -> None:
        self._config = config or PressureConfig()
        # 由外部推送的 KV 统计
        self._used_tokens: int = 0
        self._max_tokens: int = 0
        self._pending_high_priority: int = 0

        logger.debug(
            "PressureDetector initialized: threshold=%.1f%%, min_free_tokens=%d, enabled=%s",
            self._config.threshold_pct * 100,
            self._config.min_free_tokens,
            self._config.enabled,
        )

    def update_stats(
        self,
        used_tokens: int,
        max_tokens: int,
        pending_high_priority: int = 0,
    ) -> None:
        """更新 KV 使用统计（由 FrameworkAdapter 定期推送）。

        Parameters
        ----------
        used_tokens:
            当前已使用的 KV token 数量。
        max_tokens:
            KV 池的最大 token 容量。
        pending_high_priority:
            当前高优先级请求的等待数量。
        """
        self._used_tokens = used_tokens
        self._max_tokens = max_tokens
        self._pending_high_priority = pending_high_priority

    def is_under_pressure(self) -> bool:
        """判断当前系统是否处于 KV 内存压力态。

        Feature OFF 时始终返回 ``False``（零开销）。

        Returns
        -------
        bool
            True = 系统处于压力态，调度器应触发 bid 查询。
            False = 无压力或 feature 未激活。
        """
        if not self._config.enabled:
            return False

        if self._max_tokens <= 0:
            return False

        occupancy = self._used_tokens / self._max_tokens

        # 条件 1：KV 占用率超阈值
        if occupancy >= self._config.threshold_pct:
            logger.debug(
                "KV pressure detected: occupancy=%.1f%% >= threshold=%.1f%%",
                occupancy * 100,
                self._config.threshold_pct * 100,
            )
            return True

        # 条件 2：高优先级请求等待且可用 token 不足
        free_tokens = self._max_tokens - self._used_tokens
        if self._pending_high_priority > 0 and free_tokens < self._config.min_free_tokens:
            logger.debug(
                "KV pressure detected: pending_high=%d, free_tokens=%d < min_free=%d",
                self._pending_high_priority,
                free_tokens,
                self._config.min_free_tokens,
            )
            return True

        return False

    def needed_tokens(self) -> int:
        """估算当前需要释放多少 KV token。

        返回距离安全线的缺口。

        Returns
        -------
        int
            需要释放的 token 数量（>= 0）。0 表示无需额外 token。
        """
        if self._max_tokens <= 0:
            return 0
        safe_threshold = int(self._max_tokens * self._config.threshold_pct)
        gap = self._used_tokens - safe_threshold
        return max(0, gap)

    def set_enabled(self, enabled: bool) -> None:
        """动态切换 feature gate。"""
        # PressureConfig is a dataclass (not frozen), so we can mutate
        object.__setattr__(self._config, "enabled", enabled)
        logger.info("PressureDetector enabled -> %s", enabled)

    def get_kv_stats(self) -> dict[str, int]:
        """返回当前 KV 统计快照（唯一来源，Fix S07 #024）。

        Solver 和其他组件必须通过此方法获取 KV 状态，不得独立计算。

        Returns
        -------
        dict[str, int]
            包含 ``used_tokens``, ``max_tokens``, ``free_tokens``,
            ``pending_high_priority`` 字段。
        """
        free = max(0, self._max_tokens - self._used_tokens)
        return {
            "used_tokens": self._used_tokens,
            "max_tokens": self._max_tokens,
            "free_tokens": free,
            "pending_high_priority": self._pending_high_priority,
        }
