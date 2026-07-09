"""BidPoolManager — 管理活跃请求的 bid 快照（通用版）。

从 ``sagellm-kv-compress`` 的 ``bid_pool.py`` 提取并通用化。
原始版本绑定 ``ContextCompressor.get_bids()``；通用版改为由外部直接提交 bid 列表，
由 FrameworkAdapter 负责生成 bids，Pool 只管存储和查询。

Feature Gate: ``compress.scheduling_primitive.v1``（默认 OFF）
"""

from __future__ import annotations

import logging
import threading
import time

from bidkv.protocol.bid import BidPool, CompressionBid

logger = logging.getLogger(__name__)


class BidPoolManager:
    """管理所有活跃请求的 bid 快照，支持调度器轮询与增量更新。

    与原始 ``sagellm-kv-compress`` 版本的区别：
    - **通用化**：不再依赖 ``ContextCompressor``，由外部通过 ``submit_bids()`` 提交
    - **零 sagellm 依赖**：仅使用 ``bidkv.protocol`` 类型

    Feature Gate 控制：
    - ``enabled=True``（默认 False）：激活 bid 存储与查询功能
    - ``kill_switch=True``：立即停止所有操作，现有缓存视为无效

    Args:
        enabled: 是否激活 bid 功能。默认 False（feature OFF）。
        kill_switch: 紧急关闭开关。True 时等同于 enabled=False，优先级最高。
    """

    def __init__(self, *, enabled: bool = False, kill_switch: bool = False) -> None:
        self._enabled = enabled
        self._kill_switch = kill_switch
        # {request_id: list[CompressionBid]}
        self._bids: dict[str, list[CompressionBid]] = {}
        # {bid_id: CompressionBid} — O(1) lookup by bid_id
        self._bid_index: dict[str, CompressionBid] = {}
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        """返回 feature 是否处于激活状态（enabled=True 且 kill_switch=False）。"""
        return self._enabled and not self._kill_switch

    def enable(self) -> None:
        """激活 bid 功能（feature ON）。"""
        self._kill_switch = False
        self._enabled = True
        logger.info("BidPoolManager: feature enabled")

    def disable(self) -> None:
        """停用 bid 功能，清除现有缓存（feature OFF）。"""
        self._enabled = False
        with self._lock:
            self._bids.clear()
            self._bid_index.clear()
        logger.info("BidPoolManager: feature disabled, bids cleared")

    def activate_kill_switch(self) -> None:
        """激活 kill switch，立即停止所有操作并清空缓存。"""
        self._kill_switch = True
        with self._lock:
            self._bids.clear()
            self._bid_index.clear()
        logger.warning("BidPoolManager: KILL SWITCH activated, all bids cleared immediately")

    def submit_bids(self, request_id: str, bids: list[CompressionBid]) -> None:
        """提交指定请求的 bid 列表（替代原始版本的 refresh + compressor.get_bids）。

        FrameworkAdapter 负责生成 bids，Pool 只管存储和查询。

        Feature OFF 路径：直接返回，不存储任何 bid。

        Args:
            request_id: 目标请求 ID。
            bids: 由 FrameworkAdapter 生成的 bid 列表。
        """
        if not self.is_active:
            return

        with self._lock:
            # 先清除该 request 的旧 bid 索引
            old_bids = self._bids.get(request_id, [])
            for old_bid in old_bids:
                self._bid_index.pop(old_bid.bid_id, None)
            # 写入新 bid 并建立索引
            self._bids[request_id] = list(bids)
            for bid in bids:
                self._bid_index[bid.bid_id] = bid

        logger.debug(
            "BidPoolManager.submit_bids: request_id=%s, stored %d bids",
            request_id,
            len(bids),
        )

    def get_pool_snapshot(self) -> BidPool:
        """生成当前所有活跃请求的 bid 快照（供调度器使用）。

        Feature OFF：返回空快照（bids=()，snapshot_time_ns=0）。

        Returns:
            :class:`BidPool` 不可变快照。bids 按 tokens_freed 降序排列。
        """
        if not self.is_active:
            return BidPool(snapshot_time_ns=0, bids=())

        with self._lock:
            all_bids: list[CompressionBid] = []
            for bids in self._bids.values():
                all_bids.extend(bids)

        # 按 tokens_freed 降序排列（方便调度器贪心选择）
        all_bids.sort(key=lambda b: b.tokens_freed, reverse=True)

        return BidPool(
            snapshot_time_ns=time.monotonic_ns(),
            bids=tuple(all_bids),
        )

    def get_bid(self, bid_id: str) -> CompressionBid | None:
        """O(1) 查找指定 bid_id 的 bid。

        Args:
            bid_id: 目标 bid 的唯一标识。

        Returns:
            ``CompressionBid`` 实例，不存在时返回 None。
        """
        if not self.is_active:
            return None
        with self._lock:
            return self._bid_index.get(bid_id)

    def get_bids_for_request(self, request_id: str) -> list[CompressionBid]:
        """返回指定请求的当前 bid 列表。

        Feature OFF / kill_switch 激活：返回空列表。

        Args:
            request_id: 目标请求 ID。

        Returns:
            bid 列表。若该请求无缓存则返回空列表。
        """
        if not self.is_active:
            return []
        with self._lock:
            return list(self._bids.get(request_id, []))

    def remove_by_request(self, request_id: str) -> int:
        """移除该请求的所有 bid，返回移除数量。

        Args:
            request_id: 要移除的请求 ID。若不存在则返回 0。

        Returns:
            被移除的 bid 数量。
        """
        with self._lock:
            removed = self._bids.pop(request_id, None)
            if removed is not None:
                for bid in removed:
                    self._bid_index.pop(bid.bid_id, None)
        count = len(removed) if removed is not None else 0
        if count > 0:
            logger.debug(
                "BidPoolManager.remove_by_request: request_id=%s, removed %d bids",
                request_id,
                count,
            )
        return count

    def invalidate(self, request_id: str) -> None:
        """使指定请求的 bid 缓存失效（语义同 remove_by_request）。

        Args:
            request_id: 要失效的请求 ID。若不存在则静默忽略。
        """
        self.remove_by_request(request_id)

    def invalidate_all(self) -> None:
        """使所有请求的 bid 缓存失效。"""
        with self._lock:
            count = sum(len(v) for v in self._bids.values())
            self._bids.clear()
            self._bid_index.clear()
        logger.debug("BidPoolManager.invalidate_all: removed %d total bids", count)

    @property
    def active_request_count(self) -> int:
        """当前有缓存 bid 的请求数量。"""
        with self._lock:
            return len(self._bids)

    @property
    def total_bid_count(self) -> int:
        """当前缓存的 bid 总数。"""
        with self._lock:
            return sum(len(v) for v in self._bids.values())

    def get_stats(self) -> dict[str, object]:
        """返回 BidPoolManager 的运行统计信息。"""
        return {
            "enabled": self._enabled,
            "kill_switch": self._kill_switch,
            "is_active": self.is_active,
            "active_requests": self.active_request_count,
            "total_bids": self.total_bid_count,
        }
