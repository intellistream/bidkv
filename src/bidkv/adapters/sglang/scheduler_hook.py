"""SGLang Scheduler Hook — 在 batch 选择前注入 BidKV 压力检测与压缩。

**Pressure Interception Boundary（语义冻结 v7.1）**：
BidKV 必须在 SGLang 原生 eviction（RadixAttention LRU）执行前拦截。

SGLang 的调度主循环通过 ``Scheduler.get_next_batch_to_run()`` 选择下一批请求。
在此之前，如果 KV 内存不足，SGLang 会触发 RadixCache 的 LRU 驱逐。

BidKV 在 eviction path 的入口处插入钩子，使 bid-based 压缩优先于 LRU 驱逐。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def install_scheduler_hook(scheduler: Any, adapter: Any) -> None:
    """将 BidKV 压力检测钩子注入 SGLang Scheduler。

    通过 monkey-patch ``get_next_batch_to_run`` 方法，在 native batch selection
    前执行 bidkv 压缩尝试。

    Parameters
    ----------
    scheduler:
        SGLang ``Scheduler`` 实例。
    adapter:
        ``SGLangAdapter`` 实例。
    """
    if not hasattr(scheduler, "get_next_batch_to_run"):
        raise RuntimeError(
            "SGLang Scheduler does not have 'get_next_batch_to_run' method. "
            "This may indicate an incompatible SGLang version."
        )

    original_method = scheduler.get_next_batch_to_run

    def patched_get_next_batch_to_run(*args: Any, **kwargs: Any) -> Any:
        """BidKV patched: 在 native batch selection 前尝试压缩。"""
        # Pressure interception: 在原生调度前尝试 bidkv 压缩
        if adapter.config.is_active:
            freed = adapter.try_compress()
            if freed > 0:
                logger.debug("scheduler_hook: freed %d tokens before batch selection", freed)

        # 调用原始方法
        return original_method(*args, **kwargs)

    # 安装 monkey-patch
    patched_get_next_batch_to_run.__wrapped__ = original_method
    scheduler.get_next_batch_to_run = patched_get_next_batch_to_run

    # 同时 patch eviction path（如果存在）
    _hook_eviction_path(scheduler, adapter)

    logger.info("SGLang scheduler hook installed (get_next_batch_to_run patched)")


def uninstall_scheduler_hook(scheduler: Any) -> None:
    """移除 BidKV 的 scheduler hook，恢复原始方法。

    Parameters
    ----------
    scheduler:
        SGLang ``Scheduler`` 实例。
    """
    # 恢复 get_next_batch_to_run
    method = getattr(scheduler, "get_next_batch_to_run", None)
    if method is not None and hasattr(method, "__wrapped__"):
        scheduler.get_next_batch_to_run = method.__wrapped__
        logger.info("SGLang scheduler hook uninstalled (get_next_batch_to_run)")

    # 恢复 RadixCache.evict（如果被 patch 过）
    radix_cache = _get_tree_cache(scheduler)
    if radix_cache is not None:
        evict_method = getattr(radix_cache, "evict", None)
        if evict_method is not None and hasattr(evict_method, "__wrapped__"):
            radix_cache.evict = evict_method.__wrapped__
            logger.info("SGLang scheduler hook uninstalled (RadixCache.evict)")


def _hook_eviction_path(scheduler: Any, adapter: Any) -> None:
    """Hook SGLang 的 RadixCache eviction path。

    SGLang 在 KV 内存不足时通过 RadixCache.evict() 释放 LRU 节点。
    BidKV 在 evict 入口前获得压缩尝试机会。
    """
    radix_cache = _get_tree_cache(scheduler)
    if radix_cache is None:
        logger.debug("_hook_eviction_path: no tree_cache found, skip")
        return

    if not hasattr(radix_cache, "evict"):
        logger.debug("_hook_eviction_path: tree_cache has no evict(), skip")
        return

    original_evict = radix_cache.evict

    def patched_evict(num_tokens: int, *args: Any, **kwargs: Any) -> Any:
        """BidKV patched: 在 native eviction 前尝试压缩。"""
        if adapter.config.is_active:
            freed = adapter.try_compress()
            if freed >= num_tokens:
                # 压缩释放的 token 已足够，可能不再需要 native eviction
                logger.debug(
                    "scheduler_hook: eviction averted, bidkv freed %d >= needed %d",
                    freed,
                    num_tokens,
                )
                return
        return original_evict(num_tokens, *args, **kwargs)

    patched_evict.__wrapped__ = original_evict
    radix_cache.evict = patched_evict
    logger.debug("RadixCache eviction hook installed")


def _get_tree_cache(scheduler: Any) -> Any | None:
    """获取 RadixCache 实例。"""
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "tree_cache"):
            return tp.tree_cache
    if hasattr(scheduler, "tree_cache"):
        return scheduler.tree_cache
    return None
