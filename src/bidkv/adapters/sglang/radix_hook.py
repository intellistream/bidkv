"""[DEPRECATED — Mode B] RadixAttention 节点级压缩/释放钩子。

本模块在 Mode A（request-level 调度）中为死代码。
Mode A 通过 scheduler_hook 实现 request-level 调度，不做 token-level 部分释放。
保留用于潜在的 Mode B 扩展（issue #054）。

SGLang 的 RadixAttention 使用 trie 树管理 KV cache，
每个节点对应一段 token 序列。本模块提供在 radix tree 节点粒度上
释放 KV 的函数。

关键注意事项：
- 共享前缀保护：ref count > 1 的节点不可释放
- token-level 精度：SGLang 支持比 vLLM block-level 更细的释放粒度
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def free_kv_positions(
    scheduler: Any,
    request_id: str,
    positions: list[int],
) -> int:
    """在 SGLang 的 KV pool 中释放指定 token 位置。

    SGLang 的 RadixCache 支持按节点粒度操作。本函数将 token 位置
    映射到对应的 KV slot，并释放非共享的 slot。

    Parameters
    ----------
    scheduler:
        SGLang Scheduler 实例。
    request_id:
        目标请求 ID。
    positions:
        要释放的 token 位置列表（0-indexed）。

    Returns
    -------
    int
        实际释放的 token 数量。
    """
    if not positions:
        return 0

    # 获取 req_to_token_pool 和 token_to_kv_pool
    req_to_token_pool = _get_req_to_token_pool(scheduler)
    token_to_kv_pool = _get_token_to_kv_pool(scheduler)

    if req_to_token_pool is None or token_to_kv_pool is None:
        logger.warning(
            "free_kv_positions: cannot access SGLang KV pools for request %s",
            request_id,
        )
        return 0

    # 查找请求在 ReqToTokenPool 中的映射
    req_pool_idx = _find_request_pool_index(scheduler, request_id)
    if req_pool_idx is None:
        logger.debug(
            "free_kv_positions: request %s not found in ReqToTokenPool",
            request_id,
        )
        return 0

    # 获取请求的 token → KV slot 映射
    token_indices = _get_token_indices(req_to_token_pool, req_pool_idx)
    if token_indices is None:
        return 0

    # 检查共享并释放
    freed_count = 0
    for pos in positions:
        if pos >= len(token_indices):
            continue
        kv_slot = token_indices[pos]
        # 检查此 slot 是否被其他请求共享
        if _is_shared_slot(scheduler, kv_slot, request_id):
            logger.debug(
                "free_kv_positions: skipping shared slot %d at position %d",
                kv_slot,
                pos,
            )
            continue
        # 释放 KV slot
        if _free_kv_slot(token_to_kv_pool, kv_slot):
            freed_count += 1

    logger.debug(
        "free_kv_positions: request=%s, requested=%d, freed=%d",
        request_id,
        len(positions),
        freed_count,
    )
    return freed_count


def get_shared_prefix_positions(
    scheduler: Any,
    request_id: str,
    total_tokens: int,
) -> set[int]:
    """检测请求中哪些 token 位置与其他请求共享前缀。

    遍历 radix tree，找出 ref count > 1 的节点中包含的 token 位置。

    Parameters
    ----------
    scheduler:
        SGLang Scheduler 实例。
    request_id:
        目标请求 ID。
    total_tokens:
        请求的总 token 数。

    Returns
    -------
    set[int]
        与其他请求共享的 token 位置集合。
    """
    radix_cache = _get_radix_cache(scheduler)
    if radix_cache is None:
        return set()

    shared: set[int] = set()

    # 遍历 radix tree 查找共享节点
    # SGLang RadixCache 的节点有 lock_ref 表示引用计数
    if hasattr(radix_cache, "root_node"):
        _collect_shared_positions(radix_cache.root_node, 0, total_tokens, shared)

    return shared


def _collect_shared_positions(node: Any, start_pos: int, max_pos: int, shared: set[int]) -> None:
    """递归遍历 radix tree 节点，收集共享位置。"""
    if node is None or start_pos >= max_pos:
        return

    # 检查节点引用计数
    ref_count = getattr(node, "lock_ref", 0)
    node_len = 0

    if hasattr(node, "key"):
        key = node.key
        node_len = len(key) if hasattr(key, "__len__") else 0

    # ref_count > 1 表示此节点被多个请求共享
    if ref_count > 1:
        for i in range(node_len):
            pos = start_pos + i
            if pos < max_pos:
                shared.add(pos)

    # 递归子节点
    children = getattr(node, "children", {})
    if isinstance(children, dict):
        child_start = start_pos + node_len
        for child in children.values():
            _collect_shared_positions(child, child_start, max_pos, shared)


# ---------------------------------------------------------------------------
# Internal helpers — SGLang pool access
# ---------------------------------------------------------------------------


def _get_token_to_kv_pool(scheduler: Any) -> Any | None:
    """获取 TokenToKVPool。"""
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "token_to_kv_pool"):
            return tp.token_to_kv_pool
    if hasattr(scheduler, "token_to_kv_pool"):
        return scheduler.token_to_kv_pool
    return None


def _get_req_to_token_pool(scheduler: Any) -> Any | None:
    """获取 ReqToTokenPool。"""
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "req_to_token_pool"):
            return tp.req_to_token_pool
    if hasattr(scheduler, "req_to_token_pool"):
        return scheduler.req_to_token_pool
    return None


def _get_radix_cache(scheduler: Any) -> Any | None:
    """获取 RadixCache。"""
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "tree_cache"):
            return tp.tree_cache
    if hasattr(scheduler, "tree_cache"):
        return scheduler.tree_cache
    return None


def _find_request_pool_index(scheduler: Any, request_id: str) -> int | None:
    """在 scheduler 的运行队列中查找请求的 pool index。"""
    running = getattr(scheduler, "running_batch", None)
    if running is None:
        return None

    reqs = getattr(running, "reqs", None)
    if reqs is None:
        return None

    for req in reqs:
        rid = getattr(req, "rid", None) or getattr(req, "request_id", None)
        if str(rid) == str(request_id):
            return getattr(req, "req_pool_idx", None)

    return None


def _get_token_indices(req_to_token_pool: Any, req_pool_idx: int) -> list[int] | None:
    """获取请求在 ReqToTokenPool 中的 token → KV slot 映射。"""
    if hasattr(req_to_token_pool, "req_to_token"):
        mapping = req_to_token_pool.req_to_token
        if hasattr(mapping, "__getitem__"):
            try:
                indices = mapping[req_pool_idx]
                if hasattr(indices, "tolist"):
                    return indices.tolist()
                return list(indices)
            except (IndexError, KeyError):
                return None
    return None


def _is_shared_slot(scheduler: Any, kv_slot: int, request_id: str) -> bool:
    """检查 KV slot 是否被其他请求共享。

    通过遍历 running batch 中其他请求的 token→KV 映射，检查同一
    kv_slot 是否出现在不同请求中。如果是，则该 slot 被共享，不可释放。

    同时检查 RadixCache 节点的 lock_ref：ref > 1 表示该节点被多棵
    子树共享。
    """
    # 方法 1：通过 RadixCache 节点 ref count 检查
    radix_cache = _get_radix_cache(scheduler)
    if (
        radix_cache is not None
        and hasattr(radix_cache, "root_node")
        and _slot_in_shared_node(radix_cache.root_node, kv_slot)
    ):
        return True

    # 方法 2：遍历 running batch 中其他请求的 token 映射
    req_to_token_pool = _get_req_to_token_pool(scheduler)
    if req_to_token_pool is None:
        return False

    running = getattr(scheduler, "running_batch", None)
    if running is None:
        return False

    reqs = getattr(running, "reqs", None)
    if reqs is None:
        return False

    for req in reqs:
        rid = getattr(req, "rid", None) or getattr(req, "request_id", None)
        if str(rid) == str(request_id):
            continue
        pool_idx = getattr(req, "req_pool_idx", None)
        if pool_idx is None:
            continue
        indices = _get_token_indices(req_to_token_pool, pool_idx)
        if indices is not None and kv_slot in indices:
            return True

    return False


def _slot_in_shared_node(node: Any, kv_slot: int) -> bool:
    """递归检查 kv_slot 是否属于 lock_ref > 1 的 radix tree 节点。"""
    if node is None:
        return False

    ref_count = getattr(node, "lock_ref", 0)
    if ref_count > 1:
        # 检查此节点管理的 KV slot 范围
        value = getattr(node, "value", None)
        if value is not None:
            if hasattr(value, "tolist"):
                slots = value.tolist()
            elif hasattr(value, "__iter__"):
                slots = list(value)
            else:
                slots = []
            if kv_slot in slots:
                return True

    children = getattr(node, "children", {})
    if isinstance(children, dict):
        for child in children.values():
            if _slot_in_shared_node(child, kv_slot):
                return True

    return False


def _free_kv_slot(token_to_kv_pool: Any, kv_slot: int) -> bool:
    """释放单个 KV slot。

    Returns
    -------
    bool
        是否成功释放。
    """
    if hasattr(token_to_kv_pool, "free"):
        try:
            token_to_kv_pool.free(kv_slot)
            return True
        except Exception:
            logger.debug("_free_kv_slot: failed to free slot %d", kv_slot, exc_info=True)
            return False
    return False
