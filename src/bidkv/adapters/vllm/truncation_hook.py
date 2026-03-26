"""Truncation Hook — vLLM KV cache tail truncation via monkeypatch.

Implements tail truncation by dynamically adding
``truncate_request_tail()`` to vLLM's ``KVCacheManager`` and the
underlying ``SingleTypeKVCacheManager`` / ``KVCacheCoordinator``.

Design:
- All block-level operations happen inside SingleTypeKVCacheManager,
  keeping req_to_blocks and block_pool atomically consistent.
- Coordinator forwards to all single_type_managers.
- KVCacheManager is the public entry point used by the adapter.

Safety constraints (INV-1..INV-7 from issue #054):
- INV-1: req_to_blocks length matches actual block ownership
- INV-2: ref_cnt correctly decremented
- INV-3: freed blocks removed from req_to_blocks before entering free list
- INV-4: at least 1 block always retained (never full eviction)
- INV-5: num_computed_tokens updated by *caller* (adapter) to match boundary
- INV-6: only tail (contiguous end) blocks removed — causal mask preserved
- INV-7: shared prefix blocks (ref_cnt > 1) are never freed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TruncateResult:
    """Result of a tail truncation operation."""

    success: bool
    actual_freed_blocks: int = 0
    actual_freed_tokens: int = 0
    new_num_blocks: int = 0
    new_computed_token_boundary: int = 0
    fallback_required: bool = False
    reason: str = ""


def _single_type_truncate_tail_blocks(
    manager: Any,
    request_id: str,
    num_blocks_to_free: int,
) -> TruncateResult:
    """Truncate tail blocks from a SingleTypeKVCacheManager.

    This is the core atomic operation: updates req_to_blocks and
    block_pool.free_blocks in one method call.
    """
    blocks = manager.req_to_blocks.get(request_id)
    if blocks is None:
        return TruncateResult(
            success=False,
            fallback_required=True,
            reason="request not found in req_to_blocks",
        )

    if len(blocks) <= 1:
        return TruncateResult(
            success=False,
            fallback_required=True,
            reason=f"only {len(blocks)} block(s), cannot truncate",
        )

    # Limit: never free all blocks, keep at least 1
    max_freeable = len(blocks) - 1
    actual_free_target = min(num_blocks_to_free, max_freeable)

    if actual_free_target <= 0:
        return TruncateResult(
            success=False,
            fallback_required=True,
            reason="nothing to truncate",
        )

    # Take tail blocks (from the end)
    tail_candidates = blocks[-actual_free_target:]

    # INV-7: skip shared blocks (ref_cnt > 1 = prefix cache)
    safe_tail: list[Any] = []
    for blk in reversed(tail_candidates):
        if blk.ref_cnt > 1:
            break  # hit a shared prefix block, stop
        safe_tail.append(blk)
    safe_tail.reverse()

    if not safe_tail:
        return TruncateResult(
            success=False,
            fallback_required=True,
            reason="all tail blocks shared by prefix cache (ref_cnt > 1)",
        )

    # Atomic: remove from req_to_blocks, then free to block_pool
    # INV-1 + INV-3: req_to_blocks updated before blocks enter free list
    del blocks[-len(safe_tail) :]
    manager.block_pool.free_blocks(safe_tail)

    new_num_blocks = len(blocks)
    block_size = getattr(manager, "block_size", 16)
    freed_tokens = len(safe_tail) * block_size

    return TruncateResult(
        success=True,
        actual_freed_blocks=len(safe_tail),
        actual_freed_tokens=freed_tokens,
        new_num_blocks=new_num_blocks,
        new_computed_token_boundary=new_num_blocks * block_size,
        fallback_required=False,
        reason="tail truncated",
    )


def _coordinator_truncate_tail(
    coordinator: Any,
    request_id: str,
    num_blocks_to_free: int,
) -> TruncateResult:
    """Forward truncation to all single_type_managers.

    Returns the result from the first manager (main KV group).
    All managers truncate the same number of blocks.
    """
    result = None
    for mgr in coordinator.single_type_managers:
        r = _single_type_truncate_tail_blocks(mgr, request_id, num_blocks_to_free)
        if result is None:
            result = r
        if not r.success:
            return r
    return result or TruncateResult(success=False, fallback_required=True, reason="no managers")


def _kv_cache_manager_truncate_request_tail(
    kv_cache_manager: Any,
    request_id: str,
    num_blocks_to_free: int,
) -> TruncateResult:
    """Public API: safely truncate tail KV blocks for a request.

    This is the method called by the BidKV adapter.
    """
    return _coordinator_truncate_tail(kv_cache_manager.coordinator, request_id, num_blocks_to_free)


def install_truncation_support(kv_cache_manager: Any) -> None:
    """Monkeypatch truncate_request_tail() onto a KVCacheManager instance.

    Called once when BidKV tail truncation is activated.
    """
    if hasattr(kv_cache_manager, "truncate_request_tail"):
        return  # already patched

    import types

    kv_cache_manager.truncate_request_tail = types.MethodType(
        lambda self, req_id, n_blocks: _kv_cache_manager_truncate_request_tail(
            self, req_id, n_blocks
        ),
        kv_cache_manager,
    )
    logger.info("BidKV: truncate_request_tail() installed on KVCacheManager")
