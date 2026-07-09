"""bidkv.adapters.vllm — vLLM framework adapter.

vLLM 0.11+ (v1 架构) 使用 KVCacheManager + BlockPool 进行分页 KV 管理。
本模块提供 BidKV 在 vLLM 上的完整适配层。

核心注入点：
- Scheduler.schedule() 的 preemption 路径（allocate_slots 返回 None 时）
- Scheduler.update_from_output() 的 decode step 回调（Positional scoring 更新）
- Scheduler._free_request() 的请求完成回调（lifecycle cleanup）

.. note:: Architecture Decision — 为什么没有 block_manager.py

   Issue #044 spec 最初设计了 ``BidKVBlockSpaceManager(SelfAttnBlockSpaceManager)``
   子类方案，并要求通过 ``--block-manager-class`` CLI 参数注入。

   **vLLM v1（0.17+）彻底移除了这一抽象**：

   - ``vllm.core`` 模块已不存在（``SelfAttnBlockSpaceManager`` 随之移除）
   - ``EngineArgs.block_manager_class`` 参数已不存在
   - v1 使用 ``KVCacheManager`` + ``BlockPool`` 替代 ``BlockSpaceManager``
   - ``Scheduler`` 引用 ``kv_cache_manager``，不再引用 ``block_manager``

   因此采用 **Scheduler monkey-patch 方案**（``scheduler_hook.py``），
   这是 vLLM v1 架构下注入 BidKV 的唯一可行方式。
   功能上完全等价：在 ``allocate_slots()`` 失败触发 preempt 之前，
   先尝试 BidKV 压缩释放空间。
"""

from __future__ import annotations

from bidkv.adapters.vllm.adapter import VLLMAdapter

__all__ = [
    "VLLMAdapter",
]
