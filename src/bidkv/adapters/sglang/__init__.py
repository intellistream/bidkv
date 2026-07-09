"""bidkv.adapters.sglang — SGLang framework adapter.

**Mode A 架构（request-level 调度）**：
BidKV 通过 scheduler_hook monkey-patch ``get_next_batch_to_run()``，
在 native batch selection 前执行 request-level 调度决策。
不做 token-level 部分释放（radix_hook.py 已删除，Mode B 已停用）。

关键差异 vs vLLM：
- KV 管理：树状 RadixAttention（前缀共享）vs 扁平 BlockTable
- 驱逐策略：radix tree 节点 LRU vs seq_group 整体 preempt
- 调度入口：``get_next_batch_to_run()`` vs ``schedule()``
- KV 内存池：``TokenToKVPool`` + ``ReqToTokenPool`` vs ``BlockAllocator``
"""

from __future__ import annotations

from bidkv.adapters.sglang.adapter import SGLangAdapter

__all__ = [
    "SGLangAdapter",
]
