"""统一 Bids 生成器 — 将 scorer 输出的 token-level scores 转换为 CompressionBid。

所有 scorer 的 ``generate_bids()`` 统一委托此模块，避免每个 scorer
重复实现 scores→bids 逻辑。

职责
----
- 依据 compression_levels 计算 ``tokens_freed`` (r)
- 依据被移除 token 的 score 统计计算 ``quality_delta`` (δ)
- 统一生成 ``bid_id``、``metadata``、基础校验
- 提供可选的 ``confidence_fn`` 回调以支持策略级置信度

设计原则
--------
- 输入只依赖 ``list[float]`` scores（[0,1]，越高越重要）
- 不依赖具体 scorer 类型（scorer-agnostic）
- 零外部依赖（仅 stdlib）
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from bidkv.protocol.bid import CompressionBid, make_bid_id


def build_bids(
    request_id: str,
    token_ids: Sequence[int],
    scores: Sequence[float],
    compression_levels: Sequence[float],
    algorithm_id: str,
    *,
    confidence_fn: Callable[[], float] | None = None,
    latency_factor: float = 0.1,
    extra_metadata: dict[str, Any] | None = None,
) -> list[CompressionBid]:
    """将 token-level scores 和 compression_levels 统一转换为 CompressionBid 列表。

    Parameters
    ----------
    request_id:
        推理请求 ID。
    token_ids:
        Token ID 序列。
    scores:
        每个 token 的重要度分数，长度与 ``token_ids`` 相同，值域 [0, 1]，
        越高越重要。
    compression_levels:
        压缩比例列表（0~1），例如 [0.2, 0.4, 0.6]。
        0.2 表示压缩掉 20% 的 token（保留 80%）。
    algorithm_id:
        压缩算法标识符（例如 "h2o", "attention_weight", "random"）。
    confidence_fn:
        可选回调，返回当前置信度 [0, 1]。若为 None，默认 0.5。
    latency_factor:
        每个 token 的预计压缩耗时系数（ms/token）。默认 0.1。
    extra_metadata:
        策略特定的额外 metadata 字段，会合并到每个 bid 的 metadata 中。

    Returns
    -------
    list[CompressionBid]
        按 compression_levels 顺序生成的 bid 列表。
    """
    n = len(token_ids)
    if n == 0:
        return []

    # 边界校验：scores 长度必须匹配
    if len(scores) != n:
        raise ValueError(f"scores length ({len(scores)}) must match token_ids length ({n})")

    # 按分数升序排列索引（最低分 = 最不重要 = 优先移除）
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])

    confidence = confidence_fn() if confidence_fn is not None else 0.5
    base_meta = extra_metadata or {}

    bids: list[CompressionBid] = []
    for level_idx, level in enumerate(compression_levels):
        tokens_to_remove = max(1, int(n * level))
        tokens_freed = min(tokens_to_remove, n - 1)  # 至少保留 1 个 token
        if tokens_freed <= 0:
            continue

        # quality_delta: 被移除 token 的平均重要度
        removed_scores = [s for _, s in indexed_scores[:tokens_freed]]
        avg_removed_importance = sum(removed_scores) / len(removed_scores)
        quality_delta = min(1.0, max(0.0, avg_removed_importance))

        metadata = {
            "compression_level": level,
            **base_meta,
        }

        bid = CompressionBid(
            bid_id=make_bid_id(request_id, level_idx),
            request_id=request_id,
            algorithm_id=algorithm_id,
            tokens_freed=tokens_freed,
            quality_delta=quality_delta,
            compress_latency_ms=latency_factor * tokens_freed,
            confidence=confidence,
            metadata=metadata,
        )
        bids.append(bid)

    return bids
