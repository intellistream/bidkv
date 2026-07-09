"""CompressionBid 核心数据结构 (从 sagellm-protocol 提取)

本模块定义 KV 压缩层向调度器暴露"报价"能力的核心数据结构：

- :class:`CompressionBid` — 压缩层针对单个请求的一次报价
- :class:`BidPool` — 某时刻所有活跃请求 bid 的快照集合
- :class:`BidAcceptance` — 调度器接受一批 bid 的决策结果

CompressionBid 字段三层体系
---------------------------
- **Layer 1（Solver 核心）**：``tokens_freed`` (r)、``quality_delta`` (δ) → Solver 直接使用
- **Layer 2（过滤/路由）**：``compress_latency_ms`` (t_exp)、``request_id`` → BidPool 过滤
- **Layer 3（可观测/扩展）**：``confidence``、``metadata`` → instrumentation + 未来扩展

Utility 函数
-------------
.. math::

    U(r, \\delta) = \\frac{r}{\\delta + \\varepsilon}, \\quad \\varepsilon = 10^{-3}

注意：``quality_delta (δ)`` = **predicted / surrogate** quality signal（非 ground-truth）。
``U = r / (δ + ε)`` = **operational ranking signal**（非 ground-truth user utility）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Feature Gate 标识（仅用于文档标注，不产生运行时副作用）
# ---------------------------------------------------------------------------

FEATURE_GATE_ID: str = "compress.scheduling_primitive.v1"
"""Feature gate 标识符。默认 OFF，由上层配置系统控制激活。"""

# Utility 函数参数
_UTILITY_EPSILON: float = 1e-3
"""Utility 函数分母修正项，防止 delta=0 时除零。"""


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionBid:
    """压缩层向调度器的一次报价。

    语义：以牺牲 ``quality_delta`` 质量为代价，释放 ``tokens_freed`` 个 token
    的 KV 空间。

    字段分层标注
    ------------
    - **Layer 1（Solver 核心）**：
        - ``tokens_freed`` (r)：预计释放 token 数 → Solver 直接使用
        - ``quality_delta`` (δ)：predicted/surrogate quality signal → Solver 直接使用
    - **Layer 2（过滤/路由）**：
        - ``request_id``：对应推理请求 → BidPool 过滤
        - ``compress_latency_ms`` (t_exp)：预计压缩耗时 → BidPool 过滤
    - **Layer 3（可观测/扩展）**：
        - ``confidence``：质量预测置信度 → instrumentation
        - ``metadata``：算法特定扩展字段 → 未来扩展

    Attributes
    ----------
    bid_id:
        全局唯一 bid 标识，建议格式 ``"{request_id}:bid:{level}"``。
    request_id:
        [Layer 2] 对应哪个推理请求的 KV cache。
    algorithm_id:
        使用的压缩算法标识。
    tokens_freed:
        [Layer 1 — r] 预计释放的 token 数量。必须 > 0。
    quality_delta:
        [Layer 1 — δ] 预测质量损失（0.0 = 无损，1.0 = 完全损失）。
        注意：这是 **predicted / surrogate** signal，非 ground-truth。
    compress_latency_ms:
        [Layer 2 — t_exp] 执行本次压缩的预计耗时（毫秒）。必须 >= 0.0。
    confidence:
        [Layer 3] 质量预测的置信度（0.0 ~ 1.0）。越高表示预测越可靠。
    metadata:
        [Layer 3] 算法特定的扩展字段。
    """

    bid_id: str
    request_id: str
    algorithm_id: str
    # Layer 1 — Solver 核心字段
    tokens_freed: int  # r: 释放 token 数
    quality_delta: float  # δ: predicted/surrogate quality signal (非 ground-truth)
    # Layer 2 — 过滤/路由字段
    compress_latency_ms: float  # t_exp: 预计压缩耗时
    # Layer 3 — 可观测/扩展字段
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """验证字段约束。"""
        if not self.bid_id:
            raise ValueError("bid_id must be a non-empty string")
        if not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not self.algorithm_id:
            raise ValueError("algorithm_id must be a non-empty string")
        if self.tokens_freed <= 0:
            raise ValueError(f"tokens_freed must be > 0, got {self.tokens_freed}")
        if self.quality_delta < 0.0:
            raise ValueError(f"quality_delta must be >= 0.0, got {self.quality_delta}")
        if self.compress_latency_ms < 0.0:
            raise ValueError(f"compress_latency_ms must be >= 0.0, got {self.compress_latency_ms}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def utility(self) -> float:
        """计算本 bid 的 utility 值（operational ranking signal, not ground-truth）。

        公式：:math:`U(r, \\delta) = r / (\\delta + \\varepsilon)`

        用于调度器在同一请求的多个 bid 间排序（utility 越高越优先）。
        """
        return self.tokens_freed / (self.quality_delta + _UTILITY_EPSILON)

    @property
    def normalized_utility(self) -> float:
        """归一化 utility（调度器跨请求比较时使用）。

        需外部传入 ``tokens_freed_max`` 时方可真正归一化；
        此处返回 ``utility``，由调度器负责归一化。
        """
        return self.utility


@dataclass(frozen=True)
class BidPool:
    """某时刻所有活跃请求的 bid 集合快照，供调度器查询。

    ``bids`` 列表建议按 ``tokens_freed`` 降序排列，方便调度器贪心选择。

    Attributes
    ----------
    snapshot_time_ns:
        快照生成时间戳（纳秒，单调时钟）。
    bids:
        所有活跃 bid 的列表。可为空（无可压缩请求时）。
    """

    snapshot_time_ns: int
    bids: tuple[CompressionBid, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.snapshot_time_ns < 0:
            raise ValueError(f"snapshot_time_ns must be >= 0, got {self.snapshot_time_ns}")

    def bids_for_request(self, request_id: str) -> tuple[CompressionBid, ...]:
        """返回指定请求的所有 bid，按 utility 降序排列。"""
        return tuple(
            sorted(
                (b for b in self.bids if b.request_id == request_id),
                key=lambda b: b.utility,
                reverse=True,
            )
        )

    def top_k_by_utility(self, k: int) -> tuple[CompressionBid, ...]:
        """返回 utility 最高的 k 个 bid（跨请求）。

        Parameters
        ----------
        k:
            最多返回的 bid 数量，必须 >= 1。
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        return tuple(sorted(self.bids, key=lambda b: b.utility, reverse=True)[:k])

    @property
    def total_tokens_available(self) -> int:
        """所有 bid 可释放的 token 总量。"""
        return sum(b.tokens_freed for b in self.bids)


@dataclass(frozen=True)
class BidAcceptance:
    """调度器接受一批 bid 的决策结果。

    Attributes
    ----------
    accepted_bid_ids:
        被接受的 bid id 列表。
    total_tokens_freed:
        所有被接受 bid 合计释放的 token 数量。
    total_quality_delta:
        所有被接受 bid 的质量损失之和（简化模型：Σδ）。
    decision_reason:
        触发本次接受决策的原因。
    """

    accepted_bid_ids: tuple[str, ...]
    total_tokens_freed: int
    total_quality_delta: float
    decision_reason: str

    def __post_init__(self) -> None:
        if self.total_tokens_freed < 0:
            raise ValueError(f"total_tokens_freed must be >= 0, got {self.total_tokens_freed}")
        if self.total_quality_delta < 0.0:
            raise ValueError(f"total_quality_delta must be >= 0.0, got {self.total_quality_delta}")
        if not self.decision_reason:
            raise ValueError("decision_reason must be a non-empty string")

    @property
    def accepted_count(self) -> int:
        """被接受的 bid 数量。"""
        return len(self.accepted_bid_ids)

    @property
    def is_empty(self) -> bool:
        """是否未接受任何 bid。"""
        return len(self.accepted_bid_ids) == 0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def compute_utility(tokens_freed: int, quality_delta: float) -> float:
    """计算 bid utility 值（operational ranking signal, not ground-truth）。

    .. math::

        U(r, \\delta) = \\frac{r}{\\delta + \\varepsilon}, \\quad \\varepsilon = 10^{-3}

    Parameters
    ----------
    tokens_freed:
        释放的 token 数量（对应 bid 中的 ``tokens_freed``）。
    quality_delta:
        预测质量损失（对应 bid 中的 ``quality_delta``）。

    Returns
    -------
    float
        Utility 值，越大表示"性价比"越高。
        注意：这是 operational ranking signal，非 ground-truth user utility。
    """
    if tokens_freed <= 0:
        raise ValueError(f"tokens_freed must be > 0, got {tokens_freed}")
    if not (0.0 <= quality_delta <= 1.0):
        raise ValueError(f"quality_delta must be in [0.0, 1.0], got {quality_delta}")
    return tokens_freed / (quality_delta + _UTILITY_EPSILON)


def make_bid_id(request_id: str, level: int) -> str:
    """生成标准格式的 bid_id。

    Parameters
    ----------
    request_id:
        请求 ID。
    level:
        压缩级别索引（从 0 开始，代表不同力度的压缩方案）。

    Returns
    -------
    str
        格式为 ``"{request_id}:bid:{level}"`` 的 bid ID。
    """
    if not request_id:
        raise ValueError("request_id must be a non-empty string")
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    return f"{request_id}:bid:{level}"
