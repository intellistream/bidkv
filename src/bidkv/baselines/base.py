"""BaselineStrategy ABC, CompressionAction, RequestState, BaselineContext。

定义 baseline 策略的核心抽象和数据类型。所有 baseline 实现此 ABC。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RequestState:
    """某个活跃推理请求的快照状态（用于 baseline 决策）。

    Attributes
    ----------
    request_id:
        唯一请求标识。
    current_tokens:
        该请求当前在 KV cache 中占用的 token 数量。
    priority:
        请求优先级（越高越重要）。默认 0.0。
    arrival_time_ms:
        请求到达时间戳（单调毫秒）。默认 0.0。
    deadline_ms:
        SLO 截止时间戳（单调毫秒）。None 表示无 SLO。
    token_ids:
        该请求的 token ID 序列（用于 scoring）。可为空。
    num_prompt_tokens:
        Prompt token 数量。默认 0。
    num_computed_tokens:
        已计算的总 token 数量（prompt + output so far）。默认 0。
    max_output_tokens:
        最大输出 token 数量（来自 sampling_params）。默认 0（未知）。
    num_preemptions:
        该请求被 preempt 的次数。默认 0。
    """

    request_id: str
    current_tokens: int
    priority: float = 0.0
    arrival_time_ms: float = 0.0
    deadline_ms: float | None = None
    token_ids: tuple[int, ...] = ()
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    max_output_tokens: int = 0
    num_preemptions: int = 0
    private_tokens: int = 0
    """SGLang radix-tree-aware private token count.

    Number of this request's KV tokens that are NOT shared with any other
    request in the radix tree (lock_ref == 1 nodes only).

    - 0  = unknown / not available (vLLM, or SGLang without tree access)
    - >0 = actual private token count; used by BidKV as tokens_freed
           and as basis for recompute cost estimation.

    Populated by scheduler_hook._build_running_candidates() when the
    SGLang scheduler and radix tree are accessible.
    """


@dataclass(frozen=True)
class CompressionAction:
    """baseline 策略选择的一个压缩/驱逐操作。

    Attributes
    ----------
    request_id:
        目标请求 ID。
    action_type:
        操作类型：``"evict"``（驱逐整个请求）或 ``"compress"``（部分压缩）。
    target_tokens:
        需要从该请求释放的 token 数量。
        对于 evict：等于该请求的全部 token。
        对于 compress：部分 token 数。
    metadata:
        策略特定的附加信息。
    """

    request_id: str
    action_type: str  # "evict" | "compress"
    target_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_type not in ("evict", "compress"):
            raise ValueError(f"action_type must be 'evict' or 'compress', got {self.action_type!r}")
        if self.target_tokens <= 0:
            raise ValueError(f"target_tokens must be > 0, got {self.target_tokens}")


@runtime_checkable
class BaselineContext(Protocol):
    """baseline 策略所需的框架适配器接口。

    实际运行时由 FrameworkAdapter（#044/#045）实现此 Protocol。
    测试时可直接构造符合此接口的对象。
    """

    def get_active_requests(self) -> list[RequestState]:
        """返回当前所有活跃请求的快照列表。"""
        ...

    def get_kv_stats(self) -> tuple[int, int]:
        """返回 (used_tokens, max_tokens)。"""
        ...


class BaselineStrategy(ABC):
    """Baseline 调度策略抽象基类，用于对比 BidKV。

    所有 baseline 通过 ``select_victims()`` 方法决策：
    给定候选请求列表和需要释放的 token 数，返回需执行的操作列表。

    **Candidate-universe consistency**: 所有 baseline 在同一 pressure event
    中使用同一候选池（``candidates`` 参数），保证 within-platform 公平性。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（用于 registry 和日志）。"""

    @abstractmethod
    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """选择要压缩/驱逐的请求，返回操作列表。

        Parameters
        ----------
        candidates:
            当前所有可压缩的候选请求快照。所有 baseline 在同一 pressure event
            中接收同一列表（candidate-universe consistency）。
        needed_tokens:
            需要释放的最小 KV token 数量。
        **kwargs:
            策略特定参数。子类可定义额外参数（如 scoring、bids 等）。

        Returns
        -------
        list[CompressionAction]
            需执行的操作列表。可为空（无法满足需求时）。
        """
