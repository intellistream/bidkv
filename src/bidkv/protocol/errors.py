"""CompressionBid 异常类型定义。

所有 bid 相关异常均继承自 :class:`CompressionBidError`，调用方可统一捕获。
"""

from __future__ import annotations


class CompressionBidError(Exception):
    """CompressionBid 操作基类异常。

    Attributes
    ----------
    bid_id:
        触发异常的 bid ID（可为 None 表示非特定 bid 的错误）。
    message:
        人类可读的错误描述。
    """

    def __init__(self, message: str, *, bid_id: str | None = None) -> None:
        super().__init__(message)
        self.bid_id = bid_id
        self.message = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, bid_id={self.bid_id!r})"


class BidExpiredError(CompressionBidError):
    """Bid 已过期（对应 KV 块已被其他操作修改）。

    触发场景：
    - 调度器在 ``get_bids()`` 和 ``accept_bid()`` 之间，目标 KV 块已被驱逐。
    - Bid 超出最大有效时间窗口（实现者自定义）。

    Rollback 语义：调度器应重新调用 ``get_bids()`` 获取最新 bid 集合。
    """


class BidCapacityError(CompressionBidError):
    """无法满足请求的最小 tokens_freed 要求。

    触发场景：
    - KV 池已满足压缩条件，但当前请求可压缩 token 数不足。
    - 压缩后实际释放量小于 bid 声明量（块对齐导致差异）。

    Attributes
    ----------
    requested_tokens:
        调用方期望释放的最小 token 数。
    available_tokens:
        实际可释放的 token 数。
    """

    def __init__(
        self,
        message: str,
        *,
        bid_id: str | None = None,
        requested_tokens: int = 0,
        available_tokens: int = 0,
    ) -> None:
        super().__init__(message, bid_id=bid_id)
        self.requested_tokens = requested_tokens
        self.available_tokens = available_tokens


class BidExecutionError(CompressionBidError):
    """accept_bid() 执行失败（通用执行异常）。

    触发场景：
    - 底层压缩算法抛出未预期异常。
    - KV 传输失败（PD 分离场景）。
    - 算法内部状态不一致。

    Attributes
    ----------
    cause:
        导致执行失败的原始异常（可选）。
    """

    def __init__(
        self,
        message: str,
        *,
        bid_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, bid_id=bid_id)
        self.cause = cause
