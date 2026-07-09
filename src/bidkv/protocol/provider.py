"""CompressionBidProvider 接口协议定义。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bidkv.protocol.bid import CompressionBid


@runtime_checkable
class CompressionBidProvider(Protocol):
    """压缩层向调度器暴露 bid 能力的接口协议。

    实现要求
    --------
    - ``get_bids()``：纯查询，不修改状态；允许并发调用。
    - ``accept_bid()``：触发实际压缩；调度器应在持有 KV 池锁时调用以避免竞态。
    - 若 feature gate ``compress.scheduling_primitive.v1`` 为 OFF，
      ``get_bids()`` 应返回空列表，``accept_bid()`` 应为无操作（no-op）。

    实现者须知
    ----------
    本接口为结构化子类型（structural subtyping），无需继承此类，
    只需实现对应方法签名即可。
    """

    def get_bids(
        self,
        request_id: str,
        *,
        min_tokens: int = 0,
        max_delta: float = 1.0,
    ) -> list[CompressionBid]:
        """为指定请求生成可用的 bid 列表。

        Parameters
        ----------
        request_id:
            目标请求 ID。
        min_tokens:
            调用方要求的最小释放 token 数（用于过滤低效 bid）。默认 0（不过滤）。
        max_delta:
            调用方可接受的最大质量损失上限。默认 1.0（不过滤）。

        Returns
        -------
        list[CompressionBid]
            符合条件的 bid 列表，按 utility 降序排列。若无可用 bid 则返回空列表。
        """
        ...

    def accept_bid(self, bid_id: str) -> None:
        """触发指定 bid 对应的实际压缩操作。

        Parameters
        ----------
        bid_id:
            要执行的 bid 唯一标识。

        Raises
        ------
        BidExpiredError
            若 bid 已过期。
        BidCapacityError
            若无法满足 bid 所声明的 ``tokens_freed`` 数量。
        BidExecutionError
            其他执行失败情况。
        """
        ...
