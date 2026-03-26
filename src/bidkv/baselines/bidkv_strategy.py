"""BidKV baseline — 完整 bid 机制 + utility greedy。

这是 BidKV 的完整策略包装器（作为 baseline 接口的适配器）。
scorer-agnostic：支持任意实现 ScoringStrategy 的评分器，
默认使用 H2OScoring。

选择公式：U = r / (δ + ε)，greedy by U（Algorithm 1）。
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.pool import BidPoolManager
from bidkv.protocol.bid import CompressionBid
from bidkv.scoring import H2OScoring, ScoringStrategy
from bidkv.scoring.bid_builder import build_bids
from bidkv.solver import GreedyBidSolver, SolverConfig


class BidKVStrategy(BaselineStrategy):
    """BidKV 完整策略：scoring → bid → pool → solver。

    Parameters
    ----------
    scoring:
        ScoringStrategy 实例。若为 None，使用 H2OScoring 默认配置创建。
    delta_budget:
        质量损失上限。默认 0.15。
    compression_levels:
        生成 bid 时使用的压缩级别。默认 [0.2, 0.4, 0.6]。
    """

    def __init__(
        self,
        *,
        scoring: ScoringStrategy | None = None,
        delta_budget: float = 0.15,
        compression_levels: tuple[float, ...] = (0.2, 0.4, 0.6),
    ) -> None:
        self._scoring: ScoringStrategy = scoring or H2OScoring()
        self._delta_budget = delta_budget
        self._compression_levels = compression_levels
        self._solver = GreedyBidSolver(SolverConfig(enabled=True, delta_budget=delta_budget))

    @property
    def name(self) -> str:
        return "bidkv"

    @property
    def scoring(self) -> ScoringStrategy:
        """当前使用的评分策略实例。"""
        return self._scoring

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """使用完整 BidKV 流程：生成 bid → 池化 → 贪心求解。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``scoring_states``：dict[str, ScoringStrategy]。
            可选 ``delta_budget``：覆盖默认值。
            可选 ``bids_by_request``：dict[str, list[CompressionBid]]，
            预生成的 bid（用于 candidate-universe consistency）。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        scoring_states: dict[str, ScoringStrategy] = kwargs.get("scoring_states", {})
        delta_budget = kwargs.get("delta_budget", self._delta_budget)
        pre_bids: dict[str, list[CompressionBid]] | None = kwargs.get("bids_by_request")

        # 收集所有 bid 到 pool manager
        pool_mgr = BidPoolManager(enabled=True)

        if pre_bids is not None:
            # 使用预生成的 bid（candidate-universe consistency）
            for request_id, bids in pre_bids.items():
                pool_mgr.submit_bids(request_id, bids)
        else:
            # 为每个候选请求生成 bid（score → build_bids 统一链路）
            for req in candidates:
                if req.current_tokens <= 1 or not req.token_ids:
                    continue
                scorer = scoring_states.get(req.request_id, self._scoring)
                scores = scorer.score(req.token_ids)
                bids = build_bids(
                    request_id=req.request_id,
                    token_ids=req.token_ids,
                    scores=scores,
                    compression_levels=self._compression_levels,
                    algorithm_id="bidkv",
                )
                pool_mgr.submit_bids(req.request_id, bids)

        # 获取 pool 快照并求解
        pool = pool_mgr.get_pool_snapshot()
        acceptance = self._solver.solve(
            pool,
            needed_tokens,
            delta_budget,
            decision_reason="baseline_bidkv",
        )

        if acceptance.is_empty:
            return []

        # 将 acceptance 转换为 CompressionAction
        bid_index = {b.bid_id: b for b in pool.bids}
        actions: list[CompressionAction] = []
        for bid_id in acceptance.accepted_bid_ids:
            bid = bid_index.get(bid_id)
            if bid is None:
                continue
            actions.append(
                CompressionAction(
                    request_id=bid.request_id,
                    action_type="compress",
                    target_tokens=bid.tokens_freed,
                    metadata={
                        "strategy": "bidkv",
                        "bid_id": bid.bid_id,
                        "quality_delta": bid.quality_delta,
                        "utility": bid.utility,
                    },
                )
            )

        return actions
