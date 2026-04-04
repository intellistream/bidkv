"""Global-NoBid baseline — 系统自动推断 utility，无用户 bid。

**关键归因 baseline**：对比 Global-NoBid → BidKV 可揭示
bid 接口（用户显式偏好）的增量价值。

设计理由：Global-NoBid 与 BidKV 使用相同的 PositionalScoring 评分和
相同的 utility-ratio 贪心算法，但 **系统自动推断 utility 并直接
做出压缩决策**（不暴露 bid 接口给用户）。
如果 BidKV > Global-NoBid，则证明用户显式 bid 比系统推断更有价值。

与 BidKV 的精确差异：
- BidKV：scoring → build_bids → BidPoolManager → GreedyBidSolver
- Global-NoBid：scoring → 直接贪心选择（无 BidPool / Solver 协议）
- 评分策略、compression_levels、delta_budget、贪心算法完全相同

选择公式：
- U_sys = r / (δ_H2O + ε)，其中 δ_H2O 由 PositionalScoring 估算
- 多级 compression levels：与 BidKV 相同 (0.2, 0.4, 0.6)
- All options 混合按 U_sys 贪心选择，每 request 最多选 1 级
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.protocol.bid import _UTILITY_EPSILON
from bidkv.scoring import PositionalScoring


class GlobalNoBidStrategy(BaselineStrategy):
    """Global-NoBid：系统推断 utility + 多级 greedy 选择（无 bid 接口）。

    流程：
    1. 对每个候选请求，用 PositionalScoring 评分
    2. 对每个 compression_level 估算 (tokens_freed, quality_delta)
    3. 计算系统推断 utility：U_sys = tokens_freed / (δ_H2O + ε)
    4. 将所有 (request, level) options 混合，按 U_sys 降序贪心选择
    5. 每个 request 最多选取 1 个 level（约束 A）
    6. Σδ ≤ delta_budget（约束 B）

    Parameters
    ----------
    scoring:
        PositionalScoring 实例。若为 None，使用默认配置创建。
    delta_budget:
        质量损失上限（Σδ ≤ delta_budget）。默认 0.15。
    compression_levels:
        系统尝试的压缩级别列表。默认与 BidKV 相同 (0.2, 0.4, 0.6)。
    """

    def __init__(
        self,
        *,
        scoring: PositionalScoring | None = None,
        delta_budget: float = 0.15,
        compression_levels: tuple[float, ...] = (0.2, 0.4, 0.6),
    ) -> None:
        self._scoring = scoring or PositionalScoring()
        self._delta_budget = delta_budget
        self._compression_levels = compression_levels

    @property
    def name(self) -> str:
        return "global-nobid"

    @property
    def scoring(self) -> PositionalScoring:
        """当前使用的 PositionalScoring 实例。"""
        return self._scoring

    @staticmethod
    def _completion_factor(req: RequestState) -> float:
        """Compute recompute-cost penalty for near-completion candidates.

        Quadratic ramp: 0% → 1.0×, 50% → 2.0×, 80% → 3.56×, 100% → 5.0×.
        """
        if req.max_output_tokens <= 0 or req.num_computed_tokens <= 0:
            return 1.0
        num_output = max(0, req.num_computed_tokens - req.num_prompt_tokens)
        completion = min(1.0, num_output / req.max_output_tokens)
        return 1.0 + completion * completion * 4.0

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """系统自动推断 utility 并贪心选择。

        对每个 candidate × 每个 compression_level 生成 option，
        混合按 utility 贪心，每 request 最多 1 option。
        算法与 GreedyBidSolver._greedy_solve 完全对齐。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``scoring_states``：dict[str, PositionalScoring]，
            每个请求独立的 PositionalScoring 实例。
            可选 ``delta_budget``：覆盖默认 delta_budget。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        scoring_states: dict[str, PositionalScoring] = kwargs.get("scoring_states", {})
        delta_budget = kwargs.get("delta_budget", self._delta_budget)

        # 为每个 candidate × 每个 compression_level 生成 option
        # option = (request_id, RequestState, tokens_freed, quality_delta, utility, level)
        all_options: list[tuple[str, RequestState, int, float, float, float]] = []
        for req in candidates:
            if req.current_tokens <= 1:
                continue

            scorer = scoring_states.get(req.request_id, self._scoring)
            scores = scorer.score(req.token_ids) if req.token_ids else []

            # Completion-aware penalty: inflate scores for near-completion requests
            completion_factor = self._completion_factor(req)
            if completion_factor > 1.0:
                scores = [min(1.0, s * completion_factor) for s in scores]

            for level in self._compression_levels:
                tokens_freed = max(1, int(req.current_tokens * level))
                quality_delta = self._estimate_delta(scores, tokens_freed, req.current_tokens)
                utility = tokens_freed / (quality_delta + _UTILITY_EPSILON)
                all_options.append(
                    (req.request_id, req, tokens_freed, quality_delta, utility, level)
                )

        # 按 utility 降序排序（与 GreedyBidSolver 一致）
        all_options.sort(key=lambda x: x[4], reverse=True)

        # 贪心选取（约束 A：每 request 最多 1；约束 B：Σδ ≤ budget）
        actions: list[CompressionAction] = []
        freed = 0
        total_delta = 0.0
        seen_requests: set[str] = set()

        for request_id, _req, tokens_to_free, delta, utility, level in all_options:
            if freed >= needed_tokens:
                break
            # 约束 A：每 request 最多 1 option
            if request_id in seen_requests:
                continue
            # 约束 B：delta_budget
            if total_delta + delta > delta_budget:
                continue

            actions.append(
                CompressionAction(
                    request_id=request_id,
                    action_type="compress",
                    target_tokens=tokens_to_free,
                    metadata={
                        "strategy": "global-nobid",
                        "system_utility": utility,
                        "estimated_quality_delta": delta,
                        "compression_level": level,
                    },
                )
            )
            freed += tokens_to_free
            total_delta += delta
            seen_requests.add(request_id)

        return actions

    def _estimate_delta(
        self, scores: list[float], tokens_to_compress: int, total_tokens: int
    ) -> float:
        """根据 H2O scoring 估算压缩的 quality delta。

        被压缩的 token 是重要度最低的。delta 等于被压缩 token 的平均重要度。
        """
        if not scores:
            # 无 scoring 数据时，用比例启发式
            ratio = tokens_to_compress / max(1, total_tokens)
            return min(1.0, ratio * 0.5)

        sorted_scores = sorted(scores)
        n_compress = min(tokens_to_compress, len(sorted_scores))
        if n_compress == 0:
            return 0.0
        return min(1.0, sum(sorted_scores[:n_compress]) / n_compress)
