"""BidKV baseline — 完整 bid 机制 + utility greedy。

这是 BidKV 的完整策略包装器（作为 baseline 接口的适配器）。
scorer-agnostic：支持任意实现 ScoringStrategy 的评分器，
默认使用 PositionalScoring。

选择公式：U = current_tokens / (δ + ε)，greedy by U（Algorithm 1）。

Mode A 语义（request-level preemption）
--------------------------------------
SGLANG:
δ = recompute_norm + late_penalty + starvation

- tokens_freed = output_tokens（decode 阶段 KV）
  prompt KV 在 SGLang RadixAttention 中为 prefix-cached 共享，evict
  时实际只释放 output 阶段的 KV；用 output_tokens 避免高估长 prompt 请求。
- recompute_norm = prompt_tokens / 256（prompt 长度归一化）
  prefill 重算代价正比于 prompt 长度，长 prompt 请求 δ 增大 → U 降低 →
  BidKV 主动回避高重算代价的驱逐决策。
- late_penalty = completion × 2（接近完成的请求获得保护）
  completion = output_tokens / max_output_tokens；接近完成时 δ 线性增大，
  保护"很快就能自然释放 KV"的请求不被意外驱逐。
- starvation = num_preemptions × 0.5（anti-starvation）
  被多次 preempt 的请求 δ 递增，防止 cascading 连续驱逐同一请求。

Fallback（num_computed_tokens 不可用时）：
  回退到简化公式 U = current_tokens / (1 + starvation)。
VLLM：
δ = 1 + 0.5·completion + 0.3·num_preemptions（v8-frozen formula）

- tokens_freed = current_tokens（prompt + output 全部 KV，vLLM recompute-from-scratch）
- completion = output_tokens / max_output_tokens，衡量已完成比例
  接近完成的请求 δ 增大 → U 降低 → 避免临近结束的请求被驱逐
- anti-starvation = num_preemptions × 0.3
  被多次 preempt 的请求 δ 递增，防止 cascading 连续驱逐同一请求
- δ ∈ [1.0, 1.8]（completion 0→1，starvation=1 时），freed 强主导排序

注意：vLLM Mode A 采用 recompute-from-scratch，因此 current_tokens（整个 KV
footprint）是实际释放量；无 prefix cache 共享，prompt KV 也会被释放。
"""

from __future__ import annotations

import os
from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.pool import BidPoolManager
from bidkv.protocol.bid import CompressionBid, make_bid_id
from bidkv.scoring import PositionalScoring, ScoringStrategy
from bidkv.solver import GreedyBidSolver, SolverConfig

# ---------------------------------------------------------------------------
# Sensitivity analysis: environment variable overrides for δ parameters.
# These allow running sensitivity experiments without code changes.
# Default values match the published formula.
# ---------------------------------------------------------------------------
_COMPLETION_WEIGHT = float(os.environ.get("BIDKV_COMPLETION_WEIGHT", "2.0"))
_STARVATION_WEIGHT = float(os.environ.get("BIDKV_STARVATION_WEIGHT", "0.5"))
_RECOMPUTE_DIV = float(os.environ.get("BIDKV_RECOMPUTE_DIV", "256.0"))
_RECOMPUTE_FLOOR = float(os.environ.get("BIDKV_RECOMPUTE_FLOOR", "0.5"))
# "default" = normal formula, "freed-only" = δ=1 constant, "no-recompute" = recompute=1
_DELTA_MODE = os.environ.get("BIDKV_DELTA_MODE", "default")


class BidKVStrategy(BaselineStrategy):
    """BidKV 完整策略：scoring → bid → pool → solver。

    Mode A 使用 U = current_tokens / (1 + 0.5·completion + 0.3·P + ε) 排序，
    freed 强主导、completion 提供 ≤1.5× 轻量保护，δ ∈ [1.0, 1.8]。

    Parameters
    ----------
    scoring:
        ScoringStrategy 实例。若为 None，使用 PositionalScoring 默认配置创建。
    delta_budget:
        质量损失上限。默认 0.15。
    """

    def __init__(
        self,
        *,
        scoring: ScoringStrategy | None = None,
        delta_budget: float = 0.15,
    ) -> None:
        self._scoring: ScoringStrategy = scoring or PositionalScoring()
        self._delta_budget = delta_budget
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
        **kwargs: Any,  # noqa: ARG002
    ) -> list[CompressionAction]:
        """Mode A: 质量感知的请求级驱逐排序。

        U = output_tokens / (recompute_norm + late_penalty + starvation + ε)

        - output_tokens = num_computed - num_prompt（仅 decode phase KV）
        - recompute_norm = prompt_tokens / 256（重算代价归一化）
        - late_penalty = completion × 2（保护接近完成的请求）
        - starvation = num_preemptions × 0.5（防止 cascading 驱逐）

        当 num_computed_tokens == 0（信息不可用）时，回退到简化公式：
        U = current_tokens / (1 + starvation)。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``delta_budget``：覆盖默认值。

        Returns
        -------
        list[CompressionAction]
            按 utility 降序排列的驱逐操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        pool_mgr = BidPoolManager(enabled=True)

        for req in candidates:
            if req.current_tokens <= 1:
                continue

# SGLANG formula
#             output_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
#
#             if output_tokens > 2:
#                 tokens_freed = output_tokens
#
#                 if _DELTA_MODE == "freed-only":
#                     quality_delta = 1.0
#                     recompute_norm = 1.0
#                     completion = 0.0
#                     late_penalty = 0.0
#                     starvation_penalty = 0.0
#                 else:
#                     if _DELTA_MODE == "no-recompute":
#                         recompute_norm = 1.0
#                     else:
#                         recompute_norm = max(
#                             _RECOMPUTE_FLOOR, req.num_prompt_tokens / _RECOMPUTE_DIV
#                         )
#                     completion = 0.0
#                     if req.max_output_tokens > 0:
#                         completion = min(1.0, output_tokens / req.max_output_tokens)
#                     late_penalty = completion * _COMPLETION_WEIGHT
#                     starvation_penalty = req.num_preemptions * _STARVATION_WEIGHT
#                     quality_delta = max(0.1, recompute_norm + late_penalty + starvation_penalty)
#
#                 metadata: dict = {
#                     "output_tokens": output_tokens,
#                     "recompute_norm": round(recompute_norm, 4),
#                     "completion": round(completion, 4),
#                     "num_preemptions": req.num_preemptions,
#                     "mode": "A",
#                     "path": "primary",
#                 }
#             else:
#                 tokens_freed = req.current_tokens
#                 quality_delta = max(0.1, 1.0 + req.num_preemptions * 0.5)
#                 metadata = {
#                     "output_tokens": output_tokens,
#                     "completion": 0.0,
#                     "num_preemptions": req.num_preemptions,
#                     "mode": "A",
#                     "path": "fallback",
#                 }

            # ----------------------------------------------------------------
            # Dual-branch formula
            #
            # Branch A — radix-aware (SGLang, when private_tokens > 0):
            #   U = private_tokens / (δ + ε)
            #   δ = max(0.1, private_tokens/RECOMPUTE_DIV + completion·CW + P·SW)
            #   • private_tokens: only tokens NOT shared by any other request
            #     in the radix tree — the true freed amount on eviction
            #   • recompute cost ∝ private_tokens (shared prefix stays warm)
            #
            # Branch B — v8-frozen fallback (vLLM, or SGLang without tree access):
            #   U = current_tokens / (δ + ε)
            #   δ = 1 + 0.5·completion + 0.3·P
            # ----------------------------------------------------------------
            if req.private_tokens > 0:
                # --- Branch A: radix-tree-aware ---
                tokens_freed = req.private_tokens
                recompute_norm = max(_RECOMPUTE_FLOOR, req.private_tokens / _RECOMPUTE_DIV)
                output_tokens = max(0, req.num_computed_tokens - req.num_prompt_tokens)
                completion = 0.0
                if req.max_output_tokens > 0:
                    completion = min(1.0, output_tokens / req.max_output_tokens)
                late_penalty = completion * _COMPLETION_WEIGHT
                starvation_penalty = req.num_preemptions * _STARVATION_WEIGHT
                quality_delta = max(0.1, recompute_norm + late_penalty + starvation_penalty)
                metadata: dict = {
                    "private_tokens": req.private_tokens,
                    "recompute_norm": round(recompute_norm, 4),
                    "completion": round(completion, 4),
                    "num_preemptions": req.num_preemptions,
                    "mode": "A",
                    "path": "radix-aware",
                }
            else:
                # --- Branch B: v8-frozen fallback ---
                tokens_freed = req.current_tokens
                output_generated = max(0, req.num_computed_tokens - req.num_prompt_tokens)
                completion = 0.0
                if req.max_output_tokens > 0:
                    completion = min(1.0, output_generated / req.max_output_tokens)
                quality_delta = 1.0 + 0.5 * completion
                if req.num_preemptions > 0:
                    quality_delta += req.num_preemptions * 0.3
                metadata = {
                    "completion": round(completion, 4),
                    "num_preemptions": req.num_preemptions,
                    "mode": "A",
                    "path": "v8-frozen",
                }

            if tokens_freed <= 0:
                continue

            bid = CompressionBid(
                bid_id=make_bid_id(req.request_id, 0),
                request_id=req.request_id,
                algorithm_id="bidkv",
                tokens_freed=tokens_freed,
                quality_delta=quality_delta,
                compress_latency_ms=0.0,
                confidence=0.8,
                metadata=metadata,
            )
            pool_mgr.submit_bids(req.request_id, [bid])

        pool = pool_mgr.get_pool_snapshot()
        if not pool.bids:
            return []

        # Relaxed delta budget for Mode A: rank ALL candidates
        # (delta_budget only constrains how many the solver picks, but we
        # want a complete ordering for the priority cache)
        total_delta = sum(b.quality_delta for b in pool.bids)
        mode_a_budget = max(total_delta + 1.0, 100.0)

        acceptance = self._solver.solve(
            pool,
            needed_tokens,
            mode_a_budget,
            decision_reason="baseline_bidkv",
        )

        if acceptance.is_empty:
            return []

        bid_index = {b.bid_id: b for b in pool.bids}
        actions: list[CompressionAction] = []
        for bid_id in acceptance.accepted_bid_ids:
            bid = bid_index.get(bid_id)
            if bid is None:
                continue
            actions.append(
                CompressionAction(
                    request_id=bid.request_id,
                    action_type="evict",
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
