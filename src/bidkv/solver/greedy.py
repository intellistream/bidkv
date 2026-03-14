"""GreedyBidSolver — 贪心 Knapsack 求解器（通用版）。

从 ``sagellm-control-plane`` 的 ``bid_solver.py`` 提取并通用化。
仅依赖 ``bidkv.protocol`` 类型，零 sagellm 依赖。

算法：Utility-ratio 贪心（Algorithm 1，论文 §4）
  - 对所有可用 bid 按 U = tokens_freed / (quality_delta + ε) 降序排列
  - 依次选取 bid，直到满足 tokens_needed 或 delta_budget 耗尽
  - 约束 A：同一 request 最多接受 1 个 bid
  - 约束 B：Σδ ≤ delta_budget
"""

from __future__ import annotations

import logging
import time

from bidkv.protocol.bid import _UTILITY_EPSILON, BidAcceptance, BidPool
from bidkv.solver.config import SolverConfig
from bidkv.solver.execution_result import ExecutionResult

logger = logging.getLogger(__name__)


class GreedyBidSolver:
    """贪心 Knapsack 求解器：在 SLO 质量约束下最大化释放 KV token。

    算法（论文 §4 · Algorithm 1）
    --------------------------------
    1. 对所有可用 bid 按 utility 降序排列。
    2. 依次遍历 bid：
       a. 若该 bid 对应的 request_id 已被选取 → 跳过（每请求最多 1 bid）。
       b. 若加入该 bid 后 Σδ > delta_budget → 跳过。
       c. 否则接受该 bid，累加 tokens_freed 和 quality_delta。
    3. 满足 tokens_needed 或遍历完所有 bid 后返回 :class:`BidAcceptance`。

    Feature Gate / Kill Switch
    --------------------------
    - ``config.kill_switch = True``：立即返回空 acceptance（最高优先级）。
    - ``config.enabled = False``：同样返回空 acceptance。

    Parameters
    ----------
    config:
        求解器配置，若为 None 则使用默认配置（feature OFF）。
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        self._config = config or SolverConfig()
        logger.debug(
            "GreedyBidSolver initialized: enabled=%s, delta_budget=%.3f, kill_switch=%s",
            self._config.enabled,
            self._config.delta_budget,
            self._config.kill_switch,
        )

    def solve(
        self,
        pool: BidPool,
        tokens_needed: int,
        delta_budget: float | None = None,
        *,
        decision_reason: str | None = None,
    ) -> BidAcceptance:
        """求解满足约束的最优 bid 组合。

        Solver 仅使用 Layer 1 字段（tokens_freed = r, quality_delta = δ）。

        Feature OFF 或 kill switch 激活时立即返回空 :class:`BidAcceptance`。

        Parameters
        ----------
        pool:
            当前时刻的 :class:`BidPool` 快照（candidate-universe consistency 由此保证）。
        tokens_needed:
            需要释放的最小 KV token 数量（>= 0）。
        delta_budget:
            本次求解的质量损失上限，覆盖 ``config.delta_budget``（可选）。
        decision_reason:
            BidAcceptance 中记录的触发原因，覆盖 ``config.decision_reason``（可选）。

        Returns
        -------
        BidAcceptance
            被选取的 bid 组合。若 pool 为空、无解或 feature OFF，返回空 acceptance。
        """
        effective_reason = decision_reason or self._config.decision_reason

        # Kill switch — 最高优先级
        if self._config.kill_switch:
            logger.debug("GreedyBidSolver: kill_switch=True, returning empty acceptance")
            return BidAcceptance(
                accepted_bid_ids=(),
                total_tokens_freed=0,
                total_quality_delta=0.0,
                decision_reason=f"{effective_reason}:kill_switch",
            )

        # Feature gate — 未激活
        if not self._config.enabled:
            logger.debug("GreedyBidSolver: enabled=False, returning empty acceptance")
            return BidAcceptance(
                accepted_bid_ids=(),
                total_tokens_freed=0,
                total_quality_delta=0.0,
                decision_reason=f"{effective_reason}:feature_off",
            )

        # tokens_needed 为 0 时，无需选取任何 bid
        if tokens_needed <= 0:
            return BidAcceptance(
                accepted_bid_ids=(),
                total_tokens_freed=0,
                total_quality_delta=0.0,
                decision_reason=f"{effective_reason}:no_tokens_needed",
            )

        # 空 pool 处理
        if not pool.bids:
            logger.debug("GreedyBidSolver: empty BidPool, returning empty acceptance")
            return BidAcceptance(
                accepted_bid_ids=(),
                total_tokens_freed=0,
                total_quality_delta=0.0,
                decision_reason=f"{effective_reason}:empty_pool",
            )

        effective_budget = delta_budget if delta_budget is not None else self._config.delta_budget

        t_start = time.monotonic()
        result = self._greedy_solve(pool, tokens_needed, effective_budget, effective_reason)
        elapsed_ms = (time.monotonic() - t_start) * 1_000.0

        if result.is_empty:
            logger.info(
                "GreedyBidSolver: no feasible solution "
                "(tokens_needed=%d, pool_size=%d, delta_budget=%.3f, elapsed_ms=%.2f)",
                tokens_needed,
                len(pool.bids),
                effective_budget,
                elapsed_ms,
            )
        else:
            logger.info(
                "GreedyBidSolver: accepted %d bids, freed=%d tokens, delta=%.4f, elapsed_ms=%.2f",
                result.accepted_count,
                result.total_tokens_freed,
                result.total_quality_delta,
                elapsed_ms,
            )

        return result

    def update_config(self, config: SolverConfig) -> None:
        """动态更新配置（用于 kill switch 热切换）。"""
        self._config = config
        logger.info(
            "GreedyBidSolver config updated: enabled=%s, delta_budget=%.3f, kill_switch=%s",
            config.enabled,
            config.delta_budget,
            config.kill_switch,
        )

    def solve_with_detector(
        self,
        pool: BidPool,
        detector: object,
        delta_budget: float | None = None,
        *,
        decision_reason: str | None = None,
    ) -> BidAcceptance:
        """从 PressureDetector 获取 needed_tokens 并求解（Fix S07 #024）。

        Solver 不独立计算 KV 占用；所有 KV 状态来自 PressureDetector。

        Parameters
        ----------
        pool:
            当前时刻的 BidPool 快照。
        detector:
            PressureDetector 实例，必须提供 ``needed_tokens()`` 方法。
        delta_budget:
            可选的质量损失上限覆盖。
        decision_reason:
            可选的触发原因覆盖。

        Returns
        -------
        BidAcceptance
            求解结果。若 detector 不处于压力态，返回空 acceptance。
        """
        tokens_needed = detector.needed_tokens()  # type: ignore[union-attr]
        return self.solve(pool, tokens_needed, delta_budget, decision_reason=decision_reason)

    def execute_accepted(
        self,
        acceptance: BidAcceptance,
        pool: BidPool,
        executor: object,
    ) -> list[ExecutionResult]:
        """执行已接受的 bid 并记录 actual vs estimated（Fix S04 #021）。

        对每个 accepted bid，调用 ``executor.execute(request_id, tokens_freed)``
        获取实际释放量，返回 :class:`ExecutionResult` 列表。

        Parameters
        ----------
        acceptance:
            ``solve()`` 返回的 BidAcceptance。
        pool:
            ``solve()`` 使用的同一 BidPool 快照（用于查找 bid 详情）。
        executor:
            实现 ``execute(request_id, target_tokens) -> int`` 的
            CompressionExecutor 实例。

        Returns
        -------
        list[ExecutionResult]
            每个 accepted bid 的执行结果。
        """
        if acceptance.is_empty:
            return []

        # 建立 bid_id → CompressionBid 索引
        bid_index = {b.bid_id: b for b in pool.bids}

        results: list[ExecutionResult] = []
        for bid_id in acceptance.accepted_bid_ids:
            bid = bid_index.get(bid_id)
            if bid is None:
                results.append(
                    ExecutionResult(
                        bid_id=bid_id,
                        estimated_freed=0,
                        actual_freed=0,
                        success=False,
                    )
                )
                continue

            try:
                actual = executor.execute(bid.request_id, bid.tokens_freed)  # type: ignore[union-attr]
                results.append(
                    ExecutionResult(
                        bid_id=bid_id,
                        estimated_freed=bid.tokens_freed,
                        actual_freed=actual,
                        success=True,
                    )
                )
            except Exception:
                logger.exception("execute_accepted: bid %s execution failed", bid_id)
                results.append(
                    ExecutionResult(
                        bid_id=bid_id,
                        estimated_freed=bid.tokens_freed,
                        actual_freed=0,
                        success=False,
                    )
                )

        total_actual = sum(r.actual_freed for r in results)
        total_estimated = sum(r.estimated_freed for r in results)
        if total_actual < total_estimated:
            logger.warning(
                "execute_accepted: actual_freed=%d < estimated=%d (shortfall=%d), "
                "caller should consider fallback eviction",
                total_actual,
                total_estimated,
                total_estimated - total_actual,
            )

        return results

    def _greedy_solve(
        self,
        pool: BidPool,
        tokens_needed: int,
        delta_budget: float,
        decision_reason: str,
    ) -> BidAcceptance:
        """Utility-ratio 贪心选取（Algorithm 1）。

        参数同 :meth:`solve`，此处保证 pool 非空、tokens_needed > 0。
        """
        # 1. 按 utility = tokens_freed / (quality_delta + ε) 降序排列
        sorted_bids = sorted(
            pool.bids,
            key=lambda b: b.tokens_freed / (b.quality_delta + _UTILITY_EPSILON),
            reverse=True,
        )

        accepted_ids: list[str] = []
        total_freed: int = 0
        total_delta: float = 0.0
        seen_requests: set[str] = set()  # 每 request 最多 1 bid

        for bid in sorted_bids[: self._config.max_bids_per_solve]:
            # 约束 A：同一 request 最多 1 bid
            if bid.request_id in seen_requests:
                continue

            # 约束 B：delta_budget 上限（加入后不超预算）
            if total_delta + bid.quality_delta > delta_budget:
                continue

            # 选取此 bid
            accepted_ids.append(bid.bid_id)
            total_freed += bid.tokens_freed
            total_delta += bid.quality_delta
            seen_requests.add(bid.request_id)

            # 满足 tokens_needed，提前退出
            if total_freed >= tokens_needed:
                break

        return BidAcceptance(
            accepted_bid_ids=tuple(accepted_ids),
            total_tokens_freed=total_freed,
            total_quality_delta=round(total_delta, 6),
            decision_reason=decision_reason,
        )
