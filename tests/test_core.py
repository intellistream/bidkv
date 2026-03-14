"""bidkv core 层单元测试 — BidPoolManager, GreedyBidSolver, PressureDetector, CompressionExecutor

测试覆盖范围：
- BidPoolManager：submit_bids CRUD、snapshot 一致性、feature gate、kill switch
- GreedyBidSolver：贪心选择（基于 U = r/(δ+ε) ranking）、delta_budget 约束、
  per-request 约束、feature gate、kill switch、candidate-universe consistency
- PressureDetector：阈值判断、高优先级触发、feature gate
- CompressionExecutor：Protocol 结构验证
"""

from __future__ import annotations

import pytest

from bidkv.compression import CompressionExecutor
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.protocol.bid import (
    BidPool,
    CompressionBid,
)
from bidkv.solver import ExecutionResult, GreedyBidSolver, SolverConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bid(
    request_id: str = "req-1",
    level: int = 0,
    tokens_freed: int = 100,
    quality_delta: float = 0.05,
    compress_latency_ms: float = 1.0,
    confidence: float = 0.9,
) -> CompressionBid:
    """构造测试用 CompressionBid。"""
    return CompressionBid(
        bid_id=f"{request_id}:bid:{level}",
        request_id=request_id,
        algorithm_id="test_algo",
        tokens_freed=tokens_freed,
        quality_delta=quality_delta,
        compress_latency_ms=compress_latency_ms,
        confidence=confidence,
    )


# ===========================================================================
# BidPoolManager Tests
# ===========================================================================


class TestBidPoolManagerFeatureOff:
    """Feature OFF 路径：所有方法为 no-op / 返回空。"""

    def test_default_off(self) -> None:
        mgr = BidPoolManager()
        assert not mgr.is_active

    def test_submit_noop_when_off(self) -> None:
        mgr = BidPoolManager(enabled=False)
        bids = [_make_bid()]
        mgr.submit_bids("req-1", bids)
        assert mgr.total_bid_count == 0

    def test_snapshot_empty_when_off(self) -> None:
        mgr = BidPoolManager(enabled=False)
        snap = mgr.get_pool_snapshot()
        assert snap.bids == ()
        assert snap.snapshot_time_ns == 0

    def test_get_bid_none_when_off(self) -> None:
        mgr = BidPoolManager(enabled=False)
        assert mgr.get_bid("any-id") is None

    def test_get_bids_for_request_empty_when_off(self) -> None:
        mgr = BidPoolManager(enabled=False)
        assert mgr.get_bids_for_request("req-1") == []


class TestBidPoolManagerFeatureOn:
    """Feature ON 路径：bid CRUD 操作。"""

    def test_submit_and_snapshot(self) -> None:
        mgr = BidPoolManager(enabled=True)
        bids = [_make_bid("req-1", 0, 200, 0.1), _make_bid("req-1", 1, 100, 0.05)]
        mgr.submit_bids("req-1", bids)

        assert mgr.active_request_count == 1
        assert mgr.total_bid_count == 2

        snap = mgr.get_pool_snapshot()
        assert len(snap.bids) == 2
        # 按 tokens_freed 降序
        assert snap.bids[0].tokens_freed >= snap.bids[1].tokens_freed

    def test_submit_replaces_old_bids(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0, 100, 0.05)])
        assert mgr.total_bid_count == 1

        # 替换
        mgr.submit_bids("req-1", [_make_bid("req-1", 0, 200, 0.1), _make_bid("req-1", 1, 50, 0.01)])
        assert mgr.total_bid_count == 2

        # 旧 bid 不应在索引中
        snap = mgr.get_pool_snapshot()
        freed_values = [b.tokens_freed for b in snap.bids]
        assert 100 not in freed_values

    def test_multi_request(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0, 100, 0.05)])
        mgr.submit_bids("req-2", [_make_bid("req-2", 0, 200, 0.1)])

        assert mgr.active_request_count == 2
        assert mgr.total_bid_count == 2

    def test_get_bid_lookup(self) -> None:
        mgr = BidPoolManager(enabled=True)
        bid = _make_bid("req-1", 0, 100, 0.05)
        mgr.submit_bids("req-1", [bid])
        found = mgr.get_bid(bid.bid_id)
        assert found is not None
        assert found.bid_id == bid.bid_id

    def test_get_bid_not_found(self) -> None:
        mgr = BidPoolManager(enabled=True)
        assert mgr.get_bid("nonexistent") is None

    def test_get_bids_for_request(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0)])
        mgr.submit_bids("req-2", [_make_bid("req-2", 0)])

        bids = mgr.get_bids_for_request("req-1")
        assert len(bids) == 1
        assert bids[0].request_id == "req-1"

    def test_remove_by_request(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0), _make_bid("req-1", 1, 50, 0.02)])
        assert mgr.total_bid_count == 2

        count = mgr.remove_by_request("req-1")
        assert count == 2
        assert mgr.total_bid_count == 0

    def test_remove_nonexistent_request(self) -> None:
        mgr = BidPoolManager(enabled=True)
        assert mgr.remove_by_request("nonexistent") == 0

    def test_invalidate(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0)])
        mgr.invalidate("req-1")
        assert mgr.total_bid_count == 0

    def test_invalidate_all(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid("req-1", 0)])
        mgr.submit_bids("req-2", [_make_bid("req-2", 0)])
        mgr.invalidate_all()
        assert mgr.total_bid_count == 0
        assert mgr.active_request_count == 0

    def test_snapshot_time_ns_positive(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid()])
        snap = mgr.get_pool_snapshot()
        assert snap.snapshot_time_ns > 0


class TestBidPoolManagerKillSwitch:
    """Kill switch 行为。"""

    def test_kill_switch_clears_bids(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid()])
        assert mgr.total_bid_count == 1

        mgr.activate_kill_switch()
        assert not mgr.is_active
        assert mgr.total_bid_count == 0

    def test_kill_switch_blocks_submit(self) -> None:
        mgr = BidPoolManager(enabled=True, kill_switch=True)
        mgr.submit_bids("req-1", [_make_bid()])
        assert mgr.total_bid_count == 0

    def test_enable_after_kill_switch(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.activate_kill_switch()
        assert not mgr.is_active

        mgr.enable()
        assert mgr.is_active

    def test_disable_clears_bids(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid()])
        mgr.disable()
        assert not mgr.is_active
        assert mgr.total_bid_count == 0


class TestBidPoolManagerStats:
    """get_stats() 返回。"""

    def test_stats_structure(self) -> None:
        mgr = BidPoolManager(enabled=True)
        mgr.submit_bids("req-1", [_make_bid()])
        stats = mgr.get_stats()
        assert stats["enabled"] is True
        assert stats["kill_switch"] is False
        assert stats["is_active"] is True
        assert stats["active_requests"] == 1
        assert stats["total_bids"] == 1


# ===========================================================================
# GreedyBidSolver Tests
# ===========================================================================


class TestGreedyBidSolverFeatureOff:
    """Feature OFF / kill switch 路径。"""

    def test_default_off(self) -> None:
        solver = GreedyBidSolver()
        pool = BidPool(snapshot_time_ns=1, bids=(_make_bid(),))
        result = solver.solve(pool, 50)
        assert result.is_empty
        assert "feature_off" in result.decision_reason

    def test_kill_switch(self) -> None:
        config = SolverConfig(enabled=True, kill_switch=True)
        solver = GreedyBidSolver(config)
        pool = BidPool(snapshot_time_ns=1, bids=(_make_bid(),))
        result = solver.solve(pool, 50)
        assert result.is_empty
        assert "kill_switch" in result.decision_reason

    def test_kill_switch_higher_priority(self) -> None:
        """kill_switch=True 优先级高于 enabled=True。"""
        config = SolverConfig(enabled=True, kill_switch=True)
        solver = GreedyBidSolver(config)
        pool = BidPool(snapshot_time_ns=1, bids=(_make_bid(),))
        result = solver.solve(pool, 50)
        assert "kill_switch" in result.decision_reason


class TestGreedyBidSolverEmptyPool:
    """空 pool 或零 tokens 场景。"""

    def test_empty_pool(self) -> None:
        config = SolverConfig(enabled=True)
        solver = GreedyBidSolver(config)
        pool = BidPool(snapshot_time_ns=1, bids=())
        result = solver.solve(pool, 100)
        assert result.is_empty
        assert "empty_pool" in result.decision_reason

    def test_zero_tokens_needed(self) -> None:
        config = SolverConfig(enabled=True)
        solver = GreedyBidSolver(config)
        pool = BidPool(snapshot_time_ns=1, bids=(_make_bid(),))
        result = solver.solve(pool, 0)
        assert result.is_empty
        assert "no_tokens_needed" in result.decision_reason


class TestGreedyBidSolverOptimalSelection:
    """贪心最优选择（基于 U = r/(δ+ε) ranking）。"""

    def test_single_bid_selection(self) -> None:
        config = SolverConfig(enabled=True, delta_budget=0.5)
        solver = GreedyBidSolver(config)
        bid = _make_bid("req-1", 0, 200, 0.1)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        result = solver.solve(pool, 100)
        assert result.accepted_count == 1
        assert bid.bid_id in result.accepted_bid_ids
        assert result.total_tokens_freed == 200
        assert abs(result.total_quality_delta - 0.1) < 1e-6

    def test_utility_ordering(self) -> None:
        """bid 按 utility 降序选取，而非 tokens_freed 降序。"""
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        # bid_a: U = 200/(0.1+0.001) ≈ 1980
        bid_a = _make_bid("req-a", 0, 200, 0.1)
        # bid_b: U = 150/(0.01+0.001) ≈ 13636 → 更优
        bid_b = _make_bid("req-b", 0, 150, 0.01)
        pool = BidPool(snapshot_time_ns=1, bids=(bid_a, bid_b))

        result = solver.solve(pool, 300)
        assert result.accepted_count == 2
        # bid_b 应先被选取（utility 更高）
        assert result.accepted_bid_ids[0] == bid_b.bid_id

    def test_multi_bid_selection(self) -> None:
        """多个 bid 组合满足 tokens_needed。"""
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        bids = tuple(_make_bid(f"req-{i}", 0, 50, 0.02) for i in range(5))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 150)
        assert result.total_tokens_freed >= 150
        assert result.accepted_count == 3

    def test_early_exit_when_satisfied(self) -> None:
        """满足 tokens_needed 后提前退出。"""
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        bids = tuple(_make_bid(f"req-{i}", 0, 100, 0.01) for i in range(10))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 200)
        # 需要 2 个 bid（每个 100 token）
        assert result.accepted_count == 2
        assert result.total_tokens_freed == 200


class TestGreedyBidSolverDeltaBudget:
    """delta_budget 约束测试。"""

    def test_budget_constraint(self) -> None:
        """Σδ 不得超过 delta_budget。"""
        config = SolverConfig(enabled=True, delta_budget=0.05)
        solver = GreedyBidSolver(config)

        # 每个 bid 的 delta = 0.03，budget = 0.05，所以最多选 1 个
        bids = tuple(_make_bid(f"req-{i}", 0, 100, 0.03) for i in range(3))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 300)
        assert result.accepted_count == 1
        assert result.total_quality_delta <= 0.05

    def test_budget_override(self) -> None:
        """调用方可覆盖 config.delta_budget。"""
        config = SolverConfig(enabled=True, delta_budget=0.01)
        solver = GreedyBidSolver(config)

        bids = tuple(_make_bid(f"req-{i}", 0, 100, 0.05) for i in range(3))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        # 用 config 的 budget (0.01) → 选不了（每个 delta=0.05 > 0.01）
        result = solver.solve(pool, 300)
        assert result.accepted_count == 0

        # 覆盖 budget 为 0.2 → 可以选 3 个
        result = solver.solve(pool, 300, delta_budget=0.2)
        assert result.accepted_count == 3

    def test_zero_delta_bids(self) -> None:
        """delta=0 的 bid 始终可选（不消耗 budget）。

        注意：CompressionBid 的 quality_delta 最小值为 0.0，
        但 utility 公式中 ε 防止除零。
        """
        config = SolverConfig(enabled=True, delta_budget=0.0)
        solver = GreedyBidSolver(config)

        bid = _make_bid("req-1", 0, 100, 0.0)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        result = solver.solve(pool, 50)
        assert result.accepted_count == 1
        assert result.total_quality_delta == 0.0


class TestGreedyBidSolverOnePerRequest:
    """每 request 最多 1 bid 约束。"""

    def test_one_bid_per_request(self) -> None:
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        # 同一 request 的两个 bid
        bids = (
            _make_bid("req-1", 0, 200, 0.05),
            _make_bid("req-1", 1, 100, 0.02),
        )
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 300)
        # 只能选 1 个（同一 request）
        assert result.accepted_count == 1

    def test_different_requests_both_selected(self) -> None:
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        bids = (
            _make_bid("req-1", 0, 200, 0.05),
            _make_bid("req-2", 0, 100, 0.02),
        )
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 300)
        assert result.accepted_count == 2


class TestGreedyBidSolverMaxBids:
    """max_bids_per_solve 限制。"""

    def test_max_bids_limit(self) -> None:
        config = SolverConfig(enabled=True, delta_budget=1.0, max_bids_per_solve=2)
        solver = GreedyBidSolver(config)

        bids = tuple(_make_bid(f"req-{i}", 0, 50, 0.01) for i in range(10))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        result = solver.solve(pool, 500)
        assert result.accepted_count <= 2


class TestGreedyBidSolverCandidateConsistency:
    """Candidate-universe consistency：Solver 在一次 solve() 中，
    所有候选 bid 来自同一 BidPool snapshot。"""

    def test_snapshot_consistency(self) -> None:
        """BidPool 是 frozen dataclass，保证 solve 期间 bids 不变。"""
        config = SolverConfig(enabled=True, delta_budget=1.0)
        solver = GreedyBidSolver(config)

        bids = tuple(_make_bid(f"req-{i}", 0, 100, 0.01) for i in range(5))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        # BidPool 是 frozen，无法修改
        with pytest.raises(AttributeError):
            pool.bids = ()  # type: ignore[misc]

        result = solver.solve(pool, 200)
        assert result.accepted_count == 2


class TestGreedyBidSolverDecisionReason:
    """decision_reason 传播。"""

    def test_default_reason(self) -> None:
        config = SolverConfig(enabled=True)
        solver = GreedyBidSolver(config)
        bid = _make_bid()
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))
        result = solver.solve(pool, 50)
        assert result.decision_reason == "kv_pool_pressure_threshold_exceeded"

    def test_custom_reason(self) -> None:
        config = SolverConfig(enabled=True)
        solver = GreedyBidSolver(config)
        bid = _make_bid()
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))
        result = solver.solve(pool, 50, decision_reason="custom_reason")
        assert result.decision_reason == "custom_reason"


class TestGreedyBidSolverUpdateConfig:
    """动态配置更新。"""

    def test_update_config(self) -> None:
        solver = GreedyBidSolver(SolverConfig(enabled=True))
        bid = _make_bid()
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        # 先正常 solve
        result = solver.solve(pool, 50)
        assert result.accepted_count == 1

        # 更新为 kill_switch
        solver.update_config(SolverConfig(enabled=True, kill_switch=True))
        result = solver.solve(pool, 50)
        assert result.is_empty
        assert "kill_switch" in result.decision_reason


class TestSolverConfig:
    """SolverConfig 验证。"""

    def test_default(self) -> None:
        config = SolverConfig()
        assert not config.enabled
        assert config.delta_budget == 0.15
        assert config.max_bids_per_solve == 20
        assert not config.kill_switch

    def test_invalid_delta_budget(self) -> None:
        with pytest.raises(ValueError, match="delta_budget"):
            SolverConfig(delta_budget=-0.1)
        with pytest.raises(ValueError, match="delta_budget"):
            SolverConfig(delta_budget=1.1)

    def test_invalid_max_bids(self) -> None:
        with pytest.raises(ValueError, match="max_bids_per_solve"):
            SolverConfig(max_bids_per_solve=0)

    def test_is_active(self) -> None:
        assert not SolverConfig().is_active
        assert SolverConfig(enabled=True).is_active
        assert not SolverConfig(enabled=True, kill_switch=True).is_active


# ===========================================================================
# PressureDetector Tests
# ===========================================================================


class TestPressureDetectorFeatureOff:
    """Feature OFF 路径。"""

    def test_default_off(self) -> None:
        detector = PressureDetector()
        assert not detector.is_under_pressure()

    def test_off_ignores_high_occupancy(self) -> None:
        detector = PressureDetector()
        detector.update_stats(used_tokens=950, max_tokens=1000)
        assert not detector.is_under_pressure()


class TestPressureDetectorThreshold:
    """KV 占用率阈值触发。"""

    def test_under_threshold(self) -> None:
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=800, max_tokens=1000)
        assert not detector.is_under_pressure()

    def test_at_threshold(self) -> None:
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=850, max_tokens=1000)
        assert detector.is_under_pressure()

    def test_above_threshold(self) -> None:
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()

    def test_zero_max_tokens(self) -> None:
        config = PressureConfig(enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=0, max_tokens=0)
        assert not detector.is_under_pressure()


class TestPressureDetectorHighPriority:
    """高优先级请求触发条件。"""

    def test_pending_high_priority_and_low_free(self) -> None:
        config = PressureConfig(threshold_pct=0.85, min_free_tokens=512, enabled=True)
        detector = PressureDetector(config)
        # 占用率不到 85%，但 free_tokens < 512 且有高优先级等待
        detector.update_stats(used_tokens=600, max_tokens=1000, pending_high_priority=1)
        assert detector.is_under_pressure()

    def test_pending_high_priority_with_enough_free(self) -> None:
        config = PressureConfig(threshold_pct=0.85, min_free_tokens=512, enabled=True)
        detector = PressureDetector(config)
        # 高优先级等待但 free_tokens 充足
        detector.update_stats(used_tokens=400, max_tokens=1000, pending_high_priority=1)
        assert not detector.is_under_pressure()

    def test_no_pending_high_priority(self) -> None:
        config = PressureConfig(threshold_pct=0.85, min_free_tokens=512, enabled=True)
        detector = PressureDetector(config)
        # free_tokens < 512 但无高优先级等待
        detector.update_stats(used_tokens=600, max_tokens=1000, pending_high_priority=0)
        assert not detector.is_under_pressure()


class TestPressureDetectorNeededTokens:
    """needed_tokens 估算。"""

    def test_no_pressure(self) -> None:
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=800, max_tokens=1000)
        assert detector.needed_tokens() == 0

    def test_over_threshold(self) -> None:
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=900, max_tokens=1000)
        # safe_threshold = 1000 * 0.85 = 850, gap = 900 - 850 = 50
        assert detector.needed_tokens() == 50

    def test_zero_max_tokens(self) -> None:
        detector = PressureDetector(PressureConfig(enabled=True))
        detector.update_stats(used_tokens=0, max_tokens=0)
        assert detector.needed_tokens() == 0


class TestPressureDetectorSetEnabled:
    """动态 feature gate 切换。"""

    def test_enable_disable(self) -> None:
        config = PressureConfig(threshold_pct=0.85, enabled=False)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert not detector.is_under_pressure()

        detector.set_enabled(True)
        assert detector.is_under_pressure()

        detector.set_enabled(False)
        assert not detector.is_under_pressure()


class TestPressureConfig:
    """PressureConfig 验证。"""

    def test_default(self) -> None:
        config = PressureConfig()
        assert config.threshold_pct == 0.85
        assert config.min_free_tokens == 512
        assert not config.enabled

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold_pct"):
            PressureConfig(threshold_pct=0.0)
        with pytest.raises(ValueError, match="threshold_pct"):
            PressureConfig(threshold_pct=1.5)

    def test_invalid_min_free_tokens(self) -> None:
        with pytest.raises(ValueError, match="min_free_tokens"):
            PressureConfig(min_free_tokens=-1)


# ===========================================================================
# CompressionExecutor Protocol Tests
# ===========================================================================


class TestCompressionExecutor:
    """CompressionExecutor Protocol 验证。"""

    def test_protocol_structural_subtyping(self) -> None:
        """无需继承即可实现。"""

        class MyExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return target_tokens

        executor = MyExecutor()
        assert isinstance(executor, CompressionExecutor)

    def test_protocol_non_conforming(self) -> None:
        """不匹配签名不应通过 isinstance 检查。"""

        class NotAnExecutor:
            def run(self, x: int) -> int:
                return x

        assert not isinstance(NotAnExecutor(), CompressionExecutor)


# ===========================================================================
# Integration: Pool → Solver → Pressure 联动
# ===========================================================================


class TestPoolSolverIntegration:
    """Pool 生成 snapshot → Solver 求解 联动。"""

    def test_pool_snapshot_fed_to_solver(self) -> None:
        pool_mgr = BidPoolManager(enabled=True)
        pool_mgr.submit_bids("req-1", [_make_bid("req-1", 0, 200, 0.05)])
        pool_mgr.submit_bids("req-2", [_make_bid("req-2", 0, 150, 0.03)])
        pool_mgr.submit_bids("req-3", [_make_bid("req-3", 0, 100, 0.01)])

        snapshot = pool_mgr.get_pool_snapshot()

        solver = GreedyBidSolver(SolverConfig(enabled=True, delta_budget=0.1))
        result = solver.solve(snapshot, 300)

        assert not result.is_empty
        assert result.total_tokens_freed > 0
        assert result.total_quality_delta <= 0.1

    def test_pressure_triggers_solve(self) -> None:
        """Pressure 检测 → 获取 snapshot → Solver 求解 完整流程。"""
        # 1. 设置压力检测
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()

        # 2. 获取需要释放的 token 数
        needed = detector.needed_tokens()
        assert needed == 50

        # 3. 从 pool 获取 snapshot
        pool_mgr = BidPoolManager(enabled=True)
        pool_mgr.submit_bids("req-1", [_make_bid("req-1", 0, 100, 0.02)])
        snapshot = pool_mgr.get_pool_snapshot()

        # 4. Solver 求解
        solver = GreedyBidSolver(SolverConfig(enabled=True))
        result = solver.solve(snapshot, needed)

        assert result.accepted_count == 1
        assert result.total_tokens_freed >= needed


# ===========================================================================
# Top-level imports
# ===========================================================================


class TestTopLevelImports:
    """验证从 bidkv 包级别导入核心模块。"""

    def test_import_pool_manager(self) -> None:
        from bidkv import BidPoolManager

        assert BidPoolManager is not None

    def test_import_solver(self) -> None:
        from bidkv import GreedyBidSolver, SolverConfig

        assert GreedyBidSolver is not None
        assert SolverConfig is not None

    def test_import_pressure(self) -> None:
        from bidkv import PressureConfig, PressureDetector

        assert PressureDetector is not None
        assert PressureConfig is not None

    def test_import_compression_executor(self) -> None:
        from bidkv import CompressionExecutor

        assert CompressionExecutor is not None


# ===========================================================================
# Fix S01 #018: PressureDetector 瞬时值（无 rolling window）
# ===========================================================================


class TestPressureDetectorInstantaneous:
    """Fix 1 (S01 #018): 压力检测必须基于瞬时值，不做平滑。"""

    def test_instant_response_to_spike(self) -> None:
        """瞬时占用率飙升必须立即触发压力。"""
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)

        # 初始低占用
        detector.update_stats(used_tokens=100, max_tokens=1000)
        assert not detector.is_under_pressure()

        # 瞬间飙升到 90% — 必须立即检测到
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()

    def test_instant_response_to_drop(self) -> None:
        """瞬时占用率回落必须立即解除压力。"""
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)

        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()

        # 瞬间降到 50% — 必须立即解除
        detector.update_stats(used_tokens=500, max_tokens=1000)
        assert not detector.is_under_pressure()

    def test_no_smoothing_effect(self) -> None:
        """连续更新不应产生平滑效果。"""
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)

        # 多次低占用
        for _ in range(10):
            detector.update_stats(used_tokens=100, max_tokens=1000)
        assert not detector.is_under_pressure()

        # 单次飙升 — 不受之前历史影响
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()


# ===========================================================================
# Fix S04 #021: ExecutionResult + execute_accepted
# ===========================================================================


class TestExecutionResult:
    """ExecutionResult 数据结构。"""

    def test_basic_fields(self) -> None:
        result = ExecutionResult(
            bid_id="req-1:bid:0",
            estimated_freed=200,
            actual_freed=180,
            success=True,
        )
        assert result.bid_id == "req-1:bid:0"
        assert result.estimated_freed == 200
        assert result.actual_freed == 180
        assert result.success is True

    def test_shortfall(self) -> None:
        result = ExecutionResult(
            bid_id="req-1:bid:0", estimated_freed=200, actual_freed=150, success=True
        )
        assert result.shortfall == 50

    def test_no_shortfall(self) -> None:
        result = ExecutionResult(
            bid_id="req-1:bid:0", estimated_freed=200, actual_freed=220, success=True
        )
        assert result.shortfall == 0

    def test_failed_execution(self) -> None:
        result = ExecutionResult(
            bid_id="req-1:bid:0", estimated_freed=200, actual_freed=0, success=False
        )
        assert result.shortfall == 200
        assert result.success is False

    def test_frozen(self) -> None:
        result = ExecutionResult(
            bid_id="req-1:bid:0", estimated_freed=200, actual_freed=180, success=True
        )
        with pytest.raises(AttributeError):
            result.actual_freed = 999  # type: ignore[misc]


class TestExecuteAccepted:
    """GreedyBidSolver.execute_accepted() — 记录 actual vs estimated。"""

    def test_execute_returns_actual(self) -> None:
        """executor 返回实际释放量。"""

        class FakeExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return target_tokens - 20  # 总是少释放 20

        solver = GreedyBidSolver(SolverConfig(enabled=True))
        bid = _make_bid("req-1", 0, 200, 0.05)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))
        acceptance = solver.solve(pool, 100)

        results = solver.execute_accepted(acceptance, pool, FakeExecutor())
        assert len(results) == 1
        assert results[0].estimated_freed == 200
        assert results[0].actual_freed == 180
        assert results[0].success is True
        assert results[0].shortfall == 20

    def test_execute_full_match(self) -> None:
        """actual == estimated 时 shortfall 为 0。"""

        class PerfectExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return target_tokens

        solver = GreedyBidSolver(SolverConfig(enabled=True))
        bid = _make_bid("req-1", 0, 200, 0.05)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))
        acceptance = solver.solve(pool, 100)

        results = solver.execute_accepted(acceptance, pool, PerfectExecutor())
        assert len(results) == 1
        assert results[0].shortfall == 0
        assert results[0].success is True

    def test_execute_failure(self) -> None:
        """executor 抛异常 → success=False, actual_freed=0。"""

        class FailingExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                raise RuntimeError("compress failed")

        solver = GreedyBidSolver(SolverConfig(enabled=True))
        bid = _make_bid("req-1", 0, 200, 0.05)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))
        acceptance = solver.solve(pool, 100)

        results = solver.execute_accepted(acceptance, pool, FailingExecutor())
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].actual_freed == 0
        assert results[0].estimated_freed == 200

    def test_execute_multi_bid(self) -> None:
        """多个 bid 的执行结果。"""

        class PartialExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return target_tokens // 2

        solver = GreedyBidSolver(SolverConfig(enabled=True, delta_budget=1.0))
        bids = (
            _make_bid("req-1", 0, 200, 0.05),
            _make_bid("req-2", 0, 100, 0.02),
        )
        pool = BidPool(snapshot_time_ns=1, bids=bids)
        acceptance = solver.solve(pool, 300)
        assert acceptance.accepted_count == 2

        results = solver.execute_accepted(acceptance, pool, PartialExecutor())
        assert len(results) == 2
        total_actual = sum(r.actual_freed for r in results)
        total_estimated = sum(r.estimated_freed for r in results)
        assert total_actual == total_estimated // 2

    def test_execute_empty_acceptance(self) -> None:
        """空 acceptance → 空结果列表。"""
        solver = GreedyBidSolver(SolverConfig(enabled=True))
        pool = BidPool(snapshot_time_ns=1, bids=())
        acceptance = solver.solve(pool, 100)  # empty pool → empty acceptance

        class DummyExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return 0

        results = solver.execute_accepted(acceptance, pool, DummyExecutor())
        assert results == []


# ===========================================================================
# Fix S07 #024: KV 统计唯一来源 + solve_with_detector
# ===========================================================================


class TestPressureDetectorGetKvStats:
    """PressureDetector.get_kv_stats() — KV 统计唯一来源。"""

    def test_kv_stats_structure(self) -> None:
        config = PressureConfig(threshold_pct=0.85, enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=800, max_tokens=1000, pending_high_priority=2)

        stats = detector.get_kv_stats()
        assert stats["used_tokens"] == 800
        assert stats["max_tokens"] == 1000
        assert stats["free_tokens"] == 200
        assert stats["pending_high_priority"] == 2

    def test_kv_stats_free_tokens_zero(self) -> None:
        config = PressureConfig(enabled=True)
        detector = PressureDetector(config)
        detector.update_stats(used_tokens=1000, max_tokens=1000)

        stats = detector.get_kv_stats()
        assert stats["free_tokens"] == 0

    def test_kv_stats_default(self) -> None:
        detector = PressureDetector()
        stats = detector.get_kv_stats()
        assert stats["used_tokens"] == 0
        assert stats["max_tokens"] == 0
        assert stats["free_tokens"] == 0


class TestSolveWithDetector:
    """GreedyBidSolver.solve_with_detector() — Solver 从 PressureDetector 获取 needed_tokens。"""

    def test_solve_uses_detector_needed_tokens(self) -> None:
        """Solver 使用 detector.needed_tokens() 而非独立计算。"""
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=900, max_tokens=1000)

        # needed = 900 - 850 = 50
        assert detector.needed_tokens() == 50

        bid = _make_bid("req-1", 0, 100, 0.02)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        solver = GreedyBidSolver(SolverConfig(enabled=True))
        result = solver.solve_with_detector(pool, detector)

        # Solver 应选取 bid 以满足 50 tokens needed
        assert result.accepted_count == 1
        assert result.total_tokens_freed >= 50

    def test_solve_with_detector_no_pressure(self) -> None:
        """无压力时 needed_tokens=0 → Solver 不选取。"""
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=500, max_tokens=1000)
        assert detector.needed_tokens() == 0

        bid = _make_bid("req-1", 0, 100, 0.02)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        solver = GreedyBidSolver(SolverConfig(enabled=True))
        result = solver.solve_with_detector(pool, detector)
        assert result.is_empty
        assert "no_tokens_needed" in result.decision_reason

    def test_solve_with_detector_consistency(self) -> None:
        """验证 Solver 和 Detector 使用同一 KV 状态口径。"""
        detector = PressureDetector(PressureConfig(threshold_pct=0.80, enabled=True))
        detector.update_stats(used_tokens=850, max_tokens=1000)

        # detector: needed = 850 - 800 = 50
        needed = detector.needed_tokens()
        assert needed == 50

        # Solver 通过 detector 获取 needed_tokens — 一致
        bids = tuple(_make_bid(f"req-{i}", 0, 30, 0.01) for i in range(5))
        pool = BidPool(snapshot_time_ns=1, bids=bids)

        solver = GreedyBidSolver(SolverConfig(enabled=True, delta_budget=1.0))
        result = solver.solve_with_detector(pool, detector)

        assert result.total_tokens_freed >= needed

    def test_solve_with_detector_feature_off(self) -> None:
        """Solver feature OFF 时，即使 detector 有压力也不选取。"""
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=900, max_tokens=1000)

        bid = _make_bid("req-1", 0, 100, 0.02)
        pool = BidPool(snapshot_time_ns=1, bids=(bid,))

        solver = GreedyBidSolver(SolverConfig(enabled=False))
        result = solver.solve_with_detector(pool, detector)
        assert result.is_empty


# ===========================================================================
# Integration: end-to-end with actual_freed tracking
# ===========================================================================


class TestEndToEndWithActualFreed:
    """完整流程：Pressure → Snapshot → Solve → Execute → actual_freed 校验。"""

    def test_full_flow(self) -> None:
        # 1. Pressure detector 检测到压力
        detector = PressureDetector(PressureConfig(threshold_pct=0.85, enabled=True))
        detector.update_stats(used_tokens=900, max_tokens=1000)
        assert detector.is_under_pressure()

        # 2. Pool 提供 bids
        pool_mgr = BidPoolManager(enabled=True)
        pool_mgr.submit_bids("req-1", [_make_bid("req-1", 0, 100, 0.02)])
        snapshot = pool_mgr.get_pool_snapshot()

        # 3. Solver 从 detector 获取 needed_tokens 并求解
        solver = GreedyBidSolver(SolverConfig(enabled=True))
        acceptance = solver.solve_with_detector(snapshot, detector)
        assert acceptance.accepted_count == 1

        # 4. Execute — 实际释放少于预估
        class PartialExecutor:
            def execute(self, request_id: str, target_tokens: int) -> int:  # noqa: ARG002
                return target_tokens - 30  # 只释放 70

        results = solver.execute_accepted(acceptance, snapshot, PartialExecutor())
        assert len(results) == 1
        assert results[0].estimated_freed == 100
        assert results[0].actual_freed == 70
        assert results[0].shortfall == 30

        # 5. 调用方可据此判断是否需要 fallback eviction
        total_actual = sum(r.actual_freed for r in results)
        needed = detector.needed_tokens()
        if total_actual < needed:
            # 需要 fallback eviction
            pass  # 调用方处理


class TestImportExecutionResult:
    """验证从 bidkv 包级别导入 ExecutionResult。"""

    def test_import(self) -> None:
        from bidkv import ExecutionResult

        assert ExecutionResult is not None
