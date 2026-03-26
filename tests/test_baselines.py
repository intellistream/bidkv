"""bidkv baselines 单元测试。

测试覆盖：
- BaselineStrategy ABC / CompressionAction / RequestState 数据类型
- BaselineRegistry 注册与获取
- 7 个 baseline 各 ≥ 3 个测试
- Candidate-universe consistency（所有 baseline 使用同一候选池）
- H2O-Style ≠ H2OScoring 区分验证
"""

from __future__ import annotations

import pytest

from bidkv.baselines import (
    BaselineRegistry,
    BaselineStrategy,
    BidKVStrategy,
    CompressionAction,
    GlobalNoBidStrategy,
    H2OStyleStrategy,
    PreemptEvictStrategy,
    RequestState,
    SlackAwareStrategy,
    StaticRandomStrategy,
    UniformStrategy,
)
from bidkv.protocol.bid import CompressionBid
from bidkv.scoring import H2OScoring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 5, tokens_per_req: int = 200) -> list[RequestState]:
    """生成 N 个候选请求。"""
    return [
        RequestState(
            request_id=f"req-{i}",
            current_tokens=tokens_per_req,
            priority=float(i),
            arrival_time_ms=float(i * 100),
            deadline_ms=float(10000 + i * 1000),
            token_ids=tuple(range(tokens_per_req)),
        )
        for i in range(n)
    ]


def _make_candidates_varied() -> list[RequestState]:
    """生成具有不同 token 数和优先级的候选请求。"""
    return [
        RequestState(
            "req-a",
            current_tokens=500,
            priority=1.0,
            arrival_time_ms=0.0,
            deadline_ms=20000.0,
            token_ids=tuple(range(500)),
        ),
        RequestState(
            "req-b",
            current_tokens=300,
            priority=3.0,
            arrival_time_ms=100.0,
            deadline_ms=15000.0,
            token_ids=tuple(range(300)),
        ),
        RequestState(
            "req-c",
            current_tokens=100,
            priority=2.0,
            arrival_time_ms=200.0,
            deadline_ms=25000.0,
            token_ids=tuple(range(100)),
        ),
        RequestState(
            "req-d",
            current_tokens=400,
            priority=0.5,
            arrival_time_ms=300.0,
            deadline_ms=12000.0,
            token_ids=tuple(range(400)),
        ),
    ]


def _make_bids_for_request(request_id: str, token_ids: tuple[int, ...]) -> list[CompressionBid]:
    """用 H2OScoring 为请求生成 bids。"""
    scorer = H2OScoring()
    return scorer.generate_bids(
        request_id,
        token_ids,
        [0.2, 0.4, 0.6],
    )


# ===========================================================================
# Data type tests
# ===========================================================================


class TestCompressionAction:
    """CompressionAction 数据类型测试。"""

    def test_valid_evict(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="evict", target_tokens=100)
        assert action.request_id == "req-1"
        assert action.action_type == "evict"
        assert action.target_tokens == 100

    def test_valid_compress(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="compress", target_tokens=50)
        assert action.action_type == "compress"

    def test_invalid_action_type(self) -> None:
        with pytest.raises(ValueError, match="action_type"):
            CompressionAction(request_id="req-1", action_type="delete", target_tokens=10)

    def test_invalid_target_tokens(self) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            CompressionAction(request_id="req-1", action_type="compress", target_tokens=0)

    def test_frozen(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="compress", target_tokens=10)
        with pytest.raises(AttributeError):
            action.target_tokens = 20  # type: ignore[misc]


class TestRequestState:
    """RequestState 数据类型测试。"""

    def test_basic(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=200)
        assert req.request_id == "req-1"
        assert req.current_tokens == 200
        assert req.priority == 0.0
        assert req.deadline_ms is None

    def test_with_deadline(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=100, deadline_ms=5000.0)
        assert req.deadline_ms == 5000.0

    def test_frozen(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=100)
        with pytest.raises(AttributeError):
            req.current_tokens = 300  # type: ignore[misc]


# ===========================================================================
# BaselineRegistry tests
# ===========================================================================


class TestBaselineRegistry:
    """BaselineRegistry 注册表测试。"""

    def test_register_and_get(self) -> None:
        registry = BaselineRegistry()
        strategy = PreemptEvictStrategy()
        registry.register(strategy)
        assert registry.get("preempt-evict") is strategy

    def test_duplicate_register_raises(self) -> None:
        registry = BaselineRegistry()
        registry.register(PreemptEvictStrategy())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(PreemptEvictStrategy())

    def test_get_unknown_raises(self) -> None:
        registry = BaselineRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_strategies(self) -> None:
        registry = BaselineRegistry()
        registry.register(PreemptEvictStrategy())
        registry.register(UniformStrategy())
        names = registry.list_strategies()
        assert names == ["preempt-evict", "uniform"]

    def test_create_default_registry(self) -> None:
        registry = BaselineRegistry()
        registry.create_default_registry()
        assert registry.count == 7
        # 验证所有 7 个策略都已注册
        for name in [
            "preempt-evict",
            "static-random",
            "h2o-style",
            "uniform",
            "global-nobid",
            "slack-aware",
            "bidkv",
        ]:
            assert isinstance(registry.get(name), BaselineStrategy)


# ===========================================================================
# Preempt-Evict tests
# ===========================================================================


class TestPreemptEvict:
    """Preempt-Evict baseline 测试。"""

    def test_name(self) -> None:
        assert PreemptEvictStrategy().name == "preempt-evict"

    def test_evicts_lowest_priority_first(self) -> None:
        candidates = _make_candidates_varied()
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=400)
        # req-d (priority=0.5) 应该最先被驱逐
        assert actions[0].request_id == "req-d"
        assert actions[0].action_type == "evict"

    def test_evicts_until_sufficient(self) -> None:
        candidates = _make_candidates_varied()
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=500)
        freed = sum(a.target_tokens for a in actions)
        assert freed >= 500

    def test_empty_candidates(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims([], needed_tokens=100)
        assert actions == []

    def test_zero_needed(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(_make_candidates(), needed_tokens=0)
        assert actions == []

    def test_all_actions_are_evict(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=1000)
        for action in actions:
            assert action.action_type == "evict"


# ===========================================================================
# Static-Random tests
# ===========================================================================


class TestStaticRandom:
    """Static-Random baseline 测试。"""

    def test_name(self) -> None:
        assert StaticRandomStrategy().name == "static-random"

    def test_deterministic_with_seed(self) -> None:
        candidates = _make_candidates(5)
        s1 = StaticRandomStrategy(seed=42)
        s2 = StaticRandomStrategy(seed=42)
        a1 = s1.select_victims(candidates, needed_tokens=300)
        a2 = s2.select_victims(candidates, needed_tokens=300)
        assert [a.request_id for a in a1] == [a.request_id for a in a2]

    def test_fixed_compression_ratio(self) -> None:
        candidates = [RequestState("req-1", current_tokens=200)]
        strategy = StaticRandomStrategy(compression_ratio=0.5, seed=0)
        actions = strategy.select_victims(candidates, needed_tokens=50)
        assert len(actions) == 1
        assert actions[0].target_tokens == 100  # 200 * 0.5

    def test_all_actions_are_compress(self) -> None:
        strategy = StaticRandomStrategy(seed=0)
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=100)
        for action in actions:
            assert action.action_type == "compress"

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="compression_ratio"):
            StaticRandomStrategy(compression_ratio=0.0)

    def test_empty_candidates(self) -> None:
        strategy = StaticRandomStrategy(seed=0)
        assert strategy.select_victims([], needed_tokens=100) == []


# ===========================================================================
# H2O-Style tests
# ===========================================================================


class TestH2OStyle:
    """H2O-Style baseline 测试。"""

    def test_name(self) -> None:
        assert H2OStyleStrategy().name == "h2o-style"

    def test_h2o_style_is_not_h2o_scoring(self) -> None:
        """H2O-Style ≠ H2OScoring：确认 H2O-Style 是策略，使用 H2OScoring 实例。"""
        scorer = H2OScoring()
        strategy = H2OStyleStrategy(scoring=scorer)
        # H2O-Style 是 BaselineStrategy 实例
        assert isinstance(strategy, BaselineStrategy)
        # H2O-Style 持有 H2OScoring 实例
        assert strategy.scoring is scorer

    def test_compresses_with_scoring(self) -> None:
        candidates = _make_candidates_varied()
        strategy = H2OStyleStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        for action in actions:
            assert action.action_type == "compress"

    def test_respects_needed_tokens(self) -> None:
        candidates = _make_candidates_varied()
        strategy = H2OStyleStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        freed = sum(a.target_tokens for a in actions)
        assert freed >= 100

    def test_empty_candidates(self) -> None:
        strategy = H2OStyleStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_with_scoring_states(self) -> None:
        """可以传入每个请求独立的 H2OScoring 实例。"""
        candidates = [RequestState("req-1", current_tokens=100, token_ids=tuple(range(100)))]
        scorer = H2OScoring()
        # 用 decode 数据更新 scorer
        scorer.update_from_decode_step([float(i) for i in range(100)])
        strategy = H2OStyleStrategy()
        actions = strategy.select_victims(
            candidates, needed_tokens=30, scoring_states={"req-1": scorer}
        )
        assert len(actions) >= 1


# ===========================================================================
# Uniform tests
# ===========================================================================


class TestUniform:
    """Uniform baseline 测试。"""

    def test_name(self) -> None:
        assert UniformStrategy().name == "uniform"

    def test_equal_distribution(self) -> None:
        candidates = _make_candidates(4, tokens_per_req=200)
        strategy = UniformStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=200)
        # 每个请求应压缩约 200/4 = 50 个 token
        for action in actions:
            assert action.target_tokens <= 50

    def test_all_actions_compress(self) -> None:
        strategy = UniformStrategy()
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=100)
        for action in actions:
            assert action.action_type == "compress"

    def test_empty_candidates(self) -> None:
        strategy = UniformStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_preserves_at_least_one_token(self) -> None:
        """每个请求至少保留 1 个 token。"""
        candidates = [RequestState("req-1", current_tokens=2)]
        strategy = UniformStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        for action in actions:
            assert action.target_tokens < candidates[0].current_tokens


# ===========================================================================
# Global-NoBid tests
# ===========================================================================


class TestGlobalNoBid:
    """Global-NoBid baseline 测试。"""

    def test_name(self) -> None:
        assert GlobalNoBidStrategy().name == "global-nobid"

    def test_uses_same_scoring_as_bidkv(self) -> None:
        """Global-NoBid 使用与 BidKV 相同的 H2OScoring。"""
        scorer = H2OScoring()
        nobid = GlobalNoBidStrategy(scoring=scorer)
        bidkv = BidKVStrategy(scoring=scorer)
        assert nobid.scoring is bidkv.scoring

    def test_respects_delta_budget(self) -> None:
        candidates = _make_candidates_varied()
        strategy = GlobalNoBidStrategy(delta_budget=0.01)
        actions = strategy.select_victims(candidates, needed_tokens=1000)
        # 严格的 delta_budget 应限制压缩量
        total_delta = sum(a.metadata.get("estimated_quality_delta", 0) for a in actions)
        assert total_delta <= 0.01 + 0.001  # 允许浮点误差

    def test_greedy_by_utility(self) -> None:
        """按 utility 贪心选择：高 tokens / 低 delta 的请求优先。"""
        candidates = _make_candidates_varied()
        strategy = GlobalNoBidStrategy(delta_budget=1.0)
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        for action in actions:
            assert action.action_type == "compress"

    def test_empty_candidates(self) -> None:
        strategy = GlobalNoBidStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_metadata_contains_utility(self) -> None:
        candidates = _make_candidates_varied()
        strategy = GlobalNoBidStrategy(delta_budget=1.0)
        actions = strategy.select_victims(candidates, needed_tokens=100)
        for action in actions:
            assert "system_utility" in action.metadata


# ===========================================================================
# Slack-Aware tests
# ===========================================================================


class TestSlackAware:
    """Slack-Aware baseline 测试。"""

    def test_name(self) -> None:
        assert SlackAwareStrategy().name == "slack-aware"

    def test_furthest_deadline_compressed_first(self) -> None:
        """deadline 远的请求先被压缩。"""
        candidates = [
            RequestState("req-near", current_tokens=200, deadline_ms=5100.0),
            RequestState("req-far", current_tokens=200, deadline_ms=50000.0),
            RequestState("req-mid", current_tokens=200, deadline_ms=15000.0),
        ]
        strategy = SlackAwareStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=50, now_ms=5000.0)
        # req-far (最大 slack=45000) 应该最先被压缩
        assert actions[0].request_id == "req-far"

    def test_no_deadline_compressed_first(self) -> None:
        """没有 deadline 的请求优先被压缩（视为无 SLO 保证）。"""
        candidates = [
            RequestState("req-with-slo", current_tokens=200, deadline_ms=10000.0),
            RequestState("req-no-slo", current_tokens=200, deadline_ms=None),
        ]
        strategy = SlackAwareStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=50, now_ms=5000.0)
        assert actions[0].request_id == "req-no-slo"

    def test_all_actions_compress(self) -> None:
        strategy = SlackAwareStrategy()
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=100)
        for action in actions:
            assert action.action_type == "compress"

    def test_empty_candidates(self) -> None:
        strategy = SlackAwareStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="compression_ratio"):
            SlackAwareStrategy(compression_ratio=0.0)


# ===========================================================================
# BidKV tests
# ===========================================================================


class TestBidKV:
    """BidKV 完整策略测试。"""

    def test_name(self) -> None:
        assert BidKVStrategy().name == "bidkv"

    def test_uses_full_pipeline(self) -> None:
        """BidKV 走完整 scoring → bid → pool → solver 流程。"""
        candidates = _make_candidates_varied()
        strategy = BidKVStrategy(delta_budget=0.5)
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        for action in actions:
            assert action.action_type == "compress"
            assert "bid_id" in action.metadata
            assert "utility" in action.metadata

    def test_with_prebaked_bids(self) -> None:
        """可以传入预生成的 bids（candidate-universe consistency）。"""
        candidates = _make_candidates(2, tokens_per_req=100)
        bids_by_request: dict[str, list[CompressionBid]] = {}
        for req in candidates:
            bids_by_request[req.request_id] = _make_bids_for_request(req.request_id, req.token_ids)
        strategy = BidKVStrategy(delta_budget=0.5)
        actions = strategy.select_victims(
            candidates, needed_tokens=30, bids_by_request=bids_by_request
        )
        assert len(actions) >= 1

    def test_empty_candidates(self) -> None:
        strategy = BidKVStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_zero_needed(self) -> None:
        strategy = BidKVStrategy()
        assert strategy.select_victims(_make_candidates(3), needed_tokens=0) == []


# ===========================================================================
# Candidate-universe consistency tests
# ===========================================================================


class TestCandidateUniverseConsistency:
    """验证所有 baseline 在同一 pressure event 使用同一候选池。"""

    def test_all_baselines_receive_same_candidates(self) -> None:
        """所有 baseline 接收完全相同的 candidates 列表。"""
        candidates = _make_candidates_varied()
        needed = 200

        strategies: list[BaselineStrategy] = [
            PreemptEvictStrategy(),
            StaticRandomStrategy(seed=42),
            H2OStyleStrategy(),
            UniformStrategy(),
            GlobalNoBidStrategy(delta_budget=0.5),
            SlackAwareStrategy(),
            BidKVStrategy(delta_budget=0.5),
        ]

        # 每个策略都接收同一个 candidates 对象
        results: dict[str, list[CompressionAction]] = {}
        for strategy in strategies:
            actions = strategy.select_victims(candidates, needed)
            results[strategy.name] = actions
            # 验证所有 action 的 request_id 都在候选池中
            candidate_ids = {r.request_id for r in candidates}
            for action in actions:
                assert action.request_id in candidate_ids, (
                    f"{strategy.name} produced action for {action.request_id} "
                    f"not in candidates: {candidate_ids}"
                )

    def test_all_baselines_produce_valid_actions(self) -> None:
        """所有 baseline 返回有效的 CompressionAction。"""
        candidates = _make_candidates(3, tokens_per_req=200)
        needed = 100

        strategies: list[BaselineStrategy] = [
            PreemptEvictStrategy(),
            StaticRandomStrategy(seed=0),
            H2OStyleStrategy(),
            UniformStrategy(),
            GlobalNoBidStrategy(delta_budget=0.5),
            SlackAwareStrategy(),
            BidKVStrategy(delta_budget=0.5),
        ]

        for strategy in strategies:
            actions = strategy.select_victims(candidates, needed)
            for action in actions:
                assert isinstance(action, CompressionAction)
                assert action.action_type in ("evict", "compress")
                assert action.target_tokens > 0


# ===========================================================================
# Registry integration test
# ===========================================================================


class TestRegistryIntegration:
    """Registry 集成测试 — 所有策略通过 registry 获取并运行。"""

    def test_all_strategies_via_registry(self) -> None:
        registry = BaselineRegistry()
        registry.create_default_registry()

        candidates = _make_candidates(3, tokens_per_req=200)

        for name in registry.list_strategies():
            strategy = registry.get(name)
            actions = strategy.select_victims(candidates, needed_tokens=100)
            assert isinstance(actions, list)
            for action in actions:
                assert isinstance(action, CompressionAction)
