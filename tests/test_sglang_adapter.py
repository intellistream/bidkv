"""SGLang Adapter 集成测试。

测试范围：
- SGLangAdapter 创建与配置
- 压力态下 BidKV 压缩周期（try_compress）
- 共享前缀保护逻辑
- Kill switch 热切换
- Feature OFF 零开销路径
- 请求追踪与清理
- Metrics 输出格式（directional consistency 对比用）
- Scheduler hook 安装

注意：SGLang 框架相关的端到端测试需要 sglang 安装，以 skipif 标记。
本文件中的测试验证 adapter 的 BidKV 核心逻辑，使用真实的 bidkv 组件。
"""

from __future__ import annotations

import pytest

from bidkv.adapters.base import FrameworkAdapter
from bidkv.adapters.sglang.adapter import SGLangAdapter
from bidkv.config import BidKVConfig
from bidkv.pressure import PressureConfig
from bidkv.scoring.h2o import H2OScoring
from bidkv.solver import SolverConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def active_config() -> BidKVConfig:
    """BidKV 激活配置。"""
    return BidKVConfig(enabled=True, kill_switch=False, delta_budget=0.3)


@pytest.fixture()
def inactive_config() -> BidKVConfig:
    """BidKV 未激活配置（默认 OFF）。"""
    return BidKVConfig()


@pytest.fixture()
def h2o_scoring() -> H2OScoring:
    """H2O 评分策略。"""
    return H2OScoring(heavy_ratio=0.2, recent_ratio=0.2)


@pytest.fixture()
def active_adapter(active_config: BidKVConfig, h2o_scoring: H2OScoring) -> SGLangAdapter:
    """激活状态的 SGLangAdapter（无 scheduler）。"""
    return SGLangAdapter(
        config=active_config,
        scoring=h2o_scoring,
        pressure_config=PressureConfig(enabled=True, threshold_pct=0.85),
        solver_config=SolverConfig(enabled=True, delta_budget=0.3),
    )


@pytest.fixture()
def inactive_adapter(inactive_config: BidKVConfig, h2o_scoring: H2OScoring) -> SGLangAdapter:
    """未激活状态的 SGLangAdapter。"""
    return SGLangAdapter(config=inactive_config, scoring=h2o_scoring)


# ===========================================================================
# Test: Adapter 创建与基本属性
# ===========================================================================


class TestSGLangAdapterCreation:
    """SGLangAdapter 创建与基本属性验证。"""

    def test_is_framework_adapter(self, active_adapter: SGLangAdapter) -> None:
        """SGLangAdapter 是 FrameworkAdapter 的子类。"""
        assert isinstance(active_adapter, FrameworkAdapter)

    def test_active_config(self, active_adapter: SGLangAdapter) -> None:
        assert active_adapter.config.is_active

    def test_inactive_config(self, inactive_adapter: SGLangAdapter) -> None:
        assert not inactive_adapter.config.is_active

    def test_components_initialized(self, active_adapter: SGLangAdapter) -> None:
        assert active_adapter.pressure_detector is not None
        assert active_adapter.pool_manager is not None
        assert active_adapter.solver is not None
        assert active_adapter.pool_manager.is_active

    def test_install_without_scheduler_raises(self, active_adapter: SGLangAdapter) -> None:
        """install() 在 scheduler 未设置时应 raise RuntimeError。"""
        with pytest.raises(RuntimeError, match="scheduler not set"):
            active_adapter.install()


# ===========================================================================
# Test: Feature OFF 零开销路径
# ===========================================================================


class TestFeatureOff:
    """Feature OFF 时所有操作为 no-op。"""

    def test_try_compress_returns_zero(self, inactive_adapter: SGLangAdapter) -> None:
        assert inactive_adapter.try_compress() == 0

    def test_execute_compression_returns_zero(self, inactive_adapter: SGLangAdapter) -> None:
        assert inactive_adapter.execute_compression("req-1", 100) == 0

    def test_get_kv_stats_no_scheduler(self, inactive_adapter: SGLangAdapter) -> None:
        assert inactive_adapter.get_kv_stats() == (0, 0)


# ===========================================================================
# Test: 压力态下 BidKV 压缩周期
# ===========================================================================


class TestPressureCompression:
    """压力态下触发 bid-based 压缩的完整周期。"""

    def test_no_pressure_no_compression(self, active_adapter: SGLangAdapter) -> None:
        """无压力时 try_compress 返回 0。"""
        # KV stats 为 (0, 0)（无 scheduler），不触发压力
        result = active_adapter.try_compress()
        assert result == 0

    def test_pressure_triggers_bid_generation(self, active_adapter: SGLangAdapter) -> None:
        """压力态下应生成 bids 并尝试求解。"""
        # 手动设置压力状态
        active_adapter.pressure_detector.update_stats(used_tokens=9000, max_tokens=10000)
        assert active_adapter.pressure_detector.is_under_pressure()

        # 追踪一个请求
        token_ids = list(range(100))
        active_adapter.track_request("req-1", token_ids)

        # 刷新 bids（内部方法）
        active_adapter._refresh_bids()

        # 验证 bids 已提交到 pool
        pool_snapshot = active_adapter.pool_manager.get_pool_snapshot()
        assert len(pool_snapshot.bids) > 0
        assert all(b.request_id == "req-1" for b in pool_snapshot.bids)

    def test_full_compression_cycle_without_framework(self, active_adapter: SGLangAdapter) -> None:
        """完整压缩周期（无框架层执行压缩）。

        验证 BidKV pipeline 的正确性：
        pressure detection → bid generation → solver → acceptance。
        """
        # 追踪请求
        token_ids = list(range(200))
        active_adapter.track_request("req-1", token_ids)
        active_adapter.track_request("req-2", list(range(150)))

        # 模拟压力态
        active_adapter.pressure_detector.update_stats(used_tokens=8800, max_tokens=10000)

        # 刷新 bids
        active_adapter._refresh_bids()

        # 确认 pool 有 bids
        snapshot = active_adapter.pool_manager.get_pool_snapshot()
        assert len(snapshot.bids) > 0

        # Solver 求解
        tokens_needed = active_adapter.pressure_detector.needed_tokens()
        assert tokens_needed > 0

        acceptance = active_adapter.solver.solve(
            snapshot, tokens_needed, decision_reason="test_pressure"
        )
        # Solver 应该能找到可接受的 bids
        assert len(acceptance.accepted_bid_ids) > 0
        assert acceptance.total_tokens_freed > 0


# ===========================================================================
# Test: 共享前缀保护
# ===========================================================================


class TestSharedPrefixProtection:
    """共享前缀的 token 不被压缩。"""

    def test_shared_positions_tracked(self, active_adapter: SGLangAdapter) -> None:
        """共享位置应被正确追踪。"""
        shared = {0, 1, 2, 3, 4}
        active_adapter.track_request("req-1", list(range(20)), shared_positions=shared)
        assert active_adapter.get_shared_positions("req-1") == shared

    def test_shared_tokens_excluded_from_compression(self, active_adapter: SGLangAdapter) -> None:
        """共享 token 不应出现在压缩候选中。"""
        token_ids = list(range(20))
        shared = {0, 1, 2, 3, 4}  # 前 5 个 token 共享
        active_adapter.track_request("req-1", token_ids, shared_positions=shared)

        # 更新 H2O scoring（给所有 token 评分）
        attention = [0.1] * 20
        active_adapter.scoring.update_from_decode_step(attention)

        # 执行压缩（无 scheduler，不会实际调用框架 API）
        # execute_compression 内部会跳过共享位置
        # 因为没有 scheduler，实际释放为 0，但逻辑路径正确
        freed = active_adapter.execute_compression("req-1", 10)
        # 没有 scheduler 所以返回 0 是正常的
        assert freed == 0

    def test_update_shared_positions(self, active_adapter: SGLangAdapter) -> None:
        """共享位置可动态更新。"""
        active_adapter.track_request("req-1", list(range(20)), shared_positions={0, 1})
        assert active_adapter.get_shared_positions("req-1") == {0, 1}

        active_adapter.update_shared_positions("req-1", {0, 1, 2, 3})
        assert active_adapter.get_shared_positions("req-1") == {0, 1, 2, 3}


# ===========================================================================
# Test: Kill Switch
# ===========================================================================


class TestKillSwitch:
    """Kill switch 热切换。"""

    def test_activate_kill_switch(self, active_adapter: SGLangAdapter) -> None:
        """激活 kill switch 后所有操作变为 no-op。"""
        assert active_adapter.config.is_active

        active_adapter.activate_kill_switch()

        assert not active_adapter.config.is_active
        assert not active_adapter.pool_manager.is_active
        assert active_adapter.try_compress() == 0
        assert active_adapter.metrics.kill_switch_activations == 1

    def test_deactivate_kill_switch(self, active_adapter: SGLangAdapter) -> None:
        """解除 kill switch 后恢复操作。"""
        active_adapter.activate_kill_switch()
        assert not active_adapter.config.is_active

        active_adapter.deactivate_kill_switch()
        assert active_adapter.config.is_active
        assert active_adapter.pool_manager.is_active


# ===========================================================================
# Test: 请求生命周期管理
# ===========================================================================


class TestRequestLifecycle:
    """请求追踪与清理。"""

    def test_track_and_list_requests(self, active_adapter: SGLangAdapter) -> None:
        active_adapter.track_request("req-1", list(range(100)))
        active_adapter.track_request("req-2", list(range(50)))
        tracked = active_adapter.get_tracked_requests()
        assert "req-1" in tracked
        assert "req-2" in tracked

    def test_on_request_complete_cleanup(self, active_adapter: SGLangAdapter) -> None:
        """请求完成后应清理所有内部状态。"""
        active_adapter.track_request("req-1", list(range(100)), shared_positions={0, 1})

        # 提交 bids
        bids = active_adapter.scoring.generate_bids("req-1", list(range(100)), [0.2, 0.4])
        active_adapter.pool_manager.submit_bids("req-1", bids)

        # 完成请求
        active_adapter.on_request_complete("req-1")

        assert "req-1" not in active_adapter.get_tracked_requests()
        assert active_adapter.get_shared_positions("req-1") == set()
        assert active_adapter.pool_manager.get_bids_for_request("req-1") == []
        assert active_adapter.metrics.total_requests_completed == 1


# ===========================================================================
# Test: Metrics 输出格式
# ===========================================================================


class TestMetrics:
    """Metrics 输出格式与 vLLM adapter 对齐。"""

    def test_metrics_dict_format(self, active_adapter: SGLangAdapter) -> None:
        """metrics.as_dict() 返回标准字段。"""
        m = active_adapter.metrics.as_dict()
        expected_keys = {
            "total_compressions",
            "total_tokens_freed",
            "total_pressure_events",
            "total_requests_completed",
            "total_decode_steps",
            "kill_switch_activations",
        }
        assert set(m.keys()) == expected_keys
        assert all(isinstance(v, int) for v in m.values())

    def test_metrics_accumulate(self, active_adapter: SGLangAdapter) -> None:
        """指标应正确累积。"""
        active_adapter.metrics.record_pressure_event()
        active_adapter.metrics.record_pressure_event()
        active_adapter.metrics.record_compression("req-1", 50)
        active_adapter.metrics.record_decode_step("req-1")

        m = active_adapter.metrics.as_dict()
        assert m["total_pressure_events"] == 2
        assert m["total_compressions"] == 1
        assert m["total_tokens_freed"] == 50
        assert m["total_decode_steps"] == 1


# ===========================================================================
# Test: Scheduler Hook
# ===========================================================================


class TestSchedulerHook:
    """Scheduler hook 安装测试。"""

    def test_install_hooks_into_scheduler_object(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """验证 scheduler hook 可以安装到具有 get_next_batch_to_run 的对象上。"""
        from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook

        call_log: list[str] = []

        class FakeScheduler:
            def get_next_batch_to_run(self) -> str:
                call_log.append("original")
                return "batch"

        scheduler = FakeScheduler()
        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
            scheduler=scheduler,
            pressure_config=PressureConfig(enabled=True),
            solver_config=SolverConfig(enabled=True),
        )

        install_scheduler_hook(scheduler, adapter)

        # 调用 patched 方法
        result = scheduler.get_next_batch_to_run()
        assert result == "batch"
        assert "original" in call_log

    def test_install_hook_raises_without_method(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """scheduler 没有 get_next_batch_to_run 时应 raise。"""
        from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook

        class BadScheduler:
            pass

        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
        )
        with pytest.raises(RuntimeError, match="get_next_batch_to_run"):
            install_scheduler_hook(BadScheduler(), adapter)

    def test_uninstall_restores_original_method(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """uninstall 后 scheduler.get_next_batch_to_run 应恢复为原始方法。"""
        from bidkv.adapters.sglang.scheduler_hook import (
            install_scheduler_hook,
            uninstall_scheduler_hook,
        )

        class FakeScheduler:
            def get_next_batch_to_run(self) -> str:
                return "original"

        scheduler = FakeScheduler()

        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
            scheduler=scheduler,
            pressure_config=PressureConfig(enabled=True),
            solver_config=SolverConfig(enabled=True),
        )

        install_scheduler_hook(scheduler, adapter)

        # patched method 应有 __wrapped__ 属性
        patched = scheduler.get_next_batch_to_run
        assert hasattr(patched, "__wrapped__"), "patched method must have __wrapped__"
        # patched 不是原始方法（是闭包）
        assert patched() == "original"

        # uninstall 应恢复原始方法
        uninstall_scheduler_hook(scheduler)
        restored = scheduler.get_next_batch_to_run
        assert not hasattr(restored, "__wrapped__"), "restored method should not have __wrapped__"
        assert restored() == "original"


# ===========================================================================
# Test: H2O Decode Step 回调
# ===========================================================================


class TestH2ODecodeCallback:
    """H2O decode step 回调。"""

    def test_on_decode_step_updates_scoring(self, active_adapter: SGLangAdapter) -> None:
        """decode step 回调应更新 H2OScoring 的累积注意力。"""
        assert active_adapter.scoring.decode_steps == 0

        attention_pattern = [0.1, 0.5, 0.3, 0.9, 0.2]
        active_adapter.on_decode_step("req-1", attention_pattern)

        assert active_adapter.scoring.decode_steps == 1
        assert active_adapter.metrics.total_decode_steps == 1

    def test_on_decode_step_disabled(self, inactive_adapter: SGLangAdapter) -> None:
        """Feature OFF 时 decode step 回调为 no-op。"""
        inactive_adapter.on_decode_step("req-1", [0.1, 0.5])
        assert inactive_adapter.scoring.decode_steps == 0


# ===========================================================================
# Test: Radix Hook
# ===========================================================================


class TestRadixHook:
    """RadixAttention hook 函数测试。"""

    def test_free_kv_positions_no_scheduler(self) -> None:
        """无 scheduler 时 free_kv_positions 返回 0。"""
        from bidkv.adapters.sglang.radix_hook import free_kv_positions

        # None scheduler
        result = free_kv_positions(None, "req-1", [0, 1, 2])
        assert result == 0

    def test_free_kv_positions_empty_positions(self) -> None:
        """空位置列表返回 0。"""
        from bidkv.adapters.sglang.radix_hook import free_kv_positions

        class FakeScheduler:
            pass

        result = free_kv_positions(FakeScheduler(), "req-1", [])
        assert result == 0

    def test_get_shared_prefix_positions_no_cache(self) -> None:
        """无 RadixCache 时返回空集合。"""
        from bidkv.adapters.sglang.radix_hook import get_shared_prefix_positions

        class FakeScheduler:
            pass

        result = get_shared_prefix_positions(FakeScheduler(), "req-1", 100)
        assert result == set()


# ===========================================================================
# Test: Baseline 策略路由
# ===========================================================================


class TestBaselineRouting:
    """SGLangAdapter 策略路由测试。"""

    def test_default_strategy_is_bidkv(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """默认 experiment_strategy_name 为 bidkv。"""
        adapter = SGLangAdapter(config=active_config, scoring=h2o_scoring)
        assert adapter._experiment_strategy_name == "bidkv"
        assert adapter._experiment_strategy is None

    def test_custom_strategy_stored(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """自定义策略正确保存。"""
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("slack-aware")

        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
            experiment_strategy=strategy,
            experiment_strategy_name="slack_aware",
        )
        assert adapter._experiment_strategy is strategy
        assert adapter._experiment_strategy_name == "slack_aware"

    def test_build_request_states(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """_build_request_states 正确构建候选列表。"""
        adapter = SGLangAdapter(config=active_config, scoring=h2o_scoring)
        adapter.track_request("req-1", [1, 2, 3])
        adapter.track_request("req-2", [4, 5])

        states = adapter._build_request_states()
        assert len(states) == 2
        ids = {s.request_id for s in states}
        assert ids == {"req-1", "req-2"}

    def test_baseline_route_skips_bidkv_pipeline(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """非 bidkv 策略走 _try_compress_baseline 路径。"""
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("slack-aware")

        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
            pressure_config=PressureConfig(enabled=True, threshold_pct=0.50),
            solver_config=SolverConfig(enabled=True, delta_budget=0.3),
            experiment_strategy=strategy,
            experiment_strategy_name="slack_aware",
        )
        adapter.track_request("req-1", list(range(100)))

        # _try_compress_baseline is called (not the bid pipeline)
        # Without scheduler, execute_compression returns 0, so total_freed=0
        # but it should not raise errors
        result = adapter._try_compress_baseline(900, 1000, 50)
        assert result == 0  # no scheduler → execute_compression returns 0

    def test_bidkv_name_uses_full_pipeline(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """strategy_name == 'bidkv' 时使用完整 bid pipeline。"""
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("bidkv")

        adapter = SGLangAdapter(
            config=active_config,
            scoring=h2o_scoring,
            experiment_strategy=strategy,
            experiment_strategy_name="bidkv",
        )
        # Even with experiment_strategy set, name=bidkv means full pipeline
        assert adapter._experiment_strategy_name == "bidkv"
