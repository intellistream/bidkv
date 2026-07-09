"""SGLang Adapter 集成测试。

测试范围：
- SGLangAdapter 创建与配置
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
from bidkv.scoring.positional import PositionalScoring
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
def positional_scoring() -> PositionalScoring:
    """Positional 评分策略。"""
    return PositionalScoring(heavy_ratio=0.2, recent_ratio=0.2)


@pytest.fixture()
def active_adapter(
    active_config: BidKVConfig,
    positional_scoring: PositionalScoring,
) -> SGLangAdapter:
    """激活状态的 SGLangAdapter（无 scheduler）。"""
    return SGLangAdapter(
        config=active_config,
        scoring=positional_scoring,
        pressure_config=PressureConfig(enabled=True, threshold_pct=0.85),
        solver_config=SolverConfig(enabled=True, delta_budget=0.3),
    )


@pytest.fixture()
def inactive_adapter(
    inactive_config: BidKVConfig,
    positional_scoring: PositionalScoring,
) -> SGLangAdapter:
    """未激活状态的 SGLangAdapter。"""
    return SGLangAdapter(config=inactive_config, scoring=positional_scoring)


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

    def test_get_kv_stats_no_scheduler(self, inactive_adapter: SGLangAdapter) -> None:
        assert inactive_adapter.get_kv_stats() == (0, 0)


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
            "total_evictions",
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
        active_adapter.metrics.record_eviction("req-1", 50)
        active_adapter.metrics.record_decode_step("req-1")

        m = active_adapter.metrics.as_dict()
        assert m["total_pressure_events"] == 2
        assert m["total_evictions"] == 1
        assert m["total_tokens_freed"] == 50
        assert m["total_decode_steps"] == 1


# ===========================================================================
# Test: Scheduler Hook
# ===========================================================================


class TestSchedulerHook:
    """Scheduler hook 安装测试。"""

    def test_install_hooks_into_scheduler_object(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
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
            scoring=positional_scoring,
            scheduler=scheduler,
            pressure_config=PressureConfig(enabled=True),
            solver_config=SolverConfig(enabled=True),
        )

        install_scheduler_hook(scheduler, adapter)

        # 调用 patched 方法 — Mode A hook 会调用原始方法
        result = scheduler.get_next_batch_to_run()
        assert result == "batch"
        assert "original" in call_log

    def test_install_hook_raises_without_method(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """scheduler 没有 get_next_batch_to_run 时应 raise。"""
        from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook

        class BadScheduler:
            pass

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
        )
        with pytest.raises(RuntimeError, match="get_next_batch_to_run"):
            install_scheduler_hook(BadScheduler(), adapter)

    def test_uninstall_restores_original_method(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
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
            scoring=positional_scoring,
            scheduler=scheduler,
            pressure_config=PressureConfig(enabled=True),
            solver_config=SolverConfig(enabled=True),
        )

        original_fn = scheduler.get_next_batch_to_run
        install_scheduler_hook(scheduler, adapter)

        # patched method should be different from original
        patched = scheduler.get_next_batch_to_run
        assert patched is not original_fn
        # Patched still returns correct result
        assert patched() == "original"

        # uninstall 应恢复原始方法
        uninstall_scheduler_hook(scheduler)
        restored = scheduler.get_next_batch_to_run
        assert restored() == "original"
        # Should not have _bidkv_ attributes anymore
        assert not hasattr(scheduler, "_bidkv_orig_get_next_batch_to_run")


# ===========================================================================
# Test: Positional Decode Step 回调
# ===========================================================================


class TestPositionalDecodeCallback:
    """Positional scoring decode step 回调。"""

    def test_on_decode_step_updates_scoring(self, active_adapter: SGLangAdapter) -> None:
        """decode step 回调应更新 PositionalScoring 的累积注意力。"""
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
# Test: Baseline 策略路由
# ===========================================================================


class TestBaselineRouting:
    """SGLangAdapter 策略路由测试。"""

    def test_default_strategy_is_bidkv(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """默认 experiment_strategy_name 为 bidkv。"""
        adapter = SGLangAdapter(config=active_config, scoring=positional_scoring)
        assert adapter._experiment_strategy_name == "bidkv"
        assert adapter._experiment_strategy is None

    def test_custom_strategy_stored(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """自定义策略正确保存。"""
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("static-random")

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy=strategy,
            experiment_strategy_name="static-random",
        )
        assert adapter._experiment_strategy is strategy
        assert adapter._experiment_strategy_name == "static-random"

    def test_bidkv_name_uses_full_pipeline(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """strategy_name == 'bidkv' 时使用完整 bid pipeline。"""
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("bidkv")

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy=strategy,
            experiment_strategy_name="bidkv",
        )
        # Even with experiment_strategy set, name=bidkv means full pipeline
        assert adapter._experiment_strategy_name == "bidkv"


# ===========================================================================
# Test: Mode A Request-Level Scheduling
# ===========================================================================


class TestModeAScheduling:
    """Mode A request-level 调度测试（对称 vLLM Mode A）。"""

    def test_adapter_has_mode_a_attributes(self, active_adapter: SGLangAdapter) -> None:
        """adapter 包含 Mode A 所需的属性。"""
        assert hasattr(active_adapter, "_cached_preempt_priority")
        assert hasattr(active_adapter, "_last_priority_refresh")
        assert hasattr(active_adapter, "_request_arrival_ms")
        assert active_adapter._cached_preempt_priority == {}
        assert active_adapter._last_priority_refresh == 0.0
        assert active_adapter._request_arrival_ms == {}

    def test_scheduler_hook_mode_a_flow(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """Mode A hook: get_next_batch_to_run 完整流程。"""
        from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook

        call_log: list[str] = []

        class FakeReq:
            def __init__(self, rid: str, prompt_len: int = 100) -> None:
                self.rid = rid
                self.request_id = rid
                self.num_prompt_tokens = prompt_len
                self.origin_input_ids = list(range(prompt_len))

        class FakeScheduler:
            waiting_queue: list[object] = []
            running_batch = None

            def get_next_batch_to_run(self) -> str:
                call_log.append("original")
                return "batch"

        scheduler = FakeScheduler()
        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            scheduler=scheduler,
            pressure_config=PressureConfig(enabled=True),
            solver_config=SolverConfig(enabled=True),
        )

        install_scheduler_hook(scheduler, adapter)
        result = scheduler.get_next_batch_to_run()
        assert result == "batch"
        assert "original" in call_log

    def test_waiting_reorder_fcfs_for_default(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """sglang_default: waiting 队列不重排（FCFS）。"""
        from bidkv.adapters.sglang.scheduler_hook import _reorder_waiting_for_admission

        class FakeReq:
            def __init__(self, rid: str, prompt_len: int) -> None:
                self.rid = rid
                self.num_prompt_tokens = prompt_len

        class FakeScheduler:
            waiting_queue: list[object]

        scheduler = FakeScheduler()
        scheduler.waiting_queue = [FakeReq("a", 200), FakeReq("b", 50), FakeReq("c", 100)]

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy_name="sglang_default",
        )

        _reorder_waiting_for_admission(scheduler, adapter)
        # FCFS: order unchanged
        ids = [r.rid for r in scheduler.waiting_queue]
        assert ids == ["a", "b", "c"]

    def test_waiting_reorder_sjf_for_bidkv(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """bidkv: waiting 按 prompt_tokens SJF 排序。"""
        from bidkv.adapters.sglang.scheduler_hook import _reorder_waiting_for_admission

        class FakeReq:
            def __init__(self, rid: str, prompt_len: int) -> None:
                self.rid = rid
                self.num_prompt_tokens = prompt_len

        class FakeScheduler:
            waiting_queue: list[object]

        scheduler = FakeScheduler()
        scheduler.waiting_queue = [FakeReq("a", 200), FakeReq("b", 50), FakeReq("c", 100)]

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy_name="bidkv",
        )

        _reorder_waiting_for_admission(scheduler, adapter)
        ids = [r.rid for r in scheduler.waiting_queue]
        assert ids == ["b", "c", "a"]  # SJF: 50 < 100 < 200

    def test_running_reorder_skipped_for_default(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """sglang_default: running 列表不重排。"""
        from bidkv.adapters.sglang.scheduler_hook import _reorder_running_for_preemption

        class FakeReq:
            def __init__(self, rid: str) -> None:
                self.rid = rid

        class FakeBatch:
            def __init__(self, reqs: list[object]) -> None:
                self.reqs = reqs

        class FakeScheduler:
            running_batch: object

        scheduler = FakeScheduler()
        scheduler.running_batch = FakeBatch([FakeReq("a"), FakeReq("b"), FakeReq("c")])

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy_name="sglang_default",
        )

        original_order = [r.rid for r in scheduler.running_batch.reqs]
        _reorder_running_for_preemption(scheduler, adapter)
        new_order = [r.rid for r in scheduler.running_batch.reqs]
        assert original_order == new_order

    def test_priority_cache_populated(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """策略 refresh 后 _cached_preempt_priority 被填充。"""
        from bidkv.adapters.sglang.scheduler_hook import _refresh_priority_cache
        from bidkv.baselines import BaselineRegistry

        registry = BaselineRegistry()
        registry.create_default_registry()
        strategy = registry.get("static-random")

        class FakeReq:
            def __init__(self, rid: str, tokens: int) -> None:
                self.rid = rid
                self.request_id = rid
                self.num_prompt_tokens = tokens
                self.num_computed_tokens = tokens
                self.num_preemptions = 0
                self.origin_input_ids = list(range(tokens))
                self.sampling_params = None

        class FakeBatch:
            def __init__(self, reqs: list[object]) -> None:
                self.reqs = reqs

        class FakeScheduler:
            running_batch: object

        scheduler = FakeScheduler()
        scheduler.running_batch = FakeBatch([FakeReq("a", 100), FakeReq("b", 200)])

        adapter = SGLangAdapter(
            config=active_config,
            scoring=positional_scoring,
            experiment_strategy=strategy,
            experiment_strategy_name="static-random",
        )
        # Track requests
        adapter.track_request("a", list(range(100)))
        adapter.track_request("b", list(range(200)))

        _refresh_priority_cache(scheduler, adapter)
        assert len(adapter._cached_preempt_priority) >= 2
