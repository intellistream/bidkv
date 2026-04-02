"""Tests for bidkv.adapters.vllm — vLLM adapter integration tests.

测试策略：
- 单元测试使用真实 bidkv 组件（无 mock）
- vLLM 集成测试通过 importability 和接口验证确认兼容性
- 所有核心 BidKV 逻辑（scoring、solver、pool、pressure）使用真实实现
"""

from __future__ import annotations

import pytest

from bidkv.adapters.vllm.adapter import AdapterMetrics, VLLMAdapter
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring import H2OScoring, UniformScoring
from bidkv.solver import GreedyBidSolver, SolverConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def h2o_scoring() -> H2OScoring:
    """H2OScoring 实例（默认参数）。"""
    return H2OScoring()


@pytest.fixture
def uniform_scoring() -> UniformScoring:
    """UniformScoring 实例（消融基线）。"""
    return UniformScoring()


@pytest.fixture
def active_config() -> BidKVConfig:
    """启用 BidKV 的配置。"""
    return BidKVConfig(enabled=True, kill_switch=False, delta_budget=0.15)


@pytest.fixture
def inactive_config() -> BidKVConfig:
    """未启用 BidKV 的配置（默认）。"""
    return BidKVConfig()


@pytest.fixture
def adapter_active(active_config: BidKVConfig, h2o_scoring: H2OScoring) -> VLLMAdapter:
    """启用状态的 VLLMAdapter（无 scheduler）。"""
    return VLLMAdapter(
        config=active_config,
        scoring=h2o_scoring,
        pressure_config=PressureConfig(enabled=True, threshold_pct=0.85),
        solver_config=SolverConfig(enabled=True, delta_budget=0.15),
    )


@pytest.fixture
def adapter_inactive(inactive_config: BidKVConfig, h2o_scoring: H2OScoring) -> VLLMAdapter:
    """未启用状态的 VLLMAdapter。"""
    return VLLMAdapter(config=inactive_config, scoring=h2o_scoring)


# ---------------------------------------------------------------------------
# Test: VLLMAdapter construction
# ---------------------------------------------------------------------------


class TestVLLMAdapterConstruction:
    """VLLMAdapter 构造函数测试。"""

    def test_default_config_creates_inactive_adapter(
        self, inactive_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        adapter = VLLMAdapter(config=inactive_config, scoring=h2o_scoring)
        assert not adapter.config.is_active
        assert not adapter.installed

    def test_active_config_creates_active_adapter(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        adapter = VLLMAdapter(config=active_config, scoring=h2o_scoring)
        assert adapter.config.is_active
        assert not adapter.installed

    def test_custom_compression_levels(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        levels = [0.1, 0.3, 0.5]
        adapter = VLLMAdapter(
            config=active_config,
            scoring=h2o_scoring,
            compression_levels=levels,
        )
        assert adapter._compression_levels == tuple(levels)

    def test_components_initialized(self, adapter_active: VLLMAdapter) -> None:
        assert isinstance(adapter_active.pressure_detector, PressureDetector)
        assert isinstance(adapter_active.pool_manager, BidPoolManager)
        assert isinstance(adapter_active.solver, GreedyBidSolver)
        assert isinstance(adapter_active.metrics, AdapterMetrics)


# ---------------------------------------------------------------------------
# Test: Feature gate & kill switch
# ---------------------------------------------------------------------------


class TestFeatureGateKillSwitch:
    """Feature gate 和 kill switch 功能测试。"""

    def test_inactive_adapter_no_compression(self, adapter_inactive: VLLMAdapter) -> None:
        """Feature OFF 时 try_compress 返回 0。"""
        assert adapter_inactive.try_compress() == 0

    def test_inactive_adapter_no_execute(self, adapter_inactive: VLLMAdapter) -> None:
        """Feature OFF 时 execute_compression 返回 0。"""
        assert adapter_inactive.execute_compression("req-1", 100) == 0

    def test_kill_switch_stops_all_operations(self, adapter_active: VLLMAdapter) -> None:
        """Kill switch 激活后，所有操作返回 0。"""
        adapter_active.track_request("req-1", list(range(100)))
        adapter_active.activate_kill_switch()

        assert not adapter_active.config.is_active
        assert adapter_active.try_compress() == 0
        assert adapter_active.execute_compression("req-1", 50) == 0
        assert adapter_active.metrics.kill_switch_activations == 1

    def test_kill_switch_deactivation_resumes(self, adapter_active: VLLMAdapter) -> None:
        """Kill switch 解除后恢复操作能力。"""
        adapter_active.activate_kill_switch()
        assert not adapter_active.config.is_active

        adapter_active.deactivate_kill_switch()
        assert adapter_active.config.is_active


# ---------------------------------------------------------------------------
# Test: Request tracking
# ---------------------------------------------------------------------------


class TestRequestTracking:
    """请求追踪功能测试。"""

    def test_track_request(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3, 4, 5])
        assert "req-1" in adapter_active.get_tracked_requests()

    def test_track_multiple_requests(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3])
        adapter_active.track_request("req-2", [4, 5, 6])
        tracked = adapter_active.get_tracked_requests()
        assert "req-1" in tracked
        assert "req-2" in tracked

    def test_on_request_complete_clears_tracking(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3])
        adapter_active.on_request_complete("req-1")
        assert "req-1" not in adapter_active.get_tracked_requests()
        assert adapter_active.metrics.total_requests_completed == 1

    def test_on_request_complete_nonexistent_is_noop(self, adapter_active: VLLMAdapter) -> None:
        """清理不存在的请求不报错。"""
        adapter_active.on_request_complete("nonexistent")
        assert adapter_active.metrics.total_requests_completed == 1


# ---------------------------------------------------------------------------
# Test: BidKV pipeline (pressure → bid → solve → compress)
# ---------------------------------------------------------------------------


class TestBidKVPipeline:
    """BidKV 压缩管道端到端测试（真实组件，不涉及 vLLM）。"""

    def test_no_pressure_no_compression(self, adapter_active: VLLMAdapter) -> None:
        """无压力态时不触发压缩。"""
        adapter_active.track_request("req-1", list(range(100)))
        # 不更新 KV stats → 无压力 → 不压缩
        result = adapter_active.try_compress()
        assert result == 0

    def test_bid_generation_via_refresh(self, adapter_active: VLLMAdapter) -> None:
        """验证 _refresh_bids 正确生成 bids。"""
        adapter_active.track_request("req-1", list(range(50)))
        adapter_active._refresh_bids()

        pool_snapshot = adapter_active.pool_manager.get_pool_snapshot()
        assert len(pool_snapshot.bids) > 0
        # 每个 bid 都属于 req-1
        for bid in pool_snapshot.bids:
            assert bid.request_id == "req-1"

    def test_solver_selects_bids_under_pressure(self, adapter_active: VLLMAdapter) -> None:
        """验证 pressure → bid → solver 流程。"""
        adapter_active.track_request("req-1", list(range(100)))
        adapter_active.track_request("req-2", list(range(80)))

        # 刷新 bids
        adapter_active._refresh_bids()
        pool_snapshot = adapter_active.pool_manager.get_pool_snapshot()
        assert len(pool_snapshot.bids) > 0

        # Solver 求解
        acceptance = adapter_active.solver.solve(pool_snapshot, tokens_needed=50)
        # 应该选中至少一个 bid
        assert acceptance.total_tokens_freed > 0


# ---------------------------------------------------------------------------
# Test: get_kv_stats without scheduler
# ---------------------------------------------------------------------------


class TestGetKVStats:
    """get_kv_stats 在无 scheduler 时的行为。"""

    def test_no_scheduler_returns_zeros(self, adapter_active: VLLMAdapter) -> None:
        used, total = adapter_active.get_kv_stats()
        assert used == 0
        assert total == 0


# ---------------------------------------------------------------------------
# Test: install/uninstall hook
# ---------------------------------------------------------------------------


class TestInstallHooks:
    """Scheduler hook 安装/卸载测试。"""

    def test_install_without_scheduler_raises(self, adapter_active: VLLMAdapter) -> None:
        """未设置 scheduler 时 install 应抛出 RuntimeError。"""
        with pytest.raises(RuntimeError, match="scheduler not set"):
            adapter_active.install()

    def test_install_inactive_skips(
        self, inactive_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """Feature OFF 时 install 不注入。"""
        adapter = VLLMAdapter(config=inactive_config, scoring=h2o_scoring)
        adapter.install()  # 不报错，静默跳过
        assert not adapter.installed


# ---------------------------------------------------------------------------
# Test: H2O decode step callback
# ---------------------------------------------------------------------------


class TestH2ODecodeCallback:
    """H2O decode step 回调测试。"""

    def test_on_decode_step_updates_scoring(self, adapter_active: VLLMAdapter) -> None:
        """verify on_decode_step 调用 scoring.update_from_decode_step。"""
        scoring = adapter_active.scoring
        assert isinstance(scoring, H2OScoring)
        assert scoring.decode_steps == 0

        adapter_active.on_decode_step("req-1", [0.1, 0.5, 0.3, 0.8])
        assert scoring.decode_steps == 1
        assert adapter_active.metrics.total_decode_steps == 1

    def test_on_decode_step_inactive_is_noop(self, adapter_inactive: VLLMAdapter) -> None:
        """Feature OFF 时 on_decode_step 不更新。"""
        scoring = adapter_inactive.scoring
        assert isinstance(scoring, H2OScoring)

        adapter_inactive.on_decode_step("req-1", [0.1, 0.5])
        assert scoring.decode_steps == 0


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------


class TestAdapterMetrics:
    """AdapterMetrics 测试。"""

    def test_initial_metrics(self) -> None:
        m = AdapterMetrics()
        assert m.total_evictions == 0
        assert m.total_tokens_freed == 0
        assert m.total_pressure_events == 0

    def test_record_eviction(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 100)
        assert m.total_evictions == 1
        assert m.total_tokens_freed == 100

    def test_record_eviction_zero_is_noop(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 0)
        assert m.total_evictions == 0

    def test_as_dict(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 50)
        m.record_pressure_event()
        d = m.as_dict()
        assert d["total_evictions"] == 1
        assert d["total_tokens_freed"] == 50
        assert d["total_pressure_events"] == 1


# ---------------------------------------------------------------------------
# Test: h2o_hook module
# ---------------------------------------------------------------------------


class TestH2OHook:
    """h2o_hook 模块的 _generate_attention_proxy 测试。"""

    def test_generate_attention_proxy_basic(self) -> None:
        from bidkv.adapters.vllm.h2o_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(10)
        assert len(proxy) == 10
        # 所有值应为正数
        assert all(v > 0 for v in proxy)

    def test_generate_attention_proxy_empty(self) -> None:
        from bidkv.adapters.vllm.h2o_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(0)
        assert proxy == []

    def test_generate_attention_proxy_single(self) -> None:
        from bidkv.adapters.vllm.h2o_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(1)
        assert len(proxy) == 1
        assert proxy[0] > 0

    def test_attention_sink_property(self) -> None:
        """验证 position 0 (attention sink) 有较高权重。"""
        from bidkv.adapters.vllm.h2o_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(100)
        # Position 0 应比中间位置权重高
        mid = len(proxy) // 2
        assert proxy[0] > proxy[mid]


# ---------------------------------------------------------------------------
# Test: scheduler_hook module
# ---------------------------------------------------------------------------


class TestSchedulerHook:
    """scheduler_hook 模块的辅助函数测试。"""

    def test_extract_token_ids_from_object(self) -> None:
        """从模拟对象中提取 token ids。"""
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            prompt_token_ids = [1, 2, 3, 4]
            output_token_ids = [5, 6]

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == [1, 2, 3, 4, 5, 6]

    def test_extract_token_ids_no_output(self) -> None:
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            prompt_token_ids = [10, 20]
            output_token_ids = None

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == [10, 20]

    def test_extract_token_ids_empty(self) -> None:
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            pass

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == []


# ---------------------------------------------------------------------------
# Helper: check vLLM v1 importability
# ---------------------------------------------------------------------------


def _vllm_v1_importable() -> bool:
    """检查 vLLM v1 内部模块是否可导入（需要 vllm._C 与 PyTorch ABI 兼容）。"""
    try:
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Test: vLLM importability (requires vllm installed)
# ---------------------------------------------------------------------------


class TestVLLMImportability:
    """验证 vLLM 相关导入是否可用。"""

    def test_vllm_installed(self) -> None:
        """验证 vLLM 已安装。"""
        import vllm

        assert hasattr(vllm, "__version__")

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_v1_scheduler_importable(self) -> None:
        """验证 vLLM v1 Scheduler 可导入。"""
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_kv_cache_manager_importable(self) -> None:
        """验证 KVCacheManager 可导入。"""
        from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: F401

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_block_pool_importable(self) -> None:
        """验证 BlockPool 可导入。"""
        from vllm.v1.core.block_pool import BlockPool  # noqa: F401


# ---------------------------------------------------------------------------
# Test: Full pipeline with scheduler hook on fake scheduler
# ---------------------------------------------------------------------------


class TestSchedulerHookIntegration:
    """使用简易 FakeScheduler 验证 hook 安装/卸载和流程。"""

    def _make_fake_scheduler(self) -> _FakeScheduler:
        return _FakeScheduler()

    def test_install_and_uninstall_hooks(self, adapter_active: VLLMAdapter) -> None:
        """验证 hook 安装后方法被替换，卸载后恢复。"""
        from bidkv.adapters.vllm.scheduler_hook import (
            install_scheduler_hook,
            uninstall_scheduler_hook,
        )

        scheduler = self._make_fake_scheduler()
        original_schedule = scheduler.schedule
        original_update = scheduler.update_from_output

        install_scheduler_hook(scheduler, adapter_active)

        # 方法已被替换
        assert scheduler.schedule is not original_schedule
        assert scheduler.update_from_output is not original_update
        assert hasattr(scheduler, "_bidkv_adapter")

        uninstall_scheduler_hook(scheduler, adapter_active)

        # 方法已恢复
        assert scheduler.schedule == original_schedule
        assert scheduler.update_from_output == original_update
        assert not hasattr(scheduler, "_bidkv_adapter")

    def test_patched_schedule_calls_original(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched schedule 调用原始方法。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        result = scheduler.schedule()
        assert result == "schedule_called"
        assert scheduler.schedule_call_count == 1

    def test_patched_update_from_output_calls_original(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched update_from_output 调用原始方法。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        result = scheduler.update_from_output("sched_out", "model_out")
        assert result == "update_called"
        assert scheduler.update_call_count == 1

    def test_patched_free_request_cleans_bidkv(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched _free_request 清理 BidKV 状态。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        adapter_active.track_request("req-1", [1, 2, 3])
        assert "req-1" in adapter_active.get_tracked_requests()

        class FakeRequest:
            request_id = "req-1"

        scheduler._free_request(FakeRequest())
        assert "req-1" not in adapter_active.get_tracked_requests()


class _FakeRequest:
    """Minimal fake vLLM Request for testing _preempt_request path."""

    def __init__(self, request_id: str, status: str = "RUNNING") -> None:
        self.request_id = request_id
        self.status = status
        self.num_computed_tokens = 100
        self.num_prompt_tokens = 50
        self._output_token_ids: list[int] = list(range(50))
        self._all_token_ids: list[int] = list(range(100))
        self.spec_token_ids: list[int] = []
        self.num_preemptions = 0


class _FakeScheduler:
    """用于测试 hook 安装的简易 FakeScheduler。

    仅实现被 hook 的方法签名，内部计数验证调用。
    不模拟 vLLM 的完整调度逻辑。
    """

    def __init__(self) -> None:
        self.running: list[_FakeRequest] = []
        self.requests: dict[str, _FakeRequest] = {}
        self.schedule_call_count = 0
        self.update_call_count = 0
        self.free_count = 0
        self.kv_cache_manager = None
        self.preempted_requests: list[str] = []

    def add_running_request(self, request_id: str) -> _FakeRequest:
        """Helper: add a fake running request."""
        try:
            from vllm.v1.request import RequestStatus

            status = RequestStatus.RUNNING
        except ImportError:
            status = "RUNNING"
        req = _FakeRequest(request_id, status=status)
        self.requests[request_id] = req
        self.running.append(req)
        return req

    def schedule(self) -> str:
        self.schedule_call_count += 1
        return "schedule_called"

    def update_from_output(self, scheduler_output: object, model_runner_output: object) -> str:
        self.update_call_count += 1
        return "update_called"

    def _free_request(self, request: object) -> str:
        self.free_count += 1
        return "free_called"

    def _preempt_request(self, request: _FakeRequest, timestamp: float) -> None:
        self.preempted_requests.append(request.request_id)


# ---------------------------------------------------------------------------
# Test: Compression Execution (tail_truncation + native preempt)
# ---------------------------------------------------------------------------


class TestCompressionExecution:
    """压缩执行路径测试（tail truncation）。"""

    def test_no_truncation_support_returns_zero(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """无 truncation_support 时返回 0（不再 fallback 到 recompute）。"""
        scheduler = _FakeScheduler()
        scheduler.add_running_request("req-1")
        adapter = VLLMAdapter(config=active_config, scoring=h2o_scoring, scheduler=scheduler)
        adapter.track_request("req-1", list(range(100)))

        freed = adapter.execute_compression("req-1", 50)

        assert freed == 0
        assert scheduler.preempted_requests == []

    def test_no_scheduler_returns_zero(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """scheduler=None 时返回 0。"""
        adapter = VLLMAdapter(config=active_config, scoring=h2o_scoring)
        adapter.track_request("req-1", list(range(50)))

        freed = adapter.execute_compression("req-1", 50)
        assert freed == 0

    def test_untracked_request_returns_zero(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """没有追踪的 token 时返回 0。"""
        scheduler = _FakeScheduler()
        adapter = VLLMAdapter(config=active_config, scoring=h2o_scoring, scheduler=scheduler)

        freed = adapter.execute_compression("req-unknown", 50)
        assert freed == 0

    def test_empty_tokens_returns_zero(
        self, active_config: BidKVConfig, h2o_scoring: H2OScoring
    ) -> None:
        """追踪的 token 列表为空时返回 0。"""
        scheduler = _FakeScheduler()
        adapter = VLLMAdapter(config=active_config, scoring=h2o_scoring, scheduler=scheduler)
        adapter.track_request("req-1", [])

        freed = adapter.execute_compression("req-1", 50)
        assert freed == 0


# ---------------------------------------------------------------------------
# Test: tail truncation routing
# ---------------------------------------------------------------------------


class TestTruncationRouting:
    """Tail truncation 路由测试。"""

    def test_truncation_routes_to_block_truncation(self, h2o_scoring: H2OScoring) -> None:
        """execute_compression 路由到 _execute_tail_truncation (block truncation)."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        config = BidKVConfig(enabled=True, kill_switch=False)
        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        req = scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = VLLMAdapter(config=config, scoring=h2o_scoring, scheduler=scheduler)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 32)
        # Block-level truncation (32 tokens = 2 blocks × 16 tokens)
        assert freed == 32
        # Request NOT preempted (stays running)
        assert scheduler.preempted_requests == []
        # num_computed_tokens updated to new boundary
        assert req.num_computed_tokens == 48  # 3 remaining blocks × 16

    def test_kill_switch_config_stable(self, h2o_scoring: H2OScoring) -> None:
        """kill switch 激活/解除不破坏 config。"""
        config = BidKVConfig(enabled=True, kill_switch=False)
        adapter = VLLMAdapter(config=config, scoring=h2o_scoring)

        adapter.activate_kill_switch()
        assert adapter.config.kill_switch is True

        adapter.deactivate_kill_switch()
        assert adapter.config.kill_switch is False


# ---------------------------------------------------------------------------
# Fakes: KV cache hierarchy for tail truncation testing
# ---------------------------------------------------------------------------


class _FakeBlock:
    """Minimal fake vLLM KVCacheBlock."""

    def __init__(self, block_id: int, ref_cnt: int = 1) -> None:
        self.block_id = block_id
        self.ref_cnt = ref_cnt


class _FakeBlockPool:
    """Minimal fake vLLM BlockPool."""

    def __init__(self, block_size: int = 16) -> None:
        self.hash_block_size = block_size
        self.num_gpu_blocks = 100
        self.freed_blocks: list[_FakeBlock] = []

    def free_blocks(self, blocks: list[_FakeBlock]) -> None:
        for blk in blocks:
            blk.ref_cnt -= 1
            self.freed_blocks.append(blk)

    def get_usage(self) -> float:
        return 0.9


class _FakeSingleTypeKVCacheManager:
    """Minimal fake SingleTypeKVCacheManager with req_to_blocks."""

    def __init__(self, block_pool: _FakeBlockPool, block_size: int = 16) -> None:
        self.block_pool = block_pool
        self.block_size = block_size
        self.req_to_blocks: dict[str, list[_FakeBlock]] = {}

    def add_request_blocks(
        self, request_id: str, num_blocks: int, *, shared_prefix: int = 0
    ) -> None:
        blocks = []
        for i in range(num_blocks):
            ref = 2 if i < shared_prefix else 1
            blocks.append(_FakeBlock(block_id=i, ref_cnt=ref))
        self.req_to_blocks[request_id] = blocks


class _FakeCoordinator:
    """Minimal fake KVCacheCoordinator."""

    def __init__(self, single_type_manager: _FakeSingleTypeKVCacheManager) -> None:
        self.single_type_managers = (single_type_manager,)
        self.block_pool = single_type_manager.block_pool


class _FakeKVCacheManager:
    """Minimal fake KVCacheManager for tail truncation testing."""

    def __init__(self, block_size: int = 16, num_blocks_per_request: int = 5) -> None:
        self.block_pool = _FakeBlockPool(block_size)
        self._single_mgr = _FakeSingleTypeKVCacheManager(self.block_pool, block_size)
        self.coordinator = _FakeCoordinator(self._single_mgr)
        self._num_blocks_per_request = num_blocks_per_request

    def add_request(self, request_id: str, *, shared_prefix: int = 0) -> None:
        self._single_mgr.add_request_blocks(
            request_id, self._num_blocks_per_request, shared_prefix=shared_prefix
        )


class _FakeSchedulerWithKV(_FakeScheduler):
    """FakeScheduler with a fake KV cache manager for tail truncation tests."""

    def __init__(self, block_size: int = 16, num_blocks_per_request: int = 5) -> None:
        super().__init__()
        self.kv_cache_manager = _FakeKVCacheManager(block_size, num_blocks_per_request)
        self._block_size = block_size
        self._num_blocks = num_blocks_per_request

    def add_running_request(self, request_id: str) -> _FakeRequest:
        req = super().add_running_request(request_id)
        # Set token counts consistent with KV blocks
        total_tokens = self._block_size * self._num_blocks
        req.num_prompt_tokens = total_tokens // 2
        req.num_computed_tokens = total_tokens
        output_len = total_tokens - req.num_prompt_tokens
        req._output_token_ids = list(range(output_len))
        req._all_token_ids = list(range(total_tokens))
        self.kv_cache_manager.add_request(request_id)
        return req


# ---------------------------------------------------------------------------
# Test: Tail Truncation
# ---------------------------------------------------------------------------


class TestTailTruncation:
    """Tail truncation tests (token-level KV block truncation).

    Semantics: truncate tail KV blocks directly, request stays running.
    Returns 0 if truncation_support not installed.
    """

    def _make_adapter(
        self,
        scheduler: _FakeScheduler,
        h2o_scoring: H2OScoring,
    ) -> VLLMAdapter:
        config = BidKVConfig(enabled=True, kill_switch=False)
        return VLLMAdapter(config=config, scoring=h2o_scoring, scheduler=scheduler)

    def test_basic_truncation_with_support(self, h2o_scoring: H2OScoring) -> None:
        """Block-level truncation frees blocks, tracks metrics."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 32)

        assert freed == 32  # 2 blocks × 16 tokens
        assert scheduler.preempted_requests == []  # NOT preempted
        assert adapter.metrics.total_evictions == 1
        assert adapter.metrics.total_tokens_freed == 32

    def test_block_truncation_frees_tail_blocks(self, h2o_scoring: H2OScoring) -> None:
        """Tail blocks freed, request stays running."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        req = scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 32)  # 2 blocks

        # Block-level truncation: 2 blocks freed (32 tokens)
        assert freed == 32
        assert req.num_computed_tokens == 48  # 3 remaining blocks × 16
        assert scheduler.preempted_requests == []  # NOT preempted

    def test_truncation_capped_at_available_blocks(self, h2o_scoring: H2OScoring) -> None:
        """target_tokens > total → frees max blocks (keeps at least 1)."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        req = scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 1000)  # huge target

        # INV-4: keeps at least 1 block, frees 4 of 5
        assert freed == 64  # 4 blocks × 16
        assert req.num_computed_tokens == 16  # 1 remaining block × 16
        assert scheduler.preempted_requests == []  # NOT preempted

    def test_no_truncation_support_returns_zero(self, h2o_scoring: H2OScoring) -> None:
        """No truncation_support installed → returns 0."""
        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        req = scheduler.add_running_request("req-1")
        # Override: all tokens are prompt, no output
        req.num_prompt_tokens = 80
        req._output_token_ids = []
        req._all_token_ids = list(range(80))
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 32)

        assert freed == 0
        assert scheduler.preempted_requests == []

    def test_tracked_tokens_truncated_after_compression(self, h2o_scoring: H2OScoring) -> None:
        """After truncation, tracked tokens shortened to new boundary."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        adapter.execute_compression("req-1", 32)

        # Request still tracked but with fewer tokens
        assert "req-1" in adapter.get_tracked_requests()
        # Internal tracking truncated to new boundary (48 tokens = 3 blocks)
        tracked = adapter._request_tokens["req-1"]
        assert len(tracked) == 48

    def test_truncation_no_scheduler_returns_zero(self, h2o_scoring: H2OScoring) -> None:
        """scheduler=None 时返回 0。"""
        config = BidKVConfig(enabled=True, kill_switch=False)
        adapter = VLLMAdapter(config=config, scoring=h2o_scoring)
        adapter.track_request("req-1", list(range(50)))

        freed = adapter.execute_compression("req-1", 20)
        assert freed == 0

    def test_plain_scheduler_no_kv_returns_zero(self, h2o_scoring: H2OScoring) -> None:
        """Plain scheduler without kv_cache_manager returns 0."""
        scheduler = _FakeScheduler()  # no kv_cache_manager
        scheduler.add_running_request("req-1")
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(60)))

        freed = adapter.execute_compression("req-1", 20)
        assert freed == 0

    def test_partial_block_truncation(self, h2o_scoring: H2OScoring) -> None:
        """Truncate fewer tokens than output — rounds up to block boundary."""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        scheduler = _FakeSchedulerWithKV(block_size=16, num_blocks_per_request=5)
        req = scheduler.add_running_request("req-1")
        install_truncation_support(scheduler.kv_cache_manager)
        adapter = self._make_adapter(scheduler, h2o_scoring)
        adapter.track_request("req-1", list(range(80)))

        freed = adapter.execute_compression("req-1", 10)  # <1 block, rounds up to 1

        assert freed == 16  # 1 block × 16 tokens
        assert req.num_computed_tokens == 64  # 4 remaining blocks × 16
        assert scheduler.preempted_requests == []  # NOT preempted


# ---------------------------------------------------------------------------
# Test: Truncation Hook (unit-level)
# ---------------------------------------------------------------------------


class TestTruncationHook:
    """truncation_hook.py 的底层单元测试。"""

    def test_single_type_truncate_basic(self) -> None:
        """SingleType 截断：释放 2/5 blocks。"""
        from bidkv.adapters.vllm.truncation_hook import _single_type_truncate_tail_blocks

        pool = _FakeBlockPool(block_size=16)
        mgr = _FakeSingleTypeKVCacheManager(pool, block_size=16)
        mgr.add_request_blocks("req-1", 5)

        result = _single_type_truncate_tail_blocks(mgr, "req-1", 2)

        assert result.success is True
        assert result.actual_freed_blocks == 2
        assert result.actual_freed_tokens == 32
        assert result.new_num_blocks == 3
        assert len(mgr.req_to_blocks["req-1"]) == 3
        assert len(pool.freed_blocks) == 2

    def test_single_type_truncate_keep_at_least_one(self) -> None:
        """INV-4：不释放所有 blocks。"""
        from bidkv.adapters.vllm.truncation_hook import _single_type_truncate_tail_blocks

        pool = _FakeBlockPool(block_size=16)
        mgr = _FakeSingleTypeKVCacheManager(pool, block_size=16)
        mgr.add_request_blocks("req-1", 3)

        result = _single_type_truncate_tail_blocks(mgr, "req-1", 100)

        assert result.success is True
        assert result.actual_freed_blocks == 2  # keep 1
        assert len(mgr.req_to_blocks["req-1"]) == 1

    def test_single_type_truncate_shared_prefix(self) -> None:
        """INV-7：shared prefix blocks 被保护。"""
        from bidkv.adapters.vllm.truncation_hook import _single_type_truncate_tail_blocks

        pool = _FakeBlockPool(block_size=16)
        mgr = _FakeSingleTypeKVCacheManager(pool, block_size=16)
        # 5 blocks: first 3 shared (ref_cnt=2), last 2 private
        mgr.add_request_blocks("req-1", 5, shared_prefix=3)

        result = _single_type_truncate_tail_blocks(mgr, "req-1", 5)

        assert result.success is True
        assert result.actual_freed_blocks == 2  # only non-shared tail
        assert len(mgr.req_to_blocks["req-1"]) == 3

    def test_single_type_truncate_all_shared(self) -> None:
        """所有 tail blocks 都是 shared → fallback。"""
        from bidkv.adapters.vllm.truncation_hook import _single_type_truncate_tail_blocks

        pool = _FakeBlockPool(block_size=16)
        mgr = _FakeSingleTypeKVCacheManager(pool, block_size=16)
        mgr.add_request_blocks("req-1", 3, shared_prefix=3)

        result = _single_type_truncate_tail_blocks(mgr, "req-1", 2)

        assert result.success is False
        assert result.fallback_required is True
        assert "ref_cnt" in result.reason

    def test_single_type_truncate_missing_request(self) -> None:
        """请求不存在 → fallback。"""
        from bidkv.adapters.vllm.truncation_hook import _single_type_truncate_tail_blocks

        pool = _FakeBlockPool(block_size=16)
        mgr = _FakeSingleTypeKVCacheManager(pool, block_size=16)

        result = _single_type_truncate_tail_blocks(mgr, "req-missing", 2)

        assert result.success is False
        assert result.fallback_required is True

    def test_install_truncation_support_idempotent(self) -> None:
        """install_truncation_support 多次调用幂等。"""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        kv_mgr = _FakeKVCacheManager(block_size=16, num_blocks_per_request=3)
        install_truncation_support(kv_mgr)
        assert hasattr(kv_mgr, "truncate_request_tail")

        # Second call should not error
        install_truncation_support(kv_mgr)
        assert hasattr(kv_mgr, "truncate_request_tail")

    def test_install_truncation_support_callable(self) -> None:
        """安装后 truncate_request_tail 可调用。"""
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        kv_mgr = _FakeKVCacheManager(block_size=16, num_blocks_per_request=5)
        kv_mgr.add_request("req-1")
        install_truncation_support(kv_mgr)

        result = kv_mgr.truncate_request_tail("req-1", 2)

        assert result.success is True
        assert result.actual_freed_blocks == 2
