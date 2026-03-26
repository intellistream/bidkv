"""VLLMAdapter — BidKV 在 vLLM 框架上的适配器。

vLLM v1（0.17+）调度架构：
- ``Scheduler`` 在 ``schedule()`` 中通过 ``kv_cache_manager.allocate_slots()`` 分配 KV
- 分配失败时 preempt 最低优先级 running request
- BidKV 在 preemption 前注入压缩尝试，减少不必要的 preemption

注入方式：Scheduler monkey-patch（``scheduler_hook.py``）。
vLLM v1 移除了 ``BlockSpaceManager`` 抽象和 ``--block-manager-class`` 参数，
因此 issue #044 原始 spec 中的 BlockManager 子类方案不再适用。
详见 ``__init__.py`` 中的 Architecture Decision 说明。

核心职责：
1. KV stats 获取：从 ``KVCacheManager.block_pool`` 读取 usage
2. Pressure interception：在 vLLM preempt 路径前获得压缩尝试机会
3. Compression 执行：通过 block-level 操作释放 KV（标记 + 释放尾部 blocks）
4. Scoring 回调：decode step 后更新评分策略
5. Lifecycle 管理：请求完成时清理 bid 和内部状态
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from bidkv.adapters.base import BaseAdapterMetrics, FrameworkAdapter
from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring.base import ScoringStrategy
from bidkv.scoring.bid_builder import build_bids
from bidkv.solver import GreedyBidSolver, SolverConfig

if TYPE_CHECKING:
    from bidkv.protocol.bid import BidAcceptance

logger = logging.getLogger(__name__)

# 默认压缩级别（论文 §4 标准设置）
DEFAULT_COMPRESSION_LEVELS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


class VLLMAdapter(FrameworkAdapter):
    """BidKV 在 vLLM 框架上的适配器。

    通过 monkey-patch vLLM 的 Scheduler，在 preemption 路径之前注入
    BidKV 压缩尝试。如果压缩能释放足够空间，则避免 preemption。

    Parameters
    ----------
    config:
        BidKV 全局配置（feature gate + kill switch）。
    scoring:
        评分策略实例。
    scheduler:
        vLLM 的 Scheduler 实例（``vllm.v1.core.sched.scheduler.Scheduler``）。
        若为 None，需在 ``install()`` 前通过 ``set_scheduler()`` 设置。
    pressure_config:
        PressureDetector 配置。若为 None 使用默认值（threshold 与 BidKV 对齐）。
    solver_config:
        GreedyBidSolver 配置。若为 None 使用默认值。
    compression_levels:
        bid 生成使用的压缩级别。默认 (0.2, 0.4, 0.6, 0.8)。
    """

    def __init__(
        self,
        config: BidKVConfig,
        scoring: ScoringStrategy,
        *,
        scheduler: Any = None,
        pressure_config: PressureConfig | None = None,
        solver_config: SolverConfig | None = None,
        compression_levels: Sequence[float] | None = None,
        experiment_strategy: BaselineStrategy | None = None,
        experiment_strategy_name: str = "bidkv",
    ) -> None:
        super().__init__(config, scoring)
        self._scheduler = scheduler

        # Experiment strategy routing
        self._experiment_strategy: BaselineStrategy | None = experiment_strategy
        self._experiment_strategy_name: str = experiment_strategy_name

        # BidKV 核心组件
        p_cfg = pressure_config or PressureConfig(enabled=config.is_active)
        s_cfg = solver_config or SolverConfig(
            enabled=config.is_active,
            delta_budget=config.delta_budget,
        )
        self._pressure_detector = PressureDetector(p_cfg)
        self._pool_manager = BidPoolManager(
            enabled=config.is_active,
            kill_switch=config.kill_switch,
        )
        self._solver = GreedyBidSolver(s_cfg)
        self._compression_levels = tuple(compression_levels or DEFAULT_COMPRESSION_LEVELS)

        # 请求追踪
        # {request_id: list[int]} — 每个请求的 token ids
        self._request_tokens: dict[str, list[int]] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Metrics
        self._metrics = AdapterMetrics()

        logger.info(
            "VLLMAdapter created: enabled=%s, kill_switch=%s, compression_levels=%s",
            config.is_active,
            config.kill_switch,
            self._compression_levels,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pressure_detector(self) -> PressureDetector:
        """内部 PressureDetector 实例。"""
        return self._pressure_detector

    @property
    def pool_manager(self) -> BidPoolManager:
        """内部 BidPoolManager 实例。"""
        return self._pool_manager

    @property
    def solver(self) -> GreedyBidSolver:
        """内部 GreedyBidSolver 实例。"""
        return self._solver

    @property
    def metrics(self) -> AdapterMetrics:
        """适配器运行指标。"""
        return self._metrics

    @property
    def installed(self) -> bool:
        """是否已安装到 vLLM 框架。"""
        return self._installed

    # ------------------------------------------------------------------
    # FrameworkAdapter interface
    # ------------------------------------------------------------------

    def install(self) -> None:
        """将 bidkv 注入到 vLLM 调度路径。

        Monkey-patch vLLM Scheduler 的 schedule()、update_from_output()
        和 _free_request() 方法。

        Raises
        ------
        RuntimeError
            如果 scheduler 未设置。
        """
        if not self._config.is_active:
            logger.info("VLLMAdapter.install: BidKV not active, skipping injection")
            return

        if self._scheduler is None:
            raise RuntimeError(
                "VLLMAdapter.install: scheduler not set. Call set_scheduler() before install()."
            )

        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        install_scheduler_hook(self._scheduler, self)
        self._installed = True
        logger.info("VLLMAdapter: installed into vLLM scheduler")

    def uninstall(self) -> None:
        """移除 bidkv 注入，恢复 vLLM 原始行为。"""
        if not self._installed:
            return

        from bidkv.adapters.vllm.scheduler_hook import uninstall_scheduler_hook

        uninstall_scheduler_hook(self._scheduler, self)
        self._installed = False
        logger.info("VLLMAdapter: uninstalled from vLLM scheduler")

    def set_scheduler(self, scheduler: Any) -> None:
        """设置 vLLM Scheduler 实例。

        Parameters
        ----------
        scheduler:
            vLLM v1 ``Scheduler`` 实例。
        """
        self._scheduler = scheduler

    def get_kv_stats(self) -> tuple[int, int]:
        """从 vLLM KVCacheManager 获取 KV 使用统计。

        通过 ``block_pool.get_usage()`` 和 ``block_pool.get_num_free_blocks()``
        计算 token 级别的使用量。

        Returns
        -------
        tuple[int, int]
            (used_tokens, max_tokens)。以 block 粒度对齐到 token 数。
        """
        if self._scheduler is None:
            return (0, 0)

        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return (0, 0)

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return (0, 0)

        # vLLM v1: block_size stored on block_pool as hash_block_size
        block_size = getattr(block_pool, "hash_block_size", None)
        if block_size is None or block_size <= 0:
            return (0, 0)

        # 直接从 block_pool 获取 total blocks
        total_blocks = getattr(block_pool, "num_gpu_blocks", 0)
        if total_blocks <= 0:
            return (0, 0)

        usage = block_pool.get_usage()
        total_tokens = total_blocks * block_size
        used_tokens = int(total_tokens * usage)

        return (used_tokens, total_tokens)

    def execute_compression(self, request_id: str, target_tokens: int) -> int:
        """在 vLLM 中执行 KV 压缩。

        截断 output tokens 后通过 native preempt 释放 KV，
        recompute 时序列更短，降低 prefill 成本。

        Parameters
        ----------
        request_id:
            目标请求 ID。
        target_tokens:
            期望释放的 token 数量。

        Returns
        -------
        int
            实际释放的 token 数量。
        """
        if not self._config.is_active:
            return 0

        return self._execute_tail_truncation(request_id, target_tokens)

    def _execute_recompute_fallback(self, request_id: str) -> int:
        """Internal: preempt request → vLLM frees all KV → recompute on resume.

        Used as the final step of tail_truncation (after output truncation)
        and by proactive preemption in scheduler_hook.
        Zero coordinator manipulation. Zero crash risk.

        Uses ``Scheduler._preempt_request()`` — the same internal API that
        vLLM itself calls within ``schedule()`` for native preemption.

        Returns
        -------
        int
            释放的 token 数量（即该 request 的全部追踪 token）。
        """
        if self._scheduler is None:
            return 0

        token_ids = self._request_tokens.get(request_id)
        total_tokens = len(token_ids) if token_ids else 0
        if total_tokens == 0:
            return 0

        # Look up the vLLM Request object
        requests_dict = getattr(self._scheduler, "requests", None)
        if requests_dict is None:
            return 0
        request_obj = requests_dict.get(request_id)
        if request_obj is None:
            return 0

        # Only preempt RUNNING requests
        try:
            from vllm.v1.request import RequestStatus
        except ImportError:
            return 0
        if request_obj.status != RequestStatus.RUNNING:
            return 0

        # Remove from running queue (required before calling _preempt_request)
        running = getattr(self._scheduler, "running", None)
        if running is None:
            return 0
        try:
            running.remove(request_obj)
        except ValueError:
            return 0  # not in running list

        # Use vLLM native preempt — the ONLY safe way to release KV.
        # This is the same API vLLM calls inside schedule() for preemption:
        # frees KV blocks, resets computed tokens, puts request back in waiting.
        import time

        self._scheduler._preempt_request(request_obj, time.monotonic())

        # Clear prev_step tracking so the preempted request doesn't trigger
        # the "assert not scheduled_in_prev_step" in _make_cached_request_data
        # when it re-enters the waiting queue and gets re-scheduled.
        prev_ids = getattr(self._scheduler, "prev_step_scheduled_req_ids", None)
        if prev_ids is not None:
            prev_ids.discard(request_id)

        # Clean up BidKV internal state
        self._request_tokens.pop(request_id, None)
        self._pool_manager.remove_by_request(request_id)
        self._metrics.record_compression(request_id, total_tokens)

        logger.debug(
            "_execute_recompute_fallback: request=%s, freed=%d tokens (preempted)",
            request_id,
            total_tokens,
        )
        return total_tokens

    def _execute_tail_truncation(self, request_id: str, target_tokens: int) -> int:
        """Truncate output tokens + native preempt for reduced recompute cost.

        Truncates the request's output token lists first, then preempts via
        vLLM's native preempt/recompute path. When the request is re-scheduled,
        it recomputes with a shorter sequence (prompt + partial output).

        This avoids the GPUModelRunner InputBatch block-table desync that
        occurs with direct block removal. vLLM v1's InputBatch only appends
        new blocks for cached requests; removing blocks directly causes the
        model runner to reference freed block IDs → CUDA device-side assert.

        Net effect:
        - Full KV recovery (preemption frees all blocks)
        - Lower recompute cost (fewer total tokens to prefill)
        - Permanent KV footprint reduction (truncated output never returns)

        Falls back to pure preemption if the request has no output tokens.

        Returns
        -------
        int
            Actual tokens freed (full request KV via preemption).
        """
        if self._scheduler is None:
            return 0

        requests_dict = getattr(self._scheduler, "requests", None)
        if requests_dict is None:
            return self._execute_recompute_fallback(request_id)

        request_obj = requests_dict.get(request_id)
        if request_obj is None:
            return self._execute_recompute_fallback(request_id)

        # Truncate output tokens before preemption
        output_tids = getattr(request_obj, "_output_token_ids", None)
        all_tids = getattr(request_obj, "_all_token_ids", None)
        num_prompt = getattr(request_obj, "num_prompt_tokens", 0)
        current_output_len = len(output_tids) if output_tids else 0

        tokens_truncated = 0
        if current_output_len > 0 and output_tids is not None and all_tids is not None:
            tokens_to_cut = min(target_tokens, current_output_len)
            if tokens_to_cut > 0:
                new_output_len = current_output_len - tokens_to_cut
                new_boundary = num_prompt + new_output_len
                del output_tids[new_output_len:]
                del all_tids[new_boundary:]
                tokens_truncated = tokens_to_cut
                logger.debug(
                    "Truncated %d output tokens: request=%s (output: %d→%d, total: %d→%d)",
                    tokens_to_cut,
                    request_id,
                    current_output_len,
                    new_output_len,
                    num_prompt + current_output_len,
                    new_boundary,
                )

        # Delegate to the safe native preemption path.
        # This frees ALL KV blocks and moves the request to waiting.
        # When re-scheduled, vLLM rebuilds the block table from scratch
        # (resumed_from_preemption=True in GPUModelRunner).
        freed = self._execute_recompute_fallback(request_id)

        if freed > 0 and tokens_truncated > 0:
            logger.debug(
                "Truncation complete: request=%s, truncated=%d, freed=%d",
                request_id,
                tokens_truncated,
                freed,
            )

        return freed

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

    # ------------------------------------------------------------------
    # BidKV Pipeline — Pressure-triggered compression cycle
    # ------------------------------------------------------------------

    def try_compress(self) -> int:
        """执行一轮 BidKV 压缩周期（压力驱动）。

        在 vLLM scheduler 的 preemption 路径之前调用。
        流程：
        1. 更新 KV stats → PressureDetector
        2. 检查是否处于压力态
        3. 为所有追踪的请求生成/刷新 bids
        4. Solver 选择最优 bid 组合
        5. 执行压缩

        Returns
        -------
        int
            本轮实际释放的总 token 数。0 表示未触发或无需压缩。
        """
        if not self._config.is_active:
            return 0

        # Step 1: 更新 KV stats
        used, total = self.get_kv_stats()
        self._pressure_detector.update_stats(used, total)

        # Step 2: 检查压力
        if not self._pressure_detector.is_under_pressure():
            return 0

        self._metrics.record_pressure_event()
        tokens_needed = self._pressure_detector.needed_tokens()

        # Strategy routing: baseline strategies use select_victims(),
        # BidKV uses the full bid pipeline.
        if self._experiment_strategy is not None and self._experiment_strategy_name != "bidkv":
            return self._try_compress_baseline(used, total, tokens_needed)

        # --- BidKV pipeline (default) ---
        # Step 3: 为追踪的请求刷新 bids
        self._refresh_bids()

        # Step 4: Solver 求解
        pool_snapshot = self._pool_manager.get_pool_snapshot()
        acceptance = self._solver.solve(
            pool_snapshot,
            tokens_needed,
            decision_reason="vllm_kv_pressure",
        )

        if acceptance.is_empty:
            return 0

        # Step 5: 执行压缩
        total_freed = self._execute_acceptance(acceptance)
        from bidkv.adapters.vllm.scheduler_hook import _diag

        if total_freed > 0:
            pct = (used / total * 100) if total > 0 else 0.0
            _diag(
                f"compression[bidkv]: freed={total_freed} tokens "
                f"(kv={pct:.0f}% bids_accepted={acceptance.accepted_count})"
            )
        return total_freed

    def _try_compress_baseline(self, used: int, total: int, tokens_needed: int) -> int:
        """Route compression through a BaselineStrategy.select_victims()."""
        strategy = self._experiment_strategy
        assert strategy is not None  # guarded by caller

        candidates = self._build_request_states()
        if not candidates:
            return 0

        actions = strategy.select_victims(candidates, tokens_needed)
        if not actions:
            return 0

        total_freed = self._execute_baseline_actions(actions)
        from bidkv.adapters.vllm.scheduler_hook import _diag

        if total_freed > 0:
            pct = (used / total * 100) if total > 0 else 0.0
            _diag(
                f"compression[{self._experiment_strategy_name}]: "
                f"freed={total_freed} tokens "
                f"(kv={pct:.0f}% actions={len(actions)})"
            )
        return total_freed

    def _build_request_states(self) -> list[RequestState]:
        """Build RequestState list from tracked requests."""
        states: list[RequestState] = []
        for request_id, token_ids in self._request_tokens.items():
            if not token_ids:
                continue
            states.append(
                RequestState(
                    request_id=request_id,
                    current_tokens=len(token_ids),
                    token_ids=tuple(token_ids),
                )
            )
        return states

    def _execute_baseline_actions(self, actions: list[CompressionAction]) -> int:
        """Execute CompressionAction list from a baseline strategy."""
        total_freed = 0
        for action in actions:
            freed = self.execute_compression(action.request_id, action.target_tokens)
            total_freed += freed
        return total_freed

    def try_compress_for_request(self, needed_blocks: int) -> int:
        """尝试压缩以为特定请求腾出 block 空间。

        在 allocate_slots 返回 None 时，preempt 之前调用。

        Parameters
        ----------
        needed_blocks:
            需要释放的 block 数量。

        Returns
        -------
        int
            实际释放的 block 数量（以 token 计）。
        """
        if not self._config.is_active:
            return 0

        block_size = self._get_block_size()
        if block_size <= 0:
            return 0

        needed_tokens = needed_blocks * block_size

        # 更新 KV stats
        used, total = self.get_kv_stats()
        self._pressure_detector.update_stats(used, total)
        self._metrics.record_pressure_event()

        # Strategy routing
        if self._experiment_strategy is not None and self._experiment_strategy_name != "bidkv":
            return self._try_compress_baseline(used, total, needed_tokens)

        # 刷新 bids
        self._refresh_bids()

        # Solver 求解
        pool_snapshot = self._pool_manager.get_pool_snapshot()
        acceptance = self._solver.solve(
            pool_snapshot,
            needed_tokens,
            decision_reason="vllm_allocate_slots_pressure",
        )

        if acceptance.is_empty:
            return 0

        total_freed = self._execute_acceptance(acceptance)
        return total_freed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_bids(self) -> None:
        """为所有追踪的请求重新生成 bids（score → build_bids 统一链路）。"""
        for request_id, token_ids in self._request_tokens.items():
            if not token_ids:
                continue
            scores = self._scoring.score(token_ids)
            bids = build_bids(
                request_id=request_id,
                token_ids=token_ids,
                scores=scores,
                compression_levels=self._compression_levels,
                algorithm_id="bidkv",
            )
            self._pool_manager.submit_bids(request_id, bids)

    def _execute_acceptance(self, acceptance: BidAcceptance) -> int:
        """执行 Solver 接受的 bid 组合。"""
        total_freed = 0
        for bid_id in acceptance.accepted_bid_ids:
            bid = self._pool_manager.get_bid(bid_id)
            if bid is None:
                continue
            freed = self.execute_compression(bid.request_id, bid.tokens_freed)
            total_freed += freed
        return total_freed

    def _get_block_size(self) -> int:
        """获取 vLLM 的 block size。"""
        if self._scheduler is None:
            return 0
        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return 0
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return 0
        block_size = getattr(block_pool, "hash_block_size", None)
        return block_size if block_size and block_size > 0 else 0

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def track_request(self, request_id: str, token_ids: list[int]) -> None:
        """开始追踪一个请求的 token。

        Parameters
        ----------
        request_id:
            请求 ID。
        token_ids:
            请求的 token ID 列表。
        """
        self._request_tokens[request_id] = list(token_ids)
        logger.debug(
            "track_request: request=%s, tokens=%d",
            request_id,
            len(token_ids),
        )

    def get_tracked_requests(self) -> list[str]:
        """返回当前追踪的所有请求 ID。"""
        return list(self._request_tokens.keys())

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """激活 kill switch，立即停止所有 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=True,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
            execution_mode=self._config.execution_mode,
        )
        self._pool_manager.activate_kill_switch()
        self._solver.update_config(
            SolverConfig(
                enabled=self._solver._config.enabled,
                kill_switch=True,
                delta_budget=self._solver._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(False)
        self._metrics.record_kill_switch()
        logger.warning("VLLMAdapter: KILL SWITCH activated")

    def deactivate_kill_switch(self) -> None:
        """解除 kill switch，恢复 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=False,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
            execution_mode=self._config.execution_mode,
        )
        self._pool_manager.enable()
        self._solver.update_config(
            SolverConfig(
                enabled=True,
                kill_switch=False,
                delta_budget=self._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(True)
        logger.info("VLLMAdapter: kill switch deactivated, BidKV resumed")

    # ------------------------------------------------------------------
    # H2O decode step callback
    # ------------------------------------------------------------------

    def on_decode_step(self, request_id: str, attention_pattern: Sequence[float]) -> None:
        """decode step 完成后的回调，更新 H2O scoring。

        由 h2o_hook 在每个 decode step 后调用。

        Parameters
        ----------
        request_id:
            请求 ID。
        attention_pattern:
            当前 decode step 中 query token 对所有 KV token 的注意力权重。
        """
        if not self._config.is_active:
            return
        if hasattr(self._scoring, "update_from_decode_step"):
            self._scoring.update_from_decode_step(attention_pattern)
        self._metrics.record_decode_step(request_id)


class AdapterMetrics(BaseAdapterMetrics):
    """vLLM adapter 运行指标。

    继承 ``BaseAdapterMetrics`` 的 6 个跨框架共同字段，
    额外提供 vLLM 特有的 ``preemptions_avoided``。
    """

    def __init__(self) -> None:
        super().__init__()
        self.preemptions_avoided: int = 0

    def record_preemption_avoided(self) -> None:
        self.preemptions_avoided += 1

    def as_dict(self) -> dict[str, int]:
        """返回所有指标的字典形式（含 vLLM 特有字段）。"""
        d = super().as_dict()
        d["preemptions_avoided"] = self.preemptions_avoided
        return d
