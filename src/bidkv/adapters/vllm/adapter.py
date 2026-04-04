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

_DIAG_LOG = "/tmp/bidkv_diag.log"


def _diag(msg: str) -> None:
    """Write diagnostic message to a file (works in subprocesses)."""
    import os

    with open(_DIAG_LOG, "a") as f:
        f.write(f"[{os.getpid()}] adapter: {msg}\n")


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
        # {request_id: float} — 每个请求的到达时间 (monotonic ms)
        self._request_arrival_ms: dict[str, float] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Metrics
        self._metrics = AdapterMetrics()

        # Model executor reference (set by plugin.py after EngineCore init).
        # Used by truncation to sync model runner's cached block table.
        self._model_executor: Any = None

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

        .. deprecated::
            **DEPRECATED (Mode B)** — 本方法为 Mode B（token-level truncation）入口，
            在当前 Mode A（request-level preempt+recompute）实验中从未被调用。
            scheduler_hook.py 仅通过请求排序和 _preempt_request() 实现调度，
            不调用此方法。保留用于 Mode B 未来扩展（issue #054）。

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
        import warnings

        warnings.warn(
            "execute_compression() is Mode B dead code in current Mode A experiments. "
            "See issue #054 for Mode B roadmap.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self._config.is_active:
            return 0

        return self._execute_tail_truncation(request_id, target_tokens)

    def execute_abort(self, request_id: str) -> int:
        """Abort a running request via vLLM scheduler.abort_requests().

        .. deprecated::
            **DEPRECATED (Mode B)** — 本方法在 Mode A 实验中未被 scheduler_hook 调用。
            Mode A 使用 vLLM 原生 _preempt_request()，不通过 adapter 路径。
            保留用于 Mode B 未来扩展。

        The request is removed from running and requeued to waiting.
        All its KV blocks are freed immediately. The request will be
        recomputed from scratch when scheduled again.

        Returns the estimated number of tokens freed (= num_computed_tokens).
        """
        if not self._config.is_active:
            return 0

        scheduler = self._scheduler
        if scheduler is None:
            return 0

        # Find victim in running to estimate freed tokens
        freed_tokens = 0
        for req in getattr(scheduler, "running", []):
            if getattr(req, "request_id", None) == request_id:
                freed_tokens = getattr(req, "num_computed_tokens", 0)
                break

        if freed_tokens <= 0:
            return 0

        try:
            scheduler.abort_requests([request_id])
        except Exception:  # noqa: BLE001
            return 0

        self._metrics.record_eviction(request_id, freed_tokens)
        return freed_tokens

    def _execute_tail_truncation(self, request_id: str, target_tokens: int) -> int:
        """Token-level KV block truncation.

        .. deprecated::
            **DEPRECATED (Mode B)** — 本方法为 Mode B 核心实现，在 Mode A 中为死代码。
            仅通过 execute_compression()（也已废弃）间接调用。
            保留用于 Mode B 未来扩展（issue #054）。

        Removes tail KV blocks from a running request without full preemption.
        The request continues decoding with a reduced KV footprint.

        Steps:
        1. Calculate how many blocks to free from target_tokens
        2. Call kv_cache_manager.truncate_request_tail() (installed by
           truncation_hook.py) to atomically free tail blocks
        3. Update request's num_computed_tokens to match new boundary (INV-5)
        4. Update internal token tracking

        Returns
        -------
        int
            Actual tokens freed.
        """
        if self._scheduler is None:
            return 0

        # Get block size for token→block conversion
        block_size = self._get_block_size()
        if block_size <= 0:
            return 0

        # Check that truncation support is installed
        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "truncate_request_tail"):
            return 0

        # Calculate blocks to free (round up)
        num_blocks_to_free = max(1, (target_tokens + block_size - 1) // block_size)

        # Safety: synchronize GPU before freeing KV blocks.
        # vLLM v1 uses async CUDA kernel launches — future.result() returns when
        # the kernel is *launched*, not when it *finishes*.  Without sync, we
        # could free blocks that the in-flight kernel is still reading, causing
        # device-side assert.  Truncation events are rare (3 s cooldown), so the
        # synchronize cost is negligible.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass  # non-CUDA env (unit tests): skip silently

        # Attempt real KV block truncation
        result = kv_cache_manager.truncate_request_tail(request_id, num_blocks_to_free)

        if result.fallback_required:
            logger.debug(
                "Truncation failed: request=%s, reason=%s",
                request_id,
                result.reason,
            )
            return 0

        if not result.success:
            return 0

        # INV-5: Update request state to match new block boundary.
        # Must update BOTH num_computed_tokens AND output token tracking.
        # Without this, the model runner sees inconsistent state:
        #   num_computed_tokens = 16, but num_output_tokens = 38
        # causing it to prepare input_ids for positions that have
        # uninitialized token IDs in token_ids_cpu → embedding crash.
        requests_dict = getattr(self._scheduler, "requests", None)
        if requests_dict is not None:
            request_obj = requests_dict.get(request_id)
            if request_obj is not None:
                old_computed = request_obj.num_computed_tokens
                new_boundary = result.new_computed_token_boundary
                num_prompt = getattr(request_obj, "num_prompt_tokens", 0)

                request_obj.num_computed_tokens = new_boundary

                # If truncation boundary is within the prompt,
                # discard ALL output tokens (their KV is gone).
                # If boundary is past the prompt, trim output tokens
                # to match.
                output_ids = getattr(request_obj, "_output_token_ids", None)
                all_ids = getattr(request_obj, "_all_token_ids", None)
                if output_ids is not None and num_prompt > 0:
                    if new_boundary <= num_prompt:
                        # Boundary in prompt: discard all output
                        output_ids.clear()
                    else:
                        # Boundary past prompt: keep output up to boundary
                        keep_output = new_boundary - num_prompt
                        if keep_output < len(output_ids):
                            del output_ids[keep_output:]
                    # Sync _all_token_ids: prompt + remaining output
                    if all_ids is not None:
                        new_total = num_prompt + len(output_ids)
                        if len(all_ids) > new_total:
                            del all_ids[new_total:]

                _diag(
                    f"truncation state: req={request_id} "
                    f"old_computed={old_computed} new_boundary={new_boundary} "
                    f"num_prompt={num_prompt} "
                    f"output_tokens={len(output_ids) if output_ids else '?'} "
                    f"all_tokens={len(all_ids) if all_ids else '?'}"
                )

        # Truncate internal token tracking to match
        token_ids = self._request_tokens.get(request_id)
        if token_ids is not None and len(token_ids) > result.new_computed_token_boundary:
            del token_ids[result.new_computed_token_boundary :]

        # Sync model runner's cached block table so GPU attention kernel
        # only reads the new (truncated) block count.  Without this the
        # model runner's ``num_blocks_per_row`` stays stale and the
        # attention kernel accesses freed blocks → CUDA device-side assert.
        self._sync_model_runner_block_table(request_id, result.new_num_blocks)

        # Record metrics
        self._metrics.record_eviction(request_id, result.actual_freed_tokens)

        logger.debug(
            "Truncation: request=%s, freed=%d blocks (%d tokens), new_boundary=%d",
            request_id,
            result.actual_freed_blocks,
            result.actual_freed_tokens,
            result.new_computed_token_boundary,
        )
        return result.actual_freed_tokens

    def _sync_model_runner_block_table(self, request_id: str, new_num_blocks: int) -> None:
        """Sync the model runner's cached state after tail-block truncation.

        .. deprecated::
            **DEPRECATED (Mode B)** — 仅被 _execute_tail_truncation()（已废弃）调用。

        Uses ``add_row`` (full row overwrite) instead of just patching
        ``num_blocks_per_row`` to guarantee no stale block-IDs remain in
        the CPU numpy buffer that would be copied to GPU by
        ``commit_block_table``.

        Access path (UniProcExecutor, enforce-eager):
          model_executor.driver_worker.worker.model_runner
        """
        executor = self._model_executor
        if executor is None:
            _diag("sync_block_table: no model_executor reference")
            return

        try:
            # Navigate: executor → driver_worker → worker → model_runner
            driver_worker = getattr(executor, "driver_worker", None)
            if driver_worker is None:
                _diag("sync_block_table: no driver_worker")
                return
            worker = getattr(driver_worker, "worker", None)
            if worker is None:
                _diag("sync_block_table: no worker")
                return
            model_runner = getattr(worker, "model_runner", None)
            if model_runner is None:
                _diag("sync_block_table: no model_runner")
                return
            input_batch = getattr(model_runner, "input_batch", None)
            if input_batch is None:
                _diag("sync_block_table: no input_batch")
                return

            req_id_to_index = getattr(input_batch, "req_id_to_index", None)
            if req_id_to_index is None or request_id not in req_id_to_index:
                _diag(f"sync_block_table: request {request_id} not in input_batch")
                return
            req_index = req_id_to_index[request_id]

            # --- Collect real block IDs from coordinator ---
            kv_mgr = getattr(self._scheduler, "kv_cache_manager", None)
            coordinator = getattr(kv_mgr, "coordinator", None) if kv_mgr else None
            if coordinator is None:
                _diag("sync_block_table: no coordinator")
                return

            stms = getattr(coordinator, "single_type_managers", None) or []
            real_block_ids: list[list[int]] = []
            for mgr in stms:
                blocks = mgr.req_to_blocks.get(request_id, [])
                real_block_ids.append([b.block_id for b in blocks])

            if not real_block_ids:
                _diag("sync_block_table: no block IDs from coordinator")
                return

            # --- (1) Full overwrite of block table via add_row ---
            multi_bt = getattr(input_batch, "block_table", None)
            if multi_bt is not None:
                block_tables = getattr(multi_bt, "block_tables", None) or getattr(
                    multi_bt, "tables", None
                )
                if block_tables is not None:
                    for i, group_bt in enumerate(block_tables):
                        ids = real_block_ids[i] if i < len(real_block_ids) else []
                        # add_row: sets num_blocks_per_row=0, then appends
                        group_bt.add_row(ids, req_index)

            # --- (2) Overwrite model_runner.requests[req_id].block_ids ---
            mr_requests = getattr(model_runner, "requests", None)
            if mr_requests is not None:
                req_state = mr_requests.get(request_id)
                if req_state is not None:
                    block_ids_groups = getattr(req_state, "block_ids", None)
                    if block_ids_groups is not None:
                        for i, group_list in enumerate(block_ids_groups):
                            new_ids = real_block_ids[i] if i < len(real_block_ids) else []
                            group_list[:] = new_ids

            # --- (3) Sync num_output_tokens_cpu to match ---
            # The scheduler trimmed _output_token_ids, so model runner must
            # be told the new count.  Also zero out stale token IDs beyond
            # the new total to prevent the embedding layer from reading
            # garbage values.
            sched_req = getattr(self._scheduler, "requests", {}).get(request_id)
            if sched_req is not None:
                num_prompt = getattr(sched_req, "num_prompt_tokens", 0)
                n_output = len(getattr(sched_req, "_output_token_ids", []))
                new_total = num_prompt + n_output

                # Update model runner's output token count
                nout_cpu = getattr(input_batch, "num_output_tokens_cpu", None)
                if nout_cpu is not None and req_index < len(nout_cpu):
                    nout_cpu[req_index] = n_output

                # Zero stale token IDs in token_ids_cpu
                tids = getattr(input_batch, "token_ids_cpu", None)
                if tids is not None:
                    # tids might be numpy or CpuGpuBuffer; get the numpy view
                    tids_np = getattr(tids, "np", tids)
                    if hasattr(tids_np, "__setitem__"):
                        old_end = tids_np.shape[1] if tids_np.ndim > 1 else len(tids_np)
                        if new_total < old_end:
                            tids_np[req_index, new_total:] = 0
                        _diag(
                            f"sync_token_ids: zeroed positions "
                            f"{new_total}+ for req_index={req_index}"
                        )

                # Also update model_runner.requests[].num_computed_tokens
                if mr_requests is not None:
                    mr_req = mr_requests.get(request_id)
                    if mr_req is not None:
                        nt = getattr(mr_req, "num_computed_tokens", None)
                        if nt is not None:
                            actual_block_size = self._get_block_size() or 16
                            new_computed = new_num_blocks * actual_block_size
                            mr_req.num_computed_tokens = new_computed
                            _diag(f"sync: mr_req.num_computed_tokens {nt} -> {new_computed}")

            _diag(
                f"sync_block_table: add_row OK request={request_id} "
                f"req_index={req_index} blocks={[len(g) for g in real_block_ids]}"
            )
        except Exception:
            logger.warning(
                "Failed to sync model runner block table for request %s",
                request_id,
                exc_info=True,
            )
            _diag(f"sync_block_table: EXCEPTION for request={request_id}")

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._request_arrival_ms.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

    # ------------------------------------------------------------------
    # BidKV Pipeline — Pressure-triggered compression cycle
    # ------------------------------------------------------------------

    def try_compress(self) -> int:
        """执行一轮 BidKV 压缩周期（压力驱动）。

        .. deprecated::
            **DEPRECATED (Mode B)** — 本方法在 vLLM Mode A 中未被 scheduler_hook 调用。
            Mode A 通过请求排序 + vLLM 原生 preempt 实现调度，不走 adapter 压缩路径。
            SGLang adapter 中的同名方法仍为活跃代码。
            保留用于 Mode B 未来扩展（issue #054）。

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

        .. deprecated::
            **DEPRECATED (Mode B)** — 本方法在 vLLM Mode A 中未被调用。保留用于 Mode B。

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
            truncation_ratio=self._config.truncation_ratio,
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
            truncation_ratio=self._config.truncation_ratio,
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
