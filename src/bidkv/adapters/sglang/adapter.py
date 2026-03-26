"""SGLangAdapter — BidKV 在 SGLang 框架上的适配器。

SGLang 的 RadixAttention 天然支持部分 KV 释放（tree node invalidation），
比 vLLM 更适合 BidKV 的细粒度压缩。

核心职责：
1. KV stats 获取：从 ``TokenToKVPool`` 读取 used/total
2. Pressure interception：在 RadixAttention LRU 驱逐前获得压缩尝试机会
3. Compression 执行：通过 radix tree 节点级缩减释放 KV
4. Scoring 回调：decode step 后更新评分策略
5. Lifecycle 管理：请求完成时清理 bid 和前缀追踪

共享前缀保护：
- 检查 token 是否被多个请求共享（radix tree ref count > 1）
- 共享 token 不可压缩（跳过）
- 仅压缩请求独有的 token
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
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


class SGLangAdapter(FrameworkAdapter):
    """BidKV 在 SGLang 框架上的适配器。

    利用 SGLang 的 RadixAttention 树状 KV 管理进行细粒度压缩。
    SGLang 天然支持 token-level prefix sharing 和节点级 KV 释放，
    使 BidKV 能在比 vLLM 更细的粒度上操作。

    Parameters
    ----------
    config:
        BidKV 全局配置。
    scoring:
        评分策略实例。
    scheduler:
        SGLang 的 Scheduler 实例。若为 None，需在 ``install()`` 前通过
        ``set_scheduler()`` 设置。
    pressure_config:
        PressureDetector 配置。若为 None 使用默认值。
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
        audit_dir: Path | None = None,
    ) -> None:
        super().__init__(config, scoring)
        self._scheduler = scheduler
        self._experiment_strategy: BaselineStrategy | None = experiment_strategy
        self._experiment_strategy_name: str = experiment_strategy_name
        self._audit_dir: Path | None = audit_dir

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
        # {request_id: set[int]} — 每个请求中与其他请求共享的 token 位置
        self._shared_positions: dict[str, set[int]] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Metrics（与 vLLM adapter 对齐，便于跨框架对比）
        self._metrics = _AdapterMetrics()

        logger.info(
            "SGLangAdapter created: enabled=%s, kill_switch=%s, compression_levels=%s",
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
    def metrics(self) -> _AdapterMetrics:
        """适配器指标（与 vLLM adapter 对齐）。"""
        return self._metrics

    @property
    def installed(self) -> bool:
        """是否已安装到 SGLang 框架。"""
        return self._installed

    # ------------------------------------------------------------------
    # FrameworkAdapter interface
    # ------------------------------------------------------------------

    def install(self) -> None:
        """将 bidkv 注入到 SGLang 调度路径。

        需要先设置 scheduler。注入后，在 SGLang 的 eviction path 前
        会先尝试 bidkv 压缩。

        Raises
        ------
        RuntimeError
            如果 scheduler 未设置。
        """
        if not self._config.is_active:
            logger.info("SGLangAdapter.install: BidKV not active, skipping injection")
            return

        if self._scheduler is None:
            raise RuntimeError(
                "SGLangAdapter.install: scheduler not set. Call set_scheduler() before install()."
            )

        from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook

        install_scheduler_hook(self._scheduler, self)
        self._installed = True
        logger.info("SGLangAdapter: installed into SGLang scheduler")

    def set_scheduler(self, scheduler: Any) -> None:
        """设置 SGLang Scheduler 实例。

        Parameters
        ----------
        scheduler:
            SGLang ``Scheduler`` 实例。
        """
        self._scheduler = scheduler

    def get_kv_stats(self) -> tuple[int, int]:
        """从 SGLang 的 TokenToKVPool 获取 KV 使用统计。

        Returns
        -------
        tuple[int, int]
            (used_tokens, max_tokens)。
        """
        if self._scheduler is None:
            return (0, 0)

        token_to_kv_pool = _get_token_to_kv_pool(self._scheduler)
        if token_to_kv_pool is None:
            return (0, 0)

        total = token_to_kv_pool.size
        available = token_to_kv_pool.available_size()
        used = total - available
        return (used, total)

    def execute_compression(self, request_id: str, target_tokens: int) -> int:
        """在 SGLang 中执行 KV 压缩（radix tree 节点级缩减）。

        SGLang 的 RadixAttention 支持按节点粒度释放 KV，
        比 vLLM 的 block-level 释放更精细。

        共享前缀保护：与其他请求共享的 token（ref count > 1）不被压缩。

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

        token_ids = self._request_tokens.get(request_id)
        if not token_ids:
            logger.debug("execute_compression: no tokens tracked for request %s", request_id)
            return 0

        # 获取评分
        scores = self._scoring.score(token_ids)
        if len(scores) != len(token_ids):
            logger.warning(
                "execute_compression: score length mismatch (scores=%d, tokens=%d)",
                len(scores),
                len(token_ids),
            )
            return 0

        # 获取共享位置（这些位置不可压缩）
        shared = self._shared_positions.get(request_id, set())

        # 按分数升序排列（低分 = 不重要 = 优先压缩），排除共享位置
        candidates = [(pos, score) for pos, score in enumerate(scores) if pos not in shared]
        candidates.sort(key=lambda x: x[1])

        # 选取要释放的 token 位置
        positions_to_free = [pos for pos, _ in candidates[:target_tokens]]

        if not positions_to_free:
            return 0

        if self._scheduler is None:
            logger.debug(
                "execute_compression: no scheduler, cannot free KV for request %s",
                request_id,
            )
            return 0

        from bidkv.adapters.sglang.radix_hook import free_kv_positions

        actual_freed = free_kv_positions(self._scheduler, request_id, positions_to_free)

        if actual_freed > 0:
            # 更新内部追踪：从 token 列表中移除已压缩的位置
            freed_set = set(positions_to_free[:actual_freed])
            remaining = [tid for i, tid in enumerate(token_ids) if i not in freed_set]
            self._request_tokens[request_id] = remaining

        self._metrics.record_compression(request_id, actual_freed)
        logger.debug(
            "execute_compression: request=%s, target=%d, actual_freed=%d, shared_protected=%d",
            request_id,
            target_tokens,
            actual_freed,
            len(shared),
        )
        return actual_freed

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._shared_positions.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

    # ------------------------------------------------------------------
    # BidKV Pipeline — Pressure-triggered compression cycle
    # ------------------------------------------------------------------

    def try_compress(self) -> int:
        """执行一轮 BidKV 压缩周期（pressure interception boundary）。

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

        # Fairness audit: 记录压力事件和候选列表
        self._write_audit(used, total)

        # Strategy routing: baseline strategies use select_victims(),
        # BidKV uses the full bid pipeline.
        if self._experiment_strategy is not None and self._experiment_strategy_name != "bidkv":
            return self._try_compress_baseline(used, total, tokens_needed)

        # Step 3: 为追踪的请求刷新 bids
        self._refresh_bids()

        # Step 4: Solver 求解
        pool_snapshot = self._pool_manager.get_pool_snapshot()
        acceptance = self._solver.solve(
            pool_snapshot,
            tokens_needed,
            decision_reason="sglang_kv_pressure",
        )

        if acceptance.is_empty:
            return 0

        # Step 5: 执行压缩
        total_freed = self._execute_acceptance(acceptance, pool_snapshot)
        return total_freed

    def _refresh_bids(self) -> None:
        """为所有追踪的请求重新生成 bids。"""
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

    def _execute_acceptance(self, acceptance: BidAcceptance, pool_snapshot: Any) -> int:
        """执行 Solver 接受的 bid 组合。"""
        total_freed = 0
        for bid_id in acceptance.accepted_bid_ids:
            bid = self._pool_manager.get_bid(bid_id)
            if bid is None:
                continue
            freed = self.execute_compression(bid.request_id, bid.tokens_freed)
            total_freed += freed
        return total_freed

    # ------------------------------------------------------------------
    # Fairness audit
    # ------------------------------------------------------------------

    def _write_audit(self, used: int, total: int) -> None:
        """Write a fairness audit entry when pressure is detected."""
        if self._audit_dir is None:
            return
        from bidkv.experiments.sglang.collector import write_audit_entry

        candidate_ids = list(self._request_tokens.keys())
        kv_usage_pct = used / total if total > 0 else 0.0
        audit_path = self._audit_dir / f"audit_{self._experiment_strategy_name}.jsonl"
        write_audit_entry(
            audit_path,
            candidate_count=len(candidate_ids),
            candidate_request_ids=candidate_ids,
            strategy=self._experiment_strategy_name,
            kv_usage_pct=kv_usage_pct,
        )

    # ------------------------------------------------------------------
    # Baseline strategy routing
    # ------------------------------------------------------------------

    def _try_compress_baseline(self, used: int, total: int, tokens_needed: int) -> int:
        """Route compression through a BaselineStrategy.select_victims()."""
        strategy = self._experiment_strategy
        assert strategy is not None

        candidates = self._build_request_states()
        if not candidates:
            return 0

        actions = strategy.select_victims(candidates, tokens_needed)
        if not actions:
            return 0

        return self._execute_baseline_actions(actions)

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

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def track_request(
        self,
        request_id: str,
        token_ids: list[int],
        shared_positions: set[int] | None = None,
    ) -> None:
        """开始追踪一个请求的 token。

        Parameters
        ----------
        request_id:
            请求 ID。
        token_ids:
            请求的 token ID 列表。
        shared_positions:
            与其他请求共享的 token 位置集合（radix tree ref count > 1）。
            这些位置的 token 不可被压缩。
        """
        self._request_tokens[request_id] = list(token_ids)
        if shared_positions:
            self._shared_positions[request_id] = set(shared_positions)
        logger.debug(
            "track_request: request=%s, tokens=%d, shared=%d",
            request_id,
            len(token_ids),
            len(shared_positions) if shared_positions else 0,
        )

    def update_shared_positions(self, request_id: str, shared_positions: set[int]) -> None:
        """更新请求的共享前缀位置（动态变化时调用）。"""
        self._shared_positions[request_id] = set(shared_positions)

    def get_tracked_requests(self) -> list[str]:
        """返回当前追踪的所有请求 ID。"""
        return list(self._request_tokens.keys())

    def get_shared_positions(self, request_id: str) -> set[int]:
        """返回请求中受共享前缀保护的 token 位置。"""
        return set(self._shared_positions.get(request_id, set()))

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """激活 kill switch，立即停止所有 BidKV 操作。

        Kill switch 优先于 enabled，无需重启即可生效。
        """
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
        logger.warning("SGLangAdapter: KILL SWITCH activated")

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
        logger.info("SGLangAdapter: kill switch deactivated, BidKV resumed")

    # ------------------------------------------------------------------
    # H2O decode step callback
    # ------------------------------------------------------------------

    def on_decode_step(self, request_id: str, attention_pattern: Sequence[float]) -> None:
        """decode step 完成后的回调，更新评分策略。

        由 h2o_hook.py 在每个 decode step 后调用。

        Parameters
        ----------
        request_id:
            请求 ID。
        attention_pattern:
            当前 decode step 中 query token 对所有 KV token 的注意力权重。
        """
        if not self._config.is_active:
            return
        # 部分 scoring 策略（如 H2OScoring）支持 decode-step 增量更新
        if hasattr(self._scoring, "update_from_decode_step"):
            self._scoring.update_from_decode_step(attention_pattern)
        self._metrics.record_decode_step(request_id)


class _AdapterMetrics(BaseAdapterMetrics):
    """SGLang adapter 运行指标。

    直接复用 ``BaseAdapterMetrics`` 的 6 个跨框架共同字段。
    SGLang 暂无框架特有指标。
    """


def _get_token_to_kv_pool(scheduler: Any) -> Any | None:
    """从 SGLang scheduler 获取 TokenToKVPool。

    SGLang 的 KV 内存管理通过 ``TokenToKVPool`` 实现，
    它提供 ``size`` 和 ``available_size()`` 接口。
    """
    # SGLang 版本差异：pool 可能在不同属性路径下
    # 优先检查 tp_server -> token_to_kv_pool
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "token_to_kv_pool"):
            return tp.token_to_kv_pool
    # 直接属性
    if hasattr(scheduler, "token_to_kv_pool"):
        return scheduler.token_to_kv_pool
    return None
