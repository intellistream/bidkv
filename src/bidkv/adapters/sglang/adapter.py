"""SGLangAdapter — BidKV 在 SGLang 框架上的适配器。

**Mode A 架构（request-level 调度，对称 vLLM Mode A）**：
BidKV 在 SGLang 上的角色是 **请求调度插件**——控制 WHO gets evicted，
执行机制是 SGLang 原生 eviction/abort。不做 token-level 部分释放。

核心职责：
1. KV stats 获取：从 ``TokenToKVPool`` 读取 used/total
2. Request-level 调度：通过 scheduler_hook 在 ``get_next_batch_to_run()``
   前重排 waiting/running queue，影响 SGLang 原生的 admission 和 eviction
3. Scoring 回调：decode step 后更新评分策略
4. Lifecycle 管理：请求完成时清理 bid 和前缀追踪

"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bidkv.adapters.base import BaseAdapterMetrics, FrameworkAdapter
from bidkv.baselines.base import BaselineStrategy
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring.base import ScoringStrategy
from bidkv.solver import GreedyBidSolver, SolverConfig

logger = logging.getLogger(__name__)


class SGLangAdapter(FrameworkAdapter):
    """BidKV 在 SGLang 框架上的适配器（Mode A request-level 调度）。

    通过 scheduler_hook monkey-patch SGLang Scheduler 的
    ``get_next_batch_to_run()``，在 native batch selection 前执行
    request-level 调度决策（waiting/running 重排 + proactive preemption）。

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
    """

    def __init__(
        self,
        config: BidKVConfig,
        scoring: ScoringStrategy,
        *,
        scheduler: Any = None,
        pressure_config: PressureConfig | None = None,
        solver_config: SolverConfig | None = None,
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

        # 请求追踪
        # {request_id: list[int]} — 每个请求的 token ids
        self._request_tokens: dict[str, list[int]] = {}
        # {request_id: set[int]} — 每个请求中与其他请求共享的 token 位置
        self._shared_positions: dict[str, set[int]] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Mode A: request-level scheduling state
        self._cached_preempt_priority: dict[str, float] = {}
        self._last_priority_refresh: float = 0.0
        self._request_arrival_ms: dict[str, float] = {}

        # Metrics（与 vLLM adapter 对齐，便于跨框架对比）
        self._metrics = _AdapterMetrics()

        logger.info(
            "SGLangAdapter created: enabled=%s, kill_switch=%s",
            config.is_active,
            config.kill_switch,
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
        """将 bidkv request-level 调度注入到 SGLang 调度路径。

        需要先设置 scheduler。注入后，scheduler_hook 在每次
        ``get_next_batch_to_run()`` 前执行 request-level 调度。

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

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._shared_positions.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

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
    # Positional scoring decode step callback
    # ------------------------------------------------------------------

    def on_decode_step(self, request_id: str, attention_pattern: Sequence[float]) -> None:
        """评分策略 decode step 回调。

        由 positional_hook 在每个 decode step 后调用。

        Parameters
        ----------
        request_id:
            请求 ID。
        attention_pattern:
            当前 decode step 中 query token 对所有 KV token 的注意力权重。
        """
        if not self._config.is_active:
            return
        # 部分 scoring 策略（如 PositionalScoring）支持 decode-step 增量更新
        if hasattr(self._scoring, "update_from_decode_step"):
            self._scoring.update_from_decode_step(attention_pattern)
        self._metrics.record_decode_step(request_id)


class _AdapterMetrics(BaseAdapterMetrics):
    """SGLang adapter 运行指标。

    直接复用 ``BaseAdapterMetrics`` 的 6 个跨框架共同字段。
    SGLang 暂无框架特有指标。
    """


def _get_token_to_kv_pool(scheduler: Any) -> Any | None:
    """从 SGLang scheduler 获取 KV pool（allocator 或 pool）。

    SGLang >= 0.5.x 使用 ``token_to_kv_pool_allocator``（BaseTokenToKVPoolAllocator），
    直接暴露 ``size`` 属性和 ``available_size()`` 方法。
    """
    # SGLang >= 0.5.x: token_to_kv_pool_allocator（首选路径）
    if hasattr(scheduler, "token_to_kv_pool_allocator"):
        return scheduler.token_to_kv_pool_allocator
    # 旧版 SGLang: 直接属性
    if hasattr(scheduler, "token_to_kv_pool"):
        return scheduler.token_to_kv_pool
    # 旧版 SGLang: tp_server 下
    if hasattr(scheduler, "tp_server"):
        tp = scheduler.tp_server
        if hasattr(tp, "token_to_kv_pool"):
            return tp.token_to_kv_pool
    return None
