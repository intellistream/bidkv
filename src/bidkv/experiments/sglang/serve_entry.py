"""SGLang server 启动入口（BidKV 注入版）。

当 BIDKV_STRATEGY != sglang_default 时，runner 使用此模块作为 server 进程入口。
它在 SGLang server 启动前注入 BidKV scheduler hook。

注入机制（方案 B — 环境变量驱动 + monkey-patch）:
1. 读取 BIDKV_STRATEGY 环境变量
2. Monkey-patch SGLang Scheduler.__init__，在初始化后自动注入 BidKV hooks
3. 启动 SGLang launch_server

Fallback:
如果 SGLang Scheduler 路径变化（版本不兼容），以明确错误信息失败。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PATCHED = False


def _patch_sglang_scheduler(strategy_name: str) -> None:
    """Monkey-patch SGLang Scheduler.__init__ 以注入 BidKV hooks。"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from sglang.srt.managers.scheduler import Scheduler
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import sglang.srt.managers.scheduler.Scheduler. "
            "Ensure SGLang is installed: pip install sglang[all]"
        ) from exc

    original_init = Scheduler.__init__

    def _bidkv_patched_init(self: object, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        _install_bidkv_hooks(self, strategy_name)

    Scheduler.__init__ = _bidkv_patched_init  # type: ignore[method-assign]
    _PATCHED = True
    logger.info("SGLang Scheduler.__init__ patched for BidKV (strategy=%s)", strategy_name)


def _install_bidkv_hooks(scheduler: object, strategy_name: str) -> None:
    """在 Scheduler 实例上安装 BidKV hooks。"""
    from bidkv.adapters.sglang.adapter import SGLangAdapter
    from bidkv.adapters.sglang.scheduler_hook import install_scheduler_hook
    from bidkv.baselines import BaselineRegistry
    from bidkv.config import BidKVConfig
    from bidkv.experiments.sglang.config import STRATEGY_BASELINE_MAP
    from bidkv.pressure import PressureConfig
    from bidkv.scoring import H2OScoring
    from bidkv.solver import SolverConfig

    config = BidKVConfig(enabled=True, kill_switch=False, delta_budget=0.30)
    scoring = H2OScoring()

    pressure_config = PressureConfig(
        enabled=True,
        threshold_pct=0.80,
        min_free_tokens=1024,
    )
    solver_config = SolverConfig(
        enabled=True,
        delta_budget=config.delta_budget,
        max_bids_per_solve=200,
    )

    # Resolve baseline strategy from registry
    registry = BaselineRegistry()
    registry.create_default_registry()
    baseline_name = STRATEGY_BASELINE_MAP.get(strategy_name, strategy_name)
    strategy = registry.get(baseline_name)

    # Resolve audit_dir from env (set by runner)
    audit_dir_str = os.environ.get("BIDKV_AUDIT_DIR")
    audit_dir = Path(audit_dir_str) if audit_dir_str else None

    adapter = SGLangAdapter(
        config=config,
        scoring=scoring,
        scheduler=scheduler,
        pressure_config=pressure_config,
        solver_config=solver_config,
        experiment_strategy=strategy,
        experiment_strategy_name=strategy_name,
        audit_dir=audit_dir,
    )

    install_scheduler_hook(scheduler, adapter)

    # Store reference on scheduler for external access
    scheduler._bidkv_experiment_adapter = adapter  # type: ignore[attr-defined]

    logger.info("BidKV hooks installed on SGLang Scheduler (strategy=%s)", strategy_name)


def main() -> None:
    """Entry point — 注入 BidKV 后启动 SGLang server。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = os.environ.get("BIDKV_STRATEGY", "sglang_default")
    logger.info("SGLang BidKV entry point: strategy=%s", strategy)

    if strategy != "sglang_default":
        _patch_sglang_scheduler(strategy)

    # 启动 SGLang server
    import runpy

    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
