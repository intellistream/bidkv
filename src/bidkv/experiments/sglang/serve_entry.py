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


def _patch_torch_register_fake_safe() -> None:
    """将 torch.library.register_fake 替换为安全版本。

    当 sgl_kernel C 扩展未加载（torch 版本不兼容）时，
    register_fake 调用会因 op 不存在而抛出 RuntimeError。
    此补丁使其在 op 不存在时静默跳过，允许 SGLang 正常导入 BF16 推理路径。
    """
    import torch

    _original = torch.library.register_fake

    def _safe_register_fake(op_name, func=None, **kwargs):  # type: ignore[override]
        def _check_and_register(fn):
            if "::" in str(op_name):
                ns, bare = str(op_name).split("::", 1)
                ops_ns = getattr(torch.ops, ns, None)
                if ops_ns is None or not hasattr(ops_ns, bare):
                    # op does not exist — skip registration silently
                    return fn
            return _original(op_name, fn, **kwargs)

        if func is None:
            # used as decorator: @torch.library.register_fake("ns::op")
            return _check_and_register
        else:
            # used directly: torch.library.register_fake("ns::op", fn)
            return _check_and_register(func)

    torch.library.register_fake = _safe_register_fake  # type: ignore[method-assign]
    logger.debug("torch.library.register_fake patched to safe version (sgl_kernel compat)")


def _patch_sglang_scheduler(strategy_name: str) -> None:
    """Monkey-patch SGLang Scheduler.__init__ 以注入 BidKV hooks。"""
    global _PATCHED
    if _PATCHED:
        return

    # Ensure torch.library.register_fake doesn't crash on missing sgl_kernel ops
    _patch_torch_register_fake_safe()

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
    from bidkv.scoring import PositionalScoring
    from bidkv.solver import SolverConfig

    config = BidKVConfig(enabled=True, kill_switch=False, delta_budget=0.30)
    scoring = PositionalScoring()

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


def _bidkv_run_scheduler_process(*args: object, **kwargs: object) -> None:
    """Scheduler subprocess entry point with BidKV hooks.

    This module-level function replaces SGLang's default run_scheduler_process so
    that BidKV hooks are installed inside the SPAWNED subprocess (where parent's
    in-memory monkey-patches are not inherited).

    Flow:
      1. Read BIDKV_STRATEGY from environment (inherited from parent).
      2. Patch Scheduler.__init__ on the class inside this subprocess.
      3. Call the original run_scheduler_process — which creates Scheduler(),
         triggering our patched __init__ and installing the hooks.
    """
    strategy_name = os.environ.get("BIDKV_STRATEGY", "sglang_default")

    # Patch Scheduler.__init__ in this subprocess before the Scheduler is created.
    from sglang.srt.managers.scheduler import Scheduler

    _orig_scheduler_init = Scheduler.__init__

    def _patched_scheduler_init(self: object, *a: object, **kw: object) -> None:  # noqa: ANN001
        _orig_scheduler_init(self, *a, **kw)
        _install_bidkv_hooks(self, strategy_name)

    Scheduler.__init__ = _patched_scheduler_init  # type: ignore[method-assign]

    from sglang.srt.managers.scheduler import run_scheduler_process

    run_scheduler_process(*args, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    """Entry point — 注入 BidKV 后启动 SGLang server。"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = os.environ.get("BIDKV_STRATEGY", "sglang_default")
    logger.info("SGLang BidKV entry point: strategy=%s", strategy)

    # Parse ServerArgs from CLI (same as sglang.launch_server does).
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])

    try:
        # Pass our subprocess wrapper so BidKV hooks run inside the spawned
        # Scheduler process.  This replaces the old runpy.run_module approach
        # which could not inject hooks across spawn boundaries.
        if server_args.grpc_mode:
            from sglang.srt.entrypoints.grpc_server import serve_grpc
            import asyncio

            asyncio.run(serve_grpc(server_args))
        elif getattr(server_args, "encoder_only", False):
            from sglang.srt.disaggregation.encode_server import launch_server as _ls

            _ls(server_args)
        else:
            from sglang.srt.entrypoints.http_server import launch_server

            launch_server(
                server_args,
                run_scheduler_process_func=_bidkv_run_scheduler_process,
            )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
