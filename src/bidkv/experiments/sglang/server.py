"""SGLang server 生命周期管理 — BidKV 策略注入。

注入方案（拍板：方案 B — 环境变量驱动）:
- sglang_default: 原生 SGLang，不注入任何 hook
- slack_aware / bidkv: 通过 BIDKV_STRATEGY 环境变量 +
  包装启动脚本在 server 进程中注入 adapter

SGLang 没有类似 vLLM 的 general_plugins 机制，因此通过单独的
启动入口 (_sglang_with_bidkv_main) 在 server 初始化后注入 hook。
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bidkv.experiments.sglang.config import (
    STRATEGY_SGLANG_DEFAULT,
    SGLangServerConfig,
)

logger = logging.getLogger(__name__)


class SGLangServer:
    """管理 SGLang serving server 的启动、健康检查和停止。

    每个 strategy 启停一次 server，保证策略间无状态泄漏。
    """

    def __init__(self, config: SGLangServerConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: object | None = None  # log file handle
        self._log_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, strategy: str, *, audit_dir: str | None = None) -> None:
        """启动 SGLang server。

        Parameters
        ----------
        strategy:
            策略名称。sglang_default 启动原生 SGLang，
            其他策略通过环境变量 BIDKV_STRATEGY 注入 adapter。
        audit_dir:
            Fairness audit 日志目录路径（可选）。
        """
        if self._proc is not None:
            self.stop()

        env = os.environ.copy()
        env["BIDKV_STRATEGY"] = strategy
        if audit_dir:
            env["BIDKV_AUDIT_DIR"] = audit_dir

        if strategy == STRATEGY_SGLANG_DEFAULT:
            cmd = self._build_native_cmd()
            logger.info("Starting native SGLang server (no BidKV): %s", " ".join(cmd))
        else:
            cmd = self._build_bidkv_cmd()
            logger.info(
                "Starting SGLang server with BidKV (strategy=%s): %s",
                strategy,
                " ".join(cmd),
            )

        # Write server output to a log file instead of PIPE to avoid
        # deadlock when the 64KB pipe buffer fills up.
        log_dir = Path(audit_dir) if audit_dir else Path("results/sglang")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / f"server_{strategy}.log"
        self._log_file = open(self._log_path, "a")  # noqa: SIM115

        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        logger.info("SGLang server launched (pid=%d, strategy=%s)", self._proc.pid, strategy)

    def wait_ready(self, timeout_s: float | None = None) -> None:
        """轮询 /health 直到 server 就绪。

        Raises
        ------
        RuntimeError
            Server 进程退出或超时。
        """
        timeout = timeout_s or 300.0
        health_url = f"{self._config.base_url}/health"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
                log_text = ""
                if self._log_path and self._log_path.exists():
                    log_text = self._log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"SGLang server exited unexpectedly "
                    f"(code={self._proc.returncode})\n"
                    f"Output:\n{log_text}"
                )
            try:
                # Use a no-proxy opener to bypass HTTP_PROXY for local connections.
                no_proxy_handler = urllib.request.ProxyHandler({})
                opener = urllib.request.build_opener(no_proxy_handler)
                req = urllib.request.Request(health_url, method="GET")
                with opener.open(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("SGLang server healthy")
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)

        self.stop()
        raise RuntimeError(f"SGLang server did not become healthy within {timeout}s")

    def stop(self) -> None:
        """SIGTERM → wait → SIGKILL → 验证 GPU 释放。"""
        if self._proc is None:
            return

        pid = self._proc.pid
        logger.info("Stopping SGLang server (pid=%d)...", pid)

        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Server didn't stop gracefully, sending SIGKILL")
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=10)

        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        time.sleep(3)
        logger.info("SGLang server stopped")

    def _build_native_cmd(self) -> list[str]:
        """构建原生 SGLang 启动命令。"""
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            *self._config.to_cli_args(),
        ]

    def _build_bidkv_cmd(self) -> list[str]:
        """构建带 BidKV 注入的 SGLang 启动命令。

        使用 bidkv.experiments.sglang.serve 作为入口，
        在 server 进程内注入 adapter。
        """
        return [
            sys.executable,
            "-m",
            "bidkv.experiments.sglang.serve_entry",
            *self._config.to_cli_args(),
        ]
