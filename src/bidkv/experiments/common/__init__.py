"""bidkv.experiments.common — 实验公共基础设施。

共享的 frozen trace 加载/验证、实验运行器、候选池审计日志。
vLLM (#047) 和 SGLang (#048) 实验均使用本模块。
"""

from __future__ import annotations

from bidkv.experiments.common.audit import AuditLogger, PressureEventAudit
from bidkv.experiments.common.report import ExperimentReport, RunResult
from bidkv.experiments.common.runner import BaseExperimentRunner, ExperimentConfig
from bidkv.experiments.common.trace import FrozenTrace, TraceEntry, load_trace

__all__ = [
    "AuditLogger",
    "BaseExperimentRunner",
    "ExperimentConfig",
    "ExperimentReport",
    "FrozenTrace",
    "PressureEventAudit",
    "RunResult",
    "TraceEntry",
    "load_trace",
]
