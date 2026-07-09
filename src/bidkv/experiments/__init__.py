"""bidkv.experiments — Experiment metrics, runners, and analysis.

包含：
- metrics.py — 统一指标 schema
- common/ — 共享基础设施（trace, audit, runner, report）
- sglang/ — SGLang 可移植性验证实验（#048）
"""

from __future__ import annotations

from bidkv.experiments.metrics import ExperimentMetrics

__all__ = [
    "ExperimentMetrics",
]
