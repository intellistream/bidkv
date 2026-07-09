"""Unified experiment metric schema for BidKV paper experiments.

Covers Table 1 / Fig 3-6 indicators from #039, used by #047 calibration
runners and #048 experiment drivers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentMetrics:
    """Frozen experiment metrics matching the paper's evaluation schema.

    Fields correspond to the metrics reported in the BidKV paper:
    - SLO attainment and latency: ``slo_attainment_rate``, ``p99_ttft_ms``
    - Throughput: ``throughput_rps``
    - Eviction effectiveness: ``eviction_coverage``
    - Quality (optional, task-dependent): ``quality_rouge1``, ``quality_em``
    - Adapter internals: ``adapter_metrics`` (from ``BaseAdapterMetrics.as_dict()``)
    """

    slo_attainment_rate: float
    p99_ttft_ms: float
    throughput_rps: float
    eviction_coverage: float
    quality_rouge1: float | None = None
    quality_em: float | None = None
    adapter_metrics: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Export all metrics as a flat dictionary."""
        return {
            "slo_attainment_rate": self.slo_attainment_rate,
            "p99_ttft_ms": self.p99_ttft_ms,
            "throughput_rps": self.throughput_rps,
            "eviction_coverage": self.eviction_coverage,
            "quality_rouge1": self.quality_rouge1,
            "quality_em": self.quality_em,
            "adapter_metrics": dict(self.adapter_metrics),
        }
