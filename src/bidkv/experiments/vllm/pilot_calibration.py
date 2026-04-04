"""Phase 2: Pilot calibration — rate sweep to find effective eviction pressure zones.

Protocol §2-P2 & §3-Phase2: use 2 representative strategies (preempt-evict + largest-first)
with 50% request count (seed=99) to find rate_low / rate_mid / rate_high for each workload.

Usage
-----
conda run -n sagellm python -m bidkv.experiments.vllm.pilot_calibration \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.85
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bidkv.experiments.common.model import get_default_model
from bidkv.experiments.vllm.collector import RunResult
from bidkv.experiments.vllm.config import (
    WORKLOAD_LONG_CONTEXT,
    WORKLOAD_MIXED,
    ExperimentConfig,
    VLLMServerConfig,
)
from bidkv.experiments.vllm.freeze_traces import freeze_traces
from bidkv.experiments.vllm.runner import VLLMExperimentRunner

logger = logging.getLogger(__name__)

# Phase 2 pilot strategies (§3 [2-2])
PILOT_STRATEGIES = ("preempt-evict", "largest-first")

# Rate sweep sequence: start at 0.5, multiply by 1.5 each step (§3 [2-1])
MAX_RATE_STEPS = 15
RATE_START = 0.5
RATE_MULTIPLIER = 1.5

# Stop thresholds (§3 [2-1])
STOP_OOM_PCT = 0.20
STOP_TIMEOUT_PCT = 0.20


@dataclass
class PilotObservation:
    """Observation for a single (workload, rate, strategy) pilot run. §3 [2-3]."""

    workload: str
    request_rate: float
    strategy: str
    # Core metrics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    oom_count: int = 0
    timeout_count: int = 0
    abort_count: int = 0
    # Performance metrics
    throughput_rps: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    # KV pressure indicators
    eviction_count: int = 0
    peak_kv_utilization_pct: float = 0.0
    # Derived rates
    oom_pct: float = 0.0
    timeout_pct: float = 0.0
    # Status
    run_status: str = "completed"
    duration_s: float = 0.0


@dataclass
class PilotReport:
    """Full pilot calibration report for all workloads."""

    gpu_memory_utilization: float
    model: str
    observations: list[PilotObservation] = field(default_factory=list)
    # Selected rates per workload
    selected_rates: dict[str, dict[str, float | None]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _generate_rate_sweep() -> list[float]:
    """Generate rate sweep sequence: 0.5 × 1.5^n (§3 [2-1])."""
    rates = []
    rate = RATE_START
    for _ in range(MAX_RATE_STEPS):
        rates.append(round(rate, 2))
        rate *= RATE_MULTIPLIER
    return rates


def _extract_observation(
    result: RunResult,
    *,
    workload: str,
    rate: float,
    strategy: str,
) -> PilotObservation:
    """Extract pilot observation metrics from a RunResult."""
    total = len(result.request_results)
    successful = result.successful_requests
    failed = result.failed_requests

    # Classify failures
    oom_count = sum(1 for r in failed if r.error and "oom" in r.error.lower())
    timeout_count = sum(
        1
        for r in failed
        if r.error and ("timeout" in r.error.lower() or "timed out" in r.error.lower())
    )

    # P95 TTFT
    ttfts = sorted(r.ttft_ms for r in successful if r.ttft_ms is not None)
    p95_ttft = 0.0
    p99_ttft = 0.0
    if ttfts:
        p95_idx = max(0, int(len(ttfts) * 0.95) - 1)
        p99_idx = max(0, int(len(ttfts) * 0.99) - 1)
        p95_ttft = ttfts[p95_idx]
        p99_ttft = ttfts[p99_idx]

    # Eviction metrics from adapter
    _am = result.adapter_metrics
    eviction_count = int(
        _am.get("total_evictions", _am.get("total_compressions", 0)),
    )
    if eviction_count == 0:
        # Also check preemption count for preempt-evict
        eviction_count = int(result.adapter_metrics.get("total_preemptions", 0))
        eviction_count += int(result.adapter_metrics.get("total_pressure_events", 0))

    peak_kv_util = float(result.adapter_metrics.get("peak_kv_utilization_pct", 0))

    run_status = str(result.server_config.get("run_status", "completed"))

    return PilotObservation(
        workload=workload,
        request_rate=rate,
        strategy=strategy,
        total_requests=total,
        completed_requests=len(successful),
        failed_requests=len(failed),
        oom_count=oom_count,
        timeout_count=timeout_count,
        abort_count=1 if run_status == "aborted" else 0,
        throughput_rps=result.compute_throughput_rps(),
        p95_ttft_ms=p95_ttft,
        p99_ttft_ms=p99_ttft,
        eviction_count=eviction_count,
        peak_kv_utilization_pct=peak_kv_util,
        oom_pct=oom_count / total if total > 0 else 0.0,
        timeout_pct=timeout_count / total if total > 0 else 0.0,
        run_status=run_status,
        duration_s=result.duration_s,
    )


def _should_stop_sweep(obs: PilotObservation) -> bool:
    """Check if rate sweep should stop (§3 [2-1]: OOM > 20% or timeout > 20%)."""
    return obs.oom_pct > STOP_OOM_PCT or obs.timeout_pct > STOP_TIMEOUT_PCT


def run_pilot_calibration(
    *,
    model: str,
    gpu_memory_utilization: float = 0.85,
    output_dir: Path = Path("results/pilot"),
    data_dir: Path = Path("data"),
    port: int = 8000,
) -> PilotReport:
    """Run Phase 2 pilot calibration.

    For each workload, sweeps rates from 0.5 req/s upward (×1.5 per step),
    running preempt-evict and largest-first at each rate. Stops when OOM or
    timeout exceeds 20%.

    Parameters
    ----------
    model:
        Path to model weights.
    gpu_memory_utilization:
        GPU memory fraction for vLLM.
    output_dir:
        Directory for pilot results.
    data_dir:
        Directory with tokenized pool files.
    port:
        vLLM server port.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    results_dir = output_dir / "run_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = PilotReport(
        gpu_memory_utilization=gpu_memory_utilization,
        model=model,
    )

    rate_sweep = _generate_rate_sweep()
    logger.info("Rate sweep sequence: %s", rate_sweep)

    # For each workload, sweep rates
    workloads = [WORKLOAD_MIXED, WORKLOAD_LONG_CONTEXT]

    for workload in workloads:
        logger.info("=" * 60)
        logger.info("PILOT: workload=%s", workload)
        logger.info("=" * 60)

        max_rate_reached: dict[str, float] = {}  # per-strategy max rate before stop

        for rate in rate_sweep:
            logger.info("-" * 40)
            logger.info("PILOT: workload=%s rate=%.2f req/s", workload, rate)
            logger.info("-" * 40)

            # Generate pilot trace for this specific rate
            logger.info("Generating pilot trace for rate=%.2f...", rate)
            freeze_traces(
                rates=[rate],
                output_dir=traces_dir,
                data_dir=data_dir,
                seed=99,
                pilot=True,
            )

            all_strategies_stopped = True

            for strategy in PILOT_STRATEGIES:
                # Skip if this strategy already hit stop threshold
                if strategy in max_rate_reached:
                    logger.info(
                        "SKIP: %s already stopped at rate=%.2f",
                        strategy,
                        max_rate_reached[strategy],
                    )
                    continue

                all_strategies_stopped = False
                logger.info("Running: strategy=%s workload=%s rate=%.2f", strategy, workload, rate)

                # Create a minimal config for this single run
                config = ExperimentConfig(
                    strategies=(strategy,),
                    workloads=(workload,),
                    request_rates=(rate,),
                    runs_per_combo=1,
                    output_dir=results_dir,
                    server=VLLMServerConfig(
                        model=model,
                        gpu_memory_utilization=gpu_memory_utilization,
                        port=port,
                    ),
                    traces_dir=traces_dir,
                    warmup_requests=3,
                    request_timeout_s=120.0,
                    server_startup_timeout_s=300.0,
                    consecutive_timeout_abort=10,
                )

                runner = VLLMExperimentRunner(config)

                try:
                    results = runner.run_all()
                except Exception as exc:
                    logger.error(
                        "PILOT FAILED: strategy=%s workload=%s rate=%.2f error=%s",
                        strategy,
                        workload,
                        rate,
                        exc,
                    )
                    # Record failure observation
                    obs = PilotObservation(
                        workload=workload,
                        request_rate=rate,
                        strategy=strategy,
                        run_status=f"error: {exc}",
                    )
                    report.observations.append(obs)
                    max_rate_reached[strategy] = rate
                    continue

                if results:
                    result = results[0]
                    obs = _extract_observation(
                        result,
                        workload=workload,
                        rate=rate,
                        strategy=strategy,
                    )
                    report.observations.append(obs)

                    logger.info(
                        "PILOT RESULT: %s %s rate=%.2f → "
                        "throughput=%.2f, P95_TTFT=%.0fms, "
                        "evictions=%d, OOM=%.0f%%, timeout=%.0f%%, "
                        "completed=%d/%d, status=%s",
                        strategy,
                        workload,
                        rate,
                        obs.throughput_rps,
                        obs.p95_ttft_ms,
                        obs.eviction_count,
                        obs.oom_pct * 100,
                        obs.timeout_pct * 100,
                        obs.completed_requests,
                        obs.total_requests,
                        obs.run_status,
                    )

                    if _should_stop_sweep(obs):
                        logger.warning(
                            "STOP: %s hit threshold at rate=%.2f (OOM=%.0f%%, timeout=%.0f%%)",
                            strategy,
                            rate,
                            obs.oom_pct * 100,
                            obs.timeout_pct * 100,
                        )
                        max_rate_reached[strategy] = rate

            # If all strategies have stopped, no need to try higher rates
            if all_strategies_stopped:
                logger.info(
                    "All strategies stopped for workload=%s. Moving to next workload.", workload
                )
                break

    # Save full report
    _save_report(report, output_dir)

    # Print summary
    _print_summary(report)

    return report


def _save_report(report: PilotReport, output_dir: Path) -> None:
    """Save pilot report as JSON."""
    report_path = output_dir / "pilot_report.json"
    data = {
        "gpu_memory_utilization": report.gpu_memory_utilization,
        "model": report.model,
        "observations": [asdict(o) for o in report.observations],
        "selected_rates": report.selected_rates,
        "notes": report.notes,
    }
    report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Pilot report saved to %s", report_path)


def _print_summary(report: PilotReport) -> None:
    """Print a human-readable pilot summary to logger."""
    logger.info("=" * 70)
    logger.info("PILOT CALIBRATION SUMMARY")
    logger.info("=" * 70)
    logger.info("GPU memory utilization: %.0f%%", report.gpu_memory_utilization * 100)

    # Group by workload
    for workload in [WORKLOAD_MIXED, WORKLOAD_LONG_CONTEXT]:
        logger.info("-" * 50)
        logger.info("Workload: %s", workload)

        for strategy in PILOT_STRATEGIES:
            obs_list = [
                o for o in report.observations if o.workload == workload and o.strategy == strategy
            ]
            if not obs_list:
                continue

            logger.info("  Strategy: %s", strategy)
            logger.info(
                "  %-8s %-10s %-12s %-10s %-8s %-8s %-10s",
                "Rate",
                "Throughput",
                "P95_TTFT",
                "Evictions",
                "OOM%",
                "TO%",
                "Status",
            )
            for o in sorted(obs_list, key=lambda x: x.request_rate):
                logger.info(
                    "  %-8.2f %-10.2f %-12.0f %-10d %-8.1f %-8.1f %-10s",
                    o.request_rate,
                    o.throughput_rps,
                    o.p95_ttft_ms,
                    o.eviction_count,
                    o.oom_pct * 100,
                    o.timeout_pct * 100,
                    o.run_status,
                )

    logger.info("=" * 70)
    logger.info(
        "Next step: review pilot_report.json, select rate_low/mid/high "
        "for each workload, then write calibration_report.md"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Phase 2: Pilot calibration — find effective eviction pressure zones"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help="Model name or local path.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory fraction (default: 0.85). §3 [2-5] may lower to 0.80/0.75.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pilot",
        help="Output directory for pilot results.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with tokenized pool files.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port.",
    )

    args = parser.parse_args()

    run_pilot_calibration(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        port=args.port,
    )


if __name__ == "__main__":
    main()
