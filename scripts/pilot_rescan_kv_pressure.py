#!/usr/bin/env python3
"""Phase 2 Rescan: KV-pressure pilot with reduced gpu_memory_utilization.

Key finding from pilot v2: at gpu_mem=0.85, no KV eviction occurs. The system
hits compute saturation before KV memory pressure.

Per protocol §2 [2-5] Situation A, we lower gpu_memory_utilization to force
KV pressure:
  - Long-context @ gpu_mem=0.50: KV budget ≈ 8.0 GB, 32 concurrent × 2180 tok
    = 106% theoretical util → guaranteed eviction
  - Mixed @ gpu_mem=0.50: only 20% util → still no eviction; relegate to appendix
    per protocol

This script runs the long_context workload only, with:
  - gpu_memory_utilization = 0.375 (validated: KV peak=97.7%, 3 preemptions at 15 concurrent)
  - request_timeout_s = 120.0 (formal timeout)
  - Rates: 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
  - Strategies: preempt-evict, h2o-style
  - 1 run per combination, early stop when failure > 50%

Usage:
    python scripts/pilot_rescan_kv_pressure.py
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
import time
import urllib.request
from pathlib import Path

from bidkv.experiments.vllm.collector import save_run_result
from bidkv.experiments.vllm.config import ExperimentConfig, VLLMServerConfig
from bidkv.experiments.vllm.runner import VLLMExperimentRunner
from bidkv.experiments.vllm.workload import load_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pilot_rescan")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "/home/cyb/Llama-3.1-8B-Instruct"
GPU_MEM = 0.375  # validated: ~1000 KV blocks, peak KV=97.7% at 15 concurrent
TRACES_DIR = Path("results/pilot/traces")
OUTPUT_DIR = Path("results/pilot/rescan_kv")
STRATEGIES = ("preempt-evict", "h2o-style")
RATES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
WORKLOAD = "long_context"
TIMEOUT_S = 120.0
FAILURE_THRESHOLD = 0.50  # early stop when > 50% failure


def _get_kv_metrics(base_url: str = "http://127.0.0.1:8000") -> dict:
    """Fetch KV cache metrics from vLLM Prometheus endpoint."""
    try:
        req = urllib.request.Request(f"{base_url}/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode()
        metrics = {}
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            if "kv_cache_usage_perc" in line:
                metrics["kv_cache_usage_perc"] = float(line.split()[-1])
            elif "num_requests_running" in line and "HELP" not in line:
                metrics["num_requests_running"] = float(line.split()[-1])
            elif "num_requests_waiting" in line and "HELP" not in line:
                metrics["num_requests_waiting"] = float(line.split()[-1])
            elif "num_preemptions_total" in line and "HELP" not in line:
                metrics["num_preemptions_total"] = float(line.split()[-1])
        return metrics
    except Exception:
        return {}


class _KVPeakSampler:
    """Background thread that polls KV metrics and records peak usage."""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_kv_usage: float = 0.0
        self.peak_running: float = 0.0
        self.samples: int = 0

    def start(self) -> None:
        self._stop.clear()
        self.peak_kv_usage = 0.0
        self.peak_running = 0.0
        self.samples = 0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return {
            "peak_kv_usage_perc": self.peak_kv_usage,
            "peak_running": self.peak_running,
            "samples": self.samples,
        }

    def _poll(self) -> None:
        while not self._stop.is_set():
            m = _get_kv_metrics()
            if m:
                self.samples += 1
                kv = m.get("kv_cache_usage_perc", 0.0)
                run = m.get("num_requests_running", 0.0)
                if kv > self.peak_kv_usage:
                    self.peak_kv_usage = kv
                if run > self.peak_running:
                    self.peak_running = run
            self._stop.wait(self._interval)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations: list[dict] = []

    for strategy in STRATEGIES:
        logger.info("=" * 70)
        logger.info("Strategy: %s", strategy)
        logger.info("=" * 70)

        # Build config with reduced gpu_mem
        server_config = VLLMServerConfig(
            model=MODEL_PATH,
            gpu_memory_utilization=GPU_MEM,
            max_num_seqs=32,
            enforce_eager=True,
            disable_frontend_multiprocessing=True,
            max_model_len=8192,
        )
        config = ExperimentConfig(
            strategies=(strategy,),
            workloads=(WORKLOAD,),
            request_rates=tuple(RATES),
            runs_per_combo=1,
            output_dir=OUTPUT_DIR,
            server=server_config,
            traces_dir=TRACES_DIR,
            request_timeout_s=TIMEOUT_S,
            warmup_requests=3,
            consecutive_timeout_abort=10,
        )

        runner = VLLMExperimentRunner(config)

        # Manually load traces
        for rate in RATES:
            trace_path = TRACES_DIR / f"long_rate{rate}.json"
            key = f"{WORKLOAD}__{rate}"
            runner._traces[key] = load_trace(trace_path)
            logger.info("  Loaded trace: %s (%d requests)", key, runner._traces[key].num_requests)

        # Start server once per strategy
        logger.info("Starting vLLM server (gpu_mem=%.3f)...", GPU_MEM)
        runner._start_server(strategy)
        runner._warmup(strategy)
        logger.info("Server ready.")

        consecutive_high_failure = 0

        for rate in RATES:
            logger.info("-" * 50)
            logger.info("Running: %s @ %s @ rate=%.2f", strategy, WORKLOAD, rate)

            # Health check — restart server if dead
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
            except Exception:
                logger.warning("Server unhealthy before rate=%.2f, restarting...", rate)
                with contextlib.suppress(Exception):
                    runner._stop_server()
                time.sleep(5)
                runner._start_server(strategy)
                runner._warmup(strategy)
                logger.info("Server restarted successfully.")

            pre_metrics = _get_kv_metrics()
            sampler = _KVPeakSampler(interval=1.0)
            sampler.start()

            t0 = time.time()
            try:
                result = runner._run_single(
                    strategy=strategy,
                    workload=WORKLOAD,
                    request_rate=rate,
                    run_index=0,
                )
            except Exception as exc:
                peak_info = sampler.stop()
                logger.error("Run failed with exception: %s", exc)
                obs = {
                    "strategy": strategy,
                    "workload": WORKLOAD,
                    "rate": rate,
                    "gpu_mem": GPU_MEM,
                    "error": str(exc),
                    "status": "error",
                    "kv_peak": peak_info,
                }
                observations.append(obs)
                # Save error observations to disk
                obs_path = OUTPUT_DIR / "rescan_observations.json"
                obs_path.write_text(
                    json.dumps(observations, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                consecutive_high_failure += 1
                if consecutive_high_failure >= 2:
                    logger.warning("2 consecutive failures → stopping rate sweep for %s", strategy)
                    break
                continue

            duration = time.time() - t0
            peak_info = sampler.stop()
            post_metrics = _get_kv_metrics()

            # Compute preemption delta
            pre_preempt = pre_metrics.get("num_preemptions_total", 0.0)
            post_preempt = post_metrics.get("num_preemptions_total", 0.0)
            delta_preemptions = post_preempt - pre_preempt

            total = len(result.request_results)
            succeeded = len(result.successful_requests)
            failed = len(result.failed_requests)
            timed_out = sum(
                1
                for r in result.request_results
                if r.error and ("timeout" in r.error.lower() or "timed out" in r.error.lower())
            )
            failure_rate = (total - succeeded) / total if total > 0 else 0

            # Compute metrics
            throughput = result.compute_throughput_rps()
            p50_ttft = 0.0
            p95_ttft = 0.0
            p99_ttft = 0.0
            if result.successful_requests:
                ttfts = sorted(r.ttft_ms for r in result.successful_requests)
                n = len(ttfts)
                p50_ttft = ttfts[max(0, int(n * 0.50) - 1)]
                p95_ttft = ttfts[max(0, int(n * 0.95) - 1)]
                p99_ttft = ttfts[max(0, int(n * 0.99) - 1)]

            run_status = result.server_config.get("run_status", "completed")

            obs = {
                "strategy": strategy,
                "workload": WORKLOAD,
                "rate": rate,
                "gpu_mem": GPU_MEM,
                "total": total,
                "succeeded": succeeded,
                "failed_other": failed - timed_out,
                "timed_out": timed_out,
                "failure_rate": failure_rate,
                "throughput": throughput,
                "p50_ttft": p50_ttft,
                "p95_ttft": p95_ttft,
                "p99_ttft": p99_ttft,
                "duration_s": duration,
                "run_status": run_status,
                "adapter_metrics": dict(result.adapter_metrics),
                "kv_peak": peak_info,
                "delta_preemptions": delta_preemptions,
            }
            observations.append(obs)

            # Save individual run result
            save_run_result(result, OUTPUT_DIR)

            logger.info(
                "  Result: %d/%d succeeded (%.0f%% fail), throughput=%.3f rps, "
                "P95 TTFT=%.1f ms, duration=%.1fs, status=%s",
                succeeded,
                total,
                failure_rate * 100,
                throughput,
                p95_ttft,
                duration,
                run_status,
            )
            logger.info(
                "  KV peak=%.1f%%, delta_preemptions=%.0f, peak_running=%.0f",
                peak_info["peak_kv_usage_perc"] * 100,
                delta_preemptions,
                peak_info["peak_running"],
            )
            logger.info("  Adapter metrics: %s", result.adapter_metrics)

            # Save observations incrementally (BEFORE early stop check)
            obs_path = OUTPUT_DIR / "rescan_observations.json"
            obs_path.write_text(
                json.dumps(observations, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Early stop check
            if failure_rate > FAILURE_THRESHOLD:
                consecutive_high_failure += 1
                logger.warning(
                    "  HIGH FAILURE (%.0f%%) — consecutive=%d",
                    failure_rate * 100,
                    consecutive_high_failure,
                )
                if consecutive_high_failure >= 2:
                    logger.warning("  → Stopping rate sweep for %s", strategy)
                    break
            else:
                consecutive_high_failure = 0

        # Stop server
        runner._stop_server()
        logger.info("Server stopped for %s. Waiting 10s for GPU cleanup...", strategy)
        time.sleep(10)

    # Final save
    obs_path = OUTPUT_DIR / "rescan_observations.json"
    obs_path.write_text(
        json.dumps(observations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("=" * 70)
    logger.info("Rescan complete. %d observations saved to %s", len(observations), obs_path)

    # Print summary
    print("\n" + "=" * 70)
    print(f"RESCAN SUMMARY (long_context @ gpu_mem={GPU_MEM:.2f})")
    print("=" * 70)
    for obs in observations:
        if "error" in obs:
            print(f"  {obs['strategy']:16s} rate={obs['rate']:.2f}  ERROR: {obs['error']}")
        else:
            kv_peak = obs.get("kv_peak", {}).get("peak_kv_usage_perc", 0)
            delta_p = obs.get("delta_preemptions", 0)
            print(
                f"  {obs['strategy']:16s} rate={obs['rate']:.2f}  "
                f"{obs['succeeded']}/{obs['total']} ok  "
                f"thru={obs['throughput']:.3f}  "
                f"P95={obs['p95_ttft']:.0f}ms  "
                f"KVpeak={kv_peak:.1%}  "
                f"preempt={delta_p:.0f}  "
                f"adapter={obs.get('adapter_metrics', {})}"
            )


if __name__ == "__main__":
    main()
