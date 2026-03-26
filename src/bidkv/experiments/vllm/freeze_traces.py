"""Freeze reproducible workload traces for vLLM experiments.

Usage
-----
# Generate pilot traces (seed=99, 50% requests)
python -m bidkv.experiments.vllm.freeze_traces \
    --mode pilot \
    --rates 1.0,2.0,4.0 \
    --output-dir experiments/vllm/traces/pilot

# Generate formal frozen traces (seed=42, full request count)
python -m bidkv.experiments.vllm.freeze_traces \
    --mode formal \
    --rates 1.5,3.0,6.0 \
    --output-dir experiments/vllm/traces

Traces load from pre-tokenized ShareGPT JSONL pools (data/sharegpt_*_pool.jsonl),
generate Poisson arrival timestamps, and save as frozen JSON per protocol §6.
Once frozen, experiments always replay the same requests for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from bidkv.experiments.vllm.config import (
    WORKLOAD_LONG_CONTEXT,
    WORKLOAD_MIXED,
    WORKLOAD_NUM_REQUESTS,
    WORKLOAD_REQUEST_RATES,
)
from bidkv.experiments.vllm.workload import (
    RequestTrace,
    WorkloadTrace,
    save_trace,
)

logger = logging.getLogger(__name__)

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = Path("data")


def _load_pool(pool_path: Path) -> list[dict[str, object]]:
    """Load a pre-tokenized JSONL pool file.

    Each line is a JSON object with keys:
        id, prompt, output, prompt_tokens, output_tokens
    """
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Pool file not found: {pool_path}\n"
            "Run Phase 1 tokenization first to generate pool files."
        )
    records: list[dict[str, object]] = []
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded pool: %s (%d records)", pool_path, len(records))
    return records


def _generate_poisson_arrivals(
    num_requests: int,
    request_rate: float,
    rng: random.Random,
) -> list[float]:
    """Generate Poisson arrival timestamps (ms) using expovariate.

    Parameters
    ----------
    num_requests:
        Number of requests to generate.
    request_rate:
        Arrival rate in requests per second.
    rng:
        Seeded random generator for reproducibility.

    Returns
    -------
    list[float]
        Cumulative arrival timestamps in milliseconds.
    """
    rate_per_ms = request_rate / 1000.0
    timestamps: list[float] = []
    current_time = 0.0
    for _ in range(num_requests):
        inter_arrival = rng.expovariate(rate_per_ms)
        current_time += inter_arrival
        timestamps.append(current_time)
    return timestamps


def generate_trace(
    pool: list[dict[str, object]],
    *,
    workload_name: str,
    num_requests: int,
    request_rate: float,
    seed: int,
    max_output_tokens: int = 256,
) -> WorkloadTrace:
    """Generate a frozen trace from a pre-tokenized pool.

    Parameters
    ----------
    pool:
        Pre-tokenized JSONL records.
    workload_name:
        Workload identifier (mixed / long_context).
    num_requests:
        Number of requests to sample.
    request_rate:
        Poisson arrival rate (req/s).
    seed:
        Random seed for sampling and arrival generation.
    max_output_tokens:
        Cap on output token count per request.
    """
    rng = random.Random(seed)

    if len(pool) < num_requests:
        raise ValueError(
            f"Pool has {len(pool)} records but need {num_requests} for {workload_name}. "
            "Ensure pool is large enough."
        )

    # Sample without replacement
    sampled = rng.sample(pool, num_requests)

    # Generate Poisson arrival timestamps
    arrivals = _generate_poisson_arrivals(num_requests, request_rate, rng)

    # Build request traces
    prefix = "mixed" if workload_name == WORKLOAD_MIXED else "long"
    requests: list[RequestTrace] = []
    for i, (record, arrival_ms) in enumerate(zip(sampled, arrivals, strict=False)):
        output_tokens = int(record.get("output_tokens", 128))
        max_tokens = min(output_tokens, max_output_tokens)
        max_tokens = max(max_tokens, 1)  # at least 1

        requests.append(
            RequestTrace(
                request_id=f"{prefix}-{i:04d}",
                prompt=str(record["prompt"]),
                max_tokens=max_tokens,
                arrival_time_ms=arrival_ms,
                metadata={
                    "actual_prompt_tokens": str(record.get("prompt_tokens", 0)),
                    "dataset": "sharegpt",
                    "original_id": str(record.get("id", "")),
                },
            )
        )

    return WorkloadTrace(
        workload_name=workload_name,
        requests=requests,
        request_rate=request_rate,
        dataset_source="ShareGPT_Vicuna_unfiltered",
        frozen_at="",  # Set by caller for deterministic hashing
        seed=seed,
    )


def freeze_traces(
    *,
    rates: list[float] | None = None,
    workload_rates: dict[str, list[float]] | None = None,
    output_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 42,
    pilot: bool = False,
) -> dict[str, str]:
    """Generate all frozen traces and manifest.

    Parameters
    ----------
    rates:
        Fallback request rates (req/s) used when workload_rates is not provided.
    workload_rates:
        Per-workload rates mapping (e.g. {"mixed": [2.0, 3.8, 5.7], ...}).
        Takes precedence over ``rates``.
    output_dir:
        Directory to write trace JSON files.
    data_dir:
        Directory containing sharegpt_mixed_pool.jsonl and sharegpt_long_pool.jsonl.
    seed:
        Random seed (42 for formal, 99 for pilot).
    pilot:
        If True, use 50% of formal request count.

    Returns
    -------
    dict[str, str]
        Mapping from trace filename to SHA-256 hash.
    """
    if rates is None and workload_rates is None:
        raise ValueError("Either rates or workload_rates must be provided")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pools
    mixed_pool = _load_pool(data_dir / "sharegpt_mixed_pool.jsonl")
    long_pool = _load_pool(data_dir / "sharegpt_long_pool.jsonl")

    pool_map = {
        WORKLOAD_MIXED: mixed_pool,
        WORKLOAD_LONG_CONTEXT: long_pool,
    }

    hashes: dict[str, str] = {}
    frozen_at = datetime.now(tz=timezone.utc).isoformat()

    for workload_name, pool in pool_map.items():
        base_num = WORKLOAD_NUM_REQUESTS[workload_name]
        num_requests = base_num // 2 if pilot else base_num

        # Per-workload rates take precedence over fallback
        wl_rates: list[float]
        if workload_rates and workload_name in workload_rates:
            wl_rates = workload_rates[workload_name]
        elif rates is not None:
            wl_rates = rates
        else:
            continue

        for rate in wl_rates:
            trace = generate_trace(
                pool,
                workload_name=workload_name,
                num_requests=num_requests,
                request_rate=rate,
                seed=seed,
            )

            # Naming convention: mixed_rate1.5.json, long_rate3.0.json
            prefix = "mixed" if workload_name == WORKLOAD_MIXED else "long"
            filename = f"{prefix}_rate{rate}.json"
            trace_path = output_dir / filename
            save_trace(trace, trace_path)

            # Compute SHA-256 hash of the saved file content
            content = trace_path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            hashes[filename] = file_hash

            logger.info(
                "Frozen %s: %d requests, rate=%.1f req/s, seed=%d, hash=%s",
                filename,
                trace.num_requests,
                rate,
                seed,
                file_hash[:12],
            )

    # Save manifest
    manifest = {
        "frozen_at": frozen_at,
        "seed": seed,
        "pilot": pilot,
        "rates": rates,
        "workload_rates": workload_rates,
        "workloads": list(pool_map.keys()),
        "num_requests": {
            wl: WORKLOAD_NUM_REQUESTS[wl] // 2 if pilot else WORKLOAD_NUM_REQUESTS[wl]
            for wl in pool_map
        },
        "hashes": hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest saved to %s", manifest_path)

    return hashes


def verify_reproducibility(
    *,
    rates: list[float] | None = None,
    workload_rates: dict[str, list[float]] | None = None,
    output_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 42,
    pilot: bool = False,
) -> bool:
    """Regenerate all traces and verify SHA-256 hashes match manifest.

    Returns True if all hashes match.
    """
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["hashes"]

    # Regenerate into a temp location and compare
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        new_hashes = freeze_traces(
            rates=rates,
            workload_rates=workload_rates,
            output_dir=tmp_dir,
            data_dir=data_dir,
            seed=seed,
            pilot=pilot,
        )

    mismatches = []
    for filename, expected in expected_hashes.items():
        actual = new_hashes.get(filename, "MISSING")
        if actual != expected:
            mismatches.append((filename, expected[:12], actual[:12]))

    if mismatches:
        for fname, exp, act in mismatches:
            logger.error("Hash mismatch: %s expected=%s actual=%s", fname, exp, act)
        return False

    logger.info("Reproducibility verification PASSED: all %d hashes match", len(expected_hashes))
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Freeze workload traces for vLLM experiment")
    parser.add_argument(
        "--mode",
        choices=["pilot", "formal", "verify"],
        default="formal",
        help="pilot: 50%% requests seed=99; formal: full requests seed=42; verify: check hashes.",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default=None,
        help="Comma-separated fallback rates (req/s). Overridden by --mixed-rates/--long-rates.",
    )
    parser.add_argument(
        "--mixed-rates",
        type=str,
        default=None,
        help="Comma-separated rates for mixed workload.",
    )
    parser.add_argument(
        "--long-rates",
        type=str,
        default=None,
        help="Comma-separated rates for long_context workload.",
    )
    parser.add_argument(
        "--use-frozen-rates",
        action="store_true",
        help="Use WORKLOAD_REQUEST_RATES from config (recommended for formal).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/vllm/traces",
        help="Output directory for frozen traces.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory with tokenized pool JSONL files.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Build rate arguments
    rates: list[float] | None = None
    workload_rates: dict[str, list[float]] | None = None

    if args.use_frozen_rates:
        workload_rates = {k: list(v) for k, v in WORKLOAD_REQUEST_RATES.items()}
    else:
        if args.mixed_rates or args.long_rates:
            workload_rates = {}
            if args.mixed_rates:
                workload_rates["mixed"] = [float(r) for r in args.mixed_rates.split(",")]
            if args.long_rates:
                workload_rates["long_context"] = [float(r) for r in args.long_rates.split(",")]
        if args.rates:
            rates = [float(r.strip()) for r in args.rates.split(",")]

    if rates is None and workload_rates is None:
        parser.error("Provide --rates, --mixed-rates/--long-rates, or --use-frozen-rates")

    if args.mode == "pilot":
        freeze_traces(
            rates=rates,
            workload_rates=workload_rates,
            output_dir=output_dir,
            data_dir=data_dir,
            seed=99,
            pilot=True,
        )
    elif args.mode == "formal":
        freeze_traces(
            rates=rates,
            workload_rates=workload_rates,
            output_dir=output_dir,
            data_dir=data_dir,
            seed=42,
            pilot=False,
        )
    elif args.mode == "verify":
        ok = verify_reproducibility(
            rates=rates,
            workload_rates=workload_rates,
            output_dir=output_dir,
            data_dir=data_dir,
            seed=42,
            pilot=False,
        )
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
