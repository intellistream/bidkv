"""Custom vLLM API server with BidKV strategy injection.

Usage
-----
# Start with preempt-evict (no BidKV, vanilla vLLM):
BIDKV_STRATEGY=preempt-evict python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000

# Start with BidKV largest-first strategy:
BIDKV_STRATEGY=largest-first python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000

Hook mechanism
--------------
BidKV hooks are injected via the ``vllm.general_plugins`` entry point
(see ``bidkv.adapters.vllm.plugin``). The plugin reads ``BIDKV_STRATEGY``
from the environment and patches ``Scheduler.__init__`` so that BidKV hooks
are automatically installed when the Scheduler is created — including inside
the EngineCore subprocess that vLLM spawns via ``multiprocessing.spawn``.

This module only needs to ensure ``BIDKV_STRATEGY`` is set before starting
the vLLM server.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_STRATEGY_NAME: str = os.environ.get("BIDKV_STRATEGY", "preempt-evict")


def main() -> None:
    """Entry point — starts vLLM server with BidKV strategy via plugin."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = _STRATEGY_NAME
    logger.info("BidKV experiment server starting (strategy=%s)", strategy)

    if strategy == "preempt-evict":
        logger.info("preempt-evict: using vanilla vLLM (no BidKV)")
    else:
        logger.info(
            "BidKV strategy=%s: hooks will be injected via vllm.general_plugins "
            "entry point in the EngineCore subprocess",
            strategy,
        )

    # Ensure BIDKV_STRATEGY is in env for the subprocess to read
    os.environ["BIDKV_STRATEGY"] = strategy

    # Start the standard vLLM API server
    import runpy

    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
