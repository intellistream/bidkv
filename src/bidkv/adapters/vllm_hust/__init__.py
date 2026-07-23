# SPDX-License-Identifier: Apache-2.0
"""vLLM HUST adapter for BidKV victim selector plugin.

Provides :class:`BidkvVictimSelector` that implements the
``VictimSelector`` protocol from ``vllm.v1.core.sched.victim_selector``,
auto-discovered via the ``vllm.victim_selector`` entry-point group.
"""

from bidkv.adapters.vllm_hust.selector import (
    BidkvSelectorConfig,
    BidkvVictimSelector,
    UtilityCandidateScore,
)

__all__ = [
    "BidkvSelectorConfig",
    "BidkvVictimSelector",
    "UtilityCandidateScore",
]
