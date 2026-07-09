"""bidkv.adapters — Framework adapter layer.

Provides FrameworkAdapter ABC and per-framework implementations
(vLLM, SGLang) that bridge bidkv with LLM serving frameworks.
"""

from __future__ import annotations

from bidkv.adapters.base import BaseAdapterMetrics, FrameworkAdapter

__all__ = [
    "BaseAdapterMetrics",
    "FrameworkAdapter",
]
