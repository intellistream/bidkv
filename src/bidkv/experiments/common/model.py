"""Shared model defaults for experiments."""

from __future__ import annotations

import os

DEFAULT_MODEL_ENV_VAR = "BIDKV_MODEL"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def get_default_model() -> str:
    """Return the default experiment model name or path.

    Resolution order:
    1. ``BIDKV_MODEL`` environment variable
    2. Hugging Face model ID for the frozen paper model
    """
    return os.environ.get(DEFAULT_MODEL_ENV_VAR, DEFAULT_MODEL_NAME)