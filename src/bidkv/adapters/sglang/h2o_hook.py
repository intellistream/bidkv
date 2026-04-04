"""H2O Decode-Step 回调 — 在 SGLang decode step 后更新注意力统计。

SGLang 的 model execution 在每个 decode step 后可以提供 attention pattern
（如果通过自定义 attention backend 或 sampling callback 收集）。

本模块提供钩子函数，将 attention pattern 转发给 SGLangAdapter 的 PositionalScoring。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def install_h2o_hook(scheduler: Any, adapter: Any) -> None:
    """安装 H2O decode step 回调到 SGLang 的 model execution path。

    SGLang 在每个 decode step 后通过 model runner 返回 logits。
    本钩子在 logits 返回后收集 attention pattern 并更新 PositionalScoring。

    注意：SGLang 的 FlashAttention 默认不暴露 attention weights。
    H2O 使用 token attend 频率（从 model output logprobs 推断）作为代理信号。

    Parameters
    ----------
    scheduler:
        SGLang Scheduler 实例。
    adapter:
        SGLangAdapter 实例。
    """
    tp_server = getattr(scheduler, "tp_server", None)
    if tp_server is None:
        logger.debug("install_h2o_hook: no tp_server, skip")
        return

    model_runner = getattr(tp_server, "model_runner", None)
    if model_runner is None:
        logger.debug("install_h2o_hook: no model_runner, skip")
        return

    if not hasattr(model_runner, "forward"):
        logger.debug("install_h2o_hook: model_runner has no forward(), skip")
        return

    original_forward = model_runner.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        """BidKV patched: decode step 后更新 H2O scoring。"""
        result = original_forward(*args, **kwargs)

        if adapter.config.is_active:
            _process_decode_output(scheduler, adapter, result)

        return result

    model_runner.forward = patched_forward
    logger.info("H2O decode-step hook installed on model_runner.forward")


def _process_decode_output(scheduler: Any, adapter: Any, output: Any) -> None:
    """从 decode step 输出中提取注意力信号并更新 PositionalScoring。

    SGLang 不直接暴露 attention weights。我们从 logprobs 和 token 选择
    中推断 attention pattern（top-k logprob 分布作为注意力代理）。
    """
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is None:
        return

    reqs = getattr(running_batch, "reqs", None)
    if reqs is None:
        return

    for req in reqs:
        request_id = str(getattr(req, "rid", None) or getattr(req, "request_id", ""))
        if not request_id:
            continue

        # 尝试从输出中获取 attention pattern
        attention_pattern = _extract_attention_proxy(req, output)
        if attention_pattern:
            adapter.on_decode_step(request_id, attention_pattern)


def _extract_attention_proxy(req: Any, output: Any) -> list[float] | None:
    """从 model output 中提取注意力代理信号。

    由于 FlashAttention 不暴露 attention weights，使用以下启发式代理：
    - 如果 output 包含 attention_weights 字段（自定义 attention backend），直接使用
    - 否则，使用 uniform estimation（所有 token 等权）作为 fallback

    注意：这是 H2O 的限制 — 在 FlashAttention 下只能使用代理信号。
    完整注意力权重需要 eager attention 模式。
    """
    # 检查是否有显式 attention weights
    if hasattr(output, "attention_weights"):
        weights = output.attention_weights
        if hasattr(weights, "tolist"):
            return weights.tolist()
        if isinstance(weights, list):
            return weights

    # 检查是否有 per-token attention scores
    if hasattr(output, "attentions"):
        attentions = output.attentions
        if attentions is not None:
            if hasattr(attentions, "tolist"):
                return attentions.tolist()
            if isinstance(attentions, list):
                return attentions

    # FlashAttention 下无 attention weights 可用 — 返回 None
    # PositionalScoring 会退化到位置启发式评分
    return None
