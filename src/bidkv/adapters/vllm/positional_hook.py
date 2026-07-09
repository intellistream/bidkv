"""Positional Decode Hook — vLLM decode step 后的 PositionalScoring 更新回调。

在每个 decode step 完成后，从 vLLM 的 model_runner_output 中
提取位置代理注意力信息，更新 PositionalScoring 的累积统计。

vLLM 的 FlashAttention 不暴露 output_attentions，因此使用代理信号：
- 生成 token 的 position 作为 attend-to 频率的近似
- 累积统计随 decode step 增加而更精确

注入方式：
- 在 ``Scheduler.update_from_output()`` 的 patched 版本中调用
- 不单独 monkey-patch model_runner
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.vllm.adapter import VLLMAdapter

logger = logging.getLogger(__name__)


def update_positional_from_output(
    adapter: VLLMAdapter,
    scheduler: Any,
    model_runner_output: Any,
) -> None:
    """从 vLLM model_runner_output 更新 PositionalScoring。

    由 scheduler_hook 中的 _patched_update_from_output 调用。

    Parameters
    ----------
    adapter:
        VLLMAdapter 实例。
    scheduler:
        vLLM Scheduler 实例。
    model_runner_output:
        vLLM 的 ModelRunnerOutput。
    """
    if not adapter.config.is_active:
        return

    # 检查 scoring 是否有 update_from_decode_step 方法
    if not hasattr(adapter.scoring, "update_from_decode_step"):
        return

    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled_token_ids is None:
        return

    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
    if req_id_to_index is None:
        return

    running = getattr(scheduler, "running", [])

    for request in running:
        req_id = getattr(request, "request_id", None)
        if req_id is None:
            continue

        token_ids = adapter._request_tokens.get(req_id)
        if not token_ids:
            continue

        req_index = req_id_to_index.get(req_id)
        if req_index is None:
            continue

        # 生成代理注意力模式
        # 由于 FlashAttention 不暴露 attention weights，
        # 使用 position-based 代理：
        # - position 0 (attention sink) 获得高权重
        # - 最近 token 获得高权重
        # - 中间 token 获得衰减权重
        n = len(token_ids)
        attention_proxy = _generate_attention_proxy(n)

        adapter.on_decode_step(req_id, attention_proxy)


def _generate_attention_proxy(seq_len: int) -> list[float]:
    """生成 position-based 注意力代理模式。

    基于 attention sink (Xiao et al., 2023) 和 recency bias 观察：
    - Position 0~few：attention sink，权重最高
    - 最近 token：recency bias，权重较高
    - 中间 token：权重逐渐衰减

    Parameters
    ----------
    seq_len:
        序列长度。

    Returns
    -------
    list[float]
        每个位置的代理注意力权重。
    """
    if seq_len == 0:
        return []

    import math

    proxy: list[float] = []
    for i in range(seq_len):
        # Attention sink: 前几个 token
        sink_weight = 0.3 * math.exp(-i / max(1, seq_len * 0.02))
        # Recency: 越近的 token 权重越高
        recency_weight = 0.2 * (i / max(1, seq_len - 1)) if seq_len > 1 else 0.2
        # 均匀基线
        base_weight = 0.1
        proxy.append(sink_weight + recency_weight + base_weight)

    return proxy
