"""Frozen trace 定义与加载。

Frozen traces 保证 vLLM (#047) 和 SGLang (#048) 使用相同的请求序列，
确保跨框架实验结果可比。

Trace 格式 (JSON Lines):
  {"request_id": "req-001", "prompt_tokens": [1,2,...], "max_new_tokens": 128,
   "arrival_offset_ms": 0, "workload": "chat", "slo_ttft_ms": 500}
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceEntry:
    """一条 frozen trace 记录。

    Attributes
    ----------
    request_id:
        唯一请求 ID（跨框架一致）。
    prompt_tokens:
        Prompt token ID 序列。
    max_new_tokens:
        期望生成的最大 token 数。
    arrival_offset_ms:
        相对于 trace 起始的到达时间偏移（毫秒）。
    workload:
        工作负载类型（chat / qa / summarization）。
    slo_ttft_ms:
        SLO TTFT 上限（毫秒）。
    metadata:
        附加元数据。
    """

    request_id: str
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    arrival_offset_ms: float
    workload: str
    slo_ttft_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenTrace:
    """Frozen trace — 一组固定的实验请求序列。

    Attributes
    ----------
    name:
        Trace 名称（如 "chat_sharegpt_c32"）。
    entries:
        按 arrival_offset_ms 排序的请求列表。
    content_hash:
        Trace 内容的 SHA-256 哈希（跨框架一致性校验用）。
    """

    name: str
    entries: tuple[TraceEntry, ...]
    content_hash: str

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    def get_workload(self) -> str:
        """返回 trace 中的工作负载类型（假设单一类型）。"""
        if not self.entries:
            return "unknown"
        return self.entries[0].workload


def load_trace(path: str | Path) -> FrozenTrace:
    """从 JSON Lines 文件加载 frozen trace。

    Parameters
    ----------
    path:
        Trace 文件路径（JSONL 格式）。

    Returns
    -------
    FrozenTrace
        已加载并验证的 trace。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    ValueError
        格式不正确。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {p}")

    raw_bytes = p.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    entries: list[TraceEntry] = []
    for line_num, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

        entry = TraceEntry(
            request_id=str(obj["request_id"]),
            prompt_tokens=tuple(obj["prompt_tokens"]),
            max_new_tokens=int(obj["max_new_tokens"]),
            arrival_offset_ms=float(obj["arrival_offset_ms"]),
            workload=str(obj.get("workload", "unknown")),
            slo_ttft_ms=float(obj.get("slo_ttft_ms", 500.0)),
            metadata=obj.get("metadata", {}),
        )
        entries.append(entry)

    if not entries:
        raise ValueError(f"Empty trace file: {p}")

    # 按到达时间排序
    entries.sort(key=lambda e: e.arrival_offset_ms)

    trace_name = p.stem
    trace = FrozenTrace(
        name=trace_name,
        entries=tuple(entries),
        content_hash=content_hash,
    )

    logger.info(
        "Loaded trace %s: %d entries, hash=%s",
        trace_name,
        len(entries),
        content_hash[:12],
    )
    return trace
