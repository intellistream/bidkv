"""Workload trace loader — 加载和冻结 reproducible 工作负载。

支持三种工作负载：
- Chat：ShareGPT 数据集（多轮对话，prompt 长度变化大）
- Summarization：CNN/DailyMail（长输入短输出）
- QA：LongBench（长上下文问答）

冻结原则：每种工作负载在首次加载时序列化为 JSON lines，
后续实验直接读取冻结 trace，确保跨运行可复现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestTrace:
    """单条请求 trace — 冻结的输入参数。

    Attributes
    ----------
    request_id:
        唯一请求标识。
    prompt:
        输入 prompt 文本。
    max_tokens:
        最大生成 token 数。
    expected_output:
        参考输出文本（用于质量评估）。
    metadata:
        额外元数据（数据集名称、原始 ID 等）。
    """

    request_id: str
    prompt: str
    max_tokens: int
    arrival_time_ms: float = 0.0
    expected_output: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadTrace:
    """工作负载 trace — 一组冻结的请求序列。

    Attributes
    ----------
    workload_name:
        工作负载名称（chat/summarization/qa）。
    requests:
        请求列表（按到达顺序排列）。
    dataset_source:
        数据集来源描述。
    frozen_at:
        冻结时间戳（ISO 格式）。
    """

    workload_name: str
    requests: list[RequestTrace]
    request_rate: float = 0.0
    dataset_source: str = ""
    frozen_at: str = ""
    seed: int = 42

    @property
    def num_requests(self) -> int:
        return len(self.requests)


def save_trace(trace: WorkloadTrace, path: Path) -> None:
    """将工作负载 trace 序列化为 JSON 文件。

    Parameters
    ----------
    trace:
        要保存的工作负载 trace。
    path:
        输出文件路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "workload_name": trace.workload_name,
        "request_rate": trace.request_rate,
        "seed": trace.seed,
        "dataset_source": trace.dataset_source,
        "frozen_at": trace.frozen_at,
        "num_requests": trace.num_requests,
        "requests": [asdict(r) for r in trace.requests],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Saved trace: %s (%d requests) -> %s",
        trace.workload_name,
        len(trace.requests),
        path,
    )


def load_trace(path: Path) -> WorkloadTrace:
    """从 JSON 文件加载工作负载 trace。

    Parameters
    ----------
    path:
        trace 文件路径。

    Returns
    -------
    WorkloadTrace
        加载的工作负载 trace。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    """
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    requests = [
        RequestTrace(
            request_id=r["request_id"],
            prompt=r["prompt"],
            max_tokens=r["max_tokens"],
            arrival_time_ms=r.get("arrival_time_ms", 0.0),
            expected_output=r.get("expected_output", ""),
            metadata=r.get("metadata", {}),
        )
        for r in data["requests"]
    ]
    return WorkloadTrace(
        workload_name=data["workload_name"],
        requests=requests,
        request_rate=data.get("request_rate", 0.0),
        dataset_source=data.get("dataset_source", ""),
        frozen_at=data.get("frozen_at", ""),
        seed=data.get("seed", 42),
    )


def load_dataset_sharegpt(path: Path, *, max_samples: int = 200) -> list[RequestTrace]:
    """从 ShareGPT JSON 数据集加载 chat traces。

    ShareGPT 格式：每条记录包含多轮对话，取第一轮 human 输入作为 prompt，
    第一轮 gpt 输出作为 expected_output。

    Parameters
    ----------
    path:
        ShareGPT JSON 文件路径。
    max_samples:
        最大样本数。
    """
    if not path.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    traces: list[RequestTrace] = []

    for idx, item in enumerate(raw):
        if idx >= max_samples:
            break
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue
        human_turn = conversations[0]
        gpt_turn = conversations[1]
        if human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt":
            continue
        traces.append(
            RequestTrace(
                request_id=f"chat-{idx:04d}",
                prompt=human_turn["value"],
                max_tokens=min(512, max(64, len(gpt_turn["value"].split()) * 2)),
                expected_output=gpt_turn["value"],
                metadata={"dataset": "sharegpt", "original_id": item.get("id", str(idx))},
            )
        )

    logger.info("Loaded %d chat traces from ShareGPT", len(traces))
    return traces


def load_dataset_cnn_dm(path: Path, *, max_samples: int = 200) -> list[RequestTrace]:
    """从 CNN/DailyMail JSONL 数据集加载 summarization traces。

    每行格式：{"article": "...", "highlights": "..."}

    Parameters
    ----------
    path:
        CNN/DM JSONL 文件路径。
    max_samples:
        最大样本数。
    """
    if not path.exists():
        raise FileNotFoundError(f"CNN/DM dataset not found: {path}")

    traces: list[RequestTrace] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            item = json.loads(line)
            article = item.get("article", "")
            highlights = item.get("highlights", "")
            if not article:
                continue
            prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
            traces.append(
                RequestTrace(
                    request_id=f"summ-{idx:04d}",
                    prompt=prompt,
                    max_tokens=128,
                    expected_output=highlights,
                    metadata={"dataset": "cnn_dm", "original_idx": str(idx)},
                )
            )

    logger.info("Loaded %d summarization traces from CNN/DM", len(traces))
    return traces


def load_dataset_longbench(path: Path, *, max_samples: int = 200) -> list[RequestTrace]:
    """从 LongBench JSONL 数据集加载 QA traces。

    每行格式：{"input": "...", "context": "...", "answers": [...]}

    Parameters
    ----------
    path:
        LongBench JSONL 文件路径。
    max_samples:
        最大样本数。
    """
    if not path.exists():
        raise FileNotFoundError(f"LongBench dataset not found: {path}")

    traces: list[RequestTrace] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            item = json.loads(line)
            context = item.get("context", "")
            question = item.get("input", "")
            answers = item.get("answers", [])
            if not context or not question:
                continue
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            expected = answers[0] if answers else ""
            traces.append(
                RequestTrace(
                    request_id=f"qa-{idx:04d}",
                    prompt=prompt,
                    max_tokens=256,
                    expected_output=expected,
                    metadata={"dataset": "longbench", "original_idx": str(idx)},
                )
            )

    logger.info("Loaded %d QA traces from LongBench", len(traces))
    return traces
