"""Candidate-universe consistency 审计日志。

**v7.1 within-platform consistency**：每次 pressure event 记录
candidate list hash + KV snapshot hash，确保同一平台内所有 baseline
在同一 pressure event 使用同一候选池。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PressureEventAudit:
    """单次 pressure event 的审计记录。

    Attributes
    ----------
    event_id:
        Pressure event 序号。
    timestamp_ms:
        事件发生时间戳（单调毫秒）。
    strategy_name:
        当前使用的策略名称。
    kv_used:
        KV 使用量（token 数）。
    kv_total:
        KV 总容量（token 数）。
    candidate_list_hash:
        候选请求列表的哈希（request_ids + current_tokens 决定）。
    kv_snapshot_hash:
        KV 快照的哈希（per-request token 分配情况）。
    tokens_needed:
        需要释放的 token 数。
    tokens_freed:
        实际释放的 token 数。
    actions_count:
        执行的操作数（compress / evict）。
    """

    event_id: int
    timestamp_ms: float
    strategy_name: str
    kv_used: int
    kv_total: int
    candidate_list_hash: str
    kv_snapshot_hash: str
    tokens_needed: int
    tokens_freed: int
    actions_count: int


class AuditLogger:
    """Candidate-universe consistency 审计日志记录器。

    每个实验 run 创建一个 AuditLogger，记录所有 pressure events 的
    候选池和 KV 快照信息。用于事后验证 within-platform consistency。
    """

    def __init__(self, framework: str, strategy: str, run_id: str) -> None:
        self._framework = framework
        self._strategy = strategy
        self._run_id = run_id
        self._events: list[PressureEventAudit] = []
        self._event_counter: int = 0

    @property
    def events(self) -> list[PressureEventAudit]:
        return list(self._events)

    @property
    def event_count(self) -> int:
        return self._event_counter

    def record_pressure_event(
        self,
        *,
        timestamp_ms: float,
        strategy_name: str,
        kv_used: int,
        kv_total: int,
        candidate_request_ids: list[str],
        candidate_token_counts: list[int],
        tokens_needed: int,
        tokens_freed: int,
        actions_count: int,
    ) -> PressureEventAudit:
        """记录一次 pressure event。

        Parameters
        ----------
        timestamp_ms:
            事件时间戳。
        strategy_name:
            策略名称。
        kv_used, kv_total:
            KV 使用统计。
        candidate_request_ids:
            候选请求 ID 列表（按固定顺序）。
        candidate_token_counts:
            对应候选请求的 token 数。
        tokens_needed:
            需要释放的 token 数。
        tokens_freed:
            实际释放的 token 数。
        actions_count:
            执行的压缩/驱逐操作数。
        """
        self._event_counter += 1

        # 计算 candidate list hash
        candidate_data = list(zip(candidate_request_ids, candidate_token_counts, strict=False))
        candidate_data.sort(key=lambda x: x[0])  # 按 request_id 排序确保确定性
        candidate_str = json.dumps(candidate_data, sort_keys=True)
        candidate_hash = hashlib.sha256(candidate_str.encode()).hexdigest()[:16]

        # 计算 KV snapshot hash
        kv_snapshot = {
            "used": kv_used,
            "total": kv_total,
            "per_request": dict(zip(candidate_request_ids, candidate_token_counts, strict=False)),
        }
        kv_str = json.dumps(kv_snapshot, sort_keys=True)
        kv_hash = hashlib.sha256(kv_str.encode()).hexdigest()[:16]

        audit = PressureEventAudit(
            event_id=self._event_counter,
            timestamp_ms=timestamp_ms,
            strategy_name=strategy_name,
            kv_used=kv_used,
            kv_total=kv_total,
            candidate_list_hash=candidate_hash,
            kv_snapshot_hash=kv_hash,
            tokens_needed=tokens_needed,
            tokens_freed=tokens_freed,
            actions_count=actions_count,
        )
        self._events.append(audit)
        return audit

    def save(self, output_dir: str | Path) -> Path:
        """将审计日志保存为 JSONL 文件。

        Returns
        -------
        Path
            保存的文件路径。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filename = f"audit_{self._framework}_{self._strategy}_{self._run_id}.jsonl"
        filepath = out / filename

        with filepath.open("w") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event)) + "\n")

        logger.info("Audit log saved: %s (%d events)", filepath, len(self._events))
        return filepath
