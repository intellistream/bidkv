"""Largest-First baseline — 容量贪心驱逐策略。

Largest-First 按 KV 占用量（current_tokens）降序驱逐请求，
贪心地优先释放 KV 占用最大的请求以腾出空间。

历史：此策略最初命名为 h2o-style，但实际行为是 capacity-greedy
（无真实 attention 数据时退化为按 KV 大小排序），故重命名为 largest-first。
已冻结实验数据中 h2o-style 的结果等同于 largest-first。
"""

from __future__ import annotations

from typing import Any

from bidkv.baselines.base import BaselineStrategy, CompressionAction, RequestState
from bidkv.scoring import PositionalScoring


class LargestFirstStrategy(BaselineStrategy):
    """Largest-First：容量贪心驱逐，优先释放 KV 占用最大的请求。

    对每个候选请求：
    1. 使用 PositionalScoring 对 token 评分
    2. 计算可压缩 token 数（低重要度 token）
    3. 按可释放量从大到小排序，依次压缩直到满足 needed_tokens

    Parameters
    ----------
    scoring:
        PositionalScoring 实例。若为 None，使用默认配置创建。
    compressible_ratio:
        每个请求最多可压缩的 token 比例。默认 0.6
        （保留 heavy_ratio + recent_ratio = 0.4 的重要 token）。
    """

    def __init__(
        self,
        *,
        scoring: PositionalScoring | None = None,
        compressible_ratio: float = 0.6,
    ) -> None:
        if not (0.0 < compressible_ratio <= 1.0):
            raise ValueError(f"compressible_ratio must be in (0, 1], got {compressible_ratio}")
        self._scoring = scoring or PositionalScoring()
        self._compressible_ratio = compressible_ratio

    @property
    def name(self) -> str:
        return "largest-first"

    @property
    def scoring(self) -> PositionalScoring:
        """当前使用的 PositionalScoring 实例。"""
        return self._scoring

    def select_victims(
        self,
        candidates: list[RequestState],
        needed_tokens: int,
        **kwargs: Any,
    ) -> list[CompressionAction]:
        """使用 H2O scoring 选择低重要度 token 压缩。

        Parameters
        ----------
        candidates:
            候选请求列表。
        needed_tokens:
            需要释放的 token 数量。
        **kwargs:
            可选 ``scoring_states``：dict[str, PositionalScoring]，
            每个请求的独立 PositionalScoring 实例（带累积注意力）。

        Returns
        -------
        list[CompressionAction]
            压缩操作列表。
        """
        if needed_tokens <= 0 or not candidates:
            return []

        # 可选：每个请求有独立的 scoring 实例（含累积注意力数据）
        scoring_states: dict[str, PositionalScoring] = kwargs.get("scoring_states", {})

        # 为每个候选计算可压缩 token 数
        candidate_info: list[tuple[RequestState, int, float]] = []
        for req in candidates:
            if req.current_tokens <= 1:
                continue

            scorer = scoring_states.get(req.request_id, self._scoring)
            scores = scorer.score(req.token_ids) if req.token_ids else []

            # 可压缩 token = 总 token × compressible_ratio
            max_compressible = max(1, int(req.current_tokens * self._compressible_ratio))

            if scores:
                # 统计低重要度 token 数（score < 0.5）
                low_importance_count = sum(1 for s in scores if s < 0.5)
                compressible = min(max_compressible, low_importance_count)
            else:
                # 无 scoring 数据时使用最大可压缩量
                compressible = max_compressible

            if compressible > 0:
                # 平均 quality delta（score 越低的 token 被压缩，delta 越小）
                avg_delta = self._estimate_quality_delta(scores, compressible)
                candidate_info.append((req, compressible, avg_delta))

        # 按可释放量降序排序（贪心：先压缩能释放最多 token 的请求）
        candidate_info.sort(key=lambda x: x[1], reverse=True)

        actions: list[CompressionAction] = []
        freed = 0
        for req, compressible, avg_delta in candidate_info:
            if freed >= needed_tokens:
                break
            tokens_to_free = min(compressible, needed_tokens - freed)
            actions.append(
                CompressionAction(
                    request_id=req.request_id,
                    action_type="compress",
                    target_tokens=tokens_to_free,
                    metadata={
                        "strategy": "largest-first",
                        "estimated_quality_delta": avg_delta,
                        "compressible_tokens": compressible,
                    },
                )
            )
            freed += tokens_to_free

        return actions

    def _estimate_quality_delta(self, scores: list[float], tokens_to_compress: int) -> float:
        """估算压缩 token_to_compress 个低重要度 token 的质量损失。"""
        if not scores:
            return self._compressible_ratio * 0.5

        # 按重要度升序排列，取最不重要的 token
        sorted_scores = sorted(scores)
        n_compress = min(tokens_to_compress, len(sorted_scores))
        # quality delta ≈ 被压缩 token 的平均重要度（越低越好）
        if n_compress == 0:
            return 0.0
        return sum(sorted_scores[:n_compress]) / n_compress
