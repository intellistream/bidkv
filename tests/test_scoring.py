"""bidkv.scoring 单元测试。

覆盖：
- ScoringStrategy Protocol 合规性
- H2OScoring 策略
- AttentionWeightScoring 策略
- UniformScoring 策略
- RandomScoring 策略
- generate_bids() 返回合法 CompressionBid
- H2O vs AttentionWeight Spearman rank correlation
"""

from __future__ import annotations

import math

import pytest

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.attention import AttentionWeightScoring
from bidkv.scoring.base import ScoringStrategy
from bidkv.scoring.h2o import H2OScoring
from bidkv.scoring.random_score import RandomScoring
from bidkv.scoring.uniform import UniformScoring

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_IDS = list(range(100))
DEFAULT_COMPRESSION_LEVELS = [0.2, 0.4, 0.6, 0.8]


def _validate_scores(scores: list[float], n: int) -> None:
    """验证 scores 列表的基本属性。"""
    assert len(scores) == n, f"Expected {n} scores, got {len(scores)}"
    for i, s in enumerate(scores):
        assert 0.0 <= s <= 1.0, f"Score at index {i} out of range: {s}"


def _validate_bids(bids: list[CompressionBid], request_id: str) -> None:
    """验证 bids 列表的合法性。"""
    for bid in bids:
        assert isinstance(bid, CompressionBid)
        assert bid.request_id == request_id
        assert bid.tokens_freed > 0
        assert 0.0 <= bid.quality_delta <= 1.0
        assert bid.compress_latency_ms >= 0.0
        assert 0.0 <= bid.confidence <= 1.0
        assert bid.bid_id
        assert bid.algorithm_id
        # 三层体系：metadata 应为 dict
        assert isinstance(bid.metadata, dict)


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """计算 Spearman rank correlation coefficient。

    使用手动实现，不依赖 scipy。
    """
    n = len(x)
    assert n == len(y), "Lists must have the same length"
    if n < 2:
        return 1.0

    def _ranks(values: list[float]) -> list[float]:
        """计算排名（处理并列情况使用平均排名）。"""
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# ===========================================================================
# ScoringStrategy Protocol 合规性
# ===========================================================================


class TestScoringStrategyProtocol:
    """测试所有策略是否满足 ScoringStrategy Protocol。"""

    def test_h2o_is_scoring_strategy(self) -> None:
        scorer = H2OScoring()
        assert isinstance(scorer, ScoringStrategy)

    def test_attention_is_scoring_strategy(self) -> None:
        scorer = AttentionWeightScoring()
        assert isinstance(scorer, ScoringStrategy)

    def test_uniform_is_scoring_strategy(self) -> None:
        scorer = UniformScoring()
        assert isinstance(scorer, ScoringStrategy)

    def test_random_is_scoring_strategy(self) -> None:
        scorer = RandomScoring(seed=42)
        assert isinstance(scorer, ScoringStrategy)


# ===========================================================================
# H2OScoring 测试
# ===========================================================================


class TestH2OScoring:
    """H2O Heavy Hitter Oracle 评分策略测试。"""

    def test_init_defaults(self) -> None:
        scorer = H2OScoring()
        assert scorer.heavy_ratio == 0.2
        assert scorer.recent_ratio == 0.2
        assert scorer.decode_steps == 0

    def test_init_custom_params(self) -> None:
        scorer = H2OScoring(heavy_ratio=0.3, recent_ratio=0.1)
        assert scorer.heavy_ratio == 0.3
        assert scorer.recent_ratio == 0.1

    def test_invalid_heavy_ratio(self) -> None:
        with pytest.raises(ValueError, match="heavy_ratio"):
            H2OScoring(heavy_ratio=1.5)

    def test_invalid_recent_ratio(self) -> None:
        with pytest.raises(ValueError, match="recent_ratio"):
            H2OScoring(recent_ratio=-0.1)

    def test_heavy_plus_recent_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="heavy_ratio \\+ recent_ratio"):
            H2OScoring(heavy_ratio=0.6, recent_ratio=0.5)

    def test_score_empty(self) -> None:
        scorer = H2OScoring()
        assert scorer.score([]) == []

    def test_score_positional_heuristic(self) -> None:
        """无 decode 数据时使用位置启发式。"""
        scorer = H2OScoring()
        scores = scorer.score(DEFAULT_TOKEN_IDS)
        _validate_scores(scores, len(DEFAULT_TOKEN_IDS))
        # 第一个 token（attention sink）应比中间 token 更重要
        assert scores[0] > scores[len(scores) // 2]

    def test_score_with_cumulative_attention(self) -> None:
        """有 decode 数据时使用累积注意力评分。"""
        scorer = H2OScoring()
        n = 20
        token_ids = list(range(n))

        # 模拟 3 个 decode step：token 5 和 15 被频繁 attend to
        for _ in range(3):
            pattern = [0.01] * n
            pattern[5] = 0.5  # heavy hitter
            pattern[15] = 0.3  # heavy hitter
            pattern[-1] = 0.2  # recent
            scorer.update_from_decode_step(pattern)

        assert scorer.decode_steps == 3
        scores = scorer.score(token_ids)
        _validate_scores(scores, n)
        # Token 5 应该有高分（heavy hitter）
        assert scores[5] > 0.5

    def test_score_via_context(self) -> None:
        """通过 context 传递 attention_pattern。"""
        scorer = H2OScoring()
        n = 10
        pattern = [0.1] * n
        pattern[3] = 0.9
        scores = scorer.score(list(range(n)), attention_pattern=pattern)
        _validate_scores(scores, n)
        assert scorer.decode_steps == 1

    def test_generate_bids(self) -> None:
        scorer = H2OScoring()
        bids = scorer.generate_bids("req-1", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        assert len(bids) == len(DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-1")
        # bid_id 格式检查
        for i, bid in enumerate(bids):
            assert bid.bid_id == f"req-1:bid:{i}"
            assert bid.algorithm_id == "h2o"
            assert "scoring_method" in bid.metadata
            assert bid.metadata["scoring_method"] == "h2o_cumulative_attention"

    def test_generate_bids_empty(self) -> None:
        scorer = H2OScoring()
        bids = scorer.generate_bids("req-1", [], DEFAULT_COMPRESSION_LEVELS)
        assert bids == []

    def test_generate_bids_with_decode_data(self) -> None:
        """有 decode 数据时，confidence 应更高。"""
        scorer = H2OScoring()
        n = 50
        token_ids = list(range(n))
        # 添加 decode 数据
        for _ in range(5):
            scorer.update_from_decode_step([0.1] * n)

        bids = scorer.generate_bids("req-1", token_ids, [0.3])
        assert len(bids) == 1
        assert bids[0].confidence > 0.3  # 有 decode 数据时 confidence 应 > 0.3

    def test_reset(self) -> None:
        scorer = H2OScoring()
        scorer.update_from_decode_step([0.1, 0.2, 0.3])
        assert scorer.decode_steps == 1
        scorer.reset()
        assert scorer.decode_steps == 0

    def test_tokens_freed_monotonically_increases(self) -> None:
        """更高压缩级别应释放更多 token。"""
        scorer = H2OScoring()
        bids = scorer.generate_bids("req-1", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        freed_values = [b.tokens_freed for b in bids]
        for i in range(len(freed_values) - 1):
            assert freed_values[i] <= freed_values[i + 1]


# ===========================================================================
# AttentionWeightScoring 测试
# ===========================================================================


class TestAttentionWeightScoring:
    """Full Attention Weight Aggregate 评分策略测试。"""

    def test_init_defaults(self) -> None:
        scorer = AttentionWeightScoring()
        assert scorer.aggregation == "mean"

    def test_invalid_aggregation(self) -> None:
        with pytest.raises(ValueError, match="aggregation"):
            AttentionWeightScoring(aggregation="invalid")

    def test_score_empty(self) -> None:
        scorer = AttentionWeightScoring()
        assert scorer.score([]) == []

    def test_score_without_weights(self) -> None:
        """无注意力权重时返回均匀分数。"""
        scorer = AttentionWeightScoring()
        scores = scorer.score(DEFAULT_TOKEN_IDS)
        _validate_scores(scores, len(DEFAULT_TOKEN_IDS))
        # 所有分数应相等（0.5）
        assert all(s == 0.5 for s in scores)

    def test_score_with_weights_mean(self) -> None:
        """使用 mean 聚合注意力权重。"""
        scorer = AttentionWeightScoring(aggregation="mean")
        n = 10
        # 2 层 x 2 头 x 10 token
        weights = [
            [[0.1] * n, [0.2] * n],  # layer 0
            [[0.3] * n, [0.4] * n],  # layer 1
        ]
        # token 3 在所有层和头中都有高注意力
        weights[0][0][3] = 0.9
        weights[0][1][3] = 0.8
        weights[1][0][3] = 0.7
        weights[1][1][3] = 0.6

        scores = scorer.score(list(range(n)), attention_weights=weights)
        _validate_scores(scores, n)
        # Token 3 应该有最高分
        assert scores[3] == max(scores)

    def test_score_with_weights_max(self) -> None:
        """使用 max 聚合注意力权重。"""
        scorer = AttentionWeightScoring(aggregation="max")
        n = 5
        weights = [
            [[0.1, 0.2, 0.9, 0.1, 0.1]],
        ]
        scores = scorer.score(list(range(n)), attention_weights=weights)
        _validate_scores(scores, n)
        assert scores[2] == 1.0  # max 归一化后为 1.0

    def test_score_with_weights_last_layer(self) -> None:
        """使用 last_layer 聚合。"""
        scorer = AttentionWeightScoring(aggregation="last_layer")
        n = 5
        weights = [
            [[0.1, 0.1, 0.1, 0.1, 0.1]],  # layer 0 — 被忽略
            [[0.1, 0.1, 0.1, 0.9, 0.1]],  # layer 1（last）
        ]
        scores = scorer.score(list(range(n)), attention_weights=weights)
        _validate_scores(scores, n)
        assert scores[3] == 1.0

    def test_generate_bids(self) -> None:
        scorer = AttentionWeightScoring()
        bids = scorer.generate_bids("req-2", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        assert len(bids) == len(DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-2")
        for bid in bids:
            assert bid.algorithm_id == "attention_weight"

    def test_generate_bids_with_real_weights(self) -> None:
        """有真实注意力权重时 confidence 应为 0.9。"""
        scorer = AttentionWeightScoring()
        n = 20
        weights = [[[0.1] * n, [0.2] * n]]
        bids = scorer.generate_bids(
            "req-2",
            list(range(n)),
            [0.3],
            attention_weights=weights,
        )
        assert len(bids) == 1
        assert bids[0].confidence == 0.9

    def test_reset(self) -> None:
        scorer = AttentionWeightScoring()
        scorer.update_attention_weights([[[0.1, 0.2, 0.3]]])
        scorer.reset()
        scores = scorer.score([1, 2, 3])
        assert all(s == 0.5 for s in scores)


# ===========================================================================
# UniformScoring 测试
# ===========================================================================


class TestUniformScoring:
    """Uniform scoring 基线策略测试。"""

    def test_init_defaults(self) -> None:
        scorer = UniformScoring()
        assert scorer.uniform_score == 0.5

    def test_init_custom(self) -> None:
        scorer = UniformScoring(uniform_score=0.3)
        assert scorer.uniform_score == 0.3

    def test_invalid_score(self) -> None:
        with pytest.raises(ValueError, match="uniform_score"):
            UniformScoring(uniform_score=1.5)

    def test_score_empty(self) -> None:
        scorer = UniformScoring()
        assert scorer.score([]) == []

    def test_score(self) -> None:
        scorer = UniformScoring(uniform_score=0.7)
        scores = scorer.score(DEFAULT_TOKEN_IDS)
        assert len(scores) == len(DEFAULT_TOKEN_IDS)
        assert all(s == 0.7 for s in scores)

    def test_generate_bids(self) -> None:
        scorer = UniformScoring()
        bids = scorer.generate_bids("req-3", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        assert len(bids) == len(DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-3")
        for bid in bids:
            assert bid.algorithm_id == "uniform"
            assert bid.confidence == 1.0


# ===========================================================================
# RandomScoring 测试
# ===========================================================================


class TestRandomScoring:
    """Random scoring 基线策略测试。"""

    def test_init_defaults(self) -> None:
        scorer = RandomScoring()
        assert scorer.seed is None

    def test_init_with_seed(self) -> None:
        scorer = RandomScoring(seed=42)
        assert scorer.seed == 42

    def test_score_empty(self) -> None:
        scorer = RandomScoring(seed=42)
        assert scorer.score([]) == []

    def test_score_range(self) -> None:
        scorer = RandomScoring(seed=42)
        scores = scorer.score(DEFAULT_TOKEN_IDS)
        _validate_scores(scores, len(DEFAULT_TOKEN_IDS))

    def test_score_reproducible(self) -> None:
        """相同 seed 应产生相同结果。"""
        scorer1 = RandomScoring(seed=42)
        scorer2 = RandomScoring(seed=42)
        scores1 = scorer1.score(DEFAULT_TOKEN_IDS)
        scores2 = scorer2.score(DEFAULT_TOKEN_IDS)
        assert scores1 == scores2

    def test_score_varies_without_seed(self) -> None:
        """不同 seed 应产生不同结果。"""
        scorer1 = RandomScoring(seed=42)
        scorer2 = RandomScoring(seed=99)
        scores1 = scorer1.score(DEFAULT_TOKEN_IDS)
        scores2 = scorer2.score(DEFAULT_TOKEN_IDS)
        assert scores1 != scores2

    def test_generate_bids(self) -> None:
        scorer = RandomScoring(seed=42)
        bids = scorer.generate_bids("req-4", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        assert len(bids) == len(DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-4")
        for bid in bids:
            assert bid.algorithm_id == "random"
            assert bid.confidence == 0.0  # 随机评分无置信


# ===========================================================================
# H2O vs AttentionWeight Spearman rank correlation 测试
# ===========================================================================


class TestScoringCorrelation:
    """验证 H2O proxy scoring 与 full attention weight scoring 的一致性。

    验收标准 #6：Spearman rank correlation >= 0.7（在模拟数据上）。
    """

    def test_spearman_correlation_on_simulated_data(self) -> None:
        """在模拟 attention pattern 上，H2O 与 AttentionWeight 排名相关性 >= 0.7。

        模拟设置：
        - 100 个 token
        - 生成一个"真实"注意力分布（模拟重要 token 集中在某些位置）
        - AttentionWeight 使用真实权重
        - H2O 从多个 decode step 累积同一分布的注意力
        """
        n = 100
        token_ids = list(range(n))

        # 生成模拟的"真实"注意力分布
        # 一些 token 有高注意力（heavy hitter），大部分较低
        import random

        rng = random.Random(42)
        base_attention = [rng.uniform(0.01, 0.1) for _ in range(n)]
        # 设定 heavy hitters
        heavy_positions = [5, 12, 23, 45, 67, 89]
        for pos in heavy_positions:
            base_attention[pos] = rng.uniform(0.5, 1.0)
        # Recent tokens 有中等注意力
        for pos in range(n - 10, n):
            base_attention[pos] = max(base_attention[pos], rng.uniform(0.2, 0.4))

        # AttentionWeightScoring：使用真实权重
        attn_scorer = AttentionWeightScoring(aggregation="mean")
        # 构造为 [1 layer][2 heads][n tokens]
        # 两个 head 略有不同（加噪声）
        head1 = list(base_attention)
        head2 = [a * rng.uniform(0.8, 1.2) for a in base_attention]
        attn_weights = [[head1, head2]]
        attn_scores = attn_scorer.score(token_ids, attention_weights=attn_weights)

        # H2OScoring：通过多步累积同一分布
        h2o_scorer = H2OScoring(heavy_ratio=0.2, recent_ratio=0.1)
        for _step in range(10):
            # 每步加一些噪声到真实分布上
            noisy_pattern = [a * rng.uniform(0.7, 1.3) for a in base_attention]
            h2o_scorer.update_from_decode_step(noisy_pattern)

        h2o_scores = h2o_scorer.score(token_ids)

        # 计算 Spearman rank correlation
        correlation = _spearman_rank_correlation(h2o_scores, attn_scores)

        # 验收标准：Spearman rank correlation >= 0.7
        assert correlation >= 0.7, (
            f"Spearman rank correlation between H2O and AttentionWeight is {correlation:.4f}, "
            f"expected >= 0.7"
        )

    def test_h2o_better_than_random(self) -> None:
        """H2O 与 AttentionWeight 的相关性应显著高于 Random。"""
        n = 50
        token_ids = list(range(n))

        import random

        rng = random.Random(123)
        base_attention = [rng.uniform(0.01, 0.1) for _ in range(n)]
        heavy_positions = [3, 10, 20, 35, 45]
        for pos in heavy_positions:
            base_attention[pos] = rng.uniform(0.5, 1.0)

        # AttentionWeight reference
        attn_scorer = AttentionWeightScoring(aggregation="mean")
        attn_weights = [[list(base_attention)]]
        attn_scores = attn_scorer.score(token_ids, attention_weights=attn_weights)

        # H2O
        h2o_scorer = H2OScoring()
        for _ in range(5):
            noisy = [a * rng.uniform(0.8, 1.2) for a in base_attention]
            h2o_scorer.update_from_decode_step(noisy)
        h2o_scores = h2o_scorer.score(token_ids)

        # Random
        random_scorer = RandomScoring(seed=42)
        random_scores = random_scorer.score(token_ids)

        h2o_corr = _spearman_rank_correlation(h2o_scores, attn_scores)
        random_corr = abs(_spearman_rank_correlation(random_scores, attn_scores))

        assert h2o_corr > random_corr, (
            f"H2O correlation ({h2o_corr:.4f}) should be > Random correlation ({random_corr:.4f})"
        )

    def test_uniform_has_zero_info(self) -> None:
        """Uniform 评分不携带排序信息——与任何参考的相关性应为 0。"""
        n = 30
        token_ids = list(range(n))

        uniform_scorer = UniformScoring()
        uniform_scores = uniform_scorer.score(token_ids)

        # Uniform 的所有分数相同，rank correlation 应为 0
        ref_scores = [float(i) / n for i in range(n)]
        corr = _spearman_rank_correlation(uniform_scores, ref_scores)
        assert corr == 0.0, f"Uniform correlation should be 0, got {corr}"


# ===========================================================================
# generate_bids 默认实现测试
# ===========================================================================


class TestGenerateBidsCommon:
    """所有策略的 generate_bids 通用行为测试。"""

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (H2OScoring, {}),
            (AttentionWeightScoring, {}),
            (UniformScoring, {}),
            (RandomScoring, {"seed": 42}),
        ],
    )
    def test_bid_fields_complete(self, scorer_cls: type, kwargs: dict) -> None:
        """CompressionBid 包含三层体系所有必要字段。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-test", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-test")
        for bid in bids:
            # Layer 1
            assert bid.tokens_freed > 0
            assert 0.0 <= bid.quality_delta <= 1.0
            # Layer 2
            assert bid.compress_latency_ms >= 0.0
            assert bid.request_id == "req-test"
            # Layer 3
            assert 0.0 <= bid.confidence <= 1.0
            assert isinstance(bid.metadata, dict)
            assert "compression_level" in bid.metadata

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (H2OScoring, {}),
            (AttentionWeightScoring, {}),
            (UniformScoring, {}),
            (RandomScoring, {"seed": 42}),
        ],
    )
    def test_bid_id_format(self, scorer_cls: type, kwargs: dict) -> None:
        """bid_id 格式为 '{request_id}:bid:{level}'。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-format", DEFAULT_TOKEN_IDS, [0.3, 0.6])
        for i, bid in enumerate(bids):
            assert bid.bid_id == f"req-format:bid:{i}"

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (H2OScoring, {}),
            (AttentionWeightScoring, {}),
            (UniformScoring, {}),
            (RandomScoring, {"seed": 42}),
        ],
    )
    def test_empty_token_ids_returns_empty(self, scorer_cls: type, kwargs: dict) -> None:
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-empty", [], DEFAULT_COMPRESSION_LEVELS)
        assert bids == []

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (H2OScoring, {}),
            (AttentionWeightScoring, {}),
            (UniformScoring, {}),
            (RandomScoring, {"seed": 42}),
        ],
    )
    def test_single_token(self, scorer_cls: type, kwargs: dict) -> None:
        """单个 token 不应生成 bid（至少保留 1 个）。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-single", [42], [0.5])
        # tokens_freed 不能 >= total tokens（至少保留1）
        # 对于 1 个 token，tokens_freed = min(int(1*0.5)=0... 实际 max(1, 0) = 1，
        # 但 min(1, 1-1) = 0，所以 skip
        assert bids == []


# ===========================================================================
# build_bids 统一生成器测试
# ===========================================================================


class TestBuildBids:
    """测试 build_bids() 统一 bids 生成逻辑。"""

    def test_basic(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        bids = build_bids(
            request_id="req-bb",
            token_ids=[10, 20, 30, 40, 50],
            scores=scores,
            compression_levels=[0.4],
            algorithm_id="test",
        )
        assert len(bids) == 1
        bid = bids[0]
        assert bid.request_id == "req-bb"
        assert bid.algorithm_id == "test"
        assert bid.tokens_freed == 2  # int(5 * 0.4) = 2
        assert bid.bid_id == "req-bb:bid:0"
        # 被移除的应是分数最低的 2 个：0.1, 0.3 → δ = 0.2
        assert abs(bid.quality_delta - 0.2) < 1e-9

    def test_empty_tokens(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[],
            scores=[],
            compression_levels=[0.5],
            algorithm_id="test",
        )
        assert bids == []

    def test_scores_length_mismatch(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        with pytest.raises(ValueError, match="scores length"):
            build_bids(
                request_id="req-bb",
                token_ids=[1, 2, 3],
                scores=[0.5, 0.5],
                compression_levels=[0.3],
                algorithm_id="test",
            )

    def test_multiple_levels(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=list(range(10)),
            scores=[0.5] * 10,
            compression_levels=[0.2, 0.5, 0.8],
            algorithm_id="test",
        )
        assert len(bids) == 3
        freed = [b.tokens_freed for b in bids]
        assert freed[0] <= freed[1] <= freed[2]

    def test_confidence_fn(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.2, 0.4, 0.6, 0.8, 1.0],
            compression_levels=[0.4],
            algorithm_id="test",
            confidence_fn=lambda: 0.85,
        )
        assert bids[0].confidence == 0.85

    def test_extra_metadata(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            compression_levels=[0.4],
            algorithm_id="test",
            extra_metadata={"custom_key": "custom_value"},
        )
        assert bids[0].metadata["custom_key"] == "custom_value"
        assert "compression_level" in bids[0].metadata

    def test_default_confidence(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.5] * 5,
            compression_levels=[0.4],
            algorithm_id="test",
        )
        assert bids[0].confidence == 0.5


# ===========================================================================
# BidKVStrategy scorer-agnostic 测试
# ===========================================================================


class TestBidKVStrategyScorerAgnostic:
    """测试 BidKVStrategy 接受任意 ScoringStrategy 实现。"""

    def test_default_is_h2o(self) -> None:
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy()
        assert isinstance(strategy.scoring, H2OScoring)

    def test_inject_uniform_scorer(self) -> None:
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy(scoring=UniformScoring())
        assert isinstance(strategy.scoring, UniformScoring)

    def test_inject_random_scorer(self) -> None:
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy(scoring=RandomScoring(seed=42))
        assert isinstance(strategy.scoring, RandomScoring)

    def test_inject_attention_scorer(self) -> None:
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy(scoring=AttentionWeightScoring())
        assert isinstance(strategy.scoring, AttentionWeightScoring)

    def test_select_victims_with_uniform(self) -> None:
        """使用 UniformScoring 注入后 select_victims 仍正常工作。"""
        from bidkv.baselines.base import RequestState
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy(scoring=UniformScoring(), delta_budget=0.9)
        candidates = [
            RequestState(
                request_id="req-1",
                current_tokens=50,
                token_ids=tuple(range(50)),
                priority=1.0,
            ),
        ]
        actions = strategy.select_victims(candidates, needed_tokens=10)
        # 应该能产生 action（delta_budget 足够宽松）
        assert len(actions) >= 1
        assert actions[0].request_id == "req-1"
