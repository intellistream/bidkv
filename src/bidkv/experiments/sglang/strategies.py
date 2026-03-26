"""SGLang 实验策略配置（v2.3 冻结版本）。

定义 SGLang 可移植性验证实验使用的 3 个策略：
1. SGLang-Default (= Preempt-Evict) — SGLang 原生驱逐
2. Slack-Aware — 强无-bid 系统对手（SLO-deadline aware）
3. BidKV — 完整 bid pipeline

v2.3 Cross-platform baseline difference explanation:
SGLang 使用 3 个策略子集。bid 归因已由 vLLM 主实验承担，
SGLang 仅需验证 bid vs 强系统对手的 directional consistency。
"""

from __future__ import annotations

from dataclasses import dataclass

from bidkv.baselines import (
    BaselineRegistry,
    BaselineStrategy,
    BidKVStrategy,
    PreemptEvictStrategy,
    SlackAwareStrategy,
)
from bidkv.scoring import H2OScoring

# SGLang 实验使用的 3 个策略名称（v2.3 frozen）
SGLANG_STRATEGIES: tuple[str, ...] = (
    "sglang_default",
    "slack_aware",
    "bidkv",
)

# 策略名称映射：实验名称 → BaselineStrategy.name
_STRATEGY_NAME_MAP: dict[str, str] = {
    "sglang_default": "preempt-evict",  # SGLang-Default = Preempt-Evict
    "slack_aware": "slack-aware",
    "bidkv": "bidkv",
}


@dataclass(frozen=True)
class SGLangStrategyConfig:
    """单个 SGLang 实验策略的配置。

    Attributes
    ----------
    experiment_name:
        实验中使用的名称（如 "sglang_default"）。
    baseline_name:
        对应的 BaselineRegistry 名称（如 "preempt-evict"）。
    description:
        策略描述（用于报告）。
    is_framework_default:
        是否为框架默认策略（用于 Figure 7 标注）。
    """

    experiment_name: str
    baseline_name: str
    description: str
    is_framework_default: bool = False


def create_sglang_strategy_configs() -> list[SGLangStrategyConfig]:
    """创建 SGLang 实验的 3 个策略配置。

    Returns
    -------
    list[SGLangStrategyConfig]
        3 个策略配置。
    """
    return [
        SGLangStrategyConfig(
            experiment_name="sglang_default",
            baseline_name="preempt-evict",
            description="SGLang 原生驱逐（RadixAttention LRU）",
            is_framework_default=True,
        ),
        SGLangStrategyConfig(
            experiment_name="slack_aware",
            baseline_name="slack-aware",
            description="强无-bid 系统对手（SLO-deadline aware）",
        ),
        SGLangStrategyConfig(
            experiment_name="bidkv",
            baseline_name="bidkv",
            description="完整 bid pipeline（H2O scoring + bid + solver）",
        ),
    ]


def create_sglang_registry(
    *,
    h2o_scoring: H2OScoring | None = None,
    delta_budget: float = 0.15,
) -> BaselineRegistry:
    """创建包含 SGLang 3 个策略的 registry。

    Parameters
    ----------
    h2o_scoring:
        共享的 H2OScoring 实例。若为 None 使用默认配置。
    delta_budget:
        BidKV 的 delta_budget。

    Returns
    -------
    BaselineRegistry
        包含 3 个策略的 registry。
    """
    scoring = h2o_scoring or H2OScoring()
    registry = BaselineRegistry()

    registry.register(PreemptEvictStrategy())
    registry.register(SlackAwareStrategy())
    registry.register(BidKVStrategy(scoring=scoring, delta_budget=delta_budget))

    return registry


def get_baseline_for_experiment(
    registry: BaselineRegistry,
    experiment_name: str,
) -> BaselineStrategy:
    """根据实验名称获取对应的 baseline strategy。

    Parameters
    ----------
    registry:
        已注册的策略 registry。
    experiment_name:
        实验策略名称（如 "sglang_default"、"bidkv"）。

    Returns
    -------
    BaselineStrategy
        对应的策略实例。

    Raises
    ------
    KeyError
        策略未注册。
    """
    baseline_name = _STRATEGY_NAME_MAP.get(experiment_name)
    if baseline_name is None:
        raise KeyError(
            f"Unknown SGLang experiment strategy: {experiment_name!r}. "
            f"Available: {', '.join(SGLANG_STRATEGIES)}"
        )
    return registry.get(baseline_name)
