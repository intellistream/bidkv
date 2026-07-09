"""BaselineRegistry — baseline 策略注册表。

通过名称获取已注册的 baseline 策略实例。
"""

from __future__ import annotations

from bidkv.baselines.base import BaselineStrategy


class BaselineRegistry:
    """Baseline 策略注册表。

    所有 baseline 通过 ``register()`` 注册，通过 ``get()`` 获取。

    Example
    -------
    >>> registry = BaselineRegistry()
    >>> registry.register(PreemptEvictStrategy())
    >>> strategy = registry.get("preempt-evict")
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BaselineStrategy] = {}

    def register(self, strategy: BaselineStrategy) -> None:
        """注册一个 baseline 策略。

        Parameters
        ----------
        strategy:
            BaselineStrategy 实例。使用 ``strategy.name`` 作为 key。

        Raises
        ------
        ValueError
            如果同名策略已注册。
        """
        if strategy.name in self._strategies:
            raise ValueError(f"Strategy {strategy.name!r} already registered")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> BaselineStrategy:
        """获取已注册的策略。

        Parameters
        ----------
        name:
            策略名称。

        Returns
        -------
        BaselineStrategy
            已注册的策略实例。

        Raises
        ------
        KeyError
            策略未注册。
        """
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise KeyError(f"Strategy {name!r} not found. Available: {available}")
        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """返回所有已注册的策略名称（排序后）。"""
        return sorted(self._strategies.keys())

    @property
    def count(self) -> int:
        """已注册策略数量。"""
        return len(self._strategies)

    def create_default_registry(self) -> None:
        """注册所有内置 baseline（延迟导入以避免循环依赖）。"""
        from bidkv.baselines.bidkv_strategy import BidKVStrategy
        from bidkv.baselines.largest_first import LargestFirstStrategy
        from bidkv.baselines.preempt_evict import PreemptEvictStrategy
        from bidkv.baselines.preempt_evict_sjf import PreemptEvictSJFStrategy
        from bidkv.baselines.static_random import StaticRandomStrategy

        defaults: list[BaselineStrategy] = [
            PreemptEvictStrategy(),
            PreemptEvictSJFStrategy(),
            StaticRandomStrategy(),
            LargestFirstStrategy(),
            BidKVStrategy(),
        ]
        for strategy in defaults:
            if strategy.name not in self._strategies:
                self._strategies[strategy.name] = strategy
