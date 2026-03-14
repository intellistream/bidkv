# BidKV

CompressionBid protocol layer for KV cache scheduling primitives.

## Overview

`bidkv` defines the core data structures and interface protocols for KV compression bid-based scheduling:

- **CompressionBid** — 压缩层向调度器的一次报价（tokens_freed, quality_delta）
- **BidPool** — 某时刻所有活跃 bid 的快照集合
- **BidAcceptance** — 调度器接受一批 bid 的决策结果
- **CompressionBidProvider** — Protocol 接口（structural subtyping）
- **BidKVConfig** — feature gate + kill switch

### CompressionBid 字段三层体系

| Layer | 字段 | 用途 |
|-------|------|------|
| **Layer 1 (Solver 核心)** | `tokens_freed` (r), `quality_delta` (δ) | Solver 直接使用 |
| **Layer 2 (过滤/路由)** | `request_id`, `compress_latency_ms` (t_exp) | BidPool 过滤 |
| **Layer 3 (可观测/扩展)** | `confidence`, `metadata` | instrumentation + 未来扩展 |

### Utility 函数

$$U(r, \delta) = \frac{r}{\delta + \varepsilon}, \quad \varepsilon = 10^{-3}$$

> `quality_delta (δ)` = **predicted / surrogate** quality signal（非 ground-truth）  
> `U` = **operational ranking signal**（非 ground-truth user utility）

## Install

```bash
pip install -e .

# 开发模式
pip install -e ".[dev]"
```

## Quick Start

```python
from bidkv import CompressionBid, BidPool, BidKVConfig, compute_utility, make_bid_id

# 配置（默认 OFF）
config = BidKVConfig(enabled=True)
assert config.is_active

# 创建 bid
bid = CompressionBid(
    bid_id=make_bid_id("req-001", level=0),
    request_id="req-001",
    algorithm_id="token_budget",
    tokens_freed=256,
    quality_delta=0.05,
    compress_latency_ms=2.0,
    confidence=0.9,
)

# utility（operational ranking signal）
print(f"utility = {bid.utility:.2f}")

# kill switch
config_off = BidKVConfig(enabled=True, kill_switch=True)
assert not config_off.is_active
```

## Zero Dependencies

`bidkv` 仅依赖 Python stdlib，无任何外部依赖。

## Testing

```bash
pytest -v
```

## License

Apache-2.0
