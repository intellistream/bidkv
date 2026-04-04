# sglang_validation_v2 — 实验说明

## 对应代码版本

本次实验基于 KV stats 修复 + 参数调优版本（Fix 1-3），但 BidKV 公式仍使用旧版
`tokens_freed = current_tokens`。

### 修复内容（Fix 1-3）
- `_get_token_to_kv_pool()`：改为查找 `scheduler.token_to_kv_pool_allocator`
- `_build_running_candidates()`：`current_tokens` 改用 `num_computed_tokens`
- 优先级缓存刷新 3.0s → 0.5s
- `_proactive_preempt`：threshold 90% → 85%，cooldown 5s → 2s
- `_proactive_srpt`：threshold 80% → 75%，cooldown 1.5s → 0.8s

## 实验参数

- Workload: mixed | Rate: 5.7 | Runs: 3 per strategy
- Strategies: bidkv, preempt-evict-sjf, h2o-style
- max-total-tokens: 9600

## 3-run 均值结果

| strategy | p50 TTFT | p95 TTFT | tput |
|---|---|---|---|
| bidkv | 169ms | 4930ms | 4.09rps |
| **h2o-style** | **149ms** | **4734ms** | **4.14rps** |
| preempt-evict-sjf | 174ms | 4823ms | 4.12rps |

## 问题分析

BidKV 的 p95 是三者中最差（4930ms > h2o 4734ms > pe-sjf 4823ms）。

根因：
1. `tokens_freed = current_tokens` 包含了 prompt KV，但 SGLang RadixAttention 中
   prompt KV 是 prefix-cached 共享的，evict 时实际不释放，导致高估长 prompt 请求。
2. `quality_delta = 1 + 0.5*completion + starvation` penalty 过弱，
   BidKV 错误地优先驱逐高重算代价的长 prompt 请求。
3. `_proactive_preempt` 在 `select_victims()` 返回空时（output_tokens ≤ 2），
   随机驱逐了 catch-all priority=100 的非受害者。

→ v3 针对以上问题重写 `select_victims()` 公式并加 guard。
