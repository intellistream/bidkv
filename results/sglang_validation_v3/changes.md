# sglang_validation_v3 — 变更说明

## 对应代码版本

本次实验基于以下修改（相对于 v2）：

### Fix 1（上个 session）：SGLang KV stats 路径修复
- `adapters/sglang/adapter.py` `_get_token_to_kv_pool()`：SGLang ≥ 0.5.x 使用
  `scheduler.token_to_kv_pool_allocator`，旧代码查找 `scheduler.token_to_kv_pool` 始终
  返回 None → `get_kv_stats()` 返回 `(0, 0)` → proactive preempt / SRPT 从未触发。

### Fix 2（上个 session）：stale `current_tokens` 修复
- `scheduler_hook.py` `_build_running_candidates()`：`current_tokens` 改用
  `num_computed_tokens`（实时值），不再用 prompt 初始 token 列表长度。

### Fix 3（上个 session）：参数调优
- 优先级缓存刷新：3.0s → 0.5s
- `_proactive_preempt`：threshold 90% → 85%，cooldown 5s → 2s，>95% 驱逐 2 个
- `_proactive_srpt`：threshold 80% → 75%，cooldown 1.5s → 0.8s

### Fix 4（本 session）：BidKV `select_victims()` 公式重构
- `bidkv_strategy.py`：
  - `tokens_freed = output_tokens`（仅 decode 阶段 KV，而非 prompt+output）
    SGLang RadixAttention 中 prompt KV 是 prefix-cached 共享的，evict 时实际只释放
    output token 占用的 KV；旧公式高估长 prompt 请求的释放量，导致错误选出高重算代价受害者。
  - `quality_delta = recompute_norm + late_penalty + starvation_penalty`
    - `recompute_norm = max(0.5, prompt_tokens / 256)`：长 prompt → δ 增大 → U 降低 → 回避高重算代价
    - `late_penalty = completion² × 2`：接近完成的请求获得二次曲线保护
    - `starvation_penalty = num_preemptions × 0.5`（权重从 0.3 调高为 0.5）
  - fallback：`num_computed_tokens == 0` 时退回 `U = current_tokens / (1+starvation)`

### Fix 5（本 session）：`_proactive_preempt` guard
- `scheduler_hook.py`：加入 `if scored_reqs[0][0] >= 10.0: return` guard
  防止 `select_victims()` 返回空时随机驱逐非受害者（优先级 catch-all=100）。

## 实验参数

- Workload: mixed
- Rate: 5.7（冻结 trace）
- Runs: 3 per strategy
- Strategies: bidkv, preempt-evict-sjf, h2o-style
- max-total-tokens: 9600（对应 vLLM 600 blocks × 16）

## 对比基准

- v1：rate=3.8, 1 run（旧代码，KV stats 恒 0/0）
- v2：rate=5.7, 3 runs（KV stats 已修复 + 参数调优，但公式仍用 current_tokens）
- v3（本次）：rate=5.7, 3 runs（公式重构 + _proactive_preempt guard）

## v2 基准数据（3-run 均值）

| strategy | p50 TTFT | p95 TTFT | tput |
|---|---|---|---|
| bidkv | 169ms | 4930ms | 4.09rps |
| h2o-style | 149ms | 4734ms | 4.14rps |
| preempt-evict-sjf | 174ms | 4823ms | 4.12rps |
