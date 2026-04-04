# SGLang Adapter 重构：Token-Level → Request-Level 调度（对齐 vLLM Mode A）

## 任务目标

将 SGLang adapter 从 **token-level 部分释放** 语义重构为 **request-level 调度** 语义，
使其与 vLLM Mode A 完全对称。BidKV 控制"谁被驱逐"，SGLang 原生机制执行驱逐。

## 背景

### 当前问题

SGLang adapter 当前实现的是 token-level 部分释放：
- `try_compress()` → `execute_compression()` → `radix_hook.free_kv_positions()`
- 这比 SGLang 原生行为更激进（SGLang native 是 `RadixCache.evict()` = 请求级 LRU）
- 与 vLLM Mode A 的 request-level 调度语义不一致
- Smoke test (#052) 中 KV 压力从未触发，此路径从未实际执行

### 目标架构（对称 vLLM Mode A）

BidKV 在 SGLang 上的角色是 **请求调度插件**：
- 控制 **WHO** gets evicted（通过 `strategy.select_victims()` 排序）
- 执行机制是 **SGLang 原生** `RadixCache.evict()`（request-level LRU 驱逐）
- 不做 token-level 部分释放

### vLLM scheduler_hook.py 参考模式

vLLM 的 `scheduler_hook.py`（900行）已实现 Mode A 调度，是本次重构的参考模板：

```
_patched_schedule() 流程：
1. _sync_request_tracking() — 同步 running 请求
2. _track_waiting_arrival() — 记录 waiting 到达时间
3. _reorder_waiting_for_admission() — 策略决定 admission 排序
4. _refresh_priority_cache() — 每 3s 调用 strategy.select_victims() 缓存优先级
5. _proactive_srpt() — KV > 80% 时 SRPT 主动 preempt
6. _reorder_running_for_preemption() — 按缓存优先级重排 running
7. orig() — 调用原始 schedule()
```

策略分化表（3 策略）：

| 层面              | sglang_default (preempt-evict) | slack_aware       | bidkv                     |
| ----------------- | ------------------------------ | ----------------- | ------------------------- |
| Waiting 排序      | FCFS (无排序)                  | EDF (到达序)       | SJF (prompt_tokens)       |
| Running 排序      | LIFO (无排序)                  | cached prio       | cached prio               |
| select_victims()  | N/A                            | slack-based       | U = r/(δ+ε)              |
| SRPT 主动驱逐     | ❌                             | ❌                | ✅                        |
| Proactive preempt | ❌                             | ✅                | ✅                        |

## 修改范围

### 1. `src/bidkv/adapters/sglang/scheduler_hook.py` — 核心重写

当前（token-level）：
- Hook `get_next_batch_to_run()` → 调 `adapter.try_compress()`
- Hook `RadixCache.evict()` → 调 `adapter.try_compress()`

目标（request-level，对称 vLLM）：
- Hook `get_next_batch_to_run()` → request-level 调度：
  1. 同步 request tracking
  2. 记录 waiting 到达时间
  3. 重排 waiting queue（策略决定）
  4. 刷新 preemption 优先级缓存（调 `strategy.select_victims()`）
  5. Proactive SRPT（仅 bidkv）
  6. 重排 running 列表
  7. Proactive preempt（KV > 90%，跳过 preempt-evict）
  8. 调用原始 `get_next_batch_to_run()`

**关键差异**：SGLang 的调度模型与 vLLM 不同：
- SGLang 使用 `get_next_batch_to_run()` 选批，没有 vLLM 那样的 `schedule()` 分配循环
- SGLang 的 waiting/running 队列可能通过不同属性访问（`waiting_queue`、`running_batch` 等）
- SGLang 的 preemption 可能通过 `RadixCache.evict()` 或类似机制
- **你需要详细查看 SGLang Scheduler 的源码来确定正确的 hook 点和队列访问方式**

不确定 SGLang Scheduler 内部结构时，在代码中用 `getattr()` 安全地探测属性名。

### 2. `src/bidkv/adapters/sglang/adapter.py` — 重构 compression 路径

需要标记为 DEPRECATED 的方法：
- `execute_compression()`: **DEPRECATED (Mode B)**
- `try_compress()`: **DEPRECATED** — 压力检测和压缩不再由 adapter 驱动
- `_try_compress_baseline()`: **DEPRECATED**
- `_execute_baseline_actions()`: **DEPRECATED**
- `_execute_acceptance()`: **DEPRECATED (Mode B)**

保留的方法：
- `_refresh_bids()`: 保留（用于 `_refresh_priority_cache` 中的 bidkv 策略）

需要新增的属性（参考 vLLM adapter）：
- `_cached_preempt_priority: dict[str, float]` — 缓存的 preemption 优先级
- `_last_priority_refresh: float` — 上次刷新时间
- `_request_arrival_ms: dict[str, float]` — 请求到达时间

**注意**：adapter 的 docstring 也需要从 "token-level compression" 更新为 "request-level scheduling"。

### 3. `src/bidkv/adapters/sglang/radix_hook.py` — DEPRECATED

给整个模块添加 DEPRECATED 标记（同 vLLM 的 `truncation_hook.py`）：
```python
"""[DEPRECATED — Mode B] RadixAttention token-level KV 释放。

本模块在 Mode A（request-level 调度）中为死代码。
保留用于潜在的 Mode B 扩展（issue #054）。
"""
```

### 4. `src/bidkv/experiments/sglang/serve_entry.py` — 所有策略安装 hook

当前：`sglang_default` 不安装 BidKV hook。
目标：**所有策略都安装 hook**（包括 sglang_default），确保公平对比。

参考 vLLM 的 `plugin.py`：
```python
# ALL strategies install hooks for fair comparison.
# preempt-evict hooks do FCFS (no reorder) — identical to vanilla SGLang
# scheduling, but with the same infrastructure overhead for fairness.
```

### 5. `tests/test_sglang_adapter.py` — 更新测试

- 更新测试以反映 request-level 语义
- `execute_compression()` 相关测试标记 DEPRECATED 或改为测试新的调度方法
- 新增测试：priority cache 刷新、waiting/running 重排、proactive preempt

### 6. `src/bidkv/adapters/sglang/h2o_hook.py` — 保持不变

H2O scoring 更新逻辑不受影响。

## 约束

### 绝对禁止

- ❌ 修改 `baselines/*.py` 中的任何策略代码
- ❌ 修改 `adapters/vllm/` 下的任何文件
- ❌ 修改 `experiments/vllm/` 下的任何文件
- ❌ 添加任何外部依赖（零依赖原则）
- ❌ 修改已冻结文档（`docs/experiment_protocol.md`、`docs/baseline-specs.md`、`results/claim_freeze_early.md`）
- ❌ 创建 `.venv` / `venv`
- ❌ 使用 `git commit --no-verify` 或 `git push --no-verify`

### 必须遵守

- ✅ Python 3.10+，`from __future__ import annotations`
- ✅ 类型注解、Google 风格 docstring、100 字符行长
- ✅ DEPRECATED 方法保留但添加 `warnings.warn("...", DeprecationWarning, stacklevel=2)`
- ✅ 所有测试通过：`conda run -n sagellm python -m pytest tests/ -v`
- ✅ Lint 通过：`conda run -n sagellm python -m ruff check . && conda run -n sagellm python -m ruff format --check .`
- ✅ 更新 `CHANGELOG.md`
- ✅ sglang_default (preempt-evict) 安装 hook 时为完全无操作（FCFS + LIFO，纯 SGLang 默认行为）
- ✅ 3 策略的 BaselineStrategy 名称映射不变：sglang_default→preempt-evict, slack_aware→slack-aware, bidkv→bidkv

## DEPRECATED 标记模板

参考 vLLM adapter 已有的标记方式：
```python
def execute_compression(self, request_id: str, target_tokens: int) -> int:
    """[DEPRECATED — Mode B] Token-level KV 压缩。

    在 Mode A（request-level 调度）中不使用。
    保留用于潜在的 Mode B 扩展（issue #054）。
    """
    import warnings
    warnings.warn(
        "SGLangAdapter.execute_compression() is deprecated (Mode B). "
        "Mode A uses request-level scheduling via scheduler_hook.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... 原有代码保留 ...
```

## 验证清单

完成后必须确认：
1. [ ] `conda run -n sagellm python -m pytest tests/ -v` — 全部通过
2. [ ] `conda run -n sagellm python -m ruff check .` — 无 lint 错误
3. [ ] `conda run -n sagellm python -m ruff format --check .` — 格式正确
4. [ ] sglang_default 策略：hook 安装但不做任何 reorder（纯 SGLang 默认行为）
5. [ ] slack_aware 策略：EDF admission + SLO-slack preemption ordering
6. [ ] bidkv 策略：SJF admission + U-based priority + SRPT
7. [ ] `radix_hook.py` 整体标记 DEPRECATED
8. [ ] `adapter.py` 的 token-level 方法标记 DEPRECATED
9. [ ] `CHANGELOG.md` 更新
10. [ ] 不影响 vLLM 相关的任何测试

## 参考文件（只读）

阅读以下文件理解 vLLM Mode A 的实现模式：
- `src/bidkv/adapters/vllm/scheduler_hook.py` — **核心参考**（900行）
- `src/bidkv/adapters/vllm/adapter.py` — DEPRECATED 标记示例
- `src/bidkv/adapters/vllm/truncation_hook.py` — DEPRECATED 模块示例
- `src/bidkv/baselines/base.py` — BaselineStrategy ABC + RequestState
- `.github/copilot-instructions.md` — 项目约束全文
