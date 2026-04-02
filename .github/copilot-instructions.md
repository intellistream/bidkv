# BidKV Copilot Instructions

## 仓库信息

| 字段           | 值                                                                   |
| -------------- | -------------------------------------------------------------------- |
| 仓库名         | bidkv                                                                |
| PyPI 包名      | `bidkv`                                                              |
| 导入命名空间   | `bidkv`                                                              |
| Python         | ≥ 3.10                                                               |
| 外部依赖       | **零** — 仅依赖 Python stdlib                                        |
| 论文           | SC 2026 投稿，deadline 2026-04-10                                    |
| 当前 Phase     | **Phase D 优化迭代** — v8 mixed 全量完成，调优 BidKV 高压竞争力    |

## 项目定位

BidKV 是一个 **framework-portable** 的 KV cache 请求调度原语。

核心问题：当 KV cache 压力超过阈值时，**谁应该被驱逐**？
vLLM 默认使用 LIFO/FCFS 和搜索回填 — BidKV 通过质量感知的驱逐排序
替代这一策略，使得 preemption 决策基于 **U = tokens_freed / (δ + ε)**
（每单位质量损失能回收多少 KV 空间）。

BidKV **不做压缩**——它只控制“谁被 preempt”，底层执行仍是
框架原生的 preempt + recompute（vLLM）或 token-level release（SGLang）。

BidKV 是**独立 Python 包**，零 sagellm 依赖，可作为 vLLM / SGLang 的纯插件使用。

## 代码架构

```
src/bidkv/
├── _version.py              # 版本唯一来源
├── config.py                # BidKVConfig（feature gate + kill switch，默认 OFF）
├── __init__.py              # 公开 API
├── protocol/                # 核心类型：CompressionBid, BidPool, BidAcceptance
│   ├── bid.py
│   ├── errors.py
│   └── provider.py
├── scoring/                 # Token 重要度评分策略
│   ├── base.py              # ScoringStrategy Protocol
│   ├── attention.py         # Attention-weight scoring
│   ├── h2o.py               # H2O decode-step scoring
│   ├── random_score.py
│   └── uniform.py
├── pool/                    # BidPoolManager
│   └── manager.py
├── pressure/                # PressureDetector（KV 压力检测）
│   ├── config.py
│   └── detector.py
├── compression/             # CompressionExecutor
│   └── base.py
├── solver/                  # GreedyBidSolver（bid 排序 + 贪心选取）
│   ├── config.py
│   ├── execution_result.py
│   └── greedy.py
├── baselines/               # 7 策略（6 baseline + BidKV）
│   ├── base.py              # BaselineStrategy ABC, RequestState, BaselineContext
│   ├── registry.py          # BaselineRegistry（名称→策略实例）
│   ├── preempt_evict.py     # 1. vLLM 原生 baseline (FCFS+LIFO)
│   ├── preempt_evict_sjf.py # 2. PE + SJF admission
│   ├── static_random.py     # 3. 随机驱逐
│   ├── h2o_style.py         # 4. attention-based heuristic
│   ├── uniform.py           # 5. 均等驱逐
│   ├── slack_aware.py       # 6. SLO-deadline aware
│   └── bidkv_strategy.py    # 7. BidKV 完整系统
├── adapters/                # 框架适配器
│   ├── base.py              # FrameworkAdapter ABC（5 层职责边界）
│   ├── vllm/                # vLLM v1 adapter
│   │   ├── adapter.py       # VLLMAdapter
│   │   ├── plugin.py        # monkey-patch EngineCore.__init__
│   │   ├── scheduler_hook.py
│   │   └── h2o_hook.py
│   └── sglang/              # SGLang adapter
│       ├── adapter.py       # SGLangAdapter
│       ├── scheduler_hook.py # monkey-patch get_next_batch_to_run()
│       ├── radix_hook.py    # token-level KV release via RadixAttention
│       └── h2o_hook.py
└── experiments/             # 实验框架
    ├── metrics.py           # 实验指标采集
    ├── common/              # 共享：trace, audit, runner, report
    ├── vllm/                # vLLM 实验：runner, serve, config, analysis, etc.
    └── sglang/              # SGLang 实验：runner, server, config, analysis, etc.
```

总量：~12,000 LOC Python，447+ tests。

## 关键设计模式

### 1. BaselineRegistry — 策略注册表
所有 7 个策略通过 `BaselineRegistry.register()` 注册，通过名称获取。
禁止在 registry 中硬编码策略类。新策略必须实现 `BaselineStrategy` ABC。

### 2. FrameworkAdapter ABC — 跨框架适配
职责：KV stats 获取 → 策略注入（谁被 preempt）→ Lifecycle 管理。
在 vLLM Mode A 中，adapter 仅做请求排序，不执行 token-level 操作。
每个被支持的 serving framework 实现一个 adapter。

注：adapter.py 中的 execute_compression() / try_compress() 等方法已标记为
**DEPRECATED (Mode B)** — 在 Mode A 实验中为死代码。

### 3. 默认 OFF + Kill Switch
`BidKVConfig(enabled=False)` 永远是默认。
`BidKVConfig(kill_switch=True)` 立即绕过所有 bid 逻辑。

### 4. 零外部依赖
`dependencies = []`。不依赖 torch / numpy / sagellm / vllm / sglang。
vLLM / SGLang 仅在 adapter 内通过 runtime import 引用。

## 实验方案（v2.3-frozen，不可修改）

实验方案已通过 reviewer 审核冻结。以下给出关键约束，供实施时参考。

### 冻结规则索引

| Rule ID                    | 说明                                                    |
| -------------------------- | ------------------------------------------------------- |
| RULE FIG6-DEFAULT          | Figure 6 默认为 surrogate budget sensitivity            |
| RULE FIG3AB-FREEZE         | Fig 3a/b 移出 Mode A 主线，Mode B / Appendix only      |
| RULE SCENARIO-SWITCH       | Scenario A/B 切换：SLO attainment Δ_avg ≥ 10% + 无反转 |
| RULE SGLANG-FALLBACK-CLAIM | SGLang request-level fallback 下 claim 降级             |
| RULE NARRATIVE-BASELINE-SPLIT | 正文 6 策略核心 + 1 策略次级                         |
| RULE RATE-FREEZE           | rate 冻结与 trace 冻结同级                              |

### 双平台分工

| 维度         | vLLM（主量化平台）                     | SGLang（portability slice）          |
| ------------ | -------------------------------------- | ------------------------------------ |
| 策略数       | 7（完整归因链）                        | 3（精简方向验证）                     |
| 矩阵         | 7 × 2 × 3 × 3 = 126 runs             | 3 × 2 × 3 × 3 = 54 runs            |
| 执行语义     | Mode A: 请求级调度 + vLLM 原生 preempt/recompute | token-level (RadixAttention) / request-level fallback |
| 产出物       | Table 1 + Fig 3/4/5/6                 | Table 2 + Fig 7                     |
| claim 角色   | Scenario A 核心主张                    | directional consistency + portability |

### vLLM 当前架构：质量感知的请求调度

BidKV 在 vLLM 上的定位是**请求调度插件**，不是压缩引擎。

它解决的核心问题：当 KV cache 空间不足时，**应该 preempt 哪个请求？**

vLLM 原生的答案是 LIFO（最后进入 running 的先被驱逐）。
BidKV 的答案是：驱逐那个“每单位质量损失能释放最多 KV 空间”的请求（U 最高的）。

**执行机制完全是 vLLM 原生的**：被选中的请求通过 `scheduler._preempt_request()`
驱逐，其 KV blocks 全部释放，重新调度时 recompute from scratch。
BidKV 不修改这个执行路径，只控制“谁被选中”。

所有 7 个策略的调度分化：

| 层面               | PE            | PE-SJF        | Slack       | Random/H2O/Uniform | BidKV                          |
| ------------------ | ------------- | ------------- | ----------- | ------------------- | ------------------------------ |
| Waiting 排序       | FCFS (无排序) | SJF           | EDF (到达序) | SJF (prompt_tokens) | SJF (prompt_tokens)            |
| Running 排序       | LIFO (无排序) | LIFO (无排序) | cached prio | cached prio         | 95% KV 门控 + avg_prompt≤500  |
| select_victims()   | N/A           | N/A           | slack-based | 各自启发式          | **U = r/(δ+ε)** 质量感知      |
| SRPT 主动驱逐      | ❌            | ❌            | ❌          | ✅ (同等估算)       | ❌ (recompute 成本过高)        |
| Proactive preempt  | ❌            | ❌            | ✅          | ✅                  | ✅                             |

BidKV 的唯一分化点：`select_victims()` 中使用完整 scoring→bid→pool→solver
管线计算 **U = tokens_freed / (quality_delta + ε)**，实现质量感知的 preemption 排序。

### Mode B 状态（已废弃）

adapter.py 中存在 Mode B 代码（`execute_compression()` → `_execute_tail_truncation()`），
`truncation_hook.py` 提供 token-level KV block 截断基础设施。
这些代码已全部标记为 **DEPRECATED (Mode B)**，在当前实验中为死代码，
保留用于潜在的 Mode B 未来扩展（issue #054）。

### SGLang 3 策略

Preempt-Evict (Default) → Slack-Aware → BidKV

方向一致性（Directional Consistency）：
- DC-1a: BidKV ≥ Preempt-Evict
- DC-1b: BidKV ≥ Slack-Aware

**SGLang 执行语义设计问题**：

SGLang 原生驱逐机制是 `RadixCache.evict()`，本质上做的是 **请求级 LRU 驱逐**
（驱逐整个 radix tree 节点），而不是在一个 running 请求中间释放部分 token。

当前 SGLang adapter 代码（`execute_compression()` → `radix_hook.free_kv_positions()`）
实现的是 **token-level 部分释放**——这比 SGLang 原生行为更激进，
且与 vLLM Mode A 的 request-level 调度语义不一致。

双平台一致性目标：SGLang 也应走 request-level 调度（控制 WHO gets evicted），
然后让 SGLang 原生 `RadixCache.evict()` 执行驱逐。这与 vLLM 完全对称。

当前状态：smoke test (#052) 中 KV 压力从未触发，token-level 路径从未实际执行。
SGLang adapter 的 token-level 代码有待重构为 request-level 语义，
或在全量实验中确认后再决定。

### Phase 路线（vLLM）

Phase 0 策略+指标冻结 → Phase 1 数据集冻结 → Phase 2 Pilot calibration →
Phase 3 Frozen trace → Phase 4 Sanity check → Phase 5 全量 126 runs →
Phase 6 消融（可选）→ Phase 7 论文映射

### 六大原则

P1 真实性优先（ShareGPT + Poisson）| P2 先 pilot 后冻结 | P3 trace 不可回改 |
P4 最小闭环优先 | P5 失败计入结果 | P6 单一变量控制

## 核心文档（已冻结，不可修改内容方向）

| 文档路径                           | 内容                                             |
| ---------------------------------- | ------------------------------------------------ |
| `docs/experiment_protocol.md`      | 主实验方案 v2.3-frozen（§1-§14，1300+ 行）       |
| `docs/vllm-route-redesign.md`      | vLLM crash 根因 + 双轨方案（10 节）              |
| `docs/sglang-portability-slice.md` | SGLang portability slice v2.3-frozen（9 节）      |
| `docs/baseline-specs.md`           | 7 baseline 策略规格                               |
| `results/claim_freeze_early.md`    | Claim freeze + Scenario A/B 规则 v2.3-frozen     |

**⚠️ 不得修改已冻结文档的实验方向、策略列表、指标定义、figure 语义。**
仅允许 typo 修正和实施备注。

## Wave 1 关键代码变更（#049 + #051，已合入）

### #049 vLLM Mode A Recompute Fallback
- `adapters/vllm/adapter.py`：`execute_compression()` 路由到 `_execute_tail_truncation()`（**已标记为 DEPRECATED Mode B**）
- `scheduler_hook.py` 仅做请求排序 + `_preempt_request()`（调度语义）
- `execute_abort()` 调用 `scheduler.abort_requests()`（**也已 DEPRECATED**）
- **已完全移除** `_free_tail_blocks()` — 不再绕过 KVCacheCoordinator

### #051 SGLang 策略列表更新 + Audit
- `experiments/sglang/config.py`：策略列表更新为 v2.3（sglang_default, slack_aware, bidkv）
- `experiments/sglang/serve_entry.py`：修复 `active=True` → `enabled=True`；新增 BaselineRegistry 路由
- `experiments/sglang/runner.py`：CLI 默认策略更新
- `experiments/sglang/collector.py`：新增 `write_audit_entry()` 公平性审计日志

## Wave 2-3 关键变更（#052 + #055，已合入）

### #052 SGLang Smoke Test
- 3 策略 × 1 workload × 1 rate × 1 run = 3 runs，零 crash
- TTFT p50 ~85ms, TPOT p50 ~32ms, SLO attainment 100%（成功请求）
- 方向一致性 DC-1a/DC-1b 初步通过（timeout-dominated regime）
- 结果保存至 `results/sglang_smoke_mode_a/`

### #055 Pilot Calibration + Trace/Rate 冻结
- Situation A：复用 90 组已有数据（formal 72 + pilot_v3 12 + pilot_v3_mixed_high 6），无需额外 pilot
- 冻结 per-workload rates：mixed `(2.0, 3.8, 5.7)`, long_context `(0.35, 0.5, 0.7)`
- `WORKLOAD_REQUEST_RATES` 冻结字典新增（vLLM + SGLang config 同步）
- `get_rates_for_workload(workload)` 方法 + `--mixed-rates` / `--long-rates` CLI 覆盖
- seed=42 formal traces（6 文件 + manifest.json），SHA-256 验证通过
- 校准报告：`results/pilot_055/calibration_report.md`

## 当前实施阶段与 Issue 追踪

Issue 文件位于 `sagellm-docs/issues/dir1-compression-scheduling-primitive/`。

### Phase C 并行化执行计划

```
─── Wave 1：代码修复（可并行）────────────────────────
#049  vLLM Mode A Recompute Fallback   │ P0  │ ✅ 完成+验收 (2026-03-19)
#051  SGLang 策略列表更新 + Audit      │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Wave 2：Smoke Test（可并行）──────────────────────
#050  vLLM Smoke Test (7 runs)          │ P0  │ ⬜  (依赖 #049 ✅)
#052  SGLang Smoke Test (3 runs)        │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Wave 3：Pilot Calibration ──────────────────────────
#055  Pilot + Trace/Rate 冻结           │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Wave 4：全量实验 ─────────────────────────────────
#053  全量 126+54 runs + Figure/Table   │ P0  │ ⬜  (依赖 #055)
─── 独立增强（不阻塞主线）────────────────────────
#054  vLLM Mode B Kernel Extension      │ P1  │ ⬜  (可任何时候开始)
─── Phase D：论文 ───────────────────────────────────
#039  论文写作冲刺                     │ P0  │ ⬜  (依赖 #053)
```

### 关键路径

```
#049 ──→ #050 ──→ #055 ──→ #053 ──→ #039
       │                     ↑
#051 ──→ #052 ─────────┘
```

## 编码规范

- Python 3.10+，`from __future__ import annotations`
- 类型注解强制
- Docstring：Google 风格
- 行长度：100 字符
- Linter：ruff（`select = ["E", "F", "W", "I", "UP", "B", "SIM", "ARG"]`）
- **零外部依赖**：所有核心代码仅 stdlib
- 测试：`pytest tests/ -v`

## 关键约束

### 绝对禁止

- ❌ 向 `dependencies` 添加任何外部包（torch / numpy / vllm / sglang / sagellm）
- ❌ 修改已冻结文档的实验方向 / 策略列表 / figure 语义
- ❌ 在 vLLM adapter 中绕过 coordinator 操作 block_pool（参见 vllm-route-redesign.md §4）
- ❌ 使用 `_free_tail_blocks()` / null-block 替换 / `remove_skipped_blocks()` 的任何变体
- ❌ 在 Mode A 下为 Figure 6 / Figure 3a/b 赋予 task-level quality 含义
- ❌ 声称 SGLang "leverages native token-level KV release"（除非 smoke test 确认）
- ❌ 声称 BidKV 通过 `max_tokens` 获得信息优势（`max_tokens` 是标准 API 参数，所有策略平等可用）
- ❌ 在 `_get_max_tokens_estimate()` 中按策略名返回不同估算值（已移除此 asymmetry）
- ❌ 创建 `.venv` / `venv`

### 必须遵守

- ✅ `BidKVConfig(enabled=False)` 始终为默认
- ✅ Kill switch `BidKVConfig(kill_switch=True)` 必须立即旁路所有 bid 逻辑
- ✅ 新 baseline 必须通过 `BaselineRegistry.register()` 注册
- ✅ 新 adapter 必须实现 `FrameworkAdapter` ABC
- ✅ 失败计入结果（OOM / timeout / crash 全记录，禁止 cherry-pick）
- ✅ Frozen trace (seed=42) 跨所有策略共享
- ✅ Rate 冻结后不可基于策略表现调整（RULE RATE-FREEZE）

## 版本管理

`_version.py` 是唯一版本来源。不在 `pyproject.toml` / `__init__.py` 硬编码版本。

## 测试

**重要**：必须使用 `python -m pytest`（而非裸 `pytest`），确保使用 conda 环境的 Python：

```bash
cd /home/cyb/bidkv
conda run -n sagellm python -m pytest tests/ -v            # 全部 447+ tests
conda run -n sagellm python -m pytest tests/test_baselines.py -v   # 7 策略 baseline tests
conda run -n sagellm python -m pytest tests/test_vllm_adapter.py   # vLLM adapter tests
conda run -n sagellm python -m pytest tests/test_sglang_adapter.py # SGLang adapter tests
conda run -n sagellm python -m ruff check . && conda run -n sagellm python -m ruff format --check .
```

## Git 约定

- `main-dev` = 活跃开发分支
- `feature/bidkv` = 当前 BidKV 特性分支
- 不使用 `--no-verify`；hooks 失败必须先修复

## 论文关键时间线

| 节点           | 日期         | 说明                                        |
| -------------- | ------------ | ------------------------------------------- |
| Gate-A         | 2026-03-14   | ✅ 7/7 PASS，357 tests                      |
| 方案冻结       | 2026-03-19   | ✅ v2.3-frozen，reviewer APPROVED            |
| Wave 1 完成    | 2026-03-19   | ✅ #049+#051 代码修复，447 tests pass        |
| Wave 2         | 03-20~22     | #050+#052 smoke test                       |
| Wave 3         | 03-22~23     | #055 pilot calibration + trace/rate 冻结    |
| Wave 4         | 03-24~28     | #053 全量 126+54 runs                       |
| Gate-B         | 2026-03-31   | vLLM Mode A 全量 + SGLang 全量 目标         |
| Phase D        | 04-01~08     | #039 论文写作冲刺                        |
| SC 2026 投稿   | 2026-04-10   | 论文截止                                     |

## 实验硬件

- GPU：NVIDIA RTX A6000 48GB · CUDA 12.5
- 模型：Llama-3.1-8B-Instruct（bf16, 16GB）
- 推理引擎：vLLM 0.17.1 (v1 架构) + SGLang (RadixAttention)
- KV 限制：`--num-gpu-blocks-override 600`（600×16=9600 tokens，**必须指定否则无 KV 压力**）
- 并发：`--max-num-seqs 32`，`--gpu-memory-utilization 0.5`

## 冻结实验环境（v8-frozen，2026-04-02）

### 服务端参数（不可修改，所有后续实验必须使用）

```
--model           /home/cyb/.cache/huggingface/hub/Llama-3.1-8B-Instruct
--gpu-memory-utilization  0.5
--num-gpu-blocks-override 600    # 600×16=9600 tokens KV
--max-num-seqs    32
--block-size      16
--max-model-len   8192
--enforce-eager
--disable-frontend-multiprocessing
--no-enable-prefix-caching
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 工作负载参数（RULE RATE-FREEZE）

| 工作负载 | Rates (req/s) | Requests/run | Runs/combo |
|----------|--------------|-------------|------------|
| mixed | 2.0, 3.8, 5.7 | 1000 | 3 |
| long_context | 0.35, 0.5, 0.7 | 500 | 3 |

### 观测指标体系

**主表指标**（论文 Table 1，5 列）：
- Goodput(500ms) — DistServe (OSDI'24)
- SLO attainment(300ms) — S³ (ISCA'24)
- TTFT p95 — 尾部延迟主指标
- TPOT p95 — Sarathi-Serve (OSDI'24)，统一使用 P95（P99 样本量不足方差过大）
- Normalized Latency — Orca (OSDI'22)

## v8 Mixed 全量结果（63 runs，2026-04-02 冻结）

### Cross-Rate Average Ranking（7 策略 × 3 rates × 3 runs）

| Rank | Strategy | Goodput | SLO300 | TTFT95 | TPOT95 | NrmLat | Wins |
|------|----------|---------|--------|--------|--------|--------|------|
| 1 | static-random | #1 | #2 | #5 | #1 | #1 | 3 |
| 2 | **bidkv** | #3 | **#1** | **#1** | #4 | #3 | **2** |
| 3 | uniform | #2 | #3 | #4 | #2 | #2 | 0 |
| 4 | h2o-style | #4 | #4 | #3 | #6 | #4 | 0 |
| 5 | preempt-evict-sjf | #5 | #5 | #2 | #7 | #5 | 0 |
| 6 | slack-aware | #6 | #6 | #6 | #3 | #6 | 0 |
| 7 | preempt-evict | #7 | #7 | #7 | #5 | #7 | 0 |

### Cross-Rate Average Values

| Strategy | Goodput | SLO300% | TTFT p95 | TPOT p95 | NormLat |
|----------|---------|---------|----------|----------|---------|
| static-random | 2.92 | 87.0 | 1091 | 86.0 | 65.3 |
| **bidkv** | **2.79** | **87.1** | **562** | **96.5** | **67.8** |
| uniform | 2.90 | 86.9 | 1079 | 86.7 | 65.4 |
| h2o-style | 2.68 | 84.4 | 587 | 100.2 | 70.4 |
| preempt-evict-sjf | 2.46 | 82.8 | 575 | 129.5 | 74.5 |
| slack-aware | 2.24 | 72.4 | 4101 | 93.3 | 76.6 |
| preempt-evict | 2.19 | 72.2 | 5384 | 98.3 | 79.2 |

### BidKV Per-Rate Performance

| Rate | Wins/5 | Top-3/5 | 最强项 | 最弱项 |
|------|--------|---------|--------|--------|
| 2.0 | **5/5** | 5/5 | 全胜 | — |
| 3.8 | 1/5 | 4/5 | TTFT #1 | TPOT #5 |
| 5.7 | 0/5 | 4/5 | — | Goodput #3, TPOT #4 |

### 核心问题：BidKV 高压竞争力不足

BidKV 在 SLO(300ms) #1 和 TTFT p95 #1，但在 Goodput、TPOT、NormLat 输给
static-random/uniform。根因：**BidKV 禁用了 SRPT，而 static-random/uniform 启用**。

**v8b SRPT 快速测试结果**：已在 `results/vllm_v8b_srpt_quick/` 测试过简单启用
SRPT（rate=5.7, 1 run）。结果：TTFT 改善 -98ms，但 SLO(-2pp)、TPOT(+3.7ms)、
NormLat(+1.3) 均变差。**简单启用 SRPT 无效**，需要更深层优化。

### 各策略机制开关（scheduler_hook.py 当前状态）

| 机制 | PE | PE-SJF | Slack | Random/H2O/Uniform | BidKV |
|------|-----|--------|-------|---------------------|-------|
| Waiting 排序 | FCFS | SJF | EDF | SJF | SJF |
| Running reorder | ❌ LIFO | ❌ LIFO | ✅ cached prio | ✅ cached prio | ✅ 95% KV 门控 |
| Proactive preempt | ❌ | ❌ | ✅ KV>90% | ✅ KV>90% | ✅ KV>90% |
| SRPT preempt | ❌ | ❌ | ❌ | ✅ KV>80% | ❌ |

BidKV 特有：U = freed / (1.0 + 0.5×completion + 0.3×num_preemptions + ε)

### 数据目录

| 路径 | 内容 | 状态 |
|------|------|------|
| `results/vllm_v8_full_validation/` | v8 全量 7×3×3=63 runs (mixed) | **当前主数据，冻结** |
| `results/vllm_v8b_srpt_quick/` | BidKV+SRPT 快速测试 (1 run) | 参考数据 |
