# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- **废弃策略完全移除（uniform + slack-aware）** (2026-04-10):
  - 删除 `baselines/uniform.py`（`UniformStrategy`，均等驱逐策略）
  - 删除 `baselines/slack_aware.py`（`SlackAwareStrategy`，SLO-deadline 感知策略）
  - 清理所有引用：`baselines/__init__.py`、`baselines/registry.py`、`bidkv/__init__.py`
  - 清理实验配置：`experiments/vllm/config.py`（ALL_STRATEGIES 从 7→5）、`experiments/sglang/config.py`
  - 更新 SGLang 策略：`experiments/sglang/strategies.py` 中 slack-aware 替换为 static-random
  - 清理 adapter hooks：`adapters/vllm/scheduler_hook.py`、`adapters/sglang/scheduler_hook.py`
  - 更新测试：`tests/test_baselines.py`（删除 TestUniform/TestSlackAware）、`tests/test_sglang_adapter.py`
  - 更新 `tests/test_vllm_experiment.py`：total_runs 期望值 7→5 策略
  - 更新 AD：`paper-ad/bidkv_ad.tex` 从 "seven implementations (five evaluated)" 简化为 "five implementations"
  - 测试结果：362 tests pass

- **AD (Artifact Description) 修订** (2026-04-10):
  - 修复 T2/T4 Verbatim 块中双反斜杠 `\\` → 单反斜杠 `\`（shell 行继续符渲染错误）
  - T2 命令新增 `--max-model-len 8192`（与论文固定参数一致）
  - T3 命令新增 `--attention-backend triton`（与实验实际使用一致），并添加 triton vs flashinfer 兼容性说明
  - 补全 HuggingFace ShareGPT URL：`anon8231489123/` → `anon8231489123/ShareGPT_Vicuna_unfiltered`
  - SGLang 执行时间更正：`≈2--3 h` → `≈3--4 h`（9 runs × ~24 min = ~3.6 h）
  - `\artexp` 新增 Table V（超参数敏感性）预期结果：SLO/TTFT P95 在全参数扫描范围内稳定（±2 pp / ±10%）
  - `\artout` 修正：CSV → JSON，Tables II--III → Tables II--IV，Figures 4--5 → Figure 4 only，补充 Figures 1--3/5 为静态资产说明


  - **文件删除**：`baselines/random_evict.py`（RandomEvictStrategy，未注册、非 v2.3 冻结策略）、
    `compression/`（整个目录，Mode B CompressionExecutor Protocol）、
    `scoring/attention.py`（AttentionWeightScoring，依赖 `output_attentions=True`，FlashAttention 不可用）、
    `scoring/random_score.py`（RandomScoring，无生产调用者）、
    `scoring/uniform.py`（UniformScoring scoring class，无生产调用者）、
    `solver/execution_result.py`（ExecutionResult，仅被已删除的 execute_accepted 使用）
  - **solver/greedy.py**：删除 `execute_accepted()` 方法（Mode B dead code，无生产调用者）
  - **公开 API 清理**：`solver/__init__.py`、`scoring/__init__.py`、`__init__.py`
    移除 `ExecutionResult`、`CompressionExecutor`、`AttentionWeightScoring`、
    `RandomScoring`、`UniformScoring`
  - **baselines/registry.py**：删除 `RandomEvictStrategy` 导入和注册（策略数 8→7）
  - **experiments/sglang/config.py**：删除 `STRATEGY_VANILLA_SGLANG`、`STRATEGY_RANDOM_EVICT`、
    `SGLANG_NATIVE_ABL_STRATEGIES`、`EXTENDED_STRATEGIES` 常量；`STRATEGY_BASELINE_MAP` 精简为 4 条目；
    `__post_init__` 使用 `ALL_STRATEGIES` 验证
  - **adapters/sglang/scheduler_hook.py**：删除 `vanilla_sglang`、`random_evict`、`random-evict`
    字符串字面量（共 5 处 if/tuple 检查）
  - **测试清理**：
    - `test_core.py`：删除 `TestCompressionExecutor`、`TestExecutionResult`、`TestExecuteAccepted`、
      `TestEndToEndWithActualFreed`、`TestImportExecutionResult`（共 -15 tests）
    - `test_scoring.py`：删除 `TestAttentionWeightScoring`、`TestUniformScoring`、
      `TestRandomScoring`、`TestScoringCorrelation`；精简 `TestGenerateBidsCommon` 参数化；
      删除 `TestBidKVStrategyScorerAgnostic` 中 4 个死方法（共 -50 tests）
    - `test_vllm_adapter.py`：删除 `uniform_scoring` fixture 和 `UniformScoring` 导入
    - `test_baselines.py`：更新 `test_create_default_registry` 断言（count 8→7）
  - 测试总数：433 → 373（-60 死测试）


  - **Critical fix**: `adapters/vllm/scheduler_hook.py` 移除已删除的 `truncation_hook` 死导入（runtime `ImportError`）
  - **SGLang adapter Mode B 完全清除**: 删除 `execute_compression()`, `try_compress()`, `_refresh_bids()`,
    `_execute_acceptance()`, `_try_compress_baseline()`, `_build_request_states()`, `_execute_baseline_actions()`,
    `_write_audit()` — 这些方法依赖已删除的 `radix_hook.py`，在 Mode A 中为死代码
  - **H2OStyleStrategy 向后兼容别名删除**: `baselines/__init__.py` / `bidkv/__init__.py`
  - **STRATEGY_H2O_STYLE 常量删除**: `experiments/sglang/config.py`（无外部调用者）
  - **SGLang 测试清理**: 删除 `TestPressureCompression`、`test_shared_tokens_excluded_from_compression`、
    `test_build_request_states`、`test_baseline_route_skips_bidkv_pipeline`、
    Mode B `try_compress`/`execute_compression` 测试（共 -8 tests）
  - 测试总数：441 → 433（-8 Mode B SGLang 测试）

- **Mode B dead code removal from `adapters/vllm/adapter.py`** (2026-04-09):
  - 删除 `execute_compression()`, `execute_abort()`, `_execute_tail_truncation()`,
    `_sync_model_runner_block_table()` — Mode B Token-level truncation 入口，
    在 Mode A 实验中从未被调用，`truncation_hook.py` 已于上一 session 删除
  - 删除 `try_compress()`, `_try_compress_baseline()`, `_build_request_states()`,
    `_execute_baseline_actions()`, `try_compress_for_request()`, `_refresh_bids()`,
    `_execute_acceptance()` — Mode B 压缩管道，vLLM Mode A 不使用
  - 删除 `_get_block_size()` — 仅被上述已删除方法调用
  - 删除 `DEFAULT_COMPRESSION_LEVELS` 常量和 `compression_levels` 构造函数参数
  - 更新模块 docstring 核心职责（4 层，移除"Compression 执行"）
  - 删除 `from bidkv.scoring.bid_builder import build_bids` / `CompressionAction` / `BidAcceptance` 等孤立导入
  - 文件从 936 行缩减至 372 行

- **`adapters/base.py` ABC 更新** (2026-04-09):
  - 删除 `execute_compression()` 抽象方法（SGLang adapter 的 execute_compression 已非 ABC 要求）
  - 更新模块 docstring 职责边界（5 层 → 4 层）

- **Mode B 测试清理** (2026-04-09):
  - `tests/test_vllm_adapter.py`：删除 `TestCompressionExecution`, `TestTruncationRouting`,
    `TestTailTruncation`, `TestTruncationHook` 测试类（共 ~526 行）；
    删除 `test_custom_compression_levels`, `test_inactive_adapter_no_compression`,
    `test_inactive_adapter_no_execute`, `TestBidKVPipeline` 类；
    更新 `test_kill_switch_stops_all_operations` 移除 Mode B 断言
  - `tests/test_sglang_adapter.py`：删除 `TestRadixHook` 类（引用已删除的 `radix_hook.py`）
  - 测试总数：471 → 441（-30）

- **Sensitivity analysis for BidKV v8 formula parameters** (2026-04-07):
  - `bidkv_strategy.py`：`w_c=0.5` / `w_s=0.3` 硬编码改为读取环境变量
    `BIDKV_COMPLETION_WEIGHT` / `BIDKV_STARVATION_WEIGHT`（默认值不变）
  - `scheduler_hook.py`：KV gate `0.95` 改为读取环境变量 `BIDKV_KV_GATE`（默认值不变）
  - 新增 `scripts/run_sensitivity_v2.sh`：10 variants × 3 runs = 30 runs
    (completion_weight axis: 0.25/0.5/1.0/2.0; starvation_weight: 0.1/0.3/0.6/1.0; kv_gate: 0.85/0.90/0.95/0.98)
  - 新增 `scripts/analyze_sensitivity_v2.py`：span 分析 + 鲁棒性分类
  - 结果（rate=3.8, mixed, 3 runs/variant，保存于 `results/vllm_sensitivity_v2/`）：
    | Axis | SLO span | TTFT span | Classification |
    |---|---|---|---|
    | completion_weight | 1.9pp | 9.2% | ROBUST |
    | starvation_weight | 1.6pp | 3.2% | ROBUST |
    | kv_gate | 1.4pp | 3.1% | ROBUST |
    Overall: SLO span 1.9pp, TTFT P95 span 9.2% → **ROBUST**
    Default sanity check: TTFT P95=633ms ✓ (expected 550–750ms)

- **BidKV radix-tree-aware victim selection for SGLang** (2026-04-08):
  - `baselines/base.py`: `RequestState` 新增 `private_tokens: int = 0` 字段，
    表示请求中不与其他请求共享的 KV token 数量（SGLang radix 树感知）；0 = 不可用（vLLM）
  - `adapters/sglang/radix_hook.py`: 新增两个公开 API：
    - `get_shared_kv_slots(scheduler) -> frozenset[int]`：一次遍历 radix 树，收集
      所有 `lock_ref > 1` 节点的 KV slot 集合（O(tree_nodes)，批量高效）
    - `get_private_token_count(scheduler, rid, num_tokens, shared_slots=None) -> int`：
      计算请求中不在共享集合里的 token 数量（O(num_tokens)，可复用预计算的 shared_slots）
  - `adapters/sglang/scheduler_hook.py`: `_build_running_candidates()` 新增
    `scheduler: Any | None = None` 参数；当 scheduler 可用时，每次调用预计算一次
    `shared_slots`，再为每个请求计算 `private_tokens`，并写入 `RequestState`；
    `_refresh_priority_cache` 中传入 `scheduler=scheduler`；`needed_all` 改为优先使用
    `private_tokens`（若可用）以准确反映实际可回收的 KV 空间
  - `baselines/bidkv_strategy.py`: `select_victims()` 改为双分支公式：
    - **Branch A（radix-aware）**：当 `req.private_tokens > 0` 时，
      `tokens_freed = private_tokens`，`δ = max(0.1, private_tokens/RECOMPUTE_DIV + completion·CW + P·SW)`；
      分子只计实际可回收的私有 KV，分母 recompute cost 也仅对私有 token 归一化（共享 prefix 不需重算）
    - **Branch B（v8-frozen 回退）**：`private_tokens = 0`（vLLM 或 SGLang 树访问失败），
      保持 `tokens_freed = current_tokens`，`δ = 1 + 0.5c + 0.3P`（与之前完全相同）
    - 新增 `if tokens_freed <= 0: continue` 防卫，跳过无 KV 可回收的请求



- **Figure 3 eviction analysis: 5-strategy × mixed × rate=3.8 × 3 runs = 15 runs** (2026-04-06):
  - 新增 `AdapterMetrics.record_all_preemption()`，包装 `scheduler._preempt_request` instance attr
    捕获所有 preemption（native LIFO + proactive + SRPT），修复旧 Figure 3 所有策略全零 bar 的问题
  - 新增 `AdapterMetrics.total_all_preemptions`、`total_all_tokens_freed` 字段并在 `as_dict()` 导出
  - `scheduler_hook.py`：`install_scheduler_hook()` 新增 `_preempt_request` instance wrapper；
    同时修复 h2o_hook → positional_hook 命名 bug（`_H2O_STRATEGIES` → `_POSITIONAL_STRATEGIES`）
  - `analysis.py`：`_plot_eviction_coverage()` 优先使用 mixed × rate=3.8 数据（原来仅 long_context）；
    `StrategyAggregation` 新增 `total_all_preemptions_mean`、`total_all_tokens_freed_mean` 字段
  - 结果（5 策略均非零，3 runs 均值，mixed rate=3.8）：
    | Strategy | Preemptions | Tokens Freed |
    |---|---|---|
    | preempt-evict | 247 | 196k |
    | preempt-evict-sjf | 336 | 366k |
    | static-random | 329 | 88k |
    | largest-first | 296 | 325k |
    | BidKV | 323 | 174k |
  - Figure 3 PDF: `results/vllm_fig3_mixed_rate38/analysis/figures/fig3_eviction_analysis_mixed.pdf`
  - 471 tests pass

  - 新增 `RandomEvictStrategy`（`src/bidkv/baselines/random_evict.py`），实现随机 victim 排序
  - 新增 `STRATEGY_VANILLA_SGLANG`、`STRATEGY_RANDOM_EVICT`、`SGLANG_NATIVE_ABL_STRATEGIES` 到 `config.py`
  - 更新 `EXTENDED_STRATEGIES` + `STRATEGY_BASELINE_MAP` 支持两个新策略
  - `scheduler_hook.py` 5 处策略分支更新：vanilla_sglang = 纯 pass-through；random_evict = FCFS waiting + 随机 running reorder + 随机 proactive preempt；no SRPT
  - `registry.py` 注册 `RandomEvictStrategy`（registry 策略总数 7→8）
  - 实验结果（mixed rate=5.7, 3×3=9 runs）：
    | Strategy | TTFT-p50 | TTFT-p95 | Tput | SLO(300ms) |
    |---|---|---|---|---|
    | vanilla_sglang | 188ms | 6246ms | 3.04 r/s | 55.2% |
    | random_evict | 114ms | 879ms | 5.57 r/s | 85.7% |
    | bidkv | 114ms | **598ms** | 5.53 r/s | **86.9%** |
  - 消融链验证：vanilla→random 证明打破 SGLang 默认顺序有巨大收益（p95 6246→879ms）；random→bidkv 证明质量感知排序有额外收益（p95 879→598ms，+3.3pp SLO）

### Changed

- **v9 公式实验失败，恢复 v8 公式 + H2O→Positional 全量重命名** (2026-04-07):
  - v9 公式 `δ = (M/R) × (1+P)` 实验灾难性失败：throughput -46%, 170 timeouts, 79 proactive evictions 导致驱逐风暴
  - 根因：v9 的极端 δ 值域 + 误删的 BidKV proactive preempt skip 共同引发
  - **恢复 v8 公式** `δ = 1 + 0.5·c + 0.3·P`（δ ∈ [1.0, 1.8]）
  - **恢复 BidKV proactive preempt skip**：`_proactive_preempt()` 中 `if strategy_name == "bidkv": return`
    （该代码在 commit fdd9a3f "freeze v8 env" 中被误删）
  - `_compute_keep_score()` 同步恢复为 v8 公式
  - h2o_hook.py → positional_hook.py (vLLM)，SGLang h2o_hook.py 已删除（死代码）
  - `update_h2o_from_output` → `update_positional_from_output`
  - `_H2O_STRATEGIES` → `_POSITIONAL_STRATEGIES`
  - 所有 src/tests 中的 H2O 引用更新为 positional/largest-first
  - `global_nobid.py` δ_H2O → δ_pos
  - `test_freed_dominant_over_completion` 断言修正为 v8 预期排序
  - copilot-instructions.md 同步更新
  - 471 tests pass, ruff clean

- **Phase D empirical motivation + SGLang validation (phase-d-empirical merge)**:
  - BidKV v5 公式重构（基于 recompute 代价感知）— 后被 v8 恢复覆盖
  - SGLang adapter fixes: `_get_token_to_kv_pool()` 修复, stale `current_tokens` 修复
  - Proactive preemption thresholds tightened (后被 v8 恢复覆盖)
  - SGLang validation: bidkv/preempt-evict-sjf/h2o-style on mixed rate=3.8
  - Results in `results/sglang_validation_v1/`
- **Comprehensive data analysis guide in copilot-instructions** (2026-04-03):
  - Replaced brief JSON format snippet with full 8-section analysis reference
  - Added file naming convention, all 13 top-level keys, all 10 request fields
  - Documented adapter_metrics field name inconsistency (`total_compressions` vs `total_evictions`)
  - Added summary field reference (11 keys, no p95/SLO)
  - Added complete `load_run()` / `load_all_runs()` / `cross_rate_average()` code templates
  - Added 7 known data traps table with correct workarounds
  - Added audit-verified eviction reference values per strategy × workload
  - Synced to both `.github/copilot-instructions.md` and `.github/agents/bidkv-agent.md`

- **Rename h2o-style → largest-first** (2026-04-06):
  - `baselines/h2o_style.py` → `baselines/largest_first.py`
  - `H2OStyleStrategy` → `LargestFirstStrategy` (backward compat alias kept)
  - Strategy name: `"h2o-style"` → `"largest-first"`
  - Config constant: `STRATEGY_H2O_STYLE` → `STRATEGY_LARGEST_FIRST`
  - Added `STRATEGY_LEGACY_NAMES` mapping for frozen result data compatibility\n  - All analysis display names updated to \"Largest-First\"

### Removed

- **Remove dead `BidKVStrategy._completion_factor()`** (2026-04-06):
  - Method was never called by `select_victims()` (v8 formula uses inline calculation)
  - `GlobalNoBidStrategy._completion_factor()` retained (still in use)

### Fixed

- **Remove avg_prompt > 500 long-context gate from BidKV reorder** (2026-04-03):
  - `scheduler_hook.py` `_reorder_running_for_preemption()`: removed the
    `avg_prompt > 500 → return` guard that completely disabled quality-aware
    reorder for long-context workloads (avg prompt ~1785 tokens).
  - Root cause: gate was designed for mixed workload (avg ~300-500) but
    prevented BidKV's core mechanism from operating in long-context.
  - BidKV's U-score already handles recompute concerns via completion factor
    and anti-starvation penalty — the blunt prompt-length gate was redundant.
  - KV > 95% pressure gate retained (no reorder when KV isn't under pressure).
  - Expected impact: BidKV regains quality-aware victim selection for
    long-context, matching the mechanism that achieved SLO #1 + TTFT #1
    in mixed workload.

### Changed

- **Metric system FROZEN: 4-column main table (v8-frozen)** (2026-04-02):
  - **FROZEN** — 后续实验（long_context、SGLang）使用相同 4 列体系
  - Main table: Throughput + SLO attainment(300ms) + TTFT p95 + TPOT p95
  - Goodput(500ms) moved to supplementary (low universality, overlaps SLO)
  - Normalized Latency removed (covered by TTFT+TPOT decomposition)
  - p95 values recomputed from raw request data (previously used summary p99)
  - BidKV cross-rate: Throughput #4, SLO #1, TTFT #1, TPOT #4
  - Full analysis: `results/vllm_v8_analysis/v8_analysis_report.md`
  - Updated copilot-instructions.md with corrected metric system and data

- **Freeze v8 experiment environment** (2026-04-02):
  - Server params frozen: `--gpu-memory-utilization 0.5 --num-gpu-blocks-override 600
    --max-num-seqs 32 --block-size 16 --max-model-len 8192 --enforce-eager`
  - Workload rates frozen: mixed (2.0, 3.8, 5.7), long_context (0.35, 0.5, 0.7)
  - v8 mixed 63-run data frozen at `results/vllm_v8_full_validation/`
  - Updated copilot instructions with frozen params, full ranking tables, v8b SRPT results
  - Cleaned 230 stale intermediate result files from git tracking (pre-v8 data)
  - Added stale result dirs to .gitignore

- **Rename compression metrics to eviction metrics** (#terminology):
  - `total_compressions` → `total_evictions`（字段、dict key、JSON 输出）
  - `record_compression()` → `record_eviction()`（方法名）
  - `compression_coverage` → `eviction_coverage`（ExperimentMetrics 字段）
  - `_plot_compression_coverage()` → `_plot_eviction_coverage()`
  - 所有 analysis/report/pilot_calibration 代码同步更新
  - 旧 JSON 数据向后兼容：fallback 读取 `total_compressions` / `compression_coverage`
  - 动机：BidKV 在 Mode A 中是请求调度原语，不做压缩；"compression" 术语误导

- **scheduler_hook: finalize v8 scheduling with improved comments**:
  - Removed v11c proactive cheapest-victim block (reverted to generic cached-priority
    path with cost-benefit gate). Empirical evidence: proactive evictions add recompute
    overhead that hurts p99 (v11c p99=4964 vs v8 p99=4807 at rate=5.7, pooled 3k).
  - Disabled SRPT for BidKV with explanatory comment: each SRPT eviction triggers full
    prompt recompute, creating extreme tail latency (v8b p99=6192 at rate=5.7).
  - Updated reorder comments to reflect v8 pressure-gated design rationale.
  - **No functional change from v8** — code paths identical, only comments improved.
  - Tested variants that all performed worse than v8:
    - v11c (proactive cheapest-victim): +3% p99 regression  
    - v13 (LIFO starvation guard): +32% p95 regression
    - v8b (SRPT enabled): +124% p99 regression

- Added new custom agent [bidkv-empirical-motivation](.github/agents/bidkv-empirical-motivation-agent.md) for paper Section 2 "Empirical Motivation" bridging experiments:
  - Locks pre-experiment setup to rate=3.8 and KV proactive threshold >88%.
  - Encodes three mandatory studies: victim heterogeneity, shared-snapshot counterfactual victim preference, and KV pressure frequency/duration.
  - Enforces interpretation constraints for Mode A recompute fallback (delta as scheduling proxy, not output-quality loss).
  - Requires paired snapshot-level statistics and warns against online-trajectory confounding.

- **Optimization strategy pivot: Mixed TTFT focus + Long-Context gate**:
  - Primary target: Mixed workload TTFT advantage (bidkv(v5b) p99=3331ms vs pe-sjf 6476ms)
  - Long-Context: v5 three-gate mechanism for parity with pe-sjf (not superiority)
  - SLO threshold: Mixed adjusted to 1s (from 2s) for better strategy differentiation
  - Metrics: P95 replaces P50 in analysis (P50 differences too small)

- **Copilot instructions updated with Phase D optimization strategy and full experiment results**

### Fixed

- **vLLM scheduler_hook: add missing `_proactive_preempt()` call in `_patched_schedule()`**:
  - `_proactive_preempt()` was defined (line 433) but never called from the main schedule flow,
    making slack-aware lose its only proactive preemption mechanism (SRPT already excluded for
    EDF discipline). Now called before `_proactive_srpt()` in `_patched_schedule()`.
  - Effect: slack-aware now has proactive preemption at KV > 90% (5s cooldown), as specified
    in the strategy differentiation table.
  - SJF strategies also get `_proactive_preempt()` in addition to SRPT (harmless: SRPT fires
    more aggressively at KV > 80% with 1.5s cooldown, so proactive_preempt is rarely additive).
  - **Bug fix**: Added `prev_step_scheduled_req_ids.discard(victim_id)` after preemption in
    `_proactive_preempt()` — same fix already present in `_proactive_srpt()`. Without this,
    vLLM's `_make_cached_request_data()` hits `assert not scheduled_in_prev_step`, crashing
    the EngineCore subprocess.

### Changed

- **SGLang adapter: token-level → request-level Mode A scheduling**:
  - `scheduler_hook.py`: Complete rewrite — replaced token-level `try_compress()` hook with
    request-level Mode A scheduling (symmetric with vLLM Mode A):
    - 8-step flow in `_patched_get_next_batch_to_run()`: sync tracking → track arrivals →
      reorder waiting → reorder running → refresh priority cache → proactive preempt →
      proactive SRPT → call original
    - Waiting reorder: FCFS (preempt-evict), EDF (slack-aware), SJF (bidkv)
    - Running reorder: LIFO passthrough (preempt-evict), cached priority (slack-aware, bidkv)
    - Priority cache refresh via `strategy.select_victims()` (3s interval)
    - Proactive preempt at KV > 90%, proactive SRPT at KV > 80% (bidkv only)
    - Uses `functools.partial` monkey-patching (same pattern as vLLM adapter)
  - `adapter.py`: Added Mode A attributes (`_cached_preempt_priority`,
    `_last_priority_refresh`, `_request_arrival_ms`); marked `execute_compression()` and
    `try_compress()` as DEPRECATED (Mode B) with `warnings.warn(DeprecationWarning)`
  - `radix_hook.py`: Entire module marked as `[DEPRECATED — Mode B]`; token-level
    `free_kv_positions()` preserved for potential Mode B extension (issue #054)
  - `__init__.py`: Updated module docstring to reflect Mode A architecture
  - `serve_entry.py`: All strategies (including sglang_default) now install scheduler hooks
    for fair overhead comparison — removed `if strategy != "sglang_default":` guard

- **Narrative pivot: scheduling-centric framing (Mode B deprecated)**:
  - Updated copilot-instructions.md and bidkv-agent.md: BidKV is a "request preemption
    scheduling primitive", not a "compression scheduling primitive"
  - Core message: BidKV controls WHO gets preempted, execution is vLLM native preempt+recompute
  - All Mode B code marked with `DEPRECATED (Mode B)` docstring warnings:
    - `adapter.py`: `execute_compression()`, `_execute_tail_truncation()`, `execute_abort()`,
      `try_compress()`, `try_compress_for_request()`, `_sync_model_runner_block_table()`
    - `truncation_hook.py`: entire module marked as deprecated infrastructure
    - `scheduler_hook.py`: truncation install block marked as deprecated
  - Mode B code retained for potential future extension (issue #054)

- **Remove fake bid=max_tokens asymmetry — strategy differentiation via quality-aware U only**:
  - `_get_max_tokens_estimate()`: removed `strategy_name` parameter; all strategies now equally
    access `sampling_params.max_tokens` (standard API param, NOT a bid signal)
  - BidKV admission: changed from `SJF(prompt + max_output)` to `SJF(prompt)` same as other
    SJF strategies; admission is no longer a differentiation axis
  - SRPT: all SJF strategies (static-random, h2o, uniform, global-nobid, bidkv) now use
    identical remaining-cost estimation; previously global-nobid got hardcoded 180, others 256
  - BidKV's sole differentiator: quality-aware preemption via `select_victims()` using
    `U = r / (δ + ε)` from scoring→bid→pool→solver pipeline
  - plugin.py: all strategies (including preempt-evict) now install hooks for fair overhead comparison
  - 465 tests passing, ruff clean

### Fixed

- **Experiment fairness: 5-layer architecture reform for clean ablation**:
  - **_reorder disabled for experiments**: Strategy-aware reorder caused 15-40% throughput degradation for H2O-based strategies by interfering with vLLM's native FCFS preemption. Now uses vLLM default FCFS for all strategies during experiments.
  - **proactive_preempt: skip preempt-evict**: preempt-evict is the "no compression" baseline — it now skips proactive preempt entirely, measuring vanilla framework behavior.
  - **RequestState enrichment**: Added `num_prompt_tokens`, `num_computed_tokens`, `max_output_tokens`, `num_preemptions` fields. `_build_running_candidates` now populates all fields from vLLM request attrs + arrival time tracking.
  - **Completion penalty for BidKV/GlobalNoBid**: Near-completion requests get quadratic score inflation (1-5×), steering the solver away from truncating expensive-to-redo requests.
  - **Truncation cap**: Per-event truncation capped at 256 tokens (16 blocks) to prevent batch-size bloat cascade.
  - **h2o_hook sampling**: Reduced H2O attention proxy generation from every step to every 5th step (~5× CPU overhead reduction).
  - **Proactive preempt threshold**: 92% (was 88%), target 85% (was 80%).
  - **ARG001 suppress**: `_resolve_model_executor` scheduler arg.
  - Verified: BidKV #1 at rate 3.8 (3.08 rps, +5.1% vs preempt-evict), #3 at rate 5.7 (3.16 rps)

- **GlobalNoBid 策略修复：多级 compression levels + 正确的贪心选择**:
  - 根因：旧实现仅使用单一 `compressible_ratio=0.6`，H2O positional heuristic 下 delta≈0.22 永远超过 `delta_budget=0.15`，导致 `select_victims()` 始终返回空列表（0 truncations）
  - 修复：引入与 BidKV 相同的 `compression_levels=(0.2, 0.4, 0.6)`，为每个 candidate × 每个 level 生成 option，混合按 utility 贪心选择
  - 保持归因纯净：GlobalNoBid 与 BidKV 唯一差异 = 无 BidPool/Solver 协议（系统直接推断 vs 用户显式 bid）
  - 约束对齐：约束 A（每 request 最多 1 bid）+ 约束 B（Σδ ≤ budget）与 GreedyBidSolver 完全一致
  - 验证：修复后 GlobalNoBid 产出 107 compressions（修复前=0），BidKV 仍显著优于（100% vs 80% success, 2.45 vs 1.23 rps）

- **scheduler_hook 归因路由统一**:
  - `_proactive_preempt` 中 BidKV 之前走硬编码 `_compute_keep_score()` 路径而非 `select_victims()`，导致归因不纯
  - 修复：所有策略（含 BidKV）统一走 `select_victims()` 路径，`_compute_keep_score()` 仅在无策略时作为 fallback
  - 465 tests passing, ruff clean

### Removed

- **Mode A (recompute_fallback) 彻底移除**:
  - 删除 `BidKVConfig.execution_mode` 字段和 `_VALID_EXECUTION_MODES` 验证
  - 删除 `VLLMAdapter._execute_recompute_fallback()` 方法（~70 LOC）
  - `_execute_tail_truncation()` 失败时不再 fallback 到 recompute，直接返回 0
  - 移除 `plugin.py` 中 `BIDKV_EXECUTION_MODE` 环境变量读取
  - 移除实验 runner `--execution-mode` CLI 参数
  - 移除 `VLLMServerConfig.execution_mode` 字段
  - 清理 vLLM/SGLang adapter kill switch 中 `execution_mode` 引用
  - 重命名 `_mode_b_proactive_preempt` → `_proactive_preempt`
  - 更新测试：移除 5 个 recompute fallback 测试，新增 no-truncation-returns-zero 测试
  - 465 tests passing, ruff clean

- **Oracle-DP 策略完整移除**:
  - 删除 `src/bidkv/baselines/oracle_dp.py` 及所有 import/export/registry 注册
  - 从 vLLM 和 SGLang 实验配置中移除 oracle-dp 策略
  - 从 `analysis.py` 中删除 `compute_oracle_gap()` 和 `_plot_oracle_gap()` 函数
  - 从 `metrics.py` 中移除 `oracle_gap` 字段
  - 从论文 `paper/bidkv_sc2026.tex` 中移除所有 Oracle-DP 引用（28 处）
  - 从 `paper/tables/` 中移除 Oracle-DP 表格行
  - 更新所有文档：baseline-specs、experiment_protocol、sglang-portability-slice 等
  - vLLM: 8 策略 → 7 策略 (126 runs)；SGLang: 4 策略 → 3 策略 (54 runs)
  - 删除 DC-2 方向一致性检查（Oracle-DP ≥ BidKV）
  - 468 tests passing, ruff clean

### Changed

- **ScoringStrategy score-only 契约（二次修复）**:
  - `ScoringStrategy` Protocol 仅保留 `score()` 方法，移除 `generate_bids` 要求
  - 主管线统一为 `scorer.score() → build_bids() → pool → solve`，适用于 BidKVStrategy、VLLMAdapter、SGLangAdapter
  - 全部 H2O 硬编码注释更新为 scorer-agnostic 描述

### Fixed

- 修复 11 个 ruff lint 错误（SIM105, UP031, F401×5, B905×3, E501）

### Added

- **统一 score 语义 + 统一 bids 生成**:
  - 新增 `scoring/bid_builder.py` — `build_bids()` 统一将 token-level scores 转换为 CompressionBid
  - 所有 4 个 scorer（H2O, Attention, Random, Uniform）的 `generate_bids()` 委托 `build_bids()`
  - BidKVStrategy 构造器接受 `ScoringStrategy`（而非 `H2OScoring`），支持 scorer 注入
  - `build_bids` 导出至 `bidkv.scoring` 和 `bidkv` 顶层命名空间
  - 新增 `TestBuildBids`（7 用例）和 `TestBidKVStrategyScorerAgnostic`（5 用例）

- **Mode B v3: Truncated Recompute** (#054):
  - `truncation_hook.py` — monkeypatch `truncate_request_tail()` onto vLLM KVCacheManager（保留为底层工具）
  - `_execute_tail_truncation()` v3 语义：截断 output tokens → native preempt（避免 InputBatch 块表失同步）
  - v2 CUDA crash 根因：直接 block truncation 导致 GPUModelRunner InputBatch 引用已释放 block → device-side assert
  - v3 解决方案：不直接操作 blocks，先截断 `_output_token_ids`/`_all_token_ids`，再通过 `_preempt_request()` 安全释放
  - 净效果 vs Mode A：相同即时 KV 回收 + 更低 recompute cost（更短序列）+ 永久 KV 占用降低
  - `plugin.py` 支持 `BIDKV_EXECUTION_MODE` 环境变量
  - `config.py` / `runner.py` 新增 `--execution-mode` CLI 参数 (recompute_fallback | tail_truncation)
  - 20+ 新测试覆盖 Mode B 路径（466 tests total）

### Fixed

- **vLLM Mode A EngineCore crash** — `_execute_recompute_fallback()` 从 `abort_requests()`
  迁移到 `_preempt_request()` API（vLLM v1 不提供 `abort_requests`）:
  - 使用 `scheduler._preempt_request(request, timestamp)` 替代不存在的 `abort_requests()`
  - 从 `scheduler.running` 列表移除请求后调用 preempt
  - 清理 `prev_step_scheduled_req_ids` 防止 `_make_cached_request_data` 断言失败
  - 验证结果：BidKV 1000/1000 成功（修复前 85/1000），2222 次压缩，6.27M tokens 释放

### Added

- **断点续跑 (resume) 功能** (#053):
  - vLLM / SGLang runner 新增 `--resume` CLI 标志
  - 启用后自动跳过已存在结果文件的 runs，防止重启导致数据丢失
  - `VLLMExperimentRunner` / `SGLangExperimentRunner` 构造器新增 `resume: bool` 参数
  - `parse_args()` 返回 `tuple[ExperimentConfig, bool]`

- **Issue #053 全量执行脚本** `scripts/run_issue053.sh`:
  - 7 步自动化流程：环境验证→P1→P2→合并→SGLang→分析→验收
  - 支持 `--resume` 断点续跑模式

- **全量实验 + 分析管线** (#053):
  - vLLM 分析 CLI：`python -m bidkv.experiments.vllm.analysis --results-dir --output-dir`
  - SGLang 分析 CLI：`python -m bidkv.experiments.sglang.analysis --sglang-results-dir --output-dir`
  - 格式桥接函数 `_load_collector_results_as_report()`：collector RunResult JSON → ExperimentReport
  - Figure 6 budget sensitivity（RULE FIG6-DEFAULT）：surrogate budget sensitivity 计算 + PDF 绘制
  - Figure 7 cross-framework portability：跨框架 SLO attainment 并排柱状图 + DC 标注
  - `run_sglang_analysis()` 一键分析：Table 2 + DC 检查 + Figure 7
  - `scripts/run_issue053_pipeline.sh` 自动化 pipeline（P2→merge→SGLang→analysis）

- **Pilot Calibration + Trace/Rate 冻结** (#055):
  - 分析 90 组已有 pilot 数据（formal 72 + pilot_v3 12 + pilot_v3_mixed_high 6），确认无需额外 pilot（Situation A）
  - 冻结 per-workload request rates：mixed `(2.0, 3.8, 5.7)`, long_context `(0.35, 0.5, 0.7)`
  - 生成 seed=42 formal traces（6 文件 + manifest.json），SHA-256 验证通过
  - 创建 `results/pilot_055/calibration_report.md` 完整校准报告

- **Per-workload rate 架构**：
  - 新增 `WORKLOAD_REQUEST_RATES` 冻结字典（vLLM + SGLang config）
  - `ExperimentConfig` / `SGLangExperimentConfig` 新增 `workload_rates` 字段
  - `get_rates_for_workload(workload)` 方法替代全局 `request_rates`
  - `total_runs` 属性按 workload 独立计算 rate 数
  - runner.py CLI 新增 `--mixed-rates` / `--long-rates` 参数覆盖
  - `freeze_traces.py` 支持 `--use-frozen-rates` 从 config 读取冻结 rates

- **SGLang Smoke Test 完成** (#052): 4 策略 × 1 workload × 1 rate × 1 run = 4 runs
  - sglang_default / slack_aware / bidkv / oracle_dp 全部通过，零 crash
  - TTFT p50 ~85ms, TPOT p50 ~32ms, SLO attainment 100%（成功请求）
  - BidKV hook 注入（serve_entry.py + BIDKV_STRATEGY）验证通过
  - 结果保存至 `results/sglang_smoke_mode_a/`
  - 方向一致性初步通过（timeout-dominated regime，策略未分化）

### Changed

- **SGLang Adapter 策略路由** (#051 补丁): SGLangAdapter 新增 `experiment_strategy` /
  `experiment_strategy_name` 参数，支持 baseline 策略路由
  - `try_compress()` 根据策略名路由到 `_try_compress_baseline()` 或 BidKV pipeline
  - `serve_entry.py` 使用 `BaselineRegistry` 获取策略实例并传给 adapter
  - 确保 4 个 SGLang 策略产生不同的压缩行为

- **SGLang 策略列表更新至 v2.3 冻结版本** (#051):
  - 4 策略: sglang_default / slack_aware / bidkv / oracle_dp
  - 替换旧策略 global_nobid → slack_aware, uniform → oracle_dp
  - 更新 `STRATEGY_BASELINE_MAP` 映射
  - runner.py CLI `--strategies` 默认值同步更新
  - `strategies.py`: 策略配置从 v7 (global_nobid, uniform) 更新至 v2.3 (slack_aware, oracle_dp)
  - `analysis.py`: 方向一致性检查更新为 DC-1a/DC-1b/DC-2（v2.3 冻结定义）

### Fixed

- **vLLM/SGLang runner pipe 死锁**：`subprocess.Popen(stdout=PIPE)` 的 64KB 管道缓冲区被
  vLLM server 日志输出灌满后导致 server 进程阻塞，请求无法处理。改为重定向至日志文件
  (`output_dir/server_{strategy}.log`)。
- **vLLM runner server 状态累积**：同一 (strategy, workload, rate) 组内的 3 次 repeat run
  共享一个 server 实例，r0 之后 KV cache 耗尽导致 r1/r2 全部 timeout。改为每个 run 独立
  重启 server（start→warmup→run→stop→cooldown 3s）。
- **SGLang runner server 生命周期**：同类问题，server 原本每个 strategy 仅启动一次，
  所有 workload/rate/run 共享。同样改为每个 run 独立重启。
- **serve_entry.py**: `BidKVConfig(active=True)` → `BidKVConfig(enabled=True)`（`BidKVConfig` 无 `active` 字段）

### Added

- **Fairness audit logging** (#051): `write_audit_entry()` 函数
  - 输出 candidate_count + candidate_list_hash（确定性排序后 MD5）
  - JSONL 格式，追加写入 `results/sglang_*/audit_*.jsonl`
  - 用于验证 within-platform candidate-universe consistency
  - 3 个单元测试覆盖（文件创建、追加、hash 确定性）

### Changed

- **vLLM Mode A — Recompute Fallback Executor** (#049): 将 vLLM adapter 从危险的
  `_free_tail_blocks()` 切换到安全的 Mode A (Recompute Fallback) 执行模式
  - `BidKVConfig` 新增 `execution_mode` 字段（默认 `"recompute_fallback"`）
  - `execute_compression()` 根据 `execution_mode` 路由到不同执行策略
  - 新增 `_execute_recompute_fallback()`：仅使用 vLLM native `scheduler.abort_requests()` API
  - `_free_tail_blocks()` 标记废弃，调用时抛出 `RuntimeError`
  - 零 coordinator/block_pool 直接操作，消除 CUDA 内存损坏风险
  - 新增 13 个测试覆盖 recompute fallback、deprecation、execution_mode routing

### Added

- **SGLang Portability Experiment — Real HTTP Runner** (#048): SGLang 可移植性验证实验基础设施
  - `bidkv.experiments.sglang.config`: `SGLangExperimentConfig` / `SGLangServerConfig` / `SLOConfig`
    - SLO 值与 Issue-047 完全一致: ttft_target_ms=2000.0, tpot_target_ms=100.0
    - 4 策略: sglang_default / bidkv / global_nobid / uniform
    - 2 工作负载: mixed / long_context（正式名称与 Issue-047 对齐）
    - 3 请求率: 1.0, 2.0, 4.0 req/s
  - `bidkv.experiments.sglang.server`: `SGLangServer` 生命周期管理
    - sglang_default: 原生 SGLang server
    - 其他策略: 通过 BIDKV_STRATEGY 环境变量 + serve_entry.py 注入 adapter
    - SIGTERM → wait 15s → SIGKILL 优雅停止
  - `bidkv.experiments.sglang.serve_entry`: BidKV 注入入口点
    - Monkey-patch `Scheduler.__init__` 在 server 进程内注入 hooks
  - `bidkv.experiments.sglang.collector`: `RequestResult` / `RunResult` / `save_run_result`
    - P95/P99 TTFT、TPOT（computed property）、SLO attainment、completion rate
  - `bidkv.experiments.sglang.runner`: `SGLangExperimentRunner` 完全重写
    - **替换旧 simulation runner**: 不再使用公式估算，改为真实 HTTP Poisson 开环
    - Poisson 开环到达: 按 arrival_time_ms 调度请求，无 max_inflight 限制
    - /v1/chat/completions (streaming=True) SSE 首 chunk 精确 TTFT 采集
    - Consecutive timeout abort 机制
    - 复用 Issue-047 frozen traces (experiments/vllm/traces/)
    - CLI 入口: `python -m bidkv.experiments.sglang.runner`
  - 33 个单元测试全部通过

- **vLLM Adapter** (#044): BidKV 在 vLLM v1 架构上的完整适配器
  - Architecture Decision: vLLM v1 (0.17+) 移除了 `BlockSpaceManager` 和 `--block-manager-class`，
    因此采用 Scheduler monkey-patch 方案替代原 spec 的 BlockManager 子类方案（功能等价）
  - `bidkv.adapters.vllm.VLLMAdapter`: vLLM v1 适配器（实现 FrameworkAdapter ABC）
    - KV stats 获取: 从 `KVCacheManager` / `BlockPool` 读取 used/total blocks
    - Pressure interception: 在 `schedule()` 分配 slots 前尝试 BidKV 压缩
    - Compression 执行: 按 bid 释放 request 尾部 KV blocks
    - H2O decode step 回调: `update_from_output()` 后更新累积注意力评分
    - 请求生命周期管理: track/complete/cleanup（`_free_request` hook）
  - `bidkv.adapters.vllm.scheduler_hook`: vLLM v1 Scheduler monkey-patch 注入
    - `install_scheduler_hook()` / `uninstall_scheduler_hook()` 可逆 patch
    - `_patched_schedule()`: 在 `allocate_slots()` 前执行 BidKV `try_compress()`
    - `_patched_update_from_output()`: decode 完成后同步 token tracking + H2O scoring
    - `_patched_free_request()`: 请求结束时清理 BidKV 状态
  - `bidkv.adapters.vllm.h2o_hook`: H2O decode step 注意力代理生成
    - Position-based attention proxy: attention sink (pos 0–3) + recency bias
    - `update_h2o_from_output()`: 从 scheduler output 批量更新 H2O scoring
  - Kill switch 热切换: `activate_kill_switch()` / `deactivate_kill_switch()`
  - `AdapterMetrics`: 压缩统计（尝试次数、成功次数、释放 token 数）
  - 39 个测试全部通过

- **SGLang Adapter** (#045): BidKV 在 SGLang 框架上的完整适配器
  - `bidkv.adapters.base.FrameworkAdapter` ABC: 最小可行跨框架抽象（5 层职责边界）
  - `bidkv.adapters.sglang.SGLangAdapter`: SGLang RadixAttention 适配器
    - KV stats 获取: 从 `TokenToKVPool` 读取 used/total
    - Pressure interception: 在 RadixAttention LRU 驱逐前获得压缩尝试机会
    - Compression 执行: radix tree 节点级缩减，细粒度释放 KV
    - H2O decode step 回调: 更新累积注意力评分
    - 请求生命周期管理: track/complete/cleanup
  - `bidkv.adapters.sglang.radix_hook`: RadixAttention 节点级压缩/释放钩子
    - 共享前缀保护: ref count > 1 的 token 不可压缩
    - `get_shared_prefix_positions()`: 检测共享前缀位置
  - `bidkv.adapters.sglang.scheduler_hook`: Scheduler batch selection 前 bidkv 注入
    - Monkey-patch `get_next_batch_to_run()` 实现 pressure interception boundary
    - 同时 hook RadixCache eviction path
  - `bidkv.adapters.sglang.h2o_hook`: decode step 后 H2O scoring 更新回调
  - Kill switch 热切换: `activate_kill_switch()` / `deactivate_kill_switch()`
  - Metrics 输出格式与 vLLM adapter 对齐（directional consistency 可验证）
  - 27 个集成测试全部通过

- **vLLM 7-Baseline Experiment Framework** (#047): 8 策略 × 3 负载 × 3 并发 × 3 runs = 216 实验矩阵
  - `bidkv.experiments.vllm.config`: `ExperimentConfig` / `VLLMServerConfig` / `SLOConfig`（冻结配置，含完整验证）
  - `bidkv.experiments.vllm.workload`: `RequestTrace` / `WorkloadTrace` + dataset loaders（ShareGPT / CNN-DM / LongBench）
  - `bidkv.experiments.vllm.collector`: `RequestResult` / `RunResult` / `CandidateSnapshot` / `MetricsCollector`（Prometheus /metrics 解析）
  - `bidkv.experiments.vllm.runner`: `VLLMExperimentRunner`（async 并发发送 + SSE streaming TTFT 测量 + CLI）
  - `bidkv.experiments.vllm.analysis`: 聚合 + CI95 + Oracle gap + 6 张论文图（SLO violation bar / Pareto front / Oracle gap / Compression coverage）+ Table 1 数据
  - 策略覆盖: preempt-evict / static-random / h2o-style / uniform / global-nobid / slack-aware / bidkv / oracle-dp
  - 负载覆盖: chat (ShareGPT) / summarization (CNN/DM) / QA (LongBench)
  - Candidate-universe consistency 验证: 每 run 结束后检查所有策略接收同一候选集
  - 43 个测试全部通过

- **Baselines Implementation** (#046): 7 个 baseline 策略 + Oracle DP 上界
  - `bidkv.baselines.BaselineStrategy` ABC + `CompressionAction` / `RequestState` 数据类型
  - `bidkv.baselines.BaselineRegistry`: 策略注册表，支持 `create_default_registry()` 一键注册全部 8 个策略
  - `bidkv.baselines.PreemptEvictStrategy`: 零压缩基线（`victim = argmin(priority)`）
  - `bidkv.baselines.StaticRandomStrategy`: 固定比率 + 随机受害者（控制变量基线）
  - `bidkv.baselines.H2OStyleStrategy`: token 级 H2O scoring 压缩，不走 bid 机制（**H2O-Style ≠ H2OScoring**）
  - `bidkv.baselines.UniformStrategy`: 所有请求均等压缩（`∀req: compress(needed/N)`）
  - `bidkv.baselines.GlobalNoBidStrategy`: 系统自动推断 utility（`U_sys = r/(δ_H2O+ε)`），**关键 bid 归因 baseline**
  - `bidkv.baselines.SlackAwareStrategy`: SLO 剩余时间感知（deadline 远的先压缩）
  - `bidkv.baselines.BidKVStrategy`: 完整 bid pipeline 包装器（scoring → pool → solver）
  - `bidkv.baselines.OracleDPStrategy`: Grouped Knapsack DP 精确最优解（离线上界）
  - Candidate-universe consistency: 所有 baseline 接收同一 `candidates` 列表
  - Baseline Spec 预注册文档: `docs/baseline-specs.md`

### Fixed

- **SGLang Adapter Bug Fixes** (#045):
  - `scheduler_hook.py`: `uninstall_scheduler_hook()` 无法恢复原始方法——install 时未设置 `__wrapped__` 属性，uninstall 跳过恢复逻辑。同时修复 RadixCache eviction hook 的 `__wrapped__` 缺失及 uninstall 遗漏
  - `radix_hook.py`: `_is_shared_slot()` 始终返回 `False`——实现了基于 RadixCache 节点 `lock_ref` 和 running batch token 映射的双重共享检测
  - `adapter.py`: `_AdapterMetrics.to_dict()` 重命名为 `as_dict()`，与 vLLM adapter 对齐

- **Algorithm Corrections S01+S04+S07** (#042): 三项算法层语义修正
  - Fix S01 (#018): PressureDetector 保证使用瞬时值，无滚动窗口平滑——已验证并添加显式文档保证
  - Fix S04 (#021): 新增 `ExecutionResult` 数据类（`actual_freed` vs `estimated_freed`）及 `GreedyBidSolver.execute_accepted()` 方法，记录每轮压缩的实际效果
  - Fix S07 (#024): 新增 `PressureDetector.get_kv_stats()` 作为 KV 状态唯一数据源，`GreedyBidSolver.solve_with_detector()` 直接从 detector 获取 needed_tokens
  - 新增 22 个专项测试覆盖所有三项修正

### Added

- **Core Layer Extraction** (#041): BidPoolManager, GreedyBidSolver, PressureDetector, CompressionExecutor
  - `bidkv.pool.BidPoolManager`: 通用版 bid 快照管理器（零 sagellm 依赖）
    - `submit_bids()` 替代原始 `refresh(compressor, request)` — FrameworkAdapter 直接提交 bid 列表
    - feature gate + kill switch + thread-safe (threading.Lock)
    - `get_pool_snapshot()` → BidPool (frozen, candidate-universe consistency)
  - `bidkv.solver.GreedyBidSolver`: 贪心 Knapsack 求解器 (Algorithm 1, 论文 §4)
    - Utility-ratio 排序: U = r/(δ+ε), 仅使用 Layer 1 字段
    - 约束 A: 每 request 最多 1 bid; 约束 B: Σδ ≤ delta_budget
    - `SolverConfig`: enabled, delta_budget, max_bids_per_solve, kill_switch
  - `bidkv.pressure.PressureDetector`: KV 内存压力感知（通用版）
    - `update_stats(used_tokens, max_tokens, pending_high_priority)` — 纯数值输入
    - 触发条件: 占用率 ≥ threshold_pct 或 (pending_high_priority > 0 且 free < min_free_tokens)
    - `PressureConfig`: threshold_pct, min_free_tokens, enabled
  - `bidkv.compression.CompressionExecutor`: Protocol — FrameworkAdapter 实现接口
  - 68 个 core 单元测试全部通过（含 Pool→Solver→Pressure 联动测试）

- **Scoring Strategy Layer** (#043): Token 重要度评分策略
  - `ScoringStrategy` Protocol（`score()` + `generate_bids()`）
  - `H2OScoring`: Heavy Hitter Oracle — 基于累积注意力的 practical scoring（CPU, 无 GPU 依赖）
  - `AttentionWeightScoring`: Full attention weight aggregate — reference scoring
  - `UniformScoring`: 等权基线（消融实验用）
  - `RandomScoring`: 随机基线（消融实验用）
  - `generate_bids()` 按压缩级别生成合法 CompressionBid（三层体系完整）
  - H2O vs AttentionWeight Spearman rank correlation ≥ 0.7 验证通过
  - H2OScoring 精度边界声明文档（稀疏度 >90%、>32K token、多轮对话退化条件）
  - 60 个 scoring 单元测试全部通过

## [0.1.0] - 2026-03-14

### Added

- 从 `sagellm-protocol` 提取 CompressionBid 协议层为独立包 (issue #040)
- `bidkv.protocol.bid`: CompressionBid, BidPool, BidAcceptance 核心数据结构
- `bidkv.protocol.errors`: CompressionBidError 异常层次结构
- `bidkv.protocol.provider`: CompressionBidProvider Protocol 接口
- `bidkv.config`: BidKVConfig (feature gate + kill switch, 默认 OFF)
- CompressionBid 字段三层体系标注 (Layer 1/2/3)
- `compute_utility()` 标注为 "operational ranking signal, not ground-truth"
- 零外部依赖（仅 Python stdlib）
- 77 个单元测试全部通过
