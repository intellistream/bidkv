# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- **GlobalNoBid 策略修复：多级 compression levels + 正确的贪心选择**:
  - 根因：旧实现仅使用单一 `compressible_ratio=0.6`，H2O positional heuristic 下 delta≈0.22 永远超过 `delta_budget=0.15`，导致 `select_victims()` 始终返回空列表（0 truncations）
  - 修复：引入与 BidKV 相同的 `compression_levels=(0.2, 0.4, 0.6)`，为每个 candidate × 每个 level 生成 option，混合按 utility 贪心选择
  - 保持归因纯净：GlobalNoBid 与 BidKV 唯一差异 = 无 BidPool/Solver 协议（系统直接推断 vs 用户显式 bid）
  - 约束对齐：约束 A（每 request 最多 1 bid）+ 约束 B（Σδ ≤ budget）与 GreedyBidSolver 完全一致
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
