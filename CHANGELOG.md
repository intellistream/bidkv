# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

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
