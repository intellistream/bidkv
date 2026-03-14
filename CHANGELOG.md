# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

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
