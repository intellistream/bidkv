# BidKV Baseline Specifications

本文档为 #046 Baseline Spec 预注册文档，包含每个 baseline 的设计理由（design rationale）和选择公式。

## 概述

BidKV 实验包含 7 个 baseline，用于消融实验验证 bid 机制各组件的增量价值。

**Candidate-universe consistency**: 所有 baseline 在同一 pressure event 中使用同一候选池，保证 within-platform 公平性。

## 归因关系（Ablation Attribution）

```
Preempt-Evict → H2O-Style  :  加入 token-level scoring 的收益
H2O-Style → Global-NoBid   :  加入 request-level utility 推断的收益
Global-NoBid → BidKV        :  加入 bid 接口（用户显式偏好）的收益 ⭐
Uniform → BidKV             :  差异化压缩 vs 均等压缩的收益
Slack-Aware → BidKV         :  quality-aware vs SLO-only 的收益
```

## Baseline 规格

### 1. Preempt-Evict — 零压缩基线

| 字段 | 值 |
|------|-----|
| **信息量** | 最低 |
| **设计理由** | 框架默认行为（vLLM preemption / SGLang eviction）。不执行任何 KV 压缩，直接驱逐请求。作为所有策略的下界。 |
| **选择公式** | `victim = argmin(priority)` |
| **实现** | 按优先级升序驱逐，直到释放足够 token |

### 2. Static-Random — 控制变量基线

| 字段 | 值 |
|------|-----|
| **信息量** | 低 |
| **设计理由** | 固定压缩率 + 无信息量的随机选择。验证"有信息"策略相对"随机猜测"的优势。 |
| **选择公式** | `victim = random.choice(active), ratio = 0.5` |
| **实现** | 随机打乱候选列表，每个请求压缩固定 50% token |

### 3. H2O-Style — token-level scoring，不走 bid

| 字段 | 值 |
|------|-----|
| **信息量** | 中（有 scoring，无 bid） |
| **设计理由** | 使用 H2OScoring 做 token 级重要度评分，但不经过 bid → pool → solver 流程。隔离 bid 机制贡献。**H2O-Style ≠ H2OScoring**：H2O-Style 是策略名，H2OScoring 是评分实现。 |
| **选择公式** | `compress(top_k(H2OScoring, 1 - heavy_ratio - recent_ratio))` |
| **实现** | 对每个请求用 H2OScoring 评分，压缩低重要度 token，按可释放量贪心选择 |

### 4. Uniform — 均等压缩

| 字段 | 值 |
|------|-----|
| **信息量** | 中（无差异化） |
| **设计理由** | 所有请求均等压缩，不做差异化决策。验证"差异化压缩"相对"均等压缩"的优势。 |
| **选择公式** | `∀req: compress(needed_tokens / N)` |
| **实现** | 每个请求压缩 `ceil(needed / N)` 个 token |

### 5. Global-NoBid ⭐ — 关键归因 baseline

| 字段 | 值 |
|------|-----|
| **信息量** | 高（有 scoring，无 bid 接口） |
| **设计理由** | 系统自动用 H2OScoring 推断 utility（`U_sys = r / (δ_H2O + ε)`），不暴露 bid 接口给用户。**这是 bid 机制价值的关键归因 baseline**：如果 BidKV > Global-NoBid，说明用户显式 bid 比系统推断更有价值。 |
| **选择公式** | `U_sys = r / (δ_H2O + ε), greedy by U_sys` |
| **实现** | 为每个请求估算系统 utility，按 utility 降序贪心选择，同时遵守 delta_budget 约束 |

### 6. Slack-Aware — SLO 剩余时间感知

| 字段 | 值 |
|------|-----|
| **信息量** | 高（有调度信息，无 quality 信息） |
| **设计理由** | 使用 SLO deadline 信息做调度决策，但不考虑压缩对质量的影响。验证 quality-aware 信息的价值。 |
| **选择公式** | `victim = argmax(slack), slack = deadline - now`（deadline 远的先压缩） |
| **实现** | 按 SLO slack 降序排列，先压缩 deadline 最远（或无 SLO）的请求 |

### 7. BidKV — 完整 bid 机制

| 字段 | 值 |
|------|-----|
| **信息量** | 最高 |
| **设计理由** | 完整 bid 机制：H2OScoring → CompressionBid → BidPoolManager → GreedyBidSolver。包含 token-level scoring、request-level utility、和用户显式 bid 接口。 |
| **选择公式** | `U = r / (δ + ε), greedy by U`（Algorithm 1） |
| **实现** | 为每个请求生成多级 bid，提交到 pool，使用 GreedyBidSolver 求解 |

## Candidate-Universe Consistency

所有 baseline 的 `select_victims()` 方法接收同一个 `candidates` 参数。
实验 Runner 在每个 pressure event 中：
1. 获取当前活跃请求快照 → `candidates`
2. 将同一个 `candidates` 列表传递给所有 baseline
3. 分别记录各 baseline 的决策和结果

## Implementation Priority vs Paper Presentation Priority

| 实现优先级 | 策略 | 论文展示优先级 |
|-----------|------|--------------|
| P1 | Preempt-Evict, Static-Random | 必展示 |
| P1 | H2O-Style, Global-NoBid | 必展示（核心归因） |
| P1 | BidKV | 必展示 |
| P2 | Uniform, Slack-Aware | 必展示（消融补充） |
