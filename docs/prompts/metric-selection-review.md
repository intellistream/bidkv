# BidKV 论文指标体系修订评审

> **决策已定稿 (2026-04-02)**：选择 **方案 B（4 列主表）**，详见下方决策记录。

## 最终决策：方案 B（4 列主表）

| # | 主表指标 | BidKV 排名 | 理由 |
|---|----------|-----------|------|
| 1 | Throughput (req/s) | #4 | 社区标准必报指标，reviewer 期望 |
| 2 | SLO attainment(300ms) (%) | **#1** | 用户体验核心，BidKV 最强项 |
| 3 | TTFT p95 (ms) | **#1** | 尾部延迟，调度质量直接体现 |
| 4 | TPOT p95 (ms) | #4 | 完整覆盖 decode 阶段，展示诚实性 |

**Goodput(500ms)** → 补充指标（appendix / text discussion）
**Normalized Latency** → 已移除

### 决策理由

1. **Throughput 是必报指标**：几乎所有 LLM serving 论文都报告 Throughput，
   省略会引发 cherry-picking 质疑。BidKV #4 虽非最优，但差距仅 7%，
   可通过 tradeoff 叙事合理化。

2. **4 列完整覆盖四个维度**：吞吐(Throughput) + SLO 达标率(SLO) +
   prefill 尾部延迟(TTFT) + decode 尾部延迟(TPOT)，信息正交性最高。

3. **诚实性优于选择性**：仅展示 BidKV #1 的指标（3 列方案）虽然更"好看"，
   但对审稿人而言不如展示完整 4 维度再解释 tradeoff 更有说服力。
   包含弱指标展示的是自信，而非弱点。

4. **Goodput 正交性不足**：SLO attainment(300ms) 和 Goodput(500ms) 都是
   TTFT 阈值指标，正交性低。且 Goodput 仅 2-3 篇论文使用，
   reviewer 可能不熟悉。

5. **BidKV 2/4 #1 + 2/4 competitive 是强结果**：
   没有任何 serving 论文声称在所有指标上同时最优。
   BidKV 的定位是"用 7% 吞吐换显著更好的延迟质量"，4 列表完整呈现这一 tradeoff。

---

## 当前指标体系（copilot-instructions.md 已冻结，4 列）

| 指标 | 定义 | 出处 | BidKV 排名 |
|------|------|------|-----------|
| Goodput(500ms) | TTFT≤500ms 的有效吞吐 (req/s) | DistServe (OSDI'24) | #3 |
| SLO attainment(300ms) | TTFT≤300ms 达标率 (%) | S³ (ISCA'24) | **#1** |
| TTFT p95 | 第 95 百分位 TTFT (ms) | 通用 LLM serving | **#1** |
| TPOT p95 | 第 95 百分位 per-token 输出时间 (ms) | Sarathi-Serve (OSDI'24) | #4 |

已移除：Normalized Latency (Orca OSDI'22) — 被 TTFT+TPOT 分解严格覆盖，
近年 serving 论文已不再作为主指标使用。

---

## 调研结论

### 1. Goodput 的普遍性问题

Goodput(SLO) 定义为"满足 TTFT≤阈值 的请求有效吞吐"，来自 DistServe (OSDI'24)。
文献调研结果：

| 论文 | 是否使用 Goodput | 主指标选择 |
|------|-----------------|-----------|
| DistServe (OSDI'24) | ✅ 首创 | Goodput(SLO) 唯一主指标 |
| Mooncake (ATC'25) | ✅ 采用 | Goodput + P99 latency |
| Sarathi-Serve (OSDI'24) | ❌ | Serving capacity (最大吞吐@延迟约束) |
| S³ (ISCA'24) | ❌ | Throughput + SLO CDF 曲线 |
| vLLM (SOSP'23) | ❌ | Throughput, latency CDF 辅助 |
| Orca (OSDI'22) | ❌ | Throughput + Normalized Latency |
| SGLang (各版本) | ❌ | Throughput + latency breakdown |

**结论**：Goodput 仅有 2-3 篇论文使用，尚未成为社区标准指标。
而 Throughput (req/s) 是几乎所有 LLM serving 论文都报告的基础指标。

### 2. Goodput 与 Throughput 在本实验中的区别

| 指标 | BidKV | static-random | uniform | 差距 |
|------|-------|--------------|---------|------|
| Goodput(500ms) | 2.79 (#3) | 2.92 (#1) | 2.90 (#2) | BidKV 落后 4.7% |
| Throughput (req/s) | 2.99 (#5) | 3.21 (#1) | 3.21 (#2) | BidKV 落后 6.9% |

两者排名趋势一致，但 Goodput 对 BidKV 更友好（因为 BidKV 的 TTFT 优势
使更多请求满足 500ms 阈值）。**替换为 Throughput 对 BidKV 不利**。

### 3. Throughput 的论文标准性

Throughput 是 LLM serving 论文的"必报指标"：
- vLLM、Orca、SGLang、S³、Sarathi-Serve 全部使用
- Reviewer 期望看到 Throughput
- Throughput 含义直观，无需解释 SLO 阈值选择

### 4. TPOT p95 的定位

BidKV TPOT p95 排名 #4（96.5ms），弱于 static-random (#1, 86.0ms)。
原因：BidKV 不使用 SRPT，无法主动释放 decode 慢的请求。

TPOT 在 serving 论文中的地位：
- Sarathi-Serve: 使用 TPOT 作为 decode 效率指标
- DistServe: 不单独报告 TPOT（absorbed into Goodput）
- S³: 使用 TTFT+TPOT 作为 SLO 目标的两个维度

### 5. 同时赢得所有指标的可行性

文献调研：**没有任何 serving 论文声称在所有 4+ 独立指标上同时最优。**
标准做法：
- 在 1-2 个核心指标上最优
- 其他指标 competitive / 在可接受范围内
- 用 narrative 解释 tradeoff

---

## 提议的修订方案

### 方案 A：3 列主表（推荐）

| 主表指标 | BidKV 排名 | 理由 |
|----------|-----------|------|
| SLO attainment(300ms) | **#1** | 用户体验核心，BidKV 最强项 |
| TTFT p95 | **#1** | 尾部延迟，调度质量直接体现 |
| Throughput (req/s) | #5 | 社区标准必报指标 |

TPOT p95 → supplementary figure 或 appendix table。

**优势**：主表 2/3 指标 #1，Throughput 补全 reviewer 期望。
**风险**：Throughput #5 较弱，但绝对值差距 6.9% 可用 narrative 合理化。

### 方案 B：4 列主表（保留当前体系，替换 Goodput→Throughput）

| 主表指标 | BidKV 排名 |
|----------|-----------|
| SLO attainment(300ms) | **#1** |
| TTFT p95 | **#1** |
| Throughput (req/s) | #5 |
| TPOT p95 | #4 |

**优势**：信息完整，decode 效率有覆盖。
**风险**：4 列中仅 2 列 #1，另 2 列偏弱，整体印象可能不如 3 列。

### 方案 C：4 列主表（保留 Goodput，不替换）

保持当前已冻结的 4 列体系不变。

| 主表指标 | BidKV 排名 |
|----------|-----------|
| Goodput(500ms) | #3 |
| SLO attainment(300ms) | **#1** |
| TTFT p95 | **#1** |
| TPOT p95 | #4 |

**优势**：Goodput 对 BidKV 比 Throughput 更友好（#3 vs #5），且包含延迟质量信息。
**风险**：Goodput 不够普遍，reviewer 可能不熟悉；4 列仅 2 列 #1。

---

## 完整 Cross-Rate 排名数据（最终 p95 计算结果）

### 7 策略 Cross-Rate Average（3 rates 均值，p95 从原始请求数据计算）

| 策略 | Throughput | SLO300% | TTFT p95 | TPOT p95 | Goodput |
|------|-----------|---------|----------|----------|---------|
| **bidkv** | 2.99 (#4) | **87.1 (#1)** | **554 (#1)** | 96.4 (#4) | 2.79 (#3) |
| static-random | 3.21 (#1) | 87.0 (#2) | 1076 (#5) | 86.0 (#1) | 2.92 (#1) |
| uniform | 3.21 (#2) | 86.9 (#3) | 1069 (#4) | 86.7 (#2) | 2.90 (#2) |
| h2o-style | 2.92 (#6) | 84.4 (#4) | 584 (#3) | 100.1 (#6) | 2.68 (#4) |
| preempt-evict-sjf | 2.77 (#7) | 82.8 (#5) | 572 (#2) | 129.4 (#7) | 2.46 (#5) |
| slack-aware | 3.05 (#3) | 72.4 (#6) | 4023 (#6) | 93.2 (#3) | 2.24 (#6) |
| preempt-evict | 2.98 (#5) | 72.2 (#7) | 5241 (#7) | 98.3 (#5) | 2.19 (#7) |

### BidKV Per-Rate 表现

| Rate | SLO300 Rank | TTFT95 Rank | Thru Rank | TPOT95 Rank | Wins/4 |
|------|------------|------------|-----------|------------|--------|
| 2.0 | #1 | #1 | #1 | #1 | **4/4** |
| 3.8 | #2 | #1 | #5 | #5 | 1/4 |
| 5.7 | #2 | #2 | #4 | #4 | 0/4 |
| **Cross-rate** | **#1** | **#1** | #4 | #4 | **2/4** |

### 核心观察

1. **BidKV 的竞争优势集中在用户体验指标**：SLO #1 + TTFT #1
2. **系统效率指标偏弱**：Throughput #5, TPOT #4（原因：BidKV 禁用 SRPT）
3. **低压力（rate=2.0）全胜**，高压力（rate=5.7）竞争力下降
4. **static-random/uniform 在系统指标上领先**：因为启用了 SRPT

---

## 评审要求

请基于以下维度给出建议：

1. **Reviewer 接受度**：SC 2026 reviewer 对这些指标组合的接受程度
2. **指标诚实性**：是否存在 cherry-picking 嫌疑
3. **叙事一致性**：指标选择与 BidKV "quality-aware scheduling" 定位的一致性
4. **方案选择**：推荐方案 A / B / C，或提出方案 D
5. **Throughput 弱势的处理**：如何在论文中合理化 BidKV Throughput #5

## 约束

- 实验数据已冻结，不可重跑
- 策略代码已冻结，不可修改实验行为
- 7 策略列表不可变
- Rate (2.0, 3.8, 5.7) 不可变
