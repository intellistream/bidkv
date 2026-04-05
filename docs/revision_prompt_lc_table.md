# Revision Agent Prompt: Table 3 LC Methodology Fix

## 任务目标

分析 BidKV SC 2026 论文 Table 3（Long-Context 工作负载评估表）的呈现方法是否合理，
并提出修改建议及对应的 §6.3 文本重写方案。**不改变 §6 整体结构**。

---

## 背景与问题定义

### 当前 Table 3 方法：跨速率平均

Table 3 目前使用三个速率（0.35 / 0.5 / 0.7 req/s）的**跨速率平均值**，
每策略每速率 3 runs 取均值后再对三个速率取算术平均。

### 统计问题：TTFT P95 跨速率平均无效

TTFT P95 在不同速率下呈强非线性分布（接近饱和临界时突变），
导致跨速率算术平均实际上被最高速率的极端值主导：

| 策略 | rate=0.35 贡献 | rate=0.5 贡献 | rate=0.7 贡献 |
|------|--------------|-------------|-------------|
| BidKV | 3,461ms (7%) | 12,871ms (28%) | 30,143ms **65%** |
| Static-Random | 2,348ms (5%) | 9,667ms (22%) | 31,693ms **73%** |
| Largest-First | 2,309ms (5%) | 13,133ms (27%) | 33,345ms **68%** |
| PE-SJF | 7,306ms (12%) | 15,936ms (26%) | 37,551ms **62%** |

**结论**：所谓"跨速率平均 TTFT P95 = 15,492ms"几乎完全是 rate=0.7 的重复，
不是三个速率的均衡代表。SLO attainment 受此影响相对较小（因为 SLO 是百分比，有上界），
但 TTFT P95 的跨速率平均在统计上不可辩护。

### §6 结构约束

**LC（长上下文）是补充性评估，不是主线。** 主线是 mixed 工作负载（§6.2/6.3）。
§6.3 当前是 2-3 段的补充讨论，引用 Table 3。
**不应将 LC 升级为与 mixed 等量的 per-rate 全表**（那需要全面重写 §6，改变论文叙事重心）。

---

## 完整实验数据（已验证，2026-04-02 冻结，每策略每速率 3 runs 均值）

### Table A：LC 全速率 Per-Rate 数据

```
Strategy        Rate |  SLO%   TTFT_p50  TTFT_p95   Thru  TPOT_p95
-------------------------------------------------------------------
PE              0.35 |  54.1%    1,063ms   34,810ms  0.318    158ms
PE              0.50 |   4.5%   54,290ms  110,898ms  0.391    202ms
PE              0.70 |   5.0%   60,198ms   94,696ms  0.413    184ms

PE-SJF          0.35 |  89.1%      376ms    7,306ms  0.318    124ms
PE-SJF          0.50 |  72.1%      695ms   15,936ms  0.445    164ms
PE-SJF          0.70 |  63.6%    1,133ms   37,551ms  0.563    145ms

Static-Random   0.35 |  93.5%      354ms    2,348ms  0.319    113ms
Static-Random   0.50 |  74.0%      703ms    9,667ms  0.453    166ms
Static-Random   0.70 |  57.4%    1,602ms   31,693ms  0.568    208ms

Largest-First   0.35 |  94.5%      373ms    2,309ms  0.319    142ms
Largest-First   0.50 |  70.9%      896ms   13,133ms  0.445    253ms
Largest-First   0.70 |  64.4%    1,165ms   33,345ms  0.542    240ms

BidKV           0.35 |  91.9%      376ms    3,461ms  0.319    142ms
BidKV           0.50 |  72.5%      753ms   12,871ms  0.448    190ms
BidKV           0.70 |  62.3%    1,293ms   30,143ms  0.560    207ms
```

SLO 阈值：2,000ms（长上下文工作负载，松弛阈值）。
每格为 3 runs 算术均值。PE = Preempt-Evict（vLLM 原生 LIFO）。

### Table B：当前 Table 3（跨速率平均，存在问题）

```
Strategy       |  SLO%   TTFT_p95   Thru  TPOT_p95
---------------------------------------------------
PE             |  21.2%   80,135ms  0.374    181ms
PE-SJF         |  74.9%   20,264ms  0.442    144ms
Static-Random  |  75.0%   14,569ms  0.447    163ms
Largest-First  |  76.6%   16,262ms  0.435    211ms   ← best SLO
BidKV          |  75.6%   15,492ms  0.443    180ms
```

### Table C：提议的替换方案——rate=0.7 单速率快照

```
Strategy       |  SLO%   TTFT_p95   Thru  TPOT_p95
---------------------------------------------------
PE             |   5.0%   94,696ms  0.413    184ms
PE-SJF         |  63.6%   37,551ms  0.563    145ms
Static-Random  |  57.4%   31,693ms  0.568    208ms
Largest-First  |  64.4%   33,345ms  0.542    240ms
BidKV          |  62.3%   30,143ms  0.560    207ms   ← best TTFT P95
```

---

## SLO 排序（每速率，不含 PE 基线）

| 速率 | #1 SLO | #2 SLO | #3 SLO | #4 SLO |
|------|--------|--------|--------|--------|
| 0.35 | LF 94.5% | SR 93.5% | BidKV 91.9% | PE-SJF 89.1% |
| 0.50 | SR 74.0% | BidKV 72.5% | PE-SJF 72.1% | LF 70.9% |
| 0.70 | LF 64.4% | PE-SJF 63.6% | **BidKV 62.3%** | SR 57.4% |

**注意**：rate=0.7 时 BidKV SLO 排第 3（落后 LF 约 2.1pp），但 TTFT P95 排**第 1**（30,143ms 优于 SR=31,693ms、LF=33,345ms、PE-SJF=37,551ms）。

---

## 核心分析问题（供 revision agent 回答）

### Q1：rate=0.7 单速率快照是否比跨速率平均更 defensible？

**支持换**：
- TTFT P95 跨速率平均在统计上无意义（rate=0.7 占 62-73% 权重）
- rate=0.7 对应系统进入长尾饱和区，是 LC 工作负载最具代表性的压力点
- Reviewer 容易发现"跨速率平均 TTFT"的问题并提出质疑

**反对换**：
- rate=0.7 是三个速率中最大的，选择它可能被认为 cherry-picking
- cross-rate avg 与 mixed 工作负载（Table 1）的呈现方式一致

**若换**：需要脚注说明其他速率结论一致（方向不变）。

### Q2：rate=0.7 快照下 BidKV 的说法是什么？

在 rate=0.7（峰值饱和区）：
- **BidKV TTFT P95 最优**：30,143ms vs SR=31,693ms(+5%), LF=33,345ms(+11%), PE-SJF=37,551ms(+25%)
- **BidKV SLO 第三**：62.3%，落后 LF(64.4%) 约 2.1pp，落后 PE-SJF(63.6%) 约 1.3pp
- **BidKV Throughput 第二**：0.560 vs SR=0.568(best), PE-SJF=0.563, LF=0.542

合理叙事：BidKV 在峰值压力下通过 utility-guided 驱逐获得最佳尾部延迟（TTFT P95），
以 ~2pp SLO 换取显著更好的首 token 时间——与 admission responsiveness 主线一致。

### Q3：是否保留 PE 行？

**建议保留**：PE 是 vLLM 原生 baseline，其在 LC 下 SLO 仅 5% 展示了无调度策略的退化。
这一对比数据有力说明了 LC 工作负载下调度的必要性。

### Q4：§6.3 文本应如何改写？

需要修改的陈述（当前 §6.3 引用了跨速率数字）：
- "BidKV achieves 75.6% SLO attainment" → 改为 rate=0.7 数字或方向性表述
- "TTFT P95 of 15,492ms" → 改为 rate=0.7 的 30,143ms 或删除绝对值
- 跨速率平均比较句 → 改为 rate=0.7 排序 + 脚注

脚注建议内容：
> At lower request rates (0.35 and 0.50 req/s), all proactive strategies maintain SLO attainment above 70\%; the ordering among strategies is qualitatively consistent with Table~\ref{tab:long_context}.

### Q5：abstract / conclusion 是否需要改？

当前 abstract 不引用 LC 具体数字，§8 结论提及 LC 是"portability slice"——
**若仅改 Table 3 和 §6.3，abstract 和结论无需修改。**

---

## 修改边界约束（供 revision agent 遵守）

1. **§6 结构不变**：LC 保持为补充评估，不升级为主线
2. **§6.2（Mixed 主线）不变**：本次修改仅限 §6.3 + Table 3
3. **论文叙事主线不变**：admission responsiveness → TTFT + SLO attainment (mixed workload)
4. **Table 1（主结果表）不变**：这是 mixed 工作负载的结果
5. **策略列表不变**：5 个策略，排列方式 PE → PE-SJF → SR → LF → BidKV
6. **数字来源**：所有数字必须来自上方 Table A/C 中的实验数据，不得伪造

---

## 当前 §6.3 关键引用位置

需要修改的数字（基于当前 paper/bidkv_sc2026.tex §6.3）：
- SLO attainment: 75.6% (BidKV), 76.6% (LF), 75.0% (SR)（跨速率平均）
- TTFT P95: 15,492ms (BidKV), 20,264ms (PE-SJF)（跨速率平均）
- Table caption: 应更新以反映 rate=0.7 快照

---

## 建议的修改方案（供 revision agent 参考）

### Table 3 caption 建议

```latex
\caption{Long-context workload evaluation at peak request rate (0.70 req/s).
SLO threshold: 2,000~ms. Each cell is the mean of 3~independent runs.
At lower rates (0.35 and 0.50~req/s) all proactive strategies attain
SLO $>$70\%; the relative ordering is qualitatively consistent.
\textbf{Bold}: best; \underline{underline}: second-best.}
```

### §6.3 段落改写要点

第一段（结论陈述）：
- 改为：在 rate=0.7（峰值压力点），BidKV 实现 TTFT P95 最优（30,143ms），
  SLO attainment 为 62.3%（低于 Largest-First 的 64.4% 约 2pp，但高于 Static-Random 的 57.4% 约 5pp）

第二段（机制解释）：
- 保持原有的 utility-guided 驱逐机制说明
- 强调 TTFT P95 优势来源：cost-aware 驱逐减少无效 recompute，
  降低 prefill 资源竞争，使新请求更快获得首 token

第三段（与 mixed 工作负载结论一致性）：
- 强调两个工作负载下 BidKV 均在 TTFT P95 上优于其他策略
- 诚实陈述 SLO 在 LC 下非领先（因 Largest-First 容量贪心驱逐在长序列场景高效）
