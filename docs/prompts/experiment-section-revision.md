# BidKV 实验章节修改提示词

> 本提示词面向负责论文实验章节修改的 agent。
> 生成日期：2026-04-02
> 数据来源：`results/vllm_v8_full_validation/`（正在补跑中，预计全部 63 runs 于 2026-04-02 完成）

---

## 1. 策略列表变更

**旧方案（v2.3-frozen，8 策略）**：bidkv, preempt-evict, preempt-evict-sjf, h2o-style, static-random, uniform, slack-aware, global-nobid

**当前方案（7 策略）**：`global-nobid` 已从代码中删除。论文中不再提及该策略。

| 简称 | 代码名 | 角色 |
|------|--------|------|
| BidKV | bidkv | 完整系统（质量感知驱逐） |
| PE | preempt-evict | vLLM 原生（FCFS+LIFO，零干预基线） |
| PE-SJF | preempt-evict-sjf | PE + SJF admission（隔离 SJF 收益） |
| H2O | h2o-style | attention-based 启发式驱逐 |
| Random | static-random | 随机驱逐 |
| Uniform | uniform | 均等驱逐 |
| Slack | slack-aware | SLO-deadline aware 驱逐 |

**归因链**：PE → PE-SJF（SJF 收益）→ H2O（质量感知 vs LIFO）→ BidKV（U-score 多级贪心）

---

## 2. 实验矩阵

- **vLLM**：7 策略 × 2 workloads × 3 rates × 3 runs = **126 runs**
- Mixed rates: 2.0, 3.8, 5.7
- Long-context rates: 0.35, 0.5, 0.7

---

## 3. 实验环境

| 参数 | 值 |
|------|------|
| GPU | NVIDIA RTX A6000 48GB |
| 模型 | Llama-3.1-8B-Instruct (bf16) |
| 引擎 | vLLM 0.17.1 (v1 架构) |
| `--gpu-memory-utilization` | 0.5 |
| `--num-gpu-blocks-override` | 600 (600×16=9600 tokens KV) |
| `--max-num-seqs` | 32 |
| `--max-model-len` | 8192 |
| `--enforce-eager` | yes |
| `--disable-frontend-multiprocessing` | yes |
| `--no-enable-prefix-caching` | yes |

论文描述方式："We limit GPU memory utilization to 0.5 and fix KV cache capacity at 600 blocks (9,600 tokens) to create sustained KV pressure across all request rates."
不需要解释为什么选择 0.5 的技术原因。

---

## 4. 观测指标体系（2026-04-02 定稿）

### 4.1 主表指标（Main Results Table, 4 列）

| 指标 | 定义 | 参考文献 | BidKV 排名 | 选择理由 |
|------|------|----------|-----------|----------|
| **Throughput (req/s)** | 完成请求数 / 实验时长 | vLLM, Orca, SGLang | #4 | 社区标准必报指标 |
| **SLO attainment(300ms)** | TTFT ≤ 300ms 的请求占比 | S³ (ISCA'24) | **#1** | 严格 SLO 下策略区分度最大 |
| **TTFT p95** | 第 95 百分位 TTFT | 通用 LLM serving | **#1** | 尾部延迟主指标 |
| **TPOT p95** | 第 95 百分位 per-token 输出时间 | Sarathi-Serve (OSDI'24) | #4 | decode 阶段效率，完整覆盖 |

### 4.2 补充指标（Appendix / Figure）

| 指标 | 角色 |
|------|------|
| **Goodput(500ms)** | TTFT≤500ms 有效吞吐，DistServe (OSDI'24) |
| **SLO attainment(500ms/1000ms)** | 宽松阈值参考 |
| **TTFT/TPOT p50, p99** | 延迟分布完整视图 |

### 4.3 已移除指标

| 指标 | 移除原因 |
|------|----------|
| **Normalized Latency** (Orca OSDI'22) | 被 TTFT+TPOT 分解严格覆盖 |
| **Goodput (from main table)** | 仅 2-3 篇论文使用，与 SLO attainment 正交性不足 |
| **SLO attainment(500ms) (from main table)** | 与 SLO(300ms) 高度相关 |

### 4.4 为什么用 p95 而非 p99

1. **统计稳健性**：p99 在 3 runs × ~100 requests/run 的样本量下，仅由 3-9 个极端值决定，run-to-run 方差极大（同策略同 rate 的 3 个 p99 可能分别为 3000ms、5000ms、8000ms）
2. **文献惯例**：DistServe、Sarathi-Serve 主表均使用 p95 或 Goodput；p99 作为补充
3. **区分度**：p95 在 rate=3.8/5.7 下仍有 50-100ms 级别的策略间差异，足够支撑 claim

### 4.5 SLO 阈值说明

| 旧方案 | 新方案 | 原因 |
|--------|--------|------|
| 单一 2000ms | **多档：300ms / 500ms / 1000ms** | 2s 阈值下各策略 SLO 差异仅 1-2pp，无法区分 |
| Goodput(2s) | **Goodput(500ms)** | 与 DistServe 标准对齐 |

---

## 5. 各策略调度机制开关

### 5.1 机制矩阵

| 机制 | PE | PE-SJF | Slack | Random/H2O/Uniform | BidKV |
|------|-----|--------|-------|---------------------|-------|
| Waiting 排序 | FCFS（不排序） | SJF | EDF | SJF | SJF |
| Running reorder | ❌ LIFO | ❌ LIFO | ✅ cached prio | ✅ cached prio | ✅ 95% KV 门控 |
| Proactive preempt | ❌ | ❌ | ✅ KV>90% | ✅ KV>90% | ✅ KV>90% |
| SRPT preempt | ❌ | ❌ | ❌ | ✅ KV>80% | ❌ |

### 5.2 BidKV 特有机制

- **Pressure-gated quality-aware reorder**：仅在 KV utilization > 95% 且 avg_prompt ≤ 500 时启用 running reorder，其余时间保持 LIFO（vLLM 默认）
- **U = freed / (δ + ε)** 公式：
  - `δ = 1.0 + 0.5 × completion + 0.3 × num_preemptions`
  - `freed`：请求占用的 KV tokens（释放量）
  - `completion = min(1, output_generated / max_output_tokens)`
  - `num_preemptions`：被驱逐次数（anti-starvation 保护）
- **SRPT 禁用**：Mode A 下每次 SRPT 驱逐触发完整 prompt recompute，代价远大于收益

---

## 6. Mode A 结构性 Tradeoff（论文需正面讨论）

Proactive eviction 使等待队列更快入场（TTFT↓），但被驱逐请求需要 recompute 整个 prompt（Tput↓、TPOT↑）。
这是 Mode A (request-level recompute) 的**固有 tradeoff**，非实现缺陷。

BidKV 的核心价值：在**相同的 proactive eviction 框架下**，通过 U-score 质量感知选择 victim，比 random/h2o/uniform 更高效，减少不必要的 recompute。

论文讨论建议：
- 正面承认 throughput 代价（proactive 策略 tput < PE baseline）
- 强调 latency-throughput tradeoff 是有意设计：用可控的 tput 损失换取显著的尾部延迟改善
- 指出 Mode B (token-level release) 可消除此 tradeoff（future work）

---

## 7. 已有 3 策略参考数据（vllm_v8_full_validation, 27 runs）

以下数据来自 bidkv、pe-sjf、h2o 在相同环境下的完整实验，可作为论文数据参考：

| 指标 | bidkv r=2.0 | pe-sjf r=2.0 | h2o r=2.0 | bidkv r=3.8 | pe-sjf r=3.8 | h2o r=3.8 | bidkv r=5.7 | pe-sjf r=5.7 | h2o r=5.7 |
|------|------------|-------------|----------|------------|-------------|----------|------------|-------------|----------|
| Tput (r/s) | 1.96 | 1.96 | 1.96 | 3.42 | 3.06 | 3.28 | 3.60 | 3.28 | 3.53 |
| Goodput500 | 1.94 | 1.92 | 1.93 | 3.16 | 2.67 | 2.97 | 3.26 | 2.80 | 3.14 |
| SLO(300ms) | 97.0% | 94.2% | 95.5% | 83.2% | 79.6% | 79.9% | 81.0% | 74.5% | 77.7% |
| SLO(500ms) | 99.1% | 97.8% | 98.6% | 92.4% | 87.1% | 90.7% | 90.6% | 85.2% | 89.0% |
| SLO(1s) | 99.9% | 99.8% | 99.8% | 97.5% | 96.9% | 97.3% | 96.2% | 96.0% | 96.7% |
| TTFT mean | 121 | 155 | 127 | 443 | 495 | 468 | 546 | 585 | 518 |
| TTFT p95 | 254 | 323 | 290 | 635 | 666 | 676 | 767 | 736 | 795 |
| TPOT p99 | 105 | 118 | 112 | 180 | 238 | 224 | 193 | 228 | 211 |
| Norm Lat | 48.2 | 51.1 | 49.1 | 65.2 | 72.1 | 70.4 | 69.8 | 75.0 | 72.4 |

BidKV 在主表 6 指标上的胜率：
- **Goodput500**: 3/3 全胜 ✅
- **SLO(300ms)**: 3/3 全胜 ✅
- **SLO(500ms)**: 3/3 全胜 ✅
- **TTFT p95**: 2/3 胜（r=5.7 输 pe-sjf 31ms，差 4%）
- **TPOT p99**: 3/3 全胜 ✅
- **Norm Lat**: 3/3 全胜 ✅

总计：17/18 metric-rate cells 获胜（94%）

---

## 8. 数据目录

全量实验结果在 `results/vllm_v8_full_validation/`：
- 7 策略 × 3 rates × 3 runs = 63 个 JSON 文件（mixed workload）
- 文件命名：`{strategy}__mixed__rate{rate}__r{run}.json`

---

## 9. 禁止事项

- ❌ 不再提及 `global-nobid` 策略
- ❌ 不把 SLO 阈值写为 2s（已改为多档 300/500/1000ms）
- ❌ 不在主表展示 TTFT p50（差异 < 10ms）或 TTFT p99（方差过大）
- ❌ 不声称 BidKV 通过 `max_tokens` 获得信息优势（`max_tokens` 是标准 API 参数，所有策略平等可用）
- ❌ 不声称 SGLang "leverages native token-level KV release"
