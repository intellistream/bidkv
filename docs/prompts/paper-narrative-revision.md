# BidKV 论文叙事修订：从"压缩调度"到"请求调度"

## 任务目标

将 BidKV 论文中所有涉及 "compression scheduling"、"KV compression" 的叙事
修订为 **request-level preemption scheduling** 的框架。
BidKV 不做压缩——它只控制 **谁被 preempt**，底层执行仍是框架原生的 preempt + recompute。

## 核心叙事转变

### 旧叙事（已废弃）

> BidKV: A Bid-Driven KV Cache Compression Framework for LLM Serving
> 
> BidKV 通过 bid 机制让用户表达 quality preference，
> 在 KV 压力下选择性压缩 KV cache 来降低内存占用...

### 新叙事

> BidKV: Quality-Aware Request Scheduling for KV-Pressure LLM Serving
>
> BidKV 是一个 framework-portable 的 KV cache 请求调度原语。
> 核心问题：当 KV cache 压力超过阈值时，**谁应该被驱逐？**
> vLLM 默认使用 LIFO/FCFS — BidKV 通过质量感知的驱逐排序替代，
> 使得 preemption 决策基于 U = tokens_freed / (δ + ε)
>（每单位质量损失能回收多少 KV 空间）。

## 暂定标题选项

当前冻结标题（来自 claim_freeze_early.md）：
> BidKV: Compression-Aware Request Scheduling for SLO-Guaranteed LLM Serving

建议修订候选（请在论文中采用最终版本）：
1. BidKV: Quality-Aware Preemption Scheduling for KV-Pressure LLM Serving
2. BidKV: Bid-Driven Request Scheduling for SLO-Guaranteed LLM Serving
3. BidKV: Who Gets Evicted? Quality-Aware KV Scheduling for LLM Serving

## 关键术语映射

| 旧术语                  | 新术语                              | 说明                                        |
| ----------------------- | ----------------------------------- | ------------------------------------------- |
| compression scheduling  | preemption scheduling / request scheduling | BidKV 不做 token-level 压缩       |
| compress                | preempt / evict                     | 执行动作是驱逐+重算，不是压缩               |
| compression ratio       | (不使用)                            | Mode A 下无压缩比概念                       |
| quality degradation     | recompute cost                      | Mode A 下输出质量不受影响（完整 recompute） |
| BidKV framework         | BidKV scheduling primitive          | BidKV 是调度原语，不是框架                   |
| KV token-level release  | request-level preemption            | 粒度是请求级，不是 token 级                  |
| compression executor    | (不暴露给论文)                      | Mode B 代码已废弃                            |

## BidKV 核心机制描述

### Algorithm 1: Quality-Aware Victim Selection

```
Input: running_requests R, needed_tokens N
Output: victim_set V

For each request r_i in R:
    scores = H2OScoring(r_i.token_ids)            # token importance
    For each compression_level l in {0.2, 0.4, 0.6}:
        tokens_freed = |r_i| × l
        quality_delta = EstimateDelta(scores, tokens_freed)
        U_i,l = tokens_freed / (quality_delta + ε)  # utility
        Create bid(r_i, l, U_i,l)

Pool all bids, sort by U descending
Greedy select: Σfreed ≥ N, Σδ ≤ budget, 1 bid per request
V = {r_i | bid(r_i) selected}
Return V  → framework executes native preempt(r_i)
```

### 7 策略调度分化表（vLLM 主实验）

| 层面               | preempt-evict | slack-aware | random/h2o/uniform/nobid | bidkv                     |
| ------------------ | ------------- | ----------- | ------------------------ | ------------------------- |
| Waiting 排序       | FCFS (无排序) | EDF (到达序) | SJF (prompt_tokens)      | SJF (prompt_tokens)       |
| Running 排序       | LIFO (无排序) | cached prio | cached prio              | cached prio               |
| select_victims()   | N/A           | slack-based | 各自启发式               | **U = r/(δ+ε)** 质量感知 |
| SRPT 主动驱逐      | ❌            | ❌          | ✅ (同等估算)            | ✅ (同等估算)             |
| Proactive preempt  | ❌            | ✅          | ✅                       | ✅                        |

**BidKV 的唯一分化点**：`select_victims()` 中使用完整 scoring→bid→pool→solver
管线计算 U = tokens_freed / (quality_delta + ε)，实现质量感知的 preemption 排序。

所有策略共享相同的执行机制：vLLM 原生 `_preempt_request()` → recompute from scratch。

## Scenario A 核心主张

Claim（不变）：bid-driven selection ≥ system-inferred (Global-NoBid)

归因链：
1. Preempt-Evict（零信息下界）→ 任何有信息策略
2. Static-Random → H2O-Style → Global-NoBid：信息量递增
3. Global-NoBid → BidKV：同一评分+算法，唯一差异是 bid protocol 的增量价值
4. Slack-Aware：调度信息（SLO deadline）但无 quality 信息

Scenario A/B 切换规则（v2.3 冻结）：
- 保留 A：BidKV vs Global-NoBid 的 SLO attainment Δ_avg ≥ 10% 且所有 workload ΔM_w > 0
- 降级 B：Δ_avg < 10% 或任一 workload ΔM_w ≤ 0

## Figure 列表与语义

| Figure  | 内容                                  | Mode A 状态                               |
| ------- | ------------------------------------- | ----------------------------------------- |
| Table 1 | 7策略 × SLO violation + P99 TTFT + throughput | ✅ 不变，重命名策略描述为 scheduling      |
| Fig 3   | Rate sweep: throughput + TTFT P95     | ✅ 不变                                   |
| Fig 3ab | Quality 指标 (ROUGE-1, EM/F1)        | ⛔ 移出主线（RULE FIG3AB-FREEZE）         |
| Fig 4   | Oracle gap 分析                       | ✅ 不变                                   |
| Fig 5   | Compression coverage rate             | ✅ 重命名为 "Preemption event rate"       |
| Fig 6   | Surrogate budget sensitivity          | ✅ δ budget 对调度性能的影响              |
| Table 2 | Cross-framework consistency (SGLang)  | ✅ 3 策略方向一致性                       |
| Fig 7   | Cross-platform visualization          | ✅ 不变                                   |

### Fig 5 语义修订

旧名：Compression coverage rate（压缩覆盖率）
新名：Preemption event analysis（驱逐事件分析）

内容：
- 每个策略的 proactive preemption 触发频率
- 每次 preemption 释放的平均 KV 量
- Preemption-to-completion ratio（被驱逐 vs 成功完成的请求比例）

### Fig 6 语义说明

δ budget 是调度器的 surrogate quality budget 参数。
在 Mode A 下，δ 不直接对应输出质量损失（因为被驱逐的请求会完整 recompute）。
δ 控制的是调度激进程度：低 δ = 保守（少驱逐），高 δ = 激进（多驱逐以换吞吐量）。

论文必须明确说明：
> "δ is a surrogate budget parameter that controls scheduling aggressiveness,
> not a direct measure of output quality degradation. Under Mode A (recompute
> fallback), preempted requests are fully recomputed, preserving output quality."

## Mode B 论述

论文中 Mode B 只在以下位置提及：
1. §6 Discussion：提到 Mode B (tail truncation) 作为 future work
2. Appendix（如果有）：Mode B 实验数据（如果 issue #054 完成）

主线论文不依赖 Mode B 的任何数据或 claim。

## 双平台（vLLM + SGLang）叙事

| 维度        | vLLM（主量化平台）                    | SGLang（portability slice）           |
| ----------- | ------------------------------------- | ------------------------------------- |
| 策略数      | 7（完整归因链）                       | 3（精简方向验证）                      |
| 矩阵        | 7 × 2 × 3 × 3 = 126 runs            | 3 × 2 × 3 × 3 = 54 runs             |
| 执行语义    | request-level preempt + recompute     | request-level 调度（对称 vLLM）        |
| 产出物      | Table 1 + Fig 3/4/5/6                | Table 2 + Fig 7                      |
| Claim 角色  | Scenario A 核心数据                   | 方向一致性 + portability               |

SGLang 的 directional consistency 要求：
- DC-1a: BidKV ≥ Preempt-Evict
- DC-1b: BidKV ≥ Slack-Aware

## 论文各节修订要点

### Abstract
- "compression scheduling" → "quality-aware preemption scheduling"
- 强调 BidKV 是 scheduling primitive，不是 compression framework
- 核心问题：**who should be evicted** under KV pressure

### §1 Introduction
- 问题定义：KV cache 内存不足时的 preemption 决策
- vLLM default = LIFO (no intelligence)
- BidKV = quality-aware victim selection via U = r/(δ+ε)
- Contribution: scheduling primitive, not compression algorithm

### §2 Background
- KV cache management in LLM serving
- Preemption strategies in vLLM/SGLang
- 不提 token-level compression

### §3 Design
- BidKV scheduling primitive
- Bid interface: quality preference signal
- Scoring → Bid → Pool → Solver pipeline
- Algorithm 1: victim selection
- Integration: monkey-patch scheduler, framework-native execution

### §4 Experimental Setup
- 7 strategies (表格)
- vLLM Mode A: request-level preempt + recompute
- Frozen traces, rates, SLO definitions

### §5 Evaluation
- Table 1, Fig 3-7
- 归因链分析
- Scenario A/B 评估

### §6 Discussion
- Mode A limitations (recompute overhead)
- Mode B future work (tail truncation for quality-throughput tradeoff)
- Generalization to other frameworks

## 约束

### 绝对禁止
- ❌ 修改实验方向 / 策略列表 / figure 语义（已冻结）
- ❌ 声称 BidKV 做 "KV compression"（它只控制驱逐排序）
- ❌ 声称 Mode A 有 quality degradation（recompute 完整重算，无质量损失）
- ❌ 声称 BidKV 通过 max_tokens 获得信息优势（所有策略平等可用）
- ❌ 在 vLLM 主线中引用 SGLang token-level 数据
- ❌ 声称 SGLang "leverages native token-level KV release"（除非经过验证）

### 必须遵守
- ✅ Mode A 下所有输出质量相同（完整 recompute）— 明确声明
- ✅ δ 是 surrogate parameter，不是 quality measure — 明确声明
- ✅ 失败计入结果（OOM / timeout / crash 全记录）
- ✅ Frozen trace (seed=42) 跨所有策略共享
- ✅ Rate 冻结后不可基于策略表现调整

## 参考文件

- `results/claim_freeze_early.md` — Claim freeze + Scenario A/B 规则
- `docs/experiment_protocol.md` — 主实验方案 v2.3-frozen
- `docs/baseline-specs.md` — 7 baseline 策略规格
- `.github/copilot-instructions.md` — 项目完整约束
- `src/bidkv/adapters/vllm/scheduler_hook.py` — 调度实现（理解 strategy 分化）
- `src/bidkv/baselines/` — 7 个策略的具体实现
