# GPT 论文修改评审提示词

## 角色

你是一位系统领域（SIGCOMM / OSDI / SC）资深审稿人。请对以下论文摘要+§1 的修改进行评判。论文投稿目标：**SC 2026**。

---

## 1. 项目背景

**BidKV** 是一个 framework-portable 的 KV cache 请求调度原语，解决的核心问题：当 LLM serving 的 KV cache 压力超过阈值时，**谁应该被驱逐（victim selection）**？

vLLM 默认使用 LIFO/FCFS — BidKV 通过 utility-guided victim selection 替代，使 preemption 决策基于 **U = tokens_freed / (disruption_cost + ε)**。

BidKV **不做压缩，不做 token-level 操作**。它只控制"谁被 preempt"，底层执行仍是框架原生的 preempt + recompute（vLLM）或 request-level eviction（SGLang）。

---

## 2. 实验数据（驱动叙事的关键事实）

### 实验设置
- GPU: NVIDIA RTX A6000 48GB
- 模型: Llama-3.1-8B-Instruct (bf16)
- 引擎: vLLM 0.17.1 (v1 架构)
- KV 限制: 600 blocks × 16 = 9600 tokens（人为制造 KV 压力）
- 工作负载: ShareGPT mixed traces, rates = {2.0, 3.8, 5.7} req/s
- 每组 3 runs, 共 7 策略 × 3 rates × 3 runs = 63 runs

### v8 全量结果 — Cross-Rate Average（7 策略 × 3 rates × 3 runs 取均值）

| Strategy | Throughput (req/s) | SLO(300ms) % | TTFT P95 (ms) | TPOT P95 (ms) |
|----------|-----------|---------|----------|----------|
| **bidkv** | **2.99** (#4) | **87.1** (#1) | **554** (#1) | **96.4** (#4) |
| static-random | 3.21 (#1) | 87.0 (#2) | 1076 (#5) | 86.0 (#1) |
| uniform | 3.21 (#2) | 86.9 (#3) | 1069 (#4) | 86.7 (#2) |
| h2o-style | 2.92 (#6) | 84.4 (#4) | 584 (#3) | 100.1 (#6) |
| preempt-evict-sjf | 2.77 (#7) | 82.8 (#5) | 572 (#2) | 129.4 (#7) |
| slack-aware | 3.05 (#3) | 72.4 (#6) | 4023 (#6) | 93.2 (#3) |
| preempt-evict | 2.98 (#5) | 72.2 (#7) | 5241 (#7) | 98.3 (#5) |

### 关键观察

1. **BidKV 核心优势**：SLO(300ms) #1（87.1%）、TTFT P95 #1（554ms）
2. **BidKV 非领先指标**：Throughput #4（2.99 vs best 3.21, -7%）、TPOT P95 #4（96.4 vs best 86.0, -12%）
3. **vs vLLM 原生 (preempt-evict)**：SLO +14.9pp, TTFT -89.4%, Throughput +0.3%, TPOT -1.9%
4. **Tradeoff 特征**：BidKV 用适度的 throughput/TPOT 代价换取显著的 TTFT/SLO 改善
5. **Rate=2.0（低压）下 BidKV 全 4 指标 #1**；高压下 Throughput/TPOT 排名下降

---

## 3. 修改原则与理由

### 核心叙事转向

**旧叙事**：BidKV 是"全面优化"系统，声称 Goodput(500ms) +27.4%（已移除指标）
**新叙事**：BidKV 是 **admission-responsiveness-first** 策略

**原因**：
- 数据不支持"全面优化"声称 — Throughput 和 TPOT 排名第 4
- 数据强烈支持"准入响应性"聚焦 — SLO 和 TTFT 均排名第 1
- 系统论文更看重 honest tradeoff 声明，而非夸大的全面优势

### 6 条具体原则

1. **Target scenario upfront**：摘要前两句必须定义目标场景（KV 压力下的 admission responsiveness），不能让读者猜
2. **Causal chain explicit**：§1 必须显式构建因果链 — poor victim → 无效 recompute → TTFT 飙升 → SLO 违约
3. **Tradeoff must be declared**：不隐瞒 Throughput/TPOT 的非领先表现，显式声明 tradeoff
4. **Data claims prioritized**：先报告核心指标（TTFT -89%, SLO +14.9pp），Throughput 作为补充
5. **No overclaiming**：不声称"全面领先"，不暗示 throughput 优化
6. **Derive, don't assert**：¶3 的两个需求从 ¶2 的结论推导而来，而非直接断言

---

## 4. 修改前后对比

### 4.1 摘要 (Abstract)

**修改前**（旧版，聚焦 Goodput）：
> Large language model~(LLM) serving must manage a continuously changing mix of concurrent requests under limited GPU memory. When KV-cache demand exceeds available capacity, the serving engine must reclaim active KV state to admit waiting requests---a process whose efficiency depends on \emph{which} requests are selected as reclamation victims. Existing scheduling policies rely on coarse request-order heuristics (LIFO, FCFS) or independent per-request scoring rules, lacking both an explicit cost signal for reclamation and a systematic way to coordinate victim selection across the batch. We present BidKV, a bid-based scheduling abstraction for active-KV reclamation. Each active request produces a bid encoding its recoverable KV capacity and a surrogate disruption estimate; a coordinated solver selects victims that free the required capacity at minimum aggregate cost. Because scorer and solver communicate only through the bid interface, the scoring mechanism can be replaced without modifying the selection algorithm or runtime integration. The current prototype instantiates disruption cost from request-lifecycle proxies (completion progress, prompt length, preemption history) under recompute-fallback semantics. BidKV integrates non-invasively with vLLM and SGLang via a portable adapter layer. Evaluated with Llama-3.1-8B-Instruct on an NVIDIA RTX A6000, BidKV improves Goodput(500\,ms) by 27.4\% and SLO(300\,ms) attainment by 14.9\,pp over vLLM's native policy under ShareGPT traces with calibrated arrival rates.

**修改后**（新版，聚焦 admission responsiveness）：
> When KV-cache demand exceeds GPU memory capacity during online LLM serving, serving engines reclaim cache from active requests to admit waiting ones. Poor victim selection wastes recomputation, delays admission of queued requests, and degrades first-token latency---a metric directly visible to users. Existing policies rely on coarse request-order heuristics (LIFO, FCFS) or independent per-request scoring rules, lacking both explicit reclamation-cost signals and cross-request coordination. We present BidKV, a utility-guided reclamation policy that improves \emph{admission responsiveness} under KV pressure. Each active request produces a bid encoding recoverable capacity and estimated disruption cost; a coordinated solver selects victims that free the required capacity at minimum aggregate cost. The current prototype instantiates disruption cost from request-lifecycle proxies (completion progress, prompt length, preemption history) under recompute-fallback semantics. BidKV integrates non-invasively with vLLM and SGLang via a portable adapter layer. Evaluated with Llama-3.1-8B-Instruct on an NVIDIA RTX A6000 under ShareGPT traces, BidKV reduces P95 TTFT by 89\% and improves 300\,ms SLO attainment by 14.9\,pp over vLLM's native policy, at comparable throughput.

**变化要点**：
- 开头直接入场景（KV cache 压力 → victim selection）
- 第二句建立因果：poor victim → recomputation waste → admission delay → TTFT degradation
- 将 BidKV 定位从"bid-based scheduling abstraction"调整为"utility-guided reclamation policy that improves admission responsiveness"
- 数据声称从 Goodput +27.4% 改为 TTFT -89% + SLO +14.9pp + "at comparable throughput"（诚实的 tradeoff 表述）

### 4.2 §1 ¶2 尾部（新增因果链）

**修改前**（¶2 结尾）：
> ...whereas reclaiming a request that has only recently entered decoding may free similar capacity at much lower system disruption. Active-KV reclamation is therefore a cross-request victim-selection problem under heterogeneous cost.

**修改后**：
> ...whereas reclaiming a request that has only recently entered decoding may free similar capacity at much lower system disruption. Poor victim choices can also cascade: recomputation re-inflates KV demand, which in turn triggers further reclamation---a cycle that keeps KV capacity occupied by recomputing victims rather than available for waiting requests. The direct consequence is admission-latency degradation: queued requests wait longer to receive their first token, driving up time-to-first-token (TTFT) and eroding service-level objective (SLO) attainment. Active-KV reclamation is therefore a cross-request victim-selection problem whose quality directly governs admission responsiveness under sustained KV pressure.

**变化要点**：
- 新增 cascade 机制说明：bad victim → recompute → re-inflate KV → more eviction
- 显式连接到 TTFT 和 SLO：这个循环的直接后果是准入延迟恶化
- 结尾从"heterogeneous cost"升级为"whose quality directly governs admission responsiveness"

### 4.3 §1 ¶3 开头（从 ¶2 推导而非断言）

**修改前**：
> Two capabilities are missing from current scheduling policies. First, the scheduler lacks...

**修改后**：
> Because reclamation costs are heterogeneous, the scheduler needs explicit per-request cost signals to distinguish cheap victims from expensive ones; because selection is cross-request, these signals must feed into coordinated batch-level selection rather than independent per-request rules. No existing system provides both.

**变化要点**：两个需求（per-request cost signal + cross-request coordination）从 ¶2 的结论（成本异构 + 跨请求问题）逻辑推导出来，而非直接断言"Two capabilities are missing"。

### 4.4 §1 ¶5（收益具体化 + tradeoff 声明）

**修改前**：
> ...the policy does not alter final outputs; the differentiation lies entirely in scheduling efficiency, whose benefit is operating-regime dependent: under sustained mixed-length KV pressure, utility-guided selection provides substantial gains; under low pressure, victim selection rarely activates and the benefit diminishes.

**修改后**：
> ...the policy does not alter final outputs; the differentiation lies entirely in scheduling efficiency. The primary benefit is improved admission responsiveness: by selecting victims whose reclamation frees adequate capacity at low recompute cost, BidKV reduces the time waiting requests spend queued for KV allocation, yielding lower TTFT and higher SLO attainment. This comes at a modest cost in throughput and per-token decode latency, a tradeoff that favors latency-sensitive deployments. The benefit is operating-regime dependent: under sustained mixed-length KV pressure, utility-guided selection provides substantial gains; under low pressure, victim selection remains governed by the framework's native ordering, with BidKV's reorder activating only under sustained high-pressure conditions.

**变化要点**：
- 将"substantial gains"具体化为"lower TTFT and higher SLO attainment"
- 新增显式 tradeoff 声明：throughput 和 TPOT 有适度代价
- 低压场景措辞更精确："remains governed by framework's native ordering"

### 4.5 C4（措辞软化）

**修改前**：
> ...enabling structured attribution across these co-varying dimensions.

**修改后**：
> ...supporting structured attribution across these co-varying dimensions.

**变化要点**：从"enabling"（暗示是充分条件）改为"supporting"（表述为支持而非保证）。

---

## 5. 请评判以下问题

### A. 叙事一致性
1. 摘要、¶2、¶5 是否始终围绕"admission responsiveness"这一核心主张展开？是否有任何段落偏离或自相矛盾？
2. 因果链（poor victim → recompute cascade → TTFT/SLO degradation）是否每一步都成立？是否有逻辑跳跃？
3. "at comparable throughput" 这一表述是否恰当？实际数据：BidKV Throughput 2.99 vs best 3.21（-7%）、vs PE 2.98（+0.3%）。

### B. Tradeoff 声明的诚实度
4. ¶5 的 tradeoff 声明（"modest cost in throughput and per-token decode latency"）是否足够诚实？-7% throughput、-12% TPOT 算 "modest" 吗？
5. 摘要中只报告 TTFT 和 SLO（核心优势指标）+ "at comparable throughput"，是否构成 cherry-picking？是否需要在摘要中也报告 TPOT？

### C. §1 结构与可读性
6. ¶2 新增的 cascade + TTFT/SLO 段落是否过长？是否影响 ¶2 的聚焦（从"victim selection problem"角度）？
7. ¶3 从 ¶2 推导需求的方式（"Because...the scheduler needs..."）是否自然？还是显得过于刻意？
8. 整个 §1 是否流畅？五个段落的逻辑递进是否清晰？

### D. Claims 与数据匹配
9. "reduces P95 TTFT by 89%" — 基线是 vLLM native (preempt-evict)，554ms vs 5241ms。这个百分比计算是否正确？表述是否可能引起误解（例如 cross-rate average 的 P95 含义）？
10. "improves 300ms SLO attainment by 14.9pp" — 87.1% vs 72.2%。pp (percentage points) 表述是否恰当？
11. BidKV 实际排名 SLO #1, TTFT #1, Throughput #4, TPOT #4。论文是否应该在 §1 中提及具体排名而非仅比较 baseline？

### E. 审稿人视角的风险
12. 一个严格的 reviewer 看到这个摘要+§1 后，最可能的前 3 个质疑点是什么？
13. "admission responsiveness" 是否是一个被系统社区广泛认可的概念？是否需要更多定义或引用？
14. 标题中没有出现 "admission responsiveness"，摘要中头两句才引出。这个引入速度够快吗？

### F. 与竞争对手的对比
15. static-random 的 SLO 和 BidKV 仅差 0.1pp（87.0 vs 87.1），但 static-random 的 Throughput 显著更高（3.21 vs 2.99）。审稿人可能问：为什么不直接用 random eviction？§1 是否需要预防性地回应这个问题？
16. 论文声称的 +14.9pp SLO 改善是 vs preempt-evict（最差基线之一）。vs 第二名（static-random）仅 +0.1pp。这是否构成潜在的 reviewer 攻击点？

---

## 6. 当前 §1 全文

（供评审参考，五个段落 + 贡献列表）

```latex
\section{Introduction}\label{sec:intro}

% ¶1: Setting — KV cache 是动态资源
Online large language model~(LLM) serving must sustain a continuously changing
population of concurrent requests on limited GPU memory.  In production
workloads, requests are highly heterogeneous in prompt length, generation
budget, and arrival pattern; they also progress asynchronously as some requests
are still in prefill while others are already deep into
decoding~\cite{Kwon2023, Yu2022, Zheng2024lmsys}.  Under such mixed-length,
bursty workloads, the key-value~(KV) cache becomes a dominant runtime
resource~\cite{Vaswani2017, Pope2022}: new arrivals
allocate KV state, ongoing requests expand it token by token, and completed or
preempted requests release it at different points in their lifecycle.  Aggregate
KV demand therefore fluctuates on short timescales and can repeatedly exceed
available capacity during normal operation, forcing the serving engine to
reclaim KV state from active requests to make room for others.

% ¶2: Problem — victim selection + 因果链
The central question in this setting is not whether KV capacity must be
reclaimed, but which active request should be chosen as the \emph{victim}.
Two requests may occupy comparable amounts of KV memory while imposing very
different costs when interrupted.  Under the \emph{recompute-fallback}
execution model used by current serving engines~\cite{Kwon2023}, reclaiming an
active request releases its KV state but forces the request to restart through
the framework's native recovery path when it is rescheduled.  The cost of reclamation therefore
depends not only on how much memory is freed, but also on how much useful
computation is discarded and must later be redone.  Reclaiming a request that is
near completion can waste substantial prior work and incur expensive
recomputation, whereas reclaiming a request that has only recently entered
decoding may free similar capacity at much lower system disruption.  Poor victim
choices can also cascade: recomputation re-inflates KV demand, which in turn
triggers further reclamation---a cycle that keeps KV capacity occupied by
recomputing victims rather than available for waiting requests.  The direct
consequence is admission-latency degradation: queued requests wait longer to
receive their first token, driving up time-to-first-token~(TTFT) and eroding
service-level objective~(SLO) attainment.  Active-KV reclamation is therefore a
cross-request victim-selection problem whose quality directly governs admission
responsiveness under sustained KV pressure.

% ¶3: Gap — 现有方法缺什么
Because reclamation costs are heterogeneous, the scheduler needs
\emph{explicit per-request cost signals} to distinguish cheap victims from
expensive ones; because selection is cross-request, these signals must feed
into \emph{coordinated batch-level selection} rather than independent
per-request rules.  No existing system provides both.  Current scheduling
heuristics---LIFO preemption~\cite{Kwon2023}, FCFS
admission~\cite{Yu2022}, EDF-style waiting
policies~\cite{Gujarati2020}---determine victim order without any
per-request cost signal; broader serving innovations such as chunked
prefills~\cite{Agrawal2024sarathiserve} and prefill--decode
disaggregation~\cite{Zhong2024, Patel2024} advance throughput and resource
placement but inherit the same order-based victim selection, leaving the
signal gap intact.  Token-level scoring methods such as
H2O~\cite{Zhang2023h2o} and StreamingLLM~\cite{Xiao2024} do produce
request-specific importance signals, yet each applies a fixed scoring rule
independently---without coordination across the batch, a locally reasonable
per-request choice can be globally costly.  What is missing is a
\emph{signal pathway}: a structured interface through which per-request
reclamation sensitivity can be surfaced to the scheduler and used for
coordinated cross-request victim selection.

% ¶4: Solution — BidKV 架构
In this study we present \bidkv, a \emph{bid-based scheduling abstraction} for
active-KV reclamation.  The key idea is to make reclamation sensitivity an
explicit, structured signal rather than an implicit consequence of scheduling
order.  Each active request produces a \emph{bid} that summarizes the KV
capacity recoverable upon reclamation and a surrogate disruption estimate
reflecting the relative penalty of reclaiming that request.  Given bids from all
active requests, the scheduler solves a constrained cross-request selection
problem: choose the victim set that recovers the required capacity while
minimizing aggregate estimated disruption.  In this way, \bidkv turns victim
selection from an opaque runtime behavior into an explicit interface between a
\emph{scoring layer}---which estimates per-request reclamation
sensitivity---and a \emph{coordinated selection layer}---which makes
batch-level trade-offs across candidates.  Because scorer and solver
communicate only through the bid interface, \bidkv is
scorer-agnostic: the scoring mechanism can be replaced without
modifying the selection algorithm or runtime integration.

% ¶5: Implementation + Tradeoff
In the current prototype, the disruption estimate is instantiated from
request-lifecycle features (completion progress, prompt length, preemption
history) under recompute-fallback semantics~\cite{Kwon2023}; the same
architecture is designed to admit richer scoring signals (\eg
attention-based~\cite{Zhang2023h2o, Xiao2024}) without changing the solver.
\bidkv integrates non-invasively with both vLLM~\cite{Kwon2023} and
SGLang~\cite{Zheng2024sglang} via a portable adapter layer that fully reuses
each framework's native reclamation and recovery paths---the fact that both
integrations require no source-code modification confirms that the bid
interface is cleanly separated from framework-specific scheduling internals.
\bidkv controls \emph{which} request is reclaimed, not \emph{how}.  Under
recompute-fallback execution, the policy does not alter final outputs; the
differentiation lies entirely in scheduling efficiency.  The primary benefit is
improved admission responsiveness: by selecting victims whose reclamation
frees adequate capacity at low recompute cost, \bidkv reduces the time
waiting requests spend queued for KV allocation, yielding lower TTFT and
higher SLO attainment.  This comes at a modest cost in throughput and
per-token decode latency, a tradeoff that favors latency-sensitive
deployments.  The benefit is operating-regime dependent: under sustained
mixed-length KV pressure, utility-guided selection provides substantial
gains; under low pressure, victim selection remains governed by the
framework's native ordering, with \bidkv's reorder activating only under
sustained high-pressure conditions.

% Contributions
\paragraph{Contributions.}
\begin{enumerate}[leftmargin=*,nosep]
\item \textbf{Bid-based reclamation interface.}
  We introduce a structured bid abstraction that allows per-request
  reclamation sensitivity to be surfaced to the runtime scheduler, replacing
  implicit victim ordering rules with an explicit decision interface.

\item \textbf{Scorer-agnostic coordination architecture.}
  We design a decoupled scoring-to-selection pipeline that separates
  request-level disruption estimation from coordinated cross-request victim
  selection, so that the scoring strategy and the selection algorithm can
  evolve independently.

\item \textbf{Portable runtime integration.}
  We realize \bidkv as a non-invasive scheduling layer with a
  \texttt{Framework\-Adapter} abstraction, demonstrating integration with
  vLLM and SGLang that reuses each framework's native reclamation and
  recovery mechanisms without source-code modification.

\item \textbf{Structured evaluation under controlled conditions.}
  We evaluate seven strategies that vary across scheduling dimensions
  (admission ordering, running-queue reordering, victim selection logic,
  and proactive reclamation) under frozen request traces and calibrated
  arrival rates, supporting structured attribution across these co-varying
  dimensions.
\end{enumerate}
```

---

## 7. 输出格式要求

请按以下格式逐条回答 A-F 的 16 个问题，每条给出：
- **判断**：✅ 合理 / ⚠️ 有风险 / ❌ 需修改
- **理由**：1-3 句话解释
- **建议**（如需修改）：具体修改建议

最后给出一个整体评价（Overall Assessment），包括：
- 这个叙事方向的成功概率（SC 2026 审稿人视角）
- 最需要优先修复的 Top-3 风险点
- 不需要修改的强项
