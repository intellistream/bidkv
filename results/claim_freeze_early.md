# Early Claim Freeze Report (#035a)

**日期**: 2026-03-14 (初版) → 2026-03-19 (v2.3-frozen)
**Gate**: Gate-A (Phase B 完成后)
**决策人**: PI + Agent (总负责)
**状态**: ✅ v2.3-frozen — reviewer APPROVED 2026-03-19

---

## 确认的 Claim 方向

**Scenario A — quality-aware compression scheduling**

### 论文暂定标题

> BidKV: Compression-Aware Request Scheduling for SLO-Guaranteed LLM Serving

### 核心主张

BidKV = 用户 bid 显式传递 quality preference（surrogate signal），
bid-driven scheduling ≥ 系统推断（Global-NoBid），
主动压缩 ≥ 被动驱逐（Preempt-Evict）。

### 选择理由

1. **归因链完整支撑**：8 策略归因链（Preempt-Evict → H2O-Style → Global-NoBid → BidKV）
   专门为证明 bid 机制增量价值而设计
2. **Global-NoBid 天然劣势**：`U_sys = r / (δ_H2O + ε)` 完全依赖 H2OScoring proxy，
   在异质 workload 下无法区分用户对质量的差异化需求
3. **实验设计可主动倾斜**：heterogeneous quality sensitivity workload 是 bid 最佳展示场景
4. **安全网内置**：若 BidKV vs Global-NoBid <10%，可退回 Scenario B（compressive scheduling
   primitives），无需任何工程改动

### 保底方案（Scenario B）

如果实验数据显示 BidKV 与 Global-NoBid 差异不够大，退回 Scenario B：
- 主张调整为 compressive scheduling primitives（主动压缩 ≥ 被动驱逐）
- 论文侧重 compressive vs non-compressive 对比
- #035 issue 已有完整 Scenario B 转换方案

### Scenario A / B 切换的形式化规则 (RULE SCENARIO-SWITCH, v2.3)

| 项目           | 定义                                                                 |
| -------------- | -------------------------------------------------------------------- |
| **比较对象**   | BidKV vs Global-NoBid                                                |
| **主判定指标 M** | SLO attainment rate (%)（满足 TTFT < 2000ms 的请求占成功请求比例）  |
| **辅助参考**   | throughput (req/s)                                                   |
| **判定数据集 S** | { (workload, rate_mid) \| workload ∈ {Mixed, Long} }，每个取 3 runs mean |
| **聚合方式**   | ΔM_w = (M_BidKV(w) − M_GlobalNoBid(w)) / M_GlobalNoBid(w) × 100%；Δ_avg = mean(ΔM_w) |
| **保留 Scenario A** | Δ_avg ≥ 10% (relative) **且** 所有 workload 的 ΔM_w > 0           |
| **降级 Scenario B** | Δ_avg < 10% **或** 任一 workload 的 ΔM_w ≤ 0                      |
| **辅助检查**   | SLO 通过但 throughput Δ < 0 → 保留 A 但论文须注明 throughput trade-off |
| **触发时间**   | Phase 5 优先级 2 完成后（Global-NoBid 数据可用后）立即评估           |

---

## 锁定的实验 Figure 列表

v2.3 修订：Mode A 下 Fig 3a/b 移出主线，Fig 6 语义限定为 surrogate budget。

| Figure   | 内容                                           | 优先级                 | A/B 通用？ | Mode A 状态              |
| -------- | ---------------------------------------------- | ---------------------- | ---------- | ------------------------ |
| Table 1  | 8-baseline SLO violation + P99 TTFT+throughput | **Mandatory**          | ✅          | ✅ 不变                  |
| Fig 3    | Rate sweep: throughput + TTFT P95 curves       | **Mandatory**          | ✅          | ✅ 不变 (§11.2)          |
| Fig 3a/b | Quality 指标 (ROUGE-1, EM/F1)                  | Mode B / Appendix      | ❌          | ⛔ 移出主线 (RULE FIG3AB-FREEZE) |
| Fig 4    | Oracle gap 分析                                | **Mandatory**          | ✅          | ✅ 不变                  |
| Fig 5    | Compression coverage rate                      | **Mandatory**          | ✅          | ✅ 不变                  |
| Fig 6    | Surrogate budget sensitivity                   | **Mandatory** (for A)  | ⚠️          | ⚠️ 语义限定为 surrogate (RULE FIG6-DEFAULT) |
| Table 2  | Cross-framework directional consistency        | **Mandatory**          | ✅          | ✅ 不变                  |
| Fig 7    | Cross-platform consistency 可视化              | **Mandatory**          | ✅          | ✅ 不变                  |

### RULE FIG3AB-FREEZE

Under vLLM Mode A (recompute fallback), Fig 3a/b (quality metrics ROUGE-1, EM/F1) is removed from
the Mandatory figure list. Mode A recompute guarantees output equivalence — quality metrics are
identical across all strategies by construction, making quality figures trivial.

Fig 3a/b MAY be reinstated if:
- (a) Mode B tail truncation is stable and produces real quality degradation data, OR
- (b) A cross-strategy quality evaluation is added as an Appendix enhancement.

### RULE FIG6-DEFAULT

Under vLLM Mode A, Figure 6 is interpreted by default as **surrogate budget sensitivity** —
how the scheduler's performance varies as the surrogate δ budget parameter changes —
rather than a task-level quality–performance tradeoff curve.

升级条件（仅在全部满足时可升级）：
1. Mode B tail truncation 在 Gate-B 前稳定
2. 独立 task-level quality evaluation (ROUGE-1 / EM) 已跑完
3. surrogate δ → quality calibration 通过 (Pearson r ≥ 0.7 或 Spearman ρ ≥ 0.7)

---

## Gate-A 工程前置条件确认

- [x] bidkv 包可安装，零 sagellm 依赖（`dependencies = []`）
- [x] vLLM adapter + SGLang adapter 基本功能通过（67 tests）
- [x] 7 baseline + BidKV + Oracle 全部可运行（63 tests，8 策略 via Registry）
- [x] Baseline Spec 预注册（`docs/baseline-specs.md`，110 行）
- [x] Candidate-universe consistency 有专门测试
- [x] 初步 claim 方向已锁定 — **Scenario A**
- [x] 实验 Figure 列表已锁定（6 项，5 Mandatory 全场景 + 1 Mandatory for A）
- [x] #043 interface reconciliation 完成（357 tests all pass）

---

## Phase C 实验设计指引

基于 Scenario A claim，Phase C 实验应包含：
1. **异质 workload**：混合高/低 quality sensitivity 请求（展示 bid 差异化价值）
2. **压力梯度**：70%/80%/90% KV 占用率（展示 BidKV 在高压下的优势）
3. **SLO 多样性**：不同 deadline 约束（展示 quality-aware 调度意义）

---

## v2.0 修订 (2026-03-19)：执行模式变更

### 问题

所有 BidKV 注入策略在 long_context workload 下因 `_free_tail_blocks()` partial
eviction 导致 vLLM EngineCore crash。根因：vLLM v1 公开 KV 生命周期不支持
partial block release，插件层直接操作 `block_pool` / `coordinator` 内部状态
导致 block ownership violation + CUDA memory corruption。

详见：`bidkv/docs/vllm-route-redesign.md`

### 决策

采用双执行模式方案：
- **Mode A (P0)**：Recompute fallback — BidKV selection + vLLM native preempt/recompute
- **Mode B (P1)**：Tail truncation — 最小 vLLM 内核扩展（~60 LOC），增强项

### 对 Claim 的影响

- Scenario A 核心主张不变：bid-driven selection ≥ system-inferred
- Mode A 下"compression"等价于"bid-guided preemption"
- Quality 指标调整：Mode A 中输出质量不受影响（完整 recompute），Fig 3a/b
  已从 Mandatory 移至 Mode B / Appendix only (RULE FIG3AB-FREEZE, v2.3)
- Fig 6 语义限定为 surrogate budget sensitivity (RULE FIG6-DEFAULT, v2.3)
- 若 Mode B 成功：可展示 partial eviction 效率优势 + quality degradation 分析，
  此时可恢复 Fig 3a/b 并升级 Fig 6 解释
