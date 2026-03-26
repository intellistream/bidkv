# Reviewer 评审提示词 — SGLang Portability Slice 方案冻结

> **用途**：将此提示词整体发给 reviewer，请求对 SGLang 路线方案做 APPROVE / REQUEST_CHANGES / REJECT。
> 冻结后方可进入代码实现阶段。

---

## 提示词正文

---

**角色**：你是 BidKV 论文的 reviewer / 研究方法论审核人。

**任务**：评审 SGLang 路线的重新定位方案（v2.1），判断是否可以冻结并进入实现。

**背景**：
- BidKV 是一个 framework-agnostic KV compression scheduling primitive，需在论文中证明跨框架可移植性。
- **vLLM 是主实验平台**（8 策略 × 全量矩阵 = 144+ runs，Table 1 + Figure 3/4），承担全部定量归因。
- **SGLang 被重新定位为 portability validation slice**（4 策略 × 72 runs，Table 2 + Figure 7），仅验证方向一致性和适配器可行性。
- 前序决定：vLLM 路线已冻结为双执行模式（Mode A recompute fallback P0 + Mode B tail truncation P1），详见 `vllm-route-redesign.md`。

**关键架构差异**：
- vLLM：PagedAttention（扁平分块 KV），block-level eviction → BidKV 需 kernel extension 做 partial block free
- SGLang：RadixAttention（树状 KV），token-level eviction → BidKV 可原生 token-level 操作，无需 kernel extension

**本次方案核心决策（请逐项评审）**：

### 1. 定位重新校准

SGLang 从"第二主量化平台"降级为 **portability validation slice**。

> **SGLang is a portability validation slice, not a second primary quantitative platform.**

论文中 SGLang 的作用是补充"跨框架方向一致性"（Figure 7 + Table 2），不替代 vLLM 的主归因分析。

- 你是否同意此定位？
- 这是否足以支撑论文中 "portable" / "framework-agnostic" 的声明？
- 是否需要更多平台（如第三个引擎）才能做出可信的 portability claim？

### 2. 策略集变更（3 策略 v2.0）

| v1 策略 | v2.0 策略 | 变更原因 |
|---------|-----------|---------|
| SGLang-Default | Preempt-Evict | 不变，baseline |
| Global-NoBid | **Slack-Aware** | bid 归因已由 vLLM 承担；SGLang 用 Slack-Aware 展示"有配额但无竞价" vs "有竞价"的对比 |
| BidKV | BidKV | 不变，核心方法 |

> **The reduced baseline set is intentional: full attribution is performed on vLLM, while SGLang validates transferability under a structurally different runtime.**

- 你是否同意 3 策略足够支撑 portability 验证？
- Slack-Aware 替换 Global-NoBid 的逻辑是否成立？
- 是否有必须保留的策略被遗漏？

### 3. 验收标准变更

旧标准：
- ❌ "BidKV SLO 降低 ≥ 10%"（硬性数值门槛）

新标准：
- ✅ BidKV 方向性优于 Default 和 Slack-Aware（rank order 一致）
- ✅ Fairness audit 通过（candidate hash 一致）
- ✅ 72 runs 中 ≥ 90% 完成

> **The goal is directional consistency and adapter feasibility, not numerical equivalence of gains across frameworks.**

- 你是否同意移除硬数值门槛？
- "方向一致性"的定义是否需要更形式化？（当前用 rank order + 方向性改善）
- 若 SGLang 上 BidKV 与 Slack-Aware 无显著差异（方向一致但 gap 极小），你如何评判？

### 4. 执行语义差异

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV 架构 | PagedAttention (flat block) | RadixAttention (tree-based) |
| 释放粒度 | block-level → 需 kernel extension | token-level → 原生支持 |
| BidKV 执行 | Mode A: recompute fallback | Native: free_kv_positions() |
| 质量影响 | Mode A 无质量损失（完整重算） | Token-level free 后续 token 可能受影响 |

> **SGLang is not required to replicate the strongest execution semantics engineered for vLLM.**

- SGLang 使用 native token-level 操作而非 vLLM 的 recompute fallback，这是否可接受？
- 论文是否需要明确说明两个平台的执行语义差异？
- 如果 SGLang token-level smoke test 失败并降级为 request-level fallback，portability claim 是否仍成立？

### 5. Within-Platform Fairness

> **Within-platform fairness must still hold on SGLang for all compared baselines.**

审计机制：candidate_universe hash + kv_snapshot hash + frozen trace 一致 + server 参数一致。

- 审计机制是否充分？
- 是否需要额外的公平性保障措施？

### 6. 论文叙事

**推荐叙事**：
> "BidKV achieves directionally consistent improvements on both vLLM (PagedAttention) and SGLang (RadixAttention), demonstrating that compression scheduling primitives transfer across structurally different KV architectures."

**禁止叙事**：
> ❌ "SGLang results confirm vLLM findings"（暗示数据等价）
> ❌ "BidKV achieves X% improvement on SGLang"（暗示精确量化）

- 叙事定位是否合适？
- 是否存在 over-claim 或 under-claim 风险？

### 7. 风险评估

| 风险 | 缓解措施 | 你的评估 |
|------|---------|---------|
| SGLang smoke test 失败 | 降级为 request-level fallback | ? |
| 方向不一致 | 归因为 framework-specific effect | ? |
| Reviewer 质疑两个平台不够 | 论文 Limitation 声明，提供 adapter LOC 对比 | ? |
| SGLang 实验未完成（deadline） | 不阻塞 vLLM 主实验，可降级为 appendix | ? |
| Token-level free 导致质量退化 | Smoke test 检测，必要时 fallback | ? |

---

**参考文档**：
1. `bidkv/docs/sglang-portability-slice.md` — 完整 9 节 SGLang 路线设计
2. `bidkv/docs/experiment_protocol.md` §14 — 实验协议 SGLang 部分
3. `issue-048-sglang-experiment.md` — v2.0 修订版 Issue
4. `bidkv/docs/vllm-route-redesign.md` — vLLM 双轨方案（已冻结）

---

**请给出你的判断**：

```
[ ] APPROVE      — 方案无重大问题，进入实现阶段
[ ] REQUEST_CHANGES — 方案基本可行，但有以下修改要求：
    修改要求：
    1. ...
    2. ...
[ ] REJECT       — 方案存在根本性问题，需重新设计：
    原因：...
```

**补充问题**（如有）：
1. ...
2. ...

---

> 本提示词由 SGLang 实验 agent 生成，对应 v2.1 experiment_protocol.md。
