================================================================================
  BidKV SGLang 路线重新定位方案
  Issue-048 · Portability Validation Slice
  版本：v2.3-frozen（reviewer APPROVED 2026-03-19）
  日期：2026-03-19
  前置文档：docs/vllm-route-redesign.md（vLLM 双轨方案）
  修订说明：
    v1.0 — 初版：4 策略设计、portability slice 定位、fairness audit
    v2.2 — 术语统一、DC 形式化、claim 双级退化（reviewer APPROVED）
    v2.3 — §6.8 fallback claim 降级规则、§4.2 candidate_count 审计
           （reviewer APPROVED, 最终冻结版）
================================================================================

目录
  §1  路线重新定位
  §2  SGLang Adapter 目标边界
  §3  SGLang Baseline 设计（4 策略）
  §4  SGLang 内部 Fairness / Consistency
  §5  执行语义与 vLLM 的关系
  §6  论文表述模板
  §7  可选增强项 vs 明确不做项
  §8  实施排期
  §9  风险评估

================================================================================
§1  路线重新定位
================================================================================

1.1 核心定位

    ┌─────────────────────────────────────────────────────────────────┐
    │  SGLang is a portability validation slice, not a second        │
    │  primary quantitative platform.                                │
    └─────────────────────────────────────────────────────────────────┘

    SGLang 在论文中承担的角色：
    - 验证 BidKV adapter 抽象能否迁移到架构显著不同的 serving framework
    - 验证在不同 cache / prefix / scheduling 体系下收益方向是否一致
    - 提供 external validity / portability evidence

    SGLang 不承担：
    - 第二套完整主实验（那是 vLLM 的职责）
    - 第二套与 vLLM 对等的内核扩展工程
    - 与 vLLM 相同的最强执行语义
    - 跨平台数值等价的证明责任

1.2 与 vLLM 的角色分工

    ┌──────────────────────────┬───────────────────────────────────┐
    │ vLLM（主量化平台）        │ SGLang（portability slice）       │
    ├──────────────────────────┼───────────────────────────────────┤
    │ full attribution platform │ portability validation           │
    │ 完整 8-baseline 归因链    │ 精简 4-strategy 方向验证          │
    │ Table 1 + Fig 3-6 主数据  │ Table 2 + Fig 7 portability 证据 │
    │ 可接受最小源码扩展        │ 默认 adapter-only，不做内核扩展   │
    │ 承担最强 execution 语义   │ 使用 framework-native safe 执行   │
    │ 主结论承载平台            │ external validity / 辅助证据      │
    └──────────────────────────┴───────────────────────────────────┘

1.3 SGLang 回答的三个问题

    Q1: BidKV 抽象是否能迁移到另一类 serving framework？
        → adapter-based integration 是否可行？

    Q2: 在架构显著不同的 cache / prefix / scheduling 体系下，
        收益方向是否仍然一致或至少可解释？
        → directional consistency validation

    Q3: BidKV 的 adapter-based control-plane 设计是否具有跨框架可移植性？
        → FrameworkAdapter ABC 是否足够通用？

1.4 SGLang 不回答的问题

    ✗ 与 vLLM 的绝对提升幅度是否接近
    ✗ 两个框架是否共享同样大小的 candidate universe
    ✗ 两个框架是否有同样的 cache 粒度 / prefix 行为 / hook 时序
    ✗ SGLang 上某个 baseline 的排序是否与 vLLM 完全一致

================================================================================
§2  SGLang Adapter 目标边界
================================================================================

2.1 必须接入的能力

    能力 1 — Request lifecycle visibility
        能够知道请求的创建、active、completion 状态。
        实现：scheduler_hook.py monkey-patch + on_request_complete callback。

    能力 2 — KV stats / cache occupancy visibility
        能够获取 used_tokens / total_tokens。
        实现：通过 TokenToKVPool.size / available_size()。

    能力 3 — Pressure-event interception boundary
        BidKV 必须在 native eviction / preemption 之前获得行动机会。
        语义定义（非函数名绑定）：
        > BidKV must get a chance to act before native eviction is executed.
        当前实现：hook get_next_batch_to_run()，在每次 batch 选择前触发。
        注意：不过早绑定唯一函数名为唯一合法 hook 点。
        如果 SGLang 版本变化导致该函数名不可用，允许寻找等价 hook 点。

    能力 4 — Request completion callback
        请求完成时清理 bid pool、token tracking、shared position state。
        实现：scheduler_hook.py 中 hook free_request 或 output path。

    能力 5 — Candidate snapshot collection
        每次 pressure event 时，记录当前所有活跃请求快照。
        用于 within-platform candidate-universe consistency 审计。

    能力 6 — Fairness audit logging
        记录 candidate list hash + KV snapshot hash。
        确保同一 pressure event 下所有 baseline 看到同一候选池。

2.2 Pressure-interception 语义要求

    冻结的是语义，不是单一函数名：

    ┌─────────────────────────────────────────────────────────────────┐
    │  BidKV must get a chance to act before native cache eviction   │
    │  is finalized.                                                 │
    │                                                                │
    │  Acceptable hook points:                                       │
    │  - near batch selection path                                   │
    │  - before native cache eviction is finalized                   │
    │  - exact hook point determined during implementation           │
    └─────────────────────────────────────────────────────────────────┘

    当前 hook 方案：get_next_batch_to_run() monkey-patch。
    该方案已在 27 tests 中验证。若 SGLang 版本迭代导致 API 变化，
    首先尝试等价 hook 点，而非放弃整个 adapter。

2.3 不默认承诺的实现目标

    以下目标在 SGLang issue 中不作为要求：

    ❌ True partial KV truncation while continuing decode
       → SGLang 的 RadixAttention 天然支持 token-level 操作，
         但我们不承诺该能力在所有场景下安全可用。
         如果 token-level 释放在 GPU 运行中遇到问题，
         fallback 到 request-level 驱逐。

    ❌ 与 vLLM 等价的 execution semantics
       → vLLM 有 Mode A (recompute) / Mode B (tail truncation) 两条路线。
         SGLang 不需要复刻这两条路线。

    ❌ 必须零源码改动
       → 不作为约束。如果需要少量改动来稳定 hook 点，允许。
         但改动量不应超过 vLLM 内核扩展级别。

    ❌ 与 vLLM 相同的内核扩展能力
       → SGLang 的架构不同，扩展需求不同。不做对称工程。

2.4 SGLang 的执行语义策略

    SGLang 默认采用 adapter-native execution path：

    a) 首选：token-level KV 释放（RadixAttention 天然能力）
       SGLang 的 RadixAttention 支持按节点粒度释放 KV slot，
       这是 SGLang 架构的原生能力，不需要源码扩展。
       当前 adapter 已实现 radix_hook.py::free_kv_positions()。

    b) 保底：request-level 驱逐（SGLang 原生 evict 路径）
       若 token-level 释放在某些场景下不安全或不可用，
       fallback 到让 SGLang 原生驱逐被选中的请求。

    关键区别：
    - vLLM 需要内核扩展是因为 vLLM 不支持 partial release
    - SGLang 的 RadixAttention 天然支持 token-level 操作
    - 因此 SGLang 不需要 vLLM 那样的内核扩展

    但注意：SGLang 的 token-level 释放虽然在架构上可行，
    但在 GPU 运行时的安全性仍需 pilot 验证（Phase C.8 smoke test）。
    若 smoke test 发现问题 → 切换到 request-level fallback。

    c) Request-level fallback 下的 claim 降级（RULE SGLANG-FALLBACK-CLAIM, v2.3）
       若 smoke test 导致 token-level 释放被回退到 request-level：
       · portability claim 仍然有效（adapter feasibility + directional consistency）
       · 但不再暗示 native token-level execution benefits
       · 不再暗示 execution semantics advantage over vLLM Mode A
       · 论文表述必须同步降级（使用 SGLANG-FALLBACK-SENTENCE）
       · Table 2 的 "Notes" 列标注 "request-level fallback"
       · Figure 7 的 caption 注明 execution semantics 差异

       SGLANG-FALLBACK-SENTENCE（token-level 成功时不使用此句）：
       > "SGLang portability validation executed under request-level safe
       >  fallback semantics. The portability claim remains valid as
       >  evidence of adapter feasibility and directional consistency
       >  under framework-native safe execution, but does not imply
       >  native token-level execution benefits."

================================================================================
§3  SGLang Baseline 设计（4 策略）
================================================================================

3.1 保留的策略

    ┌─────┬──────────────┬───────────────────────────────────────────┐
    │  #  │ 策略名       │ 论文角色                                   │
    ├─────┼──────────────┼───────────────────────────────────────────┤
    │  1  │ Preempt-Evict│ SGLang 原生 LRU 驱逐，portability 起点     │
    │     │ (SGLang-     │ baseline。等同于 vLLM 的 Preempt-Evict。   │
    │     │  Default)    │                                            │
    ├─────┼──────────────┼───────────────────────────────────────────┤
    │  2  │ Slack-Aware  │ 强无-bid 系统级对手。                       │
    │     │              │ 回答：在另一 runtime 下，bid 信息是否仍     │
    │     │              │ 超越纯系统调度信号？                        │
    ├─────┼──────────────┼───────────────────────────────────────────┤
    │  3  │ BidKV        │ 完整系统（scoring + bid + solver）。        │
    │     │              │ 主角。                                      │
    └─────┴──────────────┴───────────────────────────────────────────┘

3.2 策略变更说明（对比 Issue-048 v1）

    Issue-048 v1 使用的 4 策略：
        SGLang-Default, BidKV, Global-NoBid, Uniform

    v2 调整为：
        SGLang-Default (Preempt-Evict), Slack-Aware, BidKV

    变更理由：

    a) 替换 Global-NoBid → Slack-Aware
       Global-NoBid 的核心归因价值（bid vs system-inferred）已由 vLLM
       主实验完成。SGLang portability slice 的关键问题不是重复 bid 归因，
       而是验证在不同 runtime 下 BidKV 是否仍优于强系统级 baseline。
       Slack-Aware（SLO deadline 感知，无 bid）是更合适的"强系统对手"。

    b) 结果：3 策略覆盖 2 层对比
       Preempt-Evict → BidKV ：compression 是否有效
       Slack-Aware → BidKV   ：bid 信息是否仍优于纯系统信号

3.3 为什么不在 SGLang 上复刻完整 baseline 链

    ┌─────────────────────────────────────────────────────────────────┐
    │  The reduced baseline set is intentional: full attribution is  │
    │  performed on vLLM, while SGLang validates transferability     │
    │  under a structurally different runtime.                       │
    └─────────────────────────────────────────────────────────────────┘

    具体理由：

    a) H2O-Style / Static-Random / Uniform / Global-NoBid 的完整归因
       任务已由 vLLM 主实验（#047 Table 1）承担。在 SGLang 上重复
       这些对比，论文收益远低于工作量。

    b) portability slice 的核心问题是"可迁移性 + 方向一致性"，
       而非"在第二个平台上重做完整归因分析"。

    c) 3 策略足以验证 2 层方向一致性：
       - compression vs no-compression (Preempt-Evict → BidKV)
       - bid vs system-only (Slack-Aware → BidKV)

    d) 减少 SGLang 工作量，确保不阻塞 vLLM 主实验进度。

================================================================================
§4  SGLang 内部 Fairness / Consistency
================================================================================

    ┌─────────────────────────────────────────────────────────────────┐
    │  Within-platform fairness must still hold on SGLang for all    │
    │  compared baselines.                                           │
    └─────────────────────────────────────────────────────────────────┘

4.1 Within-platform candidate-universe consistency

    虽然 SGLang 是 portability slice，但不是"宽松实验"。
    SGLang 内部的 4 个 baseline 必须在公平条件下比较：

    规则 1: 同一 pressure event 中所有 baseline 使用同一候选池。
            candidate list = 当前所有 active requests 的快照。
            所有 baseline 的 select_victims() 接收同一 candidates 参数。

    规则 2: 相同 accounting snapshot。
            KV usage、queue state、request metadata 在比较时一致。

    规则 3: 唯一差异是 selection / ranking policy。
            Preempt-Evict: 按 vLLM-priority / SGLang-LRU 选择
            Slack-Aware: 按 SLO slack 排序
            BidKV: 按 bid-based utility 排序


4.2 审计机制

    每次 pressure event 必须记录：

    log_entry = {
        "event_id": unique_id,
        "timestamp_ms": ...,
        "candidate_list_hash": hash(sorted(candidate_request_ids)),
        "candidate_count": len(candidate_request_ids),
        "kv_snapshot_hash": hash(used_tokens, total_tokens, per_request_kv),
        "kv_used_tokens": used_tokens,
        "kv_total_tokens": total_tokens,
        "strategy": "bidkv" | "slack_aware" | ...,
        "decision": [...],  # 选中的 victim 列表
        "tokens_freed": ...,
    }

    这些日志必须在实验结束后可审计，确认：
    - 同一 event_id 下不同 strategy 看到相同的 candidate_list_hash
    - 同一 event_id 下不同 strategy 看到相同的 kv_snapshot_hash

4.3 与 vLLM 的 candidate-universe 关系

    ┌─────────────────────────────────────────────────────────────────┐
    │  vLLM 与 SGLang 不需要跨平台 candidate universe 同构。         │
    │  但 SGLang 自己内部的各 baseline 必须在同一 candidate          │
    │  universe 上比较。                                             │
    └─────────────────────────────────────────────────────────────────┘

    两个平台的 candidate universe 必然不同，原因：
    - KV 管理粒度不同（block vs token）
    - 前缀共享机制不同（flat hash vs radix tree）
    - 驱逐策略不同（priority-based vs LRU）
    - 调度时序不同

    这是预期行为，不需要解释或消除。
    论文中明确标注："within-platform fairness is guaranteed for each
    framework independently"。

================================================================================
§5  执行语义与 vLLM 的关系
================================================================================

5.1 vLLM 的执行语义（参考 vllm-route-redesign.md）

    Mode A: recompute fallback（P0）— BidKV selection + vLLM native preempt
    Mode B: tail truncation（P1）— 最小 vLLM 内核扩展

5.2 SGLang 的执行语义

    ┌─────────────────────────────────────────────────────────────────┐
    │  SGLang is not required to replicate the strongest execution   │
    │  semantics engineered for vLLM.                                │
    └─────────────────────────────────────────────────────────────────┘

    SGLang 默认路线：
    - adapter-based integration（现有 monkey-patch 方案）
    - framework-native safe execution path
    - 首选 token-level KV 释放（RadixAttention 原生能力）
    - 保底 request-level 驱逐（SGLang 原生 evict）

    SGLang 不需要：
    - 复刻 vLLM 的 Mode A / Mode B 双轨结构
    - 实现 truncate_request_tail() 风格的内核扩展
    - 保证与 vLLM 相同的 execution semantics

    关键区别：
    vLLM CraSH 是因为其 PagedAttention 不支持 partial block release。
    SGLang 的 RadixAttention 天然支持 token-level 操作。
    因此两个平台面临的技术挑战是不同的，解决方案也应不同。

5.3 SGLang 执行语义的 pilot 验证

    在正式 72-run 实验之前，必须先完成 smoke test：

    Smoke test checklist:
    [ ] SGLang serve 启动 + warmup request 成功（4 策略各 1 个）
    [ ] Pressure event 可触发（调整 mem-fraction-static 使 KV 紧张）
    [ ] BidKV token-level 释放不导致 crash（至少运行 5 分钟）
    [ ] RadixAttention 共享前缀保护工作正常（ref_cnt > 1 不释放）

    若 smoke test 通过 → 使用 token-level 释放执行。
    若 smoke test 发现 crash / 数据不一致 → 切换到 request-level fallback。

    注意：即使需要切换到 request-level fallback，也不影响 portability slice
    的论文价值。因为 portability 验证的是 selection intelligence 的迁移能力，
    而非特定 execution mechanic 的跨框架等价性。

================================================================================
§6  论文表述模板
================================================================================

6.1 角色说明

    > SGLang serves as a portability validation slice rather than a
    > second primary quantitative platform.

6.2 目标说明

    > The goal is to validate directionally consistent benefit and
    > adapter feasibility under a structurally different cache/runtime
    > architecture.

6.3 不要求数值一致

    > The portability slice validates directional consistency and
    > adapter feasibility, not numerical equivalence of gains across
    > frameworks.

    > Numerical equivalence of gains across frameworks is not required;
    > differences in absolute improvement magnitude are expected due
    > to architectural differences (RadixAttention tree-based KV vs
    > PagedAttention flat-block KV, token-level vs block-level
    > eviction granularity, different prefix sharing mechanisms).

6.4 若方向一致

    > BidKV shows directionally consistent benefit on SGLang under
    > a structurally different runtime: selection-layer improvements
    > persist across both flat-block (vLLM) and tree-based (SGLang)
    > KV management architectures.

6.5 若方向不一致

    > The discrepancy must be analyzed as a framework-specific effect
    > or execution-semantics difference, rather than treated as a
    > failure of portability by default.

    分析清单（若出现方向不一致）：
    a) SGLang RadixAttention 的 prefix sharing 行为是否导致了不同的
       KV 压力特征？（例如：共享前缀多 → 可压缩空间少 → BidKV 增量小）
    b) SGLang LRU 驱逐策略 vs vLLM priority-based preemption 的行为差异？
    c) Hook 时序差异是否导致了 BidKV 介入时机不同？
    d) 若差异可用上述因素解释 → 论文标注为 "framework-specific effect"
    e) 若差异无法解释 → 需要更深入的根因分析，可能影响 portability claim

6.6 Table 2 表述

    Table 2: Cross-Framework Performance Summary

    | Framework | Strategy     | SLO Viol.↓ | P99 TTFT↓ | Throughput↑ | Notes |
    |-----------|-------------|-----------|----------|------------|-------|
    | vLLM      | Default     | X₁%       | Y₁ ms    | Z₁ rps     |       |
    | vLLM      | Slack-Aware | X₂%       | Y₂ ms    | Z₂ rps     |       |
    | vLLM      | BidKV       | X₃%       | Y₃ ms    | Z₃ rps     |       |
    | SGLang    | Default     | X₄%       | Y₄ ms    | Z₄ rps     |       |
    | SGLang    | Slack-Aware | X₅%       | Y₅ ms    | Z₅ rps     |       |
    | SGLang    | BidKV       | X₆%       | Y₆ ms    | Z₆ rps     |       |

    关键对比：
    - vLLM Δ(Default→BidKV) vs SGLang Δ(Default→BidKV) → 方向是否一致？
    - vLLM Δ(Slack→BidKV)  vs SGLang Δ(Slack→BidKV)  → bid 价值方向一致？

    不比较：
    - 不比较 X₁ vs X₄（绝对值不可比）
    - 不比较 Y₁ vs Y₄（框架差异使绝对值无意义）

6.7 避免的措辞

    ❌ "SGLang achieves comparable gains to vLLM"
    ❌ "cross-platform numerical equivalence"
    ❌ "identical improvement magnitude"
    ❌ "SGLang replicates vLLM results"

    ✅ "directionally consistent benefit"
    ✅ "adapter feasibility under different architecture"
    ✅ "portability evidence across framework boundary"
    ✅ "improvement trend preserved under structural change"

6.8 Request-level fallback 表述（RULE SGLANG-FALLBACK-CLAIM, v2.3 新增）

    仅在 SGLang smoke test 失败、降级为 request-level 执行时使用。
    Token-level 正常工作时不使用以下句子。

    ✅ "SGLang portability validation executed under request-level safe
        fallback semantics. The portability claim remains valid as evidence
        of adapter feasibility and directional consistency under
        framework-native safe execution, but does not imply native
        token-level execution benefits."

    ✅ "The RadixAttention architecture was used for candidate selection
        and KV state visibility; eviction execution used the framework's
        native request-level path."

    ❌ "SGLang leverages native token-level KV release"
       （降级后此句禁止使用）
    ❌ "SGLang's RadixAttention enables finer-grained compression"
       （除非 token-level 确实被执行）

================================================================================
§7  可选增强项 vs 明确不做项
================================================================================

7.1 可选增强项（不阻塞主实验）

    [ ] RadixAttention prefix sharing 特征分析
        如果时间允许，分析 SGLang prefix sharing 对 BidKV 压缩空间的影响。
        可作为论文 §7 Discussion 中的 framework-specific insight。

    [ ] SGLang 上的 H2O decode step 评分质量
        评估 logprobs-based proxy scoring 与 vLLM 侧 attention-weight
        scoring 的一致性。可作为附录材料。

    [ ] 第 5 个策略（如 Uniform）
        如果 4 策略实验完成后有余力，可补充 Uniform 用于更丰富的消融。
        但不是默认要求。

7.2 明确不做项（不应成为阻塞项）

    ❌ SGLang 内核扩展（类似 vLLM truncate_request_tail）
       → SGLang 的 RadixAttention 天然支持 token-level 操作，
         不需要内核扩展。若 token-level 不安全 → request-level fallback。

    ❌ 复刻 vLLM 的 Mode A / Mode B 双轨结构
       → SGLang 有自己的 execution path，不需要对称工程。

    ❌ 完整 8 baseline 链
       → 归因分析由 vLLM 承担。

    ❌ 与 vLLM 相同数量的 workload / concurrency 维度
       → 2 workload × 3 concurrency × 3 runs = 72 runs 足够。

    ❌ 跨框架 candidate-universe 同构证明
       → 预期不同，不需要解释或消除。

    ❌ 质量评测（ROUGE / EM）的跨框架对比
       → 若 SGLang 使用 token-level 释放，质量退化模式与 vLLM block-level
         不同。跨框架质量对比需要额外校准，收益不足。
         vLLM 主实验负责质量分析。

================================================================================
§8  实施排期
================================================================================

    ┌──────┬──────────────────────────────────────┬───────┬──────────┐
    │ 步骤 │ 任务                                  │ 依赖  │ 优先级   │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.1  │ SGLang adapter 策略列表更新            │ 无    │ P0       │
    │      │ (Global-NoBid → Slack-Aware,          │       │          │
    │      │  Uniform removed)                     │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.2  │ SGLang smoke test（4 策略 × warmup）  │ S.1   │ P0       │
    │      │ 验证 serve 启动 + pressure 触发        │       │          │
    │      │ + token-level 释放安全性               │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.3  │ Fairness audit logging 实现           │ S.1   │ P0       │
    │      │ candidate_list_hash + kv_snapshot_hash │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.4  │ 实验 runner 脚本                      │ S.3   │ P0       │
    │      │ experiments/sglang/run_experiment.py   │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.5  │ 全量实验：72 runs                     │ S.4   │ P0       │
    │      │ 4 × 2 × 3 × 3                        │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.6  │ 生成 Table 2 + Fig 7                  │ S.5   │ P0       │
    │      │ 与 vLLM 数据合并展示                   │       │          │
    ├──────┼──────────────────────────────────────┼───────┼──────────┤
    │ S.7  │ 可选增强项（如有余力）                  │ S.6   │ P2       │
    └──────┴──────────────────────────────────────┴───────┴──────────┘

    关键约束：
    - S.1-S.6 是 SGLang portability slice 的完整 critical path
    - 可与 vLLM C.1-C.4（Mode A 全量实验）并行推进
    - 不阻塞 vLLM 主实验
    - 若 S.2 smoke test 发现 token-level 释放不安全：
      → 切换到 request-level fallback，重新估算 S.5 时间
      → 不升级为内核扩展工程

================================================================================
§9  风险评估
================================================================================

    ┌──────────────────────────┬──────┬────────────────────────────────┐
    │ 风险                     │ 概率 │ 缓解措施                       │
    ├──────────────────────────┼──────┼────────────────────────────────┤
    │ SGLang token-level 释放  │ 中   │ S.2 smoke test 提前发现。       │
    │ 在 GPU 运行中不安全      │      │ 切换到 request-level fallback。 │
    │                          │      │ Portability 价值不受影响。      │
    ├──────────────────────────┼──────┼────────────────────────────────┤
    │ BidKV vs Preempt-Evict   │ 低   │ 即使 SGLang 原生 LRU 很强，    │
    │ 方向不一致（SGLang 原生   │      │ BidKV quality-aware selection  │
    │ eviction 已足够好）      │      │ 仍应有差异。若真不一致：        │
    │                          │      │ 分析为 framework-specific       │
    │                          │      │ effect（见 §6.5）。             │
    ├──────────────────────────┼──────┼────────────────────────────────┤
    │ SGLang 版本变化导致       │ 低   │ 当前 adapter 已有版本兼容层    │
    │ hook 点失效              │      │ （多属性路径 fallback）。       │
    │                          │      │ 轻量 hook 更新即可。           │
    ├──────────────────────────┼──────┼────────────────────────────────┤
    │ SGLang 路线膨胀为        │ 中   │ 本文档明确定义边界。            │
    │ 第二个大工程             │      │ §7.2 明确不做项列表。           │
    │                          │      │ 任何超出范围的需求需 reviewer   │
    │                          │      │ 审批。                         │
    ├──────────────────────────┼──────┼────────────────────────────────┤
    │ Fairness audit 实现      │ 低   │ 复用 vLLM 侧的 audit logging  │
    │ 增加工作量               │      │ 框架，适配 SGLang 数据源即可。  │
    └──────────────────────────┴──────┴────────────────────────────────┘

================================================================================
END OF DOCUMENT — 待 reviewer 审核冻结
================================================================================
