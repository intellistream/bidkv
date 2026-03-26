================================================================================
  BidKV 主实验实施方案
  Issue-047 · 论文 §6 主实验 + Issue-048 · SGLang 可移植性验证
  版本：v2.3-frozen（reviewer APPROVED 2026-03-19）
  修订说明：
    v2.0 — 引入双执行模式（Mode A / Mode B），解决 vLLM v1 partial eviction crash
    v2.1 — 新增 §14 SGLang 可移植性切片（portability validation slice）
    v2.2 — SGLang 术语统一、DC 形式化、claim 双级退化（reviewer APPROVED）
    v2.3 — Figure 3a/b 移出主线、Figure 6 语义限定、Scenario 切换形式化、
           rate-freeze、narrative baseline split、SGLang fallback claim
           （reviewer APPROVED, 最终冻结版）
================================================================================

实验总产出：
  vLLM  ：Table 1（主对比表）+ Figure 3（rate sweep）+ Figure 4（机制解释）
  SGLang：Table 2（portability 对比表）+ Figure 7（cross-platform consistency）

硬件环境  ：NVIDIA RTX A6000 48GB · CUDA 12.5
模型      ：Llama-3.1-8B-Instruct（bf16, 16GB）
推理引擎  ：vLLM 0.17.1（v1 架构）+ SGLang（RadixAttention, tree-based KV）
注入方式  ：
  vLLM  ：monkey-patch EngineCore.__init__ + --disable-frontend-multiprocessing
  SGLang：monkey-patch get_next_batch_to_run() + token-level radix hook
执行模式  ：
  vLLM  ：Mode A (recompute fallback, 默认) / Mode B (tail truncation, 增强)
  SGLang：native token-level（RadixAttention 原生支持 token-level 操作）
详见      ：
  vLLM  ：docs/vllm-route-redesign.md
  SGLang：docs/sglang-portability-slice.md

目录
  §1  实验总目标
  §2  实验实施总原则
  §3  分阶段实施路线（Phase 0–7, vLLM 主实验）
  §4  Workload 设计
  §5  Request-Rate 设计
  §6  Frozen Trace 设计
  §7  指标设计
  §8  Fail-Safe 与实验风控
  §9  主实验矩阵与资源规划
  §10 补充消融实验
  §11 结果整理与论文映射
  §12 执行版拍板清单
  §13 执行模式修订（v2.0）— vLLM Partial Eviction Crash 修复
  §14 SGLang 可移植性切片（v2.1 新增）— Portability Validation Slice

================================================================================
§1  实验总目标
================================================================================

1.1 一句话定义

    在真实 vLLM 推理引擎上，以 Poisson 开环到达 + ShareGPT 真实对话数据为
    输入，对 7 个 KV cache eviction 策略做端到端实机对比，证明 BidKV 在 KV
    压力区间内同时优于所有 heuristic baseline。

1.2 最终产出物及其依赖

    产出物           所需数据来源                           说明
    ──────────────── ────────────────────────────────────── ────────────────────
    Table 1          Phase 5 全量运行（mid-rate 结果）      8 策略 × 2 workloads
    （主对比表）      所有主指标 + eviction count             mean ± CI95

    Figure 3         Phase 5 全量运行（3 rates 全部）       4 核心策略主图
    （rate sweep）   X=rate, Y=normalized latency/throughput 其余 4 策略 appendix

    Figure 4         Phase 5 数据 + Phase 4 插桩指标        eviction count、
    （机制解释图）   + 消融 A（如已完成）                    KV util、pressure sweep

    主结论           Table 1 + Figure 3                     BidKV throughput 优于
                                                            h2o-style ≥ X%；
                                                            P95 TTFT 优于
                                                            preempt-evict ≥ Y%

================================================================================
§2  实验实施总原则
================================================================================

本节列出贯穿全实验的 6 条原则。所有阶段的操作与决策均不得违背。

原则 P1 ── 真实性优先
    使用 ShareGPT 真实对话数据 + Poisson 开环到达模型。
    禁止使用合成 filler 段落拼接 prompt。
    禁止使用闭环（Semaphore）并发模型。
    理由：合成数据导致 attention pattern 退化，使 H2O 等基于 attention score
    的策略无法产生有意义的差异化评分；闭环模型隐藏排队延迟，无法反映真实
    在线服务场景。两者均会被 reviewer 直接质疑。

原则 P2 ── 先 pilot 后冻结
    不预设所有数值参数。先用 2 个代表策略（preempt-evict + h2o-style）跑
    pilot calibration，确认各 workload 的"有效 eviction 压力区间"后，再
    冻结正式 request rate 和 frozen trace。
    禁止跳过 pilot 直接进入全量矩阵运行。

原则 P3 ── trace 冻结后不可回改
    一旦 frozen trace 通过复现性验证并进入 Phase 4 及之后的阶段，任何对
    trace 内容（prompt 文本、arrival_time_ms、请求数量、max_tokens、请求
    顺序）的修改，都必须导致 Phase 3–7 所有已有结果作废，从 Phase 3 开头
    重新执行。

原则 P4 ── 主实验最小闭环优先
    优先跑出能支撑 Table 1 + Figure 3 的最小实验集（4 个核心策略），在产出
    可用论文数据后，再补全剩余策略和消融实验。不要试图一次性做满所有维度。

原则 P5 ── 失败计入结果，不静默重试
    OOM、timeout、server crash、abort 都是合法实验结果，必须被完整记录和
    统计。不允许挑选"结果好看的 run"留下、"结果差的 run"丢弃。所有 run
    的原始 JSON 全部保留，即使被标记为 abort 或 failure。

原则 P6 ── 单一变量控制
    同一个 frozen trace 文件在所有 8 个 strategy 间共享。保证实验的唯一变量
    是 eviction 策略本身，消除请求序列差异对跨策略对比的干扰。

================================================================================
§3  分阶段实施路线
================================================================================

全实验共分 8 个阶段（Phase 0–7），严格按顺序执行。每个阶段有明确的准入条件
（Gate）和输出产物；未满足准入条件不得进入下一阶段。

阶段总览：

    Phase 0  实验对象与指标冻结        ─── 不依赖 GPU
    Phase 1  数据集与 workload 设计冻结 ─── 不依赖 GPU
    Phase 2  Pilot calibration          ─── 需要 GPU
    Phase 3  Frozen trace 生成与冻结     ─── 需要 GPU（tokenizer 验证）
    Phase 4  观测指标插桩与 sanity check ─── 需要 GPU
    Phase 5  主实验全量运行             ─── 需要 GPU（核心阶段）
    Phase 6  补充消融实验（可选）        ─── 需要 GPU
    Phase 7  结果清洗、汇总与论文映射    ─── 不依赖 GPU

────────────────────────────────────────────────────────────────────────────────
Phase 0：实验对象与指标冻结
────────────────────────────────────────────────────────────────────────────────
依赖：无
GPU ：不需要

目标：
    冻结策略列表、指标定义、SLO 阈值、vLLM server 基础参数。
    此后不允许增删策略或修改指标定义。

动作清单：
    [0-1] 确认 8 个策略名称及其在 serve.py 中的注入映射：
          preempt-evict    → 原生 vLLM（无 BidKV 注入）
          static-random    → BidKV + static-random overlay
          h2o-style        → BidKV + h2o overlay
          uniform          → BidKV + uniform overlay
          global-nobid     → BidKV + global-nobid overlay
          slack-aware      → BidKV + slack-aware overlay
          bidkv            → BidKV 完整策略
    [0-2] 逐一验证：每个策略可以启动 vLLM 服务并完成 1 个 warmup 请求。
          不要求性能正确，只要求进程不 crash。
    [0-3] 冻结主指标列表（完整定义见 §7）。
    [0-4] 冻结 SLO 阈值：
          - TTFT SLO = 2000 ms
          - TPOT SLO = 100 ms
    [0-5] 冻结 vLLM server 基础参数：
          --max-model-len              8192
          --gpu-memory-utilization     0.85（Phase 2 可能调整，见 §5）
          --enforce-eager              （禁用 CUDA graph，便于 hook）
          --disable-frontend-multiprocessing （确保 EngineCore 在同一进程）
          --block-size                 16

输出产物：
    - experiment_protocol.md
      内容：策略列表 + 指标定义 + SLO 阈值 + server 参数

Gate → Phase 1：
    ✓ 8 个策略全部通过启动 + warmup 验证
    ✓ experiment_protocol.md 已提交并冻结

────────────────────────────────────────────────────────────────────────────────
Phase 1：数据集与 workload 设计冻结
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 0 完成
GPU ：不需要（仅 tokenizer 可在 CPU 运行）

目标：
    下载 ShareGPT 数据集，确认数据质量与可用样本量，冻结 workload 定义。

动作清单：
    [1-1] 通过 hf-mirror.com 下载 ShareGPT_Vicuna_unfiltered JSON 文件。
    [1-2] 用 Llama-3.1-8B-Instruct 的 tokenizer 对全量 conversations[0]
          逐条 tokenize，输出 prompt token 长度分布统计：
          P10 / P25 / P50 / P75 / P90 / P95 / P99 / mean / max。
    [1-3] 按 §4 的 workload 定义进行分桶：
          - Workload A (Mixed)：全量 ShareGPT，不限长度
          - Workload B (Long-context)：prompt_len ≥ 1024 tokens
    [1-4] 验证每个 workload 分桶后的可用样本数 ≥ 目标请求数 × 2：
          - Mixed 需要 ≥ 2000 条可用样本
          - Long-context 需要 ≥ 1000 条可用样本
          若不满足 → 降低 Long-context 的 prompt_len 下限（如降到 512）
          或增加 Mixed 的采样范围，直到满足。
    [1-5] 检查是否存在超过 max_model_len (8192) 的 outlier，若有则从样本池
          中剔除。
    [1-6] 冻结 workload_definition.md。

输出产物：
    - data/sharegpt_raw.json            原始下载数据
    - data/sharegpt_tokenized_stats.json 全量 token 长度分布统计
    - workload_definition.md             各 workload 的筛选条件 + 样本量

Gate → Phase 2：
    ✓ 数据集文件完整、可读
    ✓ 每个 workload 可用样本数 ≥ 目标请求数 × 2
    ✓ 无超过 max_model_len 的 outlier（或已剔除）
    ✓ workload_definition.md 已提交

────────────────────────────────────────────────────────────────────────────────
Phase 2：Pilot Calibration
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 1 完成
GPU ：需要（A6000 显存必须已释放至 baseline）

目标：
    用 2 个代表策略（preempt-evict + h2o-style）做快速 rate 扫描，为每个
    workload 找到"有效 eviction 压力区间"，确定正式 request rate 点。

动作清单：
    [2-1] 为每个 workload 生成 pilot 用临时 trace：
          - 请求数 = 正式请求数的 50%（Mixed 500, Long-context 250）
          - 到达模型 = Poisson 开环
          - seed = 99（区别于正式 trace 的 seed=42，避免混淆）
          - rate 粗扫序列：从 0.5 req/s 开始，以 ×1.5 步进
            （0.5 → 0.75 → 1.1 → 1.7 → 2.5 → 3.8 → 5.6 → 8.4 ...）
            直到出现 OOM 率 > 20% 或 timeout 率 > 20% 时停止上探。
    [2-2] 对每个 (workload, rate) 分别跑 preempt-evict 和 h2o-style 各 1 run。
    [2-3] 每个 run 记录以下观测值：
          - eviction count（h2o-style 下；preempt-evict 下为 preemption count）
          - peak KV utilization (%)
          - P95 TTFT (ms)
          - OOM / timeout / abort 各自的请求数
          - throughput (completed req/s)
    [2-4] 根据观测值，为每个 workload 选定 3 个正式 rate 点：
          ┌──────────┬────────────────────────────────────────────────────┐
          │ rate_low │ eviction 刚开始触发                               │
          │          │ h2o-style eviction count > 0 但 < 总请求数 10%    │
          │          │ KV util peak ≈ 70–80%                             │
          │          │ 所有策略均可正常完成，OOM = 0                      │
          ├──────────┼────────────────────────────────────────────────────┤
          │ rate_mid │ eviction 活跃                                     │
          │          │ eviction count 占总请求数 20–50%                   │
          │          │ KV util peak ≈ 85–95%                             │
          │          │ OOM/abort < 5%                                    │
          ├──────────┼────────────────────────────────────────────────────┤
          │ rate_high│ 接近系统极限，eviction 非常频繁                    │
          │          │ 部分弱策略 OOM < 15%                              │
          │          │ 核心策略仍可跑完                                   │
          └──────────┴────────────────────────────────────────────────────┘
    [2-5] 异常处理——无法找到 3 个有效 rate 点的情况：
          情况 A：某 workload 在所有 rate 下均不触发 eviction
          → 将 gpu_memory_utilization 从 0.85 逐步降低至 0.80 → 0.75，
            人为缩减 KV 池容量，重新扫描。
          → 若降至 0.75 仍无 eviction → 该 workload 降级为 appendix 实验，
            不纳入主实验。
          → 将最终采用的 gpu_memory_utilization 值作为 Phase 0 参数的修正，
            记录修改理由。

          情况 B：某 workload 在最低 rate 就频繁 OOM
          → 降低该 workload 的 prompt 长度上限（如 Long-context 的下限从
            1024 降到 768），或降低 max_model_len。
          → 若调整后依然无法使用 → 该 workload 降级为 appendix。

          情况 C：只能找到 2 个有效 rate 点（rate_low + rate_mid，无 rate_high）
          → 接受 2 个 rate 点进入主实验，Figure 3 该 workload 的曲线少一个点。
             在论文中说明原因。

输出产物：
    - pilot_results/           每个 (workload, rate) 的观测值 JSON
    - calibration_report.md    选定的正式 rate 值 + 选择理由
                               + 最终 gpu_memory_utilization 值

Gate → Phase 3：
    ✓ 每个主实验 workload 至少有 2 个 rate 点在"有效 eviction 区间"内
    ✓ preempt-evict 和 h2o-style 在各 workload 的 rate_mid 均能跑完无 crash
    ✓ GPU 内存可在 strategy 切换间完全释放（nvidia-smi 显存回到模型 baseline）
    ✓ calibration_report.md 已提交

失败回滚规则：
    - GPU 内存泄漏 → 修复 process group kill 机制后重试 Phase 2
    - rate 区间不合理 → 调整 gpu_memory_utilization 后重试 Phase 2
    - workload 无法使用 → 修改 workload 定义，退回 Phase 1 重新冻结

────────────────────────────────────────────────────────────────────────────────
Phase 3：Frozen Trace 生成与复现性冻结
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 2 完成，正式 rate 已确定
GPU ：不需要（tokenizer 可在 CPU 运行；seed 固定的 Poisson 生成为纯 CPU 运算）

目标：
    根据 Phase 2 确定的正式 rate 生成所有 frozen trace 文件，验证复现性后
    冻结。此后 trace 内容不可修改（见原则 P3）。

动作清单：
    [3-1] 为每个 (workload, rate) 组合生成一个 frozen trace JSON 文件。
          生成参数：
          - seed = 42（所有 trace 统一）
          - 数据源 = Phase 1 下载的 ShareGPT 数据
          - 到达模型 = Poisson（expovariate）
          - 请求数 = Mixed 1000 / Long-context 500
          共计 2 workloads × 3 rates = 6 个 trace 文件。
          命名约定：mixed_rate{X.X}.json, long_rate{Y.Y}.json
    [3-2] 每个 trace 文件中包含以下字段：
          顶层：
            workload_name        字符串
            request_rate         浮点，req/s
            seed                 整数
            dataset_source       "ShareGPT_Vicuna_unfiltered"
            frozen_at            ISO 8601 时间戳
            num_requests         整数
          requests[] 数组中每个元素：
            request_id           字符串（如 "mixed-0001"）
            prompt               完整 prompt 文本
            max_tokens           整数（output 上限）
            arrival_time_ms      浮点（Poisson 生成的到达时间戳，ms）
            metadata.actual_prompt_tokens  整数（tokenizer 精确计数）
    [3-3] 为每个 trace 文件计算 SHA-256 hash。将所有 hash 和生成参数写入
          manifest.json。
    [3-4] 复现性验证：用相同参数（seed=42 + 相同 ShareGPT 数据 + 相同 tokenizer）
          重新生成一次全部 trace，比对 hash 是否完全一致。如不一致则排查
          非确定性来源（如 tokenizer 版本差异）并修复后重试。
    [3-5] Sanity check：用 preempt-evict 对每个 trace 跑 1 run（不计入正式
          结果），确认：
          - vLLM 服务正常启停
          - trace 中所有请求均已发送
          - 所有请求的 RunResult 均已收集（含失败请求的 error 记录）
    [3-6] 将 trace 文件和 manifest.json 提交到 experiments/vllm/traces/ 目录。

输出产物：
    - experiments/vllm/traces/*.json       6 个 frozen trace 文件
    - experiments/vllm/traces/manifest.json hash + 生成参数完整记录

Gate → Phase 4：
    ✓ 所有 trace 文件 SHA-256 hash 复现验证通过
    ✓ preempt-evict sanity check 在每个 trace 上均成功完成
    ✓ manifest.json 已提交

冻结规则（进入 Phase 4 后不可变更的项）：
    • prompt 文本
    • arrival_time_ms 序列
    • max_tokens
    • 请求数量
    • 请求顺序
    变更上述任何一项 → Phase 3–7 所有已有结果作废，从 Phase 3 开头重做。

────────────────────────────────────────────────────────────────────────────────
Phase 4：观测指标插桩与 Sanity Check
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 3 完成，frozen trace 已冻结
GPU ：需要

目标：
    端到端验证所有指标的采集链路，确认从请求发送到 RunResult JSON 到
    analysis.py 聚合的完整数据流无缺失、无异常。

动作清单：
    [4-1] 选取 1 个 workload（Mixed）的 rate_mid trace，分别用 preempt-evict
          和 bidkv 各跑 1 个完整 run。
    [4-2] 逐一验证以下指标已成功采集：
          ┌──────────────────────────────┬───────────────────────────────┐
          │ 指标                         │ 采集来源                      │
          ├──────────────────────────────┼───────────────────────────────┤
          │ TTFT (ms)                    │ SSE streaming 第一个 data:    │
          │ TPOT (ms)                    │ (finish - first_token) /      │
          │                              │  (output_tokens - 1)          │
          │ Total latency (ms)           │ finish - submit               │
          │ Throughput (req/s)           │ run 级计算                    │
          │ SLO attainment count         │ TTFT < 2000ms 的请求数        │
          │ Completion / abort / error   │ 请求级状态统计                │
          │ Eviction count               │ server-side 日志或 hook       │
          │ Peak KV utilization (%)      │ server-side 日志或 hook       │
          └──────────────────────────────┴───────────────────────────────┘
    [4-3] 如果 eviction count 和 KV utilization 无法从 server-side 获取：
          → 记录为"不可用"，在 §7 中标记为降级处理。
          → 论文中改用 client-side 可观测指标（TTFT 差异、throughput 差异）
            间接论证 eviction 效果，并在 Limitation 中说明。
          → 这不阻塞后续阶段，但需要调整 Figure 4 的内容方案。
    [4-4] 验证 RunResult JSON 格式包含 §7 中定义的全部指标字段。
    [4-5] 验证 analysis.py 可以正确聚合 2 个 sample run：
          - 计算 P50 / P95
          - 计算 mean ± CI95（虽然只有 2 个 run 无法得到有意义的 CI，
            此处仅验证计算逻辑不报错）
          - 生成 CSV 和 LaTeX 输出格式

输出产物：
    - sanity_check_results/   2 个 sample run 的完整 RunResult JSON
    - metrics_checklist.md    各指标可用性确认 + 降级决策记录

Gate → Phase 5：
    ✓ 所有主指标（throughput / TTFT / TPOT / SLO attainment / completion rate）
      均可正确采集，数值在合理范围内（非全零、非全 inf）
    ✓ analysis.py 聚合逻辑验证通过
    ✓ metrics_checklist.md 已完成，所有降级决策已记录

────────────────────────────────────────────────────────────────────────────────
Phase 5：主实验全量运行
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 4 完成
GPU ：需要（这是实验的核心阶段，预估 30–36 小时）

目标：
    运行完整实验矩阵（8 策略 × 2 workloads × 3 rates × 3 runs = 144 runs），
    收集 Table 1 + Figure 3 所需的全部数据。

动作清单：
    [5-1] 按以下优先级顺序逐策略运行。每个 strategy 启动一次 vLLM 服务，
          跑完该 strategy 下所有 (workload, rate, run) 组合后停止服务。

          优先级 1（核心策略，必须最先完成）：
            ① preempt-evict   baseline anchor
            ② h2o-style        最强 heuristic baseline
            ③ bidkv            本文方法
          完成后即可构建 Table 1 草稿和 Figure 3 主曲线。

          优先级 2（补全实验矩阵）：
            ⑤ static-random
            ⑥ uniform
            ⑦ global-nobid
            ⑧ slack-aware
          完成后 Table 1 第一版完整可用。

    [5-2] 每个 strategy 内部的运行顺序：
          for workload in [Mixed, Long-context]:
              for rate in [rate_low, rate_mid, rate_high]:
                  for run_idx in [0, 1, 2]:
                      执行单次 run → 立即保存 RunResult JSON

    [5-3] 每个 run 完成后立即将 RunResult JSON 写入磁盘。
          命名：{strategy}__{workload}__rate{X.X}__r{run_idx}.json
          不要等整个 strategy 跑完再批量保存。

    [5-4] 每个 strategy 运行完毕后，做 quick sanity check：
          - throughput > 0 对所有 run
          - success rate > 50%（排除全部失败的 run）
          - TTFT 分布不全为 0 或 inf
          如有异常 → 记录 warning，不自动重跑，继续下一个 strategy。

    [5-5] Overload failure 处理规则：
          如果某 (strategy, workload, rate) 的连续 3 runs 全部因 OOM 或
          全部请求 timeout 而 abort：
          → 标记该组合为 "overload_failure"
          → 不重试
          → 在最终结果汇总中显示为 "×" 或 "OOL"（out-of-limit）
          → 该数据仍保留在原始结果目录中

    [5-6] GPU 内存管理：
          每个 strategy 停止服务后，等待 3 秒，然后检查 nvidia-smi 显存。
          如果残留 > 2GB → 额外等待 30 秒后再检查一次。
          如果仍有残留 → 记录 warning，继续下一个 strategy。
          （不要因显存残留中断整个实验。）

输出产物：
    - results/vllm_YYYYMMDD/    所有 RunResult JSON 文件
    - results/vllm_YYYYMMDD/run_summary.json  运行状态汇总
      （每个组合的状态：completed / aborted / overload_failure / server_crash）

Gate → Phase 6：
    ✓ 7 个策略全部完成（含标记为 overload_failure 的组合）
    ✓ 3 个核心策略（preempt-evict / h2o-style / bidkv）在所有
      冻结 rate 上至少有 2/3 runs 成功完成
    ✓ 主指标数据无明显异常（如 bidkv throughput 低于 static-random 等反常现象；
      若出现需人工审视原因后再决定是否继续）

失败回滚规则：
    - 如果 3 个核心策略中有任何 1 个在 rate_mid 上 3 runs 全部失败
      → 退回 Phase 2 重新 calibrate rate
      → Phase 3–5 所有结果作废
    - 如果非核心策略失败但核心策略正常
      → 不影响主实验结论，该策略在 Table 1 中标注 "×"

────────────────────────────────────────────────────────────────────────────────
Phase 6：补充消融实验（可选）
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 5 完成
GPU ：需要

目标：
    补充支撑论文"为什么 BidKV 有效"的机制层消融数据。
    仅在 Phase 5 主实验结论稳固后执行。

具体规划 → 见 §10。

Gate → Phase 7：
    ✓ 消融 A 完成（或因时间限制明确跳过、在论文中注明 future work）

────────────────────────────────────────────────────────────────────────────────
Phase 7：结果清洗、汇总与论文映射
────────────────────────────────────────────────────────────────────────────────
依赖：Phase 5 完成（Phase 6 可选）
GPU ：不需要

目标：
    将原始 RunResult 转化为论文可用的 Table、Figure 和摘要统计。

动作清单：
    [7-1] 对每个 (strategy, workload, rate) 组合的 3 runs 做聚合：
          - mean ± CI95 对所有主指标
          - 如果某 run 标记为 overload_failure，在聚合结果中标注 ★
          - 如果某组合只有 2 个有效 run，CI 基于 2 个样本计算（t 分布）
    [7-2] 生成 Table 1 数据（LaTeX 格式）。
          详见 §11 的 Table 1 定义。
    [7-3] 生成 Figure 3 数据（CSV + PDF）。
          X 轴 = request rate，Y 轴 = normalized TTFT P95 和 throughput。
    [7-4] 生成 Figure 4 数据（CSV + PDF）。
          内容取决于 Phase 4 确定的可用指标方案。
    [7-5] 交叉检查：
          - 确认 bidkv 在所有 rate 点上优于所有 heuristic baseline
            （如果不是 → 论文中需针对性讨论）

输出产物：
    - paper_data/table1.tex               主对比表（LaTeX）
    - paper_data/figure3.pdf + .csv       rate sweep 曲线
    - paper_data/figure4.pdf + .csv       机制解释图
    - paper_data/full_results.csv          全部聚合后的数值表（appendix 用）

================================================================================
§4  Workload 设计
================================================================================

4.1 主线方案（已拍板）

    主实验使用 ShareGPT 数据集，通过 prompt 长度分桶形成 2 个 workload。

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Workload A：ShareGPT-Mixed（简称 "Mixed"）                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 数据源    ：ShareGPT 全量（不限 prompt 长度）                        │
    │ 采样方式  ：均匀随机采样，保持原始 prompt + output 长度自然分布        │
    │ output_len：min(原始 conversations[1] 的 token 长度, 256)            │
    │ 请求数    ：1000                                                     │
    │ 角色      ：主 workload，用于 Table 1 和 Figure 3 的核心部分          │
    │ 对标      ：与 vLLM / DistServe 论文的 ShareGPT workload 完全一致    │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Workload B：ShareGPT-Long（简称 "Long-context"）                    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 数据源    ：ShareGPT 中 prompt_len ≥ 1024 tokens 的子集              │
    │ 采样方式  ：从长 prompt 子集中均匀随机采样                            │
    │ output_len：min(原始 conversations[1] 的 token 长度, 256)            │
    │ 请求数    ：500                                                      │
    │ 角色      ：高 KV 压力 workload，放大 eviction 差异                  │
    │ 说明      ：prompt_len 下限 1024 为初始值，Phase 2 可根据实测调整     │
    └──────────────────────────────────────────────────────────────────────┘

4.2 为什么是 2 个 workload 而非 3 个

    原方案中的 "Summarization" 和 "QA" 均从 ShareGPT 按长度切片而来，在论文
    叙事上不成立：ShareGPT 是对话数据，不包含真正的摘要指令或 QA 结构，按
    长度切片并改名为 "Summarization / QA" 会被 reviewer 一眼识别。

    正确做法是诚实命名为 "Mixed" 和 "Long-context"，将 workload 差异的重点
    放在"KV 压力水平不同"上，而非伪装为"任务类型不同"。2 个 workload 已
    足够覆盖"低压力"和"高压力"两个 KV 场景。

4.3 为什么 Mixed 使用 1000 请求

    这是主 workload，承担论文最核心的结论。统计充分性要求：
    - P95 由第 50 个最慢的请求决定 → 1000 请求时有 50 个样本，方差可控
    - 1000 请求 × 3 runs 的 CI95 对 P95 的覆盖通常在 ±15% 以内
    - 500 请求的 P99 只由 5 个样本决定，方差过大，无法报告

4.4 增强版扩展（不进入主实验，视时间决定）

    - Workload C (CNN/DailyMail Summarization)
      构造 "长文章 + 摘要指令" prompt，需额外数据下载和 prompt 工程。
    - Workload D (Natural Questions QA)
      构造 "context + question" prompt，同上。
    如果完成，可在论文中增加 "generalization across task types" 一节。

================================================================================
§5  Request-Rate 设计
================================================================================

5.1 核心原则：不同 workload 的 rate 必须分别校准

    Mixed workload 平均 prompt 约 128 tokens，4 req/s 可能完全不构成 KV 压力；
    Long-context workload 平均 prompt 约 2000 tokens，2 req/s 就可能接近 OOM。
    共用一组 rate 会导致某些 workload 全部处于 underload 区间（无 eviction，
    策略无差异），某些处于 overload 区间（全部 OOM，无有效结果）。

    因此：每个 workload 拥有独立的 3 个 rate 点，在 Phase 2 pilot 中分别确定。

5.2 pilot calibration 方法

    工具  ：Phase 2 的 pilot trace（请求数减半）
    策略  ：preempt-evict + h2o-style
    扫描  ：从 0.5 req/s 起，×1.5 步进，直到 OOM/timeout > 20%

    选取 3 个正式 rate 点的目标定义：

    rate_low  ── 轻压力
        论文叙事："服务刚进入压力区，BidKV 已比 heuristic 略优"
        判据：h2o-style eviction count > 0 但 < 10%；KV util peak 70–80%；OOM = 0

    rate_mid  ── 中压力（Table 1 的主数据来源）
        论文叙事："核心对比点，BidKV 显著优于 heuristic"
        判据：eviction count 20–50%；KV util peak 85–95%；OOM/abort < 5%

    rate_high ── 高压力
        论文叙事："极端压力下各策略退化曲线差异"
        判据：eviction 非常频繁；部分弱策略 OOM < 15%；核心策略仍可跑完

5.3 异常情况处理

    见 Phase 2 动作 [2-5] 的完整描述。

5.4 冻结规则

    Rate 值在 Phase 2 结束时确定，写入 calibration_report.md。
    进入 Phase 3 后 rate 值不可修改。如在 Phase 5 发现 rate 不合适
    （如全 OOM），必须退回 Phase 2 重新 calibrate，Phase 3–5 结果作废。

    RULE RATE-FREEZE（v2.3 新增）：
    ┌─────────────────────────────────────────────────────────────────┐
    │  Once pilot calibration (Phase 2) determines the rate set for  │
    │  each workload, the rate grid is frozen with the same force    │
    │  as trace freeze (Principle P3):                               │
    │                                                                │
    │  · The rate values CANNOT be adjusted based on downstream      │
    │    strategy outcomes (e.g., "BidKV curve looks bad at this     │
    │    rate, let's re-tune").                                      │
    │  · Rate adjustment is ONLY permitted under the explicitly      │
    │    documented rollback conditions:                             │
    │    – Core strategy 3 runs all fail at rate_mid → rollback to   │
    │      Phase 2 (all Phase 3–5 results voided).                   │
    │    – Workload deemed unusable → rollback to Phase 1.           │
    │  · Any rollback voids ALL existing results, not selectively.   │
    │  · The frozen rate values are recorded in                      │
    │    calibration_report.md alongside the trace manifest.         │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
§6  Frozen Trace 设计
================================================================================

6.1 Trace 的地位

    frozen trace 是整个实验可复现性和公平性的基石。8 个策略在完全相同的请求
    序列和到达时间下运行，保证唯一变量是 eviction 策略本身（原则 P6）。
    如果 trace 不冻结，跨策略对比将被 Poisson 随机性污染。

6.2 Trace 文件组织

    数量    ：2 workloads × 3 rates = 6 个 trace 文件
    命名    ：mixed_rate{X.X}.json, long_rate{Y.Y}.json
    存放位置：experiments/vllm/traces/
    配套文件：manifest.json（SHA-256 hash + 生成参数完整记录）

6.3 Trace 字段定义

    顶层字段：
        workload_name           字符串    workload 标识
        request_rate            浮点      对应 arrival rate (req/s)
        seed                    整数      42
        dataset_source          字符串    "ShareGPT_Vicuna_unfiltered"
        frozen_at               字符串    ISO 8601 时间戳
        num_requests            整数      该 trace 中请求总数

    requests[] 数组中每个元素：
        request_id              字符串    如 "mixed-0001"
        prompt                  字符串    完整 prompt 文本
        max_tokens              整数      output 上限
        arrival_time_ms         浮点      Poisson 生成的到达时间戳 (ms)
        metadata:
            actual_prompt_tokens  整数    tokenizer 精确计数

6.4 Independent Runs 与 Trace Seed 的关系

    3 次 independent runs 使用同一个 trace 文件。

    理由：
    - run 间的随机性来自 vLLM 调度器内部决策和 GPU 时序抖动，这已经是
      "独立运行"的合理随机性来源。
    - 如果 3 runs 使用不同 seed 的 trace，跨 run 方差将混入"输入不同"的
      因素，不再是纯粹的系统噪声。
    - vLLM benchmarks/serve.py 的标准做法也是同一组 requests 多次运行。

    增强项（非主实验）：
    用 seed=43 和 seed=44 各生成一组 trace 做 cross-seed 验证，
    放 appendix 证明结论不依赖特定 prompt 抽样。

6.5 冻结规则

    进入 Phase 4 后不可变更的项目：
    • prompt 文本
    • arrival_time_ms 序列
    • max_tokens
    • 请求数量
    • 请求顺序

    需要重新生成整批 trace 的触发条件（Phase 3–7 结果全部作废）：
    • 修改 workload 的长度筛选条件
    • 修改 request rate 值
    • 修改请求数量
    • 修改 ShareGPT 数据集版本或预处理逻辑
    • 修改 seed

================================================================================
§7  指标设计
================================================================================

7.1 主指标（Table 1 / Figure 3 必选，缺一不可）

    ┌───┬───────────────────────┬──────────────────────────────────────────┐
    │ # │ 指标名称              │ 定义                                     │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 1 │ Throughput (req/s)    │ 成功完成请求数 / trace 总持续时间         │
    │   │                       │ 持续时间 = 最后请求完成时间 -             │
    │   │                       │            第一个请求到达时间             │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 2 │ TTFT P50 / P95 (ms)  │ 从请求提交到收到第一个 token 的延迟       │
    │   │                       │ 采集：SSE streaming 第一个 data: 事件     │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 3 │ TPOT P50 / P95 (ms)  │ (完成时间 - 首 token 时间) /             │
    │   │                       │ (output_tokens - 1)                      │
    │   │                       │ 反映 decode 阶段逐 token 延迟             │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 4 │ SLO Attainment (%)   │ 满足 TTFT < 2000ms 的请求占              │
    │   │                       │ 成功请求总数的比例（越高越好）            │
    └───┴───────────────────────┴──────────────────────────────────────────┘

    P99 不进入 Table 1 主列。理由：
    - Mixed 1000 请求 → P99 由 10 个样本决定，3 runs 的 CI 宽
    - Long 500 请求 → P99 由 5 个样本决定，几乎无统计意义
    处理方式：若 3 runs P99 的 CV < 20%，放 appendix；否则不报告。

7.2 辅助指标（Table 1 补充列或 Figure 4）

    ┌───┬───────────────────────┬──────────────────────────────────────────┐
    │ 5 │ Eviction Count        │ 实验期间 eviction 触发总次数              │
    │   │                       │ 区分"TTFT 好是因为 evict 更少"           │
    │   │                       │ 还是"evict 了正确目标"                   │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 6 │ Normalized Latency    │ 实际 TTFT / 空载 expected TTFT            │
    │   │                       │ 用于 Figure 3 的 Y 轴，跨 workload 可比  │
    └───┴───────────────────────┴──────────────────────────────────────────┘

7.3 机制解释指标（可选，取决于 Phase 4 插桩结果）

    ┌───┬───────────────────────┬──────────────────────────────────────────┐
    │ 7 │ Recompute Overhead ms │ 被 evict 请求的重算额外时间               │
    │   │                       │ 来源：server-side 日志（如可获取）        │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │ 8 │ Peak KV Util (%)      │ 实验期间 KV cache 最高占用率              │
    │   │                       │ 证明实验在 KV 压力区间内                  │
    └───┴───────────────────────┴──────────────────────────────────────────┘

    如果 Phase 4 确认 server-side 指标不可获取 → 论文中用 TTFT/TPOT 差异间接
    论证 eviction 效果，Figure 4 改用消融 A 的 memory pressure sweep 数据。

7.4 失败/异常指标（必须统计）

    ┌───┬───────────────────────┬──────────────────────────────────────────┐
    │ 9 │ Completion Rate (%)   │ 成功请求 / 总请求（含 OOM+timeout+abort）│
    │   │                       │ completion rate < 80% 须在论文中标注     │
    ├───┼───────────────────────┼──────────────────────────────────────────┤
    │10 │ 分类失败计数           │ OOM / Timeout / Error 各自的请求数        │
    │   │                       │ 不允许只报笼统 failure count              │
    └───┴───────────────────────┴──────────────────────────────────────────┘

7.5 关于质量指标（ROUGE / EM）

    本轮主实验不做质量评测。理由及论文推荐措辞：

    "本文关注 serving 层面的性能指标。BidKV 的 eviction 发生在 request 级别
    （evict 整个请求的 KV 后 requeue 重算），与 preempt-recompute 策略具有
    相同的质量保证——最终输出完全等价于未被 evict 的执行结果。Token 级
    eviction（如 H2O 的 attention head 修剪）的质量影响由原论文充分评估。
    跨策略的生成质量对比留作 future work。"

================================================================================
§8  Fail-Safe 与实验风控
================================================================================

8.1 设计原则

    开环 Poisson 模型的核心价值在于暴露高压下的排队效应。如果在 harness 层加
    max_inflight 硬限制，当在途请求达到上限时后续请求被阻塞等待，这实质上
    退化为闭环模型，排队延迟信息被丢失。

    因此：fail-safe 不能在"请求是否发出"层面限流，
          只能在"实验是否继续"层面做保护。

8.2 三层 fail-safe 结构

    Layer 1 ── Harness 侧（runner.py 实验编排器）

        单请求 timeout    ：120 秒
                            超时 → 记录 error="timeout"，不重试。

        单 run abort 阈值 ：连续 10 个请求 timeout
                            → 提前终止该 run
                            → 标记 run_status = "aborted"
                            → 保留已收集的结果，不删除

        连续失败终止      ：某 (strategy, workload, rate) 的 3 runs 全部 abort
                            → 标记该组合为 "overload_failure"
                            → 汇总中显示 "×" 或 "OOL"

    Layer 2 ── Server 侧

        vLLM 内部行为    ：vLLM 自身的 preemption/recompute 是引擎行为的一部分，
                           不视为"失败"，属于正常运行。

        进程崩溃处理     ：如果 vLLM 进程异常退出（非 OOM preemption，
                           而是进程死亡）：
                           → harness 检测退出 → 标记 run_status = "server_crash"
                           → 保留 crash 前已收集的数据
                           → 尝试重启服务进入下一个 run
                           → 如果连续 3 次 crash → 终止该 strategy 全部后续 run
                              → 人工介入

    Layer 3 ── 资源保护

        GPU 内存检查     ：每个 strategy 停止服务后，等 3 秒，检查 nvidia-smi。
                           显存残留 > 2GB → 等 30 秒再检查。
                           仍有残留 → 记录 warning，继续下一 strategy。
                           不因显存残留中断整个实验。

8.3 失败数据处理规则

    OOM          → 计入结果，error = "oom"
    Timeout      → 计入结果，error = "timeout"
    Server crash → 该 run 中已收集的数据保留；未完成的请求不计入分母
    所有 run     → 原始 JSON 全部保留，即使标记为 abort/failure

    禁止行为：
    • 发现某 run 结果不好就删掉重跑
    • 只保留"好"的 run、丢弃"坏"的 run
    • 在结果汇总中隐藏 failure run

8.4 实验终止条件汇总

    ┌──────────────────────────┬──────────────────────────────────────────┐
    │ 条件                     │ 处理方式                                 │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ 正常完成                 │ 所有 (strategy,workload,rate,run) 跑完   │
    │ 一个组合失败             │ 3 runs 全 abort → 标记 overload_failure  │
    │ 一个 strategy 失败       │ server 连续 crash 3 次 → 终止该 strategy │
    │ 整个实验终止             │ GPU 硬件故障 / 不可恢复的环境问题        │
    └──────────────────────────┴──────────────────────────────────────────┘
    所有提前终止均须记录原因。

================================================================================
§9  主实验矩阵与资源规划
================================================================================

9.1 实验矩阵

    8 strategies × 2 workloads × 3 rates × 3 runs = 144 runs

9.2 耗时估算

    单次 run 平均耗时估算：
    - Mixed (1000 req) @ mid rate ≈ 500–1000 秒
    - Long  (500 req)  @ mid rate ≈ 250–500 秒
    - 取中位 ≈ 10 分钟/run

    每个 strategy 运行量：
    - 2 workloads × 3 rates × 3 runs = 18 runs
    - 18 runs × 10 min = 约 3 小时/strategy
    - 额外：启动/warmup/停止 ≈ 2–3 分钟（每个 strategy 仅一次）

    总耗时：
    - 8 strategies × 3 小时 ≈ 24 小时
    - 加上 warmup、fail-safe 等待、可能的重启 → 预估 30–36 小时

9.3 执行顺序与优先级

    RULE NARRATIVE-BASELINE-SPLIT（v2.3 新增）：
    正文核心归因链（§6 主叙事，Table 1 粗体行，Figure 3 主曲线）：
      ① preempt-evict   ② h2o-style   ③ global-nobid
      ④ slack-aware      ⑤ bidkv
    次级对照 / sanity（Table 1 普通行 + Appendix）：
      ⑥ static-random   ⑦ uniform
    说明：所有 7 策略保留在实验矩阵中，全部出现在 Table 1 完整版。
    Figure 3 主曲线画 ①②⑤，其余用淡色/虚线或移入附录。

    优先级 1 ── 核心策略（必须最先完成）
        ① preempt-evict    baseline anchor
        ② h2o-style         最强 heuristic baseline
        ③ bidkv             本文方法
        产出：Table 1 草稿 + Figure 3 主曲线（3 策略版）
        预估耗时：12 小时

    优先级 2 ── 补全策略
        ④ static-random
        ⑤ uniform
        ⑥ global-nobid
        ⑦ slack-aware
        产出：Table 1 完整版
        预估耗时：12 小时

    优先级 3 ── 补充消融（见 §10）

9.4 早期结论检查点

    优先级 1 的 3 策略 × 2 workloads × 3 rates × 3 runs = 54 runs 完成后，
    即可进行初步结论判断：
    (a) BidKV 是否确实优于 h2o-style？
    (b) Figure 3 趋势是否清晰？

    如果结论明确 → 继续优先级 2 补全。
    如果结论不明确 → 暂停，审视 rate 是否在有效区间内，
    必要时退回 Phase 2 重新 calibrate（Phase 3–5 结果作废）。

================================================================================
§10  补充消融实验
================================================================================

10.1 最小必做消融

    消融 A ── Memory Pressure Sweep
    ─────────────────────────────────────────────────────
    目的  ：展示 BidKV 的优势如何随 KV 压力增大而变化。
    方法  ：Mixed workload，rate 固定在 rate_mid，
            通过调节 gpu_memory_utilization 改变 KV 池大小：
            0.70 / 0.75 / 0.80 / 0.85 / 0.90（5 个水平）
    策略  ：仅 3 个核心策略（preempt-evict + h2o-style + bidkv）
    runs  ：每个组合 3 runs
    总量  ：3 × 5 × 3 = 45 runs
    用途  ：Figure 4 的一部分——"BidKV 的优势在高压力下更显著"
    前置  ：Phase 5 完成

10.2 可选增强消融（有时间再做）

    消融 B ── Output Length Sensitivity
    ─────────────────────────────────────────────────────
    目的  ：验证不同 output 长度对 eviction 策略的影响
    方法  ：Mixed workload, rate = rate_mid,
            output_len cap = 64 / 128 / 256 / 512
    策略  ：4 核心策略
    总量  ：4 × 4 × 3 = 48 runs

    消融 C ── Cross-seed Stability（成本最低）
    ─────────────────────────────────────────────────────
    目的  ：验证结论不依赖特定 ShareGPT sample
    方法  ：用 seed=43 和 seed=44 重新生成 Mixed rate_mid trace
    策略  ：preempt-evict + bidkv
    总量  ：2 × 2 × 3 = 12 runs
    用途  ：appendix 一句话带过

10.3 不建议做的消融

    • 不同模型规模 → 单张 A6000 跑 13B 会直接 OOM
    • 不同 GPU 类型 → 只有一种硬件
    • 完全不同的数据集 → 主实验已使用社区标准 ShareGPT

================================================================================
§11  结果整理与论文映射
================================================================================

11.1 Table 1（主对比表）

    结构：
        行  = 8 个策略
        列  = Throughput | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
              SLO Attain. | Eviction Count
        数据：Phase 5 主实验的 rate_mid 结果（最有代表性的压力点）
        格式：mean ± CI95（3 runs）
        标注：abort 的组合标记 "×"，脚注说明

    按 workload 分子表：
        Table 1a：Mixed
        Table 1b：Long-context

11.2 Figure 3（rate sweep 曲线）

    X 轴 ：request rate (req/s)
    Y 轴 ：normalized TTFT P95 / throughput（双 Y 轴或分两个子图）
    曲线 ：主图画 4 个核心策略，其余 4 个放 appendix 或用淡色线
    数据点：每个 workload 的 3 个 rate 点
    误差带：3 runs 的 CI95

11.3 Figure 4（eviction 机制解释图）

    内容取决于可用指标（Phase 4 决定）：
    方案 A（server-side 指标可用）：
        (a) Eviction count vs KV utilization
        (b) Recompute overhead distribution
        (c) BidKV bid value distribution
    方案 B（仅 client-side 指标）：
        (a) Throughput vs gpu_memory_utilization（来自消融 A）
        (b) TTFT CDF 对比（展示 tail 差异）

11.3a Figure 6 — Surrogate Budget Sensitivity（RULE FIG6-DEFAULT, v2.3）

    Under vLLM Mode A (recompute fallback), Figure 6 is interpreted by
    default as **surrogate budget sensitivity** — how the scheduler's
    performance (throughput, SLO attainment) varies as the surrogate δ
    parameter changes — rather than a task-level quality budget curve.

    X 轴 ：surrogate budget parameter δ
    Y 轴 ：scheduling KPI（throughput, SLO attainment）
    不对 Y 轴承诺 task-level quality (ROUGE/EM)。

    升级为 quality budget 解释需同时满足：
    (1) Mode B 稳定  (2) task-level quality 数据可用
    (3) calibration 通过 (Pearson r ≥ 0.7 或 Spearman ρ ≥ 0.7)

11.3b Figure 3a/b — Quality 指标（RULE FIG3AB-FREEZE, v2.3）

    Under vLLM Mode A, Fig 3a/b (Quality 指标 ROUGE-1, EM/F1) 从
    Mandatory 移至 Mode B / Appendix only。
    理由：Mode A recompute 保证输出等价，quality 指标在各策略间
    identical by construction，reporting 无信息量。
    复原条件：Mode B 稳定且产生真实 quality degradation 数据。

11.4 主文 vs Appendix 分配

    主文    ：Table 1 + Figure 3 (rate sweep) + Figure 4 + Figure 5 +
              Figure 6 (surrogate budget) + Table 2 (if P-Strong) +
              Figure 7 (if P-Strong)
    Appendix：
        - Fig 3a/b Quality 指标（Mode B only）
        - P99 数据（如 CV < 20% 则报告）
        - Cross-seed 验证（消融 C）
        - 完整 8 策略 rate sweep 曲线
        - Static-Random / Uniform 详细对比
        - 原始分布图（TTFT histogram, throughput box plot）
        - 完整的 pilot calibration 数据
        - Table 2 + Figure 7（if P-Weak）

11.5 Tail 指标不稳定时的降级策略

    P95 across 3 runs 的 CV > 30%   → 仍报 P95，但 CI 标宽
    P99 across 3 runs 的 CV > 50%   → 不报 P99，脚注说明"样本量限制"
    SLO attainment 波动 > 10pp       → 改报 mean ± range 而非 CI95

    这些降级不影响主结论：throughput 和 P50 通常非常稳定。

================================================================================
§12  执行版拍板清单
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ 立刻冻结的项（无需 pilot 即可确定）                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✅ 到达模型             = Poisson 开环（与 vLLM benchmark 对齐）             │
│ ✅ 数据集               = ShareGPT_Vicuna_unfiltered                        │
│ ✅ 策略列表             = 8 个，不增不减                                     │
│ ✅ 正文主叙事 baseline  = 6 个核心 + 2 个次级 (RULE NARRATIVE-BASELINE-SPLIT)│
│ ✅ Fig 6 语义           = surrogate budget sensitivity (RULE FIG6-DEFAULT)  │
│ ✅ Fig 3a/b 状态        = Mode B / Appendix only (RULE FIG3AB-FREEZE)       │
│ ✅ Scenario A/B 切换    = 形式化规则 (RULE SCENARIO-SWITCH)                 │
│ ✅ Rate 冻结            = 与 trace 冻结同级 (RULE RATE-FREEZE)              │
│ ✅ 模型/硬件/引擎       = Llama-3.1-8B / A6000 / vLLM 0.17.1              │
│ ✅ SLO 阈值             = TTFT 2000ms, TPOT 100ms                          │
│ ✅ Independent runs      = 3 次                                             │
│ ✅ Trace seed            = 42                                               │
│ ✅ Fail-safe 规则       = timeout 120s, abort 阈值, 失败计入结果            │
│ ✅ 主指标列表           = throughput + TTFT P50/P95 + TPOT P50/P95 +        │
│                            SLO attainment + eviction count                  │
│ ✅ Workload 数量         = 2 个（Mixed + Long-context）                      │
│ ✅ 请求数量             = Mixed 1000, Long-context 500                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 必须先 pilot 后再冻结的项                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ⏳ 每个 workload 的 3 个正式 request rate 数值                               │
│ ⏳ gpu_memory_utilization 是否需要从 0.85 调整（取决于能否触发 eviction）     │
│ ⏳ Long-context workload 的 prompt_len 下限（初始 1024，可能调整）            │
│ ⏳ Eviction/KV util 等 server-side 指标是否可采集（决定 Figure 4 方案）       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 暂不执行、避免分散资源的项                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ❌ 第三个 workload（CNN/DM / NQ）               → 主实验不需要               │
│ ❌ 质量评测（ROUGE / EM）                        → 留 future work           │
│ ❌ 不同 GPU 类型 / 模型规模对比                   → 硬件条件不允许           │
│ ❌ 消融 B / D                                    → 主实验+消融A 后视时间    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 如果不按此路线执行，会直接伤害论文可信度的项                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 🚫 继续使用合成 filler 段落     → attention pattern 退化，实验结论不可信     │
│ 🚫 继续使用闭环 Semaphore 模型  → 隐藏排队延迟，reviewer 会质疑延迟数据     │
│ 🚫 跳过 pilot 直接拍板 rate     → 可能全部 underload 或 overload            │
│ 🚫 不报 TPOT                   → 缺失 decode 阶段关键指标                  │
│ 🚫 不统计 eviction count        → 无法解释 BidKV 的优势来源                 │
│ 🚫 允许 cherry-pick run        → 数据操纵，违反原则 P5                     │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
§13  执行模式修订（v2.0 新增）— vLLM Partial Eviction Crash 修复
================================================================================

背景：
    Phase C 实验执行中发现，所有 BidKV 注入策略在 long_context workload 下
    约 180s 后 crash。根因是 _free_tail_blocks() 绕过 vLLM coordinator 直接
    操作 block_pool，导致 block ownership violation 和 CUDA memory corruption。
    详见 docs/vllm-route-redesign.md §1。

修订内容：引入双执行模式（Mode A / Mode B）。

────────────────────────────────────────────────────────────────────────────────
§13.1  Mode A — Recompute Fallback（当前最高优先级）
────────────────────────────────────────────────────────────────────────────────

执行语义：
    - BidKV 负责 selection（pressure detection → bid → solver ranking）
    - vLLM 负责 execution（request-level preempt + full KV free + recompute）
    - 不修改 vLLM 源码

对主实验矩阵的影响：
    - 策略列表不变（8 个）
    - Workload / rate / trace 不变
    - 唯一变化：BidKV 注入策略的 compression execution 从 partial block free
      改为 bid-guided request-level preempt + recompute
    - Preempt-evict baseline 不受影响（本来就是 vLLM 原生 preempt）

指标语义调整：
    - "compression" = "bid-guided preemption"
    - tokens_freed = victim 请求的全部 KV token
    - 质量：recompute 后输出质量不受影响（完整重算），但有额外延迟

────────────────────────────────────────────────────────────────────────────────
§13.2  Mode B — Tail Truncation（增强）
────────────────────────────────────────────────────────────────────────────────

前提：vLLM 最小内核扩展（truncate_request_tail() API）已实现并通过测试。

执行语义：
    - BidKV 负责 selection
    - vLLM 扩展负责 safe partial KV tail truncation
    - 请求继续 decode（不 recompute），但尾部 KV 数据丢失
    - 安全条件不满足时自动 fallback 到 Mode A

对指标的影响：
    - tokens_freed = 实际被 truncate 的部分 KV token
    - 质量：尾部 KV 丢失导致输出质量可能下降（ROUGE/EM 退化）

────────────────────────────────────────────────────────────────────────────────
§13.3  实验执行优先级
────────────────────────────────────────────────────────────────────────────────

    P0  Mode A 全量 216 runs → 生成 Table 1 + Figure 3/4/5 数据
    P1  Mode B 核心策略子集 → Mode A vs Mode B 效率对比（可选增强）

    若 Mode B 在 Gate-B deadline (2026-03-31) 前未稳定，
    论文直接使用 Mode A 数据。Mode B 可作为附录或 future work。

================================================================================
  文档结束（vLLM 主实验部分）
================================================================================


================================================================================
§14  SGLang 可移植性切片（v2.1 新增）— Portability Validation Slice
================================================================================

背景与定位：
    SGLang is a portability validation slice, not a second primary quantitative
    platform. vLLM（§1–§13）承担全部定量归因（8 策略 × 全量矩阵），SGLang 仅
    验证方向一致性（directional consistency）和适配器可行性（adapter feasibility）。

    The goal is directional consistency and adapter feasibility, not numerical
    equivalence of gains across frameworks.

    关键架构差异：
    - vLLM：PagedAttention, flat block KV, block-level eviction
    - SGLang：RadixAttention, tree-based KV, token-level eviction
    因此 SGLang 的绝对数值与 vLLM 不可跨平台比较，只看平台内排序（rank order）。

    详细设计：docs/sglang-portability-slice.md

────────────────────────────────────────────────────────────────────────────────
§14.1  策略集（4 个，v2.0 修订）
────────────────────────────────────────────────────────────────────────────────

    ┌────────────────┬────────────────────────────────────────────────────────┐
    │ 策略名         │ 角色                                                   │
    ├────────────────┼────────────────────────────────────────────────────────┤
    │ Preempt-Evict  │ SGLang-Default baseline（原生逐出语义）               │
    │ Slack-Aware    │ 非拍卖竞争参照（带配额但无 bidding）                  │
    │ BidKV          │ 本文核心方法                                          │
    │ Oracle-DP      │ 理论上界（exhaustive offline DP）                     │
    └────────────────┴────────────────────────────────────────────────────────┘

    与 v1 的差异（rationale）：
    - Global-NoBid → Slack-Aware：Slack-Aware 具有非拍卖配额语义，更适合
      展示"有预算感知但无竞价"与"有竞价"的差异
    - Uniform → Oracle-DP：补上理论上界，使方向性比较有上界参照

    省略 Static-Random / H2O-Style / Global-NoBid / Uniform 的原因：
    **full attribution is performed on vLLM（§9, Table 1）**。SGLang 作为
    portability slice 不重复完整归因。
    The reduced baseline set is intentional: full attribution is performed on
    vLLM, while SGLang validates transferability under a structurally different
    runtime.

────────────────────────────────────────────────────────────────────────────────
§14.2  实验矩阵
────────────────────────────────────────────────────────────────────────────────

    4 strategies × 2 workloads × 3 rates × 3 runs = 72 runs

    Workload 和 rate 点复用 vLLM 的 §4 / §5 定义。
    Frozen trace 复用同一 ShareGPT trace 文件（seed=42）。
    SGLang 可能需要独立的 rate pilot（如 SGLang 原生预分配机制不同），
    但 trace 内容不变。

────────────────────────────────────────────────────────────────────────────────
§14.3  执行语义
────────────────────────────────────────────────────────────────────────────────

    SGLang 使用 native token-level 语义：
    - RadixAttention 原生支持 token-level KV 操作
    - adapter 通过 monkey-patch get_next_batch_to_run() 拦截调度决策
    - radix_hook.free_kv_positions() 执行 token-level KV 释放
    - 不需要 vLLM 的 recompute fallback（Mode A）或 kernel extension（Mode B）

    SGLang is not required to replicate the strongest execution semantics
    engineered for vLLM.

    Smoke test 要求（进入全量实验前必须通过）：
    - 4 个策略 × warmup (≤10 请求) 启动无 crash
    - free_kv_positions() 调用后 RadixAttention tree state 一致
    - 被释放的 token 在后续 lookup 中不再命中
    如果 smoke test 失败 → 降级为 request-level fallback（整请求逐出），
    不阻塞 portability 价值，但论文须注明。

────────────────────────────────────────────────────────────────────────────────
§14.4  Within-Platform Fairness（公平性保障）
────────────────────────────────────────────────────────────────────────────────

    Within-platform fairness must still hold on SGLang for all compared
    baselines. 虽然 SGLang 不做跨平台数值比较，但平台内的策略排序必须公平。

    强制审计项：
    (1) candidate_universe hash — 所有策略看到完全相同的候选请求集
    (2) kv_snapshot hash — 压力触发时 KV 状态快照一致
    (3) trace 一致 — 所有策略运行相同的 frozen trace
    (4) SGLang server 参数一致 — 除策略注入外，所有配置参数完全相同

    审计实现：
    - 在 scheduler_hook.py 的拦截点记录 candidate hash + KV snapshot hash
    - 每个 run 的审计日志保存到 results 目录
    - Phase 7 汇总时验证同一 (workload, rate) 下所有策略的 audit hash 一致

────────────────────────────────────────────────────────────────────────────────
§14.5  验收标准（Acceptance Criteria）
────────────────────────────────────────────────────────────────────────────────

    必须满足以下条件才视为 SGLang portability slice 实验成功：

    (1) BidKV 在 SGLang 上的 throughput rank 与 vLLM 方向一致
        （即 BidKV > Default, BidKV > Slack-Aware）
    (2) fairness audit 通过（各策略在同一 run 内 candidate hash 一致）
    (3) 3 × 2 × 3 × 3 = 54 runs 中 ≥ 90% 完成（无 crash/OOM）

    不要求：
    - SGLang 的绝对 throughput 与 vLLM 数值接近
    - SGLang 上 BidKV 的改进幅度与 vLLM 一致
    - SGLang 执行 7 策略全集

────────────────────────────────────────────────────────────────────────────────
§14.6  结果整理与论文映射
────────────────────────────────────────────────────────────────────────────────

    Table 2（Portability 对比表）：
        行  = 3 个 SGLang 策略
        列  = Throughput | TTFT P50 | TTFT P95 | Eviction Count
        数据：rate_mid 结果（与 Table 1 平行）
        格式：mean ± CI95（3 runs）
        注  ：与 Table 1 分开呈现，明确标注 "SGLang (RadixAttention)"

    Figure 7（Cross-platform consistency）：
        左 panel ：vLLM rank order（3 核心策略子集）
        右 panel ：SGLang rank order（3 策略）
        用 rank / normalized bar chart 展示方向一致性
        不报绝对数值的跨平台比较

    论文叙事模板：
        ✅ "BidKV achieves directionally consistent improvements on both
            vLLM and SGLang, despite their fundamentally different KV
            architectures."
        ✅ "The adapter integration required <N> LOC (SGLang) vs <M> LOC
            (vLLM), demonstrating practical portability."
        ❌ 避免 "BidKV achieves X% improvement on SGLang"（暗示量化精度）
        ❌ 避免 "SGLang results confirm vLLM findings"（暗示数据等价）

────────────────────────────────────────────────────────────────────────────────
§14.7  实施步骤与优先级
────────────────────────────────────────────────────────────────────────────────

    S.1  策略注册 — 在 BaselineRegistry 中添加 Slack-Aware
         的 SGLang binding（如当前仅有 vLLM binding）
    S.2  Smoke test — 3 策略 × warmup（≤10 请求）× 3 轮 → 无 crash
    S.3  Fairness audit 日志 — scheduler_hook.py 中嵌入 candidate hash 记录
    S.4  SGLang rate pilot — 用 Preempt-Evict + BidKV 做粗 rate 扫描
    S.5  Frozen trace + SGLang 兼容性验证
    S.6  全量 54 runs
    S.7  Table 2 + Figure 7 生成

    优先级关系（不阻塞 vLLM）：
    - S.1–S.3 可与 vLLM Phase C（Mode A 实现）并行
    - S.4–S.7 依赖 S.1–S.3 完成
    - SGLang 全流程不阻塞 vLLM 任何 Phase

────────────────────────────────────────────────────────────────────────────────
§14.8  Non-Goals（明确排除项）
────────────────────────────────────────────────────────────────────────────────

    ❌ 在 SGLang 上复现 vLLM 的 7 策略全集
    ❌ 在 SGLang 上做 kernel extension（Mode B）
    ❌ 在 SGLang 上做 Memory Pressure Sweep 消融
    ❌ 报告 SGLang vs vLLM 的跨平台绝对数值比较
    ❌ 使用 SGLang 数据替代 vLLM 主实验数据
    ❌ 将 SGLang 作为 production deployment 推荐平台

================================================================================
  文档结束
================================================================================
