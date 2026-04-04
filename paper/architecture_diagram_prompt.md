# BidKV 系统架构图绘制提示词

> SC 2026 论文 Figure。展示 BidKV 如何作为可插拔调度层嵌入 LLM Serving 框架。

---

## 提示词（中文版）

为一篇 SC 2026 系统论文绘制 **系统架构图**。矢量风格，学术模块框图，配色简洁（蓝/绿/橙/灰），白底，适合双栏宽度（~17cm）。

**核心风格要求**：
- **编号流程驱动**——全图用 ❶❷❸❹❺❻ 六个步骤编号标在对应箭头/连线旁。读者沿编号顺序即可理解完整流程
- **精炼文字**——模块标注简短，公式用数学排版，不加纯装饰图标
- **三层结构**：上方=LLM Serving 框架 / 中间薄条=FrameworkAdapter / 下方=BidKV 可插拔层
- 红/橙色调串联全部六步，形成视觉主线
- 图只讲 BidKV 自身机制，不与其他策略做对比

---

### 一、总体布局（三层结构 + 编号路径）

```
┌──────────────────────────────────────────────────────────────────┐
│  LLM Serving Framework (vLLM / SGLang)                [实线框]   │
│                                                                  │
│   ┌──────────┐  ─green admit─▶  ┌──────────────┐   ┌──────────┐│
│   │ Waiting  │                   │ Running Batch │◄─►│ KV Cache ││
│   │ Queue    │                   │ R1 R2 [R3] R4 │   │ ████▓░░  ││
│   │ ▫ ▫ ▫   │  ◀──── ❻ ────── R3 ❺             │   │ 95% ❶⚠ ││
│   └──────────┘  preempt+recomp  └──────┬─────────┘   └──┬───────┘│
│                                   ❹↑select victim  ❷↑read usage │
╞══════════════ FrameworkAdapter  ·  vLLM │ SGLang ════════════════╡
│                                   ❹↑                ❷↑          │
│  BidKV Scheduling Layer                              [虚线框]    │
│                                                                  │
│   ┌───────────┐  ❸→   ┌───────────┐  ❸→   ┌───────────┐       │
│   │❷Pressure │──────▶│   Bid     │──────▶│  Greedy   │       │
│   │ Detector  │        │Generation │        │  Solver   │       │
│   │ KV > 90%  │        │ δ=f(c,p,P)│       │ U=r/(δ+ε) │──┐    │
│   └───────────┘        └───────────┘       └───────────┘  │    │
│                                                     ❹ select│    │
│                                                     victim  ↑    │
└──────────────────────────────────────────────────────────────────┘

❷ = Detector 主动向上读取 KV 使用率（穿过 Adapter），检测到 >90% 后激活管线
❹ = Solver 向上穿过 Adapter 选出 victim R3
阅读路径：❶ → ❷ → ❸ → ❹ → ❺ → ❻
```

> **FrameworkAdapter 位置**：独立为一条 **灰色薄横条**，位于上方框架层和下方 BidKV 层**之间**。不属于任何一层，而是两层之间的桥接接口。表示所有跨层交互（❷ 读取 KV stats、❹ select victim、❻ preempt）都经过此 adapter。
>
> **❻ 驱逐表现**：R3 被选中（❺）后，从 Running Batch 中直接画一条 **红色粗直线箭头** 水平指向 Waiting Queue，箭头旁标 ❻ + `preempt + recompute`。表示 R3 被踢出 Running、回到 Waiting Queue 队尾。不使用弯曲弧线，直接横向箭头更直观。

---

### 二、上方区域：LLM Serving Framework

实线蓝灰色背景框。三个模块横向排列：

**Waiting Queue**（左）
- 圆角矩形，内画 3-4 个小方块纵向堆叠表示排队请求
- 底部小标签：`SJF reorder`
- 步骤 ❻ 的红色箭头终点：R3 被送回此处队尾

**Running Batch**（中）
- 画 4-6 个不同高度的彩色竖条 (R1–R6)，高度正比于 KV 占用量，直观展示请求异构性
- 步骤 ❺：其中 R3 用红色虚线圈出，旁边标 ❺，表示被选中的 victim
- 步骤 ❹：从下方穿过 FrameworkAdapter 有粗红色箭头向上指到 R3，旁边标 ❹
- 底部小标签：`GPU decode`

**KV Cache**（右，与 Running Batch 紧邻）
- 竖向水位条，填充到 ~95%，顶部 **红色溢出区域**
- 步骤 ❶：在红色区域旁标注 ❶ + `95% ⚠`，这是全流程的起点（可观测状态）
- 步骤 ❷ 的终点在此：Pressure Detector **向上** 读取 KV 使用率（穿过 FrameworkAdapter），箭头旁标 ❷ + `read usage`。注意箭头方向是 **从下往上**（Detector 主动轮询），不是 KV Cache 向下发信号
- KV Cache 与 Running Batch 之间用双向细箭头连接

**连接箭头**：
- Waiting → Running：绿色实线箭头，标注 `admit`（正常调度流，非 BidKV 流程）
- 步骤 ❻：从 Running Batch 中被圈出的 R3，画一条 **红色粗直线箭头水平指向 Waiting Queue**，箭头旁标 ❻ + `preempt + recompute`。R3 直接从 Running 被推入 Waiting Queue 尾部——这是 BidKV 的最终执行效果。不用弧线，直线箭头更直观。可在箭头上方小标注 `KV freed`

---

### 三、下方区域：BidKV Scheduling Layer

虚线边框 + 浅绿色背景（虚线=可插拔、非侵入）。

步骤 ❸ 覆盖此层内部的三阶段管线，用 **蓝色实线箭头** 串联：

| 卡片 | 标题 | 核心内容 |
|------|------|---------|
| Pressure Detector | 步骤 ❷：主动向上读取 KV 使用率，发现 >90% 后激活管线 | 阈值 `KV > 90%` |
| Bid Generation | 为每个 running 请求计算 disruption cost 并生成 bid | `δ = f(completion, prompt, preemptions)` |
| Greedy Solver | 按 utility 排序贪心选取 | **U = r / (δ + ε)** |

> **为什么不叫 "Scoring → Bid"**：当前 Mode A 中没有 token-level 注意力评分。每个请求的 disruption cost (δ) 直接从请求状态（完成进度、prompt 长度、被驱逐次数）计算，然后包装为 CompressionBid 对象。因此中间阶段叫 "Bid Generation" 更准确。

步骤 ❸ 标在管线箭头上方，概括为 `compute δ → create bids → rank by U`。
管线箭头标注简短数据名：`N requests` → `BidPool` → `Accepted`。

步骤 ❹ 从 Solver 输出一条 **粗红色实线箭头** 向上穿过 FrameworkAdapter 薄条和分界线，指向 Running Batch 中的 R3，旁标 ❹ + `select victim`。

---

### 四、编号标注的视觉规范

- 每个编号 ❶–❻ 用 **红色/橙色圆圈包裹数字** 的形式，大于周围文字（约 11pt），放在对应箭头起点或旁边
- 编号之间的箭头/弧线全部用 **红色/橙色** 色调，比非编号箭头更粗（2-3px vs 1px）
- BidKV 内部管线箭头用蓝/绿色，与红色编号路径视觉分开
- 绿色 `admit` 箭头是背景流，不编号
- Detector 到 KV Cache 的 ❷ 向上箭头用红/橙虚线，与向下的 ❹ 实线视觉区分

---

### 五、视觉规范

**色彩** （4 色系）：
| 语义 | 色彩 | 用于 |
|------|------|------|
| 框架原生 | 蓝色系 #4A90D9 | Waiting, Running, KV Cache 模块 |
| BidKV | 绿色系 #5CB85C | 三个阶段卡片、BidKV 层背景 |
| 编号流程 | 红/橙 #E74C3C | ❶–❻ 编号圈、因果链箭头、victim 标记 |
| 辅助 | 灰色 #95A5A6 | 背景框、辅助虚线、次要标注 |

**边框**：实线=框架原生 / 虚线=BidKV 可插拔

**字号**：模块标题 10pt，编号圈 11pt，公式 9pt，箭头标注 8pt。全部无衬线体。

**不要画以下内容**：
- 长段落文字（每处标注 ≤ 5 个词）
- 与其他策略的对比框 / callout
- 纯装饰图标
- 内嵌数据结构卡片 / callout 气泡
- Mode B / token-level truncation
- Attention weight 细节
- 多 GPU 拓扑

---

## 提示词（English Version）

Draw a **system architecture diagram** for an SC 2026 paper. BidKV is a pluggable scheduling layer for LLM serving that decides **which request to evict** when KV cache is full.

**Style**: Academic vector block diagram. Colors: blue (framework), green (BidKV), red/orange (numbered flow), gray (auxiliary). White background. Double-column width (~17cm). **Concise labels, no decorative icons, no inset data-structure cards.** The figure focuses solely on BidKV's own mechanism, no comparison with other strategies.

**Core organizing principle — Six numbered steps ❶–❻**: The entire figure is driven by a numbered flow path. Each step number appears as a **red/orange circled digit** next to the corresponding arrow. A reader follows ❶→❷→❸→❹→❺→❻ to understand the complete mechanism. The numbered arrows use red/orange and are thicker (2-3px) than other arrows.

**Layout — Three Layers**:

```
┌──────────────────────────────────────────────────────────────────┐
│  LLM Serving Framework                                  [solid] │
│                                                                  │
│   ┌──────────┐  ─admit (green)─▶  ┌─────────────┐  ┌──────────┐│
│   │ Waiting  │                     │Running Batch│◄►│ KV Cache ││
│   │ Queue    │                     │R1 R2 [R3] R4│  │ ████▓░░  ││
│   │ (SJF)    │◀══ ❻ preempt ═════ R3 ❺          │  │ 95% ❶⚠ ││
│   └──────────┘                     └──────┬──────┘  └──┬───────┘│
│                                      ❹↑ select    ❷↑ read     │
╞═══════════════ FrameworkAdapter · vLLM │ SGLang ════════════════╡
│                                      ❹↑            ❷↑          │
│  BidKV Scheduling Layer                              [dashed]    │
│  ┌───────────┐  ❸→  ┌───────────┐  ❸→  ┌───────────┐          │
│  │❷Pressure │─────▶│   Bid     │─────▶│  Greedy   │          │
│  │ Detector  │      │Generation │      │  Solver   │──❹──┐   │
│  │ KV > 90%  │      │ δ=f(c,P)  │      │ U=r/(δ+ε) │     │   │
│  └───────────┘      └───────────┘      └───────────┘     │   │
│                                               select victim↑   │
└──────────────────────────────────────────────────────────────────┘
❷ = Detector actively reads KV usage UPWARD (through Adapter); detects >90%
❹ = Solver selects victim UPWARD (through Adapter) to R3 in Running Batch
```

**Design notes**:
- **FrameworkAdapter is an independent thin horizontal bar** between the Framework layer and BidKV layer. NOT inside either layer. Visually ~1/5 the height of either layer. Gray background, small label centered. Shows it is the bridge between the two systems.
- ❷ and ❹ both go **UPWARD through** the FrameworkAdapter bar: ❷ is the Detector reading KV stats (dashed red upward arrow), ❹ is the Solver's victim selection (solid red upward arrow). Both reinforce the Adapter's role as the interface.
- Step ❻: R3 uses a **direct horizontal red arrow** from its position in Running Batch pointing left to Waiting Queue. NOT a curved arc. This direct arrow clearly shows the preempted request being pushed back to the queue.

**Top: LLM Serving Framework** (solid blue-gray border)
Three modules side-by-side:
1. **Waiting Queue** (left): stacked blocks = queued requests. Label: "SJF reorder". Step ❻ direct arrow ends here (R3 returns to queue tail).
2. **Running Batch** (center): 4-6 vertical bars of varying height (= KV footprint per request), showing request heterogeneity. Step ❺: R3 circled in red dashed line with ❺ label. Step ❹: bold red arrow from below passes through FrameworkAdapter and points to R3 with ❹ label.
3. **KV Cache** (right, adjacent to Running Batch): vertical fill-level bar ~95% full, top portion red. Step ❶: red zone with ❶ + `95% ⚠` (observable state, the starting point). Step ❷ terminates here: Pressure Detector's **upward** arrow reads KV usage through FrameworkAdapter, with ❷ + `read usage`. Arrow direction is **bottom-up** (Detector actively polls), NOT top-down.

KV Cache ↔ Running Batch: thin bidirectional arrow.
Green `admit` arrow (Waiting→Running): normal scheduling flow, not numbered.
Step ❻: **bold red straight horizontal arrow** from R3 pointing left to Waiting Queue tail, with ❻ + `preempt + recompute`. Small annotation above arrow: `KV freed`. Not a curved arc — a direct push.

**Middle: FrameworkAdapter** (independent thin gray bar)
Spans the full width between the two layers. Height ~1/5 of either layer. Gray background. Centered label: `FrameworkAdapter · vLLM | SGLang`. Arrows ❷ and ❹ pass through it. This shows it is the interface bridging framework and BidKV.

**Bottom: BidKV Layer** (dashed green border = pluggable)
Three stage cards in a pipeline (Step ❸ spans across these):

| Card | Title | Content |
|------|-------|---------|
| Pressure Detector | Step ❷: actively reads KV usage upward, detects >90%, activates pipeline | threshold: `KV > 90%` |
| Bid Generation | Compute disruption cost per request, wrap as bid | `δ = f(completion, prompt, preemptions)` |
| Greedy Solver | Rank by utility, greedy select | **U = r/(δ+ε)** |

Step ❸ label above pipeline arrows: `compute δ → create bids → rank by U`.
Pipeline arrows (blue): `N requests` → `BidPool` → `Accepted`.
Step ❹: bold red arrow from Solver upward through FrameworkAdapter to R3 in Running Batch.

**Do NOT include**: comparison callouts with other strategies, decorative icons, inset data-structure cards or callout bubbles, Mode B token-level truncation, attention weight internals, multi-GPU topology, long text (max 5 words per label).

---

## 与论文章节的对应关系

| 图中区域 | 论文章节 |
|---------|---------|
| Waiting Queue + SJF | §5.1 (a) |
| Running Batch + Reorder | §5.1 (d) |
| KV Cache + Pressure | §2.1 |
| ① Pressure Detector | §4.4 |
| ② Bid Generation | §4.2, Eq.2 |
| ③ Constrained Solver | §4.3, Alg.1, Eq.1 |
| ④ BidAcceptance | §4.1 |
| Preempt + Recompute arrow | §5.1 |
| FrameworkAdapter (middle bar) | §4.4 + §5.1/§5.2 |

## 参考格式

- 双栏宽度 ~17cm，高度 6-8cm
- PDF 矢量图优先；PNG ≥ 300 DPI
- 无衬线字体（Helvetica/Arial），与 ACM sigconf 协调
