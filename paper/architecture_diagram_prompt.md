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
│   │ ▫ ▫ ▫   │  ◀──── ❻ ────── R3 ──❺──▶        │   │ 95% ❶⚠ ││
│   └──────────┘  requeue+recomp  └──────┬─────────┘   └──┬───────┘│
│                                   ❹↑victim R3     ❷↑read usage │
╞══════════════ FrameworkAdapter  ·  vLLM │ SGLang ════════════════╡
│                                   ❹↑                ❷↑          │
│  BidKV Scheduling Layer                              [虚线框]    │
│                                                                  │
│   ┌───────────┐  ❸→   ┌───────────┐  ❸→   ┌───────────┐       │
│   │❷Pressure │──────▶│   Bid     │──────▶│  Greedy   │       │
│   │ Detector  │        │Generation │        │  Solver   │       │
│   │ KV > 90%  │        │ δ=f(c,p,P)│       │ U=r/(δ+ε) │──┐    │
│   └───────────┘        └───────────┘       └───────────┘  │    │
│                                                     ❹ victim│    │
│                                                     select  ↑    │
└──────────────────────────────────────────────────────────────────┘

❶ = KV Cache 水位 95%（可观测状态，起点）
❷ = Adapter 从 Framework 读取 KV stats → 传给 BidKV Detector，Detector 检测到 >90%
❸ = BidKV 内部管线：Detector → Bid Generation → Solver
❹ = BidKV Solver 输出 victim=R3 → 返回给 Adapter
❺ = Adapter 调用 Framework 的 _preempt_request(R3) → Framework 释放 R3 的 KV blocks
❻ = 同一 _preempt_request() 的另一效果：Framework 将 R3 移入 Waiting Queue 尾部
阅读路径：❶ → ❷ → ❸ → ❹ → ❺ → ❻
```

> **FrameworkAdapter 位置与角色**：独立为一条 **灰色薄横条**，位于上方框架层和下方 BidKV 层**之间**。Adapter 是**编排者**：它同时持有 Framework API 句柄和 BidKV 管线引用，所有跨层交互都由它发起。
>
> **三层职责划分**：
> - **Framework**（上层）：拥有 Waiting Queue、Running Batch、KV Cache、Scheduler。提供可读取的 KV stats API 和可调用的 `_preempt_request()` API。不知道 BidKV 的存在。
> - **Adapter**（中间条）：编排者。向上调 Framework API 读 KV stats，向下调 BidKV 管线获取 victim 决策，再向上调 Framework 的 `_preempt_request()` 执行驱逐。
> - **BidKV**（下层）：纯计算层。接收 KV stats + 请求信息，返回 victim 列表。从不直接操作 Framework。
>
> **❹❺❻ 三步分解**（选中 R3 之后的完整流）：
>
> R3 始终是**被动对象**，从不是信息的接收者。
>
> - **❹ 决策返回** [BidKV → **Adapter**]：BidKV Solver 计算完成，输出 `victim=R3`，返回给 Adapter。视觉效果：粗红色箭头从 Solver **向上**，终点是 **FrameworkAdapter 薄条**。箭头旁标 ❹ + `victim = R3`。R3 在上方 Running Batch 模块内用红色虚线圈出，作为**旁注标识**（标识谁被选中）。
> - **❺ KV 释放** [**Adapter** 调用 Framework]：Adapter 收到 victim 后，调用 Framework 的 `_preempt_request(R3)`。Framework 释放 R3 的全部 KV blocks。视觉效果：从被圈出的 R3 画一条 **橙色箭头** 指向 KV Cache 水位条，KV 水位从 95% 回落到 ~80%。箭头旁标 ❺ + `KV freed`。
> - **❻ 回队重算** [同一 Framework API 的另一效果]：同一 `_preempt_request()` 调用将 R3 从 running 列表移除，放入 waiting 列表尾部。视觉效果：从 R3 画一条 **红色粗直线箭头水平指向 Waiting Queue** 尾部。箭头旁标 ❻ + `requeue + recompute`。
>
> 三步的职责归属：❹ = BidKV 返回决策给 Adapter；❺❻ = Adapter 调用 Framework API，Framework 执行原生驱逐。BidKV 从不直接接触 Framework，Adapter 是唯一的中间者。

---

### 二、上方区域：LLM Serving Framework

实线蓝灰色背景框。三个模块横向排列：

**Waiting Queue**（左）
- 圆角矩形，内画 3-4 个小方块纵向堆叠表示排队请求
- 底部小标签：`SJF reorder`
- 步骤 ❻ 的红色箭头终点：R3 被送回此处队尾

**Running Batch**（中）
- 画 4-6 个不同高度的彩色竖条 (R1–R6)，高度正比于 KV 占用量，直观展示请求异构性
- R3 用红色虚线圈出，作为**旁注标识**（表明谁被选中）
- 步骤 ❹：粗红色箭头从下方 BidKV 层向上，**终点是 FrameworkAdapter 薄条**（表示决策交给 Adapter）。旁标 ❹ + `victim = R3`
- 步骤 ❺ [Adapter 调用 Framework]：从被圈出的 R3 画一条橙色箭头指向右侧 KV Cache 水位条，旁标 ❺ + `KV freed`。这是 Adapter 调用 `_preempt_request(R3)` 后，Framework 释放 KV 的效果
- 步骤 ❻ [同一 API 调用]：从 R3 画一条红色粗直线箭头水平指向左侧 Waiting Queue 尾部，旁标 ❻ + `requeue + recompute`。同一 `_preempt_request()` 的另一效果
- 底部小标签：`GPU decode`

**KV Cache**（右，与 Running Batch 紧邻）
- 竖向水位条，填充到 ~95%，顶部 **红色溢出区域**
- 步骤 ❶：在红色区域旁标注 ❶ + `95% ⚠`，这是全流程的起点（可观测状态）
- 步骤 ❷ 的起点在此：**Adapter** 主动从 Framework 的 KV Cache 读取使用率，传给 BidKV Detector。箭头方向是 **从下往上**（Adapter 主动轮询 Framework），箭头旁标 ❷ + `read usage`
- KV Cache 与 Running Batch 之间用双向细箭头连接

**连接箭头**：
- Waiting → Running：绿色实线箭头，标注 `admit`（正常调度流，非 BidKV 流程）
- ❹❺❻ 的箭头已在 Running Batch 模块中描述（见上）。❺ 和 ❻ 从 R3 分出两条不同方向的箭头：❺ 向右到 KV Cache（释放），❻ 向左到 Waiting Queue（回队）。两者都是 Adapter 调用 Framework 的 `_preempt_request()` 后的效果

---

### 三、下方区域：BidKV Scheduling Layer

虚线边框 + 浅绿色背景（虚线=可插拔、非侵入）。

步骤 ❸ 覆盖此层内部的三阶段管线，用 **蓝色实线箭头** 串联：

| 卡片 | 标题 | 核心内容 |
|------|------|---------|
| Pressure Detector | 步骤 ❷：Adapter 从 Framework 读取 KV 使用率，传入此模块。发现 >90% 后激活管线 | 阈值 `KV > 90%` |
| Bid Generation | 为每个 running 请求计算 disruption cost 并生成 bid | `δ = f(completion, prompt, preemptions)` |
| Greedy Solver | 按 utility 排序贪心选取 | **U = r / (δ + ε)** |

> **为什么不叫 "Scoring → Bid"**：当前 Mode A 中没有 token-level 注意力评分。每个请求的 disruption cost (δ) 直接从请求状态（完成进度、prompt 长度、被驱逐次数）计算，然后包装为 CompressionBid 对象。因此中间阶段叫 "Bid Generation" 更准确。

步骤 ❸ 标在管线箭头上方，概括为 `compute δ → create bids → rank by U`。
管线箭头标注简短数据名：`N requests` → `BidPool` → `Accepted`。

步骤 ❹ 从 Solver 输出一条 **粗红色实线箭头** 向上，**终点是 FrameworkAdapter 薄条**（决策交给 Adapter）。旁标 ❹ + `victim = R3`。R3 在上方 Running Batch 模块内被圈出作为旁注标识。这是 BidKV 层唯一的向上输出——Adapter 收到决策后调用 Framework API 执行 ❺（KV 释放）和 ❻（回队）。

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
│   │ (SJF)    │◀══ ❻ requeue ═════ R3 ──❺──▶     │  │ 95% ❶⚠ ││
│   └──────────┘                     └──────┬──────┘  └──┬───────┘│
│                                      ❹↑ victim    ❷↑ read     │
╞═══════════════ FrameworkAdapter · vLLM │ SGLang ════════════════╡
│                                      ❹↑            ❷↑          │
│  BidKV Scheduling Layer                              [dashed]    │
│  ┌───────────┐  ❸→  ┌───────────┐  ❸→  ┌───────────┐          │
│  │❷Pressure │─────▶│   Bid     │─────▶│  Greedy   │          │
│  │ Detector  │      │Generation │      │  Solver   │──❹──┐   │
│  │ KV > 90%  │      │ δ=f(c,P)  │      │ U=r/(δ+ε) │     │   │
│  └───────────┘      └───────────┘      └───────────┘     │   │
│                                               victim R3  ↑   │
└──────────────────────────────────────────────────────────────────┘
❶ = KV Cache at 95% (observable state, starting point)
❷ = Adapter reads KV stats from Framework → passes to BidKV Detector; detects >90%
❸ = BidKV internal pipeline: Detector → Bid Generation → Solver
❹ = BidKV Solver returns victim=R3 to Adapter (decision output)
❺ = Adapter calls Framework's _preempt_request(R3) → Framework frees R3's KV blocks
❻ = Same _preempt_request() also moves R3 to Waiting Queue tail for future recompute
```

**Design notes**:
- **FrameworkAdapter is an independent thin horizontal bar** between the Framework layer and BidKV layer. NOT inside either layer. Visually ~1/5 the height of either layer. Gray background, small label centered.
- **Adapter is the orchestrator**: it holds references to both Framework APIs and BidKV pipeline. All cross-layer interactions are initiated by the Adapter.
- **Three-layer responsibilities**:
  - **Framework** (top): owns Waiting Queue, Running Batch, KV Cache, Scheduler. Exposes read API (KV stats) and write API (`_preempt_request()`). Unaware of BidKV.
  - **Adapter** (middle bar): orchestrator. Reads KV stats FROM Framework (❷), passes them DOWN to BidKV (❸), receives victim decision FROM BidKV (❹), then calls `_preempt_request()` ON Framework (❺❻).
  - **BidKV** (bottom): pure computation. Receives data, returns victim list. Never directly touches Framework.
- ❷ goes **UPWARD through** the Adapter bar (Adapter reads KV stats from Framework, dashed red). ❹ goes **UPWARD to** the Adapter bar (BidKV returns victim to Adapter, solid red). Both reinforce the Adapter as the sole intermediary.
- **❹❺❻ three-step decomposition** (complete flow after victim selected):
  - **Core principle: R3 is always a passive object**, never the recipient of information.
  - **❹ Decision return** [BidKV → **Adapter**]: BidKV Solver outputs `victim=R3`, returns to Adapter. Bold red arrow from Solver upward, **terminating at the Adapter bar**. Label: ❹ + `victim = R3`. R3 is circled with red dashed line inside Running Batch as an **annotation** (identifying who was selected).
  - **❺ KV freed** [**Adapter** calls Framework]: Adapter calls Framework's `_preempt_request(R3)`. Framework frees R3's KV blocks. Orange arrow from circled R3 rightward to KV Cache bar, KV level drops from 95% to ~80%. Label: ❺ + `KV freed`.
  - **❻ Requeue** [same Framework API]: Same `_preempt_request()` call also moves R3 from running to waiting tail. **Direct horizontal red arrow** from R3 leftward to Waiting Queue tail. Label: ❻ + `requeue + recompute`. NOT a curved arc.
- Key distinction: ❹ = BidKV returns decision **to Adapter**; ❺❻ = **Adapter calls Framework API**, Framework executes native preemption. BidKV never directly touches Framework.

**Top: LLM Serving Framework** (solid blue-gray border)
Three modules side-by-side:
1. **Waiting Queue** (left): stacked blocks = queued requests. Label: "SJF reorder". Step ❻ arrow ends here (R3 returns to queue tail).
2. **Running Batch** (center): 4-6 vertical bars of varying height (= KV footprint per request), showing request heterogeneity. R3 circled in red dashed line (annotation = selected victim). Three arrows at/around R3:
   - Step ❹: bold red arrow from **below**, **terminating at the Adapter bar** (BidKV returns decision to Adapter). Label: ❹ + `victim = R3`. R3's circle is an annotation above, not the arrow's target.
   - Step ❺ [Adapter → Framework]: orange arrow from R3 points **right** to KV Cache bar, label ❺ + `KV freed` (KV bar drops from 95% to ~80%)
   - Step ❻ [same API call]: bold red arrow from R3 points **left** to Waiting Queue tail, label ❻ + `requeue + recompute`
3. **KV Cache** (right, adjacent to Running Batch): vertical fill-level bar ~95% full, top portion red. Step ❶: red zone with ❶ + `95% ⚠` (observable state, the starting point). Step ❷: **Adapter** reads KV usage from here (upward arrow from below through Adapter bar), with ❷ + `read usage`. Arrow direction is **bottom-up** (Adapter actively polls Framework). Step ❺ arrow arrives here from R3, causing the bar to visually drop.

KV Cache ↔ Running Batch: thin bidirectional arrow (normal data flow, not numbered).
Green `admit` arrow (Waiting→Running): normal scheduling flow, not numbered.

**Middle: FrameworkAdapter** (independent thin gray bar)
Spans the full width between the two layers. Height ~1/5 of either layer. Gray background. Centered label: `FrameworkAdapter · vLLM | SGLang`. Arrows ❷ and ❹ pass through it. This shows it is the interface bridging framework and BidKV.

**Bottom: BidKV Layer** (dashed green border = pluggable)
Three stage cards in a pipeline (Step ❸ spans across these):

| Card | Title | Content |
|------|-------|---------|
| Pressure Detector | Step ❷: Adapter reads KV usage from Framework, passes to this module. Detects >90%, activates pipeline | threshold: `KV > 90%` |
| Bid Generation | Compute disruption cost per request, wrap as bid | `δ = f(completion, prompt, preemptions)` |
| Greedy Solver | Rank by utility, greedy select | **U = r/(δ+ε)** |

Step ❸ label above pipeline arrows: `compute δ → create bids → rank by U`.
Pipeline arrows (blue): `N requests` → `BidPool` → `Accepted`.
Step ❹: bold red arrow from Solver upward, **terminating at the FrameworkAdapter bar** (BidKV returns decision to Adapter). Label: ❹ + `victim = R3`. R3 is circled inside Running Batch as annotation. This is BidKV's **only upward output** — Adapter then calls Framework's `_preempt_request()` to execute ❺ (KV freed) and ❻ (requeue).

**Do NOT include**: comparison callouts with other strategies, decorative icons, inset data-structure cards or callout bubbles, Mode B token-level truncation, attention weight internals, multi-GPU topology, long text (max 5 words per label).

---

## 与论文章节的对应关系

| 图中区域 | 论文章节 |
|---------|---------|
| Waiting Queue + SJF | §5.1 (a) |
| Running Batch + Reorder | §5.1 (d) |
| KV Cache + Pressure | §2.1 |
| ❶ KV 95% 可观测状态 | §2.1 |
| ❷ Pressure Detector | §4.4 |
| ❸ Bid Generation + Greedy Solver | §4.2 Eq.2 + §4.3 Alg.1 Eq.1 |
| ❹ Victim 决策返回 (BidKV → Adapter) | §4.1 |
| ❺ KV 释放 (Adapter 调用 Framework 执行) | §5.1 |
| ❻ 回队重算 (同一 API 的另一效果) | §5.1 |
| FrameworkAdapter (middle bar) | §4.4 + §5.1/§5.2 |

## 参考格式

- 双栏宽度 ~17cm，高度 6-8cm
- PDF 矢量图优先；PNG ≥ 300 DPI
- 无衬线字体（Helvetica/Arial），与 ACM sigconf 协调
