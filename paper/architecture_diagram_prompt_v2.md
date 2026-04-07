# BidKV Architecture Diagram Prompt

## 总体要求

为 SC 2026 论文绘制 BidKV 系统架构图（单栏宽度，约 3.3 inch）。

核心叙事：**BidKV 在后台预计算一张 victim 排名表，当 KV 压力到达门槛时，把排名投射回引擎的运行队列，让引擎自己的 LIFO 驱逐机制自动选中最优 victim。**

图必须同时传达两件事：
1. **三层架构分层**（LLM Framework / Runtime Adapter / BidKV Core）— 这是论文 §4 的核心设计
2. **两个过程**（后台排名 + 门控驱逐）在三层之间的数据流动

---

## 整体布局：三层 + 两个过程

采用 **三个水平层带（horizontal bands）** 从上到下排列，代表三个架构层。两个过程（A 蓝色、B 红色）的箭头在三层之间穿行。

**绘图时最重要的一条规则**：三个层带必须是图中最显眼的视觉结构——用 **不同背景色 + 明确的层边界线 + 左侧竖排层标签** 让读者 **先看到三层，再看到箭头**。

```
┌═══════════════════════════════════════════════════════════════════┐
│ L1  LLM Serving Engine (e.g., vLLM)                 浅灰 #F5F5F5 │
│                                                                   │
│  ┌──────────┐    ┌────────────────────────┐    ┌──────────────┐  │
│  │ Waiting  │    │     Running Batch      │    │  KV Block    │  │
│  │ Queue    │    │                        │    │  Pool        │  │
│  │          │    │  R5   R6   R7   [R4]▮  │    │  ████████░   │  │
│  │ R1 R2 R3 │◄─B4─ (requeue)    tail↑    │    │         95%↑ │  │
│  └──────────┘    └──────▲────────▲────────┘    └──────┬───────┘  │
│                    B3↑  │   A1↓  │ B2↑                │          │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─│
│ L2  Runtime Adapter Layer                      蓝虚线 #E8F0FE    │
│                                                                   │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────────────┐   │
│  │              │  │  Pressure  │  │                         │   │
│  │    State     │  │   Gate     │  │   Cached Priority Map   │   │
│  │  Collector   │  │            │  │                         │   │
│  │              │  │  ◇ KV≥95%? │  │  { R4:0, R7:1, R6:2,   │   │
│  │  c, P, r    │  │  │yes  │no │  │    R5:3 }               │   │
│  │  per request │  │  ▼    ▼   │  │  (lower = more          │   │
│  │              │  │ [B2] LIFO │  │   expendable)            │   │
│  └──────┬───────┘  └─────▲─────┘  └──────────▲──────────────┘   │
│     A2↓ │            B1↑ │              A3↑   │                  │
│ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─│
│ L3  BidKV Core                             橙实线 #FFF3E0        │
│                                                                   │
│  ┌───────────────────┐        ┌──────────────────────────────┐   │
│  │ Bid Generation    │ bids   │ Utility-Ranked Selection     │   │
│  │ Layer             │───────→│ Layer                        │   │
│  │                   │        │                              │   │
│  │ Per request:      │        │ Sort all bids by:            │   │
│  │  r = KV tokens    │        │  U = r / (δ + ε)             │   │
│  │  δ = 1+0.5c+0.3P  │        │                              │   │
│  │  → Scheduling Bid │        │ Output: ranked victim list   │   │
│  └───────────────────┘        └──────────────────────────────┘   │
│                                                                   │
└═══════════════════════════════════════════════════════════════════┘
```

---

## Layer 1：LLM Serving Engine（浅灰色 #F5F5F5）

最上方层带。**左侧竖排标签** "L1" 或 "LLM Serving Engine"。

包含三个并排组件，内部细节如下：

### Waiting Queue（左侧）
- 圆角矩形，内有 3 个请求色块 R1 R2 R3
- 色块从左到右按 prompt 长度升序排列（体现 SJF admission）
- 底部小标注 "SJF ordered"

### Running Batch（中央，最大的组件）
- 圆角矩形，内有 4 个请求色块 R5 R6 R7 R4
- **R4 末尾色块用红色填充 + 红色虚线边框**，附标注 "← tail (victim)"
- 其余色块用不同浅色区分请求（如浅绿/浅蓝/浅黄），表示异构性
- 色块内只写请求编号（R5, R6, R7, R4）
- **重要**：Running Batch 有 **两个接口点**（左下角供 A1 出发，右下角供 B2 到达），箭头起止点要明确

### KV Block Pool（右侧）
- 竖向或横向条形图，填充到 ~95%
- 在 95% 位置画 **红色虚线阈值**，标注 "95%"
- 填充区用深灰色，空闲区用白色

**Layer 1 内部交互**（B3 + B4，纯红色，发生在 Layer 1 内部 不跨层）：
- B3：Running Batch 末尾的 R4 被弹出（画一个小的红色弧线箭头从 R4 弯出 Running Batch）
- B4：R4 从 Running Batch 飞向 Waiting Queue（红色箭头，标注 "Requeue, $P$++"）

---

## Layer 2：Runtime Adapter Layer（浅蓝色 #E8F0FE + 蓝色虚线边框）

中间层带。**左侧竖排标签** "L2" 或 "Runtime Adapter"。

包含三个模块，**每个模块内部显示简短的职责描述**：

### State Collector（左侧）
- 圆角矩形
- 内部文字（两行）：
  - "Collect $c$, $P$, $r$"
  - "per request"
- **上方接口**：A1 蓝色虚线箭头从 Layer 1 的 Running Batch **向下** 进入
- **下方接口**：A2 蓝色虚线箭头 **向下** 出发去 Layer 3

### Pressure Gate（中央）
- **菱形判断框**，标注 "KV ≥ 95%?"
- **上方接口**：B1 红色实线箭头从 Layer 1 的 KV Block Pool **向下** 进入
- **Yes 出口**（向上）：红色实线箭头 B2 **向上** 穿回 Layer 1 的 Running Batch，标注 "Reorder"
- **No 出口**（向右或向下）：灰色虚线短箭头，标注 "LIFO (default)"，终止于一个小灰色圆点
- **关键**：Yes 出口箭头在向上之前，先经过 Cached Priority Map（画一条红色短线从 Map 汇入 B2 箭头，或 B2 箭头标注 "read cached ranking"）

### Cached Priority Map（右侧）
- **圆柱图标**（数据库/存储符号），**加粗边框 + 浅黄色填充**，视觉最突出
- 内部文字示例（可选，如果空间允许）：
  - "{ R4: 0, R7: 1,"
  - "  R6: 2, R5: 3 }"
  - 小注释 "0 = most expendable"
- **下方接口**：A3 蓝色虚线箭头从 Layer 3 **向上** 进入
- **左侧接口**：红色虚线连接到 Pressure Gate 的 Yes 分支（表示 B2 读取此缓存）
- **这是两个过程的唯一耦合点**——A 过程写入，B 过程读取

---

## Layer 3：BidKV Core（浅橙色 #FFF3E0 + 橙色实线边框）

最下方层带。**左侧竖排标签** "L3" 或 "BidKV Core"。

包含两个子模块 + 一条内部流箭头：

### Bid Generation Layer（左侧）
- 圆角矩形，橙色浅底
- 内部文字（三行）：
  - "$r$ = KV tokens held"
  - "$\delta = 1 + 0.5c + 0.3P$"
  - "→ Scheduling Bid"
- **上方接口**：A2 蓝色虚线箭头从 Layer 2 的 State Collector **向下** 进入

### 内部流箭头
- 橙色实线箭头，从 Bid Generation → Selection，标注 "bids"
- 表示 bid 数据在 BidKV Core 内部从生成层流向选择层

### Utility-Ranked Selection Layer（右侧）
- 圆角矩形，橙色浅底
- 内部文字（两行）：
  - "Sort by $U = r/(\delta + \varepsilon)$"
  - "→ ranked victim list"
- **上方接口**：A3 蓝色虚线箭头 **向上** 出发去 Layer 2 的 Cached Priority Map

---

## 过程 A：Background Ranking（蓝色虚线箭头 #4285F4）

每 ~3 秒执行一次。箭头主要走图的 **左侧**，形成一个 **↓下→右→上↑** 的回路。

| 步骤 | 跨层 | 起点 → 终点 | 箭头标注 | 视觉要求 |
|------|------|------------|---------|---------|
| **A1** | L1→L2 | Running Batch 底部 → State Collector 顶部 | "Collect state (~3 s)" | 垂直向下穿越层边界 |
| **A2** | L2→L3 | State Collector 底部 → Bid Generation 顶部 | "Request states" | 垂直向下穿越层边界 |
| **A3** | L3→L2 | Selection Layer 顶部 → Cached Priority Map 底部 | "$U$-ranked list" | **垂直向上**穿越层边界 |

A 过程不触及 Layer 1 的 KV Block Pool 和 Waiting Queue。

---

## 过程 B：Gated Reclamation（红色实线箭头 #DB4437）

每个 scheduling tick 执行。箭头主要走图的 **右侧**。

| 步骤 | 跨层 | 起点 → 终点 | 箭头标注 | 视觉要求 |
|------|------|------------|---------|---------|
| **B1** | L1→L2 | KV Block Pool 底部 → Pressure Gate 顶部 | "Read usage" | 垂直向下穿越层边界 |
| **B2** | L2→L1 | Pressure Gate (yes) → Running Batch 底部 | "Reorder" | **垂直向上**穿越层边界；在离开 L2 前，画一条红色短连线从 Cached Priority Map 汇入此箭头 |
| **B3** | L1 内部 | Running Batch tail (R4) → 弹出 | "Native eviction" | R4 色块向左弹出的弧线箭头 |
| **B4** | L1 内部 | R4 → Waiting Queue | "Requeue ($P$++)" | 红色箭头从 R4 飞向左侧 Waiting Queue |

**B2 的视觉重点**：这步同时使用两个数据源（Pressure Gate 的 yes 判断 + Cached Priority Map 的排名数据），两条线汇合后 **一起向上** 回到 Running Batch。可以画成 Y 形汇合或在箭头旁标注 "read cached ranking + gate pass"。

B 过程不触及 Layer 3（BidKV Core）。

---

## 视觉规范

### 颜色

| 元素 | 颜色 | 说明 |
|------|------|------|
| Layer 1 背景 | 浅灰 (#F5F5F5) | 表示"引擎原生，BidKV 不拥有" |
| Layer 2 背景 + 边框 | 浅蓝 (#E8F0FE) + 蓝色虚线边框 | 表示"中间层，桥接" |
| Layer 3 背景 + 边框 | 浅橙 (#FFF3E0) + 橙色实线边框 | 表示"BidKV 核心逻辑" |
| Cached Priority Map | 浅黄填充 (#FFF9C4) + 加粗边框 | 唯一耦合点，最突出 |
| 过程 A 箭头 + 编号 | 蓝色 (#4285F4) 虚线 | 后台排名 |
| 过程 B 箭头 + 编号 | 红色 (#DB4437) 实线 | 门控驱逐 |
| Victim R4 色块 | 红色填充 (#FFCDD2) + 红色边框 | 被选中的驱逐目标 |
| 其他请求色块 | 各一种浅色（绿/蓝/黄/紫） | 体现请求异构性 |
| L1/L2/L3 层边界 | 深灰虚线 | 清晰分隔三层 |

### 编号样式

- A1, A2, A3：蓝色实心圆圈 + 白色文字
- B1, B2, B3, B4：红色实心圆圈 + 白色文字
- 编号紧贴箭头起点，不要放在箭头中间

### 图例（右下角，紧凑一行）

`── ── ──` Background ranking (~3 s)　　`────────` Per-tick gated reclamation

### 简洁原则

- **不写**任何函数名或代码标识符
- 每支**跨层箭头**旁 **最多 3 个英文单词**
- 请求色块内只写 R1–R7 编号
- **模块框内**可以写 1–2 行简短说明（如 δ 公式、U 公式），但不写代码
- 三层的 **层标签** 必须是图中最大的文字（14pt+），模块名次之（11pt），箭头标注最小（9pt）

---

## 论文术语对齐

| 正确术语 | 禁用术语 |
|----------|----------|
| disruption estimate ($\delta$) | quality delta |
| reclamation utility ($U$) | compression ratio |
| reclaim / preempt | compress |
| Scheduling Bid | compression bid |
| Bid Generation | scoring (as proper noun) |
| Utility-Ranked Selection | solver |
| Runtime Adapter | hook layer |

---

## Caption

> **BidKV scheduling overview.** Two processes govern victim selection. *Background ranking* (blue, A1–A3): every ~3 s, the Runtime Adapter collects request-lifecycle state (completion progress $c$, preemption count $P$) from the running batch, passes it to the BidKV Core, which computes a disruption estimate $\delta = 1 + 0.5c + 0.3P$ for each request and ranks all candidates by reclamation utility $U = r/(\delta + \varepsilon)$; the resulting ordering is cached. *Gated reclamation* (red, B1–B4): at each scheduling tick the adapter checks KV utilization (B1); when usage $\geq 95\%$, it projects the cached ranking onto the running queue as native keep-priority, placing the highest-$U$ candidate at the queue tail (B2). The framework's native LIFO eviction removes the tail request and frees its KV blocks (B3); the preempted request re-enters the waiting queue with $P{+}{+}$ (B4), raising its $\delta$ in future ranking cycles (anti-starvation). Below the pressure threshold, no reorder occurs and the framework's default LIFO ordering governs eviction.
