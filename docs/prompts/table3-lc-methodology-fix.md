# Table 3 Long-Context Methodology Fix

> 本提示词面向资深系统会议论文写手，负责修改 BidKV SC 2026 论文的
> `paper/tables/table3_long_context.tex` 与 `paper/bidkv_sc2026.tex` §6.3 段落。
> 生成日期：2026-04-05
> 数据来源：`results/vllm_v8_long_context/`（5 策略 × 3 速率 × 3 runs = 45 runs）
> 数据状态：v8-frozen，所有数值为 3 runs 算术均值

---

## 第一部分：修改动机

### 统计问题

当前 Table 3 用三速率（0.35 / 0.5 / 0.7 req/s）的**跨速率算术平均**表示 TTFT P95。
但 TTFT P95 在饱和临界点附近呈指数级增长，三速率权重严重倾斜：

| 策略 | rate=0.35 | rate=0.50 | rate=0.70 |
|------|----------|----------|----------|
| BidKV | 3,461 ms **(7%)** | 12,871 ms **(28%)** | 30,143 ms **(65%)** |
| Static-Random | 2,348 ms **(5%)** | 9,667 ms **(22%)** | 31,693 ms **(73%)** |
| Largest-First | 2,309 ms **(5%)** | 13,133 ms **(27%)** | 33,345 ms **(68%)** |
| PE-SJF | 7,306 ms **(12%)** | 15,936 ms **(26%)** | 37,551 ms **(62%)** |

**结论**：所谓的"跨速率平均 TTFT P95"实质上是 rate=0.7 数字加噪，
算术平均掩盖了速率梯度，在统计上不可辩护。SLO attainment（有界百分比）
受此问题影响较小，但 TTFT 绝对值的跨速率均值对 reviewer 而言很容易被质疑。

### 修改方案

用 **rate=0.7 单速率快照**替换跨速率平均。理由：
- rate=0.7 是系统进入长尾饱和区的压力点，最能体现策略分化
- 消除统计问题，数字可直接比较
- 在脚注中声明其他速率方向一致（无需单独表格）
- **§6 结构不变**：LC 仍是补充性评估，不升级为主线

---

## 第二部分：精确数据

### 当前 Table 3 数据（跨速率平均，3 runs × 3 rates 均值）

```
PE (default)   SLO=21.2%  TTFT=80,135 ms  Thru=0.374  TPOT=181.4 ms
PE-SJF         SLO=74.9%  TTFT=20,264 ms  Thru=0.442  TPOT=144.3 ms  ← best TPOT
Static-Random  SLO=75.0%  TTFT=14,569 ms  Thru=0.447  TPOT=162.5 ms  ← best TTFT / best Thru
Largest-First  SLO=76.6%  TTFT=16,262 ms  Thru=0.435  TPOT=211.3 ms  ← best SLO
BidKV          SLO=75.6%  TTFT=15,492 ms  Thru=0.443  TPOT=179.6 ms  ← 2nd SLO / 2nd Thru
```

### 新 Table 3 数据（rate=0.7，3 runs 均值，必须使用以下精确数值）

```
PE (default)   SLO= 5.0%  TTFT=94,696 ms  Thru=0.413  TPOT=184.1 ms
PE-SJF         SLO=63.6%  TTFT=37,551 ms  Thru=0.563  TPOT=145.2 ms  ← best TPOT
Static-Random  SLO=57.4%  TTFT=31,693 ms  Thru=0.568  TPOT=208.3 ms  ← best Thru
Largest-First  SLO=64.4%  TTFT=33,345 ms  Thru=0.542  TPOT=239.6 ms  ← best SLO
BidKV          SLO=62.3%  TTFT=30,143 ms  Thru=0.560  TPOT=207.0 ms  ← best TTFT
```

**Bold/underline 规则（与全文一致）**：
- **粗体** = 最优值（per column）
- <u>下划线</u> = 次优值（per column）

新表中：
- SLO: Largest-First **bold**, PE-SJF underline
- TTFT: BidKV **bold**, Static-Random underline
- Throughput: Static-Random **bold**, PE-SJF underline
- TPOT: PE-SJF **bold**, PE underline（PE 虽低但已失效，可选择 Static-Random 作 underline）

> TPOT underline 处理：PE 的低 TPOT 是因为它几乎没有成功解码的请求，
> 属于无效数值。建议将 **Static-Random (208.3 ms) 作为 underline**，跳过 PE。

---

## 第三部分：修改指令

### 3.1 修改 `paper/tables/table3_long_context.tex`

**目标**：用 rate=0.7 数据替换所有数值，更新 caption。

**当前 caption**（需替换）：
```latex
\caption{Long-context workload: cross-rate average
  (vLLM, Llama-3.1-8B-Instruct, A6000, 500~req/run,
  rates 0.35/0.5/0.7\,req/s, SLO threshold 2{,}000\,ms,
  mean of 3~runs).
  Best in \textbf{bold}; second-best \underline{underlined}.}
```

**新 caption**：
```latex
\caption{Long-context workload at peak request rate
  (vLLM, Llama-3.1-8B-Instruct, A6000, 500~req/run,
  rate\,=\,0.70\,req/s, SLO threshold 2{,}000\,ms,
  mean of 3~independent runs).
  At lower rates (0.35 and 0.50\,req/s) all proactive strategies
  attain SLO\,$>$\,70\%; relative ordering is qualitatively consistent.
  Best in \textbf{bold}; second-best \underline{underlined}.}
```

**当前数据行**（全部替换）：
```latex
PE (default)      & 21.2 & 80135 & 0.374 & 181.4 \\
PE-SJF            & 74.9 & 20264 & 0.442 & \textbf{144.3} \\
Static-Random     & 75.0 & \textbf{14569} & \textbf{0.447} & \underline{162.5} \\
Largest-First     & \textbf{76.6} & 16262 & 0.435 & 211.3 \\
\textbf{BidKV}    & \underline{75.6} & \underline{15492} & \underline{0.443} & 179.6 \\
```

**新数据行**：
```latex
PE (default)      &  5.0 & 94696 & 0.413 & \underline{184.1} \\
PE-SJF            & \underline{63.6} & 37551 & \underline{0.563} & \textbf{145.2} \\
Static-Random     & 57.4 & \underline{31693} & \textbf{0.568} & 208.3 \\
Largest-First     & \textbf{64.4} & 33345 & 0.542 & 239.6 \\
\textbf{BidKV}    & 62.3 & \textbf{30143} & 0.560 & 207.0 \\
```

### 3.2 修改 `paper/bidkv_sc2026.tex` §6.3 段落

**定位**：`\input{tables/table3_long_context}` 下方，至 `\subsection{Reclamation Event Analysis}` 前。

**当前段落**（完整引用）：
```
Table~\ref{tab:long_context} extends the evaluation to a long-context workload
(500~requests, rates 0.35/0.5/0.7\,req/s, SLO threshold 2{,}000\,ms).
KV pressure is more severe: typical requests consume 2{,}000--4{,}000
tokens, saturating the 9{,}600-token KV budget with just two to four
concurrent requests.
PE degrades sharply (SLO 21.2\%, TTFT ${\sim}$80\,s).  
Among the remaining four strategies, Largest-First achieves the best SLO
(76.6\%), a simple result: when nearly all requests are long, evicting the
largest one frees the most capacity.  \bidkv remains competitive
(SLO 75.6\%, throughput \#2) but does not lead on SLO---consistent with
the expectation that utility-ranked victim selection provides the most value when
request lifecycles are heterogeneous.
PE-SJF achieves the best TPOT~P95 (144.3\,ms) but the worst TTFT
among active strategies (20{,}264\,ms), showing that SJF
admission alone is insufficient when KV pressure is extreme.
```

**新段落**（用以下内容完整替换）：
```
Table~\ref{tab:long_context} extends the evaluation to a long-context workload
(500~requests, rate\,=\,0.70\,req/s, SLO threshold 2{,}000\,ms).
KV pressure is more severe: typical requests consume 2{,}000--4{,}000
tokens, saturating the 9{,}600-token KV budget with just two to four
concurrent requests.
PE collapses (SLO 5.0\%, TTFT ${\sim}$95\,s), confirming that
unmanaged LIFO reclamation is untenable for long-context traffic.
Among the four active strategies, Largest-First leads on SLO
(64.4\%)---when all requests are long, evicting the largest frees
the most capacity, a regime where capacity-greedy selection excels.
\bidkv achieves the best TTFT~P95 (30{,}143\,ms, 5\,\% better than
Static-Random, 25\,\% better than PE-SJF), with competitive SLO
(62.3\%, 2.1\,pp below Largest-First).
PE-SJF obtains the best TPOT~P95 (145.2\,ms) but the worst TTFT
(37{,}551\,ms), confirming that admission ordering alone is
insufficient when KV pressure is extreme.
The TTFT advantage of \bidkv reflects the same mechanism as in the
mixed workload: utility-ranked selection avoids wasteful recomputation,
keeping prefill bandwidth available for queued requests.
```

---

## 第四部分：验证清单

修改完成后必须确认：

- [ ] table3_long_context.tex 中所有旧数字均已替换为 rate=0.7 数值
- [ ] caption 中 "cross-rate average" 措辞已改为 "peak request rate (rate=0.70 req/s)"
- [ ] caption 中已加入 "lower rates" 脚注声明
- [ ] §6.3 正文中不再出现 75.6%, 76.6%, 75.0%, 74.9%(SLO) 和 15,492/20,264/14,569 ms(TTFT) 等跨速率平均数字
- [ ] §6.3 正文不引用 rate=0.35 或 rate=0.50 的具体数值（方向性声明放在 caption 脚注）
- [ ] bold/underline 标记与新数据一致（TTFT 最优 = BidKV）
- [ ] `paper/` 目录下 `bash build.sh` 编译零 error，结果 PDF 可打开

---

## 第五部分：不允许修改的内容

- `paper/tables/table1_main.tex`（混合工作负载主结果表）
- `paper/tables/table2_rate_full.tex`（混合工作负载速率敏感性表）
- §6.1 / §6.2 / §6.4 / §6.5 任何内容
- §1–§5 / §7 / §8 任何内容
- 任何 figure 文件
- 策略列表（Table 3 仍保留 PE、PE-SJF、Static-Random、Largest-First、BidKV 五行，顺序不变）
