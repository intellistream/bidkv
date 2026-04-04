# §2 Background and Motivation — Review Prompt

用于与另一个 agent 讨论 §2 内容修改。引用补充将在内容定稿后单独进行。

---

```
You are a senior systems researcher preparing a submission to SC 2026
(Supercomputing). You are reviewing §2 (Background and Motivation) of the
paper "BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM
Serving."

Below is the current §2 text, followed by a list of issues identified
during internal review. Please revise §2 to address ALL issues while
respecting the constraints listed at the end.

====================================================================
CURRENT §2 TEXT
====================================================================

\section{Background and Motivation}\label{sec:background}

\subsection{KV Cache in LLM Serving}\label{sec:bg:kvcache}

Autoregressive Transformer decoders~\cite{Vaswani2017} compute attention as
$\text{Attn}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{d_k})\,V$, caching the
key and value projections of all previously generated tokens.  For a model with
$L$~layers, $H$~attention heads, and head dimension~$d_h$, the KV cache of a
single request of length~$n$ occupies $2LHd_h n$ elements.  With bf16
precision, Llama-3.1-8B ($L{=}32, H{=}8_{\text{kv}}, d_h{=}128$) allocates
128\,KiB per token per request.

Modern serving systems such as vLLM~\cite{Kwon2023} manage KV cache via
PagedAttention, allocating physical memory in fixed-size \emph{blocks} mapped
through a block table.  Despite this optimization, KV memory scales linearly
with sequence length: a single 4096-token request occupies $\sim$512\,MiB,
placing severe pressure on the $\sim$32\,GiB of KV-allocatable memory on a
48\,GiB A6000.

\subsection{The Preemption Problem}\label{sec:bg:preemption}

When KV cache utilization exceeds capacity, the serving engine must
\emph{preempt} running requests to free memory.  vLLM implements preemption via
\texttt{\_preempt\_request()}: the victim's KV blocks are released, and the
request re-enters the waiting queue for recomputation from scratch.  The default
victim selection is LIFO---the most recently scheduled request is evicted
first---which ignores request progress and recomputation cost.

Consider three concurrent requests under memory pressure:
\begin{itemize}[leftmargin=*,nosep]
\item $R_1$: 128-token chat reply, 90\% generation complete.
  Preempting wastes nearly all invested computation.
\item $R_2$: 2048-token document summary, 10\% complete.
  Preempting frees significant KV at low recomputation cost.
\item $R_3$: 512-token code completion, 50\% complete.
  Moderate recomputation cost.
\end{itemize}

\noindent
LIFO preempts whichever request was most recently scheduled, regardless of
these differences.  A scheduler aware of per-request recomputation cost would
preempt~$R_2$ first: it frees the most tokens at the lowest relative cost,
yielding the highest utility ratio.  This example illustrates two requirements
that motivate our design:

\begin{description}[style=unboxed,leftmargin=0em,nosep]
\item[R1 --- Explicit preemption-sensitivity signals.]
  The scheduler should accept per-request signals that capture heterogeneity in
  preemption cost, rather than relying solely on arrival order or a single
  fixed heuristic.

\item[R2 --- Cross-request coordination.]
  Victim selection should consider the \emph{batch-level} trade-off---how much
  KV capacity is freed versus how much scheduling disruption is
  incurred---rather than scoring each request in isolation.

\item[R3 --- Scorer-agnostic design.]
  Because the best importance signal may vary across workloads and models, the
  scoring strategy (\eg attention weights, uniform, random) should be
  pluggable without changing the selection algorithm.

\item[R4 --- Framework portability.]
  The mechanism should integrate with heterogeneous engines (\eg vLLM, SGLang)
  without requiring source-code modifications, to facilitate adoption across
  the serving ecosystem.
\end{description}

====================================================================
IDENTIFIED ISSUES (ordered by priority)
====================================================================

### PRIORITY-1 (must fix — structural / logical)

**ISSUE 1 — §2.1/§2.2 heavily overlaps with §1.**
§1 P1 already states "KV cache is the dominant memory resource" and
"scales linearly with sequence length." §1 P2 already explains
recompute-fallback and preemption cascades. §1 P3 names LIFO/FCFS.
§2 re-derives all of these. After reading §1→§2, a reviewer sees the
same story twice. §2 should ADVANCE the reader's understanding, not
repeat P1/P2/P3.

Recommendation: §2.1 should provide ONLY the quantitative foundation
that §1 deliberately omitted (the 128 KiB/token formula, the concrete
GPU memory budget, the block-table mechanism). Remove any sentences
that merely re-state §1's qualitative claims. §2.2 should formalize
the selection problem (input/output/objective), not re-explain "LIFO
ignores heterogeneity" which §1 already covered.

**ISSUE 2 — The R1/R2/R3 motivating example is too artificial.**
Three hand-crafted requests with stated completion percentages do not
convince a systems reviewer. The scenario is constructed to make the
answer obvious, providing no analytical insight. A real motivating
example should show WHY heterogeneity matters, not just THAT it
exists.

Recommendation: Either (a) replace with a 1-paragraph empirical
observation from your actual trace data (e.g., "In our ShareGPT
trace at rate 3.8, the ratio between the largest and smallest prompt
in any given preemption window is N×, confirming that …"), or
(b) formalize the problem as a constrained optimization and show that
LIFO is suboptimal in the formal sense. Option (a) is stronger
because it ties §2 directly to your evaluation.

**ISSUE 3 — R3 and R4 are engineering desiderata, not problem
requirements.**
R1 (signal pathway) and R2 (cross-request coordination) define what
the preemption-scheduling PROBLEM needs. R3 (scorer-agnostic) and R4
(portability) describe properties of a GOOD SOLUTION. Mixing them
weakens the argument: the reader cannot distinguish between what makes
the problem hard and what makes BidKV's design good.

Recommendation: Keep R1 and R2 in §2 as problem requirements. Move R3
and R4 to §4 (Design) opening, where they become design goals that
motivate BidKV's architecture. This is a standard structure in systems
papers (e.g., Borg, Mesos).

### PRIORITY-2 (should fix — narrative quality)

**ISSUE 4 — The attention formula is unnecessary in 2026.**
$\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})\,V$ has
appeared in 1000+ papers since 2017. Writing it out signals an
audience mismatch for SC. The KV size formula ($2LHd_h n$) IS useful
and should stay.

Recommendation: Remove the attention formula. Start §2.1 directly
from "Autoregressive LLM serving caches…" and then give the KV size
formula.

**ISSUE 5 — "utility ratio" forward-references §4 without definition.**
The phrase "yielding the highest utility ratio" appears in §2.2 but
U = r/(δ+ε) is only defined in §4. Either give a one-line informal
definition here or remove the term from §2 entirely.

Recommendation: Replace "yielding the highest utility ratio" with
"maximizing reclaimed tokens per unit scheduling disruption" — this is
self-explanatory and avoids forward reference.

**ISSUE 6 — Terminology inconsistency with §1.**
§1 uses "preemption-sensitivity", "per-request scheduling cost",
"recompute-fallback cost". §2 introduces "preemption cost",
"scheduling disruption", "recomputation cost" as if they are different
concepts. A reviewer will wonder whether these are the same thing.

Recommendation: Converge on a single term set. Preferred: "preemption
cost" (the cost of evicting a request) and "recompute cost" (the cost
of replaying the prompt when the request re-enters). Use consistently
in both §1 and §2.

### PRIORITY-3 (nice to have)

**ISSUE 7 — §2.2 has zero citations.**
The entire §2.2 (preemption problem) cites no prior work. This is
unusual for a Background section. At minimum, cite Kwon2023 for the
LIFO mechanism, and consider citing real workload characterization
studies for request heterogeneity.

Note from the user: Citations will be added AFTER the content
revision is finalized. Do not add citations in this round.

====================================================================
STRUCTURAL RECOMMENDATION
====================================================================

**Option B (recommended): Compress §2, move R3/R4 to §4.**

Target structure:
  §2.1 KV Cache Memory Model (quantitative only, no overlap with §1)
  §2.2 Problem Statement (formal or empirical, R1+R2 only)

Expected length: roughly 0.6 columns (current §2 is ~0.9 columns).
The 0.3 columns saved can accommodate a stronger §4 opening that
presents R3+R4 as design goals.

====================================================================
HARD CONSTRAINTS (do not violate)
====================================================================

1. Title remains: "BidKV: Utility-Guided Preemption Scheduling for
   KV-Pressure LLM Serving"
2. Section labels (\label{sec:background}, \label{sec:bg:kvcache},
   \label{sec:bg:preemption}) must be preserved for cross-references.
3. Do NOT add citations in this round (will be done separately).
4. Do NOT forward-reference experimental numbers or placeholder macros.
5. Keep ALL content in valid LaTeX (acmart sigconf).
6. The paper's core claim: BidKV provides a bid-based signal pathway
   for utility-guided preemption scheduling, not compression.
7. R1 and R2 (under whatever heading) must appear in §2.
8. R3 and R4 should move to §4 opening.

====================================================================
OUTPUT FORMAT
====================================================================

Return ONLY the revised §2 LaTeX text, starting from
\section{Background and Motivation} and ending just before
\section{Related Work}. No commentary outside the LaTeX block.
```
