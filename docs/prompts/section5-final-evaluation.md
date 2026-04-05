# Section 5 Final-Pass Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026) titled
"BidKV: Utility-Guided KV Cache Reclamation for Admission-Responsive LLM
Serving". Section 5 (Implementation) has just completed its final cleanup
round. Please evaluate whether the section is ready to submit, and flag
any remaining issues. **Do not re-raise items in the "already resolved"
table below.**

---

## Paper context

- **Core problem**: victim selection under KV cache pressure in LLM serving.
- **Value proposition**: improved *admission responsiveness* — reducing queued-
  request admission delay, measured as TTFT P95 and SLO attainment at 300 ms.
  Throughput and TPOT are explicitly acknowledged tradeoff dimensions, not
  primary claims.
- **Execution model**: recompute-fallback. BidKV only controls *which* request
  is preempted; vLLM executes the preemption natively (releases KV blocks,
  requeues for full recomputation). BidKV does not modify vLLM's KV cache
  coordinator or block pool.
- **Section 4 architecture**: four layers —
  Bid Signal → Bid Generation → Constrained Solver → Runtime Adapter.
  The core abstraction is a bid-based utility ranking: U = r / (δ + ε),
  where r = tokens freed, δ = disruption estimate, ε = stability term.
- **Section 6 acknowledgment**: five evaluated strategies form a layered
  comparison; multiple scheduling dimensions co-vary; performance is attributed
  to the combined configuration, not isolated mechanisms.
- **BidKV's unique mechanism**: step (b) — bid-based utility ranking via the
  full scoring → bid generation → constrained solver pipeline. All other
  scheduling dimensions (admission ordering, proactive preemption, queue
  reorder) are shared scaffolding or baseline-specific.

---

## Changes made in this final round (two edits)

### Change 1 — Strategy differentiation paragraph restructured

**Before:**
```
\paragraph{Strategy differentiation.}
Table~\ref{tab:strategy_diff} summarizes how the five experimental strategies
differ across scheduling dimensions.  All share the same reclamation execution
path (vLLM native preemption); the differentiation is in victim selection
logic, admission ordering, running-queue reordering, and proactive preemption
policy.  Because multiple dimensions co-vary across strategies,
Section~\ref{sec:eval} attributes performance differences to the combined
scheduling configuration rather than isolated mechanisms.
```

**After:**
```
\paragraph{Strategy differentiation.}
Table~\ref{tab:strategy_diff} summarizes how the five experimental strategies
differ across scheduling dimensions.  All share the same reclamation execution
path (vLLM native preemption).  \bidkv's core differentiator is
step~(b): the bid-based utility ranking that determines reclamation priority.
Steps~(a), (c), and~(e) configure admission ordering, proactive preemption
triggers, and queue management that are shared across multiple evaluated
strategies; step~(d) is used only by heuristic baselines.  Because these
dimensions co-vary across strategies, Section~\ref{sec:eval} attributes
performance differences to the combined scheduling configuration rather than
to isolated mechanisms.
```

**Intent**: explicitly name step (b) as BidKV's sole differentiating
mechanism; declare steps (a)/(c)/(e) as shared scaffolding and (d) as
baseline-only — resolving the risk that readers mistake the full policy bundle
for BidKV's core contribution.

---

### Change 2 — Table 1 Victims column for PE / PE-SJF

**Before:**
```
PE (default)    & FCFS & LIFO & N/A       & \ding{55} & \ding{55} \\
PE-SJF          & SJF  & LIFO & N/A       & \ding{55} & \ding{55} \\
```
Caption had no note about N/A.

**After:**
```
PE (default)    & FCFS & LIFO & LIFO$^*$  & \ding{55} & \ding{55} \\
PE-SJF          & SJF  & LIFO & LIFO$^*$  & \ding{55} & \ding{55} \\
```
Caption footnote added:
```
LIFO$^*$ = no managed victim-selection; vLLM's native last-in-first-out
eviction determines which request is reclaimed.
```

**Intent**: N/A was factually misleading — LIFO *is* the victim-selection
policy for PE/PE-SJF. The change makes explicit what BidKV's utility ranking
replaces. LIFO* distinguishes it from the Run column LIFO (which is
running-queue ordering) to avoid column-conflation confusion.

---

## Current Section 5 — complete verbatim text

```latex
\section{Implementation}\label{sec:impl}

\bidkv is implemented as a standalone Python package with \textbf{zero external
dependencies} (Python $\geq$~3.10 standard library only), totaling
$\sim$12,000~lines of code.  The system defaults to \textbf{disabled}
and includes a global kill switch for instant bypass in production.
Strategies are managed through a pluggable registry; adding a new strategy
requires implementing a single interface and one registration call.

\subsection{vLLM Integration}\label{sec:impl:vllm}

We integrate with vLLM 0.17.1 (v1 architecture) using its plugin system.
\bidkv's fundamental integration point is victim selection: the adapter
ranks running requests by reclamation utility and delegates execution
entirely to vLLM's native preemption mechanism---releasing all KV blocks
and requeueing the request for full recomputation.  \bidkv does not modify
vLLM's KV cache coordinator or block pool; it only controls \emph{which}
request is reclaimed.  Under recompute-fallback semantics, the
victim-selection policy does not alter final outputs.

The adapter intercepts scheduling at three points:

\begin{enumerate}[nosep]
\item \textbf{Pre-schedule.}  Before each native scheduling call, the hook
  performs five sub-steps:
  (a)~\emph{Admission reorder}: strategies order the waiting queue by SJF
  (prompt length) or FCFS (see Table~\ref{tab:strategy_diff}).
  (b)~\emph{Victim-priority refresh} (every 3\,s): each strategy's
  \texttt{select\_victims()} is invoked on the running set to produce a
  cached priority ordering.  BidKV uses the full bid pipeline
  (Eq.~\ref{eq:utility}--\ref{eq:delta}), ranking candidates by utility
  $U{=}r/(\delta{+}\varepsilon)$; other strategies use simpler heuristics.
  (c)~\emph{Proactive preemption} (KV${>}90\%$, waiting queue non-empty,
  ${\geq}3$ running, 5\,s cooldown): the lowest-priority running request
  from the cached ordering is preempted via vLLM's native mechanism.
  PE and PE-SJF skip this step (no-intervention baselines).
  (d)~\emph{SRPT preemption} (Static-Random, Largest-First only,
  KV${>}80\%$, 1.5\,s cooldown): a running request whose estimated
  remaining work exceeds 1.2$\times$ the total cost of the cheapest waiting
  request is preempted.  BidKV disables SRPT: under recompute-fallback
  semantics, the re-prefill cost of a long-prompt victim typically
  outweighs the scheduling gain.
  (e)~\emph{Running-queue reorder}: strategies with this enabled sort the
  running list by cached priority so that vLLM's native LIFO eviction
  removes the lowest-priority request.  BidKV gates this reorder to
  KV utilization ${>}95\%$ and mean prompt length ${\leq}500$\,tokens.

\item \textbf{Post-decode tracking.}  After each decode step, newly sampled
  tokens are appended to per-request state for scoring.

\item \textbf{Lifecycle cleanup.}  On request completion, bids and state are
  cleared.
\end{enumerate}

\paragraph{Strategy differentiation.}
Table~\ref{tab:strategy_diff} summarizes how the five experimental strategies
differ across scheduling dimensions.  All share the same reclamation execution
path (vLLM native preemption).  \bidkv's core differentiator is
step~(b): the bid-based utility ranking that determines reclamation priority.
Steps~(a), (c), and~(e) configure admission ordering, proactive preemption
triggers, and queue management that are shared across multiple evaluated
strategies; step~(d) is used only by heuristic baselines.  Because these
dimensions co-vary across strategies, Section~\ref{sec:eval} attributes
performance differences to the combined scheduling configuration rather than
to isolated mechanisms.

[Table 1: Strategy differentiation
 Columns: Strategy | Wait | Run | Victims | Proactive | SRPT
 PE (default)   FCFS  LIFO  LIFO*  no  no
 PE-SJF         SJF   LIFO  LIFO*  no  no
 Static-Random  SJF   prio  random yes yes
 Largest-First  SJF   prio  capacity yes yes
 BidKV          SJF   gated U=r/(δ+ε) yes no
 
 Caption footnotes: prio=completion-aware keep score;
 gated=reorder when KV>95% and mean prompt ≤500;
 SRPT=KV>80%; LIFO*=no managed victim-selection,
 vLLM native LIFO determines reclamation.]

\subsection{SGLang Integration}\label{sec:impl:sglang}

The \texttt{SGLangAdapter} integrates with SGLang's RadixAttention
engine by hooking the scheduling entry point.
To test portability, we deploy \bidkv's complete scheduling policy
(SJF admission ordering, pressure-gated running-queue reorder, and
utility-ratio victim selection) on SGLang without modifying SGLang's
internals, and compare against Vanilla SGLang (SGLang's native LRU-based
eviction with no managed scheduling) and Random-Evict (\bidkv's adapter
with random victim selection replacing the utility-ratio solver).
```

---

## Already resolved — do NOT re-raise these

| Issue | Resolution |
|---|---|
| Execution semantics buried after enumeration | Moved to opening paragraph of §5.1 in prior round |
| Pre-schedule title was verbose feature list | Simplified to `Pre-schedule.` in prior round |
| "four sub-steps" but actually five | Fixed to explicit (a)–(e) in prior round |
| §5 had no co-varying acknowledgment | Added sentence in prior round |
| `Mode A` / `Mode B` terminology | Removed in earlier round; absent from §5 |
| `quality_delta` / quality-optimization framing | Removed in earlier round; absent from §5 |
| N/A in Victims column misrepresented PE/PE-SJF | Fixed to LIFO* with caption footnote this round |
| No distinction between BidKV core mechanism and shared scaffolding | Fixed in Strategy differentiation paragraph this round |

---

## Your evaluation task

For each question, rate severity: **critical / moderate / minor / none**,
and suggest a specific fix if not "none".

### Q1 — Does naming step (b) as "BidKV's core differentiator" successfully
separate the contribution from the shared scaffolding?

After the change, the Strategy differentiation paragraph reads:
"BidKV's core differentiator is step (b): the bid-based utility ranking that
determines reclamation priority. Steps (a), (c), and (e) configure ... shared
across multiple evaluated strategies; step (d) is used only by heuristic
baselines."

Does this successfully neutralize the "full policy bundle" risk? Or does
the five-step enumeration in the preceding Pre-schedule item still overpower
this framing?

### Q2 — LIFO* in Table 1: does it help or introduce confusion?

The Run column for PE/PE-SJF already says "LIFO" (running-queue ordering).
The Victims column now also says "LIFO*". 
Could a reader confuse Run=LIFO (queue ordering) with Victims=LIFO* (eviction
policy), despite the caption footnote? Is the asterisk sufficient
disambiguation, or would a different cell value (e.g., "native") be cleaner?

### Q3 — Does §5.1 make sufficiently clear the relationship between
step (b) and Section 4's four-layer architecture?

Section 4 describes: Bid Signal → Bid Generation → Constrained Solver →
Runtime Adapter. Step (b) in §5.1 says "BidKV uses the full bid pipeline
(Eq. utility--delta), ranking candidates by utility U=r/(δ+ε)".
Is this cross-reference adequate? Or should step (b) explicitly say
"completing the bid-generation and solver layers from Section 4"?

### Q4 — SGLang Integration: is "complete scheduling policy" accurate given
that the co-varying acknowledgment just declared step (b) as BidKV's core
differentiator?

§5.2 says "we deploy BidKV's complete scheduling policy (SJF admission
ordering, pressure-gated running-queue reorder, and utility-ratio victim
selection) on SGLang".
After the Strategy differentiation paragraph now clearly separates core
mechanism from scaffolding, does calling it "complete scheduling policy" in
§5.2 re-introduce the bundle framing? Or is "complete" justified here because
we are describing the full evaluated configuration (not making a contribution
claim)?

### Q5 — Does the section ever imply "scheduling efficiency" as the goal
rather than "admission responsiveness"?

Check the entire §5 text above for any phrasing that might lead a reader to
believe the paper's primary goal is throughput optimization or abstract
scheduling efficiency, rather than reducing queued-request admission latency
(TTFT / SLO attainment). Flag any specific sentences.

### Q6 — Table 1 column header "Victims": is this self-explanatory?

The column header is just "Victims". A reader unfamiliar with preemptive
scheduling might not immediately understand this means "victim-selection
policy". Would "Victims (policy)" or a brief mention in the caption be helpful?
Or is the current header adequate given the surrounding text?

### Q7 — Overall readiness of Section 5 after this final cleanup

On a scale of:
- **needs another round** (structural or factual issues remain)
- **minor polish only** (small wording tweaks, no structural changes)
- **ready to submit** (section is done)

Where does Section 5 stand? Please justify briefly, considering that this is
a conference paper under space constraints (10 pages total) and that §5 should
be concise while providing sufficient implementation detail for reproducibility.
