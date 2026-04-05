# Section 5 Revision Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026) titled
"BidKV: Utility-Guided KV Cache Reclamation for Admission-Responsive LLM Serving".
Section 5 (Implementation) has just been revised. Please evaluate whether the
revision successfully addresses the stated goals — **do not re-raise issues
listed in the "already resolved" table**.

---

## Paper context

- **Core problem**: victim selection under KV cache pressure in LLM serving.
- **Value proposition**: improved *admission responsiveness* (TTFT P95 and SLO
  attainment at 300 ms). Throughput/TPOT are acknowledged tradeoff dimensions.
- **Execution model**: recompute-fallback. BidKV only controls *which* request
  is preempted; vLLM executes the preemption natively.
- **Section 4 framing**: four-layer architecture (Bid Signal → Bid Generation →
  Constrained Solver → Runtime Adapter). Core abstraction = bid-based utility
  ranking of victims.
- **Section 6 framing**: five strategies form a layered comparison; multiple
  scheduling dimensions co-vary; performance is attributed to combined
  configuration not isolated mechanisms.

---

## What the revision aimed to fix

Three targeted changes were made to §5.1 (vLLM Integration):

### Change 1 — Execution semantics moved to opening paragraph

**Before** (execution semantics appeared *after* the enumeration as a separate
`\noindent\textbf{...}` block):
```
[enumeration of 3 hooks with 4 sub-steps]

\noindent\textbf{Execution semantics (recompute fallback).}
When the solver selects a victim, the adapter invokes vLLM's native preemption
mechanism, releasing all KV blocks and moving the request to the waiting queue
for full recomputation.  BidKV does not modify vLLM's KV cache coordinator or
block pool---it only controls which request is reclaimed.  Under
recompute-fallback semantics, the policy does not alter final outputs; the
differentiation lies in victim selection.
```

**After** (execution semantics as the *first* paragraph of §5.1):
```
We integrate with vLLM 0.17.1 (v1 architecture) using its plugin system.
BidKV's fundamental integration point is victim selection: the adapter
ranks running requests by reclamation utility and delegates execution
entirely to vLLM's native preemption mechanism---releasing all KV blocks
and requeueing the request for full recomputation.  BidKV does not modify
vLLM's KV cache coordinator or block pool; it only controls which request
is reclaimed.  Under recompute-fallback semantics, the victim-selection
policy does not alter final outputs.
```

---

### Change 2 — Pre-schedule item restructured

**Before**:
```
\item \textbf{Pre-schedule: admission reorder + victim selection + proactive
  preemption.}
  The hook executes four sub-steps before the native scheduling loop:
  (a) Waiting-queue reorder ...
  (b) Priority cache refresh ...
  (c) Proactive preemption ...
  (d) SRPT preemption ...
  Finally, for strategies with running-queue reorder enabled, the hook sorts
  the running list by cached priority ...
  BidKV gates this reorder ...
```
Problems: title is a verbose feature list; claimed "four sub-steps" but
described five (the "Finally" clause was a hidden fifth step).

**After**:
```
\item \textbf{Pre-schedule.}  Before each native scheduling call, the hook
  performs five sub-steps:
  (a) Admission reorder: strategies order the waiting queue by SJF (prompt
  length) or FCFS.
  (b) Victim-priority refresh (every 3s): each strategy's select_victims()
  is invoked on the running set.  BidKV uses the full bid pipeline
  (Eq. utility--delta), ranking candidates by utility U = r/(δ+ε); other
  strategies use simpler heuristics.
  (c) Proactive preemption (KV>90%, waiting queue non-empty, ≥3 running,
  5s cooldown): the lowest-priority running request from the cached ordering
  is preempted via vLLM's native mechanism.  PE and PE-SJF skip this step
  (no-intervention baselines).
  (d) SRPT preemption (Static-Random, Largest-First only, KV>80%, 1.5s
  cooldown): a running request whose estimated remaining work exceeds
  1.2× the total cost of the cheapest waiting request is preempted.
  BidKV disables SRPT: under recompute-fallback semantics, the re-prefill
  cost of a long-prompt victim typically outweighs the scheduling gain.
  (e) Running-queue reorder: strategies with this enabled sort the running
  list by cached priority so that vLLM's native LIFO eviction removes the
  lowest-priority request.  BidKV gates this reorder to KV utilization
  >95% and mean prompt length ≤500 tokens.
```

---

### Change 3 — Co-varying sentence added to Strategy differentiation paragraph

**Before**:
```
Table 1 summarizes how the five strategies differ in their scheduling behavior.
All strategies share vLLM's native reclamation execution; the differentiation
is in waiting-queue ordering, running-queue ordering, victim selection logic,
and whether proactive reclamation is enabled.
```

**After**:
```
Table 1 summarizes how the five experimental strategies differ across scheduling
dimensions.  All share the same reclamation execution path (vLLM native
preemption); the differentiation is in victim selection logic, admission
ordering, running-queue reordering, and proactive preemption policy.
Because multiple dimensions co-vary across strategies, Section 6 attributes
performance differences to the combined scheduling configuration rather than
isolated mechanisms.
```

---

## Current Section 5 text (verbatim, simplified LaTeX)

```latex
\section{Implementation}

BidKV is implemented as a standalone Python package with zero external
dependencies (Python >= 3.10 standard library only), totaling ~12,000 lines of
code.  The system defaults to disabled and includes a global kill switch for
instant bypass in production.  Strategies are managed through a pluggable
registry; adding a new strategy requires implementing a single interface and one
registration call.

% --- §5.1 vLLM Integration ---

We integrate with vLLM 0.17.1 (v1 architecture) using its plugin system.
BidKV's fundamental integration point is victim selection: the adapter ranks
running requests by reclamation utility and delegates execution entirely to
vLLM's native preemption mechanism---releasing all KV blocks and requeueing the
request for full recomputation.  BidKV does not modify vLLM's KV cache
coordinator or block pool; it only controls which request is reclaimed.  Under
recompute-fallback semantics, the victim-selection policy does not alter final
outputs.

The adapter intercepts scheduling at three points:

1. Pre-schedule.  Before each native scheduling call, the hook performs five
   sub-steps:
   (a) Admission reorder: strategies order the waiting queue by SJF (prompt
       length) or FCFS (see Table 1).
   (b) Victim-priority refresh (every 3s): each strategy's select_victims() is
       invoked on the running set to produce a cached priority ordering.  BidKV
       uses the full bid pipeline (Eq. utility--delta), ranking candidates by
       utility U = r/(δ+ε); other strategies use simpler heuristics.
   (c) Proactive preemption (KV>90%, waiting queue non-empty, ≥3 running,
       5s cooldown): the lowest-priority running request from the cached
       ordering is preempted via vLLM's native mechanism.  PE and PE-SJF skip
       this step (no-intervention baselines).
   (d) SRPT preemption (Static-Random, Largest-First only, KV>80%, 1.5s
       cooldown): a running request whose estimated remaining work exceeds
       1.2× the total cost of the cheapest waiting request is preempted.
       BidKV disables SRPT: under recompute-fallback semantics, the re-prefill
       cost of a long-prompt victim typically outweighs the scheduling gain.
   (e) Running-queue reorder: strategies with this enabled sort the running
       list by cached priority so that vLLM's native LIFO eviction removes the
       lowest-priority request.  BidKV gates this reorder to KV utilization
       >95% and mean prompt length ≤500 tokens.

2. Post-decode tracking.  After each decode step, newly sampled tokens are
   appended to per-request state for scoring.

3. Lifecycle cleanup.  On request completion, bids and state are cleared.

[Table 1: Strategy differentiation across scheduling dimensions.
 Strategies: PE (FCFS/LIFO/N/A/no/no), PE-SJF (SJF/LIFO/N/A/no/no),
 Static-Random (SJF/prio/random/yes/yes), Largest-First (SJF/prio/capacity/yes/yes),
 BidKV (SJF/gated/U=r/(δ+ε)/yes/no)]

Strategy differentiation.  Table 1 summarizes how the five experimental
strategies differ across scheduling dimensions.  All share the same reclamation
execution path (vLLM native preemption); the differentiation is in victim
selection logic, admission ordering, running-queue reordering, and proactive
preemption policy.  Because multiple dimensions co-vary across strategies,
Section 6 attributes performance differences to the combined scheduling
configuration rather than isolated mechanisms.

% --- §5.2 SGLang Integration ---

The SGLangAdapter integrates with SGLang's RadixAttention engine by hooking the
scheduling entry point.  To test portability, we deploy BidKV's complete
scheduling policy (SJF admission ordering, pressure-gated running-queue reorder,
and utility-ratio victim selection) on SGLang without modifying SGLang's
internals, and compare against Vanilla SGLang (SGLang's native LRU-based
eviction with no managed scheduling) and Random-Evict (BidKV's adapter with
random victim selection replacing the utility-ratio solver).
```

---

## Already resolved — do NOT re-raise these

| Issue | Resolution |
|---|---|
| Execution semantics buried after enumeration | Moved to opening paragraph of §5.1 |
| Pre-schedule title is verbose feature list | Simplified to `\textbf{Pre-schedule.}` |
| "four sub-steps" but actually five | Fixed to explicit (a)–(e) labeling |
| Co-varying dimensions not acknowledged | Added sentence: "Because multiple dimensions co-vary across strategies, Section 6 attributes performance differences to the combined scheduling configuration rather than isolated mechanisms." |
| `Mode A` / `Mode B` usage | Removed in prior round; not present in §5 |
| `quality_delta` / quality terminology | Removed in prior round; not present in §5 |

---

## Your evaluation task

For each question below, rate severity:
**critical** / **moderate** / **minor** / **none** — and suggest a specific fix
if not "none".

### Q1 — Does the opening paragraph of §5.1 successfully anchor victim selection as the core?
After the revision, the first sentence now reads: "BidKV's fundamental
integration point is victim selection: the adapter ranks running requests by
reclamation utility and delegates execution entirely to vLLM's native preemption
mechanism."
Does this successfully prevent the section from reading like a patch-note log?
Or does the subsequent five-step enumeration still overpower the framing?

### Q2 — Is the (a)–(e) structure clear and appropriately weighted?
Steps (a)–(e) vary in importance: (b) victim-priority refresh and (c) proactive
preemption are BidKV's core differentiation; (a) admission reorder and (e)
running-queue reorder are shared across multiple strategies; (d) SRPT is only
used by Static-Random and Largest-First (and explicitly disabled by BidKV).
Does the current structure give appropriate visual/narrative weight to the most
important steps, or does it flatten everything to equal weight?

### Q3 — Co-varying acknowledgment: right place, right wording?
The new sentence "Because multiple dimensions co-vary across strategies, Section
6 attributes performance differences to the combined scheduling configuration
rather than isolated mechanisms" appears in the Strategy differentiation
paragraph.
Is this the right location? Is the wording strong enough to set reader
expectations before they reach §6? Or is it too tentative?

### Q4 — Does §5.1 risk making BidKV look like a full scheduling policy bundle
rather than a bid-based victim-selection primitive?
The five sub-steps describe both BidKV-specific behavior (utility ranking,
gated reorder, SRPT disabled) and shared infrastructure (admission reorder,
priority refresh used by all non-PE strategies). A reviewer could conclude
the paper's contribution is the entire bundle, not the victim-selection
abstraction.
Does the current text adequately distinguish BidKV's unique mechanism from the
shared scaffolding? If not, what is the minimal fix?

### Q5 — SGLang Integration paragraph (§5.2): sufficient for portability claim?
The paragraph describes: (1) hooking the scheduling entry point, (2) deploying
BidKV's complete policy without modifying SGLang internals, (3) comparison
baselines (Vanilla SGLang, Random-Evict).
Is this enough to support the portability claim in §1 and §6.5? Or does it
leave reviewers uncertain about what "hooking the scheduling entry point" means
in practice?

### Q6 — Table 1 (Strategy differentiation): caption and column clarity
The table has five columns: Wait, Run, Victims, Proactive, SRPT.
Caption footnotes: "prio" = completion-aware keep score; "gated" = reorder
enabled only when KV>95% and mean prompt ≤500; SRPT = KV>80%.
Is the table self-explanatory after reading the caption? Does "N/A" in the
Victims column for PE/PE-SJF need clarification (these strategies have no
explicit victim-selection logic; vLLM's LIFO handles eviction)?

### Q7 — Terminology: any remaining drift from the paper's main narrative?
Check §5 for any uses of:
- "scheduling efficiency" (instead of admission responsiveness)
- "quality" in any form
- "compression" (when preemption is meant)
- description of SRPT or admission reorder as a "BidKV contribution" rather
  than shared infrastructure
Rate each finding separately.

### Q8 — Overall readiness of §5 after this revision
On a scale of:
- **needs another round** (structural or factual issues remain)
- **minor polish only** (small wording tweaks, no structural changes)
- **ready to submit** (section is done)

Where does Section 5 stand? Please justify briefly.
