# Section 4 Theoretical Fix + Global Narrative Cleanup — Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026) titled
"BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM Serving".
A comprehensive revision round has just been completed. Please evaluate
whether the revision is internally consistent, technically accurate, and
narratively sound. **Do not re-raise items listed in the "already resolved"
table.**

---

## Paper context

- **Problem**: victim selection under KV cache pressure in LLM serving.
- **Value proposition**: improved *admission responsiveness* — lower TTFT P95
  and higher SLO attainment at 300 ms. Throughput and TPOT are acknowledged
  tradeoffs, not primary claims.
- **Core mechanism**: each active request produces a bid with two fields:
  `r` (KV tokens freed if preempted) and `δ` (disruption estimate from
  lifecycle features). Utility U = r / (δ + ε). The runtime maintains a
  utility-ranked ordering of all active bids; at each pressure event, the
  top-ranked entry is selected as the intended victim and preempted via the
  framework's native mechanism.
- **Execution model**: recompute-fallback throughout. Reclaiming a request
  releases KV blocks and re-queues for full prompt recomputation; the policy
  does not alter final output tokens.
- **Integration**: non-invasive adapter abstraction; vLLM and SGLang
  integrations require no framework source-code modification.
- **What this paper does NOT claim**: exact optimization, provably minimum
  TTFT, throughput leadership, output-quality improvement.

---

## What changed in this revision round (12 changes)

### Change 1 — Abstract: solver description updated

**Before:**
```
a coordinated solver selects victims that free the required capacity
at minimum aggregate cost.
```

**After:**
```
at each pressure event, an online utility-ranked selection policy identifies
the intended highest-utility victim and preempts it via the framework's native
mechanism.
```

---

### Change 2 — Intro ¶1: reclamation trigger reframed

**Before:**
```
can repeatedly exceed available capacity during normal operation, forcing the
serving engine to reclaim KV state from active requests to make room for others.
```

**After:**
```
routinely approaches hardware capacity limits; modern serving engines reclaim
KV state from active requests---triggered at a high-watermark threshold before
saturation---to preserve scheduling flexibility and ensure that low-cost victim
choices remain available as new requests arrive.
```

---

### Change 3 — Intro ¶4: BidKV description updated

**Before:**
```
the scheduler solves a constrained cross-request selection problem: choose the
victim set that recovers the required capacity while minimizing aggregate
estimated disruption.
```

**After:**
```
the scheduler performs an online utility-ranked cross-request victim selection:
at each pressure event, the request offering the most recoverable KV capacity
per unit of estimated disruption is selected as the intended victim and
preempted via the framework's native mechanism.
```

---

### Change 4 — §2.2 Eq.(1): bridge sentence added

Inserted immediately after the sentence "Section 4 instantiates $c_i$ with a
surrogate derived from request-level lifecycle features":

**New text (inserted):**
```
Minimizing aggregate reclamation cost is monotonically connected to admission
responsiveness: cheaper victims keep less KV capacity occupied by
non-productive re-prefill recomputation, accelerating queue drainage and
shortening the wait before queued requests are admitted.
```

---

### Change 5 — §2.2: global execution-model declaration added

Inserted between the two requirements paragraph and the "Empirical evidence"
paragraph:

**New text (inserted):**
```
\noindent\textbf{Execution model.}  Throughout this paper we adopt the
standard recompute-fallback execution model used by current serving
engines: reclaiming a request releases its KV blocks and re-queues it for
full prompt recomputation when rescheduled; the reclamation policy does not
alter final output tokens.
```

---

### Change 6 — §3 Positioning: solver wording updated

**Before:**
```
a constrained solver coordinates selection across the batch
```

**After:**
```
a utility-ranked online selection pass coordinates the choice across the batch
```

---

### Change 7 — §4 Overview: layer name updated

**Before:**
```
the Constrained Solver Layer selects reclamation victims
```

**After:**
```
the Utility-Ranked Selection Layer produces a cached reclamation-priority ordering
```

---

### Change 8 — §4.1 BidAcceptance sentence updated

**Before:**
```
The solver output is a BidAcceptance containing the accepted bid IDs,
total freed tokens, and cumulative cost.
```

**After:**
```
The solver output is a utility-ranked ordering of active bids; the runtime
adapter consumes the current top-ranked entry at each pressure event.
```

---

### Change 9 — §4.3 COMPLETE REWRITE (core structural fix)

**Old title:** `4.3 Constrained Solver Layer`
**New title:** `4.3 Online Utility-Ranked Victim Selection`

**Old body (key elements deleted):**
- "when the pressure detector signals that N tokens must be freed"
- disruption budget constraint (Σδ ≤ Δmax)
- cumulative coverage constraint (Σr ≥ N)
- BidAcceptance acceptance set
- "Relation to knapsack problems" paragraph

**New body (full replacement):**
```
Equation (victim) states the idealized batch reclamation objective;
practical serving requires an online approximation that fits within the
engine's tick-based scheduling loop.  At each scheduling step, the
GreedyBidSolver computes a reclamation utility score for every
active bid and produces a utility-ranked ordering of all candidates
(Algorithm 1).  This ordering is cached and refreshed periodically;
at each pressure event, the runtime adapter consumes the current
top-ranked entry and executes one native preemption.

The utility score is drawn directly from Eq. (utility):
U = r / (δ + ε).  Higher utility means the request frees more KV
capacity per unit of estimated reclamation disruption; the top-ranked
request is the intended highest-utility victim.  Because δ serves as
an ordinal ranking signal, the selection requires only that relative
ordering be preserved across bids---exact magnitude calibration is not
needed.

The ranking procedure runs in O(B log B) time.  In practice B < 100,
so ranking takes microseconds---negligible compared with a decode step.
```

---

### Change 10 — Algorithm 1 COMPLETE REWRITE

**Old caption:** `Utility-Ratio Greedy Knapsack Solver`
**Old inputs:** Bid pool P; tokens needed N; disruption budget Δmax
**Old output:** Acceptance set A
**Old logic:** Iterative greedy with two break conditions (Σr ≥ N, Σδ > Δmax);
  accumulates A, Σr, Σδ; returns (A, Σr, Σδ)

**New caption:** `Utility-Ranked Victim Ordering`
**New inputs:** Bid pool P only
**New output:** Utility-ranked ordering O
**New logic:**
```
Require: Bid pool P = {b1, b2, ..., bB}
Ensure: Utility-ranked ordering O
for all b in P:
    U(b) = b.r / (b.δ + ε)   // reclamation utility, ε = 1e-3
Sort P by U(b) descending → O
Return O   // cached; adapter pops top entry at each pressure event
```

---

### Change 11 — §5.1 step (e): runtime projection note added

To the end of step (e) "Running-queue reorder", appended:

**New text (appended):**
```
the adapter maps high reclamation utility to low keep-priority, so the
lowest-priority request removed by the native LIFO back-end corresponds
to the intended highest-utility victim from the ranking.
```

---

### Change 12 — §8 Conclusion: solver wording updated

**Before:**
```
solves a coordinated cross-request victim-selection problem, replacing
implicit ordering heuristics
```

**After:**
```
performs online utility-ranked cross-request victim selection, replacing
implicit ordering heuristics
```

---

## Current state of key sections (verbatim, simplified LaTeX)

### Abstract

```
When KV-cache demand exceeds GPU memory capacity during online LLM serving,
serving engines reclaim cache from active requests to admit waiting ones. Poor
victim selection wastes recomputation, delays admission of queued requests, and
degrades first-token latency---a metric directly visible to users. Existing
policies rely on coarse request-order heuristics (LIFO, FCFS) or independent
per-request scoring rules, lacking both explicit reclamation-cost signals and
cross-request coordination. We present BidKV, a utility-guided reclamation
policy that improves admission responsiveness under KV pressure. Each active
request produces a bid encoding recoverable capacity and estimated disruption
cost; at each pressure event, an online utility-ranked selection policy
identifies the intended highest-utility victim and preempts it via the
framework's native mechanism. The current prototype instantiates disruption cost
from request-lifecycle proxies (completion progress, prompt length, preemption
history) under recompute-fallback semantics. BidKV integrates non-invasively
with vLLM and SGLang via a portable adapter layer. Evaluated with
Llama-3.1-8B-Instruct on an NVIDIA RTX A6000 under ShareGPT traces, BidKV
reduces P95 TTFT by 89% and improves 300ms SLO attainment by 14.9pp over
vLLM's native policy, at a modest throughput cost.
```

### §2.2 Victim-Selection Problem (Eq.1 + bridge + global declaration)

```
Under recompute-fallback semantics, each reclamation event releases the
victim's KV blocks and re-queues the request; when rescheduled, its prompt is
recomputed from scratch.  The scheduler's task at each such event is to choose
a victim set V ⊆ R such that total freed capacity meets the shortfall while
minimizing aggregate reclamation cost:

    min_{V ⊆ R} Σ_{i ∈ V} c_i  s.t.  Σ_{i ∈ V} r_i ≥ N          ... (1)

where r_i is the KV capacity freed, c_i is the reclamation cost, and N is the
capacity shortfall.  The dominant component of c_i is the work already invested
in the victim; Section 4 instantiates c_i with a surrogate derived from
request-level lifecycle features.
Minimizing aggregate reclamation cost is monotonically connected to admission
responsiveness: cheaper victims keep less KV capacity occupied by
non-productive re-prefill recomputation, accelerating queue drainage and
shortening the wait before queued requests are admitted.

This formulation is an instance of the minimum-cost covering problem.
[... analysis motivating two requirements ...]

Execution model.  Throughout this paper we adopt the standard recompute-fallback
execution model used by current serving engines: reclaiming a request releases
its KV blocks and re-queues it for full prompt recomputation when rescheduled;
the reclamation policy does not alter final output tokens.

[Empirical evidence paragraph follows]
```

### §4 Overview

```
BidKV organizes its logic into four layers (Figure 2): the Runtime Adapter Layer
detects KV pressure and executes reclamations via the framework's native
mechanism; the Bid Generation Layer computes disruption cost from
request-lifecycle features and produces structured bids; the Utility-Ranked
Selection Layer produces a cached reclamation-priority ordering; and the Bid
Signal Layer defines the shared data structures used across all layers.
```

### §4.1 Bid Signal Layer (BidPool sentence)

```
Bids are collected in a BidPool---an immutable snapshot of all active bids at
a given instant.  The solver output is a utility-ranked ordering of active bids;
the runtime adapter consumes the current top-ranked entry at each pressure event.
```

### §4.3 Online Utility-Ranked Victim Selection (complete)

```
Equation (1) states the idealized batch reclamation objective; practical serving
requires an online approximation that fits within the engine's tick-based
scheduling loop.  At each scheduling step, the GreedyBidSolver computes a
reclamation utility score for every active bid and produces a utility-ranked
ordering of all candidates (Algorithm 1).  This ordering is cached and refreshed
periodically; at each pressure event, the runtime adapter consumes the current
top-ranked entry and executes one native preemption.

[Algorithm 1 inserted here]

The utility score is drawn directly from Eq. (utility): U = r / (δ + ε).
Higher utility means the request frees more KV capacity per unit of estimated
reclamation disruption; the top-ranked request is the intended highest-utility
victim.  Because δ serves as an ordinal ranking signal, the selection requires
only that relative ordering be preserved across bids---exact magnitude
calibration is not needed.

The ranking procedure runs in O(B log B) time.  In practice B < 100, so
ranking takes microseconds---negligible compared with a decode step.
```

### Algorithm 1

```
Require: Bid pool P = {b1, ..., bB}
Ensure:  Utility-ranked ordering O

for all b in P:
    U(b) = b.r / (b.δ + ε)   // reclamation utility, ε = 1e-3
Sort P by U(b) descending → O
Return O                       // cached; adapter pops top entry at each pressure event
```

### §5.1 step (e) (key sentence added)

```
(e) Running-queue reorder: strategies with this enabled sort the running list
by cached priority so that vLLM's native LIFO eviction removes the
lowest-priority request.  BidKV gates this reorder to KV utilization >95% and
mean prompt length ≤500 tokens; the adapter maps high reclamation utility to
low keep-priority, so the lowest-priority request removed by the native LIFO
back-end corresponds to the intended highest-utility victim from the ranking.
```

---

## Already resolved — do NOT re-raise these

| Issue | Resolution |
|---|---|
| Abstract claimed "minimum aggregate cost" via "coordinated solver" | Changed to "online utility-ranked selection" in this round |
| Intro framed reclamation as OOM rescue | Changed to high-watermark-triggered / preserve scheduling flexibility |
| §4.3 had knapsack/Σδ/Δmax/Σr≥N constraints | Complete rewrite to ranking-only semantics in this round |
| Algorithm 1 was a greedy knapsack with acceptance set | Replaced with sort-and-cache algorithm in this round |
| BidAcceptance acceptance-set semantics in §4.1 | Replaced with utility-ranked ordering sentence in this round |
| No bridge from Eq.(1) to admission responsiveness | Added monotonic connection sentence in this round |
| No global execution-model declaration | Added Execution model block in §2.2 in this round |
| §3 Positioning used "constrained solver" | Changed to "utility-ranked online selection pass" in this round |
| §4 Overview named "Constrained Solver Layer" | Renamed to "Utility-Ranked Selection Layer" in this round |
| §5.1 step (e) had a semantic gap between utility ranking and native LIFO | Added projection note in this round |
| §8 Conclusion used "coordinated cross-request victim-selection problem" | Changed to "online utility-ranked cross-request victim selection" in this round |
| Mode A / Mode B terminology | Removed in prior round |
| quality_delta / quality-optimization framing | Removed in prior round |
| N/A in Table 1 Victims column for PE/PE-SJF | Changed to LIFO* with caption footnote in prior round |

---

## Your evaluation task

For each question, rate severity: **critical / moderate / minor / none**, and
suggest a specific fix if not "none".

### Q1 — Does Eq.(1) remain consistent with the new online approximation claim?

Section 2.2 retains Eq.(1) (the idealized batch optimization). Section 4.3 now
explicitly calls the algorithm an "online approximation" of this ideal. The
bridge sentence in §2.2 says minimizing aggregate cost is "monotonically
connected to" admission responsiveness.

Is this three-layer structure (idealized formulation → online approximation →
admission responsiveness) narratively sound? Or does retaining Eq.(1) while
calling the runtime algorithm an "online approximation" create a tension that
reviewer would flag as "the paper claims to solve a batch optimization problem
but actually does sorting"?

### Q2 — Does Algorithm 1's simplicity weaken the systems contribution?

The new Algorithm 1 is: compute U per bid → sort descending → return ordering.
This is a 3-line procedure. For a systems conference, is this sufficient as
"Algorithm 1"? Or does the simplification risk making the contribution appear
trivial? If so, what can be added without re-introducing the old knapsack
complexity?

### Q3 — Does the §4.3 section body adequately motivate WHY an online
approximation is used instead of batch solving?

The current text says "practical serving requires an online approximation that
fits within the engine's tick-based scheduling loop." This is stated but not
elaborated. Is one sentence of motivation sufficient, or should the section
briefly explain why the batch formulation is impractical in the online setting
(e.g., latency constraints, need for incremental updates)?

### Q4 — §5.1 projection note: does "corresponds to" create an over-claim?

The new sentence reads: "the adapter maps high reclamation utility to low
keep-priority, so the lowest-priority request removed by the native LIFO
back-end corresponds to the intended highest-utility victim from the ranking."

Is "corresponds to" too strong? It implies the native LIFO invariably selects
the top-ranked victim, which is only guaranteed if the ranking is faithfully
projected into the LIFO priority and no concurrent scheduling event
intervenes. Could a reviewer flag this as an unverified
implementation-correctness claim?

### Q5 — Does the intro ¶4 "coordinated selection layer" label survive the §4
redesign?

Intro ¶4 (unchanged) still refers to "a coordinated selection layer---which
makes batch-level trade-offs across candidates". After §4.3 is now an
online ranking (not a batch solver), does "batch-level trade-offs" in the
intro become misleading? Should "batch-level" be softened to "cross-request"?

### Q6 — Does §3 Positioning remain consistent?

§3 now says "a utility-ranked online selection pass coordinates the choice
across the batch". The same paragraph still cites Martello1990/Kellerer2004
for "constrained subset selection---the same combinatorial structure that
underlies the victim-selection formulation".
After §4.3 no longer does constrained subset selection, is the connection to
the knapsack literature in §3 still accurate? Or does it now mislead readers
into thinking BidKV solves a knapsack problem?

### Q7 — Three uses of "coordinated" still in text: correct or legacy?

After the changes, "coordinated" still appears in:
- Intro ¶3: "coordinated batch-level selection rather than independent per-request rules" (requirement statement)
- Intro ¶4: "coordinated selection layer" (architecture description)
- Abstract still removed "coordinated solver" (confirmed)

Are these two remaining uses of "coordinated" consistent with the new online
ranking semantics? Or do they imply a solver-style mechanism that no longer
exists?

### Q8 — Overall consistency check

After this round of 12 changes, scan the key narrative claim chain:
Abstract → Intro ¶1 → §2.2 formulation → §4 architecture → §4.3 algorithm →
§5.1 integration → §8 conclusion.

Is this chain now internally consistent with the "online utility-ranked
approximation of the idealized batch problem" framing? Or are there remaining
seams where the old "constrained solver / minimum aggregate cost" framing leaks
through?

### Q9 — Overall readiness after this revision round

On a scale of:
- **needs another round** (structural or factual issues remain)
- **minor polish only** (small wording tweaks, no structural changes)
- **ready to submit** (sections §1–§4 are done)

Where do Sections 1–4 stand after this revision? Please justify briefly.
