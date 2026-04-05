# Section 6–7 Post-Polish Final Readiness Check — Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026) titled
"BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM Serving".
A terminology-alignment polish round has just been completed on §6 and §7.
This evaluation checks whether the paper is now ready to submit, with a focus
on three residual risks that were flagged in the previous audit but not yet
addressed.

---

## Paper context

- **Problem**: victim selection under KV cache pressure in LLM serving.
- **Core mechanism (§4.3, finalized)**: utility-ranked ordering via
  U = r / (δ + ε); cached; top-ranked entry consumed at each pressure event.
- **Primary value prop**: admission responsiveness — TTFT P95 and SLO(300ms).
  Throughput and TPOT are acknowledged tradeoffs (~7% throughput cost).
- **Execution model**: recompute-fallback throughout.

---

## What was fixed in the most recent §6–§7 polish round (do NOT re-raise)

| Change | Location |
|---|---|
| `bids → utility-ratio greedy solver` → `bids → utility-ranked victim ordering` | §6.1 BidKV bullet |
| `coordinated utility-ratio selection` (×2) → `utility-ranked cross-request victim selection` | §6.1 closing ¶, §6.2 attribution step 4 |
| `utility-ratio selection` → `utility-ranked victim selection` | §6.4 long-context ¶ |
| `utility-ratio victim selection` → `utility-ranked victim selection` | §6.5 SGLang ¶ |
| `the solver's degree of freedom is which requests to reclaim and how often` → `the policy's degree of freedom is which request is reclaimed at each pressure event` | §7.1 |
| BidKV bullet expanded with cached-ordering semantics | §6.4 Reclamation Event Analysis |

---

## Current state of sections under review (verbatim)

### §1 Contributions #4

```
Structured evaluation under controlled conditions. We evaluate five strategies
that vary across scheduling dimensions (admission ordering, running-queue
reordering, victim selection logic, and proactive reclamation) under frozen
request traces and calibrated arrival rates, supporting structured attribution
across these co-varying dimensions.
```

### §6.1 BidKV baseline bullet (after fix)

```
BidKV: SJF admission, pressure-gated running reorder (KV > 95%),
completion-aware cost estimation (δ, Eq.) → bids → utility-ranked
victim ordering; proactive reclamation enabled (KV > 90%), SRPT explicitly
disabled.
```

### §6.1 closing paragraph (after fix)

```
BidKV replaces all per-request heuristics with utility-ranked cross-request
victim selection. Each step in this progression changes multiple dimensions
simultaneously; we attribute performance differences to groups of co-varying
features rather than claim strict single-variable isolation.
```

### §6.4 BidKV bullet (after fix)

```
BidKV maintains a cached utility-ranked ordering of victim candidates using
U = r / (δ + ε); the top-ranked entry is consumed at each pressure event. The
disruption term δ grows with completion progress and prior preemption count,
steering selection toward requests whose eviction frees adequate capacity at low
reclamation cost. This produces the best TTFT P95 at rate = 3.8 while
maintaining competitive SLO.
```

### §7.1 Surrogate Disruption Estimate (after fix)

```
The disruption estimate δ is a ranking signal derived from request-lifecycle
features, not a precise reclamation-cost oracle. Exact cost prediction is
unnecessary: under recompute-fallback semantics the scheduler's task is to rank
victim candidates so that requests whose reclamation is relatively cheap are
preferred over expensive ones.

Because reclaimed requests are fully recomputed by the framework's native
recovery path, the policy preserves output correctness; the policy's degree of
freedom is which request is reclaimed at each pressure event. This degree of
freedom directly governs admission responsiveness: cost-aware victim ranking
reduces wasted recomputation and keeps KV capacity available for queued
requests, reducing TTFT and improving SLO attainment.
```

### §7.3 Broader Impact (unchanged)

```
Because bids make per-request disruption sensitivity an explicit input to the
scheduler, the interface can encode SLO-differentiated serving: higher-priority
requests express higher disruption cost, receiving stronger protection against
reclamation. A consequence is that lower-priority requests may absorb a
disproportionate share of preemption events. Per-request preemption caps---
limiting how often any single request can be reclaimed within a scheduling
window---provide a straightforward guard; exploring richer fairness-aware solver
extensions is left to future work.
```

### §8 Conclusion (key sentences)

```
BidKV surfaces per-request reclamation sensitivity as an explicit, structured
signal and performs online utility-ranked cross-request victim selection,
replacing implicit ordering heuristics with a utility-guided decision interface
(U = r / (δ + ε)). Under recompute-fallback semantics the policy preserves
output correctness; the differentiation lies in which requests are reclaimed
and how often---choices that directly govern how quickly queued requests receive
KV allocation and, consequently, first-token latency and SLO attainment.

Through a five-strategy evaluation on vLLM with Llama-3.1-8B-Instruct under
mixed-length ShareGPT workloads, we showed that BidKV achieves the best
cross-rate SLO attainment (87.1%) and TTFT P95 (562 ms), outperforming both
capacity-greedy heuristics and default scheduling baselines at a modest
throughput cost (~7%).
```

---

## Your evaluation task

For each question, rate severity: **critical / moderate / minor / none**, and
suggest a specific fix if not "none".

### Q1 — §7.3 "fairness-aware solver extensions": residual "solver" legacy

The §7.3 future-work sentence reads: "exploring richer **fairness-aware solver
extensions** is left to future work." After §4.3 dropped the "Constrained
Solver Layer" label and §7.1 was updated to use "policy" instead of "solver",
is "solver" in §7.3 a legacy inconsistency?

Note: "solver" in §7.3 refers to a _future extension_ (fairness-aware
richer optimization), not to the current implementation. Is this use of
"solver" defensible as forward-looking language, or will a reader who just
read §4.3 find it inconsistent?

Suggested fix if needed: "exploring richer fairness-aware scheduling
extensions is left to future work."

### Q2 — §8 Conclusion: "how often" reappears after §7.1 fix

Section 7.1 was updated to remove "how often" (overstatement of policy scope).
However, §8 Conclusion retains:

> "the differentiation lies in **which** requests are reclaimed and
> **how often**---choices that directly govern how quickly queued requests
> receive KV allocation."

This is now inconsistent with §7.1's corrected version. Is "how often" in
§8 also an overstatement (frequency is governed by pressure thresholds, not
by the ranking policy), or is it defensible in the conclusion as a higher-level
framing? Should §8 be aligned to "which request is reclaimed at each pressure
event" to match §7.1?

### Q3 — Contributions #4 vs evaluation scope: "five strategies"

The Contributions claim (§1) says: "We evaluate **five** strategies."
The §6.1 Baselines list explicitly names exactly five strategies (PE, PE-SJF,
Static-Random, Largest-First, BidKV), and the §8 conclusion also references
"a five-strategy evaluation."

The question is whether the paper's result tables (Table 2 rate sensitivity,
Table 3 long-context, Table 4 SGLang) contain any additional strategies beyond
these five. If Slack-Aware or Uniform appear in any table, the "five strategies"
claim in §1 Contributions #4 and §8 would be factually incorrect.

Please check: given the paper's narrative, is the "five strategies" claim
self-consistent with the tables presented? If additional strategies appear in
the tables without being named in the Contributions claim, is a brief
disclosure ("five primary strategies; two additional baselines appear in
supplementary comparison") sufficient?

### Q4 — §6.4 cached-ordering description: is the new sentence over-specified?

The new BidKV bullet (§6.4) now reads:

> "BidKV maintains a cached utility-ranked ordering of victim candidates
> using U = r / (δ + ε); the top-ranked entry is consumed at each pressure
> event."

This is accurate but introduces a mechanistic level of detail (caching,
pop-per-event) in an analysis section whose neighboring bullets for PE,
PE-SJF, and Largest-First describe behavior at a higher level. Does this
create a depth inconsistency between the BidKV bullet and the others? Or is
the extra specificity justified because BidKV's mechanism is more novel and
warrants explicit explanation?

### Q5 — §7.1 "the policy's degree of freedom is which request"

After the fix, §7.1 reads: "the policy's degree of freedom is **which** request
is reclaimed at each pressure event." Two follow-up questions:

(a) Is replacing "the solver's" with "the policy's" sufficient, or does
  "degree of freedom" itself feel overly formal for what is essentially a
  one-sentence summary of BidKV's design scope?

(b) The sentence originally claimed two degrees of freedom ("which" and
  "how often"). After removing "how often", is the single remaining degree
  of freedom ("which request at each event") complete enough, or does it
  leave out a relevant dimension (e.g., threshold tuning) that the original
  sentence captured via "how often"?

### Q6 — Overall final readiness

After the §6–§7 polish round and this final check, on a scale of:
- **needs another round** (structural or factual issues remain)
- **minor polish only** (1–2 isolated wording fixes)
- **ready to submit** (the full paper §1–§8 is submission-ready)

Where does the paper stand? Please justify and list the single highest-priority
remaining action, if any.
