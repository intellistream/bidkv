# Section 6–7 Consistency Audit — Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026) titled
"BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM Serving".
Sections 1–5 have been revised in prior rounds and are considered stable.
This round focuses on checking Sections 6 (Evaluation) and 7 (Discussion
and Limitations) for consistency with the §4.3 redesign that was completed
in the previous round.

---

## Paper context

- **Problem**: victim selection under KV cache pressure in LLM serving.
- **Value proposition**: improved *admission responsiveness* — lower TTFT P95
  and higher SLO attainment at 300 ms. Throughput and TPOT are acknowledged
  tradeoffs.
- **Core mechanism (§4.3, finalized)**: each active request produces a bid
  with `r` (KV tokens freed) and `δ` (disruption estimate). Utility
  U = r / (δ + ε). The `GreedyBidSolver` sorts all bids by U descending and
  returns a **utility-ranked ordering** (Algorithm 1). This ordering is cached;
  the runtime adapter **pops the top-ranked entry at each pressure event**.
  This is explicitly called an "online utility-ranked approximation" of
  the idealized batch problem — **not a constrained solver, not a greedy
  knapsack accumulator, not an acceptance-set producer**.
- **Key terminology shift** (§4.3 rewrite in prior round):
  - OLD: "Constrained Solver Layer", "knapsack", "BidAcceptance acceptance set",
    Σδ ≤ Δmax, Σr ≥ N, "minimum aggregate cost"
  - NEW: "Online Utility-Ranked Victim Selection", "utility-ranked ordering",
    "online approximation", "sort-and-cache semantics"

---

## What was fixed in the §4.3 rewrite (already resolved — do NOT re-raise)

| Issue | Resolution |
|---|---|
| §4.3 title was "Constrained Solver Layer" | Renamed to "Online Utility-Ranked Victim Selection" |
| Algorithm 1 was greedy knapsack with acceptance set | Replaced with sort-descending-by-U, return ordering |
| BidAcceptance acceptance-set semantics in §4.1 | Replaced with "utility-ranked ordering" sentence |
| Abstract used "coordinated solver / minimum aggregate cost" | Changed to "online utility-ranked selection" |
| Intro ¶3 used "coordinated batch-level selection" | Changed to "coordinated cross-request selection" |
| Intro ¶4 used "a coordinated selection layer---making batch-level trade-offs" | Changed to "ranks candidates by cross-request reclamation utility" |
| §3 knapsack paragraph implied BidKV solves constrained problem | Rewritten to explicitly position BidKV as online approximation |
| §5.1 step (e) had semantic gap between utility ranking and native LIFO | Added projection note |
| §8 Conclusion used "coordinated cross-request victim-selection problem" | Changed to "performs online utility-ranked cross-request victim selection" |

---

## Current state of §6 and §7 (verbatim, simplified LaTeX)

### §6.1 Experimental Setup — Baselines paragraph

```
Five strategies form a layered comparison that varies waiting-queue ordering,
running-queue reordering, proactive reclamation, and victim selection logic:

1. PE (vLLM default): FCFS admission, LIFO preemption, no proactive reclamation.

2. PE-SJF: PE plus SJF waiting-queue ordering; isolates the contribution of
   admission-queue reordering.

3. Static-Random: SJF admission, proactive reclamation (KV > 90%), random
   victim selection, SRPT preemption (KV > 80%).

4. Largest-First: SJF admission, proactive reclamation, capacity-greedy
   victim---evicts the request occupying the most KV blocks; SRPT enabled.

5. BidKV: SJF admission, pressure-gated running reorder (KV > 95%),
   completion-aware cost estimation (δ, Eq.) → bids → utility-ratio
   greedy solver; proactive reclamation enabled (KV > 90%), SRPT explicitly
   disabled.

The comparison spans a range of reclamation philosophies: PE applies no
active scheduling; PE-SJF isolates the contribution of admission ordering;
Static-Random and Largest-First add proactive reclamation with simple
heuristics; BidKV replaces all per-request heuristics with coordinated
utility-ratio selection.
```

### §6.2 Main Comparison — Attribution chain (step 4)

```
(4) Largest-First → BidKV (+3.3 pp SLO, TTFT 677→631 ms):
    replacing capacity heuristics with coordinated utility-ratio selection
    and disabling SRPT, confirming that reclamation-cost-aware victim
    coordination provides incremental value beyond simple size-based heuristics.
```

### §6.4 Reclamation Event Analysis — BidKV bullet

```
BidKV uses the utility ratio U = r / (δ + ε) to rank victims. The disruption
term δ grows with completion progress and prior preemption count, steering
selection toward requests whose eviction frees adequate capacity at low
reclamation cost. This produces the best TTFT P95 at rate = 3.8 while
maintaining competitive SLO.
```

### §7.1 Surrogate Disruption Estimate (complete)

```
The disruption estimate δ (Eq.) is a ranking signal derived from
request-lifecycle features, not a precise reclamation-cost oracle. Exact cost
prediction is unnecessary: under recompute-fallback semantics the scheduler's
task is to rank victim candidates so that requests whose reclamation is
relatively cheap are preferred over expensive ones. The three features---
completion progress, prompt length, and preemption history---capture the axes
of reclamation heterogeneity established in Section 2 and need only preserve
relative ordering across these dimensions to steer selection toward low-cost
victims.

Because reclaimed requests are fully recomputed by the framework's native
recovery path, the policy preserves output correctness; the solver's degree of
freedom is which requests to reclaim and how often. This degree of freedom
directly governs admission responsiveness: cost-aware victim ranking reduces
wasted recomputation and keeps KV capacity available for queued requests,
reducing TTFT and improving SLO attainment.
```

### §7.2 Limitations

```
- Single GPU. All experiments use a single A6000. Multi-GPU deployments may
  introduce coordination overhead for bid collection.
- Single model scale. We evaluate on Llama-3.1-8B-Instruct. Larger models
  (70B+) with different KV-to-weight ratios may exhibit different pressure
  dynamics.
- Execution model. All experiments use recompute-fallback semantics, the
  default recovery path in vLLM and SGLang. Alternative execution models
  (e.g., swap-to-host, partial truncation) may change the cost structure;
  adapting the scorer to such models is future work.
```

### §8 Conclusion (already fixed, shown for reference)

```
We presented BidKV, a bid-based scheduling abstraction for active-KV
reclamation in online LLM serving. BidKV surfaces per-request reclamation
sensitivity as an explicit, structured signal and performs online
utility-ranked cross-request victim selection, replacing implicit ordering
heuristics with a utility-guided decision interface (U = r / (δ + ε)).
```

---

## Your evaluation task

For each question, rate severity: **critical / moderate / minor / none**, and
suggest a specific fix if not "none".

### Q1 — "utility-ratio greedy solver" in §6.1 BidKV bullet

The §6.1 description of BidKV's pipeline ends with `bids → utility-ratio
greedy solver`. After §4.3 was renamed to "Online Utility-Ranked Victim
Selection" and the algorithm was changed to sort-and-return (not greedy
accumulation), is "utility-ratio greedy solver" now factually incorrect
terminology that will confuse a reader who just read §4.3?

Compare: §4.3 and Algorithm 1 now use "utility-ranked ordering" and
"GreedyBidSolver" only as a class name (not as a description). Should `bids →
utility-ratio greedy solver` be changed to something like `bids → utility-
ranked victim ordering`?

### Q2 — "coordinated utility-ratio selection" in §6 body text (×2 occurrences)

After the §6.1 bullet list, the paragraph closes: "BidKV replaces all
per-request heuristics with **coordinated utility-ratio selection**." The same
phrase appears in the Attribution chain (step 4): "replacing capacity heuristics
with **coordinated utility-ratio selection**."

The canonical phrasing in §4 is "utility-ranked cross-request victim selection"
(and in §3/§8: "utility-ranked approximation"). Is "utility-ratio selection" a
harmless shorthand or a terminology inconsistency that a reviewer tracking
nomenclature would notice? Should these two occurrences be brought into
alignment with the canonical phrasing?

### Q3 — "the solver's degree of freedom" in §7.1

Section 7.1 contains the sentence: "the **solver's** degree of freedom is which
requests to reclaim and how often." After §4.3 dropped the "Constrained Solver
Layer" label, is "solver" here a legacy reference that should be changed to
"policy" or "scheduler"? Or is "solver" acceptable as a generic term for the
component that decides victim order?

Note: the code class is still called `GreedyBidSolver`; the paper's §4.3 title
is "Online Utility-Ranked Victim Selection". Is the word "solver" in §7.1 a
cross-section inconsistency?

### Q4 — §7.1 "how often" claim: does the policy actually control eviction frequency?

The sentence reads: "the solver's degree of freedom is **which** requests to
reclaim and **how often**." BidKV controls *which* request to pop via the
utility ranking. But does it also control "how often" reclamation happens?
Proactive reclamation is triggered by pressure thresholds (KV > 90%), not by
the utility ranking itself. Is "how often" an overstatement? Should the claim
be narrowed to "which request is reclaimed at each pressure event"?

### Q5 — §6.4 "rank victims": correct representation of pop-once-per-event semantics?

The reclamation analysis bullet says BidKV "uses the utility ratio U = r/(δ+ε)
to **rank victims**." This matches the new §4.3 semantics (sort, cache, pop top
at each event). Is this description complete enough, or should it also note
the cached-ordering aspect ("the ordering is cached; the top-ranked entry
is consumed at each pressure event") to avoid the impression that BidKV
re-sorts the full bid pool on every pressure event?

### Q6 — Contributions #4 says "five strategies" but full table has more

The Contributions paragraph (Intro §1) ends with: "We evaluate **five**
strategies that vary across scheduling dimensions." However, the Evaluation
chapter's §6.1 explicitly lists five baselines (PE, PE-SJF, Static-Random,
Largest-First, BidKV). If the paper's full result tables contain additional
strategies (e.g., Slack-Aware, Uniform), is there a mismatch between the
Contributions claim and the actual evaluation scope? Conversely, if the
narrative legitimately focuses on five while tables may include more, is that
sufficiently disclosed?

### Q7 — §3 "utility-ranked approximation" forward ref vs §6 "utility-ratio"

Section 3 (Related Work, already revised) uses "utility-ranked approximation"
and forward-references §4.3. Section 6 uses "utility-ratio selection" and
"utility-ratio greedy solver". Will a reader notice that §3 says "ranked" while
§6 says "ratio"? Are these two distinct descriptors that should be unified, or
is the distinction intentional (§3 describes the approach class; §6 names a
specific formula)?

### Q8 — Global framing consistency: §6 opens with "reclamation event analysis"

The §6 overview paragraph lists dimension (4) as "reclamation event analysis."
This is consistent with the mechanism-first narrative. Does the opening
paragraph of §6 need any update to reflect the new "online utility-ranked
approximation" framing (e.g., does it currently use any language that echoes
"constrained solver" or "batch optimization")?

### Q9 — Overall readiness of §6–§7

After this audit, on a scale of:
- **needs another round** (structural or factual issues remain)
- **minor polish only** (small wording tweaks, no structural changes)
- **ready to submit** (sections §6–§7 are done)

Where do Sections 6 and 7 stand? Please justify briefly and list the
highest-priority fix if any issues remain.
