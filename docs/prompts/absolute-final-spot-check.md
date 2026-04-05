# Absolute Final Pre-Submission Spot-Check — Evaluation Prompt

You are performing a single-pass final spot-check on a systems conference paper
(SC 2026) titled "BidKV: Utility-Guided Preemption Scheduling for KV-Pressure
LLM Serving" before submission. All major revision rounds are complete. This
check is narrow: confirm no sentence-level issues remain and give a submission
verdict.

---

## All accumulated fixes (do NOT re-raise anything in this list)

| Round | Changes |
|---|---|
| §4 audit | §4.3 terminology cleanup, Mode A/B removal |
| §5 structural + final | Victim-selection framing, Table 1 LIFO* |
| §4.3 + Alg 1 rewrite | Knapsack → ranking semantics |
| 12-change global cleanup | Abstract, Intro ×2, §2.2 ×2, §3, §4 overview, §4.1, §4.3 title+body, §5.1, §8 |
| 3 follow-up fixes | "batch-level" ×2, §3 knapsack paragraph |
| §6–§7 polish (7 fixes) | greedy solver, utility-ratio ×5, solver's degree, how often |
| 2-fix round | §8 "how often" removed; §7.3 "solver" → "scheduling" |
| Final micro-fix round | §7.1 "degree of freedom" → "governs / choice"; §8 "directly governs" → "directly shapes" |

---

## Current verbatim text of the four sections most recently changed

### §7.1 Surrogate Disruption Estimate (final)

```
The disruption estimate δ is a ranking signal derived from request-lifecycle
features, not a precise reclamation-cost oracle. Exact cost prediction is
unnecessary: under recompute-fallback semantics the scheduler's task is to rank
victim candidates so that requests whose reclamation is relatively cheap are
preferred over expensive ones. The three features---completion progress, prompt
length, and preemption history---capture the axes of reclamation heterogeneity
established in Section 2 and need only preserve relative ordering across these
dimensions to steer selection toward low-cost victims.

Because reclaimed requests are fully recomputed by the framework's native
recovery path, the policy preserves output correctness; the policy governs
which request is reclaimed at each pressure event.
This choice directly shapes admission responsiveness: cost-aware victim ranking
reduces wasted recomputation and keeps KV capacity available for queued
requests, reducing TTFT and improving SLO attainment.
```

### §7.3 Broader Impact (final)

```
Because bids make per-request disruption sensitivity an explicit input to the
scheduler, the interface can encode SLO-differentiated serving: higher-priority
requests express higher disruption cost, receiving stronger protection against
reclamation. A consequence is that lower-priority requests may absorb a
disproportionate share of preemption events. Per-request preemption caps---
limiting how often any single request can be reclaimed within a scheduling
window---provide a straightforward guard; exploring richer fairness-aware
scheduling extensions is left to future work.
```

### §8 Conclusion, opening paragraph (final)

```
BidKV surfaces per-request reclamation sensitivity as an explicit, structured
signal and performs online utility-ranked cross-request victim selection,
replacing implicit ordering heuristics with a utility-guided decision interface
(U = r / (δ + ε)). Under recompute-fallback semantics the policy preserves
output correctness; the differentiation lies in which request is selected for
reclamation at each pressure event---a choice that directly shapes how quickly
queued requests receive KV allocation and, consequently, first-token latency
and SLO attainment.
```

### Abstract (unchanged, for reference)

```
When KV-cache demand exceeds GPU memory capacity during online LLM serving,
serving engines reclaim cache from active requests to admit waiting ones. Poor
victim selection wastes recomputation, delays admission of queued requests, and
degrades first-token latency. Existing policies rely on coarse request-order
heuristics (LIFO, FCFS) or independent per-request scoring rules, lacking both
explicit reclamation-cost signals and cross-request coordination. We present
BidKV, a utility-guided reclamation policy that improves admission
responsiveness under KV pressure. Each active request produces a bid encoding
recoverable capacity and estimated disruption cost; at each pressure event, an
online utility-ranked selection policy identifies the intended highest-utility
victim and preempts it via the framework's native mechanism. BidKV integrates
non-invasively with vLLM and SGLang via a portable adapter layer. Evaluated
with Llama-3.1-8B-Instruct on an NVIDIA RTX A6000 under ShareGPT traces,
BidKV reduces P95 TTFT by 89% and improves 300ms SLO attainment by 14.9pp
over vLLM's native policy, at a modest throughput cost.
```

---

## Spot-check questions (narrow scope — final pass only)

### Q1 — §7.1 sentence break: "the policy governs ... This choice"

The revision produced two short sentences:

> "the policy governs which request is reclaimed at each pressure event.
> This choice directly shapes admission responsiveness: ..."

Does this two-sentence structure flow naturally, or does the period + "This
choice" feel choppy compared to the prior single-sentence form? Would a dash
or a relative clause ("...at each pressure event — a choice that directly
shapes...") read more smoothly without changing the meaning?

### Q2 — Abstract: bid described as "encoding recoverable capacity and estimated disruption cost"

The Abstract says bids "encode recoverable capacity and estimated disruption
cost." The conclusion says BidKV uses "U = r / (δ + ε)". Is the Abstract's
description precise enough that a reader can infer U = r/(δ+ε), or is
"encoded as a ratio" missing and potentially confusing?

Note: the Abstract deliberately avoids the formula. Is this level of
abstraction standard and acceptable for SC, or should the Abstract include
the ratio structure?

### Q3 — §8 "directly shapes": is this weaker than needed?

After changing "directly governs" to "directly shapes", does the causal
claim now feel too weak for the conclusion? Compare:

- "directly governs how quickly" — strong, arguably over-claims direct causation
- "directly shapes how quickly" — softer, arguably understates the causal chain
- "determines how quickly" — middle ground?

Is "shapes" the right word here, or would "determines" or "influences" better
match the empirical evidence presented in §6?

### Q4 — §7.3 "how often" in preemption caps: any residual confusion?

§7.3 still contains: "Per-request preemption caps---limiting how often any
single request can be reclaimed within a scheduling window."

This "how often" is per-request frequency control (a fairness mechanism),
not a statement about BidKV's ranking policy scope. After the §7.1 and §8
"how often" fixes, does this remaining instance feel semantically consistent
with the rest of the paper, or does it create a confusing echo with the
now-fixed passages?

### Q5 — Final submission verdict

Given all the changes across all rounds, does the paper §1–§8 now read as
internally consistent with the "online utility-ranked approximation of the
idealized batch problem" framing throughout?

Please provide:
- A one-line verdict: **ready to submit** / **1 micro-fix remaining** (name it)
- The single most important action remaining, if any
