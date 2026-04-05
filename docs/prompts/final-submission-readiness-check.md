# Final Submission Readiness Check — Evaluation Prompt

You are acting as a final-pass reviewer for a systems conference paper (SC 2026)
titled "BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM
Serving". Multiple revision rounds have been completed. This is the last check
before submission. All structural and terminological issues from prior rounds
are resolved. The only questions here are about residual correctness,
consistency, and whether any sentence-level issues remain.

---

## Summary of all revision rounds completed

| Round | Primary changes |
|---|---|
| §4 audit | §4.3 terminology cleanup, Mode A/B removal, figure caption |
| §5 structural | Victim-selection framing, (a)–(e) steps, co-varying note |
| §5 final | Step (b) named as BidKV core differentiator; Table 1 LIFO* |
| §4.3 + Alg 1 rewrite | Knapsack → ranking semantics; "Constrained Solver Layer" → "Online Utility-Ranked Victim Selection" |
| 12-change global cleanup | Abstract, Intro, §2.2, §3, §4 overview, §4.1, §5.1, §8 |
| 3 follow-up fixes | "batch-level" → "cross-request"; §3 knapsack paragraph rewritten |
| §6–§7 polish | 7 fixes: greedy solver, utility-ratio (×5), solver's degree, how often |
| Final 2-fix round | §8 "how often" removed; §7.3 "solver extensions" → "scheduling extensions" |

---

## Current state of the complete paper (post all revisions)

### Abstract

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

### §7.1 Surrogate Disruption Estimate (final version)

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

### §7.3 Broader Impact (final version)

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

### §8 Conclusion (final version, key sentences)

```
BidKV surfaces per-request reclamation sensitivity as an explicit, structured
signal and performs online utility-ranked cross-request victim selection,
replacing implicit ordering heuristics with a utility-guided decision interface
(U = r / (δ + ε)). Under recompute-fallback semantics the policy preserves
output correctness; the differentiation lies in which request is selected for
reclamation at each pressure event---a choice that directly governs how quickly
queued requests receive KV allocation and, consequently, first-token latency
and SLO attainment.

Through a five-strategy evaluation on vLLM with Llama-3.1-8B-Instruct under
mixed-length ShareGPT workloads, we showed that BidKV achieves the best
cross-rate SLO attainment (87.1%) and TTFT P95 (562 ms), outperforming both
capacity-greedy heuristics and default scheduling baselines at a modest
throughput cost (~7%).
```

---

## Already resolved — do NOT re-raise

| Issue | Round resolved |
|---|---|
| §4.3 knapsack/Σδ/Δmax/Σr≥N constraints | §4.3 rewrite |
| Algorithm 1 greedy acceptance-set semantics | §4.3 rewrite |
| "Constrained Solver Layer" title | §4.3 rewrite |
| "minimum aggregate cost" / "coordinated solver" in Abstract | 12-change global cleanup |
| "batch-level trade-offs" in Intro | Follow-up fixes |
| §3 knapsack paragraph implying constrained solver | Follow-up fixes |
| §5.1 step (e) semantic gap with LIFO | 12-change global cleanup |
| §6.1 "utility-ratio greedy solver" | §6–§7 polish |
| "coordinated utility-ratio selection" (×2) | §6–§7 polish |
| "utility-ratio selection / victim selection" (×3) | §6–§7 polish |
| §7.1 "solver's degree of freedom / how often" | §6–§7 polish |
| §7.3 "solver extensions" | Final 2-fix round |
| §8 "how often" | Final 2-fix round |

---

## Your evaluation task

This is a final submission readiness check. For each question, rate severity:
**critical / moderate / minor / none**.

### Q1 — Abstract: "recoverable capacity and estimated disruption cost" vs "utility" framing

The Abstract describes bids as "encoding recoverable capacity and estimated
disruption cost" and then says "an online utility-ranked selection policy
identifies the intended highest-utility victim." The utility formula
U = r / (δ + ε) is not mentioned in the Abstract.

Is the Abstract's description of the bid and the utility framing sufficiently
consistent with §4?  Or does not mentioning U = r/(δ+ε) in the Abstract make
the connection between "recoverable capacity / disruption cost" and
"highest-utility victim" opaque?

### Q2 — §7.1 "the policy's degree of freedom" phrasing

The current sentence reads: "the policy's degree of freedom is which request
is reclaimed at each pressure event."

Is "degree of freedom" a natural phrasing in a systems paper, or does it
sound more like optimization/control-theory jargon that a systems reviewer
might find awkward? A simpler alternative: "the policy governs which request
is reclaimed at each pressure event."

### Q3 — §8 "a choice that directly governs how quickly": causal strength

The revised §8 sentence reads: "the differentiation lies in which request is
selected for reclamation at each pressure event---a choice that directly
governs how quickly queued requests receive KV allocation and, consequently,
first-token latency and SLO attainment."

Is "directly governs" too strong? The causal chain is:
victim selection → less wasted recompute → more available KV → shorter queue
wait → lower TTFT. This is a 3-hop chain, not a direct effect. Should
"directly governs" be softened to "directly shapes" or "has a direct effect on"?

### Q4 — §7.3 remaining "how often" in the same paragraph

Note that §7.3 still contains: "Per-request preemption caps---limiting **how
often** any single request can be reclaimed within a scheduling window."

This "how often" refers to *per-request preemption frequency*, not to the
ranking policy's scope. Is this a different, defensible use of "how often" that
should be retained? Or does its presence in the same section as the recently
fixed "how often" create a confusing echo?

### Q5 — Overall submission readiness

After all revision rounds, is the paper:
- **ready to submit** (no further wording changes needed)
- **1–2 final micro-touches** (specific sentence-level changes, name them)
- **needs structural attention** (explain what)

Please give a direct verdict and name the single most important remaining
action, if any.
