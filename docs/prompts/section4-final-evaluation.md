# Section 4 Final Evaluation Prompt

You are acting as a reviewer for a systems conference paper (SC 2026). Please evaluate Section 4 (Design) of a paper titled "BidKV: Utility-Guided KV Cache Reclamation for Admission-Responsive LLM Serving". Your task is a **final pass** — the section has already gone through several rounds of revision. Please identify only remaining issues that genuinely matter for a careful systems conference reviewer.

---

## Paper context (please read carefully before evaluating §4)

- **Core problem**: victim selection under KV cache pressure in LLM serving. When KV capacity is insufficient, which running request should be preempted?
- **Value proposition**: improved *admission responsiveness* — queued requests receive their first token faster, measured by TTFT P95 and SLO attainment (300 ms threshold). Throughput and TPOT are acknowledged tradeoff dimensions.
- **Execution model**: recompute-fallback — the preempted request is fully evicted and recomputed later. BidKV does not modify final model outputs; all differentiation is in victim selection.
- **Main claim**: BidKV reduces P95 TTFT by 89% and improves 300 ms SLO attainment by 14.9 pp over vLLM's native LIFO policy, at a modest throughput cost (~7%).
- **Section 2** frames the problem as: aggregate reclamation cost minimization under KV pressure.
- **Section 3** positions BidKV as: explicit per-request reclamation-cost signals + coordinated cross-request victim selection.
- **Sections 7–8** conclude that δ is a ranking signal; reclaim-frequency and victim-choice jointly determine admission latency (TTFT / SLO).

---

## Section 4 current text (verbatim, simplified LaTeX)

```latex
\section{Design}

The victim-selection problem identified in Section 2 imposes two requirements:
explicit reclamation-sensitivity signals and cross-request coordination.
Two additional design goals guide our architecture:
  scorer-agnostic interface — because the best importance signal may vary across
  workloads and models, the scoring strategy should be interchangeable without
  modifying the selection algorithm or the runtime integration; and
  framework portability — the mechanism should integrate with heterogeneous
  serving engines (e.g. vLLM, SGLang) by reusing their native reclamation and
  recovery paths, without requiring source-code modifications.

These goals separate naturally into a general architecture — the bid-based data
path and layered decomposition that are execution-model agnostic — and a current
instantiated path — the specific disruption estimator (δ, Eq. 3) that uses
request-lifecycle features under recompute-fallback semantics.  The architecture
prescribes how scoring signals flow to the solver; the instantiation determines
which features populate δ in this paper.

BidKV organizes its logic into four layers (Figure 2):
  - Runtime Adapter Layer: detects KV pressure and executes reclamations via the
    framework's native mechanism
  - Bid Generation Layer: computes disruption cost from request-lifecycle features
    and produces structured bids
  - Constrained Solver Layer: selects reclamation victims
  - Bid Signal Layer: defines the shared data structures used across all layers

% ---- §4.1 Bid Signal Layer ----

The atomic unit of communication is the scheduling bid, a frozen (immutable)
data class with the following fields:
  - bid_id: unique identifier.
  - request_id: the inference request this bid applies to.
  - r (tokens_freed): the number of KV tokens freed if this request is preempted.
  - δ (disruption estimate): a surrogate scheduling cost (≥ 0) estimating the
    system-level disruption of reclaiming this request. Lower values indicate
    requests whose preemption incurs less scheduling disruption. Under recompute
    fallback (this work), δ is derived from request-level features (completion
    progress, prompt length, and preemption history; Eq. 3); other request-level
    scorers can populate δ without architectural change.
  - confidence ∈ [0,1]: the scorer's confidence in δ.
  - metadata: strategy-specific information.

Each bid carries a derived utility property:
  U(b) = b.r / (b.δ + ε),   ε = 10^{-3}

A high utility ratio means preempting this request frees many tokens at low
reclamation cost.  By preferring high-utility victims, the solver reduces
aggregate reclamation cost, keeping KV capacity available for queued requests
and thereby improving admission responsiveness (TTFT and SLO attainment).

Bids are collected in a BidPool — an immutable snapshot of all active bids at a
given instant.  The solver output is a BidAcceptance containing the accepted bid
IDs, total freed tokens, and cumulative cost.

% ---- §4.2 Bid Generation Layer ----

The bid generation layer translates per-request state into a structured bid.
The choice of features that populate δ is an instantiation decision; this paper
uses request-lifecycle proxies under recompute-fallback semantics, as detailed
below.  Each running request contributes one bid with r equal to its current KV
footprint and δ computed as:
  δ = max(δ_min,  r_hat  +  w_c · c²  +  w_s · P)
where r_hat = max(β, n_prompt/n_0) (prompt-normalized recompute cost, floored
at β), c = min(1, n_output/n_max) is the completion ratio, and P counts prior
preemptions.

All three terms share the same unit: recompute cost of an n_0=256-token prompt.
  - w_c = 2.0: a fully completed request is as disruptive to evict as
    re-prefilling a 512-token prompt.
  - w_s = 0.5: half a baseline recompute cost per prior preemption.
  - β = 0.5: even trivially short prompts carry non-trivial reclamation cost.
  - Completion term is quadratic: evicting at 90% wastes 4× more accumulated
    decode work than at 45%.

Because δ functions purely as a ranking signal, the solver requires only that
the relative ordering be preserved; exact magnitudes need not be calibrated to
absolute costs.

% ---- §4.3 Constrained Solver Layer ----

When the pressure detector signals that N tokens must be freed, the
GreedyBidSolver selects bids from the pool (Algorithm 1).

The solver enforces two constraints:
  At-most-one bid per request: prevents double-counting freed tokens.
  Disruption budget: the cumulative δ must not exceed Δ_max, controlling
    per-event reclamation aggressiveness.  Under the current recompute-fallback
    path, the adapter invokes the solver with a relaxed budget to produce a
    complete utility ordering of all running candidates, then caches this ranking.
    When proactive preemption fires (KV > 90%), the adapter selects the
    top-ranked entry from the cache and executes one native preemption per event.

The algorithm runs in O(B log B) time.  In practice B < 100, so solving takes
microseconds — negligible compared with a decode step.

The victim selection problem is a variant of the 0-1 knapsack with a conflict
graph and a budget constraint.  While the general problem is NP-hard, the greedy
approach achieves strong empirical performance because the utility ratio provides
an effective ranking signal and the number of items is small.

% ---- §4.4 Runtime Adapter Layer ----

To achieve framework portability, BidKV defines a FrameworkAdapter ABC with
five responsibilities:
  1. KV stats visibility: returns current and maximum token counts for
     pressure detection.
  2. Pressure-event interception: intercepts the framework's native preemption
     path and invokes BidKV's solver first.
  3. Reclamation execution: delegates to the framework's native preemption
     mechanism.
  4. Scoring callback: provides decode-step signals (e.g. newly generated
     tokens) to the scoring strategy for priority updates.
  5. Lifecycle management: cleans up bids and state when a request completes.

This contract is intentionally minimal: any serving framework exposing KV
statistics and a reclamation hook can implement the adapter.
```

---

## What has already been revised — do NOT re-raise these

| Issue | Resolution |
|---|---|
| `quality_delta` / `quality budget` terminology | Replaced everywhere with `disruption estimate` / `disruption budget` |
| `Mode A` / `Mode B` (deprecated internal terms) | Removed; replaced with "recompute-fallback path" |
| `aggregate recomputation burden` | Changed to `aggregate reclamation cost` |
| §4.3 Δ_max semantics (was: "conservative budget → 1 victim") | Corrected: relaxed budget → complete ordering → cache → top-1 per event |
| Figure 2 caption numbering ①②③④⑥⑥ (skip ⑤, repeat ⑥) | Fixed to sequential ①②③④⑤⑥ |
| Scoring example "attention-based or gradient-based" (too forward-looking) | Changed to "other request-level scorers" |
| §4.2 instantiation framing missing | Added "choice of features that populate δ is an instantiation decision" |
| Bridge sentence from utility formula to admission responsiveness | Added sentence connecting high-utility victims → aggregate reclamation cost → KV available → TTFT / SLO |

---

## Your evaluation task

For each question below, please:
- Rate severity: **critical** (blocks understanding) / **moderate** (may confuse reviewers) / **minor** (polish only) / **none** (current version is fine)
- Suggest a specific fix if applicable
- If severity is "none", just say so briefly

### Q1 — Terminology consistency
Are there any remaining uses of deprecated or ambiguous terms — e.g. quality, mode A/B, compression (when preemption is meant), scheduling efficiency — that an SC reviewer would flag?

### Q2 — Architecture vs. instantiation boundary
Is the separation between "general architecture" and "current-paper instantiation" sufficiently clear? Does the opening paragraph successfully frame this distinction, or could a reader still mistake §4 as describing a single fixed design?

### Q3 — Narrative alignment with admission responsiveness
Does §4 consistently serve the "admission responsiveness under KV pressure" framing introduced in §1–§3? Identify any paragraph that drifts toward generic scheduling efficiency or output-quality talk.

### Q4 — §4.3 solver / cache / preemption flow
Is the explanation "relaxed budget → complete utility ordering → cache ranking → top-1 preemption per event" clear and logically coherent to a reviewer who has not seen the code? Is the relationship between the Disruption Budget constraint and the relaxed-budget operational mode confusing or acceptable?

### Q5 — Δ_max tension
The disruption budget Δ_max is defined as a hard constraint in the algorithm, but then §4.3 prose says the adapter uses a "relaxed budget" producing a complete ordering. Does this create a contradiction in the reader's mind? Is there a cleaner way to resolve the tension, or is the current explanation sufficient?

### Q6 — §4.4 Runtime Adapter Layer sufficiency
The five-point list is intentionally minimal. Does it leave reviewers with unresolved questions about how the adapter actually intercepts scheduling (especially for vLLM's internal scheduler)? Or is deferring details to §5 (Implementation) acceptable?

### Q7 — Overall readiness
On a scale of:
- **needs another round of revision** (structural or factual issues remain)
- **minor polish only** (small wording tweaks, no structural changes needed)
- **ready to submit** (this section is done)

Where does Section 4 stand? Please justify briefly.
