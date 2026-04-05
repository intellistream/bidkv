# Post-Terminology-Sweep Final Confirmation — Evaluation Prompt

You are performing a narrow spot-check on a systems conference paper (SC 2026)
titled "BidKV: Utility-Guided Preemption Scheduling for KV-Pressure LLM
Serving" after a final terminology sweep. All prior structural and naming
issues are resolved. This check confirms no new seams were introduced and
gives a submission verdict.

---

## What was changed in the final terminology sweep (do NOT re-raise)

| Change | Location |
|---|---|
| `The solver selects the victim (④)` → `The selection layer identifies the top-ranked victim (④)` | fig1_architecture.tex caption |
| `utility-ratio victim selection) on SGLang` → `utility-ranked victim selection) on SGLang` | §5.2 SGLang Integration |
| `replacing the utility-ratio solver)` → `replacing the utility-ranked ordering)` | §5.2 SGLang Integration |
| `its utility-ratio scoring already accounts for` → `BidKV's utility-ranked selection already accounts for` | §6.2 Main Comparison, SRPT bullet |
| `The scorer--solver separation enables` → `The scorer--selection separation enables` | §8 Conclusion, final ¶ |
| `fairness-aware solver extensions` → `fairness-aware scheduling extensions` | §8 Conclusion, final ¶ |

All previously resolved issues (knapsack, Σδ/Δmax, greedy solver label, §4.3
title, §6–§7 utility-ratio, §7.1/§7.3/§8 wording) remain unchanged.

---

## Current verbatim text of changed passages

### Figure 1 caption (step ④–⑤ only)

```
ranks candidates by utility U = r/(δ + ε) (④). The selection layer
identifies the top-ranked victim (⑤); the framework adapter preempts
the chosen request, freeing its KV blocks (⑥). The preempted request
re-enters the waiting queue for later recomputation (⑦).
```

### §5.2 SGLang Integration (relevant sentences, after fix)

```
To test portability, we deploy BidKV's complete scheduling policy
(SJF admission ordering, pressure-gated running-queue reorder, and
utility-ranked victim selection) on SGLang without modifying SGLang's
internals, and compare against Vanilla SGLang (SGLang's native LRU-based
eviction with no managed scheduling) and Random-Evict (BidKV's adapter
with random victim selection replacing the utility-ranked ordering).
```

### §6.2 SRPT bullet (after fix)

```
BidKV explicitly disables SRPT: BidKV's utility-ranked selection already
accounts for estimated reclamation cost via δ, avoiding the cascading
preemptions that aggressive reclamation can trigger.
```

### §8 Conclusion final paragraph (after fix)

```
BidKV integrates with both vLLM and SGLang via a portable adapter
abstraction, reusing each framework's native reclamation and recovery paths
without source-code modification. The scorer--selection separation enables
independent evolution of scoring strategies and selection algorithms. Future
work includes token-level truncation for partial KV release, fairness-aware
scheduling extensions, and multi-GPU coordination.
```

---

## Remaining "solver" instances in the paper (acceptable — do NOT flag)

The following occurrences of "solver" remain in the paper and are intentional:

| Line context | Reason acceptable |
|---|---|
| §3: "offline solver is available" | Describes knapsack literature assumption, not BidKV |
| §4 overview: "scoring signals flow to the solver" | Generic architecture description; code class ref |
| §4.1: "By preferring high-utility victims, the solver reduces..." | Code-class reference (GreedyBidSolver); valid |
| §4.1: "The solver output is a utility-ranked ordering" | Explicitly correct description of output |
| §4.3 body: "the solver requires only that the relative ordering be preserved" | Correct semantics |
| §4.3 subsection label: `\label{sec:design:solver}` | Internal LaTeX label, not visible text |
| §4.4: "invokes BidKV's solver first" | Implementation description, code class |

---

## Your evaluation task

### Q1 — Figure 1 caption: "The selection layer identifies the top-ranked victim"

The new caption now reads "the selection layer identifies the top-ranked victim
(④)". Is "identifies" the right verb — does it accurately convey that the
selection layer **produces a utility-ranked ordering** and the top-ranked entry
is what gets consumed? Or is "identifies" too strong (implying a discrete
lookup) and should it be "selects" (which was the old word) or "ranks and
surfaces"?

Note: §4.3 says "at each pressure event, the runtime adapter consumes the
current top-ranked entry." The selection layer's job is to produce and maintain
the ordering; the adapter consumes it. Does "The selection layer identifies the
top-ranked victim" correctly split these responsibilities, or does it conflate
the ranking step with the consumption step?

### Q2 — §6.2 SRPT bullet: "BidKV's utility-ranked selection already accounts for..."

The new sentence reads: "BidKV's utility-ranked selection already accounts for
estimated reclamation cost via δ."

The original sentence was: "its utility-ratio scoring already accounts for..."
The key claim is that the **scoring formula** (δ in the denominator) captures
reclamation cost, which is why SRPT is redundant. By changing "utility-ratio
scoring" to "utility-ranked selection", has the precision of this claim been
reduced? "Scoring" referred specifically to the formula; "selection" refers to
the whole victim-picking process. Does this matter for the argument being made?

### Q3 — §8 "scorer--selection separation": is this a recognized architectural pattern name?

The new phrase "scorer--selection separation" replaces "scorer--solver
separation". Is "scorer--selection" a natural architectural description that a
systems reviewer would understand, or does it sound awkward? Compare with
alternatives:
- "scoring--selection separation" (noun form)
- "scoring-to-selection architecture" (already used in §1 Contributions #2)
- "bid-based scoring and selection separation"

Should this be aligned with the Contributions #2 phrasing ("decoupled
scoring-to-selection architecture") for consistency?

### Q4 — Overall: any "utility-ratio" or "solver" legacy wording still visible?

After this sweep, do any of the following phrases appear in the paper in a
context where they misrepresent BidKV's current "online utility-ranked
approximation" design?

- utility-ratio [as a canonical naming term, not a formula description]
- greedy solver [as a mechanism description, not a class name]
- acceptance set / BidAcceptance
- constrained solver
- minimum aggregate cost [as a primary claim]

### Q5 — Final submission verdict

Given all revision rounds, does the paper §1–§8 + Figure 1 caption now read
as fully internally consistent with the "online utility-ranked approximation of
the idealized batch problem" framing?

Verdict options:
- **ready to submit** — no further changes needed
- **1 micro-fix** — name it precisely (one sentence)
- **needs review** — explain what

Please give a direct one-line verdict.
