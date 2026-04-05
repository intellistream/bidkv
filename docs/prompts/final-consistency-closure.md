# Final Consistency Closure — Evaluation Prompt

You are performing the absolute final consistency check on a systems conference
paper (SC 2026) titled "BidKV: Utility-Guided Preemption Scheduling for
KV-Pressure LLM Serving" after one must-fix and four optional harmonization
edits. The paper's framing, contributions, experiments, and structure are all
frozen. This prompt checks only whether the latest edits introduced any seams
and whether the paper now reads as fully consistent.

---

## What was changed in this round (do NOT re-raise)

### Must-fix (1)
| Change | Location |
|---|---|
| `utility-ratio victim selection respectively` → `utility-ranked victim selection respectively` | Table 4 caption (table4_sglang.tex) |

### Optional harmonization (4)
| Change | Location |
|---|---|
| `scoring signals flow to the solver` → `scoring signals flow to the selection layer` | §4 Design opening paragraph |
| `By preferring high-utility victims, the solver reduces aggregate reclamation cost` → `the selection policy reduces aggregate reclamation cost` | §4.1 following Eq. (2) |
| `The solver output is a utility-ranked ordering of active bids` → `The selection output is a utility-ranked ordering of active bids` | §4.1 BidPool paragraph |
| `the solver requires only that the relative ordering be preserved` → `the ranking procedure requires only that the relative ordering be preserved` | §4.2 Bid Generation tail sentence |

### Intentionally kept (do NOT flag)
| Instance | Reason |
|---|---|
| `GreedyBidSolver computes...` (§4.3) | Code class name; implementation description |
| `invokes BidKV's solver first` (§4.4) | Implementation description |
| `offline solver is available` (§3) | Describes knapsack literature, not BidKV |

---

## Current verbatim text of changed passages

### Table 4 caption (relevant sentence)
```
Vanilla SGLang applies no managed victim selection;
Random-Evict and BidKV deploy BidKV's adapter with random and
utility-ranked victim selection respectively.
```

### §4 Design opening paragraph (relevant sentence)
```
The architecture prescribes how scoring signals flow to the selection layer;
the instantiation determines which features populate δ in this paper.
```

### §4.1 following Eq. (2) (two changed sentences)
```
A high utility ratio means preempting this request frees many tokens at low
reclamation cost.  By preferring high-utility victims, the selection policy
reduces aggregate reclamation cost, keeping KV capacity available for queued
requests and thereby improving admission responsiveness (TTFT and SLO attainment).

Bids are collected in a BidPool—an immutable snapshot of all active bids at a
given instant.  The selection output is a utility-ranked ordering of active
bids; the runtime adapter consumes the current top-ranked entry at each
pressure event.
```

### §4.2 Bid Generation tail sentence (changed sentence)
```
Because δ functions purely as a ranking signal, the ranking procedure requires
only that the relative ordering be preserved; exact magnitudes need not be
calibrated to absolute costs (Section 7.1).
```

---

## Your evaluation task

### Q1 — §4 "selection layer" vs. §4.3 subsection title

§4 Design opening now says "scoring signals flow to the selection layer." §4.3
is titled "Online Utility-Ranked Victim Selection" (not "Selection Layer"). Is
"selection layer" in the opening paragraph consistent with §4.3's title, or
does it leave a loose end where a reader might ask "which layer is the
selection layer — is it §4.3?"

Note: the four-layer architecture is formally named: Runtime Adapter Layer,
Bid Generation Layer, and then §4.3 "Online Utility-Ranked Victim Selection"
(which is effectively the selection layer). Does this implicitly define the
term?

### Q2 — §4.1 "the selection policy reduces" — subject consistency

The surrounding sentences use "By preferring high-utility victims" as the
subject lead-in. "the selection policy reduces" now replaces "the solver
reduces". Does "selection policy" read correctly as a noun functioning as the
subject here, or is there a subject-echo issue (" By preferring high-utility
victims, the selection policy reduces...")?

### Q3 — §4.2 "the ranking procedure requires only"

The sentence now reads: "Because δ functions purely as a ranking signal, the
ranking procedure requires only that the relative ordering be preserved."

§4.3 immediately follows and says: "Because δ serves as an ordinal ranking
signal, the selection requires only that relative ordering be preserved
across bids."

These two sentences are near-duplicates with slightly different wording. Is
this a problem (redundancy between §4.2 and §4.3), or does the first
occurrence serve as a forward-pointing summary and the second as a restatement
in context?

### Q4 — Table 4 caption: "utility-ranked victim selection" — is it now
consistent with §5.2 and §6.5?

§5.2 SGLang Integration already uses "utility-ranked victim selection" for the
BidKV strategy. §6.5 SGLang portability results also use "utility-ranked
victim selection". Does Table 4 caption now align with both?

### Q5 — Final submission verdict

After all rounds of revision (§4.3 Algorithm 1 rewrite, §1–§8 narrative
cleanup, §6–§7 polish, terminology sweep, two micro-fix rounds, and this
closure round), does the paper §1–§8 + all tables and figure captions read as
fully terminology-consistent with the "online utility-ranked approximation of
the idealized batch problem" framing?

Remaining "solver" instances (GreedyBidSolver class name in §4.3, "invokes
BidKV's solver first" in §4.4, "offline solver" in §3) are intentionally
preserved as implementation/literature references. Do these cause any
terminology inconsistency at the paper-level framing?

Verdict options:
- **ready to submit** — no further changes needed
- **1 micro-fix** — name it precisely (one sentence)
- **needs review** — explain what

Please give a direct one-line verdict.
