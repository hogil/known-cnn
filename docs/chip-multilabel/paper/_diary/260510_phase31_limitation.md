# 2026-05-10 — Phase 31 methodological transparency disclosure

## What changed

Three sections of the paper now carry an explicit disclosure that the
training set (`classification_chips/`) and the evaluation set
(`chip_multilabel_v15direct/`) are **independently sampled but built
by separate scripts that share the same synthesis primitives**
(palette, alpha-modulation matched-filter mechanism, defect-type spec).

Files edited:

- `03_data.md` — new §3.10a "Train and evaluation are independently
  sampled from the same synthesis pipeline" between §3.9 (FAR
  decomposition) and §3.10 (v5.2 baseline reset). States the shared
  primitives, confirms no chip overlap (different RNG seeds 42 vs
  999, different generation modes), notes that combos and OOD
  wafer-canvas patterns are unseen during training, and explicitly
  disclaims real-factory deployment performance.

- `07_discussion.md` — new §7.6 "Limitations and scope of evaluation"
  with three subsections:
  - §7.6.1 same-pipeline concern (honest framing of the
    reviewer-vulnerable point).
  - §7.6.2 real-factory validation needed (separates the
    methodology contribution — independent of synth-data scale —
    from the headline absolute number).
  - §7.6.3 mitigations already in place (proactive framing): four
    OOD wafer-canvas patterns absorbed into `ni_FAR = 0 %`,
    24_LS030 single-model fragility (22.5–68 %) demonstrating
    non-memorisation, eval `min`-blend operator differing from any
    training operator. The original §7.6 / §7.7 / §7.8 (TTA
    rotation rule, current limits, iter narrative) renumbered to
    §7.7 / §7.8 / §7.9.

- `abstract.md` — appended a final paragraph "Methodological
  transparency" disclosing the shared synthesis pipeline, listing
  the OOD wafer-canvas patterns as the in-paper distribution-shift
  evidence, and pointing forward to §7.6 for the real-factory
  caveat. Headline number 0.9953 / 0 % unchanged.

## Why now

Reviewer-vulnerability concern: a paper that reports v15direct
n = 500 bit-F1 = 0.9953 / `ni_FAR = 0 %` on a controlled
synthetic benchmark **cannot** claim those numbers transfer to
real-fab deployment without explicit scope statement. Earlier
draft did not state the shared-synthesis-pipeline fact directly,
which is the kind of methodology-detail omission that turns a
clean technical paper into a defensive Q & A in review. Adding
the disclosure now (a) is honest, (b) reframes the paper's
contribution as **methodology + controlled-benchmark validation**
rather than absolute-deployment-number, and (c) elevates the
already-in-paper mitigations (OOD wafer-canvas, single-model
fragility, operator mismatch) from incidental observations to
structural design evidence.

## Framing choices

The §7.6.3 mitigations are framed as **proactive**, not
apologetic. The four OOD wafer-canvas patterns and the
24_LS030 ensemble-from-fragility example were both already
in-paper findings; surfacing them under the limitations
heading makes their distribution-shift role explicit rather
than buried in §6.17.2.

We do **not** modify the headline number. The 4-bag pure-hard
MAIN at v15direct n = 500 bit-F1 = 0.9953 / `ni_FAR = 0 %`
remains the paper's production headline. The §7.6 disclosure
clarifies its **scope** as a synth-benchmark ceiling, not its
**magnitude**.

## Source: user directive

User flagged that the paper currently does not state explicitly
that train and eval are independently sampled from the SAME
synthesis pipeline — a reviewer-vulnerable point. Total new
text under 600 words across all four files. Honest, transparent,
not over-disclaiming.
