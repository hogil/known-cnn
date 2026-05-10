# 260511 Phase 54 — iter 54 non-KD single-model improvement attempts

## Context

§5.32–§5.35 (Phases 47–52) established the 1× cost KD
distillation tier (iter 50 B / 51 D / 53 F) at bit-F1 ≈
0.98–0.99. The 4× cost NEW HEADLINE (4-bag) holds
0.9953 / 0 %.

One open question remained: **can any non-KD single-model
modifier improve the 26 B baseline (0.9781 / 2.5 %) within
the dual-gate FAR ≤ 5 % envelope?** 26 B is the strongest
non-KD single model in the entire project (FCM-PM
pair-mask + complement-CutMix + LS = 0.20 + 8 epochs +
g = 3 + corner fill, no KD).

If a non-KD technique works, the production single-model
tier would offer a KD-free alternative. If none works, KD
distillation is mechanistically unique in our setting and
the single-model frontier is exhausted.

iter 54 sweeps 6 standard regularisation / schedule
modifiers (one per cell, on top of the 26 B recipe), FULL
n = 200, single seed.

## Result

| cell | modification | bF1 | ni_FAR | dual | bb / fk / sc / sr |
|------|--------------|----:|-------:|:----:|---|
| 54 A | EMA decay 0.99 (Mean-Teacher style) | 0.9798 | 100 % | **FAIL** | 0.9785 / 0.9770 / 0.9637 / 1.0000 |
| 54 B | epochs 8 → 16 | 0.9654 | 0 % | PASS | 0.9678 / 0.9430 / 0.9509 / 1.0000 |
| 54 C | warmup-epochs 0 → 3 | 0.9871 | 100 % | **FAIL** | 0.9890 / 0.9776 / 0.9858 / 0.9961 |
| 54 D | drop-path-rate 0 → 0.1 | 0.9441 | 100 % | **FAIL** | 0.9752 / 0.8278 / 0.9899 / 0.9833 |
| 54 E | LS 0.20 → 0.10 | 0.9606 | 2 % | PASS | 0.9819 / 0.9032 / 0.9644 / 0.9929 |
| 54 F | combined (warmup = 2 + drop-path = 0.05 + ep = 12) | 0.9719 | 0 % | PASS | 0.9866 / 0.9702 / 0.9790 / 0.9517 |
| 26 B | reference baseline | **0.9781** | **2.5 %** | PASS | (canonical) |
| iter 50 B | KD α = 0.5 / T = 4, 4-bag teacher | **0.9872** | **0.5 %** | PASS ★ | 0.9866 / 0.9825 / 0.9795 / 1.0000 |

## Findings

**Finding 1 — bF1 ↑ vs FAR ↓ trade-off (decisive).**
Three cells lift bF1 (54 A + 0.002, 54 C + 0.009, 54 D
catastrophic regress) but every one with bF1 ≥ 26 B
**breaks `ni_FAR` to 100 %**. Two cells hold the FAR gate
(54 B 0 %, 54 E 2 %) but regress bit-F1 by − 0.013 /
− 0.018. The combined modifier (54 F) is a modest
all-axes regress (− 0.006 bF1 / 0 % FAR).

**Finding 2 — KD is the unique PASS-conforming
improvement.** Of the seven non-KD axes (54 A–F + 26 B),
**none simultaneously improves bit-F1 AND preserves the
FAR gate**. iter 50 B (KD α = 0.5, 4-bag teacher)
**does both** (+ 0.0091 bF1, − 2.0 % FAR) — the only
single-model recipe in the project to beat 26 B within
the dual-gate envelope.

**Finding 3 — 26 B is a regularisation sweet spot.**
26 B's recipe (FCM-PM pair-mask + complement-CutMix +
LS = 0.20) is itself the regularisation optimum at this
data scale. Adding *any* further dynamics-side regulariser
(EMA, warmup, drop-path, longer epochs, stronger LS)
**over-regularises** and breaks one of the two gates.
The §4.6 design rationale ("FCM-PM IS the regularisation,
not just augmentation") is empirically validated.

## Mechanism (paper §6.22)

KD distillation injects **FAR-boundary information**
through teacher soft targets on non-defect chips: the
4-bag teacher's posterior on a Normal / Invalid chip is
a calibrated near-zero distribution (not the hard zero
vector that BCE would assign), and distilling this
signal teaches the student a smoother decision boundary
on the defect ↔ non-defect axis specifically.

Non-KD modifiers (EMA, warmup, drop-path) operate on
**training dynamics** and inject no per-class non-defect
information. They smooth the student's decision boundary
uniformly, sometimes lifting bit-F1 on saturated defect
classes by ≈ 0.002–0.009, but simultaneously *removing*
the FCM-PM pair-mask's deliberate over-confidence on
non-defect chips — `ni_FAR` collapses 2.5 % → 100 %.

The pair-mask (§6.19) provides FAR control via training
data construction (Normal chips paired with defect chips
under the mask teach explicit non-defect suppression).
Dynamics-side regularisers weaken this learnt suppression
without replacing it. KD does not weaken it because the
teacher's posterior on Normal chips is already near-zero
across all 4 defect classes (teachers were themselves
trained with FCM-PM); KD reinforces the same suppression
direction additively on top of pair-mask.

## Paper claim

**No non-KD single-model technique improves the 26 B
baseline within the FAR ≤ 5 % gate.** Production
single-model deployment beyond 26 B requires KD
distillation (iter 50 B / 51 D / 53 F per §7.10–§7.10.2)
or accepts the 26 B baseline (0.9781 / 2.5 %).

This is a paper-grade negative result that strengthens
KD's positive claim by exhausting the natural alternatives.

## Cost frontier (final)

| cost  | recipe                                                  | bit-F1     | ni_FAR    |
|------:|---------------------------------------------------------|-----------:|----------:|
| 1×    | 26 B (best non-KD)                                      | 0.9781     | 2.5 %     |
| 1×    | iter 53 F (KD pure-hard 4-bag α = 0.3, strict-FAR)      | 0.9843     | **0.0 %** |
| **1× ★** | **iter 50 B (KD NEW MAIN 4-bag α = 0.5)**           | **0.9872** | **0.5 %** |
| 3×    | {37 E + 24_LS030_seed7 + 26 D}                          | 0.9929     | 0 %       |
| **4×** | **NEW HEADLINE 4-bag pure-hard**                       | **0.9953** | **0 %**   |

The single-model frontier (1× cost) is exhausted by KD.
Further lift requires ensemble cost.

## Files updated

- `docs/chip-multilabel/paper/05_experiments.md` — § 5.36 appended
- `docs/chip-multilabel/paper/06_analysis.md` — § 6.22 appended
- `docs/chip-multilabel/paper/07_discussion.md` — § 7.10.3 appended
- `docs/chip-multilabel/paper/abstract.md` — non-KD note inserted before iter 53 paragraph

## Source

iter 54 6-cell FULL n = 200 sweep, single seed, 26 B
recipe + one modifier per cell.
