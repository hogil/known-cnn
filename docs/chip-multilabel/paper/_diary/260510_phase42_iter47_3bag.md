# 2026-05-10 — Phase 42: iter 47 g=2 LS axis precision map + 3-bag re-evaluation at n=200

## Summary

Two findings closed today, both feeding the paper:

1. **iter 47 (six cells)** finishes the g = 2 LS axis precision map.
2. **3-bag re-evaluation at n = 200** surfaces a Pareto-better mid-cost
   production option.

Headline 0.9953 / 0 % (4-bag NEW HEADLINE at n = 500) is unchanged.

## 1. iter 47 — g = 2 LS axis precision mapping

Six new cells (47 A–F) on previously-untested LS values.

| cell | g = 2 LS | fill        | bF1    | ni_FAR    | dual    |
|------|---------:|-------------|-------:|----------:|---------|
| 47 A | 0.05     | corner      | 0.7988 |  0 %      | PASS (low) |
| 47 B | 0.10     | corner      | 0.7446 |  1 %      | PASS (low) |
| 47 C | 0.15     | corner      | 0.7221 |  0 %      | PASS (low) |
| 47 D | **0.25** | corner      | 0.9459 | **100 %** | **FAIL** |
| 47 E | 0.35     | corner      | 0.9125 | **100 %** | **FAIL** |
| 47 F | **0.50** | **white-fill** | **0.9795** | **5.00 %** | **PASS** |

Combined with iter 22 D / 24 / 30 D / 36 / 40 A, the **full g = 2 LS map**:

- **PASS**: 0.05, 0.10, 0.15, 0.20, 0.30, 0.50 (white-fill ONLY), 0.55, 0.80, 1.00.
- **FAIL**: 0.25, 0.35, 0.40, 0.45, 0.50 (corner-fill), 0.60, 0.65, 0.70, 0.90.

Two findings escalate to the paper:

### F1. Pair-fill alters the boundary at LS = 0.50

iter 30 D (corner-fill) FAILs at 100 % `ni_FAR`; iter 47 F (white-fill,
identical recipe otherwise) PASSes at 5 %. Same `(g, LS, seed)`,
opposite verdict. The PASS / FAIL boundary depends on
`(g, LS, seed, pair-fill)`. Pair-fill is therefore a fifth axis on the
FCM-PM hyperparameter surface, not a fixed implementation detail.

Mechanism (consistent with §6.19 pair-mask analysis): corner-fill leaks
class-correlated palette signal into the non-A region; at LS ≥ 0.25 the
leak amplifies and `ni_FAR` collapses. White-fill is class-neutral under
the chip palette and removes the leak.

### F2. The PASS region is fragmented, not continuous

47 D falsifies the earlier reading of "continuous PASS region 0.05–0.30":
LS = 0.25 corner-fill collapses 100 % `ni_FAR` while both immediate
neighbours (0.20 PASS, 0.30 PASS) clear the gate. The g = 2 LS axis is
**fragmented narrow basins separated by isolated FAIL points**.

Practitioner implication: hyperparameter interpolation along the LS axis
is not safe. Each candidate `(g, LS, pair-fill)` must be co-validated
against the dual gate. This refinement strengthens §6.17's ensemble-from-
fragility thesis: the fragility is **locally bimodal in LS**, not just
in seed.

## 2. 3-bag re-evaluation at n = 200 (paper canonical)

Five 3-bag candidates re-evaluated at the n = 200 robust eval.

| 3-bag                                          | bF1     | ni_FAR | per-class bb / fk / sc / sr  |
|------------------------------------------------|--------:|-------:|------------------------------|
| **37 E + 24_LS030_seed7 + 26 D**               | **0.9929** | **0 %** | 0.9873 / 0.9865 / 0.9992 / 0.9984 |
| 26 B + 24_LS030_seed7 + 26 D                   |  0.9921 |  1 %  | 0.9945 / 0.9841 / 0.9913 / 0.9984 |
| 26 B + 24_LS030_seed42 + 26 D                  |  0.9915 |  0 %  | 0.9921 / 0.9873 / 0.9905 / 0.9961 |
| 37 E + 24_LS030_seed42 + 26 D                  |  0.9907 |  0 %  | 0.9817 / 0.9881 / 0.9969 / 0.9961 |
| 26 B + 26 D + 26 H (3 pure-hard)               |  0.9884 |  0 %  | 0.9977 / 0.9865 / 0.9694 / 1.0000 |

**Cost frontier (n = 200 robust)**:

- 1× cost: 33 A KD-student → 0.9840 / 0 %
- **3× cost: 37 E + 24_LS030_seed7 + 26 D → 0.9929 / 0 %** (★ paper §7.8)
- 4× cost: NEW HEADLINE 24_LS030_seed42 + 26 B + 26 D + 26 H → 0.9953 / 0 %

The 3 × → 4 × delta is **0.0024** bit-F1 (≈ 5 chips out of ≈ 2 000
defect chips at n = 200). The 3-bag delivers **25 % cost reduction**
at a paper-indistinguishable bit-F1 penalty.

24_LS030_seed7 alone fails dual-gate; in the 3-bag it contributes
positively via majority-vote absorption — identical pattern to §6.17.2.
Ensemble-from-fragility holds at the 3-bag scale.

## Paper deltas

- §5.29 added (g = 2 LS axis precision mapping).
- §6.20 added (pair-fill-dependent boundary; fragmented basins).
- §7.8 added (3-bag production option, updated cost frontier).
- Abstract: 3-bag mention inserted before strength-curve paragraph.

Headline number 0.9953 / 0 % unchanged. The 3-bag is added as a
recommended production tier; 4-bag retained for SOTA.
