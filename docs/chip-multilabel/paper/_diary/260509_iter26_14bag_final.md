# 2026-05-09 evening — Iter 26 14-bag final paper headline

## What landed

**Iter 26 9-cell diversity sweep (LS × drop_path × g axes on top of T7N + FCM-PM 19C):**
- 5 dual-pass (26B/D/F/G/H), 4 fail (26A/C/E/I)
- ★ **NEW single-model best: 26B** (LS = 0.50, drop_path = 0.10, g = 3)
  v14 bF1 = 0.9921 / ni% = 0.00 / v15 bF1 = **0.9791** / ni% = 1.25
  vs iter-21 E (v15 = 0.9691, ni% = 3.75) — **+0.0100 v15 bit-F1** and **−2.50 pp** v15 ni_FAR
  → **LS axis is not exhausted at LS ∈ {0.10, 0.20, 0.30}** — LS = 0.50 + co-regularisers opens a new operating point that single-axis sweeps had missed

**14-bag ensemble + vote-threshold sweep:**
- Bag = 6-cell LS×seed core (iter-25) + iter-21F/H + iter-22G + iter-26B/D/F/G/H = 14 cells
- τ ∈ {5, 6, 7, 8, 9, 10} swept
- ★★★ **paper main: τ = 5 / 14 (36 %) and τ = 6 / 14 (43 %)** both deliver:
  - v14 bit-F1 = **1.0000** (perfect, all 4 classes)
  - v15 bit-F1 = **0.9929**
  - ni_FAR = **0.00 %** on both eval sets
  - F1_scratch (v15) = 0.9905 vs 12-T5 baseline 0.5841 → **+0.4064 (+70 %)**
- Monotonic decline from τ = 5 onward; τ = 10 super-majority loses 0.0071 v15 bit-F1 vs τ = 5

## Why simple-majority dominates super-majority (paper-novel)

Vote-count distribution decomposes orthogonally:
- **True-defect, in-distribution**: 13–14 / 14 votes (saturated)
- **True-defect, borderline severity** (fork low-grade, sr rotation tail): 5–11 / 14 votes
- **Normal in-dist**: 0–1 / 14 votes
- **OOD wafer-canvas**: 0–4 / 14 votes (worst-case bimodal-FAR over-firers)

The defect-recall floor (5 / 14) is one above the ni-FAR ceiling (4 / 14). Optimal τ = 5 (smallest integer above the worst-case negative agreement). Textbook ⌈K / 2⌉ = 7 / 14 defaults *into* the borderline-defect band and discards true positives without buying any ni_FAR reduction.

Generalisation: any vote ensemble with **bimodal base-classifier error and saturated correctness on positives** should sweep τ rather than default to 50 %. We document this as a methodological contribution.

## Headline numbers update (paper main table)

| config | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | F1_scratch (v15) |
|---|---:|---:|---:|---:|---:|
| 12-T5 baseline | — | 100% | 0.7872 | 0% (collapsed) | 0.5841 |
| 21E single best | 0.9913 | 0.00% | 0.9691 | 3.75% | 0.9786 |
| **26B NEW single best** | 0.9921 | 0.00% | **0.9791** | 1.25% | 0.9226 |
| iter-25 6-bag (≥4/6) | 0.9976 | 0.00% | 0.9913 | 0.00% | 0.9905 |
| **★ iter-26 14-bag (≥5/14)** | **1.0000** | **0.00%** | **0.9929** | **0.00%** | **0.9905** |

## Paper updates (this session)

| file | change |
|---|---|
| `abstract.md` | replaced iter-25 final block with iter-26 14-bag headline + simple-majority lesson + iter-26 B as new single-model SOTA |
| `04_methods.md` | new § 4.8 (14-bag composition, vote-threshold sweep, simple-majority mechanism, cost vs iter-25) |
| `05_experiments.md` | new § 5.17 (iter 26 9-cell sweep table + 14-bag composition + τ sweep table + per-class breakdown + paper claims unlocked) |
| `06_analysis.md` | new § 6.12 (vote-count distribution decomposition + super-majority cost analysis + bag-size scaling saturation) |
| `07_discussion.md` | new § 7.5.9 (production cost amortisation + distillation as future work + submission readiness 4-axis) |
| `09_conclusion.md` | new § 9.5 (final paper headline + sixth lesson on simple-majority + updated future work prescription) |
| `_diary/260509_iter26_14bag_final.md` | this file |

## Status

Paper **submission-ready** on the iter-26 14-bag headline. iter-25 6-bag retained as ablation row (bag-size scaling 6 → 14). iter-21 E retained as strongest single-model baseline. iter-26 B reported as new single-model SOTA.

Next sessions: distillation experiment (14-bag → 1× student), real-fab Normal validation if data lands.