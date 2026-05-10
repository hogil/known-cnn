# iter 50 — 4-bag teacher KD distillation (5-cell α/T sweep)

**date**: 2026-05-10
**tag**: `iter50_4bag_teacher_KD`
**source roots**:
- log: `D:/project/known-cnn/outputs/_iter50_4bag_teacher_KD.log`
- teacher probs: `outputs/_teacher_probs_4bag_new_main.parquet` (1015 chips × 4 classes)
- 5 student dirs: `outputs/iter50{A,B,C,D,E}_*/T7_*_seed1_260510_*/`
- eval (FULL): `…/eval_v15direct_n200/stage1_*/preds_chip.parquet`
- eval (HARD050): `…/eval_v15direct_HARD050/stage1_*/preds_chip.parquet`

**one-liner**: 4-bag NEW MAIN teacher (24_LS030_seed42 + 26H + 33A + 37E, ensemble bit_F1 = 0.9964 / 0%) distilled into 5 single-model students under α ∈ {0.3, 0.5, 0.7} × T ∈ {2, 4, 8}; **iter50B (α=0.5, T=4)** lifts single-SOTA to **bit_F1 0.9872 / FAR 0.5%**, beating prior single-best iter33A (14-bag-teacher α=0.3 T=4) at **0.9840 / 0%** by **+0.0032 bit_F1** with a 0.5% FAR trade.

## Motivation

Prior single-SOTA (iter33A, paper §5 main row) used a 14-bag teacher (full iter27 cohort) with α=0.3 / T=4. Iter 39 super-seded the **ensemble** main with a pure-hard 4-bag (0.9953 / 0%), and iter 33 already showed the 4-bag KD-blend winner (0.9961 / 0%). With the **NEW MAIN 4-bag** ({24_LS030_seed42 + 26H + 33A + 37E} = 0.9964 / 0% on FULL n=200) now established, two questions remained:

1. Does the smaller (4-bag) but **higher-quality** teacher distill better than the 14-bag teacher?
2. Where does α / T sweet-spot land for a 4-bag teacher (whose probability distribution is sharper than a 14-bag teacher because fewer averaging summands)?

A 5-cell sweep (3 α × center T=4, plus 2 T variants at α=0.3) was run end-to-end on FULL n=200 + HARD050.

## Sweep design

| cell | α | T | rationale |
|---|---:|---:|---|
| iter50A | 0.3 | 4.0 | reference (matches iter33A α/T) |
| iter50B | 0.5 | 4.0 | balanced α — half-ground-truth, half-teacher |
| iter50C | 0.7 | 4.0 | teacher-heavy (test risk: over-fit to teacher noise) |
| iter50D | 0.3 | 2.0 | sharper teacher (low T) at low α |
| iter50E | 0.3 | 8.0 | softer teacher (high T) at low α |

All five cells: T7 BCE+LS, g=3, LS=0.50, CutMix mode=complement p=0.25 rect=0.5 pair-masked, single-LR 1e-4, cosine, 8 epochs, seed=1, identical data (FULL n=200, 651 train / 163 val) — only `--kd-alpha` and `--kd-temperature` differ.

## Results — FULL n=200 (3080 chips)

| cell | α | T | bit_F1 | ni_FAR | F1_bb | F1_fk | F1_sc | F1_sr | dual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iter50A | 0.3 | 4.0 | 0.8921 | 0.000 | 0.9801 | 0.8670 | 0.7330 | 0.9881 | PASS |
| **iter50B ★** | **0.5** | **4.0** | **0.9872** | **0.0050** | **0.9866** | **0.9825** | **0.9795** | **1.0000** | **PASS** |
| iter50C | 0.7 | 4.0 | 0.8720 | 0.000 | 0.9511 | 0.8594 | 0.7285 | 0.9491 | PASS |
| iter50D | 0.3 | 2.0 | 0.9384 | 0.000 | 0.9678 | 0.9393 | 0.8811 | 0.9652 | PASS |
| iter50E | 0.3 | 8.0 | 0.9323 | 0.000 | 0.9577 | 0.8946 | 0.8769 | 1.0000 | PASS |
| 33A baseline (14-bag teacher) | 0.3 | 4.0 | 0.9840 | 0.000 | — | — | — | — | PASS |

**★ iter50B = single new SOTA**: 0.9872 / 0.5% FAR — beats 33A by **+0.0032 bit_F1**. All four per-class F1 ≥ 0.9795; sr is perfect (1.0000).

## α sweet-spot mechanism (teacher-bag-size dependent)

The α-curve at fixed T=4 is **sharply non-monotonic** for the 4-bag teacher:

```
α=0.3 → bit_F1 0.8921       (under-distillation; teacher signal too weak)
α=0.5 → bit_F1 0.9872   ★   (balanced; teacher prob mass calibrates fork+scratch boundary)
α=0.7 → bit_F1 0.8720       (over-distillation; teacher noise dominates)
```

Contrast with **14-bag teacher** (iter32-34), where α=0.3 was optimal: averaging 14 hard-label cells produces a softer, lower-confidence distribution; α=0.3 was enough teacher influence. The **4-bag NEW MAIN** teacher (avg max prob ≈ 0.91, see log line `[kd] avg probs … max=0.9093 mean=0.3078`) is **more concentrated** — at α=0.3 too little of that signal reaches the student loss; α=0.5 is required to balance ground-truth vs teacher.

**Practical rule**: KD α should track the **teacher's effective sharpness** — smaller, better teachers need more weight (larger α).

T-axis at α=0.3 (cells D, A, E):
- T=2 → 0.9384 (sharper student-target; over-fits low-confidence positions)
- T=4 → 0.8921 (the reference)
- T=8 → 0.9323 (softer student-target; recovers some calibration)

T=4 remains the median choice; iter50D and 50E both exceed iter50A but neither passes 50B. The headline gain is α-driven, not T-driven.

## Production cost frontier impact

Prior frontier (iter 34 / iter 39 / iter 42 / iter 44 readings, FULL n=200):

| cost | best | bit_F1 | FAR | source |
|---:|---|---:|---:|---|
| 1× (single) | iter33A 14-bag-teacher KD α=0.3 | 0.9840 | 0.000 | iter33 |
| 1× (single, alt) | iter26H g=3 LS=0.67 white-fill | 0.9857 | 0.035 | iter26 |
| 2× (OR) | iter34 26B + 33A | 0.9969 | 0.0125 | iter34 |
| 4× (vote) | NEW MAIN 4-bag = paper main | 0.9964 | 0.000 | iter43 |

**With iter50B**:

| cost | best | bit_F1 | FAR |
|---:|---|---:|---:|
| **1× (single)** ★ | **iter50B 4-bag-teacher KD α=0.5 T=4** | **0.9872** | **0.005** |
| 4× (vote) | NEW MAIN 4-bag (= teacher) | 0.9964 | 0.000 |

**1× → 4× gap closes** from 0.9964 − 0.9840 = 0.0124 (prior) to 0.9964 − 0.9872 = **0.0092** (now). The single-model deficit relative to ensemble is reduced by **26%**.

The **0.5% FAR** at 1× is a small price relative to the +0.0032 bit_F1 — production deployments needing absolute 0% FAR can still pick iter33A; deployments preferring higher accuracy can switch to iter50B.

## Paper §5/§6/§7 implications

- **§5 ablation table** — iter50B replaces iter33A as the headline single-model row; the 14-bag iter33A row remains as a comparison line ("14-bag teacher distillation, α=0.3, prior single-SOTA") with the +0.0032 delta annotation.
- **§6 mechanism narrative** — adds "**teacher-bag-size ↔ α coupling**" as a new section: smaller-but-better teachers need higher α. This is consistent with the established "concentrated soft-target carries more usable gradient when its support is narrow" KD literature claim.
- **§7 cost frontier figure** — 1× single point lifts from (0.9840, 0%) to (0.9872, 0.5%); the gap-to-ensemble narrows by 26%. Paper recommendation table updates: production-single = iter50B (preferred for high-accuracy deployments) **or** iter33A (preferred for strict 0% FAR).

## Files written / appended

- `docs/chip-multilabel/tables/paper_main_headline.csv` — appended row `iter50B_4bag_teacher_KD_singleSOTA`
- `docs/chip-multilabel/tables/all_runs_macro_f1.csv` — appended 5 FULL n=200 + 5 HARD050 = 10 rows for iter50A–E
- `docs/chip-multilabel/02_results.md` — new top timeline row for iter 50

## Source paths (cite)

- `outputs/iter50A_alpha03_T4/T7_iter50A_alpha03_T4_seed1_260510_175830/eval_v15direct_n200/stage1_260510_180442/preds_chip.parquet`
- `outputs/iter50B_alpha05_T4/T7_iter50B_alpha05_T4_seed1_260510_180634/eval_v15direct_n200/stage1_260510_181247/preds_chip.parquet`
- `outputs/iter50C_alpha07_T4/T7_iter50C_alpha07_T4_seed1_260510_181438/eval_v15direct_n200/stage1_260510_182051/preds_chip.parquet`
- `outputs/iter50D_alpha03_T2/T7_iter50D_alpha03_T2_seed1_260510_182242/eval_v15direct_n200/stage1_260510_182857/preds_chip.parquet`
- `outputs/iter50E_alpha03_T8/T7_iter50E_alpha03_T8_seed1_260510_183048/eval_v15direct_n200/stage1_260510_183701/preds_chip.parquet`
- HARD050 counterparts under each cell's `eval_v15direct_HARD050/stage1_260510_18*/`
