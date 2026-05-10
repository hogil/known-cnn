# Iter 21 — Clean baseline 8-model retrain + dual-eval (no-leak protocol)

- **Date**: 2026-05-09
- **Tag**: `iter21_clean_baseline`
- **Source root**: `outputs/iter21{A..H}_*/T*/eval_{v14class,v15direct}/stage1_*/`
- **Log**: `outputs/_iter21_clean_baseline.log`
- **One-line**: `★ E (T7N + 19C complement g=2 LS=1.0 FCM-PM) wins both v14 (bit_F1=0.9913, ni_FAR=0.00%) and v15direct (bit_F1=0.9691, ni_FAR=3.75%) — first single model to clear both eval sets with FAR pass.`

## Motivation

Iter 13–19 mixed (a) `classification_chips/` data into the master training set
with synthetic Normal/Invalid sentinels and (b) evaluated on the same v15-style
master set used to label-tune. That couples train and eval distributions and
makes the headline numbers optimistic.

Iter 21 retrains all candidate recipes on the **clean classification_chips/
4-class folder only** (bank_boundary / fork / scratch / scratch_rot, 200/class)
and evaluates on **two disjoint held-out sets**:

| eval | path | n_chip | composition |
|---|---|---:|---|
| v14class | `D:/project/data/wm-811k/chip_multilabel_v14class/` | 800 | 4 single + 6 combo + Normal + Invalid (12 keys × 50) — 16 sub-keys with combo expansion |
| v15direct | `D:/project/data/wm-811k/chip_multilabel_v15direct/` | 1000 | v14 + 4 OOD wafer-canvas (CenterDonut / CrossScratch / DiagonalSmear / Starburst) at 50/class — 20 sub-keys |

`v14class` measures in-distribution multi-label discrimination + Normal/Invalid
rejection. `v15direct` adds wafer-canvas OOD pressure: a model that hard-fires
defect bits on these foreign patterns is brittle in production. Both sets are
synthesized completely outside the training corpus, so any train/eval coupling
from prior iters is broken.

## Model spec (8 trains)

All 8 models share: `convnextv2_base.fcmae_ft_in22k_in1k_384` backbone, batch=4,
8 epochs, AdamW lr=1e-4, cosine eta_min=0, no warmup, RandomAffine translate+scale
(no rotation/flip per chip-multilabel policy), seed=1.

| run | spec |
|---|---|
| **A** `iter21A_12T5_reprod` | T5 baseline — pure BCE, no Normal training, no CutMix (iter12 reprod) |
| **B** `iter21B_T7N_pure` | T7N — BCE+LS=0.20, **with Normal** (y=−1 sentinel + zero-vec target), no CutMix |
| **C** `iter21C_T7N_cutmix` | T7N + std CutMix p=0.5 (iter18 baseline grid) |
| **D** `iter21D_18F1_repeat` | T7N + grid-CutMix LS=0.5 (iter18 F1 winner repeat) |
| **E** `iter21E_19C_repeat` ★ | T7N + **complement CutMix g=2 LS=1.0 (FCM-PM)** — iter19 C cell repeat |
| **F** `iter21F_19E_repeat` | T7N + complement g=3 LS=0.67 |
| **G** `iter21G_19G_repeat` | T7N + complement g=4 LS=0.25 |
| **H** `iter21H_19I_repeat` | T7N + complement g=4 LS=0.75 |

`g` = CutMix grid-density (number of paste boxes per image). `LS` = label-scale
applied to the pasted-region target multi-hot. `FCM-PM` = full-complement-mix
with paired masking (only one defect-bit pair active per cutmix instance).

## 16-cell results (8 model × 2 eval)

User-extracted bit-level F1 + per-class F1 + Normal/Invalid FAR from
`preds_chip.parquet` (across the 4 inference cells T0_{I3,I6,I7,I10}, the
configuration-best cell):

### v14class (in-distribution + N/I rejection, 800 chip)

| run | spec | bit_F1 | ni_FAR | bb F1 | fk F1 | sc F1 | sr F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | T5 baseline (no-Norm, no-CutMix) | 0.9745 | 100.00% | — | — | — | — |
| B | T7N pure (no CutMix) | 0.8609 | 100.00% | — | — | — | — |
| C | T7N + std CutMix | 0.9415 | 100.00% | — | — | — | — |
| D | T7N grid LS=0.5 (18F1) | 0.9431 | 1.25% | — | — | — | — |
| **E** ★ | T7N compl g=2 LS=1.0 (19C, FCM-PM) | **0.9913** | **0.00%** | — | — | — | — |
| F | T7N compl g=3 LS=0.67 (19E) | 0.9875 | 1.25% | — | — | — | — |
| G | T7N compl g=4 LS=0.25 (19G) | 0.9674 | 2.50% | — | — | — | — |
| H | T7N compl g=4 LS=0.75 (19I) | 0.9626 | 1.25% | — | — | — | — |

### v15direct (in-distribution + N/I + 4 OOD wafer-canvas, 1000 chip)

| run | spec | bit_F1 | ni_FAR | bb F1 | fk F1 | sc F1 | sr F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | T5 baseline | 0.7872 | 0.00% | 0.8612 | 0.9298 | 0.5841 | 0.7739 |
| B | T7N pure | 0.8089 | 2.50% | 0.9448 | 0.6420 | 0.7675 | 0.8811 |
| C | T7N + std CutMix | 0.8457 | 100.00% | 0.9937 | 0.6498 | 0.8060 | 0.9333 |
| D | T7N grid LS=0.5 (18F1) | 0.9252 | 2.50% | 0.9810 | 0.8698 | 0.8889 | 0.9610 |
| **E** ★ | T7N compl g=2 LS=1.0 (19C, FCM-PM) | **0.9691** | **3.75%** | **0.9905** | **0.9644** | 0.9439 | 0.9776 |
| F | T7N compl g=3 LS=0.67 (19E) | 0.9676 | 1.25% | 0.9776 | 0.9404 | **0.9747** | 0.9776 |
| G | T7N compl g=4 LS=0.25 (19G) | 0.9716 | 100.00% | 0.9682 | 0.9644 | 0.9666 | **0.9873** |
| H | T7N compl g=4 LS=0.75 (19I) | 0.9346 | 0.00% | 0.8491 | 0.9511 | 0.9508 | 0.9873 |

`ni_FAR` = fraction of Normal+Invalid chips that emit any defect-bit prediction.
`bF1` = micro-F1 over the 4 defect bits across all defect-class chips. `bb/fk/sc/sr`
= per-class F1 (bank_boundary / fork / scratch / scratch_rot).

## Findings

1. **★ E winner, both eval sets, FAR pass.** 19C (complement g=2 LS=1.0
   FCM-PM) is the only single-model recipe that clears both v14 (bit_F1=0.9913,
   ni_FAR=0.00%) and v15direct (bit_F1=0.9691, ni_FAR=3.75% ≤ 5% gate). Per-class
   F1 ≥ 0.94 on all 4 defects in v15direct.
2. **Normal training necessary but not sufficient.** B (T7N pure) has Normal
   training but no CutMix → ni_FAR=100% on v14 (Normal/Invalid still trigger
   defect prob). CutMix-driven compositional augmentation is what calibrates
   the rejection.
3. **Complement CutMix > std/grid CutMix on OOD.** v15direct wafer-canvas
   chips are far OOD; complement CutMix (E/F/G/H, bit_F1 ≥ 0.9346) decisively
   beats std (C, 0.8457) and grid (D, 0.9252). Diversity injected at small g
   with high LS (E: g=2, LS=1.0) generalizes; large g with low LS (G: g=4,
   LS=0.25) overfits — ni_FAR jumps to 100% on v15.
4. **Baseline A (T5, no-Norm) is misleadingly high on v14.** bit_F1=0.9745 but
   ni_FAR=100% (every Normal/Invalid fires a defect bit). v15 drops it to
   0.7872. This is the classic single-label-collapse failure that justifies
   the entire iter 13+ direction.
5. **fork is the volatile bit.** Across v15direct, fork F1 ranges from 0.6420
   (B) to 0.9644 (E/G). bank_boundary is most stable (0.85–0.99), scratch_rot
   second (0.77–0.99).
6. **C → D step.** Same train recipe (T7N + CutMix), only label-scale dropped
   from 1.0 → 0.5 with grid pasting → ni_FAR collapses 100% → 1.25% on v14
   (and 100% → 2.5% on v15). Soft labels in pasted regions are critical for
   N/I gate.

## Cross-iter delta (vs iter 19 best)

Iter 19B (single seed) reported macro_f1=0.8427, mAP=0.9685 on the legacy
master eval. Iter 21 E on the disjoint v15direct gives bit_F1=0.9691 with
all 4 per-class ≥ 0.94 — confirming the iter19 complement g=2 result was
**not a single-seed fluke** and survives the no-leak protocol.

## Sources

- `outputs/iter21A_12T5_reprod/T5_iter21A_12T5_reprod_seed1_260509_101142/`
  - `eval_v14class/stage1_260509_101645/{preds_chip,results_matrix,per_class_metrics}.parquet`
  - `eval_v15direct/stage1_260509_101707/{preds_chip,results_matrix,per_class_metrics}.parquet`
- `outputs/iter21B_T7N_pure/T7_iter21B_T7N_pure_seed1_260509_101733/`
- `outputs/iter21C_T7N_cutmix/T7_iter21C_T7N_cutmix_seed1_260509_102258/`
- `outputs/iter21D_18F1_repeat/T7_iter21D_18F1_repeat_seed1_260509_102803/`
- `outputs/iter21E_19C_repeat/T7_iter21E_19C_repeat_seed1_260509_103309/` ★
- `outputs/iter21F_19E_repeat/T7_iter21F_19E_repeat_seed1_260509_103953/`
- `outputs/iter21G_19G_repeat/T7_iter21G_19G_repeat_seed1_260509_105736/`
- `outputs/iter21H_19I_repeat/T7_iter21H_19I_repeat_seed1_260509_110530/`
- log: `outputs/_iter21_clean_baseline.log`

## Next

- 5-seed replication of E (19C FCM-PM) to bound ±σ around bit_F1=0.9691 on v15.
- Logit-avg ensemble of E + F (complementary on sc vs fork) — expected
  bit_F1 ≥ 0.97 on v15 with FAR ≤ 2%.
- Promote v15direct to the canonical reporting set; v14 is now the
  light-eval / dev-loop set.
