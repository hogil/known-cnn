# 00 — Problem setup

## ★★★ 절대 규칙 (260512) — train / eval composition

**학습**: 4 single defect class 만 (`bank_boundary`, `fork`, `scratch`, `scratch_rot`).
Normal / Invalid / OOD-wafer-pattern chip 은 학습 데이터에 포함 안 함. 모든
train script 에 `--no-normal` flag 명시.

**평가**: 5 group composition —
- (a) 4 single defect (positive)
- (b) 2-combo (현재 5: bb+fork, bb+scratch, bb+scratch_rot, fork+scratch, fork+scratch_rot; scratch+scratch_rot 은 same-family 제외) (positive)
- (c) Normal (negative)
- (d) Invalid (negative)
- (e) OOD wafer-pattern: CenterDonut, CrossScratch, DiagonalSmear, Starburst 등 (negative)

**Metric**:
- **bit F1** = positive cells (single + combo) macro-F1. `macro_f1` (전체 평균) 과 혼동 금지.
- **FAR** = `(Normal_fp + Invalid_fp + OOD_fp) / (N_Normal + N_Invalid + N_OOD)`. 3 group 분리 (NI_far / OOD_far / Total_far) + Total 통일.
- **NI-only FAR (OOD 빼고)** single-report 금지. Phase 87 lesson — paper "ni_far=0%" claim 이 OOD 포함 시 1.07% 로 정정된 사고.

상세 + lesson: `~/.claude/projects/D--project-known-cnn/memory/feedback_chip_multilabel_train_eval_composition.md`,
`D:/project/known-cnn/CLAUDE.md` "★★★ 절대 규칙: chip multi-label train/eval composition".

## Task statement

A wafer chip image (200×200 grayscale) may contain **zero, one, or more
defects** simultaneously. We train a chip-level CNN on the **4 single defect
classes only** (single-label 4-way cross-entropy / BCE-LS) and use it as a
**multi-label** predictor over the real evaluation distribution: 4 single
defects + 5 two-defect combos + Normal + Invalid + OOD wafer patterns.

Single-label train (4 defects only), multi-label predict (positive + negative
groups).

Normal handling: inference-side max-prob threshold + dist-band gate + ensemble
(not a training class). Invalid handling: 5th-head heuristic at inference.
OOD handling: enters FAR denominator only — never trained on, never predicted
as class.

## Class set (11)

| Group              | Class                       | Notes                                  | n eval |
|--------------------|-----------------------------|----------------------------------------|-------:|
| Single defect      | `bank_boundary`             | Trained class                          |    240 |
|                    | `fork`                      | Trained class                          |    240 |
|                    | `scratch`                   | Trained class                          |    160 |
|                    | `scratch_rot`               | Trained class (rotated scratch)        |    160 |
| Combo (2 defects)  | `bank_boundary+fork`        | Combo                                  |    160 |
|                    | `bank_boundary+scratch`     | Combo                                  |    160 |
|                    | `bank_boundary+scratch_rot` | Combo                                  |    160 |
|                    | `fork+scratch`              | Combo                                  |    160 |
|                    | `fork+scratch_rot`          | Combo                                  |    160 |
| Other              | `Normal`                    | No defect                              |    160 |
|                    | `Invalid`                   | Invalid wafer (predicted via 5th head) |     40 |
| **Total**          |                             |                                        | **2200** |

`scratch + scratch_rot` is **excluded** (same defect family — labels are
ill-defined).

_Counts derived from `outputs/stage1_260505_162842/preds_chip.parquet`
n_eval=1760 over the four trained classes plus combos; Invalid handled
separately._

## Data synthesis pipeline (sister repo)

1. **Source distributions**: WM-811K wafer maps grouped by class.
2. **Sister repo `D:/project/known-cnn` `dist_apply/`** stamps fail-bits on a
   blank wafer using the per-class chip distributions, producing 200×200 chip
   images. For combo classes, two distributions are stamped on the same chip.
3. **Output tree**: `D:/project/data/wm-811k/unknown/<class>/*.png`.
4. The chip CNN backbone (`chip5_round4_v14_…`) is a TAPT variant: ImageNet
   FCMAE→supervised on the same synthetic chips.

## Backbone under test (T0)

- ckpt: `D:/project/known-cnn/outputs/logs_chip/chip5_round4_v14_260505_061558_running/best_model.pth`
- backbone: `convnextv2_base.fcmae_ft_in22k_in1k_384`
- img_size: **384** (chips are 200×200 → upsampled)
- training: **single-label CE**, 5 classes (`bank_boundary`, `fork`,
  `invalid_main`, `scratch`, `scratch_rot`)
- val_macro_f1 on 5-class single-label val set: **1.000** at epoch 1
- training compute: trivial (327 train / 82 val chips)

This is a strong but narrow model: it can recognise each defect on its own
but has never been asked to assert two defects in one chip.

## Evaluation harness

- File: `chip_multilabel_eval.py` (skill: `chip-multilabel-pipeline`)
- Inputs: `(model checkpoint)` × `(inference variant)`
- Output: `outputs/stage{1,2}_<TS>/`
  - `results_matrix.parquet` — one row per (train_id, inference_id) cell
  - `per_class_metrics.parquet`, `confusion_11class.parquet`,
    `errors.parquet`, `errors/<cell>/<error_type>/*.png`
  - `eval_summary.json`, `report.md`, `thresholds.json`

## Evaluation metrics (per cell)

- **`macro_f1`** — mean of per-class F1 over the 4 defect classes (combos
  decompose into multi-hot ground truth). Primary headline.
- **`micro_f1`**, **`mAP`**, **`hamming_loss`**, **`subset_accuracy`** — standard multi-label aggregates.
- **`top1_11class`** — argmax-decoded 11-class accuracy (combo recovery).
- **`ece_pre` / `ece_post`** — calibration before/after temperature.

Unless stated otherwise, “best cell” = highest `macro_f1`.

## Hard rules (constant across iters)

- **TTA permanently disallowed.** Rotation augmentation collapses
  `scratch` and `scratch_rot` (one ablation: −0.018 macro-F1). Never
  re-enabled even where it might help, because the rotation invariance
  destroys a class boundary.
- **GPU = 1 job at a time.** Sweeps run strictly sequentially.
- **`scratch + scratch_rot` combo excluded** from the 11-class set.

## Sister repos

- `D:/project/known-cnn` — supervised CNN (TAPT backbone supplier), data synthesis.
- `D:/project/mapviewer` — composite-map visualization (read-only).
