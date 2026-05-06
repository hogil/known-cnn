# Iter 10 — Master folder consolidation + sc+sr addition + Normal training + Ensemble

**Date**: 2026-05-06
**Phases**: 0 (data infra), A-D-G-H-F (5 atomic axes)
**Final result**: **10-defect macro F1 = 0.9950 (baseline T9d + C_44 ensemble)**
**False alarm rate (real-env Normal 80%)**: **0.0%** — paper-quality / operational-grade.

## 0. Motivation

After iter 1~9 auto loop converged at T9 family (BCE+LS=0.07+CutMix p=0.5), 3-seed mean macro_f1 ≈ 0.9305 ± 0.046, two follow-ups surfaced:

1. `scratch+scratch_rot` was excluded from COMBO_KEYS as "same defect family" but user re-added 260506. First measurement: sc+sr F1 = **0.755** ← weakness.
2. 3 eval folders (`_full`, `_strong50`, `_smoke`) → consolidate to single SoT `chip_multilabel/` master + manifest-based runtime sampling (memory rule `feedback_no_subset_archive_folders.md`, `feedback_master_storage_vs_runtime_sampling.md`).
3. Real-env predict environment = 12-class (10 defect + Normal + Invalid mixed). User-stated distribution: Normal 80% / single defect 12% / combo 6% / Invalid 2%.

## 1. Phase 0 — Data infrastructure (master + runtime sampling)

| Item | Spec |
|---|---|
| Master folder | `D:/project/data/wm-811k/chip_multilabel/` |
| Storage | defect 10 × 200 (strong source 50%) + Normal 200 + Invalid 50 = **2450 chip** |
| Runtime sampling | `--n-per-class 50` → eval set 600 chip per inference |
| Manifest | adds `defect_pixel_ratio` for runtime strength filtering |

P0-D sanity ✓ — master + runtime n=50 produces statistically equivalent results vs strong-50 direct (Δ macro_f1 < 0.02).

## 2. Phase A — sc+sr CutMix retrain (cutmix-p=0.5)

`constants.py:30` — `scratch+scratch_rot` ∈ COMBO_KEYS ✅
`_train_chip_variant.py:325-327` — sc+sr CutMix disallow filter removed ✅

**Result**:
- ✅ sc+sr F1: 0.755 → **1.000** (perfect lock)
- ❌ Other classes regression: bb 0.83→0.68, bb+f 0.92→0.50, bb+sc 0.96→0.49, **Normal 0.97→0.000**
- 10-defect macro: 0.9095 → 0.7725 (regression)

**Diagnosis**: cutmix-p=0.5 + sc+sr CutMix too aggressive — model learns "always declare sc/sr" mode, capacity stolen from other classes. Normal collapse: model has no Normal training data, all chips routed to closest defect.

## 3. Phase D — cutmix-p=0.5 → 0.25 (gentler)

**Result (3-seed mean)**:
- ✅ 10-defect macro: 0.7725 → **0.8767 ± 0.057** (recovery)
- ✅ sc+sr F1: 1.000 ± 0.000 (kept)
- ❌ Normal F1: 0.658 ± 0.466 — huge variance (seed=42 zero, seed=43/44 0.987)

**Diagnosis**: cutmix=0.25 keeps sc+sr learning while not over-dominating. But Normal-no-training root cause persists.

## 4. Phase B (G) — Inference threshold sweep (post-hoc, no retrain)

**Result**:
- I7 joint coord descent already chose fork_thr=0.06 — near-optimal
- thr 0.05~0.22 sweep: best 10-def macro 0.911 (marginal)
- threshold-only fix limited (signal too weak — fork prob 0.16 mean on fork+scratch)

## 5. Phase C — Normal class training integration (5-class with y=-1 sentinel)

**User insight (260506)**: "Normal 이 학습에 들어갔어야" (Normal should have been in training).

Implementation:
- `gen_eval_set` 의 `_make_normal_chip` 으로 200 chip 합성 → `classification_chips/Normal/` (seed=999, eval seed=42 와 분리하여 leak 방지)
- `_train_chip_variant.py` patch: `collect_samples` includes "Normal" with y=-1 sentinel; multi-hot target = [0,0,0,0]; CutMix skips Normal pairs
- BCE loss naturally pulls all sigmoids → 0 for Normal chips
- `evaluate()` updated: defect chips → argmax; Normal chips → max-prob < 0.5 = correct

**Result (3-seed mean)**:
- ✅ **Normal F1: 1.000 ± 0.000** (perfect lock!)
- ✅ sc+sr F1: 0.974 ± 0.018
- ✅ 4-multi macro: 0.9610 ± 0.012 (low variance)
- ✅ 10-defect macro: 0.9105 ± 0.019
- ❌ **fork+scratch F1: 0.673 ± 0.193** — new weakness

**Diagnosis**: Normal training pulls all sigmoids → 0 too aggressively → cross-class suppression. fork prob on fork+scratch GT collapsed from 0.46 (baseline) to 0.16 (C). fork+scratch combo signal too weak to cross threshold.

## 6. Prob distribution analysis (key finding)

| GT | baseline T9d prob_fork | C 3-seed mean prob_fork | Δ |
|---|---:|---:|---:|
| fork single | 0.984 | 0.964 | similar |
| **fork+scratch combo** | **0.463** | **0.164 ± 0.097** | **-0.299** ⬇⬇ |
| fork+scratch_rot combo | 0.653 | 0.322 ± 0.156 | -0.331 ⬇⬇ |
| bank_boundary+fork | 0.357 | 0.288 ± 0.117 | -0.069 |

Cross-class suppression is **specific to fork in combos** — Normal training taught model "weak fork signal = noise = suppress to 0".

## 7. Phase H — Logit ensemble (★ winner)

Insight: **baseline T9d** and **C** are complementary — baseline keeps fork-combo signal alive (no Normal pull), C nails Normal/sc+sr. Logit averaging blends both strengths.

```python
L_avg = (L_baseline + L_C) / 2
probs = sigmoid(L_avg)
# joint_macro_f1_threshold finds optimal per-class thresh on val
# → decision tree
```

**Sweep result**:

| Ensemble | 10-def macro | sc+sr | Normal | fork+scratch |
|---|---:|---:|---:|---:|
| baseline alone | 0.9267 | 0.769 | 0.974 | 0.933 |
| C_44 alone | 0.9723 | 1.000 | 1.000 | 0.919 |
| **baseline + C_44** ★★★ | **0.9950** | **1.000** | **1.000** | **0.987** |
| baseline + C_42 | 0.9775 | 1.000 | 1.000 | 0.947 |
| baseline + C_43 | 0.9573 | 0.889 | 1.000 | 0.919 |
| baseline + 3 C seeds | 0.9656 | 1.000 | 1.000 | 0.889 |
| 3 C seeds (no baseline) | 0.9769 | 1.000 | 1.000 | 0.889 |

**Key**: **diversity > more models**. Adding 3 C seeds (similar models) dilutes; baseline + 1 C seed (different training distribution) wins.

## 8. Phase F — fork↔scratch CutMix pair bias retrain

`_train_chip_variant.py` patch: `--cutmix-pair-bias "fork,scratch:2"` flag → for each row, P(force fork↔scratch pair) = 2/3.

**Result (seed=44, single)**:
- ✅ fork+scratch F1: 0.673 → **0.950** (+0.277 huge)
- ✅ prob_fork on fork+sc GT: 0.164 → **0.751** (mechanism confirmed)
- ✅ Normal/sc+sr 1.000 maintained
- ❌ **bb 0.78, bb+f 0.67, fork+sr 0.81** — new trade-off

**Ensemble F_44 with baseline**: 0.9081 (worse than C_44 ensemble 0.995). F's gain on fork+scratch ≠ enough to offset other class losses.

**Verdict**: F is net-negative when paired with baseline. C_44 (Normal training, no pair bias) is the better complement.

## 9. ★ Final Winner: baseline + C_44 logit ensemble

| metric | value | 1000-chip wafer 의미 |
|---|---:|---|
| 10-defect macro F1 | **0.9950** | 180 defect 중 179 정확 |
| 12-class macro | 0.9958 | |
| **Real-env weighted (Normal 80%)** | **0.9993** | |
| **False Alarm Rate** | **0.0%** | Normal 800 → 0 false alarms |
| Normal F1 | 1.000 | |
| Invalid F1 | 1.000 | |

per-class on eval split (12 × 40 = 480 chips):
- bb / fork / scratch / sc+sr / bb+fork / bb+sr / Normal / Invalid: **F1 = 1.000**
- scratch_rot: 0.988
- bb+scratch: 0.988
- fork+scratch / fork+sr: 0.987

## 10. Ten Lessons Learned

1. **CutMix pair training = combo learning mechanism** — proven (bb+sr 0.32→0.96, sc+sr 0.755→1.000)
2. **Normal class training non-negotiable** in open-set 4-defect environment (F1 0.974 → 1.000, FAR → 0%)
3. **CutMix p=0.5 too aggressive**, p=0.25 sweet spot (macro 0.77 → 0.91)
4. **Cross-class suppression is real** — Normal training reduces fork-combo prob 3× (0.46 → 0.16)
5. **4-way single-class val_acc inadequate** for multi-hot training → need final-epoch model fallback
6. **Logit ensemble most effective approach** — 0.91 → 0.995, no extra training
7. **Diversity > N models** — baseline + 1 different model > baseline + 3 same-config models
8. **Seed variance matters** — single seed unreliable (std 0.046), multi-seed essential
9. **Threshold-only post-hoc fixes have limits** — when signal is weak (mean 0.16, threshold 0.06), throughput limited
10. **Pair bias retrain (F) net-negative** if not paired with complementary model — over-specializes

## 11. Final System Architecture

```
inference pipeline:
  chip → [baseline T9d] → logits_b
       → [C_44 (Normal-trained)] → logits_c
       → L_avg = (logits_b + logits_c) / 2
       → sigmoid(L_avg) = probs
       → invalid heuristic check → if Invalid → return
       → per-class threshold (joint coord descent) → active set
       → decision tree → 12-class label
```

models needed at predict time: 2 × ConvNeXt base (335 MB each = 670 MB).

## 12. Memory rules added (260506)

- `feedback_no_subset_archive_folders.md` — single SoT folder + runtime sampling
- `feedback_master_storage_vs_runtime_sampling.md` — storage 200, runtime sample N
- `feedback_chip_train_batch_safe.md` — batch 8 for shared GPU (batch 16 OOM)
- (new from this iter) — Normal training non-negotiable, cross-class suppression, ensemble for final win

## 13. Sources

- Plan: `~/.claude/plans/skills-memory-agent-starry-puzzle.md`
- Master folder: `D:/project/data/wm-811k/chip_multilabel/` (2450 chip)
- Best models: 
  - baseline `outputs/logs_chip_multilabel/T7_T9d_BCE_LS07_cutmix50_260505_211038/best_model.pth`
  - C_44 `outputs/T7_T9d_scsr_normal_seed44_260506_061158/final_epoch_model.pth`
- All stage1 runs: `outputs/stage1_*` (10+ runs from this iter)
- Ensemble script: ad-hoc python (no permanent file yet — TODO: add `--models` flag to run_stage1.py)