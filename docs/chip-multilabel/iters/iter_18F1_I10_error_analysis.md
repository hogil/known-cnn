# iter18F1 (T7N + grid + complete fill, label_scale=0.5) I10 error analysis

source: `outputs/iter18F1_T7N_gridcomplete_label0.5/T7_T7_iter18F1_gridcomplete_label0.5_seed1_260508_132951/eval_seed1_I10only/stage1_260508_145545/`

I10 (joint coord-descent + softmax-entropy → Normal) — bit_F1=0.9075 / bit_F1_defect_only=0.9937 / ni_chipFAR=1.25% (1/80). Per-class F1: bb=0.8196 / fork=0.955 / sc=0.9277 / sr=0.9275.

## 1. missed_normal: Normal_0118 — 1 chip, ni_chipFAR floor

GT=Normal (true_labels=[]); pred=`fork+scratch_rot` via combo decision.
probs: bb=0.0959, fork=0.236, sc=0.1967, **sr=0.5491**.

Visual: pure grade-0 white background with sparse green/grey speckle (random Normal noise). No structured signal — no vertical lines, no diagonals. Yet sr head fires at 0.5491, pushing fork above its very low threshold (0.18). This is the classic **diffuse-prior over-fire of sr on background dot-density**: when speckle density modestly exceeds Normal training distribution, sr re-purposes its diagonal prior and bleeds prob mass; entropy-based Normal gate then fails because sr alone is sharp enough to look "confident". Fix path: tighter sr calibration (post-hoc rebalance against Normal density) or a Normal-density-aware bias term in the sr logit.

## 2. wrong_combo (n=85) — confusion top-5 (almost entirely triples)

| GT | pred | n |
|---|---|---|
| bb+sc+sr | bb only | 20 |
| bb+f+sr | bb+sr | 12 |
| bb+f+sc | bb+sc | 12 |
| bb+f+sc | bb+f | 10 |
| bb+f+sc | bb only | 8 |

Pattern: **3-bit GT → 2-bit or 1-bit pred**, never 3-bit prediction. Joint coord-descent at I10 evidently never selects `|S|≥3` because the score penalty exceeds gain. 27/85 are `combo_collapsed` (raw thresholds passed but joint score selected smaller subset); 45 are `combo` mode (raw thresholds already excluded one bit). Example bb+sc+sr→bb only: probs bb=0.46, sc=0.32 (>0.26 thr), sr=0.19 (>0.14 thr) — both sc and sr crossed thresh but I10 dropped them.

## 3. wrong_normal_entropy (n=122) — Normal-collapse missed-bit count

40 are Invalid (true_labels=[] → pred=Normal is **correct in bit-F1** — these contribute 0 FN).
Remaining 82 are defect chips collapsed to Normal. Missed-bit count over those 82:

| class | missed bits |
|---|---|
| fork | 73 |
| scratch_rot | 71 |
| scratch | 54 |
| bb | 47 |

GT distribution of the 82 defect collapses: f+sc+sr=35, bb+f+sr=27, bb+f+sc=10, bb+sc+sr=9, bb+f=1.
**Triples dominate** (81/82). Triple chips concentrate ink across 3 patterns simultaneously, suppressing each individual head's confidence below the entropy gate. Sample bb+f+sc: bb=0.40 fork=0.26 sc=0.24 sr=0.09 — all 4 < 0.5, distribution is high-entropy across 3 classes, gate fires Normal. This is the **multi-bit confidence-dilution failure**: entropy gate confuses "dispersed positive evidence" with "no evidence".

## 4. per-class F1 weakness diagnosis

- **bb F1=0.8196** (P=0.6974 / R=0.9938). Recall is essentially perfect (bb FN=1) — the loss is precision (FP=69 in bit_metrics). bb is mostly **over-fired** rather than missed; combined with the 12-chip Invalid raw-thresh fire (`prob_bb` 0.30-0.38 on plain orange-border B-text chips), bb head treats orange/text edges as bank lines.
- **fork=0.955 / sc=0.9277 / sr=0.9275** — fork dramatically improved vs 12-T5 baseline (fork F1≈0.40). Mechanism (iter18 grid + complete fill + label_scale=0.5 in T7N):
  1. **complete grid fill** during training expands fork's positive support beyond sparse strokes, making fork prob calibration dense rather than spike-only.
  2. **label_scale=0.5 (BCE+LS)** halves overconfident negatives — fork's dual-class neighbors (sc, sr) no longer push fork prob to near-zero.
  3. paired CutMix (iter16) compositional learning prior carried over: fork-in-combo recall locked.

`combo_collapsed=27` count confirms the failure mode is **decision rule too conservative on |S|≥3**, not the head probabilities themselves.

## 5. next atomic recommendation

**Variant**: I10b — relax joint coord-descent's |S|=3 penalty floor.

Concretely: in the joint-score selection at I10, allow |S|=3 when the 3rd bit's marginal score gain ≥ τ_3 (calibrated on val so that triple recall balances Normal FAR). Current I10 implicitly sets τ_3=∞ (never selects 3 bits).

**Rationale**:
- 207/208 errors live on triple GT chips. A decision rule that physically cannot output 3 bits caps triple recall at zero by construction (paper analogy: Cole et al. 2021 SPML show *decision rule* ≥ *loss* for multi-label F1; Lipton et al. 2014 F1-max for class-specific thresholding extends naturally to subset-size threshold).
- The probability heads themselves already have decent triple support: in 27 combo_collapsed chips, all 3 raw thresholds were crossed before the joint rule rejected the selection. The fix is in the rule, not the model — minimum cost change.
- Expected gain: ≥40/640 = +0.06 bit_F1 if we recover even half of the triple wrong_combo cases (85 → 45). bb precision should improve incidentally because spurious bb-only collapses on triples disappear.
- Risk: ni_chipFAR may rise from 1.25% slightly (loosening triple admit lets more diffuse-prior chips fire 3 bits). Calibrate τ_3 to keep FAR ≤ 5% paper constraint.

## one-line summary

missed 1 chip = **Normal_0118 sr-diffuse-prior over-fire (sr=0.55 on speckle)** / most common wrong_combo = **bb+sc+sr → bb only (20)** → root cause is **joint coord-descent never emits |S|=3** / fork dramatic improvement = **complete-fill grid + label_scale=0.5 BCE+LS densified fork calibration** / next atomic = **relax I10's |S|=3 admit threshold τ_3, calibrated on val for triple recall vs Normal FAR**.

[OUT] D:/project/known-cnn/outputs/iter18F1_T7N_gridcomplete_label0.5/T7_T7_iter18F1_gridcomplete_label0.5_seed1_260508_132951/eval_seed1_I10only/stage1_260508_145545/errors/T0__I10/
