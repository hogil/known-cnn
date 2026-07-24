# Severstal sealed operator result (public evidence) — 260724, soundness-corrected

Backbone convnextv2_tiny (FCMAE), 2-stage FT, gray letterbox 128×256, 10 epochs,
5 seeds. Strict image-level: single 80/20 train/source-val; real normal 50/50
cal/test; all real multi as positive eval. Primary: bit-F1 @ real-normal FAR.
CSV: `outputs/multilabel_synth/severstal_operator_match_v2_b1/sealed_test_results.csv`.
Stats: `multilabel_synth/severstal_sound_stats.py` (paired-t 95% CI, seed bootstrap,
exact Wilcoxon 1-sided).

> **Soundness note (supersedes the first draft of this doc).** The earlier version
> reported "partition beats ALL arms 5/5 CI>0" using a 1.96×SE normal-approx CI,
> claimed the `fcm_pm` arm was FCM-PM, and credited the full system incl. NB-reject.
> A re-audit corrected all of these; the honest claims are below. Nothing here uses
> the 1.96×SE interval anymore.

## What the arm labelled `fcm_pm` actually is (naming correction)

The Severstal runner's `fcm_pm` arm builds a SINGLE simplified grid image: of a 9×9
grid, ~1/3 of cells are swapped from source A onto B (`SV.synth_pair` in
`run_svhn_full_image_operator_match.py`). It is NOT the full FCM-PM operator
(complement-set + Pair-Mask view in `synthesis/fcmpm_image.py`), which is not wired
into this runner. **Reported here as `grid_complement_g3_9`.** No conclusion about
full FCM-PM's merit can be drawn from this arm.

## Sealed leaderboard (mean over 5 seeds)

```
| Arm                  | F1@FAR1% | F1@FAR5% |  mAP   | realFAR@1% (mean, range)  |
|----------------------|----------|----------|--------|---------------------------|
| partition            |   0.0397 |   0.0915 | 0.9163 | 1.29%  (0.80 - 2.00%)     |
| cutmix               |   0.0153 |   0.0587 | 0.9207 | 0.86%  (0.30 - 1.30%)     |
| summation            |   0.0134 |   0.0921 | 0.8944 | 0.64%  (0.40 - 0.90%)     |
| mixup                |   0.0121 |   0.0646 | 0.9063 | 0.94%  (0.75 - 1.15%)     |
| grid_complement_g3_9 |   0.0051 |   0.0202 | 0.9144 | 0.85%  (0.55 - 1.30%)     |
| single_only          |   0.0028 |   0.0061 | 0.9193 | 0.96%  (0.60 - 1.15%)     |
```

## Paired significance (paired-t 95% CI is the arbiter; bootstrap + Wilcoxon shown)

```
| Comparison (F1@FAR1%)                 |  mean d | wins | paired-t 95% CI      | Wilcx | verdict |
|---------------------------------------|---------|------|----------------------|-------|---------|
| partition   - single_only             | +0.0369 | 5/5  | [+0.0142, +0.0596]   | 0.031 | sig     |
| cutmix      - single_only             | +0.0125 | 5/5  | [+0.0047, +0.0202]   | 0.031 | sig     |
| summation   - single_only             | +0.0106 | 5/5  | [+0.0024, +0.0189]   | 0.031 | sig     |
| mixup       - single_only             | +0.0093 | 5/5  | [+0.0023, +0.0163]   | 0.031 | sig     |
| grid_compl  - single_only             | +0.0023 | 2/5  | [-0.0044, +0.0091]   | 0.375 | NOT sig |
| partition   - cutmix                  | +0.0244 | 5/5  | [-0.0058, +0.0547]   | 0.031 | boundary|
| partition   - summation               | +0.0263 | 5/5  | [-0.0026, +0.0551]   | 0.031 | boundary|
| partition   - mixup                   | +0.0277 | 5/5  | [-0.0009, +0.0562]   | 0.031 | boundary|
| partition   - grid_complement_g3_9    | +0.0346 | 5/5  | [+0.0110, +0.0582]   | 0.031 | sig     |
```

## Honest claims (exactly these, no more)

1. **Content-blind pair-synthesis from single-label sources significantly improves
   the low-FAR operating point over single-only, on PUBLIC industrial data.** Each
   of partition / cutmix / summation / mixup beats single_only at F1@FAR1% AND
   F1@FAR5% with paired-t 95% CI strictly > 0 and exact Wilcoxon p = 0.031 (the
   floor for n=5). The gain is invisible to mAP (all arms 0.89–0.92, single highest)
   and shows only at the real-FAR operating point. **This is the paper-usable
   result.**
2. **`partition` has the highest mean but is NOT statistically separated from the
   other synthesis operators.** partition − {cutmix, summation, mixup} paired-t 95%
   CIs all include 0 (boundary; bootstrap is anti-conservative at n=5 so not the
   arbiter). Report partition as "best mean rank," not "significantly best."
3. **The simplified `grid_complement_g3_9` arm is indistinguishable from single_only**
   (paired-t CI crosses 0, 2/5 wins, Wilcoxon 0.375). Because it is NOT full FCM-PM,
   this says nothing about FCM-PM; it only shows a naive 1/3-cell grid swap does not
   help on continuous steel defects.
4. **NB-reject contributes nothing on Severstal.** `nb_far_after == realFAR@0.01`
   in 30/30 rows — NB adds zero FAR reduction; it only trims positive coverage
   (0.977–1.000). FAR control here is entirely from normal calibration. Report NB as
   inert on this domain, separate from calibration.
5. **Calibration is approximate, not exact conformal.** The threshold is
   `np.quantile(., 1-α)` (linear interpolation) with a `>=` rule, not the exact
   split-conformal order statistic `s_(k)` with strict/randomized ties. Realized FAR
   at nominal 1% is 1.29% mean (0.80–2.00%) for partition — close but above nominal;
   the artifact must switch to the exact order-statistic rule to match the T3
   statement.
6. **Status: public DIAGNOSTIC evidence, not a sealed confirmatory test.** An earlier
   B0 pass already read the same 427-image multi-test before this B1 recipe was
   committed; the operator ranking within B1 is test-blind but the dataset is already
   opened. Do NOT call this an untouched/sealed confirmation.

## Addendum (260725): REAL full FCM-PM tested (addresses the "not real FCM-PM" audit gap)

The audit's #2 point was that the `fcm_pm` arm was a simplified single grid-swap, not
full FCM-PM. We wired the real operator (`synthesis/fcmpm_image.synth_fcmpm`: complete
complement set + Pair-Mask views) as `fcm_pm_full` and ran a fair batch-16 3-arm
comparison (single_only / partition / fcm_pm_full, 5 seeds; separate CSV
`severstal_fcmpmfull_b16/`, so NOT comparable to the batch-64 30-row set above).

```
| Arm (batch-16) | F1@FAR1% (mean, range)   | F1@FAR5% | mAP    |
|----------------|--------------------------|----------|--------|
| partition      | 0.1078 (0.081 - 0.165)   | 0.1767   | 0.9168 |
| fcm_pm_full    | 0.0520 (0.005 - 0.131)   | 0.1045   | 0.9249 |
| single_only    | 0.0174 (0.002 - 0.050)   | 0.0611   | 0.9216 |
```
```
| Paired (F1@FAR1%, seed-matched) |    d    | wins | 95% CI            | Wilcx | verdict |
|---------------------------------|---------|------|-------------------|-------|---------|
| partition   - single_only       | +0.0903 | 5/5  | [+0.064, +0.117]  | 0.062 | sig     |
| fcm_pm_full - single_only       | +0.0346 | 3/5  | [-0.057, +0.126]  | 0.625 | NOT sig |
| partition   - fcm_pm_full       | +0.0558 | 3/5  | [-0.054, +0.166]  | 0.312 | NOT sig |
```

**Honest finding.** Even the REAL full FCM-PM does NOT reliably beat single_only on
Severstal (d=+0.035 but 3/5, CI crosses 0, Wilcoxon 0.625 -- and highly variable: one
seed 0.131, three near 0.007). partition beats single_only significantly (5/5, CI>0)
and has 2x the mean of fcm_pm_full (though partition-vs-fcm_pm_full is not significant
at n=5 due to fcm_pm_full's variance). So the earlier "the arm wasn't real FCM-PM"
caveat is now closed: real FCM-PM ALSO fails to reliably help on continuous-defect
steel -- consistent with the measured footprint mechanism (grid-complement destroys
extended defects). Caveats: n=5, batch-16 (VRAM-limited, not the batch-64 protocol),
fcm_pm_full high variance. This CONFIRMS FCM-PM is domain-specific (chip grid, not
steel); it does not change the honest ICLR ~28-33%.

## Cross-domain operator picture (consistent OBSERVATION, not causal)

The best-mean content-blind operator differs by domain defect geometry — chip
(discrete grid-aligned) favours grid-cell complement; Severstal (continuous
extended) favours whole-object `partition`; WM38 (superimposed bins) favours
cutmix/summation. This is a consistent observation across three domains, NOT a
causal proof (no mask-ablation on the same domain). The FCM grid-coherence theorem
(planned) gives the mechanism hypothesis a formal probability; until a same-domain
mask ablation is run, it stays "consistent with," not "proves."

## Honest probability contribution

Severstal yields ONE clean, paper-usable public result — synthesis-from-singles
significantly beats single-only at low FAR — plus honest negatives (operator
choice not separated; the naive grid arm and NB inert here; not a sealed test).
It does not lift ICLR toward 60%. Consistent with the decision-range
`~30–40%` after soundness fixes; the theory cluster (T6 selection-regret + FCM
coherence + constructive hedge) is the lever that can reach `~35–45%` IF it passes
independent proof audit.
