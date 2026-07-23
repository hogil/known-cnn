# Severstal sealed operator-match result (public confirmatory lever) — 260724

Backbone convnextv2_tiny (FCMAE), 2-stage FT (head-warmup 2ep → two-LR unfreeze,
warmup/cosine), gray letterbox 128×256, 10 epochs, 5 seeds. Strict image-level:
single 80/20 train/source-val; **real** normal 50/50 cal/test; **all real multi**
sealed positive. Primary: bit-F1 @ real-normal 1% FAR (τ from normal-cal, FAR on
untouched normal-test). NB-reject + conformal on. Proxy frozen test-blind BEFORE
sealed test. CSV: `outputs/multilabel_synth/severstal_operator_match_v2_b1/sealed_test_results.csv`.

## Sealed leaderboard (6 arms × 5 seeds, mean)

```
| Arm         | F1@FAR0.5% | F1@FAR1% | F1@FAR5% |  mAP   | proxy rank | note                     |
|-------------|------------|----------|----------|--------|------------|--------------------------|
| partition   |     0.0176 |   0.0397 |   0.0982 | 0.9163 |     4th    | SEALED WINNER            |
| cutmix      |     0.0106 |   0.0153 |   0.0654 | 0.9207 |     2nd    | 2nd                      |
| summation   |     0.0093 |   0.0134 |   0.0987 | 0.8944 |     3rd    | strong@FAR5%             |
| mixup       |     0.0065 |   0.0121 |   0.0646 | 0.9063 |     1st    | proxy pick (mis-selected)|
| fcm_pm      |     0.0019 |   0.0051 |   0.0269 | 0.9144 |     5th    | OURS -- FAILS here       |
| single_only |     0.0019 |   0.0028 |   0.0061 | 0.9193 |     --     | no-synth floor           |
```

## Paired, seed-matched CIs (95%)

```
| Comparison (F1@FAR1%)      |    d    | wins | CI_low  | sig  |
|----------------------------|---------|------|---------|------|
| partition vs single_only   | +0.0369 |  5/5 | +0.0209 | CI>0 |
| partition vs fcm_pm        | +0.0346 |  5/5 | +0.0179 | CI>0 |
| partition vs mixup         | +0.0277 |  5/5 | +0.0075 | CI>0 |
| partition vs summation     | +0.0263 |  5/5 | +0.0059 | CI>0 |
| partition vs cutmix        | +0.0244 |  5/5 | +0.0031 | CI>0 |
| mixup(proxy) vs single     | +0.0093 |  5/5 | +0.0043 | CI>0 |
| mixup(proxy) vs partition  | -0.0277 |  0/5 | -0.0478 | lose |
```

At FAR5% partition beats single_only (+0.086, CI>0) and fcm_pm (+0.071, CI>0)
significantly; vs cutmix/summation/mixup it leads but CI touches 0.

## Honest findings (report as-is)

**POSITIVE (survives):**
1. Content-blind pair-synthesis from single-label sources **substantially beats
   single-only** at the low-FAR operating point on PUBLIC industrial data. Every
   synthesis arm > single_only; best (partition) is ~14× single at FAR1%, ~16× at
   FAR5%, all paired CI>0. mAP is ~flat across arms (0.89–0.92) — i.e. the gain is
   invisible to mAP and shows ONLY at the real-FAR operating point (the paper's
   central measurement point).
2. Full system (val-margin checkpoint + NB-reject + conformal FAR) runs end-to-end
   on real normals; realFAR tracks the 1%/5% targets (0.005–0.020).

**NEGATIVE (must be reported, undermines two prior claims):**
1. **The test-blind proxy MIS-SELECTED.** It picked `mixup` (superposition op);
   the sealed winner was `partition` (whole-object side-by-side), which the proxy
   ranked 4th. Automatic operator selection is NOT reliable here.
2. **FCM-PM (our flagship grid-complement op) FAILS on Severstal** — 2nd-worst,
   statistically indistinguishable from single_only, and partition beats it 5/5
   CI>0 at both FAR1% and FAR5%. Root cause: Severstal defects are continuous,
   spatially-extended (thin scratches / patches); the 9×9 grid-cell complement
   chops each defect into scattered cells and destroys its coherence. FCM-PM's
   grid geometry is matched to the **chip** domain (grid-aligned discrete chip
   objects), not to continuous-defect steel.

## Cross-domain operator picture (descriptive law, 3 domains)

The best content-blind operator tracks the domain's **defect composition geometry**,
not a single universal choice:

```
| Domain    | data    | defect geometry             | best admissible op        |
|-----------|---------|-----------------------------|---------------------------|
| chip      | private | discrete grid-aligned cells | fcm_pm (grid complement)  |
| Severstal | public  | continuous extended regions | partition (whole-object)  |
| WM38      | public  | superimposed wafer bins     | cutmix/summation (overlay)|
```

The law "match the synthesis operator to defect-composition geometry" holds
**descriptively** across all three, but requires KNOWING the geometry. The
test-blind proxy (evidence-margin) is a heuristic that works on some domains and
failed on Severstal — an honest open problem, not a solved contribution.

## Paper implication (honest)

- The FCM-PM-as-universal-operator framing does NOT survive public validation.
- The robust, defensible claim is the **empirical study**: (i) synthesis-from-
  singles helps low-FAR multi-label detection, invisibly to mAP; (ii) the right
  operator is geometry-dependent and we characterize which per geometry; (iii)
  automatic (test-blind) selection remains unsolved (honest negative).
- This is aligned with the earlier pivot ("not an FCM-PM-only paper"). The
  system pieces (val-margin, NB-reject, conformal, the FAR-operating-point metric)
  are the transferable contribution; FCM-PM is one domain-specific instance.

## Honest probability

Severstal did NOT deliver the FCM-PM confirmatory win the paper wanted; it delivered
a clean "synthesis helps" public result PLUS an honest negative on FCM-PM generality
and proxy reliability. Net: ICLR stays ~**35–38%** (not lifted toward 60%). Strongest
honest venue framing is an empirical study; TMLR ~55–65% remains the realistic target.
