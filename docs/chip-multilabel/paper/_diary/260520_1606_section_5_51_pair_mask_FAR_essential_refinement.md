# Diary — 2026-05-20 16:06 autoloop · §5.51 FCM-PM pair-mask FAR-essential refinement

## Trigger

autoloop cycle 16:06 — §5.50 (Row 5 22-variant sweep, cron #313, 2026-05-20 00:55)
had concluded "pair-mask is not bit_F1-driving in Row 5", but that conclusion was
drawn entirely within the cutmix-mode=single context. Open question: does the
pair-mask carry a method-essential contribution within the FCM-PM
(cutmix-mode=complement) context where it was originally designed?

## Single-cell ablation

Base recipe (canonical FCM-PM = §5.49.7 E22 member iter116J_s1 family):
`LS=0.30, 8ep, cutmix-mode=complement, g=3, complete_label_scale=0.5`.

One isolated diff: `cutmix-pair=none` vs baseline `cutmix-pair=masked`.

Evaluation: n=2000 POS9 strict, I10 inference.

| variant | bit_F1 | Total FAR | delta_bit_F1 | delta_FAR |
|---------|--------|-----------|--------------|-----------|
| FCM-PM pair=masked (baseline) | 0.9927 | 0.00 % | — | — |
| FCM-PM pair=none (ablation)   | 0.9943 | 11.81 % | +0.0016 | +11.81 pp |

## Finding

- bit_F1 effect: **negligible** (+0.0016, within seed noise).
- FAR effect: **catastrophic** (+11.81 pp Total FAR; pair-mask is the single
  component preventing 11.81 % of negative chips from being false-asserted).

## Refined interpretation

- Pair-mask's true role: **FAR-essential within FCM-PM**, NOT bit_F1-driving in
  any context.
- Mechanism: FCM-PM complement-mode cutmix pastes complementary-class chip into
  host chip's empty regions. Without paired binary mask, complement pixels are
  trained under a soft target that does not strictly assign those pixels to the
  complement label, leaking complement-class confidence onto host-only and
  Normal/Invalid chips at inference (11.81 % Total FAR).
- With paired mask, complement-class pixels are bound to the complement label
  via the mask, severing the leakage path (0 % Total FAR).
- §5.50 Row 5 null result explained: Row 5 uses cutmix-mode=single (replace
  random rect with random other-class crop under label that already includes
  other class) — the FAR-leakage mechanism that pair-mask suppresses in FCM-PM
  does not exist in cutmix-single, so adding/removing pair-mask has no FAR effect
  to measure in Row 5.

## Paper change

- §5.51 appended (motivation → result → refined interpretation, 3 paragraphs).
- §5.50 correction-note appended (1 line pointing to §5.51).
- Corrected claim: "pair-mask is FAR-essential within FCM-PM only" (not "not
  effective in Row 5" or "method-essential everywhere").
- §4 method pair-mask design rationale should be re-anchored to FCM-PM
  complement-mode FAR-leakage mechanism (downstream §4 edit not done in this
  cycle — narrator-recorder cycle scope is §5.x narrative append only).

## Sources

- FCM-PM pair-off run: `outputs/fcm_pm_pair_none_260520_*/` family
  (eval_n2000_pred/stage1_*/preds_chip.parquet + eval_summary.json).
- Baseline iter116J_s1: §5.49 / §5.49.7 E22 member reference.
- Prior §5.50: cron #313 close-out (diary
  `260520_0055_cron313_r5n2k_phase2_complete.md`).
