# Live FCMPM Experiment Audit

updated: 2026-06-03 19:41:59

Purpose: keep training/eval running, record split effects, prune weak checkpoints, and identify next additions.

## Active Processes

- `"C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe" -u -m chip_multilabel.recipe_sweep --datasets frozen_original,sota_gapstress_seed31_260531,sota_gapstress_seed97_260531,frozen_original_200_snapshot,frozen_original_2015_candidate --diag-devic`
- `C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe -u -m chip_multilabel._train_chip_variant --num-workers 0 --lr 1e-4 --no-normal --grad-checkpointing --val-criterion margin_max --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size`

## Split Summary

| axis | value | n | dataset_n | F1 mean | F1 std | FAR mean | gap mean | gap std | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 2 | 2 | 0.9966 | 0.0010 | 0.62 | 0.186 | 0.066 | promote: use for 2-factor/seed repeat |
| neg_target | neg002 | 1 | 1 | 0.9964 | 0.0000 | 0.04 | 0.212 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p030 | 1 | 1 | 0.9964 | 0.0000 | 1.75 | 0.233 | 0.000 | repeat: promising but dispersion unknown |
| neg_target | neg005 | 1 | 1 | 0.9954 | 0.0000 | 0.06 | 0.204 | 0.000 | promote: use for 2-factor/seed repeat |
| abpos_Avar_B100 | A090_B100 | 1 | 1 | 0.9949 | 0.0000 | 0.33 | 0.130 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p060 | 1 | 1 | 0.9944 | 0.0000 | 0.08 | 0.183 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p040 | 1 | 1 | 0.9929 | 0.0000 | 3.14 | 0.034 | 0.000 | observe: keep evidence, no expansion yet |
| cutmix_p | p020 | 1 | 1 | 0.9865 | 0.0000 | 1.01 | 0.098 | 0.000 | prune: delete pth and avoid expansion |
| neg_target | neg010 | 1 | 1 | 0.9847 | 0.0000 | 0.37 | 0.028 | 0.000 | prune: delete pth and avoid expansion |
| abpos_Avar_B100 | A080_B100 | 1 | 1 | 0.9837 | 0.0000 | 15.54 | -0.246 | 0.000 | prune: delete pth and avoid expansion |
| abpos_Avar_B100 | A070_B100 | 1 | 1 | 0.9820 | 0.0000 | 0.54 | 0.044 | 0.000 | prune: delete pth and avoid expansion |

## Queue / Prune Suggestions

- add/confirm: cutmix_p p055 and p065 around p060; seed-repeat p060
- combine: neg002 x best cutmix_p and neg002 x T10
- combine: neg005 x best cutmix_p and neg005 x T10
- prune: A070_B100 A target down-weight is weak; prefer ASL/T10
- prune: A080_B100 A target down-weight is weak; prefer ASL/T10

## Response Curve Fit

Internal score: if `FAR<=1%`, `score = bit_F1 + 0.05*gap`; otherwise the score subtracts `0.02*(FAR-1)`.
This keeps high-F1 rows from winning if the negative tail leaks.

| axis | points | fit | R2 | observed best x | suggested x | note |
|---|---:|---|---:|---:|---:|---|
| abpos_Avar_B100 | 4 | quadratic | 0.413 | 1.0000 | 1.0000 | best observed |
| cutmix_p | 5 | quadratic | 0.358 | 0.5000 | 0.5000 | best observed |
| neg_target | 4 | quadratic | 0.998 | 0.0200 | 0.0238 | quadratic vertex |

## Operating Rule

- Delete low-value `.pth` only; keep CSV/MD/log/probability evidence.
- Expand only conditions with high mean F1, controlled FAR, and stable gap.
- Treat one-off high rows as candidates, not conclusions.
- Prefer ASL/T10 over lowering A target when A target down-weight reduces combo POS min.
