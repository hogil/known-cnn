# Live FCMPM Experiment Audit

updated: 2026-06-04 22:31:43

Purpose: keep training/eval running, record split effects, prune weak checkpoints, and identify next additions.

## Active Processes

- `"C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe" -u -m chip_multilabel.recipe_sweep --datasets frozen_original,sota_gapstress_seed31_260531,sota_gapstress_seed97_260531,frozen_original_200_snapshot,frozen_original_2015_candidate --diag-devic`
- `C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe -u -m chip_multilabel._train_chip_variant --num-workers 0 --lr 1e-4 --no-normal --grad-checkpointing --val-criterion margin_max --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size`

## Split Summary

| axis | value | n | dataset_n | F1 mean | F1 std | FAR mean | gap mean | gap std | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cutmix_p | p0575 | 1 | 1 | 0.9975 | 0.0000 | 0.40 | 0.285 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p080 | 1 | 1 | 0.9971 | 0.0000 | 1.96 | 0.234 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_neg | neg002_s42 | 1 | 1 | 0.9970 | 0.0000 | 42.66 | -0.010 | 0.000 | prune: delete pth and avoid expansion |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 2 | 2 | 0.9966 | 0.0010 | 0.62 | 0.186 | 0.066 | promote: use for 2-factor/seed repeat |
| neg_target | neg002 | 1 | 1 | 0.9964 | 0.0000 | 0.04 | 0.212 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p065 | 1 | 1 | 0.9964 | 0.0000 | 1.54 | 0.171 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p030 | 1 | 1 | 0.9964 | 0.0000 | 1.75 | 0.233 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p055 | 1 | 1 | 0.9963 | 0.0000 | 0.49 | 0.203 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_abpos_Avar_B100 | A080_B100_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.00 | 0.258 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p060_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.31 | 0.225 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p100 | 1 | 1 | 0.9961 | 0.0000 | 1.43 | 0.253 | 0.000 | repeat: promising but dispersion unknown |
| neg_target | neg0015 | 1 | 1 | 0.9960 | 0.0000 | 0.11 | 0.180 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg002_s13 | 1 | 1 | 0.9960 | 0.0000 | 0.60 | 0.143 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_neg | neg003_s13 | 1 | 1 | 0.9958 | 0.0000 | 0.71 | 0.198 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg0015_s13 | 1 | 1 | 0.9958 | 0.0000 | 2.62 | 0.163 | 0.000 | observe: keep evidence, no expansion yet |
| cutmix_p | p070 | 1 | 1 | 0.9957 | 0.0000 | 1.01 | 0.226 | 0.000 | repeat: promising but dispersion unknown |
| neg_target | neg005 | 1 | 1 | 0.9954 | 0.0000 | 0.06 | 0.204 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_baseline | s42 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | -0.093 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg005_s42 | 1 | 1 | 0.9953 | 0.0000 | 1.93 | 0.216 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_neg | neg002_s99 | 1 | 1 | 0.9952 | 0.0000 | 0.08 | 0.192 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p0575_s13 | 1 | 1 | 0.9952 | 0.0000 | 1.91 | 0.204 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_p | p060_s99 | 1 | 1 | 0.9950 | 0.0000 | 0.77 | 0.234 | 0.000 | promote: use for 2-factor/seed repeat |
| abpos_Avar_B100 | A090_B100 | 1 | 1 | 0.9949 | 0.0000 | 0.33 | 0.130 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p060 | 1 | 1 | 0.9944 | 0.0000 | 0.08 | 0.183 | 0.000 | promote: use for 2-factor/seed repeat |
| neg_target | neg0025 | 1 | 1 | 0.9944 | 0.0000 | 3.32 | 0.145 | 0.000 | observe: keep evidence, no expansion yet |
| cutmix_p | p0625 | 1 | 1 | 0.9943 | 0.0000 | 0.25 | 0.113 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_baseline | s13 | 1 | 1 | 0.9942 | 0.0000 | 2.00 | 0.190 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_p | p055_s13 | 1 | 1 | 0.9940 | 0.0000 | 11.89 | 0.117 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p0575_s99 | 1 | 1 | 0.9938 | 0.0000 | 0.32 | 0.233 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg0025_s13 | 1 | 1 | 0.9937 | 0.0000 | 0.43 | 0.126 | 0.000 | repeat: promising but dispersion unknown |
| neg_target | neg003 | 1 | 1 | 0.9931 | 0.0000 | 0.08 | 0.138 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p040 | 1 | 1 | 0.9929 | 0.0000 | 3.14 | 0.034 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_p | p0575_s42 | 1 | 1 | 0.9924 | 0.0000 | 52.98 | -0.084 | 0.000 | prune: delete pth and avoid expansion |
| cutmix_p | p090 | 1 | 1 | 0.9921 | 0.0000 | 0.96 | 0.175 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_abpos_Avar_B100 | A090_B100_s13 | 1 | 1 | 0.9920 | 0.0000 | 4.06 | 0.211 | 0.000 | observe: keep evidence, no expansion yet |
| grid_g3 | grid12 | 1 | 1 | 0.9917 | 0.0000 | 0.11 | 0.181 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_baseline | s99 | 1 | 1 | 0.9904 | 0.0000 | 1.51 | 0.192 | 0.000 | observe: keep evidence, no expansion yet |
| grid_g3 | grid6 | 1 | 1 | 0.9902 | 0.0000 | 1.32 | 0.111 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_neg | neg005_s99 | 1 | 1 | 0.9892 | 0.0000 | 6.15 | 0.024 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p060_s42 | 1 | 1 | 0.9874 | 0.0000 | 3.94 | 0.017 | 0.000 | prune: delete pth and avoid expansion |
| cutmix_p | p020 | 1 | 1 | 0.9865 | 0.0000 | 1.01 | 0.098 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg005_s13 | 1 | 1 | 0.9863 | 0.0000 | 16.11 | 0.047 | 0.000 | prune: delete pth and avoid expansion |
| neg_target | neg010 | 1 | 1 | 0.9847 | 0.0000 | 0.37 | 0.028 | 0.000 | prune: delete pth and avoid expansion |
| abpos_Avar_B100 | A080_B100 | 1 | 1 | 0.9837 | 0.0000 | 15.54 | -0.246 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A070_B100_s13 | 1 | 1 | 0.9835 | 0.0000 | 1.30 | 0.155 | 0.000 | prune: delete pth and avoid expansion |
| abpos_Avar_B100 | A070_B100 | 1 | 1 | 0.9820 | 0.0000 | 0.54 | 0.044 | 0.000 | prune: delete pth and avoid expansion |
| grid_g3 | grid3 | 1 | 1 | 0.9813 | 0.0000 | 8.15 | -0.037 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T6 | 1 | 1 | 0.9602 | 0.0000 | 8.48 | -0.337 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T10 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | -0.089 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T4 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | -0.089 | 0.000 | prune: delete pth and avoid expansion |

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
| cutmix_p | 13 | linear | 0.097 | 0.5750 | 0.5750 | best observed |
| neg_target | 7 | linear | 0.047 | 0.0200 | 0.0200 | best observed |

## Operating Rule

- Delete low-value `.pth` only; keep CSV/MD/log/probability evidence.
- Expand only conditions with high mean F1, controlled FAR, and stable gap.
- Treat one-off high rows as candidates, not conclusions.
- Prefer ASL/T10 over lowering A target when A target down-weight reduces combo POS min.
