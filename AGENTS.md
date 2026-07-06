# Codex Project Instructions

## Absolute Rule: Always Use Absolute Paths

When reporting any local file, directory, script, log, image, model, dataset, or
output location to the user, always write the full absolute path with a drive
letter.

Required examples:

- `D:/project/known-cnn/docs/chip-multilabel/manager_report/FCMPM_CUTMIX_P_TRADEOFF_260608_no_snapshot.md`
- `E:/data/images/classification_chips`
- `D:/project/known-cnn/outputs/frozen_original`

Forbidden examples:

- `docs/chip-multilabel/manager_report/...`
- `outputs/frozen_original`
- `./outputs/...`
- `_recipe_sweep_p_dataset_matrix_260607.out.log`

This applies to final answers, progress updates, tables, report links, command
summaries, generated images, model checkpoints, logs, and experiment results.

If a relative path appears in command output, convert it to an absolute path
before showing it to the user.
