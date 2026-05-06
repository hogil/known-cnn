# 8. Iteration Protocol

We treat the experimental protocol as a methodological contribution
in its own right. The protocol below was followed without
deviation across iters 1–5 and is designed to be portable to other
small-data, single-GPU defect-classification settings.

## 8.1 The GPU=1-job rule

All training and evaluation runs are strictly sequential. No two
training jobs ever share the GPU. Why:

- **Reproducibility.** Concurrent jobs introduce non-determinism via
  GPU memory pressure and AMP autocast scheduling. Sequential runs
  produce bit-identical training-loss trajectories given the same
  seed, which makes regression debugging tractable.
- **Cost transparency.** Every iter has a known wall-clock cost. The
  +0.1966 macro-F1 result is the product of ~115 GPU-min — easy to
  cite and reproduce.
- **Failure containment.** A buggy training script that hangs the
  GPU does not corrupt other runs in flight.

The dispatcher (`chip_multilabel/run_phase_a.py`) enforces sequencing
by waiting on the previous run's directory to finalise its
`train_summary.json` before launching the next.

## 8.2 Hard rules

These never relax across iters:

- **TTA permanently disallowed** (iter 1 measured −0.018; the rule
  applies to *any* augmentation that violates the class taxonomy,
  including rotation, flip, transpose).
- **`scratch + scratch_rot` combo excluded** from the 11-class set
  (same defect family).
- **Result directories are append-only.** No `outputs/<run>/` is ever
  deleted or overwritten by the agent. New experiments get new
  directory names. This is a strict rule in the user's CLAUDE.md
  and is enforced by giving each run a timestamp suffix.
- **No emojis in technical artefacts.** Research-grade prose.

## 8.3 Iteration template

Every iter follows the same six-step template:

1. **Define hypothesis.** Name the failure mode the next change
   targets, and predict the direction of the macro-F1 change. If
   the hypothesis cannot be falsified by the iter's measurement,
   redesign the iter.
2. **List the change.** A single-axis change (one inference variant,
   one loss family, one hyperparameter sweep). Multi-axis changes
   are split across iters so that root-cause attribution stays clean.
3. **Run on the same eval set.** All 5 iters use the same 2200-chip
   11-class eval set. New eval sets are introduced by new iter
   numbers; old numbers are never silently replaced.
4. **Capture parquet artefacts.** `results_matrix.parquet`,
   `per_class_metrics.parquet`, `confusion_11class.parquet`,
   `errors.parquet`, plus `errors/<cell>/<error_type>/*.png`
   (capped at 200 per type).
5. **Write iter log.** `docs/chip-multilabel/iters/iter_NN_<topic>.md`
   summarising prior result → hypothesis → change → outcome →
   insight → next hypothesis. **Append-only**.
6. **Update memory.** `chip_multilabel/notes.md` gets a new top-of-file
   entry. Cross-iter best timeline lives in `02_results.md`.

## 8.4 Agent automation

The work is split across four specialised agents, all under the
team-lead's coordination:

- **`paper-logger`** — writes the per-iter `iter_NN_*.md` log and
  updates `02_results.md` / `03_ablations.md` / `04_error_analysis.md`
  with new numbers. Runs after each iter. Read-only on
  `chip_multilabel/` source.
- **`paper-narrator`** (this agent) — composes the paper-grade
  manuscript in `docs/chip-multilabel/paper/`. Runs once iter 1–5
  numbers stabilise. Read-only on logger artefacts and source.
- **`error-analyst`** — examines `errors/<cell>/<error_type>/*.png`
  with chip-level visualisation, surfaces patterns that explain
  numerical regressions. Used in iter 2 (errors_review_T0__I7.md).
- **`chip-multilabel-runner`** — dispatches the actual training and
  evaluation jobs sequentially. Honours the GPU=1-job rule and the
  append-only directory rule.

Coordination is via a typed task system: each agent claims tasks,
marks them in-progress, and reports completion. blockedBy
dependencies between tasks are honoured (a logger task cannot start
until the runner task completes).

## 8.5 Coordinate-descent sweep design

When sweeping multiple hyperparameters (Phase A1: LS; Phase A2: LR;
Phase A3: epochs), we sweep one axis at a time and lock the others
at their current best values. This is *not* full grid search —
intentionally — because the budget for grid search is prohibitive
on a single GPU.

The risk of coordinate descent is missing axis-interaction sweet
spots (e.g. LS=0.30 + LR=5e-5 might beat LS=0.20 + LR=1e-4 even if
both axis-aligned sweeps say otherwise). We accept this risk because
(a) the LS curve is sharp enough that interactions seem unlikely
to flip the optimum, and (b) Phase F (best-known-method combination)
will revisit cross-axis interactions on the top cells.

## 8.6 What gets written down

Three types of artefact are saved per iter:

- **Numbers.** `results_matrix.parquet` (one row per cell), per-class
  metrics, confusion matrix, error rows. All in parquet, dump-able to
  CSV with `compare_runs.py`.
- **Chips.** Up to 200 PNG thumbnails per `(cell, error_type)`. Used
  by error-analyst.
- **Prose.** The iter log. The single source of truth for the *why*.

The contract is that any future agent (or human) can reproduce a
result by running `chip_multilabel/run_phase_*.py` with the
hyperparameters in the iter log, and the resulting
`results_matrix.parquet` should match the iter log's numbers to 4
decimals.

## 8.7 What does *not* go in the protocol

- **No early stopping based on val accuracy.** Single-label val acc
  is a poor selector for multi-label macro-F1 (§5.5 bullet 4). We
  always run the full epoch budget.
- **No model selection across runs based on the eval set.** The eval
  set is held out from all training and tuning decisions. Best-cell
  choice on eval is a *reporting* convention, not a model-selection
  signal.
- **No agent self-modifies the protocol.** Changes to the protocol
  (e.g. relaxing the GPU=1 rule, allowing TTA on a new task)
  require an explicit user directive recorded in CLAUDE.md.
