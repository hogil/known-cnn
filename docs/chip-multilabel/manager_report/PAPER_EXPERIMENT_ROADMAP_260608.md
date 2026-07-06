# FCMPM Paper Experiment Roadmap - 260608

## Current Main Finding

`cutmix_p=0.575` is the primary paper candidate after excluding
`frozen_original_200_snapshot`.

Measured condition:

- train/eval protocol: `train=200/class`, `eval=2000/class`
- recipe: `T7 / LS=0.295 / g=3 / grid=9x9 / cmp=1.0 / A,B=1.00,1.00 / neg=0.0 / mpos=0.65 / seed=7`
- datasets used:
  - `frozen_original`
  - `sota_gapstress_seed31_260531`
  - `sota_gapstress_seed97_260531`
  - `frozen_original_2015_candidate`
- excluded dataset:
  - `frozen_original_200_snapshot`

Result:

| p | n | F1 mean | F1 min | FAR mean | FAR max | posmin | negmax | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.575 | 4 | 0.9949 | 0.9902 | 3.35% | 10.31% | 0.753 | 0.513 | 0.240 |

Interpretation:

- `p=0.575` gives the strongest observed probability separation.
- `p=0.50` is the conservative FAR-control reference.
- `p>=0.70` is no longer a promote region; it is a tail-risk region.

## Immediate Experiment Queue

### 1. Narrow `cutmix_p` Peak Confirmation

Purpose: verify whether `p=0.575` is a real peak or a local single-seed artifact.

Hold fixed:

- `T7`
- `LS=0.295`
- `g=3`
- `grid=9x9`
- `cmp=1.0`
- `A,B=1.00,1.00`
- `neg=0.0`
- `mpos=0.65`
- `train=200/class`
- `eval=2000/class`

Run:

| axis | values |
|---|---|
| `cutmix_p` | `0.50 / 0.5375 / 0.55 / 0.5625 / 0.575 / 0.5875 / 0.60 / 0.625 / 0.65 / 0.675` |
| seed | `7 / 13 / 42 / 99` |
| dataset | `frozen_original / sota_gapstress_seed31_260531 / sota_gapstress_seed97_260531 / frozen_original_2015_candidate` |

Success condition:

- high mean bit_F1
- low mean FAR
- controlled max FAR
- positive mean gap
- low variance across seed and dataset

### 2. NB Reject Evidence

Purpose: show why threshold-only multi-label prediction fails and why 4-bit
pattern likelihood is needed.

Required reports:

- single POS probability pattern
- 2-combo POS probability pattern
- OOD/Normal probability pattern
- NB likelihood score table
- threshold-only vs NB-reject FAR/F1 comparison

Key example:

- threshold failure: OOD has one bit moderately or strongly high
- NB rescue: the full 4-bit vector does not match any valid single/combo class

### 3. Essential Ablation Table

Purpose: prove FCMPM is a method, not only a tuned parameter set.

Required rows:

| component | compare |
|---|---|
| FCM region copy | FCMPM vs Mixup |
| complement grid | FCMPM vs standard CutMix |
| pair-mask | masked vs no-mask |
| hard target | hard target vs soft/area target |
| `cutmix_p` | peaked response curve |
| grid/group | aligned grid split |
| NB reject | threshold-only vs likelihood reject |

### 4. External Dataset Transfer

Purpose: move from internal wafer evidence to publishable method evidence.

Minimum target:

- one additional public or independently generated compositional defect dataset
- same one-axis protocol
- same baselines: Mixup, CutMix, FCMPM, FCMPM+NB reject

## Paper Writing Tasks

1. Freeze terminology:
   - method name: `FCMPM`
   - model-selection metric: `probability gap`
   - rejection module: `class-conditional NB likelihood reject`
2. Write method section with equations:
   - FCMPM sample generation
   - target construction
   - probability-gap metric
   - NB likelihood and rejection threshold
3. Write result section around four claims:
   - FCMPM beats Mixup/CutMix on compositional defect learning
   - `cutmix_p` has a peaked trade-off
   - probability gap explains F1/FAR behavior better than F1 alone
   - NB reject catches OOD tails that threshold-only prediction misses
4. Add limitations:
   - snapshot dataset has severe OOD-bank-boundary tail
   - current wafer evidence is strong but external transfer is still required

## Required Absolute Paths

- `D:/project/known-cnn/docs/chip-multilabel/manager_report/FCMPM_CUTMIX_P_TRADEOFF_260608_no_snapshot.md`
- `D:/project/known-cnn/docs/chip-multilabel/manager_report/FCMPM_CUTMIX_P_TRADEOFF_260608_no_snapshot.png`
- `D:/project/known-cnn/docs/chip-multilabel/manager_report/PAPER_EXPERIMENT_ROADMAP_260608.md`
