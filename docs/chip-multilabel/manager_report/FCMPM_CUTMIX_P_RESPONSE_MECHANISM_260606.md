# FCMPM cutmix_p Response Mechanism

updated: 2026-06-06

## Scope

This note explains why `cutmix_p` is expected to show an optimum basin instead of a monotonic curve.

Current fixed condition:

| field | value |
|---|---|
| model | T7 |
| LS | 0.295 |
| n_groups | 3 |
| grid | 9x9 |
| cmp | 1.0 |
| A/B target | 1.00 / 1.00 |
| mpos | 0.65 |
| train cap | 200/class |
| main eval cap | 2000/class |

All final claims must use multi-dataset and multi-seed mean/variance, not a single frozen-original row.

## Current Queue Policy

`cutmix_p=0.40` is only a low-exposure control. The proof queue should prioritize the observed optimum basin.

| priority | cutmix_p | role |
|---:|---:|---|
| 1 | 0.55 | candidate basin |
| 2 | 0.60 | candidate basin |
| 3 | 0.65 | candidate basin |
| 4 | 0.575 | fine point |
| 5 | 0.625 | fine point |
| 6 | 0.50 | baseline |
| 7 | 0.70 | high-exposure boundary |
| 8 | 0.80 | high-exposure stress |
| 9 | 1.00 | always-FCMPM stress |
| 10 | 0.90 | high-exposure stress |
| 11 | 0.40 | low-exposure control |

## Mechanism Hypothesis

`cutmix_p` controls how often FCMPM synthetic combo samples appear during training.

Let `p` be `cutmix_p`.

| p region | expected effect | probability signature |
|---|---|---|
| too low | insufficient combo exposure | weak combo `min_pos`, lower bit_F1 |
| optimum basin | enough combo exposure without overwhelming real/negative anchors | combo `min_pos` rises while Normal/OOD `max_prob` stays controlled |
| too high | synthetic distribution dominates real single/negative anchors | OOD/Normal tail rises; seed/dataset variance increases |

The expected response is therefore not "larger p is always better".
It is closer to a concave or peaked response:

```text
score(p) = bit_F1 - lambda * FAR + mu * gap - nu * seed_variance
```

The peak is where combo learning improves faster than OOD tail leakage.
Past the peak, extra FCMPM exposure keeps POS confidence high but increases synthetic bias and tail risk.

## Why Performance Rises Before The Peak

At low `p`, the model sees too few valid two-defect compositions. The single-defect decision boundary is learned, but weak combo classes such as `bank_boundary+scratch` or `fork+scratch` have low `min_pos`.

Raising `p` increases positive evidence for co-occurrence. This usually:

- raises weak combo `min_pos`;
- improves per-bit F1 for the weak partner bit;
- improves the global POS/NEG probability gap when NEG tails remain unchanged.

## Why Performance Falls After The Peak

At high `p`, FCMPM samples become too frequent relative to real single and negative anchors.

This can create two failure modes:

| failure mode | symptom |
|---|---|
| synthetic over-generalization | OOD classes receive moderately high defect-bit probability |
| negative-anchor undertraining | Normal/Invalid max probability crosses or approaches weak combo `min_pos` |

In reports this appears as:

```text
worst POS min  ~= weak combo bit
worst NEG max ~= OOD/Normal/Invalid tail bit
gap           = worst POS min - worst NEG max
```

If `gap` shrinks or becomes negative, high bit_F1 alone is not enough. The condition is not robust.

## Current Evidence Direction

Early frozen-original rows show that `p=0.55~0.65` is more promising than `p=0.40`, but single-dataset evidence is not enough.

The active proof target is:

```text
dataset x seed x p
```

For each p value, report:

| metric | reason |
|---|---|
| bit_F1 mean/std | average recall/precision quality |
| FAR mean/max | tail safety and worst-case risk |
| gap mean/std | probability separation stability |
| worst POS min | weak combo bottleneck |
| worst NEG max | OOD/Normal/Invalid tail bottleneck |

Promotion rule:

```text
promote only if high bit_F1, low FAR max, positive stable gap,
and no repeated seed/dataset tail collapse.
```

