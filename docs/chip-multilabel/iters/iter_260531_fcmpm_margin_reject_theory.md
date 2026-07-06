# Iter 260531 - FCM-PM + Margin Checkpoint + NB Reject Theory Note

## Claim

The method is not only a recipe sweep. It can be framed as a gap-maximizing
multi-label anomaly classifier:

1. FCM-PM constructs vicinal positive samples for unseen or rare label
   combinations.
2. Margin-based checkpoint selection chooses the model with the largest
   separation between weak positive bits and the strongest negative tail.
3. NB reject converts the remaining ambiguous probability region into a
   calibrated abstention decision, using train/validation statistics only.

The core evidence metric is:

```text
global_gap = min_{positive classes c, positive bits k} P_theta(y_k=1 | x in c)
           - max_{negative classes c, bits k} P_theta(y_k=1 | x in c)
```

A useful model has high bit_F1 and low FAR, but a robust model must also keep
`global_gap > 0` across train/eval/dataset shifts.

## Model View

For a chip image `x`, the network estimates four Bernoulli logits:

```text
z_theta(x) in R^4
p_theta,k(x) = sigmoid(z_theta,k(x))
```

The decision problem is not a plain closed-set classification problem. It is a
multi-label recognition problem with two asymmetric errors:

- false negative on a real defect combination: weak `min_pos`
- false positive on Normal/Invalid/OOD: high `max_neg`

Therefore the practical objective is not just average BCE. It is worst-case
separation:

```text
maximize  min_pos(theta) - max_neg(theta)
subject to bit_F1(theta) high and FAR(theta) near zero
```

## Why FCM-PM Helps

Single-class training data gives strong supervision for four axes, but it does
not fully define how the logits should behave on combined defects. FCM-PM adds
structured vicinal samples:

```text
x_mix = M * x_a + (1 - M) * x_b
y_mix = y_a + cmp * y_b_complement
```

where `M` is a cut/mask pattern and `cmp` controls how strongly the complement
label is trusted.

This changes the learned boundary in the combo regions. If `cmp` is too low,
the model sees combo evidence but receives weak labels, so combo `min_pos`
stays near the threshold. If `cmp` is too high, synthetic mixed images are
treated as fully real combinations, which can broaden positive support into
Normal/OOD textures and raise `max_neg`.

The useful region is where synthetic combos are positive enough to lift weak
combo bits, but not so positive that every scratch-like or boundary-like
negative pattern becomes a positive tail. Current evidence puts that region
near:

```text
LS ~= 0.2975
g = 3
cmp ~= 0.6975
cutmix_p ~= 0.2475
pair_bias = fork,scratch:2
mask_pos_target = 0.65
```

## Why Margin Checkpoint Selection Matters

Validation F1 can select a model with good average performance but poor tail
separation. For this task, the failure is often class-local:

- one combo class has a weak positive bit
- one negative class has one bit with high probability

Margin checkpoint selection should score checkpoints by:

```text
score(theta) =
    min_pos_val(theta)
  - max_neg_val(theta)
  + lambda_f1 * bit_F1_val(theta)
  - lambda_far * FAR_val(theta)
```

The important point is that checkpoint selection must prefer the model whose
worst positive bit is safely above the strongest negative bit, not the model
with the best mean confidence.

## Why NB Reject Is a Natural Final Layer

Even after FCM-PM and margin selection, the boundary can contain ambiguous
chips. NB reject should not change the classifier by looking at eval labels.
It should learn fixed distributions from train/validation predictions:

```text
P(p_k | y_k = 1),  P(p_k | y_k = 0)
```

Then prediction accepts a bit only when its posterior odds are sufficiently
positive; otherwise it rejects the chip or bit:

```text
accept positive if log P(p | pos) - log P(p | neg) >= tau_pos
accept negative if log P(p | neg) - log P(p | pos) >= tau_neg
reject otherwise
```

This makes the final evaluation a coverage-risk result:

```text
coverage = fraction not rejected
risk     = error rate among accepted samples
FAR      = false positive rate among accepted negative/OOD samples
```

The reject rule is valid only if `tau_pos`, `tau_neg`, and the probability
models are fixed on train/validation before eval.

## Current Evidence

On `frozen_iter116J_orig814_v15direct_n2000`:

```text
train_root = E:/data/images/classification_chips_iter116J_orig814_260529
eval_root  = E:/data/images/chip_multilabel_v15direct_n2000

adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10
eval bit_F1 = 0.9965
eval Total FAR = 0.02%
worst_pos_min = 0.573 fk @ fork+scratch
worst_neg_max = 0.373 sc @ Normal
global_gap = +0.200
```

This supports the mechanism: weak combo positives are above 0.5 while the
strongest Normal/OOD bit remains below 0.5.

## Required External Evidence

The remaining proof is cross-dataset repeatability:

1. frozen original old eval
2. frozen original v15direct_n2000
3. frozen_original canonical skill dataset
4. generated gapstress seed31
5. generated gapstress seed97
6. at least one external or independently generated wafer/chip dataset

For each dataset, report:

```text
train_root, eval_root
bit_F1, NI_FAR, OOD_FAR, Total_FAR
train/eval pos_prob, neg_prob
worst_pos_min, worst_neg_max, global_gap
per-class 4-bit probability table
coverage-risk if NB reject is enabled
```

If the same recipe family keeps `bit_F1 >= 0.99`, `FAR ~= 0`, and positive
`global_gap` across these datasets, the idea is strong enough to become the
central method section.
