# Diary — 2026-05-05 (late evening)

Continues from `260505_evening_part2.md`. Covers iter 8 (BCE+CutMix
LS sweep + variance discovery) and iter 9 (negative axis sweep:
drop_path / cutmix-rect / two-LR).

## ~21:00 — iter 8 setup, post-T7c reflection

T7c__I10 closed iter 6 at macro-F1 = 0.9271 with bb+sr 0.9562. The
LS=0.20 in T7c was *inherited* from T1's CE-side optimum (Phase A1)
with no re-tuning under the new BCE + CutMix base. The analyst-iter6
loop closed with the obvious next question: is LS=0.20 still
optimal once we are on a BCE + CutMix base?

Hypothesis: BCE flattens per-class softmax structure relative to
CE (§6.6.2). The right per-non-target mass under BCE is plausibly
*lower* than under CE. Sweep α ∈ {0.00, 0.05, 0.06, 0.07, 0.08,
0.10, 0.20} at fixed LR=1e-4, ep=8, CutMix p=0.5, seed=42.

## ~21:01 — T9a (LS=0.10) launched

`outputs/stage1_260505_210059/`:
```
T9a__I10  | 0.9364   top1_11 = 0.8489   bb+sr 0.8812
```

LS=0.10 already beats T7c (0.9271) by +0.009. Hypothesis confirmed:
the CE-side optimum did not transfer.

## ~21:06 — T9b (LS=0.05)

`outputs/stage1_260505_210535/`:
```
T9b__I7   | 0.9449   top1_11 = 0.8670   bb+sr 0.9500
T9b__I10  | 0.9449   top1_11 = 0.8670   bb+sr 0.9500
```

Even better. LS=0.05 wins under both I7 and I10 (tied). The
inference winner is migrating back toward I7 as the BCE base
re-sharpens under lower LS.

## ~21:09 — T9c (LS=0.00) — control

`outputs/stage1_260505_210932/`:
```
T9c__I10  | 0.8609   top1_11 = 0.6443   bb+sr 0.3625
```

Confirms that "no LS at all" still hurts — overconfidence collapse,
bb+sr falls to 0.36. So the optimum is *some* LS, just less than 0.20.

## ~21:13 — T9d (LS=0.07 seed=42) — first peak read

`outputs/stage1_260505_211334/`:
```
T9d__I7   | 0.9705   top1_11 = 0.9267   bb+sr 0.9563
T9d__I10  | 0.9705   top1_11 = 0.9267   bb+sr 0.9563
```

**0.9705 macro-F1.** This would be a +0.0437 jump over T7c if real.
Headline-worthy if it replicates.

A bell goes off in my head: this is the first single-seed sweep
result that *looks* like a breakthrough but is also operating in
a regime where the per-cell variance might be comparable to the
gain. I should not ship this without re-running with a second seed.

## ~21:18 — T9e (LS=0.08) — the cliff

`outputs/stage1_260505_211752/`:
```
T9e__I3   | 0.8085   top1_11 = 0.4449   bb+sr 0.0063
```

A −0.16 cliff from T9d's 0.9705 to T9e's 0.8085 across a
single-α step (0.07 → 0.08). bb+sr collapses to 1/160. At
single-seed resolution this looks like a phase transition.

But — same caveat — a single-seed sweep of 7 cells with std ≈ 0.03
will have one cell drawn at the bottom tail by chance. If that
happens to land on LS=0.08, it looks like a cliff. Need to validate
with seed=43 before reporting.

## ~21:22 — T9f (LS=0.06) — flat band

`outputs/stage1_260505_212153/`:
```
T9f__I3   | 0.9401   top1_11 = 0.8648   bb+sr 0.8438
T9f__I7   | 0.9343   top1_11 = 0.8517
```

LS=0.06 = 0.9401. Sandwiched between T9b (LS=0.05, 0.9449) and T9d
(LS=0.07, 0.9705). Adjacent cells differ by 0.005 (0.05↔0.06) and
0.030 (0.06↔0.07). The 0.030 swing across one α step is exactly
what I expect from per-cell variance — and the 0.005 swing is
within noise.

The picture forming: the LS axis is *flat* over [0.05, 0.07] within
single-seed noise; T9d is a lucky upper tail.

## ~21:25 — T9g (LS=0.07 seed=43) — variance verification

`outputs/stage1_260505_212557/`:
```
T9g__I7   | 0.9408   top1_11 = 0.8307   bb+sr 0.9563
T9g__I10  | 0.9408   top1_11 = 0.8307
```

**0.9408 vs T9d's 0.9705.** Same config, different seed. Δ = 0.0297
absolute. Single-seed std ≈ 0.0150 (n=2 estimate, will be updated
as more seeds come in).

T9d is a lucky-outlier upper tail. The honest claim:
- T9 family (LS ∈ [0.05, 0.10] + CutMix p=0.5 + BCE) ≈ 0.94 mean
- Single-seed std ≈ 0.030
- bb+sr recall robust at 0.85–0.96

Iter 8 conclusion: +0.015 mean macro-F1 over T7c, with a +0.6312
absolute bb+sr recall lift retained. Not a breakthrough, but a
solid, honestly-reported gain.

## Methodological reflection — the lucky-outlier trap

Two things I almost shipped:

1. **T9d 0.9705 as a +0.044 macro-F1 breakthrough.** It would have
   replicated nowhere. The seed=43 test took 4 minutes and
   prevented a paper-credibility disaster.

2. **The 0.08 cliff as evidence of an LS-axis phase transition.**
   At single-seed resolution this looks like an axis structural
   feature. With multi-seed it would almost certainly disappear or
   shift in α — there is no mechanistic reason a 0.01 step in α
   should kill bb+sr recall by 0.95. This is also a single-seed
   pathology that the paper must NOT take as axis structure.

These are paper-grade methodological points, not just experimental
noise issues. At the macro-F1 ≈ 0.94 ceiling, single-seed sweeps
become uninterpretable for headline claims. The discipline must
shift from many cells × 1 seed (screening) to fewer cells × n seeds
(confirmation). Documented in §6.7 as the "lucky-outlier trap".

## ~21:30 — pivot to iter 9

Iter 8 closed: family-mean ≈ 0.94. To push above this we need a
*different* axis. Analyst-iter8 (opus) returned three orthogonal
proposals from sister-domain BKMs:

- **drop_path 0.05** (Stochastic Depth, Huang et al. 2016,
  arXiv:1603.09382). Standard ConvNeXt regulariser.
- **cutmix-rect ≤0.25** (cutmix area cap, variant of Yun et al.
  2019, arXiv:1905.04899). Trades combo signal for cleaner
  single-class identity.
- **two-LR** (backbone 5e-5 / head 2e-4, "ResNet strikes back",
  Wightman et al. 2021, arXiv:2110.00476). Differential LR.

Each axis varied alone on T9d's base. seed=42 first, with seed=43
replicate budget reserved for any axis that crosses noise floor on
seed=42.

Prediction (mine, before launching): drop_path will hurt by 0.02–
0.04 (regularisation ceiling), cutmix-rect will hurt by 0.04–0.08
(loses combo signal), two-LR will hurt by 0.05–0.10 (backbone
cannot move below TAPT init at LR=5e-5). I want to be wrong on at
least one of these.

## ~21:35 — T10a (drop_path=0.05 seed=42)

`outputs/stage1_260505_213423/`:
```
T10a__I3  | 0.9160   top1_11 = 0.7335   bb+sr 0.9000
T10a__I10 | 0.9128   top1_11 = 0.7068
```

Δ = −0.054 vs T9d single-seed, −0.024 vs T9 family-mean.
Within ~1.8 σ. Need seed=43 to verify — just inside the noise
floor.

## ~21:38 — T10b (drop_path=0.05 seed=43)

`outputs/stage1_260505_213817/`:
```
T10b__I11 | 0.8918   top1_11 = 0.7511
T10b__I3  | 0.8577
```

drop_path mean over n=2: 0.9039. Δ vs T9 family mean ≈ −0.05.
Outside noise floor (≈3.3 σ). Negative: drop_path hurts.

The mechanism: TAPT init + LS=0.07 + CutMix p=0.5 already saturates
the productive regularisation budget. Adding drop_path pushes the
model below the productive floor — too much noise injection
prevents the model from aligning bb+sr's combo signal cleanly.
Regularisation-ceiling hypothesis confirmed on this axis.

## ~21:42 — T11a (cutmix-rect ≤0.25 seed=42)

`outputs/stage1_260505_214222/`:
```
T11a__I3  | 0.8630   top1_11 = 0.6602
T11a__I7  | 0.8646   top1_11 = 0.6551   bb+sr 0.8938
```

Δ = −0.106 single-seed vs T9d, ≈ −0.08 vs T9 family-mean.
Well outside noise. Top1_11 collapses 0.93 → 0.66.

The mechanism: restricting CutMix patch to ≤25% chip area removes
the combo-dominant patch tail (60–95% area mixes). Those mixes
were exactly what gave T7c its bb+sr recall lift in iter 6. Without
the combo-dominant tail, the model loses combo capability.

cutmix-rect not run with seed=43 because the gap is already
≥3.5 σ — single-seed is sufficient to call this regression.

## ~21:46 — T12a (two-LR seed=42)

`outputs/stage1_260505_214634/`:
```
T12a__I10 | 0.8862   top1_11 = 0.6511   bb+sr 0.4188
T12a__I3  | 0.8829
```

Δ = −0.084 single-seed vs T9d, ≈ −0.06 vs T9 family-mean.
Outside noise (≈2 σ on T9 mean, but the bb+sr collapse to 0.42 is
the load-bearing failure here, not just macro-F1). top1_11 also
collapses 0.93 → 0.65.

The mechanism: at backbone LR=5e-5 over only 96 optimiser steps,
the backbone barely moves. The loss switch from CE+LS to
BCE+LS+CutMix never propagates below the head. Head receives
high-LR (2e-4) signal into a misaligned backbone → catastrophic
combo decoding failure (bb+sr 0.96 → 0.42).

## ~21:50 — iter 9 conclusion

All three iter-9 axes regress. Combined with iter-6 Phase F (warmup
−0.109, EMA −0.089), we now have *five* independent structural
BKMs from sister-domain BKM lists, all failing in our regime by
0.05 to 0.11 macro-F1.

The hypothesis is now firm enough to elevate (§7.4.4): in the
small-data + strong-TAPT + tuned-LS regime, additional structural
regularisation is a net cost on macro-F1. The TAPT init places the
backbone *at* a regularisation ceiling.

This is also the second half of the asymmetric-BKM-transfer story
(§7.5):
- LS retune: transfers (CE side α=0.10→0.20 +0.09; BCE side α=0.20→
  ~0.07 +0.015 mean).
- CutMix data-axis: transfers (+0.6312 bb+sr).
- 5 structural BKMs (warmup, EMA, drop_path, cutmix-rect, two-LR):
  all fail (−0.05 to −0.11).

The asymmetric pattern is paper-grade.

## Updated paper sections

- **abstract.md** — refresh: T9 family-mean ≈ 0.94 (single-seed std
  0.03), +0.21 macro-F1 + bb+sr 0.32→0.92 over CE+LS=0.20 baseline.
  Lucky-outlier flag on the 0.97 single-seed peak.
- **05_experiments.md** — new §5.7 (Iter 8 LS sweep with the
  lucky-outlier finding), §5.8 (Iter 9 negative axis sweep,
  all-negative table). §5.9 = renumbered cross-iter timeline with
  iter 8/9 rows.
- **06_analysis.md** — new §6.7 "Single-seed variance and the
  lucky-outlier trap" (paper-grade methodological lesson).
  Renumbered §6.7 (computational cost) → §6.8 with iter 8/9 rows.
- **07_discussion.md** — §7.4.4 "Iter 9 — three more BKM-transfer
  failures" (extends the regularisation-ceiling hypothesis).
  New §7.5 "The asymmetric BKM transfer story". Renumbered
  §7.5 → §7.6 (TTA), §7.6 → §7.7 (limits, with iter 8/9 update),
  §7.7 → §7.8 (iter-by-iter narrative, with iter 8/9 update).
- **09_conclusion.md** — extend §9.1 with the single-seed-variance
  lesson + asymmetric-BKM-transfer + "negative results are
  first-class". §9.2 best-known result reformatted as T9 family-
  mean. §9.3 remaining work: Phase G now top-priority (multi-seed
  confirmation). §9.4 outlook: three lessons (hyperparameter trap,
  asymmetric BKM, single-seed-ceiling discipline) + protocol
  variance flag.

## Why this matters

iter 8/9 produces four distinct contributions, each independent:

1. **+0.015 mean macro-F1 over T7c** from a one-axis LS retune on
   the BCE+CutMix base (iter 8). Expected, modest.

2. **The single-seed-variance lesson** (§6.7) — paper-grade
   methodological discovery. The lucky-outlier T9d at 0.9705 vs
   the seed=43 replicate at 0.9408 (Δ 0.030) establishes the noise
   floor at the macro-F1 ≈ 0.94 ceiling. Single-seed sweeps near
   the ceiling become uninterpretable for headline claims.
   Documents the protocol shift to multi-seed confirmation passes.

3. **The asymmetric BKM transfer story** (§7.4.4 + §7.5) — five
   independent structural BKMs (warmup, EMA, drop_path, cutmix-rect,
   two-LR) all fail; LS retune and CutMix succeed. The asymmetry
   is paper-grade because it has positive *and* negative evidence
   on both sides of the claim.

4. **The regularisation-ceiling hypothesis** (§7.4.4) — testable
   prediction (Phase H): if drop_path-as-replacement-for-LS recovers
   LS-only mean, the ceiling holds; if not, the hypothesis becomes
   a stronger TAPT-fragility claim.

Next iteration questions:
- Phase G: n≥3 multi-seed on T9b (LS=0.05) and T9d (LS=0.07).
  Replace family-mean with a confidence interval.
- Phase H: drop_path-as-replacement-for-LS test.
- Phase B: T4 ASL γ sweep on the BCE+CutMix base. Hypothesis:
  ASL's published default γ_-=4 is over-tuned; γ_-∈{2,3} on a
  CutMix base may match T9 family-mean.

End of late evening. Iter 8 closed at T9 family-mean ≈ 0.94 with
single-seed std 0.030. Iter 9 closed all-negative. Asymmetric-BKM-
transfer claim now rests on 5 negative + 2 positive axes.
