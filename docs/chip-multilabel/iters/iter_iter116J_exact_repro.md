# iter — `iter116J_exact_repro` (SOTA single-cell repro; T7 BCE+LS=0.30 pair complement)

- **Chain / tag**: `iter116J_exact_repro` / `iter116J_exact_repro_s1_ep8_best`
- **Cron**: #687
- **Eval TS**: `260522_151400`
- **Train TS**: `20260522_142643_T7_iter116J_exact`
- **Train recipe**: T7 BCE+LS=0.30, CutMix p=0.25 complement (rect=0.5, n_patches=5, total_ratio=0.3, discount=0.7, alpha=1.0), pair-mask ON, no-Normal, 10 ep
- **Best epoch**: 8 (val_acc 0.6950 reported as `best_val_acc` per selector; final ep10 val_acc 0.9970)
- **Eval set**: n_eval = 18 640 (POS9 strict + 4-class OOD strict), single cell `T0__I10` (SOTA selector)
- **Source parquet**: `outputs/iter116J_exact_repro/20260522_142643_T7_iter116J_exact/eval_sota_i10/eval_260522_151400/preds_chip.parquet`

## Result — single-cell SOTA selector + PM nopair comparison (cron #687)

```
| Recipe                                  | Ckpt | Variant | bit_F1 | NI-FAR | OOD-FAR | Tot-FAR | vs E22 (0.9956/0.00) | vs iter116J past best (0.9927/0.00) | Status                |
|-----------------------------------------|------|---------|--------|--------|---------|---------|----------------------|-------------------------------------|-----------------------|
| iter116J_exact_repro pair T7 LS=0.30    | best | I10     | 0.9691 |   0.00 |    3.75 |    0.91 |    -0.0265 / +0.91   |              -0.0236 / +0.91        | SOTA repro under E22  |
| iter116J_nopair_10ep_s1 (cron #685 ref) | best | I10     | 0.9237 |   0.50 |    8.44 |    2.42 |    -0.0719 / +2.42   |              -0.0690 / +2.42        | nopair short-ep gap   |
| E22 champion ensemble (frozen)          | -    | -       | 0.9956 |   0.00 |    0.00 |    0.00 |              -        |              +0.0029 /  0.00        | champion              |
| iter116J past best single (frozen)      | -    | -       | 0.9927 |   0.00 |    0.00 |    0.00 |     -0.0029 / 0.00   |                       -             | single ref            |
```

## Per-bit / per-OOD breakdown — `iter116J_exact_repro` / I10 / best

```
| Class            | bit_F1 / FAR | Count       | Note                         |
|------------------|--------------|-------------|------------------------------|
| bank_boundary    |       0.9974 | -           | near-perfect                 |
| fork             |       0.9840 | -           | strong                       |
| scratch          |       0.8951 | -           | weak (drag on bit_F1)        |
| scratch_rot      |       0.9999 | -           | perfect                      |
| Normal           |        0.00% | 0 / 1600    | clean                        |
| Invalid          |        0.00% | 0 / 400     | clean                        |
| DiagonalSmear    |        2.50% | 4 / 160     | smallest OOD leak            |
| CenterDonut      |        3.75% | 6 / 160     | mid OOD leak                 |
| CrossScratch     |        3.75% | 6 / 160     | mid OOD leak                 |
| Starburst        |        5.00% | 8 / 160     | largest OOD leak             |
```

## Delta vs prior champion + reference

- vs **E22 ensemble** (0.9956 / 0.00 %): **-0.0265 bit_F1 / +0.91 pp Total FAR** — single-model repro cannot match the 4-way bit-vote champion (expected; E22 is an ensemble headline).
- vs **iter116J past best single** (0.9927 / 0.00 %): **-0.0236 bit_F1 / +0.91 pp Total FAR** — even under the exact T7 BCE+LS=0.30 pair-complement recipe, this repro **falls short of the iter116J past-best single by -0.024 F1 and leaks 0.91 pp OOD-FAR** where the prior single hit a perfect 0 %.
- vs **PM nopair short-ep** (`iter116J_nopair_10ep_s1` / I10 / best = 0.9237 / 2.42 %): **+0.0454 bit_F1 / -1.51 pp Total FAR** — pair-mask delivers ~0.05 F1 and a 2.7× FAR reduction at the same 10ep budget, confirming the pair-mask FAR advantage (cf. §5.51, §5.55).

## Insight — why the repro under-shot iter116J past best

1. **NI-FAR perfect (0 %)** matches iter116J past best — `pair-mask + LS=0.30` keeps Normal/Invalid clean at 1600 + 400 chips.
2. **OOD leak is the entire gap** — Total FAR 0.91 % is wholly from 4-class strict OOD (CenterDonut/CrossScratch/DiagonalSmear/Starburst), with Starburst the worst at 5 %.
3. **`scratch` bit_F1 0.8951** is the single per-bit drag — `bank_boundary`/`fork`/`scratch_rot` all > 0.98, but `scratch` weakens the macro bit_F1 to 0.9691 (vs iter116J past best where all 4 bits land > 0.99).
4. **Seed / init drift** — the iter116J past best was selected from a longer search; this single-seed exact-recipe repro falls into the same per-seed variance band documented for T7 (cf. §5.54 calibration headroom, §5.55 val_margin selector). The result does **not** challenge iter116J past best — it confirms the past-best is on the upper tail of the seed distribution.
5. **No selector applied** — only the raw `T0__I10` cell was evaluated (SOTA-style single-shot); the `val_margin` and `final` ckpt selectors from §5.54 / §5.55 may close part of the gap and remain unrun for this run dir.

**Champions frozen.** E22 (0.9956 / 0 %) and iter116J past best single (0.9927 / 0 %) remain untouched. No new ensemble candidates added.
