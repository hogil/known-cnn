# iter116J_nopair_val 01 — iter116J_nopair_10ep_s1

- Chain: `iter116J_nopair_val`
- Iter: 1
- Tag: `iter116J_nopair_10ep_s1`
- TS: `20260522_135641` (eval `260522_142722` best / `260522_142725` final)
- Source: `outputs/iter116J_nopair_validation/20260522_135641_T7_iter116J_nopair_10ep_s1/eval_n2000_{best,final}/eval_*/preds_chip.parquet`
- Rows per ckpt: 74 560

## Result (n2000, 4 variants × best ckpt)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Note                |
|---------|--------|--------|---------|-----------|---------------------|
| I3      | 0.8816 | 100.00 |  100.00 |    100.00 | raw-thr collapse    |
| I7      | 0.8813 | 100.00 |  100.00 |    100.00 | raw-thr collapse    |
| I10     | 0.9237 |   0.50 |    8.44 |      2.42 | val_margin selector |
| I13     | 0.8394 |   0.55 |    8.28 |      2.42 | conservative gate   |
```

best ckpt and final ckpt produced **identical** per-variant numbers (4/4 cells match) — same effective weights at ep10 (no late-epoch drift across the val_margin window).

## Hparam (vs iter116J past best)

- T7 nopair, 10 ep, seed 1 (validation repro of the iter116J recipe with the nopair fork)
- vs iter116J past best (0.9927 / 0.00 %): **delta -0.0690 / +2.42 pp** on I10 best cell

## Insight

- 10-ep nopair repro under-shoots iter116J (-0.069 bit_F1) and breaks the 0 % FAR claim (+2.42 pp). OOD-FAR 8.44 % is the dominant leak; NI side stays clean (0.50 %).
- I3 / I7 (raw-threshold) **fully collapse** (100 % FAR) — consistent with the v5 / v6 pattern where unselected inference cells leak everywhere in short-epoch runs without the val_margin selector.
- I10 vs I13 — I10 holds +0.0843 bit_F1 for the same 2.42 % Total FAR; I13's tighter gate sacrifices bit_F1 without buying FAR.
- best == final at all 4 cells -> the val_margin selector saturated by ep10 in this nopair recipe. No headroom from longer training under this seed / nopair config (cf. §5.54 pair-mask plateau which requires final to recover).

Champion frozen: E22 (0.9956 / 0 %) ensemble, iter116J (0.9927 / 0 %) single-model. This run posts no new high.
