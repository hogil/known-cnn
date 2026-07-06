# Diary 260520 21:15 — §5.52 FCM-PM nopair g=4 preliminary signal

## Cron context
10min cron #436. fcm_nopair_g4 (2ep, under-converged) eval just landed. Triggered §5.52 append as preliminary-signal narrative (NOT a confirmed claim).

## New metric snapshot
| Cell | Train ep | bit_F1 | Total FAR | vs §5.51 nopair g=3 8ep |
|------|---------:|-------:|----------:|-------------------------|
| fcm_nopair g=3 s7 8ep (§5.51 ref) | 8 | 0.9943 | 11.81 % | baseline |
| fcm_nopair g=4 2ep | 2 | 0.9328 | 0.72 % | −0.0615 bit_F1 / −11.09 pp FAR |
| **Champion E22 ensemble (§5.49.7) FROZEN** | — | 0.9956 | 0.00 % | not challenged |

## Why this is preliminary (not a §4 design-axis promotion)
- N=1 per cell — no seed variance estimate.
- 2 ep (g=4) vs 8 ep (g=3) — training-budget mismatch confounds bit_F1 deficit.
- Under-fit 2ep model may produce mechanically lower FAR via blanket under-confidence rather than g-axis dilution effect.
- Single-cell ablation cannot separate g-effect from epoch-effect.

## Refined paper claim wording (conservative)
"g=4 nopair (under-converged 2ep) shows a preliminary FAR-leak reduction signal of −11.09 pp vs g=3 nopair 8ep at the cost of −0.0615 bit_F1; whether this signal survives at matched 8ep training and across seeds is the next experiment."

## Next experiment to confirm or kill the signal
- fcm_nopair g=4 at **8 ep** (matched to §5.51 g=3 baseline) — preferably ≥2 seeds.
- If FAR remains <5 % at bit_F1 ≥0.99: promote to §4 second FAR-control axis (pair-mask × g-group two-axis design space).
- If FAR climbs back to ~11 % at fair-train: section §5.52 stands as documented negative-control / failed-promotion case per §1 methodology commitment.

## Champion E22 status
**FROZEN.** §5.52 is method-section design-space probe only, not a champion challenge. E22 ensemble (bit_F1 0.9956 / 0.00 % Total FAR) remains the paper champion.

## Files touched
- `docs/chip-multilabel/paper/05_experiments.md` — appended §5.52 after §5.51.
- `docs/chip-multilabel/paper/_diary/260520_2115_section_5_52_fcm_nopair_g4_preliminary.md` — this entry.

---

## Addendum 21:25 — cron #438 cls-axis paragraph append

**New cell.** `fcm_pair_cls03` (1ep) surfaced in cron #437: bit_F1 0.8876 / Total FAR 4.55 %.

| Cell | Train ep | cls | bit_F1 | Total FAR | vs cls=0.5 ref |
|------|---------:|----:|-------:|----------:|----------------|
| fcm_pair_cls05 (iter116J ref) | 8 | 0.5 | 0.9927 | 0.00 % | baseline |
| fcm_pair_cls03 | 1 | 0.3 | 0.8876 | 4.55 % | −0.1051 bit_F1 / +4.55 pp FAR |

**Why append to §5.52 (not new section).** Same FCM-PM design-space probe family as g-axis; cls (complement label scale) is the natural second axis after g-group cardinality. Keeping both axes documented in one section preserves the "two-axis design-space probe" framing for §6 discussion.

**Confound.** N=1, 1ep vs 8ep mismatch — identical methodology issue as g=4 above. cls=0.3 may simply be under-fit; trade-off (tighter complement supervision vs bit_F1 cost) requires fair 8ep ≥2 seed confirmation before §4 promotion.

**Next experiment.** fcm_pair_cls03 at 8 ep ≥2 seeds, matched to iter116J cls=0.5 protocol. If FAR stays <2 % at bit_F1 ≥0.99 → §4 second design axis (g-group × cls). Otherwise documented negative-control.

**Champion E22 status.** Still FROZEN.
