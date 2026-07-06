# 260517 — chain v6+v7 progression narrative appended to §5

Target: `docs/chip-multilabel/paper/05_experiments.md`, new section
"Chain v6+v7 progression narrative" appended after `_Raw data: tables/all_runs_n2000.csv (rows v6,1..4)._`.

## Story arc (one paragraph each, ~500-800 words total)

1. **Setup** — iter116J carried as headline 9 days; two unresolved
   questions (seed robustness + KD viability) blocked paper claim.
2. **Chain v5** — 4 corner sweeps all under-shot iter116J; envelope
   0.9049 ± 0.0464 placed SOTA at +1.9σ. LS=0.30→0.25 atomic ablation
   cost -0.0806. Working hypothesis: seed=1 was an outlier basin.
3. **Chain v6** — s=11 ep1 (under-trained), s=23 ep9 (fork-weak
   collapse), s=77 ep8 (+0.0038 bit_F1 micro-win at +0.76 pp FAR).
   Spread ±0.21 dominated by **ckpt-selection criterion**, not seed.
4. **KD collapse fix (Phase 4)** — `--kd-skip-on-cutmix` resolved
   7th KD attempt to bit_F1 0.9265 / FAR 0.00% (first non-collapse).
   Acts as clean regulariser; I3 FAR drops from 87-100% → 3.75%.
5. **★ Ensemble headline** — `vote_majority_bits` of (iter116J s=1,
   s=77 micro-win, KD_v7) → **bit_F1 0.9941 / Total FAR 0.00%** at I10
   (+0.0014 vs SOTA, no FAR cost). `vote_union_bits` 0.9965 (+0.0038,
   +0.76 pp FAR trade-off).
6. **Paper contribution** — diversity (seed + KD-axis) converts
   within-recipe variance into usable signal. Hinton 1503.02531 +
   snapshot-ensemble 1704.00109 inline cited.
7. **Negative results** — vote_unanimous 0.9495 (bb+scratch 0.6669),
   vote_intersection_bits 0.9735 (bb+scratch 0.8518). Unanimity and
   AND-aggregation too strict for combo recovery.

## Sources cross-referenced

- `docs/chip-multilabel/iters/iter_v5_0[1-4]*.md`
- `docs/chip-multilabel/iters/iter_v6_0[1-4]*.md`
- `docs/chip-multilabel/02_results.md` (2026-05-17 update section)
- `outputs/_ensemble_v7_5mode.json` (5-mode ensemble eval, 18640 chips)

## Headline numbers used (4-decimal)

| run                                  | I10 bit_F1 | Total FAR |
|--------------------------------------|------------|-----------|
| iter116J SOTA (s=1, val_f1, ep6)     |     0.9927 |      0.00 |
| s=11 (margin_max, ep1)               |     0.8456 |     70.53 |
| s=23 (margin_max, ep9)               |     0.4738 |     66.44 |
| s=77 (margin_max, ep8)               |     0.9786 |      0.76 |
| KD_v7 skipcutmix (ep7)               |     0.9265 |      0.00 |
| vote_majority_bits (3-model)         |     0.9941 |      0.00 |
| vote_union_bits (3-model)            |     0.9965 |      0.76 |
| vote_unanimous (3-model)             |     0.9495 |      0.00 |
| vote_intersection_bits (3-model)     |     0.9735 |      0.00 |
| vote_majority (3-model variant cell) |     0.9936 |      0.00 |

## Why one more section vs editing existing 종합

Existing chain v6 종합 already records all four individual phase
results. The new section's job is the **joint reading across v5 + v6 +
v7**: the diversity argument and the ensemble headline (first cell to
exceed iter116J SOTA without FAR cost) only emerges when the three
non-degenerate students are read together. WHY append rather than
amend: keeps single-iter reports immutable (260512 absolute rule) and
lets the paper-side narrative compose across iters.

## What's not in this entry

- Multi-teacher KD (Hinton classical, T-a from chain v6 종합
  next-hypothesis) — pending future iter.
- val_f1 re-selection of s=11/s=23/s=77 (T-b) — pending future iter.
- Ensemble vote-threshold ablation beyond 5 modes — pending.
