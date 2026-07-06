# 2026-05-24 cron #983 — n200 frozen-match retrain closes github gap

Appended **§7.14.3** to `07_discussion.md`.

## What landed
- n200 frozen-match retrain (200/class × 4 = 800 chips, exact frozen pool size)
- T7 iter116J recipe, 10ep, val_margin ep5 = 0.7011 (> frozen 0.6829), 629s
- I10: **bit-F1 0.9793 / Total FAR 0.45%**
- Per-bit ≥ 0.96: bb=0.9901, fk=0.9702, sc=0.9646, sr=0.9923

## WHY this matters
Closes audit chain: §7.14 (hazards) → §7.14.1 (exact-record) → §7.14.2 (fresh clone n400) → **§7.14.3 (n400→n200 match)**. Last variable (training-pool cardinality) isolated; residual −0.0125 F1 / +0.45% FAR attributable to cuDNN nondeterminism across machine boundary, not pipeline drift.

## Honest reproducibility band
Frozen 0.9918 = lucky-seed instance of nondeterministic recipe. mega_matrix path delivers **0.97-0.98 bit-F1 at < 1% Total FAR** consistently. Practitioners cloning bundle should expect this band, not the peak.

## Framing constraint honored
Single SOTA only — no ensemble, no KD reference in §7.14.3.
