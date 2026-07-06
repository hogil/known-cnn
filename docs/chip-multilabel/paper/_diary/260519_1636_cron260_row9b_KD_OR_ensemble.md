# cron #260 — 2026-05-19 16:36 — Row 9b KD OR ensemble added

**Trigger.** paper-recorder cron #260, 1-line addendum request.

**Change.** §5.49.9 appended to `05_experiments.md`:
- Row 9b = **{KDv7 + KDv12} OR ensemble** → **bit_F1 = 0.9930 / Total FAR = 0 %**.
- First pure-KD (no base-seed, no LS-axis member) ensemble to clear strict gate.
- Isolates T-diversity (T=2 KDv7 + T=3 KDv12) per-bit OR vote, no base seeds.

**Comparison vs §5.49.7 E22 champion.**
- E22 5-member mixed (2 base + 1 LS + 2 KD): 0.9956 / 0 % — champion.
- Row 9b 2-member KD-only OR: 0.9930 / 0 % — KD-axis-only ceiling.
- Δ = +0.0026 bit_F1 = quantified contribution of base-seed members inside E22.

**Why paper-worth.** Bounds the KD-axis-only contribution and lets §5.49.7's T-diversity claim be cited in its minimal 2-member form (no confound from base seeds), strengthening the mechanistic interpretation already in §5.49.7.

Champion table unchanged at E22 / 0.9956 / 0 %.
