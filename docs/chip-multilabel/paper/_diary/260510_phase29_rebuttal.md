# Phase 29 — n = 50 → n = 200 rebuttal

_2026-05-10. Diary entry. Source: Phase 27 v15direct n = 200
re-evaluation; supersedes Phase 24 / 26 paper headline._

## What happened

The paper's iter-39 headline (0.9992 / 0 %) was measured at
v15direct n = 50 / class (≈ 770 chips). Phase 27 re-runs the
candidate 4-bag configurations on a **4 × larger eval set**
(n = 200 / class, 3 080 chips) drawn from the same v15direct
generator with disjoint seeds.

## Findings

| 4-bag                                                    | n = 50 | **n = 200** | Δ bit-F1 |
|----------------------------------------------------------|-------:|------------:|---------:|
| ★ pure-hard MAIN (24_LS030_seed42 + 26 B + 26 D + 26 H)  | 0.9992 |  **0.9955** |  − 0.0037 |
| pure-hard alt (26 B + 21 H + 26 D + 24_LS030_seed42)     | 0.9945 |      0.9953 |  + 0.0008 |
| Hard + KD (24_LS030 + 26 B + 26 H + 33 D)                | 0.9984 |      0.9953 |  − 0.0031 |
| iter-37 KD + asym (26 B + 26 D + 33 A + 37 E)            | 0.9976 |      0.9945 |  − 0.0031 |
| seed = 7 alt (24_LS030_seed7 + 26 B + 26 D + 26 H)       | 0.9992 |      0.9959 |  − 0.0033 |

**ni_FAR.** All n = 200 PASS rows hold 0 % FAR (or 1.25 % /
4.50 % on the seed-7 alt). Single-component diagnostic at
n = 200: 24_LS030_seed42 alone best ni_FAR = 20.5 %;
24_LS030_seed7 alone best ni_FAR = 46 %. **Both fail
dual-gate solo, both PASS inside the 4-bag.**

## Thesis revisions

1. **"Pure-hard wins by + 0.0016" is FALSIFIED.** All four
   4-bag composition types land within 0.0014 v15 bit-F1 at
   n = 200 — indistinguishable from sampling noise. The
   n = 50 ordering pure-hard ≻ hard + KD ≻ KD + asym was
   real but dominated by eval-set-size variance.
2. **n = 50 was systematically over-confident** by 0.003–
   0.004 v15 bit-F1 across all configurations. Inside the
   0.99-ceiling regime, any sweep maximum at n = 50 carries
   this upward bias.
3. **4-bag-at-global-optimum survives qualitatively.** 4-bag
   still beats 5- / 6- / 14-bag at like protocol; the
   global optimum cluster sits at v15direct bit-F1 ≈ 0.995.
4. **Ensemble-from-fragility (§6.17.2) STRENGTHENS.** The
   24_LS030 single-cell example is now the cleanest paper
   instance: 21 % single-cell FAR → 0 % ensemble FAR via
   majority-vote absorption. The earlier 37 E example
   (100 % single FAR → 1.25 % ensemble FAR) is preserved
   as an asymmetric-axis case study; the 24_LS030 example
   becomes the headline because it sits inside the
   production MAIN bag.

## Files edited

- `abstract.md` — replaced final iter-39 block with Phase 27
  rebuttal; new headline 0.9955 / 0 %.
- `05_experiments.md` — added new §5.25 "Robust evaluation
  (n = 200)" with 5-row comparison; appended caveat
  pointers to §5.22 / §5.23.
- `06_analysis.md` — rewrote §6.17 thesis ("All 4-bag
  composition types converge at the eval-noise floor");
  rewrote §6.17.2 example to 24_LS030 single-vs-4-bag at
  n = 200 (21 % → 0 % absorption).
- `07_discussion.md` — replaced cost frontier table with
  n = 200 numbers; updated deployment note ("deploy any
  4-bag axis blend; majority voting absorbs single-cell
  FAR fragility").
- `09_conclusion.md` — replaced final headline 0.9992 →
  0.9955; updated per-class line bb / fk / sc / sr =
  0.9984 / 0.9881 / 0.9953 / 1.0000.

## What stays unchanged

- The 4-bag's qualitative dominance over 14-bag and the
  diversity-rank methodological recipe (§6.14).
- The §6.17.1 seed-fragility on 37 E asymmetric cell (still
  a valid asymmetric-axis case study).
- §6.18 majority-vote-vs-prob-averaging finding (n = 200
  re-eval pending; expected to hold qualitatively).

## Open follow-ups

- n = 200 re-eval of the 1× and 2× cost-frontier rows
  (KD-student, 26 B + 33 A OR-mode) — currently held at
  n = 50.
- n = 200 re-eval of §6.18 prob-averaging configurations.
- Sanity check: does the 4-bag MAIN at v14class n = 200
  still hold 1.0000 / 0 % (perfect in-distribution)?

## Headline reset

**Paper headline locked at v15direct n = 200 bit-F1 =
0.9955 / `ni_FAR = 0.00 %`** for the iter-39 4-bag
{24_LS030_seed42 + 26 B + 26 D + 26 H} at τ = 2 / 4. The
production claim shifts from "pure-hard wins" to "any
well-spread 4-bag axis blend reaches the global optimum;
ensemble robustness comes from majority voting absorbing
single-cell fragility".
