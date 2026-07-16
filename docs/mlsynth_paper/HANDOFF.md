# Handoff — "Multi-label from single-label via content-blind synthesis"

> **Positioning correction, 2026-07-16 (FINAL — density-shift refutation; READ
> FIRST; supersedes ALL earlier die-budget / faithful-operator framing below).**
> A decisive **density-shift stress test REFUTED** the die-budget /
> faithful-operator thesis. On WM38, whole-image **max-union** (= Shin et al. 2022
> Summation Mixup under the binary encoding; `summation_mixup_shin22`/`overlay`
> both compute `np.maximum(ca, cb)`) **BEATS the partition-style FCM-PM on EVERY
> real-mix density stratum AND every mix order** (2-mix 0.855/0.830/0.805 vs
> 0.787/0.787/0.714; 3-mix 0.725 vs 0.634; 4-mix 0.655 vs 0.484), at higher bit-F1
> (0.80 vs 0.65) and lower FAR (0.010 vs 0.147). **"Over-density hurts" is FALSE.**
> **DROPPED (do not reuse):** "FCM-PM is the faithful WM38 method"; "max-union is
> distributionally mismatched / EXCLUDED / die-budget-violating"; "die-budget
> partition makes complement beat summation"; any claim to beat Shin22 on the wafer
> operator; the TV lower bound used as a *preference* argument. **On WM38 the best
> content-blind operator is summation/union (= Shin 2022 Summation Mixup); we claim
> NO operator novelty on wafer.** Density facts are kept only as a characterization
> (max-union over-dense 0.50; FCM matches real 0.29) — empirically harmless, hence
> a modeling property, NOT a performance advantage. The paper is repositioned as an
> **annotation-free, reliability-guaranteed, cross-domain FRAMEWORK**: (i)
> single→multi setting/method (zero multi-label annotation); (ii) a
> **label-fidelity / operator-match criterion** that selects the right operator per
> domain (summation/union on superposition-structured wafer/digits/audio; averaging
> on disjoint-coordinate text) and correctly selects summation on wafer; (iii) an
> **annotation-free distribution-free split-conformal FAR guarantee** (Shin22 lacks
> it) — strongest asset; (iv) an excess-risk theory as a general, one-directional
> bound (not a superiority proof); (v) cross-domain validation + honest VOC
> boundary. FCM-PM is reported honestly as an **alternative** operator (does NOT
> beat summation on wafer; useful on chip-internal ~0.99 + FAR control among weaker
> arms cutmix/mixup/FCM-no-PM). The reframed paper (DRAFT.md, latex/main.tex,
> THEORY.md) is fully reconciled to this. Evidence:
> `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` (sections
> "DECISIVE density-shift stress test" and "Conformal FAR guarantee COMPLETE").
> **Everything below this banner is HISTORICAL and superseded.**

Self-contained state of the mlsynth paper project, for continuation by another
agent/tool. The measured base state is committed to `hogil/known-cnn` on
`main`; the 2026-07-10 correctness audit is currently a local, uncommitted
change set. Paper artifacts live in
`D:/project/known-cnn/docs/mlsynth_paper/`.

## 0. Active ICLR evidence chain (2026-07-15)

The active claim is source-only compositional multi-label learning built on a
**die-budget partition insight + a generative-model-match criterion + the faithful
operator (FCM-PM) + theory + annotation-free guarantee**. The content-blind
operator is SELECTED per modality by a two-part criterion (evidence survival AND
generative-model/density match). On WM38 the real combination law is a **die-budget
partition**, so whole-image **max-union** (= Shin et al. 2022 Summation Mixup under
the binary encoding) is EXCLUDED as die-budget-violating (over-dense 0.50 vs real
0.31) despite its high raw bit-F1 (0.80, an over-dense artifact). The faithful WM38
operator is **FCM-PM** (for `g` groups, the `g` complementary full-cover FCM views
plus `g` corresponding Pair Mask views), which matches the real die budget (0.29)
and has the best F1-FAR trade-off among die-budget-faithful arms (0.654/0.147);
FCM-PM also excels on chip (~0.99). On FSD50K, exact waveform summation is the
modality-compatible operator for that (genuine-superposition) domain; FCM and
FCM-PM remain tested operators there. The WM38 FCM/FCM-PM geometry work in the
queue below now serves the primary WM38 method (finding the best faithful FCM-PM
layout), not merely a chip-vs-wafer diagnostic.

Active/pending order:

1. Finish the historical WM38 FCM/FCM-PM geometry reproduction. At the latest
   audit it had `40/48` five-seed conditions and was running FCM random
   `g=4`, `16x16`. Runner:
   `D:/project/known-cnn/scripts/run_wm38_fcm_geometry_hard_5seed.ps1`.
2. Immediately run the equal-area `balanced` replication. Its first 18
   conditions compare `random`, `random_fixed`, and `checkerboard` at
   `g2/4x4`, `g3/9x9`, and `g4/12x12` for FCM and FCM-PM over five paired
   seeds. This separates per-pair mask diversity from periodic topology before
   reusing those results in the full 48-condition factorial. Runners:
   `D:/project/known-cnn/scripts/run_wm38_layout_diversity_balanced_5seed.ps1`
   and
   `D:/project/known-cnn/scripts/run_wm38_fcm_geometry_balanced_5seed.ps1`.
   Mechanism audit:
   `D:/project/known-cnn/docs/mlsynth_paper/WM38_LAYOUT_AND_CELL_BOUNDS_AUDIT_260715.md`.
3. Strict FCM-PM negative-target `.02/.20`, synthetic held-out checkpoint
   selection, then equal-budget single-only/Mixup/CutMix/exact Shin22
   variants/UnionMixup/FCM baselines. Chain:
   `D:/project/known-cnn/scripts/start_wm38_strict_fcmpm_neg_compare_after_geometry.ps1`.
   A separate 40-row queue adds standard Beta-area CutMix with byte-identical
   hard-presence and area-soft target variants:
   `D:/project/known-cnn/scripts/run_wm38_strict_cutmix_beta_neg002_5seed.ps1`.
   The same chain then runs a preregistered method-aligned checkpoint-proxy
   sensitivity analysis while retaining the common FCM-PM NB bank. Protocol:
   `D:/project/known-cnn/docs/mlsynth_paper/WM38_PROXY_POLICY_SENSITIVITY_PROTOCOL_260715.md`.
   The executable source-role, evaluation-slice, train-budget, and proxy-
   cardinality audit is **PASS**:
   `D:/project/known-cnn/docs/mlsynth_paper/WM38_STRICT_PROTOCOL_AUDIT_260715.md`.
4. Practical real-Normal train/calibrate confirmation and NB operating curves:
   `D:/project/known-cnn/scripts/start_wm38_real_normal_after_strict.ps1`.
5. Independent prospective FSD50K source-only-to-sealed-real-multi benchmark:
   `D:/project/known-cnn/scripts/start_fsd50k_source_only_when_ready.ps1`.

The FSD50K information split, primary endpoint, safety gate, and failure rule
were frozen before model input download in
`D:/project/known-cnn/docs/mlsynth_paper/FSD50K_PROSPECTIVE_PROTOCOL_260715.md`.
The broader acceptance gates are frozen in
`D:/project/known-cnn/docs/mlsynth_paper/ICLR_EVIDENCE_GATE_260715.md`.
The runner saves per-seed sealed predictions and reports both seed-paired CIs
and crossed two-way paired CIs over model seeds and shared evaluation clips.
The executable pre-result information-split audit is **PASS**: all five
manifests have zero pairwise audio-path overlap; feature targets, five-seed
selection/Gaussian-fit/threshold roles, and all 15 waveform-cache provenance
keys agree. Report and machine-readable evidence:
`D:/project/known-cnn/docs/mlsynth_paper/FSD50K_INFORMATION_SPLIT_AUDIT_260715.md`
and
`D:/project/known-cnn/outputs/multilabel_synth/fsd50k_information_split_audit.json`.

Current FSD50K code paths:

- `D:/project/known-cnn/multilabel_synth/run_fsd50k_source_only.py`
- `D:/project/known-cnn/multilabel_synth/synthesis/fsd50k_waveform.py`
- `D:/project/known-cnn/multilabel_synth/analyze_fsd50k_source_only.py`
- `D:/project/known-cnn/multilabel_synth/analyze_fsd50k_operating_curves.py`
- `D:/project/known-cnn/multilabel_synth/audit_fsd50k_information_split.py`
- official audio/data/cache root:
  `E:/data/fsd50k_source_only_protocol`

Latest verification: all tests under
`D:/project/known-cnn/tests/multilabel_synth` pass (`121 passed`). No pending
result may be promoted into the paper until its frozen evidence gate passes.

## 1. Thesis / novelty (one sentence)

In **blind-combination domains** (where co-occurring signals combine by a known
content-blind operator — pixelwise max for inked digits and spectrograms, a
**die-budget partition** for wafer maps, coordinate averaging for bag-of-words
text), you can **train a multi-label recognizer using only single-label data** by
synthesizing multi-label examples with the domain's **true content-blind operator**
(no location/mask annotation), recover a large fraction of a fully-supervised
oracle on real mixes, and exceed an oracle that has not seen held-out combinations.
The contributions are (a) the **die-budget partition insight** (WM38 real mixing
is a partition, not a superposition: density 0.31 vs max-union 0.50) + a
**generative-model-match criterion** that selects the faithful operator (FCM
complement) over the unfaithful one (max-union); (b) **FCM-PM** as the faithful
WM38 synthesis with the best legitimate F1-FAR trade-off (0.654/0.147); (c) the
**theory** (the correct generative operator makes blind synthesis oracle-faithful —
superposition-equivalence generalized to the domain's TRUE mixing operator, here
partition); (d) the **annotation-free conformal FAR guarantee** (val-margin +
NB-reject + split-conformal) that operator-only prior work lacks; (e) cross-domain
generality (Reuters text). Exact oracle equivalence is a conditional theoretical
result, not a universal empirical claim. Generality is demonstrated primarily
across IMAGE datasets (MixedWM38 public wafer with FCM-PM, chip-internal ~0.99
FCM-PM, MultiMNIST with overlay); text (Reuters) is an additional modality, and
audio (FSD50K spectrogram summation) is an OPTIONAL bonus extension — never a
load-bearing claim.

Our method has three parts:
1. **Label-fidelity / generative-model-match criterion (operator selection)** —
   measure, before training, which content-blind operator both preserves every
   source's evidence (survival) AND reproduces the domain's true combination law
   (for wafer maps, the die budget / density). In genuine superposition domains
   (inked digits, spectrograms) the operator is overlay / max-union; on **wafer
   maps** max-union is EXCLUDED as die-budget-violating (over-dense 0.50 vs real
   0.31) and the faithful operator is the **full-cover complement**; in
   disjoint-coordinate text it is vector averaging. **FCM-PM** (full-cover
   complement + Pair-Mask) is the **faithful WM38 method** (0.654/0.147; Pair-Mask
   is the FAR-control lever, 0.147 vs 0.384) and also excels on chip (~0.99).
   Synthetic-normal + negative-target is a FAR-control auxiliary.
2. **Val-margin checkpoint selection** — select checkpoints on a disjoint
   held-out-source synthetic proxy by the pos-neg margin; source-only, never
   touches real multi-label data.
3. **NB-reject** — a synthetic-only class-conditional Gaussian (naive-Bayes)
   pattern-likelihood reject/decode stage at a selective operating point.
   Split-conformal on real known-good normals additionally gives a
   finite-sample marginal FAR guarantee under exchangeability. Parts 2-3 (plus
   synthetic normals + neg-target) are the **annotation-free FAR guarantee** that
   operator-only prior work (Shin22) does not provide.

## 2. Theory (`D:/project/known-cnn/docs/mlsynth_paper/THEORY.md`, ported to LaTeX)

- **Def 1 (max-superposition domain)**: `x_{a,b} = x_a ∨ x_b` (evidence joined,
  never replaced). Wafer/spectrogram/ink qualify; opaque RGB photos do NOT
  (occlusion replaces).
- **Theorem 1 (excess risk)**: `R_real(f_syn) − R_real(f*) ≤ 2B·TV(D_real,D_syn)
  ≤ 2B·[ TV(π_real,π_syn) + E TV(K_real,K_syn) ]`. Split into a **support term**
  (co-occurrence prior the oracle knows, singles don't) and an **independence
  term** (correlated placement / interaction). Short proof (add/subtract R_syn +
  TV mixture chain rule).
- **Cor 1 (matched-law risk equivalence)**: superposition + independence +
  matched subset prior ⇒ TV=0 ⇒ equal population risk within the same
  hypothesis class. If normals are evaluated, the empty-set conditional must
  also be matched.
- **Cor 2 (limit of the guarantee)**: a non-join conditional on a positive-mass
  subset makes the real and synthetic laws differ. This removes the zero-shift
  guarantee but does not alone prove a positive classification-risk gap; VOC is
  the empirical boundary case.
- **Theorem 2 (finite-sample)**: ERM adds loss-class complexity and
  concentration terms; asymptotic excess risk has `limsup <= 2B·TV`, not
  equality.
- **Prop 2** proves label-content inconsistency at the catastrophe rate;
  monotonic F1 consequences require a margin model. **Prop 3** uses strict
  `score > tau` (or explicit tie handling).

## 3. Measured results (correctness audit is local/uncommitted)

Metrics: **bit_F1** = macro-F1 over class bits @0.5; **FAR** = false alarms on
negative bits; **NORMAL FAR** = false-alarm rate on real all-negative samples;
mAP; exact-match; pos/neg prob. **Always report bit_F1 AND FAR together.**

Metric audit on 2026-07-10: MixedWM38 bits 5 and 7 occur in singles but have no
positive example in the real mixed pool. The legacy `bit_f1` conditionally
included an unsupported class only when it had a false positive, making the
macro denominator model-dependent. Code now computes macro-F1 over classes
with positive target support and leaves unsupported-bit false positives in
FAR. Existing headline F1 numbers below are legacy until the saved-probability
main rerun is complete.

### MixedWM38 (public wafer benchmark — the industrial headline)
- **Die-budget partition + density (decisive)**: real 2-mix defect-die fraction
  0.305 ≈ single 0.290; max-union/overlay/Shin22 0.501 (64% denser than real; 91%
  exceed real-2mix 95th pct); FCM/FCM-PM 0.293 (matches real). max-union EXCLUDED
  as die-budget-violating (over-dense generative model). Evidence: RESULTS.md
  section "max-union violates wafer die-budget".
- **Strict operator comparison (5 seeds, pick=val_tail_margin_guarded, neg 0.02)**:
  FCM-PM (faithful) **0.654 / FAR 0.147** = best F1-FAR among die-budget-faithful
  arms; FCM-no-PM 0.665/0.384, cutmix 0.691/0.439, mixup 0.537/0.225, single-only
  0.473/0.602. Pair-Mask = FAR-control lever (0.147 vs 0.384). max-union scores the
  highest raw bit-F1 (0.800 / FAR 0.010) but is EXCLUDED (over-dense) — a
  modeling-faithfulness argument, not F1. Recovery vs literature-grade oracle
  (0.974): ~67% with the faithful operator.
- **Excluded over-dense arm provenance (audit only)**: the max-union / Summation
  Mixup arm reaches 0.841±0.034 (9-seed headline, +sn+neg, NORMAL FAR 0.0008, 6/9=0)
  and 0.795 (FULL, all 7015 singles, NORMAL FAR 0.0003) — NOT a faithful result,
  it is the over-dense-training artifact quantified above, retained to make the
  exclusion auditable. Oracle NORMAL FAR is 0.563 in the headline protocol; the
  reported checkpoint remains 0.799 under tau=0.99; full-scale oracle FAR is
  seed-sensitive (reaches 0.001 in one seed), so no universal impossibility.
- **FULL confirmation, over-dense arms (all 7015 singles, 14k test, 3 seeds)**:
  oracle 0.984, overlay/max-union (excluded) 0.795 (NORMAL FAR 0.0003), cutmix
  0.855 (FAR 0.618, unusable), mixup 0.581 (0.758), single_only 0.409 (0.390).
  Synthetic-normal applied identically to every arm including the oracle. Consistent
  with the label-content inconsistency mechanism, but does not directly estimate the
  TV term.
- **Corrected causal fidelity (SmallCNN, five paired model/data splits)**:
  retained evidence `f=0/.1/.25/.5/.75/1` gives supported-class F1
  `0.631/.724/.806/.811/.814/.793` and NORMAL FAR
  `0.179/.507/.253/.005/.012/.015`. Full versus erased changes F1 by +0.162
  [0.129,0.208], gap by +0.229 [0.191,0.276], and FAR by -0.164
  [-0.234,-0.087]. Spearman associations are 0.662/-0.656 for F1/FAR, but
  adjacent-step monotonicity is only 60%/52%: promote a catastrophe-threshold
  mechanism, not a global monotonic law.

### MultiMNIST (controlled mechanism)
- **FULL (400/class, 8k train, 3k test, 25ep, 3 seeds)**: overlay **0.868 mAP /
  0.883 holdout** EXCEEDS oracle 0.846 / **0.685** (+0.198 on held-out combos =
  compositional generalization). mixup 0.738, single 0.619, cutmix 0.606.

### Reuters-21578 (4th modality = text; operator flip)
- **FULL (top-20 cats, 3 seeds)**: oracle 0.603, **vec_avg 0.433 (72%)**, concat
  0.398, single 0.254. Operator FLIPS vs images (averaging wins) because topic
  evidence is on disjoint coordinates — invariant is evidence preservation, not
  the operator.

### Loss-engineering control (ICLR rebuttal to "just use a better loss")
- WM38, SmallCNN, atomic (only arm×loss vary, no synth-normal): single_only+BCE
  0.243 / +ASL 0.316 / +Focal 0.242 vs **overlay 0.607**. ASL adds only +0.073
  AND blows NORMAL FAR to 1.00; synthesis adds +0.364 at lower FAR. Bottleneck is
  structural (co-occurrence), not loss.

### VOC 2007 (boundary / negative control — NOT a success case)
- oracle 0.410, copypaste(content-AWARE, uses boxes) 0.379, single_only 0.303,
  cutmix 0.285, mixup 0.162. Blind synthesis HURTS (entity-type: objects do not
  superimpose, single-only already ~74% without us). This illustrates the
  conditional-mismatch boundary; the theorem does not by itself prove a
  positive risk gap for every photo dataset.

### ChestX-ray14 (medical — probed, INCONCLUSIVE, not in paper)
- ResNet18/128px/20ep/subsampled: oracle itself weak (mAP 0.31), overlay bitF1
  ~2× single but mAP tie + worse FAR. Undertrained/hard at this scale. A clean
  CXR result needs 224px / 50ep+ / full 100k data / more GPU = a separate
  project. Held for a decision; do NOT claim CXR.

### Conformal FAR guarantee
- Real known-good calibration: α=0.05→realized 0.040, α=0.01→0.006 (n=500).
  Synthetic-normal calibration fails exchangeability (realized 0.97).

## 4. Positioning vs SPML (the nearest paradigm)
Information ordering: full supervision > SPML (multi-label images, 1 positive
observed, rest are false negatives) > **ours (no multi-label image at all)**.
Ours is strictly weaker yet can be risk-equivalent under Corollary 1's matched-
law assumptions. SPML's toolkit is
INAPPLICABLE: our zeros are true negatives (no false negatives to correct), so
our `single_only` IS SPML's Assume-Negative baseline made unbiased — and it
still fails (0.24–0.41). An oracle SPML method reduces to that failing baseline.
Deficit is structural, remedy is synthesis. Orthogonal + composable; head-to-head
benchmark is ill-posed (different inputs) ⇒ conceptual ordering only.

## 5. Hard constraints / decisions (do not violate)
- **Content-blind**: no location/mask annotation at train time (copy-paste is a
  content-AWARE reference probe only, clearly scoped).
- **Report bit_F1 AND FAR** together, never FAR-hidden.
- **VOC/COCO = boundary, not showcase.** Do not add photo datasets as "wins".
- **Tier comes from theory + guarantees + the boundary, NOT dataset count.**
- Augmentation policy: no rotation/flip that breaks class identity; the paper's
  operators are overlay/Mixup/CutMix/copy-paste/FCM/FCM-PM as synthesis, not aug.
- Fairness: equal-condition oracle (same backbone/epochs/budget).

## 6. Code / harness (`D:/project/known-cnn/multilabel_synth/`)
- `D:/project/known-cnn/multilabel_synth/metrics.py` — bit_f1, far, compute_map, exact_match, pos_neg_prob.
- `D:/project/known-cnn/multilabel_synth/models/small_cnn.py` (0.62M, spatial-preserving), `D:/project/known-cnn/multilabel_synth/models/resnet.py`
  (build_resnet18, build_resnet18_small [52px], build_resnet18_gray [128px CXR]).
- Synthesis modules under `D:/project/known-cnn/multilabel_synth/synthesis/` —
  synthesize_arm/synth_wm38: oracle/overlay(max)/CutMix/Mixup/FCM/FCM-PM/
  single_only, +synth-normal.
- Runners: `run_matrix.py` (MNIST), `run_wm38.py` (--backbone {small,resnet18}
  --loss {bce,asl,focal}), `run_condition.py` (generic condition-type, --backbone),
  `run_reuters.py`, `run_cxr_hf.py`, `run_voc*.py`, `run_coco*.py`.
- Datasets: `datasets/{multimnist,mixedwm38,voc,coco,cxr14,plant2021}.py`.
- Result CSVs: `D:/project/known-cnn/outputs/multilabel_synth/*.csv` (wm38_FULL_3s, mnist_FULL_3s,
  aslbase_{bce,asl,focal}, reuters/voc/coco runs).

## 7. Paper artifacts
- `D:/project/known-cnn/docs/mlsynth_paper/DRAFT.md` — full prose draft (abstract, 6 sections).
- `D:/project/known-cnn/docs/mlsynth_paper/THEORY.md` — theory working doc.
- `D:/project/known-cnn/docs/mlsynth_paper/latex/main.tex` + `D:/project/known-cnn/docs/mlsynth_paper/latex/refs.bib` — LaTeX (all sections; theorem/
  corollary defined; `\bibliography{refs}` active; env/brace/cite balance verified;
  **no LaTeX compiler was available in-env — not yet compiled to PDF**).
- `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` — running measured-results log.
- `D:/project/known-cnn/docs/superpowers/specs/` — design specs.

## 8. Required next evidence
1. **Closest-method reproduction**: Shin et al. 2022 Summation Mixup and
   Shim--Kang 2023 under the common MixedWM38 8-bit/FAR protocol.
2. **Causal fidelity intervention**: hold the learner fixed and erase controlled
   fractions of one source's evidence.
3. **Fair operating curves**: calibration-selected bit_F1--NORMAL-FAR curves
   for oracle, overlay, CutMix, Mixup, and closest-prior arms.
4. **Paired replication**: same split and at least five common seeds for every
   main arm; report paired confidence intervals.
5. **Conformal repetition**: multiple calibration sizes, alpha values, and
   resampled calibration splits, with exact tie handling.
6. **Paper finishing**: compile the LaTeX, add the fidelity curve, Pareto
   frontier, and held-out-combination figure.

Detailed gates and venue estimates are in
`D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.

## 9. Environment notes
- Windows, RTX 4060 Ti 16GB (shared — user runs their own GPU jobs; prefer CPU,
  or use GPU only when free; ResNet-full on CPU is impractical).
- Data on E:\ (torchvision MNIST; `E:/data/mixedwm38/Wafer_Map_Datasets.npz`;
  `E:/data/cxr14_hf/cxr_split.npz` cached).
- HF token in `.hf_token` (gitignored — never commit). Account choichoichoi123.
- Recent commits (main): SPML positioning (5c0c6cc), CXR probe note, CXR backbone,
  WM38 FULL, finite-sample Thm 2, excess-risk Thm 1, MNIST/Reuters FULL, ASL/Focal.

## 10. Active ICLR Evidence Chain (2026-07-15)

The current queue is intentionally serialized to avoid GPU races:

1. WM38 FCM/FCM-PM geometry: `48` method/layout/group/grid conditions, five seeds.
2. Strict WM38 FCM-PM negative-target and fair prior-method comparison.
3. Practical disjoint real-Normal training/calibration comparison.
4. Prospective FSD50K source-only benchmark with exact waveform summation as the frozen primary candidate.
5. WM38 original-style FCM-PM `dual_loss` fidelity ablation.
6. Exploratory FSD50K replication of the same FCM-PM training-mode correction.

The live manager audit is refreshed every five minutes at:

`D:/project/known-cnn/docs/mlsynth_paper/LIVE_ICLR_EVIDENCE_STATUS.md`

The key implementation correction is that the original chip trainer applies
Pair Mask as an auxiliary second forward,
`L = L_mixed + lambda_PM L_pair_mask`. The first external harness instead
split a fixed total view budget between mixed and mask samples, halving FCM-PM
mixed-view exposure versus FCM. Both protocols are now preserved and compared;
the preregistered primary runs retain the old `equal_total_views` default.

Relevant absolute paths:

- `D:/project/known-cnn/multilabel_synth/run_wm38_strict_selection.py`
- `D:/project/known-cnn/multilabel_synth/run_wm38_strict_fcmpm_neg_compare.py`
- `D:/project/known-cnn/multilabel_synth/run_fsd50k_source_only.py`
- `D:/project/known-cnn/scripts/start_wm38_fcmpm_dual_loss_after_fsd.ps1`
- `D:/project/known-cnn/scripts/start_fsd50k_fcmpm_dual_loss_after_wm38_dual.ps1`
- `D:/project/known-cnn/docs/mlsynth_paper/FCMPM_CORRECTION_AND_EVIDENCE_PLAN_260713.md`

Verification: all `D:/project/known-cnn/tests/multilabel_synth` tests pass
(`121 passed` after the balanced-layout, aligned dual-loss, CutMix-semantics,
and factorial-analysis additions).
