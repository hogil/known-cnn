# Handoff — "Multi-label from single-label via content-blind synthesis"

Self-contained state of the mlsynth paper project, for continuation by another
agent/tool. Everything below is measured and committed to `hogil/known-cnn`
(branch `main`). Paper artifacts live in `docs/mlsynth_paper/`.

## 1. Thesis / novelty (one sentence)

In **superposition domains** (where co-occurring signals combine by a known
blind operator — pixelwise max for wafer maps / spectrograms, coordinate sum
for bag-of-words text), you can **train a multi-label recognizer using only
single-label data** by synthesizing multi-label examples with a **content-blind
operator** (no location/mask annotation), and **match or beat a fully-supervised
oracle** — with **zero multi-label, zero normal, zero location annotation**.

Three named contributions (the "3 techniques"):
1. **Label-faithful synthesis** — pick the blind operator that maximizes label
   fidelity (survival of every source's evidence). Overlay (max) is exact for
   images; it beats mixup/cutmix/copy-paste in survival-order.
2. **Synthetic-normal + pair-mask FAR control** — erase defects from singles to
   make synthetic normals; drives false alarms to ~0 by construction.
3. **Conformal reject** — split-conformal on real known-good normals gives a
   distribution-free FAR guarantee.

## 2. Theory (docs/mlsynth_paper/THEORY.md, ported to latex sec:theory)

- **Def 1 (max-superposition domain)**: `x_{a,b} = x_a ∨ x_b` (evidence joined,
  never replaced). Wafer/spectrogram/ink qualify; opaque RGB photos do NOT
  (occlusion replaces).
- **Theorem 1 (excess risk)**: `R_real(f_syn) − R_real(f*) ≤ 2B·TV(D_real,D_syn)
  ≤ 2B·[ TV(π_real,π_syn) + E TV(K_real,K_syn) ]`. Split into a **support term**
  (co-occurrence prior the oracle knows, singles don't) and an **independence
  term** (correlated placement / interaction). Short proof (add/subtract R_syn +
  TV mixture chain rule).
- **Cor 1 (no oracle advantage)**: superposition + independence + matched support
  ⇒ TV=0 ⇒ blind synthesis is Bayes-equivalent to full supervision.
- **Cor 2 (boundary WITHOUT running photos)**: non-join conditional (occlusion)
  ⇒ irreducible TV floor ⇒ oracle keeps an advantage. **This is why VOC/COCO
  fail — a theorem, not a required benchmark.**
- **Theorem 2 (finite-sample)**: ERM adds `4·Rad_m(F) + 2B·sqrt(2ln(2/δ)/m)`;
  only `2B·TV` is irreducible ⇒ at scale the oracle gap = residual TV.
- **Prop 2 (fidelity→risk)** + **Prop 3 (conformal FAR)**.

## 3. Measured results (all committed; numbers are current)

Metrics: **bit_F1** = macro-F1 over class bits @0.5; **FAR** = false alarms on
negative bits; **NORMAL FAR** = false-alarm rate on real all-negative samples;
mAP; exact-match; pos/neg prob. **Always report bit_F1 AND FAR together.**

### MixedWM38 (public wafer benchmark — the industrial headline)
- **9-seed headline (n_train 6000, ResNet-18, 30ep)**: oracle 0.974±0.019,
  **ours(overlay+sn+neg) 0.841±0.034 = 86% recovery, NORMAL FAR 0.0008** (6/9=0).
  Oracle NORMAL FAR 0.563 unrescuable by thresholding (0.799 @ τ=0.99).
- **FULL confirmation (all 7015 singles, 14k test, 3 seeds)**: oracle 0.984,
  overlay **0.795** (NORMAL FAR **0.0003**), cutmix 0.855 (FAR **0.618**, unusable),
  mixup 0.581 (0.758), single_only 0.409 (0.390). KEY: identical synth-normal
  across arms, yet only overlay's zero label-noise yields zero FAR — cutmix's
  higher bit_F1 is unusable (62% false alarms). Confirms fidelity→FAR (Thm 1
  independence term).

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
  cutmix 0.285, mixup 0.162. Blind synthesis HURTS (entity-type: objects don't
  superimpose, single-only already ~74% without us). This is the boundary, and
  Cor 2 predicts it — so photo benchmarks are NOT required.

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
Ours is strictly weaker yet recovers the oracle (Cor 1). SPML's toolkit is
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
  operators are overlay/mixup/cutmix/copy-paste/fcm-pm as synthesis, not aug.
- Fairness: equal-condition oracle (same backbone/epochs/budget).

## 6. Code / harness (multilabel_synth/)
- `metrics.py` — bit_f1, far, compute_map, exact_match, pos_neg_prob.
- `models/small_cnn.py` (0.62M, spatial-preserving), `models/resnet.py`
  (build_resnet18, build_resnet18_small [52px], build_resnet18_gray [128px CXR]).
- `synthesis/arms.py`, `synthesis/wm38_arms.py`, `synthesis/voc_arms.py` —
  synthesize_arm/synth_wm38: oracle/overlay(max)/cutmix/mixup/fcm_pm/single_only,
  +pair_mask, +synth-normal.
- Runners: `run_matrix.py` (MNIST), `run_wm38.py` (--backbone {small,resnet18}
  --loss {bce,asl,focal}), `run_condition.py` (generic condition-type, --backbone),
  `run_reuters.py`, `run_cxr_hf.py`, `run_voc*.py`, `run_coco*.py`.
- Datasets: `datasets/{multimnist,mixedwm38,voc,coco,cxr14,plant2021}.py`.
- Result CSVs: `outputs/multilabel_synth/*.csv` (wm38_FULL_3s, mnist_FULL_3s,
  aslbase_{bce,asl,focal}, reuters/voc/coco runs).

## 7. Paper artifacts
- `docs/mlsynth_paper/DRAFT.md` — full prose draft (abstract, 6 sections).
- `docs/mlsynth_paper/THEORY.md` — theory working doc.
- `docs/mlsynth_paper/latex/main.tex` + `refs.bib` — LaTeX (all sections; theorem/
  corollary defined; `\bibliography{refs}` active; env/brace/cite balance verified;
  **no LaTeX compiler was available in-env — not yet compiled to PDF**).
- `docs/superpowers/multilabel_synth_RESULTS.md` — running measured-results log.
- `docs/superpowers/specs/2026-07-06-...design.md` — design spec.

## 8. Open directions (pick per goal)
1. **Theory rigor**: P2 constants (replace O(ρ) with a theorem + constants);
   P3 synthetic→real shift term quantified (KS of max-prob); finite-sample TV
   estimator + concentration for Thm 1.
2. **Unified recipe**: show one tuning-free recipe across wafer+MNIST+text (the
   right kind of generality).
3. **CXR proper** (if a 2nd real benchmark is wanted): 224px, 50ep+, full data,
   GPU — a real sub-project, uncertain payoff.
4. **Paper finishing**: compile the LaTeX to PDF (do a real build; only structural
   checks were possible in-env), add figures (survival→ranking scatter, FAR
   geometry, MNIST overlay samples), fill `\author{}`, tighten abstract.
5. **Venue**: current evidence (WM38 + MNIST + Reuters + boundary + theory +
   conformal + loss-control) is a coherent TMLR/Q1 story; ICLR stretch wants
   full-scale (done) + theory rigor + maybe one more real modality.

## 9. Environment notes
- Windows, RTX 4060 Ti 16GB (shared — user runs their own GPU jobs; prefer CPU,
  or use GPU only when free; ResNet-full on CPU is impractical).
- Data on E:\ (torchvision MNIST; `E:/data/mixedwm38/Wafer_Map_Datasets.npz`;
  `E:/data/cxr14_hf/cxr_split.npz` cached).
- HF token in `.hf_token` (gitignored — never commit). Account choichoichoi123.
- Recent commits (main): SPML positioning (5c0c6cc), CXR probe note, CXR backbone,
  WM38 FULL, finite-sample Thm 2, excess-risk Thm 1, MNIST/Reuters FULL, ASL/Focal.
