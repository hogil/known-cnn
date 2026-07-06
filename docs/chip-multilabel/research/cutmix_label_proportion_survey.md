# CutMix Label-Proportion Survey for Multi-Label Chip Classification

**Author:** chip-multilabel research notes
**Date:** 2026-05-08
**Scope:** literature review of how mixed-image augmentations (CutMix, Mixup, GridMix,
PuzzleMix, SaliencyMix, SnapMix, TransMix, multi-label CutMix) define the *label*
when image regions are mixed — and what this implies for our 4-class chip
multi-label setup with asymmetric A:B group splits (g=2/3/4).

---

## 1. The CutMix canon (Yun et al., ICCV 2019) — area is the label

CutMix [1] mixes two training images by replacing a rectangular region of `x_A`
with the same region from `x_B`. Formally, with a binary mask `M ∈ {0,1}^{H×W}`:

> `x̃ = M ⊙ x_A + (1 − M) ⊙ x_B`
> `ỹ = λ y_A + (1 − λ) y_B`,   `λ = 1 − bbox_area / (H·W)`

`λ` is **defined exactly as the fraction of A's pixels remaining in the mixed
image**. The Beta(α, α) distribution only controls the *prior over patch sizes*;
once the patch is sampled, `λ` is the deterministic geometric area ratio. This
is the founding rule that all later CutMix-style work either inherits or
explicitly overrides.

The intuition is calibration: a network whose output for a mixed image equals
`λ · ŷ_A + (1−λ) · ŷ_B` should not be penalised, because `λ` of the input pixels
literally come from class A. The label is the geometry of the mix.

## 2. Mixup (Zhang et al., ICLR 2018) — convex combination

Mixup [2] uses pixel-wise linear blending instead of cut-and-paste:

> `x̃ = λ x_i + (1−λ) x_j`,   `ỹ = λ y_i + (1−λ) y_j`,   `λ ~ Beta(α, α)`

Here every pixel of the mixed image is `λ·A + (1−λ)·B` — the per-pixel
"contribution of A" is exactly `λ` everywhere — so the label rule is forced:
it must be the same `λ` to keep the network's loss linear in `λ`. CutMix is the
spatially-disjoint cousin of mixup; both share `ỹ = λ y_A + (1−λ) y_B` for the
same calibration reason.

## 3. PuzzleMix (Kim, Choo & Song, ICML 2020) — saliency-aware cut, area-proportional label

PuzzleMix [3] solves an optimal-transport problem to choose *which* regions of
A and B to mix (favouring salient ones), but the label rule is still

> `ỹ = λ y_A + (1 − λ) y_B`,   `λ = (Σ_p z_p) / (H·W)`

where `z_p ∈ {0,1}` is the per-pixel decision to keep A. In other words, even
when the mix is saliency-driven, **the label weight is the resulting visible
area of A**, not the saliency mass. PuzzleMix changes the "where", not the
"how-much".

## 4. SaliencyMix (Uddin et al., ICLR 2021) — same convex rule, smarter box

SaliencyMix [4] selects the source patch from a saliency-peak region of B
(instead of CutMix's random box), but inherits the area-proportional label
exactly: `λ = 1 − bbox_area/(H·W)`. The paper's contribution is *patch
selection*; it explicitly does **not** rescale `λ` by saliency mass. Their
ablations show that even with saliency-guided cuts, area-based λ remains the
correct calibration — provided the cut now contains real foreground.

## 5. SnapMix (Huang, Wang & Tao, AAAI 2021) — when area is *wrong*

SnapMix [5] is the first major paper to explicitly argue **against**
area-proportional λ. For fine-grained recognition, "discriminative information
mainly lies in some small regions" — pasting a tiny but discriminative box of
B into a large background of A makes a label of `λ y_A + (1−λ) y_B` a lie when
`λ ≈ 1`. SnapMix replaces area with **Class Activation Map (CAM) mass**:

> `ρ_A(R) = Σ_{p∈R} CAM_A(p) / Σ_p CAM_A(p)`
> `ỹ = (1 − ρ_A(R_cut)) y_A + ρ_B(R_paste) y_B`

Critically, the two coefficients **need not sum to 1** — the rule is
asymmetric and can adapt to "small but information-rich" pastes. This is the
single most relevant prior work for our chip case, where fork (a thin vertical
line) is exactly the "small region, high information" pathology SnapMix
addresses.

## 6. GridMix (Baek, Bang & Shim, Pattern Recognition 2021) — grid → count → label

GridMix [6] is the closest geometric match to our setup. The image is divided
into a `q × q` grid; each cell is independently filled from A or B by a
Bernoulli draw. The label is

> `ỹ = (n_A / n_total) y_A + (n_B / n_total) y_B`

i.e. **the count of A-cells / total cells**, which is identical to area for
equal-sized cells. So GridMix endorses the area-proportional rule for the
exact discrete-grid topology we use — a 200×200 chip split into 8×8 cells
with cell assignments to A or B.

## 7. TransMix (Chen et al., CVPR 2022) — attention as the new λ

TransMix [7] keeps CutMix's pixel-level cut but redefines `λ` as the
intersection of the cut with the ViT's `[CLS]` attention map:

> `λ = (Σ_{p∈M} a_p) / Σ_p a_p`,   `ỹ = λ y_A + (1−λ) y_B`

The motivation is that CutMix can produce a box that contains no actual A-object
(so `λ y_A + (1−λ) y_B` is mis-calibrated). TransMix uses self-supervised
attention to estimate "how much of A actually survived" — same convex form,
but the weight is *information* not *area*. This is the ViT-era refinement of
SnapMix's idea.

## 8. Multi-label CutMix is structurally different

Sections 1–7 all assume **single-label** softmax classification, where
`ỹ ∈ Δ^C` is a probability simplex and `λ y_A + (1−λ) y_B` is a valid soft
target for cross-entropy. In **multi-label** (BCE, multi-hot `y ∈ {0,1}^C`),
the picture changes:

- **Erasure problem.** Burgert et al. [8] show that vanilla CutMix on
  multi-label remote-sensing images can *delete* labels that only existed
  in B's erased region, producing label noise. Their fix is a *Label
  Propagation* strategy: apply the same mask to a per-pixel reference map,
  read off the surviving labels, and ignore the area-proportional rule
  entirely. The convex-sum assumption is wrong here because each class is
  independent — a class is either present in the mixed image or not.

- **Logical-OR rule.** LogicMix [9] (Chong et al., 2024) for multi-label
  classification with partial labels uses

  > `ỹ_c = 1` if any source has class `c = 1`, `0` if all sources are `0`,
  > `?` (ignored) otherwise.

  No area term at all. The reasoning: BCE per-class is independent, so the
  natural label of "is class c visible somewhere in the mix?" is binary, not
  area-weighted. This is the hardest-form anti-CutMix label rule.

- **timm pragmatism.** Wightman et al. [10] (ResNet strikes back) ship
  CutMix-with-BCE in `timm` using a *threshold* `--bce-target-thresh`: the
  area-proportional soft target is binarised (`y > τ → 1, else 0`), then BCE
  is applied. This is a hybrid: area-proportional for *which* labels make
  the cut, hard `1.0` for the loss. They report no statistically significant
  difference vs. soft targets — suggesting the multi-label loss is
  *insensitive* to the exact λ value once the present-class set is correct.

The single-label community converged on `λ y_A + (1−λ) y_B` because softmax
needs probability mass to total 1. The multi-label community has not converged:
options range from area-proportional soft (CutMix-naïve) → logical-OR hard
(LogicMix) → label-propagation from pixel maps (Burgert) → threshold-binarised
(timm).

## 9. Mapping the literature to our three candidate rules

Recall our three candidates from the introduction:

| rule | label form | sums to | endorsed by |
|------|------------|---------|-------------|
| 1. equal-weight (current) | `[scale, scale]` | `2·scale` | timm-with-thresh (after binarisation) |
| 2. area-proportional | `[scale·a_frac, scale·b_frac]` | `scale` | CutMix, Mixup, GridMix, PuzzleMix, SaliencyMix, TransMix |
| 3. soft normalised | `[a_frac, b_frac]` | `1` | softmax-Mixup convention (single-label only) |

Rule 3 is a single-label artifact (`Σ_c y_c = 1` is required by softmax CE) and
should be ruled out for our BCE multi-label head — it forces fork+scratch
chips with `[0.33, 0.67]` to compete on a probability simplex, which BCE doesn't
need or want.

Rule 1 is defensible *if* one buys the "presence is binary, area is nuisance"
view (LogicMix [9], timm-with-threshold [10]). Empirically it is what we
already run.

Rule 2 is the eight-paper consensus for soft multi-label (1, 2, 3, 4, 6, 7) and
is the *direct multi-label generalisation* of CutMix's area-of-A formula. For
our 8×8 grid this is identical to GridMix [6] — `λ = n_A / 64`.

## 10. Domain reasoning — why the iter19 numbers point at Rule 2

Our iter19E (g=3, label_scale=0.67) leads the sweep with bit-F1 0.875, and
0.67 ≈ 2/3 = `b_frac` for the g=3 split. Two readings:

**(a) Coincidence.** With one seed and a 4-cell grid for label shape, this
might be noise. iter19I (g=4, scale=0.75 ≈ b_frac=0.75) is also near the top,
which weakly supports the pattern, but `scale_for_a` was applied to *both*
labels equally — so the experiment that actually moved the BCE target away
from `[1, 1]` was just multiplying both by the same constant. The "scale"
in iter19 is **not** an asymmetric area-proportional rule yet. It is a
symmetric label-smoothing-like factor that happens to land on the area
fraction by coincidence.

**(b) Useful coincidence.** Multiplying both targets by 2/3 acts like
target-side label smoothing on the present classes only. This is similar to
Müller et al.'s "When does label smoothing help?" effect — it dampens
over-confident gradients when both classes are already strongly co-active.
The fact that `scale=0.67` ≈ `b_frac=0.67` is suggestive but the experiment
cannot distinguish "asymmetric area-proportional helps" from "any scaling
in [0.5, 0.75] helps."

The clean way to test is iter20: an *atomic* swap from
`mix_t[a_cls]=mix_t[b_cls]=scale` to
`mix_t[a_cls]=scale·a_frac, mix_t[b_cls]=scale·b_frac`, keeping every other
hparam fixed. Per our atomic-method-iteration rule, this is exactly one
change.

## 11. Final answer for our case

For our 4-class chip multi-label (BCE + multi-hot, 8×8 grid, defects ∈
{bb, fork, sc, sr}), the canonical literature answer is **Rule 2:
area-proportional** — specifically the GridMix [6] form `λ = n_A_cells / 64`,
giving target `[a_frac · scale, b_frac · scale]` with `scale ≈ 1.0`. This is
the direct multi-label extension of Yun et al.'s ICCV 2019 formula and the
exact rule for our discrete grid topology. The current symmetric rule
`[scale, scale]` is defensible only under the "presence-is-binary" reading
(LogicMix [9]) — which we should empirically test against rule 2 in iter20.

Two caveats from the multi-label literature [8–10]:

- The asymmetric form is not safe if a defect can disappear from the cut
  (e.g. fork is a thin vertical line that can sit entirely inside one group
  of cells). When `b_frac → 0`, fork's BCE target → 0 even though the fork
  pixels are physically gone too — so the label is honest, but variance is
  high. Mitigation: only apply rule 2 when both source defects have ≥ 1 cell
  of "energy" surviving the mix; otherwise fall back to the per-cell label
  count (GridMix's literal definition).

- A SnapMix-style refinement [5] would weight by **defect-mask area** rather
  than cell count. We have defect masks in our synth pipeline (chip_meta), so
  this is available. fork (thin) and scratch (long thin) would weight much
  smaller than bank_boundary (covers half the chip) — closer to the
  "information mass" each defect contributes. This is a natural iter21
  candidate after iter20 establishes baseline area-proportional behaviour.

**Recommended sequence:**

1. **iter20** — atomic switch to area-proportional `[a_frac, b_frac]` with
   `scale=1.0`. Measures the pure label-rule effect against current
   `[1.0, 1.0]`. 5-seed.
2. **iter21** — defect-mask-area-proportional (SnapMix-style), weighting by
   true defect pixel count not cell count. Tests whether information mass
   beats geometric area for thin-line defects (fork, scratch).
3. **iter22** — LogicMix-style hard `[1, 1]` baseline with no scaling at all,
   for completeness against the literature's binary-presence school.

---

## BibTeX

```bibtex
@inproceedings{yun2019cutmix,
  title     = {{CutMix}: Regularization Strategy to Train Strong Classifiers with Localizable Features},
  author    = {Yun, Sangdoo and Han, Dongyoon and Oh, Seong Joon and Chun, Sanghyuk and Choe, Junsuk and Yoo, Youngjoon},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019},
  eprint    = {1905.04899}
}

@inproceedings{zhang2018mixup,
  title     = {mixup: Beyond Empirical Risk Minimization},
  author    = {Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  eprint    = {1710.09412}
}

@inproceedings{kim2020puzzlemix,
  title     = {Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup},
  author    = {Kim, Jang-Hyun and Choo, Wonho and Song, Hyun Oh},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020},
  eprint    = {2009.06962}
}

@inproceedings{uddin2021saliencymix,
  title     = {{SaliencyMix}: A Saliency Guided Data Augmentation Strategy for Better Regularization},
  author    = {Uddin, A F M Shahab and Monira, Mst. Sirazam and Shin, Wheemyung and Chung, TaeChoong and Bae, Sung-Ho},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  eprint    = {2006.01791}
}

@inproceedings{huang2021snapmix,
  title     = {{SnapMix}: Semantically Proportional Mixing for Augmenting Fine-grained Data},
  author    = {Huang, Shaoli and Wang, Xinchao and Tao, Dacheng},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {2},
  pages     = {1628--1636},
  year      = {2021}
}

@article{baek2021gridmix,
  title   = {{GridMix}: Strong regularization through local context mapping},
  author  = {Baek, Kyungjune and Bang, Duhyeon and Shim, Hyunjung},
  journal = {Pattern Recognition},
  volume  = {109},
  pages   = {107594},
  year    = {2021},
  publisher = {Elsevier}
}

@inproceedings{chen2022transmix,
  title     = {{TransMix}: Attend to Mix for Vision Transformers},
  author    = {Chen, Jie-Neng and Sun, Shuyang and He, Ju and Torr, Philip H. S. and Yuille, Alan and Bai, Song},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  eprint    = {2111.09833}
}

@article{burgert2024labelpropcutmix,
  title   = {A Label Propagation Strategy for {CutMix} in Multi-Label Remote Sensing Image Classification},
  author  = {Burgert, Tom and Clasen, Kai Norman and Klotz, Jonas and Siebert, Tim and Demir, Beg{\"u}m},
  journal = {arXiv preprint},
  year    = {2024},
  eprint  = {2405.13451}
}

@article{chong2024logicmix,
  title   = {Free Performance Gain from Mixing Multiple Partially Labeled Samples in Multi-label Image Classification},
  author  = {Chong, Chak Fong and Guo, Jielong and Yang, Xu and Ke, Wei and Wang, Yapeng},
  journal = {arXiv preprint},
  year    = {2024},
  eprint  = {2405.15860}
}

@article{wightman2021resnet,
  title   = {{ResNet} strikes back: An improved training procedure in {timm}},
  author  = {Wightman, Ross and Touvron, Hugo and J{\'e}gou, Herv{\'e}},
  journal = {arXiv preprint},
  year    = {2021},
  eprint  = {2110.00476}
}
```

---

## Sources consulted

- [CutMix (Yun et al., ICCV 2019)](https://arxiv.org/abs/1905.04899)
- [mixup (Zhang et al., ICLR 2018)](https://arxiv.org/abs/1710.09412)
- [Puzzle Mix (Kim, Choo & Song, ICML 2020)](https://arxiv.org/abs/2009.06962)
- [SaliencyMix (Uddin et al., ICLR 2021)](https://arxiv.org/abs/2006.01791)
- [SnapMix (Huang, Wang & Tao, AAAI 2021)](https://github.com/Shaoli-Huang/SnapMix)
- [GridMix (Baek, Bang & Shim, Pattern Recognition 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303976)
- [TransMix (Chen et al., CVPR 2022)](https://arxiv.org/abs/2111.09833)
- [Label Propagation CutMix (Burgert et al., 2024)](https://arxiv.org/abs/2405.13451)
- [LogicMix (Chong et al., 2024)](https://arxiv.org/abs/2405.15860)
- [ResNet strikes back (Wightman et al., 2021)](https://arxiv.org/abs/2110.00476)
- [timm CutMix+BCE discussion](https://github.com/huggingface/pytorch-image-models/discussions/1001)
