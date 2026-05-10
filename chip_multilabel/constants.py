"""Constant definitions for chip multi-label evaluation.

11 evaluation classes:
- 4 single defects (training set, sigmoid-active class slot)
- 5 combo (2-defect overlay; scratch+scratch_rot excluded — same defect family)
- Normal (no defect active)
- Invalid (heuristic-detected: white + orange border)

The model is trained on TRAIN_CLASSES only (4-way single-label CE/Focal/ASL/BCE).
"""
from __future__ import annotations

from typing import Tuple

TRAIN_CLASSES: Tuple[str, ...] = (
    "bank_boundary",
    "fork",
    "scratch",
    "scratch_rot",
)

SINGLE_KEYS: Tuple[str, ...] = TRAIN_CLASSES

COMBO_KEYS: Tuple[str, ...] = (
    # 2-combo (C(4,2) = 6).
    "bank_boundary+fork",
    "bank_boundary+scratch",
    "bank_boundary+scratch_rot",
    "fork+scratch",
    "fork+scratch_rot",
    "scratch+scratch_rot",
)

# 260508 — 3-combo (C(4,3) = 4) 재도입. eval set 14-class spectrum 확장.
# 사용자 directive 260508: "2 + 3-class combo 10 total". gen_eval_set.py 의
# --include-triples flag 로만 사용. 학습 정책 (TRAIN_CLASSES 4 single) 은 그대로.
TRIPLE_COMBO_KEYS: Tuple[str, ...] = (
    "bank_boundary+fork+scratch",
    "bank_boundary+fork+scratch_rot",
    "bank_boundary+scratch+scratch_rot",
    "fork+scratch+scratch_rot",
)

ALL_COMBO_KEYS: Tuple[str, ...] = COMBO_KEYS + TRIPLE_COMBO_KEYS

SPECIAL_KEYS: Tuple[str, ...] = ("Normal", "Invalid")

# 260506 — 5 wafer-pattern classes added (replaces single 'unknown' class).
# Each = chip-level defect from corresponding wafer-canvas pattern (b>=200 bin-defect chips
# from wafer's defect region). Eval-only, NEVER in training. The model has 4 sigmoid outputs
# for 4 trained obj — these 5 classes test how the model handles OOD chips that show wafer-level
# pattern at chip level (no chip-level obj match).
WAFER_PATTERN_KEYS: Tuple[str, ...] = (
    # 260507 user directive — Row 삭제 (fork bit FP 73.8% dominant cause: Row 의 horizontal dot
    # pattern 이 fork horizontal top 과 시각 유사해 모델이 fork 라고 잘못 fire). 4 OOD 만 유지.
    "DiagonalSmear",
    "CenterDonut",
    "CrossScratch",
    "Starburst",
)

# 260507 — 2 trained + 1 OOD overlay 4 class. GT bits = 2 trained 만 active (OOD 는 visual
# robustness 시험). class_key 형식: '<a>+<b>+ood_<OOD>' — KEY_TO_LABELS 가 2 trained 추출.
OOD_OVERLAY_KEYS: Tuple[str, ...] = (
    "fork+scratch+ood_DiagonalSmear",
    "bank_boundary+fork+ood_CenterDonut",
    "fork+scratch_rot+ood_CrossScratch",
    "scratch+scratch_rot+ood_Starburst",
)

ALL_CLASS_KEYS: Tuple[str, ...] = (SINGLE_KEYS + COMBO_KEYS + TRIPLE_COMBO_KEYS
                                   + SPECIAL_KEYS + WAFER_PATTERN_KEYS + OOD_OVERLAY_KEYS)

NUM_TRAIN = len(TRAIN_CLASSES)
NUM_ALL = len(ALL_CLASS_KEYS)

INFERENCE_VARIANTS: Tuple[str, ...] = (
    "I0",  # argmax (single-only baseline)
    "I1",  # softmax + per-class F1-max threshold
    "I2",  # sigmoid + fixed 0.5 threshold
    "I3",  # sigmoid + per-class F1-max threshold
    "I4",  # TS + sigmoid + per-class F1-max threshold
    # I5 (TTA stack) PERMANENTLY DROPPED: see notes.md "Hard Rules"
    # iter1 measured -0.018 macro_f1 vs I4. chip patterns rotation-sensitive.
    "I6",  # sigmoid + F1-max threshold + floor 0.3 (anti-degenerate)
    "I7",  # sigmoid + joint macro-F1 coord descent threshold (vs per-class binary F1)
    "I8",  # sigmoid + top-2 margin gating (combo only when 2nd >= margin*top1)
    "I9",  # per-class T (vs scalar T) + sigmoid + F1-max threshold
    "I10",  # I7 + softmax-entropy Normal short-circuit (high entropy -> Normal)
    "I11",  # I7 + bb+sr pair-aware rescue (analyst's pair-prob refinement, arXiv:2404.16193)
    "I12",  # 260506 — I7 + sc+sr pair-aware rescue (mirror of I11 for sc+sr weakness)
    "I13",  # 260506 — I7 + max-prob low-confidence Normal gate (real-env Normal 80% lever)
)
TRAIN_VARIANTS: Tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4", "T5", "T6")

DEFAULT_BACKBONE_CKPT = (
    "D:/project/known-cnn/outputs/logs_chip/"
    "chip5_round4_v14_260505_061558_running/best_model.pth"
)
DEFAULT_CLASSIFICATION_CHIPS = "D:/project/data/wm-811k/classification_chips"


def sort_label_pair(a: str, b: str) -> str:
    """Canonical 'a+b' combo key — alphabetic order."""
    if a == b:
        raise ValueError(f"combo of identical class: {a}")
    lo, hi = sorted([a, b])
    return f"{lo}+{hi}"


def _build_key_to_labels() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for k in SINGLE_KEYS:
        out[k] = frozenset({k})
    for k in COMBO_KEYS:
        out[k] = frozenset(k.split("+"))
    for k in TRIPLE_COMBO_KEYS:
        out[k] = frozenset(k.split("+"))
    out["Normal"] = frozenset()
    out["Invalid"] = frozenset({"__INVALID__"})
    # 260506 — 4 wafer-pattern OOD eval classes (chip-level pattern from wafer-canvas defect region)
    for wp in WAFER_PATTERN_KEYS:
        out[wp] = frozenset({f"__WP_{wp}__"})
    # 260507 — 4 OOD overlay classes (2 trained + 1 OOD).
    # GT bits = 2 trained 만 active. format: '<a>+<b>+ood_<OOD>' → split by 'ood_' → trained pair.
    for ok in OOD_OVERLAY_KEYS:
        # extract '<a>+<b>' (before '+ood_')
        trained_part = ok.split("+ood_")[0]
        labels = trained_part.split("+")
        out[ok] = frozenset(labels)
    return out


KEY_TO_LABELS: dict[str, frozenset[str]] = _build_key_to_labels()
LABELS_TO_KEY: dict[frozenset[str], str] = {v: k for k, v in KEY_TO_LABELS.items()}


def labels_to_class_key(active: frozenset[str], is_invalid: bool = False) -> str:
    """Map (active defect set, invalid flag) -> class key.

    - is_invalid True -> 'Invalid'
    - active empty   -> 'Normal'
    - active size 1  -> single key
    - active size 2  -> 2-combo key (canonical sort)
    - active size 3  -> 3-combo key (canonical sort) — 260508 재도입
    - active size >= 4 -> caller must handle (no 4+combo class)
    """
    if is_invalid:
        return "Invalid"
    if len(active) == 0:
        return "Normal"
    if len(active) == 1:
        (k,) = active
        return k
    if len(active) == 2:
        sorted_keys = sorted(active)
        key = "+".join(sorted_keys)
        if key not in COMBO_KEYS:
            raise KeyError(f"unknown combo {key}; not in COMBO_KEYS")
        return key
    if len(active) == 3:
        sorted_keys = sorted(active)
        key = "+".join(sorted_keys)
        if key not in TRIPLE_COMBO_KEYS:
            raise KeyError(f"unknown 3-combo {key}; not in TRIPLE_COMBO_KEYS")
        return key
    raise ValueError(f"active set size {len(active)} >= 4 — no 4+combo class defined")
