"""Multi-label loss library for Stage 4 Phase B/C 학습.

LOSS_DESIGN.md 의 모든 mix 조합 (M1-M7) 의 base class 들을 단독으로 제공.
cnn_train.py / cnn_train_compound.py 통합 시 import 해서 사용:

    from multi_label_losses import AsymmetricLoss, AdaGCLoss, ChainedLoss

지원 loss:
- BCEWithLogitsLoss (PyTorch native, 참조용)
- AsymmetricLoss (Ridnik et al. ICCV 2021) — Stage 4 Phase C
- AdaGCLoss (Verelst et al. 2024 arXiv:2510.08269) — Stage 4 Phase B
- ChainedLoss (warmup → main) — Stage 4 mix M5

모든 loss 가 multi-hot target 받음 (shape: B × C, float).
single-positive 환경 (한 sample = 1 positive) 도 multi-hot 으로 변환해서 입력.
"""
from __future__ import annotations
import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_multihot(target_int: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Single-label int target (B,) → multi-hot (B, C)."""
    return F.one_hot(target_int.long(), num_classes).float()


# ---------------------------- BCE (참조) ----------------------------

class BCELossWithSmoothing(nn.Module):
    """BCEWithLogitsLoss + label smoothing (multi-hot).

    label_smoothing: 0.0 → standard. 0.05 권장 (학계 default).
    pos_weight: per-class weight for positive samples (class imbalance 보정).
    """
    def __init__(self, label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight",
                              pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            t = target * (1 - self.label_smoothing) + self.label_smoothing / target.size(-1)
        else:
            t = target
        pw = self.pos_weight if self.pos_weight.numel() > 0 else None
        if pw is not None:
            pw = pw.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, t.float(), pos_weight=pw)


# ---------------------------- Asymmetric Loss (Ridnik 2021) ----------------------------

class AsymmetricLoss(nn.Module):
    """ASL — multi-label SOTA loss.

    L+_c = (1 - σ(z))^γ_pos × log(σ(z))
    L-_c = (σ(z) - clip)_+^γ_neg × log(1 - σ(z) + clip)

    Default (Ridnik 2021): γ_pos=1, γ_neg=4, clip=0.05.
    학계 결과: COCO mAP 81.3 (BCE) → 86.6 (ASL) +5.3.
    """
    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, eps: float = 1e-8,
                 disable_torch_grad_focal_loss: bool = True):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # numerically stable sigmoid
        p = torch.sigmoid(logits)
        target = target.float()

        # asymmetric clipping (negative)
        if self.clip is not None and self.clip > 0:
            p_neg = (p - self.clip).clamp(min=self.eps)
        else:
            p_neg = p

        # standard BCE terms (positive + negative)
        log_pos = torch.log(p.clamp(min=self.eps))
        log_neg = torch.log((1 - p_neg).clamp(min=self.eps))

        loss_pos = target * log_pos                                                       # (-) sign 곱하지 않음 — 마지막에 -mean
        loss_neg = (1 - target) * log_neg

        # asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    p_pos_t = p
                    p_neg_t = (p - self.clip).clamp(min=0) if (self.clip and self.clip > 0) else p
                    pt0 = (1 - p_pos_t) ** self.gamma_pos
                    pt1 = p_neg_t ** self.gamma_neg
                loss_pos = loss_pos * pt0
                loss_neg = loss_neg * pt1
            else:
                pt0 = (1 - p) ** self.gamma_pos
                pt1 = p_neg ** self.gamma_neg
                loss_pos = loss_pos * pt0
                loss_neg = loss_neg * pt1

        loss = -(loss_pos + loss_neg)
        return loss.mean()


# ---------------------------- AdaGC Loss (Verelst 2024) ----------------------------

class AdaGCLoss(nn.Module):
    """Adaptive Gradient Calibration for Single-Positive Multi-Label Learning.

    L_total = L_BCE_AN + λ × L_GC
    L_GC = -(1/n) × Σ_i Σ_c 1(y_{i,c}=0) × log(1 - p_{i,c} × t_{i,c})

    pseudo-label t = dual EMA teacher confidence.
    학계 결과 (OpenImages SPML): BCE-AN 73.8 → AdaGC 78.1 (+4.3 mAP).

    사용법:
        loss_fn = AdaGCLoss(num_classes=33, lambda_gc=0.5)
        # 학습 loop
        student_logits = model(x)
        teacher_logits = teacher_model(x)    # EMA copy of student
        loss = loss_fn(student_logits, teacher_logits, target)
    """
    def __init__(self, num_classes: int = None,
                 lambda_gc: float = 0.5,
                 alpha_pred: float = 0.95,
                 lambda_warmup_epochs: int = 3,
                 eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_gc = lambda_gc
        self.alpha_pred = alpha_pred
        self.lambda_warmup_epochs = lambda_warmup_epochs
        self.eps = eps
        # smoothed pseudo-label (dual EMA second stage)
        self.register_buffer("smoothed_pred", torch.zeros(0))
        self.epoch = 0

    def update_pseudo(self, teacher_p: torch.Tensor) -> torch.Tensor:
        """teacher_p: (B, C) sigmoid prob from teacher. smoothed_pred = α × prev + (1-α) × teacher_p."""
        with torch.no_grad():
            if self.smoothed_pred.numel() == 0:
                self.smoothed_pred = teacher_p.detach().clone()
            else:
                # batch averaging: cumulative running mean
                if self.smoothed_pred.shape != teacher_p.shape:
                    # different batch size — pad/truncate per-class average
                    cls_avg = teacher_p.mean(dim=0, keepdim=True)
                    self.smoothed_pred = (
                        self.alpha_pred * self.smoothed_pred.mean(dim=0, keepdim=True)
                        + (1 - self.alpha_pred) * cls_avg
                    ).expand_as(teacher_p).contiguous()
                else:
                    self.smoothed_pred = (
                        self.alpha_pred * self.smoothed_pred
                        + (1 - self.alpha_pred) * teacher_p.detach()
                    )
        return self.smoothed_pred

    def get_lambda_eff(self) -> float:
        """λ warmup for early epochs (학계 권장 — 초기 dual EMA noisy)."""
        if self.epoch < self.lambda_warmup_epochs:
            return self.lambda_gc * (self.epoch / max(1, self.lambda_warmup_epochs))
        return self.lambda_gc

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        student_p = torch.sigmoid(student_logits)
        teacher_p = torch.sigmoid(teacher_logits.detach())

        t = self.update_pseudo(teacher_p)                                                 # (B, C)
        if t.shape != student_p.shape:
            t = teacher_p                                                                  # fallback to current teacher

        # Standard BCE (assumed-negative for unknown)
        L_BCE = F.binary_cross_entropy(
            student_p.clamp(self.eps, 1 - self.eps), target.float(), reduction="none")
        L_BCE = L_BCE.mean()

        # Gradient calibration: 약화 negative penalty
        is_negative = (target == 0).float()
        gc_term = is_negative * torch.log((1 - student_p * t).clamp(min=self.eps))
        L_GC = -gc_term.mean()

        lam_eff = self.get_lambda_eff()
        return L_BCE + lam_eff * L_GC

    def step_epoch(self):
        """학습 loop 의 epoch 끝마다 호출."""
        self.epoch += 1


def update_teacher_ema(student: nn.Module, teacher: nn.Module, alpha: float = 0.99):
    """First-stage EMA: teacher_weights = α × teacher_weights + (1-α) × student_weights."""
    with torch.no_grad():
        for p_s, p_t in zip(student.parameters(), teacher.parameters()):
            p_t.data.mul_(alpha).add_(p_s.data, alpha=1 - alpha)


def init_teacher(student: nn.Module) -> nn.Module:
    """Create teacher as EMA copy of student. Eval mode, no_grad."""
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ---------------------------- ChainedLoss (warmup → main) ----------------------------

class ChainedLoss(nn.Module):
    """Two-phase loss: warmup_loss for first N epoch, then main_loss.

    LOSS_DESIGN.md M5 — BCE warmup → ASL fine-tune.
    """
    def __init__(self, main_loss: nn.Module, warmup_loss: nn.Module,
                 warmup_epochs: int = 5):
        super().__init__()
        self.main_loss = main_loss
        self.warmup_loss = warmup_loss
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def forward(self, *args, **kwargs):
        if self.epoch < self.warmup_epochs:
            return self.warmup_loss(*args, **kwargs)
        return self.main_loss(*args, **kwargs)

    def step_epoch(self):
        self.epoch += 1


# ---------------------------- Mix loss (M4 AdaGC + ASL) ----------------------------

class MixLoss(nn.Module):
    """L = base + α × aux. M4 의 AdaGC + ASL hybrid 등.

    Note: AdaGC 는 (student_logits, teacher_logits, target) signature.
          ASL 은 (logits, target). Mix 시 target 만 공유, teacher_logits 는 AdaGC 만.
    """
    def __init__(self, base_loss: nn.Module, aux_loss: nn.Module,
                 aux_weight: float = 0.5,
                 base_needs_teacher: bool = False,
                 aux_needs_teacher: bool = False):
        super().__init__()
        self.base_loss = base_loss
        self.aux_loss = aux_loss
        self.aux_weight = aux_weight
        self.base_needs_teacher = base_needs_teacher
        self.aux_needs_teacher = aux_needs_teacher

    def forward(self, student_logits, target, teacher_logits=None):
        if self.base_needs_teacher:
            base = self.base_loss(student_logits, teacher_logits, target)
        else:
            base = self.base_loss(student_logits, target)
        if self.aux_needs_teacher:
            aux = self.aux_loss(student_logits, teacher_logits, target)
        else:
            aux = self.aux_loss(student_logits, target)
        return base + self.aux_weight * aux

    def step_epoch(self):
        if hasattr(self.base_loss, "step_epoch"):
            self.base_loss.step_epoch()
        if hasattr(self.aux_loss, "step_epoch"):
            self.aux_loss.step_epoch()


# ---------------------------- factory ----------------------------

def build_loss(loss_name: str, num_classes: int, **kwargs) -> nn.Module:
    """LOSS_DESIGN.md 의 M1-M7 factory.

    name 매핑:
    - 'M1' / 'ce'        → CrossEntropy (single-label, 학계 baseline) — single_label_ce 권장
    - 'M2' / 'asl'       → ASL default
    - 'M3' / 'asl_eff'   → ASL + LS=0.05 (CW + LS 는 dataset/sampler 에서 처리)
    - 'M4' / 'adagc_asl' → AdaGC + 0.5 × ASL hybrid
    - 'M5' / 'bce_warmup_asl' → ChainedLoss (BCE warmup → ASL)
    - 'M6' / 'focal_asl' → MixLoss(focal + ASL)  ※ focal 은 cnn_train.py 의 FocalLoss reuse
    - 'M7' / 'adagc_ls'  → AdaGC with LS-aware target (외부 LS 적용)
    """
    name = loss_name.lower()
    label_smoothing = float(kwargs.get("label_smoothing", 0.0))
    pos_weight = kwargs.get("pos_weight", None)

    if name in ("m1", "ce", "bce"):
        return BCELossWithSmoothing(label_smoothing=label_smoothing, pos_weight=pos_weight)
    if name in ("m2", "asl"):
        return AsymmetricLoss(
            gamma_pos=float(kwargs.get("gamma_pos", 1.0)),
            gamma_neg=float(kwargs.get("gamma_neg", 4.0)),
            clip=float(kwargs.get("clip", 0.05)))
    if name in ("m3", "asl_eff"):
        return AsymmetricLoss(
            gamma_pos=float(kwargs.get("gamma_pos", 1.0)),
            gamma_neg=float(kwargs.get("gamma_neg", 4.0)),
            clip=float(kwargs.get("clip", 0.05)))
    if name in ("m4", "adagc_asl"):
        adagc = AdaGCLoss(num_classes=num_classes,
                          lambda_gc=float(kwargs.get("lambda_gc", 0.5)),
                          alpha_pred=float(kwargs.get("alpha_pred", 0.95)),
                          lambda_warmup_epochs=int(kwargs.get("lambda_warmup_epochs", 3)))
        asl = AsymmetricLoss(
            gamma_pos=float(kwargs.get("gamma_pos", 1.0)),
            gamma_neg=float(kwargs.get("gamma_neg", 4.0)),
            clip=float(kwargs.get("clip", 0.05)))
        return MixLoss(adagc, asl, aux_weight=float(kwargs.get("aux_weight", 0.5)),
                       base_needs_teacher=True, aux_needs_teacher=False)
    if name in ("m5", "bce_warmup_asl"):
        warmup = BCELossWithSmoothing(label_smoothing=label_smoothing, pos_weight=pos_weight)
        main = AsymmetricLoss(
            gamma_pos=float(kwargs.get("gamma_pos", 1.0)),
            gamma_neg=float(kwargs.get("gamma_neg", 4.0)),
            clip=float(kwargs.get("clip", 0.05)))
        return ChainedLoss(main, warmup,
                           warmup_epochs=int(kwargs.get("warmup_epochs", 5)))
    if name in ("m6", "focal_asl"):
        # focal as positive-focusing wrapper of ASL (γ_pos boost)
        return AsymmetricLoss(
            gamma_pos=float(kwargs.get("gamma_pos", 2.0)),                                # higher pos focus
            gamma_neg=float(kwargs.get("gamma_neg", 4.0)),
            clip=float(kwargs.get("clip", 0.05)))
    if name in ("m7", "adagc_ls"):
        # AdaGC with label smoothing (target adjustment in dataset; loss accepts as-is)
        return AdaGCLoss(num_classes=num_classes,
                          lambda_gc=float(kwargs.get("lambda_gc", 0.5)),
                          alpha_pred=float(kwargs.get("alpha_pred", 0.95)),
                          lambda_warmup_epochs=int(kwargs.get("lambda_warmup_epochs", 3)))
    raise ValueError(f"unknown loss name: {loss_name}")


# ---------------------------- self-test ----------------------------

if __name__ == "__main__":
    """Smoke test — 모든 loss 가 forward + backward 가능한지."""
    import sys
    torch.manual_seed(42)
    B, C = 4, 33
    logits = torch.randn(B, C, requires_grad=True)
    teacher_logits = torch.randn(B, C)
    target = torch.zeros(B, C); target[range(B), [3, 7, 12, 22]] = 1                     # single positive

    # Standard 2-arg signatures (no teacher)
    print("[*] testing 2-arg losses (no teacher)...")
    for name in ["bce", "asl", "m3", "m5", "m6"]:
        loss = build_loss(name, num_classes=C)
        out = loss(logits, target)
        out.backward(retain_graph=True)
        print(f"  {name:20s}: loss={out.item():.4f}  ok")

    # AdaGC variants (needs teacher)
    print("[*] testing teacher-needing losses...")
    for name in ["adagc_asl", "adagc_ls"]:
        loss = build_loss(name, num_classes=C)
        if isinstance(loss, MixLoss):
            out = loss(logits, target, teacher_logits=teacher_logits)
        else:
            out = loss(logits, teacher_logits, target)
        out.backward(retain_graph=True)
        print(f"  {name:20s}: loss={out.item():.4f}  ok")

    # Direct AdaGC
    print("[*] direct AdaGCLoss class...")
    adagc = AdaGCLoss(num_classes=C, lambda_gc=0.5)
    out = adagc(logits, teacher_logits, target)
    out.backward(retain_graph=True)
    print(f"  AdaGCLoss           : loss={out.item():.4f}  ok")

    print("\n[OK] all losses pass smoke test.")
