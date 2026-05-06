"""Loss implementations for Stage 2 training variants.

T1: CE + label_smoothing 0.1 (PyTorch native)
T4: ASL (Asymmetric Loss, Ridnik 2021)
T5: BCE with one-hot multi-hot target (single-positive)
T6: BCE -> ASL switch at warmup_epochs
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL — Ridnik 2021.

    Defaults: gamma_pos=1, gamma_neg=4, clip=0.05.
    Target must be multi-hot (B, C) {0,1}.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        target = target.float()
        p_neg = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        log_pos = torch.log(p.clamp(min=self.eps))
        log_neg = torch.log((1 - p_neg).clamp(min=self.eps))
        with torch.no_grad():
            pt0 = (1 - p) ** self.gamma_pos
            p_neg_d = (p - self.clip).clamp(min=0) if self.clip > 0 else p
            pt1 = p_neg_d ** self.gamma_neg
        loss_pos = target * log_pos * pt0
        loss_neg = (1 - target) * log_neg * pt1
        return -(loss_pos + loss_neg).mean()


class BCEMultiHot(nn.Module):
    """BCE on multi-hot target. Single-positive case = one-hot target."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.smoothing > 0:
            # 1 -> 1 - smoothing/2, 0 -> smoothing/2 (symmetric BCE smoothing)
            target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, target)


class CEWithSmoothing(nn.Module):
    """CE with label smoothing on class-index target."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target.long(), label_smoothing=self.smoothing)


class CESoftLabel(nn.Module):
    """CE with soft (multi-positive) target + label smoothing.

    Target is (B, C) float, possibly multi-positive (e.g. CutMix). Sums per row need
    not be 1. This loss normalizes target to sum=1 via row-wise division (preserving
    relative weights), then applies LS, then computes soft-CE = -mean(target * log_softmax(logits)).

    Used for T8 = CE+LS+CutMix where CutMix target = λ·one_hot_a + (1-λ)·one_hot_b.
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        # row-wise normalize to sum=1
        sums = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = target / sums
        if self.smoothing > 0:
            C = target.size(-1)
            target = target * (1.0 - self.smoothing) + self.smoothing / C
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()
        return loss


class BCEThenASL(nn.Module):
    """T6: BCE for warmup_epochs, then ASL. Caller must call .set_epoch(ep) each epoch."""

    def __init__(self, warmup_epochs: int = 5, asl_kwargs: dict | None = None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.bce = BCEMultiHot()
        self.asl = AsymmetricLoss(**(asl_kwargs or {}))
        self.epoch = 0
        self.last_active = "bce"

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.epoch < self.warmup_epochs:
            self.last_active = "bce"
            return self.bce(logits, target)
        self.last_active = "asl"
        return self.asl(logits, target)


class FocalLoss(nn.Module):
    """Focal loss (Lin 2017, RetinaNet) for multi-class classification.

    L = -alpha_c * (1 - p_c)^gamma * log(p_c)  for true class c.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            probs = log_probs.exp()
        target = target.long()
        nll = F.nll_loss(log_probs, target, reduction="none")
        pt = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        focal_term = (1.0 - pt).clamp(min=0).pow(self.gamma)
        loss = focal_term * nll
        return loss.mean()


class SigmoidFocalLoss(nn.Module):
    """Sigmoid Focal Loss (Lin 2017, RetinaNet) for multi-label classification.

    L = -alpha * (1-p)^gamma * log(p)         for positives (target=1)
        - (1-alpha) * p^gamma * log(1-p)       for negatives (target=0)

    Multi-hot version of Focal — combines BCE structure with focal down-weighting.
    Standard in multi-label image classification (e.g., MS-COCO, OpenImages).

    T9 (260506): added as paper-quality variant. Compare with T7 (BCE+LS).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.smoothing > 0:
            target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        focal_term = (1.0 - p_t).clamp(min=1e-8).pow(self.gamma)
        if self.alpha > 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * focal_term * bce
        else:
            loss = focal_term * bce
        return loss.mean()


def build_loss(loss_name: str, **kw):
    if loss_name == "ce_ls01":
        return CEWithSmoothing(smoothing=kw.get("ls", 0.1)), "class_index"
    if loss_name == "ce_ls":
        return CEWithSmoothing(smoothing=kw.get("ls", 0.1)), "class_index"
    if loss_name == "ce_soft_ls":
        return CESoftLabel(smoothing=kw.get("ls", 0.20)), "soft_multihot"
    if loss_name == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0),
                         label_smoothing=kw.get("ls", 0.0)), "class_index"
    if loss_name == "asl":
        return AsymmetricLoss(
            gamma_pos=kw.get("gamma_pos", 1.0),
            gamma_neg=kw.get("gamma_neg", 4.0),
            clip=kw.get("clip", 0.05),
        ), "multi_hot"
    if loss_name == "bce":
        return BCEMultiHot(label_smoothing=kw.get("ls", 0.0)), "multi_hot"
    if loss_name == "bce_ls":
        return BCEMultiHot(label_smoothing=kw.get("ls", 0.20)), "multi_hot"
    if loss_name == "bce_then_asl":
        return BCEThenASL(warmup_epochs=kw.get("warmup_epochs", 5),
                         asl_kwargs={
                             "gamma_pos": kw.get("gamma_pos", 1.0),
                             "gamma_neg": kw.get("gamma_neg", 4.0),
                             "clip": kw.get("clip", 0.05),
                         }), "multi_hot"
    if loss_name == "sigmoid_focal":
        # T9 (260506) — RetinaNet-style sigmoid focal for multi-label
        return SigmoidFocalLoss(
            alpha=kw.get("alpha", 0.25),
            gamma=kw.get("gamma", 2.0),
            label_smoothing=kw.get("ls", 0.0),
        ), "multi_hot"
    raise ValueError(f"unknown loss: {loss_name}")
