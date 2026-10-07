"""
All loss terms used in the repo, isolated and documented.

Two places losses live in the original code:

  A. Test-time adaptation (eval_tta.adapt_and_predict / run_pttea.py)
       loss = energy_loss(E(softmax(f(x))))            # PTTEA, always on
            + pseudolabel term                          # Nicole's addition
       where the pseudolabel term is one of
         'hard'       : CE(f(x), warp(prev_pred))
         'confidence' : mean( CE_map(f(x), warp(prev_pred)) * warp(prev_conf) )
         'entropy'    : mean pixel entropy of softmax(f(x))   (no registration)
       Only BatchNorm affine params are updated (see adapt.configure_model_for_tent).

  B. Training
       energy model (train_energy.py):  BCEWithLogits(E(perturbed_mask), patch_labels)
         patch_labels = 1 if the L1 diff between clean and perturbed one-hot
         masks inside a patch is <= threshold else 0  (utils.create_labels)
         or the IoU variant (utils.create_labels_iou)
       2-channel seg model (train_pseudolabel.py): 0.5*Dice + 0.5*CE

Every function here is a pure function of tensors. `tests/test_losses.py`
checks each one against the original repo code bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# A. Test-time adaptation losses
# ──────────────────────────────────────────────────────────────────────

def energy_loss(energy_logits: torch.Tensor) -> torch.Tensor:
    """PTTEA energy term.

    The energy model outputs one logit per patch (B, 1, n, n); during energy
    training label 1 == "this patch of the mask looks like a clean GT patch".
    Original code:  loss = -BCEWithLogitsLoss()(energy_out, zeros)
    BCE against a zero target is softplus(logit), so this equals
    -mean(softplus(logit)); minimising it pushes every patch logit UP, i.e.
    toward the "clean" class. Gradient ascent on the energy, as the paper says.
    """
    target = torch.zeros_like(energy_logits)
    return -F.binary_cross_entropy_with_logits(energy_logits, target)


def pseudolabel_ce_loss(seg_logits: torch.Tensor,
                        warped_label: torch.Tensor,
                        pixel_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pseudolabel cross-entropy against the previous slice's prediction
    warped into the current slice (Demons or affine registration).

    seg_logits   : (B, C, H, W)
    warped_label : (B, H, W) long
    pixel_weights: (B, H, W) float in [0,1] or None
                   None  -> 'hard' strategy: plain mean CE
                   given -> 'confidence' strategy: mean(CE_map * weights)
                            (weights are the warped max-softmax confidence of
                            the previous prediction; NOT re-normalised, so
                            low-confidence slices simply contribute less)
    """
    if pixel_weights is None:
        return F.cross_entropy(seg_logits, warped_label)
    ce_map = F.cross_entropy(seg_logits, warped_label, reduction="none")  # (B,H,W)
    return (ce_map * pixel_weights).mean()


def pseudolabel_gce_loss(seg_logits: torch.Tensor, warped_label: torch.Tensor,
                         q: float = 0.7, pixel_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Generalized cross-entropy (Zhang & Sabuncu, NeurIPS 2018):
        L_q = (1 - p_y^q) / q,   q -> 0 gives CE, q = 1 gives MAE.
    Bounded per pixel, so pixels whose pseudolabel is wrong (p_y small) cannot
    dominate the gradient the way CE lets them. Week 4: "pseudolabel
    consistency loss, such as ... generalized cross-entropy"."""
    p = torch.softmax(seg_logits, dim=1)
    p_y = p.gather(1, warped_label.unsqueeze(1)).squeeze(1).clamp_min(1e-6)   # (B,H,W)
    l = (1.0 - p_y ** q) / q
    if pixel_weights is not None:
        l = l * pixel_weights
    return l.mean()


def pseudolabel_loss(seg_logits: torch.Tensor, warped_label: torch.Tensor, kind: str = "ce",
                     pixel_weights: Optional[torch.Tensor] = None, q: float = 0.7) -> torch.Tensor:
    """kind: 'ce' (repo's hard/confidence term) | 'gce' (generalized CE, robust to label noise)."""
    if kind == "ce":
        return pseudolabel_ce_loss(seg_logits, warped_label, pixel_weights)
    if kind == "gce":
        return pseudolabel_gce_loss(seg_logits, warped_label, q, pixel_weights)
    raise ValueError(f"unknown pseudolabel loss {kind!r}")


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """'entropy' strategy: mean per-pixel Shannon entropy of the softmax.
    Self-supervised, needs no previous slice and no registration.
    Original code uses log(p + 1e-8), reproduced exactly (not log_softmax)."""
    ent = -(probs * torch.log(probs + eps)).sum(dim=1)  # (B, H, W)
    return ent.mean()


def entropy_loss(seg_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return entropy_from_probs(torch.softmax(seg_logits, dim=1), eps)


@dataclass
class TTALossParts:
    total: torch.Tensor
    energy: torch.Tensor
    pseudolabel: Optional[torch.Tensor]  # None when no pseudolabel/entropy term applied
    energy_score: float = float("nan")   # mean sigmoid(E): fraction of patches the energy model calls clean
    lambda_e: float = 1.0
    lambda_p: float = 1.0

    def as_floats(self) -> dict:
        return {
            "total": self.total.item(),
            "energy": self.energy.item(),
            "pseudolabel": None if self.pseudolabel is None else self.pseudolabel.item(),
            "energy_score": self.energy_score, "lambda_e": self.lambda_e, "lambda_p": self.lambda_p,
        }


def tta_loss(seg_logits: torch.Tensor,
             energy_model: nn.Module,
             strategy: str = "hard",
             warped_label: Optional[torch.Tensor] = None,
             pixel_weights: Optional[torch.Tensor] = None,
             pl_weight: float = 1.0,
             lambda_e: float = 1.0,
             lambda_p: Optional[float] = None,
             pl_kind: str = "ce",
             lambda_mode: str = "fixed",
             gce_q: float = 0.7) -> TTALossParts:
    """The complete per-iteration TTA objective (Week 4 form):

        L_adapt = lambda_E * L_energy + lambda_P * L_pseudo

    strategy   : 'none' (pure PTTEA), 'hard', 'confidence', 'entropy'.
    pl_kind    : 'ce' (repo) | 'gce' (generalized CE, robust to wrong pseudolabels).
    lambda_mode: 'fixed'  -> use lambda_e, lambda_p as given (repo: 1, 1).
                 'energy' -> scale lambda_p by how BAD the energy model thinks the
                             current prediction is: lambda_p_eff = lambda_p * 2 * (1 - s),
                             s = mean sigmoid(E(probs)) in [0,1] (s=1: all patches clean).
                             So a prediction the energy model already likes gets little
                             pseudolabel supervision; a poor one leans on the warped label.
    For 'hard'/'confidence' the pseudolabel term is skipped when warped_label
    is None (first slice of a case) -> reduces to pure PTTEA.
    pl_weight is the old name for lambda_p (kept for the tests / repo parity).
    """
    if lambda_p is None:
        lambda_p = pl_weight
    probs = torch.softmax(seg_logits, dim=1)
    e_logits = energy_model(probs)
    e = energy_loss(e_logits)
    score = torch.sigmoid(e_logits).mean().item()
    lp = lambda_p
    if lambda_mode == "energy":
        lp = lambda_p * 2.0 * (1.0 - score)
    elif lambda_mode != "fixed":
        raise ValueError(f"unknown lambda_mode {lambda_mode!r}")
    pl = None
    if strategy == "entropy":
        pl = entropy_from_probs(probs)   # same softmax tensor as the energy term (as in the repo)
    elif strategy in ("hard", "confidence") and warped_label is not None:
        w = pixel_weights if strategy == "confidence" else None
        pl = pseudolabel_loss(seg_logits, warped_label, pl_kind, w, gce_q)
    elif strategy not in ("none", "hard", "confidence"):
        raise ValueError(f"unknown strategy {strategy!r}")
    total = lambda_e * e if pl is None else lambda_e * e + lp * pl
    return TTALossParts(total=total, energy=e, pseudolabel=pl, energy_score=score, lambda_e=lambda_e, lambda_p=lp)


# ──────────────────────────────────────────────────────────────────────
# B. Training losses
# ──────────────────────────────────────────────────────────────────────

def _unfold_patches(x: torch.Tensor, n_blocks: int):
    B, C, H, W = x.shape
    kh, kw = H // n_blocks, W // n_blocks
    return F.unfold(x, kernel_size=(kh, kw), stride=(kh, kw))  # (B, C*kh*kw, n*n)


def patch_labels_l1(mask_clean_onehot: torch.Tensor,
                    mask_perturbed: torch.Tensor,
                    n_blocks: int = 16,
                    threshold: float = 50) -> torch.Tensor:
    """utils.create_labels. Per-patch binary label for training the energy model.
    label = 1 (clean) if sum |onehot_clean - perturbed| over the patch <= threshold.
    Inputs (B, C, H, W); output (B, 1, n_blocks, n_blocks) long."""
    diff = (_unfold_patches(mask_clean_onehot, n_blocks)
            - _unfold_patches(mask_perturbed, n_blocks)).abs().sum(dim=1)
    label = torch.where(diff > threshold, 0, 1)
    return label.view(-1, 1, n_blocks, n_blocks)


def patch_labels_iou(mask_clean_onehot: torch.Tensor,
                     mask_perturbed: torch.Tensor,
                     n_blocks: int = 16,
                     iou_threshold: float = 0.6) -> torch.Tensor:
    """utils.create_labels_iou. Soft-IoU per patch, label 1 if IoU > threshold."""
    a = _unfold_patches(mask_clean_onehot, n_blocks)
    b = _unfold_patches(mask_perturbed, n_blocks)
    inter = (a * b).sum(dim=1)
    union = (a + b - a * b).sum(dim=1)
    iou = inter / (union + 1e-6)
    label = torch.where(iou > iou_threshold, 1, 0)
    return label.view(-1, 1, n_blocks, n_blocks)


def energy_train_loss(energy_logits: torch.Tensor, patch_labels: torch.Tensor) -> torch.Tensor:
    """train_energy.py: BCEWithLogitsLoss(E(perturbed_mask), patch_labels)."""
    return F.binary_cross_entropy_with_logits(energy_logits.float(), patch_labels.float())


class DiceCrossEntropyLoss(nn.Module):
    """train_pseudolabel.DiceCrossEntropyLoss, copied verbatim (that script
    can't be imported because it parses argv at module level)."""

    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        target_one_hot = torch.zeros_like(pred)
        for c in range(self.num_classes):
            target_one_hot[:, c] = (target == c).float()
        dice_losses = []
        for c in range(self.num_classes):
            pred_c, target_c = pred[:, c], target_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            dice_coef = 2.0 * intersection / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_losses.append(1.0 - dice_coef)
        return torch.stack(dice_losses).mean()

    def forward(self, pred, target):
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target[:, 0]
            elif target.shape[1] == self.num_classes:
                target = target.argmax(dim=1)
        target = target.long()
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(torch.softmax(pred, dim=1), target)
        return self.dice_weight * dice + self.ce_weight * ce
