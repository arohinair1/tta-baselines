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

    def as_floats(self) -> dict:
        return {
            "total": self.total.item(),
            "energy": self.energy.item(),
            "pseudolabel": None if self.pseudolabel is None else self.pseudolabel.item(),
        }


def tta_loss(seg_logits: torch.Tensor,
             energy_model: nn.Module,
             strategy: str = "hard",
             warped_label: Optional[torch.Tensor] = None,
             pixel_weights: Optional[torch.Tensor] = None,
             pl_weight: float = 1.0) -> TTALossParts:
    """The complete per-iteration TTA objective from eval_tta.adapt_and_predict.

    strategy: 'none' (pure PTTEA, run_pttea.py without --use_pseudolabel),
              'hard', 'confidence', 'entropy'.
    For 'hard'/'confidence' the pseudolabel term is skipped when
    warped_label is None (first slice of a case) -> reduces to pure PTTEA.
    pl_weight is 1.0 in the repo (hard-coded "# Weight = 1.0").
    """
    probs = torch.softmax(seg_logits, dim=1)
    e = energy_loss(energy_model(probs))
    pl = None
    if strategy == "entropy":
        pl = entropy_from_probs(probs)   # same softmax tensor as the energy term (as in the repo)
    elif strategy in ("hard", "confidence") and warped_label is not None:
        w = pixel_weights if strategy == "confidence" else None
        pl = pseudolabel_ce_loss(seg_logits, warped_label, w)
    elif strategy not in ("none", "hard", "confidence"):
        raise ValueError(f"unknown strategy {strategy!r}")
    total = e if pl is None else e + pl_weight * pl
    return TTALossParts(total=total, energy=e, pseudolabel=pl)


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
