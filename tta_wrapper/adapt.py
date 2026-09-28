"""
Test-time adaptation loop, isolated from data loading and evaluation.

Two entry points:

  adapt_slice(...)       clean re-implementation (uses losses.tta_loss).
                         Parametrised so it can reproduce BOTH original loops:
                           run_pttea.py  -> variant="pttea"
                           eval_tta.py   -> variant="eval_tta"
  adapt_slice_repo(...)  calls Nicole's eval_tta.adapt_and_predict directly and
                         fixes its return-arity bug (see below). Used by the
                         tests to prove adapt_slice is equivalent.

Differences between the two original loops (both reproduced):
  * eval_tta has early stopping (patience=3, min_delta=1e-4) on the
    pseudolabel loss (or total loss when there is no pseudolabel term);
    run_pttea always runs num_iterations.
  * eval_tta does a final no-grad forward AFTER the last optimizer step;
    run_pttea's saved prediction is the forward from the last iteration,
    i.e. BEFORE the last step.
  * Both deep-copy the pretrained model for every slice (no state carried
    across slices; eval_tta supports carrying state but its callers don't).

Known bug in external/pseudolabel/eval_tta.py (not edited, worked around here):
  adapt_and_predict returns 6 values (…, out_state, actual_iters) but every
  caller unpacks 5 -> `python eval_tta.py` crashes with
  "ValueError: too many values to unpack" on the first slice.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ._bootstrap import bootstrap
from .losses import tta_loss

bootstrap()

import registration_utils as _reg_demons              # noqa: E402
import registration_utils_affine as _reg_affine       # noqa: E402

REGISTRATION = {
    "demons": _reg_demons.register_and_warp_2d,
    "affine": _reg_affine.register_and_warp_2d,
}


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    """TENT-style: freeze everything, train only BatchNorm2d affine params,
    and force BN to use batch statistics (running stats removed).
    Identical to eval_tta.configure_model_for_tent / run_pttea.configure_model."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


@dataclass
class SliceResult:
    pred: np.ndarray                 # (H, W) int
    probs: torch.Tensor              # (1, C, H, W)
    confidence: np.ndarray           # (H, W) max softmax prob
    losses: List[dict] = field(default_factory=list)  # per-iteration {total, energy, pseudolabel}
    iters_run: int = 0
    warped_label: Optional[np.ndarray] = None         # the pseudolabel actually used


def warp_previous(curr_img: np.ndarray, prev_img: np.ndarray, prev_label: np.ndarray,
                  prev_conf: Optional[np.ndarray], registration: str):
    """Register prev_img -> curr_img and warp prev_label (and prev_conf)."""
    reg = REGISTRATION[registration]
    warped = reg(curr_img, prev_img, prev_label)
    warped_conf = None
    if prev_conf is not None and warped is not None:
        warped_conf = reg(curr_img, prev_img, prev_conf.astype(np.float32))
    return warped, warped_conf


def adapt_slice(seg_model_base: nn.Module,
                energy_model: nn.Module,
                image: torch.Tensor,
                *,
		loss_fn=tta_loss,
                strategy: str = "hard",
                prev_img: Optional[np.ndarray] = None,
                prev_label: Optional[np.ndarray] = None,
                prev_conf: Optional[np.ndarray] = None,
                registration: str = "demons",
                num_iterations: int = 10,
                lr: float = 0.01,
                variant: str = "eval_tta",
                min_delta: float = 1e-4,
                patience: int = 3,
                device="cpu") -> SliceResult:
    """Adapt a fresh copy of seg_model_base on ONE image (1,1,H,W) and predict.

    strategy   : 'none' | 'hard' | 'confidence' | 'entropy'
    variant    : 'eval_tta' (early stop + final forward) or 'pttea'
                 (fixed iterations, prediction from last iteration's forward)
    """
    assert variant in ("eval_tta", "pttea")
    image = image.to(device)
    model = configure_model_for_tent(copy.deepcopy(seg_model_base)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # pseudolabel preparation
    warped_t = None
    weights_t = None
    warped_np = None
    if strategy in ("hard", "confidence") and prev_img is not None and prev_label is not None:
        curr_np = image[0, 0].detach().cpu().numpy()
        warped_np, warped_conf = warp_previous(
            curr_np, prev_img, prev_label,
            prev_conf if strategy == "confidence" else None, registration)
        if warped_np is not None:
            warped_t = torch.from_numpy(warped_np).long().to(device).unsqueeze(0)
            if strategy == "confidence" and warped_conf is not None:
                weights_t = torch.from_numpy(np.clip(warped_conf, 0.0, 1.0)).float().to(device).unsqueeze(0)

    res = SliceResult(pred=None, probs=None, confidence=None, warped_label=warped_np)
    best, since_improve = float("inf"), 0
    last_probs = None
    for _ in range(num_iterations):
        opt.zero_grad()
        logits = model(image)
        parts = loss_fn(logits, energy_model, strategy=strategy,
                         warped_label=warped_t, pixel_weights=weights_t)
        res.losses.append(parts.as_floats())
        last_probs = torch.softmax(logits, dim=1).detach()
        parts.total.backward()
        opt.step()
        res.iters_run += 1
        if variant == "eval_tta":
            cur = parts.pseudolabel if parts.pseudolabel is not None else parts.total
            cur = cur.detach().item()
            if cur + min_delta < best:
                best, since_improve = cur, 0
            else:
                since_improve += 1
            if since_improve >= patience:
                break

    if variant == "eval_tta":
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(image), dim=1)
    else:
        probs = last_probs
    res.probs = probs
    res.pred = probs.argmax(dim=1).cpu().numpy()[0]
    res.confidence = probs.max(dim=1)[0].cpu().numpy()[0]
    return res


def adapt_slice_repo(seg_model_base, energy_model, image, *, strategy="hard",
                     prev_img=None, prev_label=None, prev_conf=None,
                     registration="demons", num_iterations=10, lr=0.01,
                     device="cpu") -> SliceResult:
    """Run Nicole's eval_tta.adapt_and_predict unchanged (fixing the 6-vs-5
    unpack) so results can be compared against adapt_slice(variant='eval_tta')."""
    import eval_tta
    eval_tta.register_and_warp_2d = REGISTRATION[registration]
    out = eval_tta.adapt_and_predict(
        seg_model_base, energy_model, image.to(device),
        prev_img_np=prev_img, prev_label_np=prev_label, prev_confidence_np=prev_conf,
        pseudolabel_strategy=strategy, num_iterations=num_iterations, lr=lr, device=device)
    pred, probs, conf, iter_losses, _state, actual_iters = out   # <- 6 values
    return SliceResult(pred=pred, probs=probs, confidence=conf,
                       losses=[{"pseudolabel": v} for v in iter_losses],
                       iters_run=actual_iters)


def adapt_sequence(seg_model_base, energy_model, images: List[torch.Tensor], **kw) -> List[SliceResult]:
    """Run adapt_slice over an ordered list of slices from one case, feeding
    each prediction forward as the next slice's pseudolabel source.
    (This is the per-case loop of eval_tta.eval_*_tta.)"""
    results = []
    prev_img = prev_label = prev_conf = None
    for x in images:
        r = adapt_slice(seg_model_base, energy_model, x,
                        prev_img=prev_img, prev_label=prev_label, prev_conf=prev_conf, **kw)
        results.append(r)
        prev_img = x[0, 0].detach().cpu().numpy()
        prev_label = r.pred.astype(np.uint8)
        prev_conf = r.confidence
    return results


def predict_no_adapt(seg_model, image: torch.Tensor, device="cpu") -> SliceResult:
    """Source-model baseline (eval.py): plain forward, no adaptation."""
    seg_model.eval()
    with torch.no_grad():
        probs = torch.softmax(seg_model(image.to(device)), dim=1)
    return SliceResult(pred=probs.argmax(dim=1).cpu().numpy()[0], probs=probs,
                       confidence=probs.max(dim=1)[0].cpu().numpy()[0])


def adapt_batch(seg_model_base, energy_model, images: torch.Tensor, *,
                num_iterations: int = 10, lr: float = 0.01, device="cpu") -> List[SliceResult]:
    """run_pttea.py semantics: adapt ONE fresh model copy on a BATCH of slices
    jointly (BatchNorm statistics pooled over the batch), energy loss only,
    fixed iterations, prediction from the last iteration's forward.
    images: (B,1,H,W). Returns one SliceResult per image."""
    images = images.to(device)
    model = configure_model_for_tent(copy.deepcopy(seg_model_base)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses, last_probs = [], None
    for _ in range(num_iterations):
        opt.zero_grad()
        logits = model(images)
        parts = tta_loss(logits, energy_model, strategy="none")
        losses.append(parts.as_floats())
        last_probs = torch.softmax(logits, dim=1).detach()
        parts.total.backward()
        opt.step()
    out = []
    for i in range(images.shape[0]):
        p = last_probs[i:i + 1]
        out.append(SliceResult(pred=p.argmax(dim=1).cpu().numpy()[0], probs=p,
                               confidence=p.max(dim=1)[0].cpu().numpy()[0],
                               losses=losses, iters_run=num_iterations))
    return out


def energy_mask_pred(probs: torch.Tensor, energy_model: nn.Module, threshold: float = 0.5) -> np.ndarray:
    """run_pttea.py's `pred_mask`: score each 16x16 patch with the energy model,
    upsample sigmoid(score) bilinearly to image size, zero the class probabilities
    where score < threshold (those pixels fall to background), then argmax.
    probs: (1,C,H,W). Returns (H,W) int."""
    with torch.no_grad():
        score = torch.sigmoid(energy_model(probs))
        up = torch.nn.functional.interpolate(score, size=probs.shape[-2:], mode="bilinear", align_corners=False)
        masked = probs * (up >= threshold)
    return masked.argmax(dim=1).cpu().numpy()[0]
