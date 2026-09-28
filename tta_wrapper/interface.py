"""
High-level interface for PTTEA test-time adaptation.

This class provides a simple API for:
    - segmentation prediction
    - energy calculation
    - TTA loss calculation
    - test-time adaptation

The underlying implementations remain in models.py, losses.py, and adapt.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .adapt import SliceResult, adapt_slice, predict_no_adapt
from .losses import TTALossParts, tta_loss


class PTTEA:
    """Simple interface around the PTTEA segmentation and energy models."""

    def __init__(
        self,
        seg_model: nn.Module,
        energy_model: nn.Module,
        device: str = "cpu",
    ):
        self.device = device
        self.seg_model = seg_model.to(device)
        self.energy_model = energy_model.to(device)

        self.seg_model.eval()
        self.energy_model.eval()

        # The energy model is fixed during test-time adaptation.
        for p in self.energy_model.parameters():
            p.requires_grad_(False)

    def predict(self, image: torch.Tensor) -> SliceResult:
        """Predict a segmentation without adaptation."""
        return predict_no_adapt(
            self.seg_model,
            image,
            device=self.device,
        )

    def energy(self, image: torch.Tensor) -> torch.Tensor:
        """Return the patch-wise energy logits for the current prediction."""
        self.seg_model.eval()
        self.energy_model.eval()

        with torch.no_grad():
            logits = self.seg_model(image.to(self.device))
            probs = torch.softmax(logits, dim=1)
            energy_logits = self.energy_model(probs)

        return energy_logits

    def loss(
        self,
        image: torch.Tensor,
        *,
        strategy: str = "none",
        warped_label: torch.Tensor | None = None,
        pixel_weights: torch.Tensor | None = None,
        pl_weight: float = 1.0,
    ) -> TTALossParts:
        """Calculate the TTA objective without updating the model."""
        self.seg_model.eval()

        logits = self.seg_model(image.to(self.device))

        if warped_label is not None:
            warped_label = warped_label.to(self.device)

        if pixel_weights is not None:
            pixel_weights = pixel_weights.to(self.device)

        return tta_loss(
            logits,
            self.energy_model,
            strategy=strategy,
            warped_label=warped_label,
            pixel_weights=pixel_weights,
            pl_weight=pl_weight,
        )

    def adapt(
        self,
        image: torch.Tensor,
        *,
	loss_fn=tta_loss,
        strategy: str = "none",
        prev_img=None,
        prev_label=None,
        prev_conf=None,
        registration: str = "demons",
        num_iterations: int = 10,
        lr: float = 0.01,
        variant: str = "pttea",
        min_delta: float = 1e-4,
        patience: int = 3,
    ) -> SliceResult:
        """Adapt a fresh copy of the segmentation model and return its prediction."""
        return adapt_slice(
            self.seg_model,
            self.energy_model,
            image,
	    loss_fn=loss_fn,
            strategy=strategy,
            prev_img=prev_img,
            prev_label=prev_label,
            prev_conf=prev_conf,
            registration=registration,
            num_iterations=num_iterations,
            lr=lr,
            variant=variant,
            min_delta=min_delta,
            patience=patience,
            device=self.device,
        )
