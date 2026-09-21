"""Checkpoint loading. Thin wrappers over the repo's LoadableModel.load."""
from __future__ import annotations

import torch

from ._bootstrap import bootstrap, SEG_CKPT, ENERGY_CKPT

bootstrap()

from models.segmentation import Unet3DSeg          # noqa: E402
from models.energy import ShapeEnergyModel          # noqa: E402


def load_seg_model(path: str = SEG_CKPT, device="cpu") -> Unet3DSeg:
    """1-channel UNet segmentation model (3 classes: bg / LV endo / myo).
    Upstream checkpoint config: numclasses=3, inshape=(256,256)."""
    m = Unet3DSeg.load(path, device)
    m.eval()
    return m


def load_energy_model(path: str = ENERGY_CKPT, device="cpu") -> ShapeEnergyModel:
    """Patch-wise shape energy model. Input softmax probs (B,3,H,W),
    output logits (B,1,n,n). Upstream checkpoint: patch_size=16 on 256x256
    -> n_downsample=16 branch -> 16x16 patch grid."""
    m = ShapeEnergyModel.load(path, device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def checkpoint_config(path: str) -> dict:
    return torch.load(path, map_location="cpu")["config"]
