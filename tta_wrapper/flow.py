"""
Learned registration (AdaCS / VoxelMorph) as a drop-in alternative to the
Demons and affine registration already used for pseudolabels.

Upstream: external/AdaCS (Zhang et al., "Adaptive Correspondence Scoring for
Unsupervised Medical Image Registration", ECCV 2024). Its VxmDense network
takes (source, target) images and outputs a displacement field
flow (B, 2, H, W) such that SpatialTransformer(source, flow) ~= target.
Its models/ package name collides with Nicole's models/ package, so it is
imported under the name `voxelmorph` by putting external/AdaCS/models on
sys.path instead of external/AdaCS.

Conventions (match registration_utils.register_and_warp_2d):
    register_and_warp_2d(fixed_img, moving_img, moving_label) -> warped_label
    fixed  = current slice (target), moving = previous slice (source).

    Y~_t = Warp( Y^_{t-1}, D_{t-1 -> t} )        with D = flow(prev -> curr)

Nothing in external/ is edited.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ._bootstrap import ROOT, _patch_getargspec, _patch_torch_load

ADACS_DIR = os.path.join(ROOT, "external", "AdaCS")
_vxm = None


def vxm():
    """Lazy import of AdaCS's voxelmorph package (pytorch backend)."""
    global _vxm
    if _vxm is None:
        import sys
        _patch_getargspec(); _patch_torch_load()
        os.environ["VXM_BACKEND"] = "pytorch"
        os.environ["NEURITE_BACKEND"] = "pytorch"
        p = os.path.join(ADACS_DIR, "models")
        if p not in sys.path:
            sys.path.insert(0, p)
        import voxelmorph  # noqa: F401
        _vxm = voxelmorph
    return _vxm


def load_flow_model(ckpt: str, device="cpu"):
    """Load a VxmDense checkpoint written by AdaCS train_vxm.py (motion_XXXX.pt)."""
    m = vxm().networks.VxmDense.load(ckpt, device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class FlowRegistrar:
    """Callable with the register_and_warp_2d signature, backed by VxmDense.

    estimate_flow(prev, curr) -> (2, H, W) numpy, cached by image content so
    the flow for a pair is computed once (and can be precomputed / saved).
    __call__(fixed, moving, moving_label) -> warped label (nearest) or,
    if moving_label is float, warped bilinearly (for probability maps).
    """

    def __init__(self, ckpt: str, device="cpu", cache_file: Optional[str] = None):
        self.device = device
        self.model = load_flow_model(ckpt, device)
        self.shape = tuple(self.model.config["inshape"])
        L = vxm().layers.SpatialTransformer
        self.st_nearest = L(self.shape, mode="nearest").to(device)
        self.st_bilinear = L(self.shape, mode="bilinear").to(device)
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            self.cache = dict(np.load(cache_file))

    @staticmethod
    def _key(prev: np.ndarray, curr: np.ndarray) -> str:
        h = hashlib.md5(); h.update(np.ascontiguousarray(prev, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(curr, dtype=np.float32).tobytes())
        return h.hexdigest()

    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None].to(self.device)

    def estimate_flow(self, prev_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
        k = self._key(prev_img, curr_img)
        if k not in self.cache:
            with torch.no_grad():
                _, flow, _ = self.model(self._t(prev_img), self._t(curr_img), registration=True)
            self.cache[k] = flow[0].cpu().numpy()
        return self.cache[k]

    def warp(self, moving: np.ndarray, flow: np.ndarray, nearest: bool = True) -> np.ndarray:
        """moving: (H,W) label or image, or (C,H,W) probabilities. flow: (2,H,W)."""
        st = self.st_nearest if nearest else self.st_bilinear
        x = torch.from_numpy(np.asarray(moving, dtype=np.float32))
        x = x[None, None] if x.ndim == 2 else x[None]
        f = torch.from_numpy(flow)[None].to(self.device)
        with torch.no_grad():
            out = st(x.to(self.device), f)[0].cpu().numpy()
        return out[0] if moving.ndim == 2 else out

    def __call__(self, fixed_img, moving_img, moving_label):
        flow = self.estimate_flow(moving_img, fixed_img)
        is_label = np.issubdtype(np.asarray(moving_label).dtype, np.integer)
        out = self.warp(moving_label, flow, nearest=is_label)
        return out.astype(np.asarray(moving_label).dtype) if is_label else out

    def save_cache(self, path: Optional[str] = None):
        path = path or self.cache_file
        if path:
            np.savez_compressed(path, **self.cache)
        return path


def register(name: str, registrar: FlowRegistrar):
    """Expose a registrar to adapt.REGISTRATION so adapt_slice(registration=name) works."""
    from . import adapt
    adapt.REGISTRATION[name] = registrar
