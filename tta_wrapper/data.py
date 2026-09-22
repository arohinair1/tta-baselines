"""Loading + preprocessing for the processed ACDC .mat slices
(keys: im, endo_seg, myo_seg; 256x256). Mirrors eval_tta.eval_acdc_tta's
.mat branch and dataset.load_data_mri_2d: min-max normalise, build label
(myo=2, endo=1), centre-crop 256 around the myocardium centroid (GT-derived,
exactly as the repo does), group files by patient in sorted order."""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from scipy.io import loadmat

from ._bootstrap import bootstrap, EXAMPLE_DATA

bootstrap()
from dataset import im_normalize, crop_to_centroid   # noqa: E402


@dataclass
class Slice:
    name: str
    case: str
    image: np.ndarray   # (H, W) float in [0,1]
    label: np.ndarray   # (H, W) int {0,1,2}

    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.image[np.newaxis, np.newaxis]).float()


def case_id(fname: str) -> str:
    m = re.match(r"(patient\d+)", fname)
    return m.group(1) if m else fname.split(".")[0]


def load_mat_slice(path: str, crop_size: int = 256) -> Slice:
    mat = loadmat(path)
    img = im_normalize(mat["im"].astype(float))
    gt = np.zeros_like(mat["myo_seg"], dtype=int)
    gt[mat["myo_seg"] > 0.5] = 2
    gt[mat["endo_seg"] > 0.5] = 1
    img_c, gt_c = crop_to_centroid(img, gt, crop_size=crop_size)
    name = os.path.basename(path)
    return Slice(name=name, case=case_id(name), image=img_c, label=gt_c)


def load_mat_dir(data_dir: str = EXAMPLE_DATA, crop_size: int = 256) -> "OrderedDict[str, List[Slice]]":
    """Returns {case_id: [Slice, ...]} with files in sorted order, the same
    order eval_tta uses to define 'previous slice'."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    cases: Dict[str, List[Slice]] = OrderedDict()
    for f in files:
        s = load_mat_slice(os.path.join(data_dir, f), crop_size)
        cases.setdefault(s.case, []).append(s)
    return cases


# ──────────────────────────────────────────────────────────────────────
# Target datasets for PTTEA Table 1 (ACDC source -> LVQuant / MyoPS / M&M)
# Each loader mirrors the corresponding branch of eval_tta.eval_*_tta
# (same resampling to 1mm, same remap, same 256 crop, same slice order),
# reusing eval_tta's own helper functions so the preprocessing is identical.
# ──────────────────────────────────────────────────────────────────────
from eval_tta import (resample_slice, remap_mnm, remap_lvquant,   # noqa: E402
                      center_crop, find_crop_center)


def load_mnm_dir(data_dir: str, crop_size: int = 256) -> "OrderedDict[str, List[Slice]]":
    """M&Ms raw: <patient>/<patient>_sa.nii.gz + _sa_gt.nii.gz (H,W,S,T).
    One case per labeled frame (ED/ES), slices in z order. Mirrors eval_mnm_tta."""
    import nibabel as nib
    cases: Dict[str, List[Slice]] = OrderedDict()
    for patient in sorted(p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))):
        pdir = os.path.join(data_dir, patient)
        img_nii = nib.load(os.path.join(pdir, f"{patient}_sa.nii.gz"))
        gt_nii = nib.load(os.path.join(pdir, f"{patient}_sa_gt.nii.gz"))
        img_vol, gt_vol = img_nii.get_fdata(), gt_nii.get_fdata().astype(int)
        pixdim = img_nii.header["pixdim"][1:3]
        for t in [t for t in range(gt_vol.shape[3]) if len(np.unique(gt_vol[:, :, :, t])) > 1]:
            case = f"{patient}_t{t:02d}"
            for s in range(img_vol.shape[2]):
                img_r, gt_r = resample_slice(img_vol[:, :, s, t], pixdim, label=gt_vol[:, :, s, t])
                img_r = im_normalize(img_r)
                gt_r = remap_mnm(gt_r)
                img_c, gt_c = crop_to_centroid(img_r, gt_r, crop_size=crop_size)
                cases.setdefault(case, []).append(Slice(f"{case}_s{s:02d}", case, img_c, gt_c))
    return cases


def load_lvquant_dir(data_dir: str, crop_size: int = 256) -> "OrderedDict[str, List[Slice]]":
    """LVQuant train .mat: image/endo/epi (H,W,20 frames) + pix_spacing.
    One case per file, 20 frames in order, crop centred on the frame with the
    most myocardium. Mirrors _eval_lvquant_tta_with_gt."""
    cases: Dict[str, List[Slice]] = OrderedDict()
    for fname in sorted(f for f in os.listdir(data_dir) if f.endswith(".mat")):
        mat = loadmat(os.path.join(data_dir, fname))
        img_vol, endo, epi = mat["image"], mat["endo"], mat["epi"]
        pix_sp = np.array([mat["pix_spacing"].flatten()[0]] * 2)
        n = img_vol.shape[2]
        myo_counts = [(remap_lvquant(endo[:, :, t], epi[:, :, t]) == 2).sum() for t in range(n)]
        ref = int(np.argmax(myo_counts))
        ref_lbl = remap_lvquant(endo[:, :, ref], epi[:, :, ref])
        _, ref_lbl_r = resample_slice(ref_lbl.astype(float), pix_sp, label=ref_lbl)
        centre = find_crop_center(ref_lbl_r, class_id=2)
        case = os.path.splitext(fname)[0]
        for t in range(n):
            lbl = remap_lvquant(endo[:, :, t], epi[:, :, t])
            img_r, lbl_r = resample_slice(img_vol[:, :, t].astype(float), pix_sp, label=lbl)
            img_r = im_normalize(img_r)
            img_c, lbl_c = center_crop(img_r, crop_size, label=lbl_r, center=centre)
            cases.setdefault(case, []).append(Slice(f"{case}_t{t:02d}", case, img_c, lbl_c))
    return cases


def load_myops_dir(data_dir: str, crop_size: int = 256) -> "OrderedDict[str, List[Slice]]":
    """MyoPS processed by setup_myops.py: CaseXXX_slice_YY.mat with im/endo_seg/myo_seg.
    Mirrors eval_myops_tta (case id = text before '_slice')."""
    cases: Dict[str, List[Slice]] = OrderedDict()
    for fname in sorted(f for f in os.listdir(data_dir) if f.endswith(".mat")):
        mat = loadmat(os.path.join(data_dir, fname))
        img = im_normalize(mat["im"].astype(float))
        gt = np.zeros_like(img, dtype=int)
        if "myo_seg" in mat: gt[mat["myo_seg"] > 0.5] = 2
        if "endo_seg" in mat: gt[mat["endo_seg"] > 0.5] = 1
        img_c, gt_c = crop_to_centroid(img, gt, crop_size=crop_size)
        case = fname.split("_slice")[0]
        cases.setdefault(case, []).append(Slice(fname, case, img_c, gt_c))
    return cases


LOADERS = {"acdc_mat": load_mat_dir, "mnm": load_mnm_dir, "lvquant": load_lvquant_dir, "myops": load_myops_dir}
