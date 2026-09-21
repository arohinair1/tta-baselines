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
