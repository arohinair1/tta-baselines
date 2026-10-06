#!/usr/bin/env python3
"""
Turn the processed MyoPS slices (CaseXXX_slice_YY.mat from prep_myops.py) into
adjacent-slice PAIRS in the .mat format AdaCS's train_vxm.py / test_vxm.py read:
    im_ED, im_ES   : (H, W) images         (here: slice s, slice s+1)
    myo_ED, myo_ES : (H, W) binary myocardium (only used by test_vxm.py)
    lbl_ED, lbl_ES : (H, W) full 3-class labels (ours, for eval_registration.py)

Writes <out>/train/*.mat and <out>/holdout/*.mat. train_vxm.py trains on
train/*.mat + val/*.mat, so holdout cases never touch registration training
and remain a clean test set for the TTA experiments. Registration training is
unsupervised: labels are written for evaluation only.

Usage:
  python scripts/prep_myops_pairs.py ~/Dataset/MyoPS_Processed/all ~/Dataset/MyoPS_pairs --holdout 5
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.io import loadmat, savemat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--holdout", type=int, default=5, help="last N cases (sorted) held out of training")
    a = ap.parse_args()
    src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
    for d in ("train", "val", "holdout"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    by_case = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(src, "*.mat"))):
        m = re.match(r"(Case\d+)_slice_(\d+)\.mat", os.path.basename(f))
        by_case[m.group(1)].append((int(m.group(2)), f))
    cases = sorted(by_case)
    hold = set(cases[-a.holdout:]) if a.holdout else set()
    n = {"train": 0, "holdout": 0}
    for c in cases:
        slices = sorted(by_case[c])
        split = "holdout" if c in hold else "train"
        for (i, fa), (j, fb) in zip(slices[:-1], slices[1:]):
            if j != i + 1:
                continue  # non-adjacent (a slice without labels was dropped); skip
            A, B = loadmat(fa), loadmat(fb)
            lblA = np.zeros_like(A["im"], dtype=np.uint8); lblA[A["myo_seg"] > .5] = 2; lblA[A["endo_seg"] > .5] = 1
            lblB = np.zeros_like(B["im"], dtype=np.uint8); lblB[B["myo_seg"] > .5] = 2; lblB[B["endo_seg"] > .5] = 1
            savemat(os.path.join(out, split, f"{c}_s{i:02d}_s{j:02d}.mat"), {
                "im_ED": A["im"].astype(np.float32), "im_ES": B["im"].astype(np.float32),
                "myo_ED": (A["myo_seg"] > .5).astype(np.uint8), "myo_ES": (B["myo_seg"] > .5).astype(np.uint8),
                "lbl_ED": lblA, "lbl_ES": lblB, "case": c, "s0": i, "s1": j})
            n[split] += 1
    print(f"{len(cases)} cases -> train pairs: {n['train']} ({len(cases)-len(hold)} cases), "
          f"holdout pairs: {n['holdout']} ({sorted(hold)})")
    print(f"written to {out}/{{train,holdout}}  (val/ left empty on purpose)")


if __name__ == "__main__":
    main()
