#!/usr/bin/env python
"""
Consecutive-frame training pairs from the frame sequences written by prep_mnm_cine.py.

The flow model trained on ED<->ES pairs only has seen large motion. Chaining it
frame by frame (chain_flow.py) needs it to be accurate on SMALL motion too, so
this adds pairs (t, t+k) for small k, in the same AdaCS .mat format (im_ED = source
frame, im_ES = target frame; no labels needed for unsupervised training).

It also builds a mixed training dir of symlinks (ED/ES pairs + consecutive pairs)
that bash/adacs_train.sh can point at directly.

Usage (Bouchet):
  python scripts/prep_mnm_consec_pairs.py --frames ~/Dataset/MnM_cine/frames/train \
      --edes ~/Dataset/MnM_cine/pairs_edes/train --out ~/Dataset/MnM_cine/pairs_mixed --per_slice 6 --ks 1 2 3
  sbatch bash/adacs_train.sh ~/Dataset/MnM_cine/pairs_mixed ~/adacs_mnm_mixed 150
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.io import savemat


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True)
    a.add_argument("--edes", required=True, help="pairs_edes/train dir to include via symlinks")
    a.add_argument("--out", required=True, help="output root; writes <out>/consec/ and <out>/{train,val}/")
    a.add_argument("--per_slice", type=int, default=6, help="consecutive pairs sampled per slice")
    a.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3], help="frame gaps to sample")
    a.add_argument("--seed", type=int, default=0)
    args = a.parse_args()
    rng = np.random.default_rng(args.seed)
    consec = os.path.join(args.out, "consec"); train = os.path.join(args.out, "train"); val = os.path.join(args.out, "val")
    for d in (consec, train, val):
        os.makedirs(d, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.frames, "*.npz")))
    n = 0
    for f in files:
        z = np.load(f); fr = z["frames"].astype(np.float32); T = fr.shape[0]
        name = os.path.basename(f)[:-4]
        for _ in range(args.per_slice):
            k = int(rng.choice(args.ks)); t = int(rng.integers(0, T))
            u = (t + k) % T
            if rng.random() < 0.5:      # both temporal directions
                t, u = u, t
            p = os.path.join(consec, f"{name}_t{t:02d}_t{u:02d}.mat")
            savemat(p, {"im_ED": fr[t], "im_ES": fr[u], "k": k})
            n += 1
    # mixed train dir = symlinks to ED/ES pairs + consecutive pairs
    for d in (os.path.abspath(args.edes), consec):
        for p in sorted(glob.glob(os.path.join(d, "*.mat"))):
            dst = os.path.join(train, os.path.basename(p))
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(p), dst)
    print(f"{len(files)} slices -> {n} consecutive pairs in {consec}")
    print(f"mixed train dir {train}: {len(os.listdir(train))} pairs ({len(os.listdir(args.edes))} ED/ES + {n} consecutive); val/ empty")


if __name__ == "__main__":
    main()
