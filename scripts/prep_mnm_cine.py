#!/usr/bin/env python
"""
M&Ms cine -> time-frame registration data (the "frames, not slices" setup).

For every patient and every short-axis slice z where BOTH labelled frames
(ED, ES) contain myocardium, this writes:

  <out>/pairs_edes/{train,holdout}/<patient>_s<zz>.mat
        im_ED, im_ES, lbl_ED, lbl_ES, myo_ED, myo_ES      (AdaCS .mat format,
        the same keys prep_myops_pairs.py wrote, so bash/adacs_train.sh and
        scripts/diagnose_flow.py work unchanged)
  <out>/frames/{train,holdout}/<patient>_s<zz>.npz
        frames (T,256,256) float16, ed, es, lbl_ED, lbl_ES  (all time frames of
        that slice, for flow-across-all-frames / consistency checks)

Preprocessing: resample in-plane to 1 mm (same as the TTA loaders), min-max
normalise each frame, and crop 256x256 around ONE centre per slice: the midpoint
of the ED and ES myocardium centroids. Every frame of a slice uses the same
window, so, unlike the per-slice centroid crop, the heart genuinely moves
between frames and the registration has to find that motion.

Label convention (same as the rest of the repo): 0 bg, 1 LV cavity, 2 myocardium
(M&Ms RV label 3 -> 0). ED = the labelled frame with the larger LV area.

Usage (Bouchet):
  python scripts/prep_mnm_cine.py --data ~/Dataset/MnM/OpenDataset/Testing --out ~/Dataset/MnM_cine --holdout 36
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tta_wrapper._bootstrap import bootstrap  # noqa: E402

bootstrap()
from eval_tta import resample_slice, remap_mnm, center_crop, find_crop_center  # noqa: E402
from dataset import im_normalize  # noqa: E402


def process_patient(pdir: str, patient: str, crop: int, min_myo: int):
    import nibabel as nib
    img_nii = nib.load(os.path.join(pdir, f"{patient}_sa.nii.gz"))
    gt_nii = nib.load(os.path.join(pdir, f"{patient}_sa_gt.nii.gz"))
    img, gt = img_nii.get_fdata(), gt_nii.get_fdata().astype(int)
    pixdim = img_nii.header["pixdim"][1:3]
    labelled = [t for t in range(gt.shape[3]) if (gt[:, :, :, t] > 0).any()]
    if len(labelled) != 2:
        return []
    # ED = larger LV cavity
    lv = [(gt[:, :, :, t] == 1).sum() for t in labelled]
    ed, es = (labelled[0], labelled[1]) if lv[0] >= lv[1] else (labelled[1], labelled[0])
    out = []
    for z in range(img.shape[2]):
        g_ed, g_es = remap_mnm(gt[:, :, z, ed]), remap_mnm(gt[:, :, z, es])
        if (g_ed == 2).sum() < min_myo or (g_es == 2).sum() < min_myo:
            continue
        _, g_ed_r = resample_slice(img[:, :, z, ed], pixdim, label=g_ed)
        _, g_es_r = resample_slice(img[:, :, z, es], pixdim, label=g_es)
        c_ed, c_es = find_crop_center(g_ed_r, 2), find_crop_center(g_es_r, 2)
        centre = ((c_ed[0] + c_es[0]) // 2, (c_ed[1] + c_es[1]) // 2)      # ONE centre for all frames
        frames = []
        for t in range(img.shape[3]):
            f_r = im_normalize(resample_slice(img[:, :, z, t], pixdim))
            frames.append(center_crop(f_r, crop, center=centre).astype(np.float32))
        _, l_ed = center_crop(frames[ed], crop, label=g_ed_r, center=centre)
        _, l_es = center_crop(frames[es], crop, label=g_es_r, center=centre)
        out.append(dict(name=f"{patient}_s{z:02d}", frames=np.stack(frames), ed=ed, es=es,
                        lbl_ED=l_ed.astype(np.uint8), lbl_ES=l_es.astype(np.uint8)))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True, help="M&Ms dir of <patient>/<patient>_sa.nii.gz (+ _sa_gt)")
    a.add_argument("--out", required=True)
    a.add_argument("--holdout", type=int, default=36, help="last N patients (sorted) -> holdout")
    a.add_argument("--crop", type=int, default=256)
    a.add_argument("--min_myo", type=int, default=50, help="min myocardium pixels in BOTH ED and ES for a slice to be used")
    a.add_argument("--max_patients", type=int, default=0)
    args = a.parse_args()

    patients = sorted(p for p in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, p)))
    if args.max_patients:
        patients = patients[:args.max_patients]
    hold = set(patients[-args.holdout:]) if args.holdout else set()
    for split in ("train", "holdout", "val"):
        os.makedirs(os.path.join(args.out, "pairs_edes", split), exist_ok=True)   # val/ stays empty (train_vxm globs it)
    for split in ("train", "holdout"):
        os.makedirs(os.path.join(args.out, "frames", split), exist_ok=True)

    n = {"train": 0, "holdout": 0}; skipped = []
    for i, p in enumerate(patients):
        split = "holdout" if p in hold else "train"
        try:
            slices = process_patient(os.path.join(args.data, p), p, args.crop, args.min_myo)
        except Exception as e:  # noqa: BLE001
            skipped.append((p, str(e))); continue
        if not slices:
            skipped.append((p, "no slice with myocardium in both ED and ES")); continue
        for s in slices:
            fr = s["frames"]
            savemat(os.path.join(args.out, "pairs_edes", split, s["name"] + ".mat"), {
                "im_ED": fr[s["ed"]], "im_ES": fr[s["es"]],
                "lbl_ED": s["lbl_ED"], "lbl_ES": s["lbl_ES"],
                "myo_ED": (s["lbl_ED"] == 2).astype(np.uint8), "myo_ES": (s["lbl_ES"] == 2).astype(np.uint8),
                "patient": p, "z": int(s["name"][-2:]), "ed": s["ed"], "es": s["es"]})
            np.savez_compressed(os.path.join(args.out, "frames", split, s["name"] + ".npz"),
                                frames=fr.astype(np.float16), ed=s["ed"], es=s["es"],
                                lbl_ED=s["lbl_ED"], lbl_ES=s["lbl_ES"])
            n[split] += 1
        print(f"[{i+1}/{len(patients)}] {p}: {len(slices)} slices, {fr.shape[0]} frames -> {split}", flush=True)
    print(f"\nED/ES pairs: train {n['train']}, holdout {n['holdout']} ({len(hold)} patients: {sorted(hold)[:3]}...)")
    if skipped:
        print(f"skipped {len(skipped)}: " + "; ".join(f"{p} ({r})" for p, r in skipped[:5]))
    print(f"written to {args.out}/pairs_edes/{{train,holdout,val}} and {args.out}/frames/{{train,holdout}}")


if __name__ == "__main__":
    main()
