#!/usr/bin/env python3
"""
Prepare MyoPS 2020 for the ACDC->MyoPS Table 1 comparison.

Input : the challenge download folder containing train25.zip and
        train25_myops_gd.zip (flat: myops_training_101_C0.nii.gz ... and
        myops_training_101_gd.nii.gz ...).
Output: <out>/Case101_slice_00.mat ... for ALL 25 subjects, produced by
        Nicole's setup_myops.process_subject unchanged (bSSFP/C0 sequence,
        labels 500->endo, {200,1220,2221}->myo, resample to 1 mm, 256 crop
        around the mid-slice myocardium centroid).

Nicole's setup_myops.main() splits the 25 into train/val/test and keeps only
5 test cases; the PTTEA paper evaluates on all 25 (test-only), so we call
process_subject directly on every case.

Usage:
  python scripts/prep_myops.py "~/Dataset/MyoPS/MyoPS 2020 Dataset" ~/Dataset/MyoPS_Processed/all
"""
import glob
import os
import re
import sys
import types
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "external", "pseudolabel"))

# setup_myops imports sklearn only for its own main(); stub it if absent
try:
    import sklearn.model_selection  # noqa: F401
except ImportError:
    sk = types.ModuleType("sklearn"); ms = types.ModuleType("sklearn.model_selection")
    ms.train_test_split = None; sk.model_selection = ms
    sys.modules["sklearn"] = sk; sys.modules["sklearn.model_selection"] = ms

from setup_myops import process_subject  # noqa: E402


def main(src, out):
    src, out = os.path.expanduser(src), os.path.expanduser(out)
    raw = os.path.join(os.path.dirname(out.rstrip("/")), "raw")
    os.makedirs(raw, exist_ok=True); os.makedirs(out, exist_ok=True)

    for z in ("train25.zip", "train25_myops_gd.zip"):
        zp = os.path.join(src, z)
        if not os.path.exists(zp):
            sys.exit(f"missing {zp}")
        with zipfile.ZipFile(zp) as f:
            f.extractall(raw)
        print(f"unzipped {z} -> {raw}")

    c0 = sorted(glob.glob(os.path.join(raw, "**", "*_C0.nii.gz"), recursive=True))
    gd = {re.search(r"(\d+)_gd", os.path.basename(p)).group(1): p
          for p in glob.glob(os.path.join(raw, "**", "*_gd.nii.gz"), recursive=True)}
    print(f"found {len(c0)} C0 images, {len(gd)} gd labels")

    n = 0
    for img in c0:
        cid = re.search(r"(\d+)_C0", os.path.basename(img)).group(1)
        if cid not in gd:
            print(f"  no label for {cid}, skipping"); continue
        case_dir = os.path.join(raw, "cases", f"Case{cid}")
        os.makedirs(case_dir, exist_ok=True)
        for p in (img, gd[cid]):
            dst = os.path.join(case_dir, os.path.basename(p))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(p), dst)
        process_subject(case_dir, f"Case{cid}", out)
        n += 1
    mats = glob.glob(os.path.join(out, "*.mat"))
    print(f"\nprocessed {n} subjects -> {len(mats)} slices in {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
