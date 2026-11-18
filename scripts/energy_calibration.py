#!/usr/bin/env python
"""
Is the energy model an oracle for pseudolabel quality?  (Prof. Wong, 11/17)

For every ED/ES pair with labels, build the pseudolabel the pipeline would use
(the source model's ED prediction warped to ES by the flow model; also the GT_ED
warped, and GT_ED unwarped = identity) and feed it through the energy model.
The energy model scores each 16x16 patch: sigmoid(E) ~ "this patch looks like a
clean ground-truth patch". Compare that score, patch by patch, with how correct
the patch ACTUALLY is against GT_ES (patch IoU, the same criterion the energy
model was trained with). If score and correctness agree, energy can be used as a
per-pixel weight on the pseudolabel loss at test time, where there is no GT.

Reports, per pseudolabel source: mean energy, Dice vs GT_ES, Spearman / Pearson
correlation between patch score and patch IoU (patches touching the heart only),
and AUC for "patch IoU > 0.6". Writes CALIBRATION.md, calibration.csv, a scatter
PNG and a few overlay PNGs.

Usage (Bouchet):
  python scripts/energy_calibration.py --pairs ~/Dataset/MnM_cine/pairs_edes/train --flow_ckpt ~/adacs_mnm/motion_0160.pt \
      --out results/energy_calib --device cuda --max_pairs 300
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tta_wrapper as tw  # noqa: E402
from tta_wrapper import flow as F  # noqa: E402
from tta_wrapper.metrics import compute_dice  # noqa: E402

N_BLOCKS = 16


def onehot(lbl, n=3):
    return np.stack([(lbl == c) for c in range(n)]).astype(np.float32)


def energy_map(en, lbl, device):
    """sigmoid patch scores (16,16) of a hard label."""
    x = torch.from_numpy(onehot(lbl))[None].to(device)
    with torch.no_grad():
        return torch.sigmoid(en(x))[0, 0].cpu().numpy()


def patch_iou(pred, gt, n=N_BLOCKS):
    """soft-IoU per patch over the one-hot channels, as in utils.create_labels_iou."""
    a = torch.from_numpy(onehot(pred))[None]; b = torch.from_numpy(onehot(gt))[None]
    H = a.shape[-1]; k = H // n
    A = Fn.unfold(a, kernel_size=k, stride=k); B = Fn.unfold(b, kernel_size=k, stride=k)
    inter = (A * B).sum(1); union = (A + B - A * B).sum(1)
    return (inter / (union + 1e-6)).view(n, n).numpy()


def patch_has_heart(pred, gt, n=N_BLOCKS):
    m = torch.from_numpy(((pred > 0) | (gt > 0)).astype(np.float32))[None, None]
    k = m.shape[-1] // n
    return (Fn.unfold(m, kernel_size=k, stride=k).sum(1).view(n, n).numpy() > 0)


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score)) if 0 < label.mean() < 1 else float("nan")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pairs", required=True); a.add_argument("--flow_ckpt", required=True)
    a.add_argument("--out", default="results/energy_calib"); a.add_argument("--device", default="cpu")
    a.add_argument("--max_pairs", type=int, default=0); a.add_argument("--n_vis", type=int, default=4)
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from scipy.io import loadmat
    from scipy.stats import spearmanr, pearsonr

    seg = tw.load_seg_model(device=args.device); en = tw.load_energy_model(device=args.device)
    reg = F.FlowRegistrar(os.path.expanduser(args.flow_ckpt), device=args.device)
    files = sorted(glob.glob(os.path.join(args.pairs, "*.mat")))
    if args.max_pairs:
        files = files[:args.max_pairs]
    print(f"{len(files)} pairs")

    rows = []        # per patch
    per_pair = []
    for i, f in enumerate(files):
        m = loadmat(f); n = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        ed, es = n(m["im_ED"].astype(np.float32)), n(m["im_ES"].astype(np.float32))
        g_ed, g_es = m["lbl_ED"].astype(np.uint8), m["lbl_ES"].astype(np.uint8)
        pred_ed = tw.adapt.predict_no_adapt(seg, torch.from_numpy(ed[None, None]), device=args.device).pred.astype(np.uint8)
        pred_es = tw.adapt.predict_no_adapt(seg, torch.from_numpy(es[None, None]), device=args.device).pred.astype(np.uint8)
        sources = {
            "warp(pred_ED)  [pipeline pseudolabel]": reg(es, ed, pred_ed),
            "warp(GT_ED)": reg(es, ed, g_ed),
            "GT_ED unwarped (identity)": g_ed,
            "pred_ES (model's own output at ES)": pred_es,
            "GT_ES (ceiling)": g_es,
        }
        for name, lbl in sources.items():
            s = energy_map(en, lbl, args.device); iou = patch_iou(lbl, g_es); keep = patch_has_heart(lbl, g_es)
            d = compute_dice(lbl, g_es)[1:]
            per_pair.append(dict(pair=os.path.basename(f), source=name, energy_mean=float(s.mean()), energy_heart=float(s[keep].mean()) if keep.any() else np.nan,
                                 dice_lv=d[0], dice_myo=d[1], patch_iou_heart=float(iou[keep].mean()) if keep.any() else np.nan))
            for (r, c) in zip(*np.where(keep)):
                rows.append(dict(pair=i, source=name, score=float(s[r, c]), iou=float(iou[r, c])))
            if i < args.n_vis and name.startswith("warp(pred"):
                save_fig(args.out, f"{i:02d}_{os.path.basename(f)[:-4]}", es, lbl, g_es, s, iou)
        if i % 50 == 0:
            print(f"[{i+1}/{len(files)}]", flush=True)

    import pandas as pd
    R = pd.DataFrame(rows); P = pd.DataFrame(per_pair)
    R.to_csv(os.path.join(args.out, "calibration_patches.csv"), index=False); P.to_csv(os.path.join(args.out, "calibration_pairs.csv"), index=False)
    md = [f"# Energy model as a pseudolabel oracle  ({len(files)} pairs, {args.pairs})", "",
          "Per-patch (16x16) sigmoid energy score vs. the patch's actual IoU against GT_ES, over patches touching the heart.", "",
          "| pseudolabel source | n patches | mean energy (heart) | Dice LV / Myo vs GT_ES | mean patch IoU | Spearman | Pearson | AUC (IoU>0.6) |", "|---|---|---|---|---|---|---|---|"]
    for name in sources:
        r = R[R.source == name]; p = P[P.source == name]
        sp = spearmanr(r.score, r.iou).correlation; pe = pearsonr(r.score, r.iou)[0]
        md.append(f"| {name} | {len(r)} | {p.energy_heart.mean():.3f} | {100*p.dice_lv.mean():.1f} / {100*p.dice_myo.mean():.1f} | {p.patch_iou_heart.mean():.3f} | {sp:.3f} | {pe:.3f} | {auc(r.score.values, (r.iou > 0.6).values):.3f} |")
    # pair-level: does mean energy of the pseudolabel predict its Dice?
    p = P[P.source.str.startswith("warp(pred")]
    md += ["", f"Pair level, pipeline pseudolabel: Spearman(mean heart energy, Myo Dice) = {spearmanr(p.energy_heart, p.dice_myo).correlation:.3f}, "
           f"Spearman(mean heart energy, LV Dice) = {spearmanr(p.energy_heart, p.dice_lv).correlation:.3f}", "",
           "Reading: Spearman/AUC near 0.5-0.6 = energy barely knows which patches are wrong; > 0.7 = usable as a per-pixel weight.",
           "The GT_ES row is the ceiling (how high energy goes on a perfect label); identity vs warp shows whether energy notices the warp helped."]
    text = "\n".join(md); print("\n" + text); open(os.path.join(args.out, "CALIBRATION.md"), "w").write(text + "\n")
    scatter(args.out, R)
    print(f"\nwritten to {args.out}/")


def save_fig(out, name, img, lbl, gt, s, iou):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(img, cmap="gray"); ax[0].contour(gt == 2, [0.5], colors="cyan"); ax[0].contour(gt == 1, [0.5], colors="lime")
    ax[0].contour(lbl == 2, [0.5], colors="magenta", linestyles="dashed"); ax[0].contour(lbl == 1, [0.5], colors="red", linestyles="dashed"); ax[0].set_title("GT_ES (solid) vs pseudolabel (dashed)")
    ax[1].imshow(img, cmap="gray"); ax[1].imshow(np.kron(s, np.ones((16, 16))), cmap="RdYlGn", alpha=0.5, vmin=0, vmax=1); ax[1].set_title("energy score per patch (green = clean)")
    ax[2].imshow(img, cmap="gray"); ax[2].imshow(np.kron(iou, np.ones((16, 16))), cmap="RdYlGn", alpha=0.5, vmin=0, vmax=1); ax[2].set_title("actual patch IoU vs GT_ES")
    ax[3].imshow((lbl != gt) & ((lbl > 0) | (gt > 0)), cmap="hot"); ax[3].set_title("pseudolabel errors")
    for x in ax: x.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out, f"{name}.png"), dpi=70); plt.close()


def scatter(out, R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    srcs = [s for s in R.source.unique() if s.startswith("warp(pred")]
    r = R[R.source == srcs[0]]
    plt.figure(figsize=(5, 5)); plt.scatter(r.iou, r.score, s=3, alpha=0.2); plt.xlabel("patch IoU vs GT_ES"); plt.ylabel("energy score (sigmoid)")
    plt.title("pipeline pseudolabel, heart patches"); plt.tight_layout(); plt.savefig(os.path.join(out, "scatter.png"), dpi=90); plt.close()


if __name__ == "__main__":
    main()
