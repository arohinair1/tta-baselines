#!/usr/bin/env python3
"""
Week 3 check: how good are the pseudolabels?

For every adjacent slice pair (t-1, t) in a dataset, warp the TRUE label of
slice t-1 into slice t with each registration method and score it against the
true label of slice t. This isolates registration quality from the
segmentation model. Methods: identity (no warp), demons, affine (both already
in Nicole's repo), and vxm (AdaCS/VoxelMorph trained on MyoPS pairs).

Also writes overlay figures: current image, warped previous label (red),
true current label (yellow), one row per method.

Usage:
  python scripts/eval_registration.py --dataset myops --data ~/Dataset/MyoPS_Processed/all \
      --flow_ckpt ~/adacs_myops/motion_0150.pt --cases Case121 Case122 Case123 Case124 Case125 \
      --out results/registration_myops
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tta_wrapper as tw  # noqa: E402
from tta_wrapper import flow as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="myops", choices=list(tw.data.LOADERS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--flow_ckpt", default=None, help="AdaCS motion_XXXX.pt; omit to skip vxm")
    ap.add_argument("--flow_cache", default=None, help="npz to load/save precomputed flows")
    ap.add_argument("--methods", nargs="*", default=["identity", "demons", "affine", "vxm"])
    ap.add_argument("--cases", nargs="*", default=None, help="restrict to these case ids (e.g. the holdout)")
    ap.add_argument("--n_fig", type=int, default=6, help="number of overlay figures")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cases = tw.data.LOADERS[a.dataset](os.path.expanduser(a.data))
    if a.cases:
        cases = {k: v for k, v in cases.items() if k in set(a.cases)}
    methods = dict(identity=lambda fixed, moving, lbl: lbl)
    if "demons" in a.methods: methods["demons"] = tw.adapt.REGISTRATION["demons"]
    if "affine" in a.methods: methods["affine"] = tw.adapt.REGISTRATION["affine"]
    if "vxm" in a.methods and a.flow_ckpt:
        methods["vxm"] = F.FlowRegistrar(os.path.expanduser(a.flow_ckpt), device=a.device,
                                         cache_file=a.flow_cache and os.path.expanduser(a.flow_cache))

    dice = {m: [] for m in methods}; secs = {m: 0.0 for m in methods}; figs = []
    for cid, slices in cases.items():
        for prev, cur in zip(slices[:-1], slices[1:]):
            row = {}
            for m, fn in methods.items():
                t0 = time.time(); w = fn(cur.image, prev.image, prev.label.astype(np.uint8)); secs[m] += time.time() - t0
                dice[m].append(tw.compute_dice(w, cur.label)); row[m] = w
            if len(figs) < a.n_fig:
                figs.append((cid, cur, prev, row))
    if "vxm" in methods and a.flow_cache:
        methods["vxm"].save_cache()

    n = len(next(iter(dice.values())))
    lines = [f"pseudolabel quality: Dice(warp(GT_{{t-1}}), GT_t) over {n} adjacent pairs, {len(cases)} cases, {a.dataset}",
             "", "| method | LV Dice | Myo Dice | fg mean | sec/pair |", "|---|---|---|---|---|"]
    summary = {}
    for m in methods:
        d = np.asarray(dice[m]).mean(0); summary[m] = {"lv": float(d[1]), "myo": float(d[2]), "sec_per_pair": secs[m] / n}
        lines.append(f"| {m} | {100*d[1]:.2f} | {100*d[2]:.2f} | {100*d[1:].mean():.2f} | {secs[m]/n:.3f} |")
    md = "\n".join(lines); print(md)
    open(os.path.join(a.out, "REGISTRATION.md"), "w").write(md + "\n")
    json.dump({"args": vars(a), "summary": summary}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for k, (cid, cur, prev, row) in enumerate(figs):
        fig, ax = plt.subplots(1, len(row) + 1, figsize=(3.2 * (len(row) + 1), 3.4))
        ax[0].imshow(prev.image, cmap="gray"); ax[0].contour(prev.label == 2, colors="r", linewidths=.6)
        ax[0].contour(prev.label == 1, colors="orange", linewidths=.6); ax[0].set_title(f"{prev.name}\n(prev, its GT)", fontsize=8)
        for i, (m, w) in enumerate(row.items(), 1):
            d = tw.compute_dice(w, cur.label)
            ax[i].imshow(cur.image, cmap="gray")
            ax[i].contour(w == 2, colors="r", linewidths=.6); ax[i].contour(w == 1, colors="orange", linewidths=.6)
            ax[i].contour(cur.label > 0, colors="yellow", linewidths=.5, linestyles="dashed")
            ax[i].set_title(f"{m}: warped prev (red/orange)\nvs GT_t (yellow)  myo {100*d[2]:.1f}", fontsize=8)
        for x in ax: x.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(a.out, f"overlay_{k:02d}_{cur.name}.png"), dpi=90); plt.close()
    print(f"\nwritten to {a.out}/ (REGISTRATION.md, summary.json, {len(figs)} overlay PNGs)")


if __name__ == "__main__":
    main()
