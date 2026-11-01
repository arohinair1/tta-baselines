#!/usr/bin/env python
"""
Week 5: sequential test-time adaptation over cine TIME FRAMES (M&Ms).

For every held-out slice sequence (frames/<split>/*.npz from prep_mnm_cine.py) the
model walks from the start frame (ED) to the end frame (ES), every `--stride`-th
frame, adapting on each visited frame. The pseudolabel for frame t is the
prediction made at the previously visited frame, warped forward with the learned
flow (stride chosen from chain_flow.py: 5 frames beats 1-frame chaining and the
one-shot warp). Ground truth exists at ED and ES only, so Dice is scored at both
ends of the sequence; everything in between is logged for the Week 6 dataset.

Settings (the four comparisons of the plan):
  no_adapt   source model, no adaptation
  energy     PTTEA energy loss only                              L = L_E
  pl         pseudolabel loss only                               L = L_P
  energy_pl  both                                                L = lambda_E L_E + lambda_P L_P
plus options: --trust (weight L_P by a per-pixel agreement mask: the pseudolabel from
the previous visited frame vs. the one from TWO frames back, both warped to t; pixels
where they disagree get weight 0), --start gt (oracle: propagate from the ED ground
truth instead of the ED prediction, isolating propagation quality), --carry_state.

Outputs <out>/RESULTS.md, metrics.json, per_frame.csv (one row per visited frame per
setting: energy score before/after, pseudolabel loss, displacement, trust fraction,
Dice where GT exists), and a few overlay PNGs.

Usage (Bouchet, GPU):
  python scripts/run_cine_tta.py --frames ~/Dataset/MnM_cine/frames/holdout --flow_ckpt ~/adacs_mnm/motion_0160.pt \
      --out results/cine_tta --device cuda --stride 5 --energy_mask
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tta_wrapper as tw  # noqa: E402
from tta_wrapper import flow as F  # noqa: E402
from tta_wrapper.metrics import compute_dice  # noqa: E402

SETTINGS = {
    #            strategy  lambda_e lambda_p
    "no_adapt": None,
    "energy":   ("none", 1.0, 0.0),
    "pl":       ("hard", 0.0, 1.0),
    "energy_pl": ("hard", 1.0, 1.0),
}


def frame_path(a, b, T, stride):
    out, t = [a], a
    while t != b:
        t = (t + 1) % T
        out.append(t)
    if stride > 1:
        sub = out[::stride]
        out = sub + ([out[-1]] if out[-1] != sub[-1] else [])
    return out


def run_sequence(name, seg, en, reg: F.FlowRegistrar, frames, path, lbl_start, lbl_end, args):
    """Walk `path`; returns per-frame rows and the final prediction."""
    cfg = SETTINGS[name]
    rows, preds, imgs = [], [], []
    state = {} if (args.carry_state and cfg) else None
    for i, t in enumerate(path):
        x = torch.from_numpy(frames[t][None, None]).float()
        prev_img = imgs[-1] if imgs else None
        prev_lbl = preds[-1] if preds else None
        if i == 0 and args.start == "gt":
            prev_img, prev_lbl = None, None          # first frame: no pseudolabel; GT is injected below
        mask, trust_frac = None, float("nan")
        if args.trust and cfg and cfg[2] > 0 and len(preds) >= 2:
            # agreement between the pseudolabel from one step back and from two steps back
            a = reg(frames[t], imgs[-1], preds[-1]); b = reg(frames[t], imgs[-2], preds[-2])
            mask = (a == b).astype(np.float32)
            heart = (a > 0) | (b > 0)
            trust_frac = float(mask[heart].mean()) if heart.any() else float("nan")
        t0 = time.time()
        if cfg is None:
            r = tw.adapt.predict_no_adapt(seg, x, device=args.device)
            e_before = e_after = float("nan"); pl_loss = float("nan"); iters = 0
        else:
            strategy, le, lp = cfg
            r = tw.adapt.adapt_slice(seg, en, x, strategy=strategy if prev_lbl is not None else "none",
                                     prev_img=prev_img, prev_label=prev_lbl, registration="vxm",
                                     num_iterations=args.num_iterations, lr=args.lr, variant="eval_tta",
                                     device=args.device, lambda_e=le, lambda_p=lp, pl_kind=args.pl_kind,
                                     lambda_mode=args.lambda_mode, state=state, pixel_mask=mask)
            if args.energy_mask:
                r.pred = tw.adapt.energy_mask_pred(r.probs.to(args.device), en)
            e_before = r.losses[0]["energy_score"]; e_after = r.losses[-1]["energy_score"]
            pl_loss = r.losses[-1]["pseudolabel"]; iters = r.iters_run
        pred = r.pred.astype(np.uint8)
        if i == 0 and args.start == "gt":
            pred = lbl_start.copy()                     # oracle start
        disp = float("nan")
        if prev_img is not None:
            fl = reg.estimate_flow(prev_img, frames[t]); mag = np.hypot(fl[0], fl[1])
            disp = float(mag[pred > 0].mean()) if (pred > 0).any() else float("nan")
        d_start = compute_dice(pred, lbl_start)[1:] if i == 0 else [np.nan, np.nan]
        d_end = compute_dice(pred, lbl_end)[1:] if i == len(path) - 1 else [np.nan, np.nan]
        rows.append(dict(setting=name, step=i, frame=int(t), n_steps=len(path) - 1,
                         energy_before=e_before, energy_after=e_after, pl_loss=pl_loss, iters=iters,
                         disp_px=disp, trust_frac=trust_frac, fg_px=int((pred > 0).sum()),
                         dice_lv_start=d_start[0], dice_myo_start=d_start[1],
                         dice_lv_end=d_end[0], dice_myo_end=d_end[1], sec=time.time() - t0))
        preds.append(pred); imgs.append(frames[t])
    return rows, preds


def save_fig(out, name, frames, path, preds, lbl_start, lbl_end):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(path); fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    ax = np.atleast_1d(ax)
    for i, (t, p) in enumerate(zip(path, preds)):
        ax[i].imshow(frames[t], cmap="gray", vmin=0, vmax=1)
        ax[i].contour(p == 1, [0.5], colors="red"); ax[i].contour(p == 2, [0.5], colors="magenta")
        gt = lbl_start if i == 0 else (lbl_end if i == n - 1 else None)
        if gt is not None:
            ax[i].contour(gt == 1, [0.5], colors="lime", linestyles="dashed"); ax[i].contour(gt == 2, [0.5], colors="cyan", linestyles="dashed")
        ax[i].set_title(f"t={t}" + ("  (GT dashed)" if gt is not None else "")); ax[i].axis("off")
    plt.tight_layout(); p = os.path.join(out, f"{name}.png"); plt.savefig(p, dpi=70); plt.close(); return p


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True)
    a.add_argument("--flow_ckpt", required=True)
    a.add_argument("--seg_ckpt", default=None); a.add_argument("--energy_ckpt", default=None)
    a.add_argument("--out", default="results/cine_tta")
    a.add_argument("--device", default="cpu")
    a.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a.add_argument("--stride", type=int, default=5)
    a.add_argument("--directions", default="ed2es", choices=["both", "ed2es", "es2ed"])
    a.add_argument("--start", default="pred", choices=["pred", "gt"])
    a.add_argument("--trust", action="store_true")
    a.add_argument("--carry_state", action="store_true")
    a.add_argument("--energy_mask", action="store_true")
    a.add_argument("--num_iterations", type=int, default=10); a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--pl_kind", default="ce"); a.add_argument("--lambda_mode", default="fixed")
    a.add_argument("--max_slices", type=int, default=0); a.add_argument("--n_vis", type=int, default=3)
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seg = tw.load_seg_model(args.seg_ckpt or tw._bootstrap.SEG_CKPT, args.device)
    en = tw.load_energy_model(args.energy_ckpt or tw._bootstrap.ENERGY_CKPT, args.device)
    reg = F.FlowRegistrar(os.path.expanduser(args.flow_ckpt), device=args.device)
    tw.adapt.REGISTRATION["vxm"] = reg

    files = sorted(glob.glob(os.path.join(args.frames, "*.npz")))
    if args.max_slices:
        files = files[:args.max_slices]
    print(f"{len(files)} slices, settings {args.settings}, stride {args.stride}, start={args.start}, trust={args.trust}", flush=True)

    all_rows, vis = [], 0
    end_preds = {s: {} for s in args.settings}; end_lbls = {}       # per patient+direction volumes
    for k, fpath in enumerate(files):
        z = np.load(fpath); frames = z["frames"].astype(np.float32); T = frames.shape[0]
        ed, es = int(z["ed"]), int(z["es"]); l_ed, l_es = z["lbl_ED"], z["lbl_ES"]
        sl = os.path.basename(fpath)[:-4]; pat = sl.split("_")[0]
        runs = []
        if args.directions in ("both", "ed2es"): runs.append(("ed2es", frame_path(ed, es, T, args.stride), l_ed, l_es))
        if args.directions in ("both", "es2ed"): runs.append(("es2ed", frame_path(es, ed, T, args.stride), l_es, l_ed))
        for d, path, l0, l1 in runs:
            key = f"{pat}_{d}"
            end_lbls.setdefault(key, []).append(l1)
            for s in args.settings:
                rows, preds = run_sequence(s, seg, en, reg, frames, path, l0, l1, args)
                for r in rows:
                    r.update(slice=sl, patient=pat, direction=d)
                all_rows += rows
                end_preds[s].setdefault(key, []).append(preds[-1])
                if vis < args.n_vis * len(args.settings):
                    save_fig(args.out, f"{s}_{d}_{sl}", frames, path, preds, l0, l1); vis += 1
        if k % 10 == 0:
            last = {s: [r for r in all_rows if r["setting"] == s and r["slice"] == sl and r["dice_myo_end"] == r["dice_myo_end"]] for s in args.settings}
            print(f"[{k+1}/{len(files)}] {sl}  end-frame Myo: " + "  ".join(f"{s} {np.mean([r['dice_myo_end'] for r in v]):.1f}" for s, v in last.items() if v), flush=True)

    # summary
    md = [f"# Sequential TTA over cine frames  ({len(files)} slices, {args.frames})",
          f"stride {args.stride}, start={args.start}, trust={args.trust}, carry_state={args.carry_state}, energy_mask={args.energy_mask}, "
          f"lr {args.lr}, {args.num_iterations} iters, pl_kind {args.pl_kind}, lambda_mode {args.lambda_mode}, flow {args.flow_ckpt}", "",
          "Dice at the START frame = plain per-frame TTA (no pseudolabel possible yet). Dice at the END frame = after walking the sequence.", "",
          "| setting | start LV | start Myo | end LV (slice) | end Myo (slice) | end LV (vol) | end Myo (vol) | mean disp px | trust % | sec/frame |", "|---|---|---|---|---|---|---|---|---|---|"]
    metrics = {}
    for s in args.settings:
        rs = [r for r in all_rows if r["setting"] == s]
        g = lambda key: float(np.nanmean([r[key] for r in rs]))
        vol = np.array([compute_dice(np.stack(end_preds[s][k]), np.stack(end_lbls[k]))[1:] for k in end_lbls]).mean(0)
        metrics[s] = dict(start_lv=g("dice_lv_start"), start_myo=g("dice_myo_start"), end_lv=g("dice_lv_end"), end_myo=g("dice_myo_end"),
                          end_lv_vol=float(vol[0]), end_myo_vol=float(vol[1]), disp=g("disp_px"), trust=g("trust_frac"), sec=g("sec"))
        m = metrics[s]
        md.append(f"| {s} | {100*m['start_lv']:.2f} | {100*m['start_myo']:.2f} | {100*m['end_lv']:.2f} | {100*m['end_myo']:.2f} | "
                  f"{100*m['end_lv_vol']:.2f} | {100*m['end_myo_vol']:.2f} | {m['disp']:.2f} | {100*m['trust']:.0f} | {m['sec']:.2f} |")
    text = "\n".join(md); print("\n" + text)
    open(os.path.join(args.out, "RESULTS.md"), "w").write(text + "\n")
    json.dump({"args": vars(args), "metrics": metrics}, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(args.out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\nwritten to {args.out}/ (RESULTS.md, metrics.json, per_frame.csv, PNGs)")


if __name__ == "__main__":
    main()
