#!/usr/bin/env python
"""
Where do the errors come from?  (Prof. Wong, 11/17: "errors are heteroskedastic; freeze the
network, make the input a parameter, backprop the loss against ground truth, look at the gradient")

For held-out slices at the ES frame (where GT exists), for the source model and for an
adapted model (energy + pseudolabel, carried from ED along the sequence), this computes
  * the error map  (pred != GT)
  * the input-gradient saliency: d loss(pred, GT) / d image, with the network frozen.
    High |gradient| = pixels whose intensity the loss is most sensitive to, i.e. where the
    model "looks" when it gets things wrong.
and summarises, per slice, where the error mass sits: fraction on the myocardium boundary
band, inside the cavity, in the background; plus per-slice covariates (energy score, heart
size, displacement) so error can be attributed to image features (Week 6).

Writes ERRORS.md, errors.csv and overlay PNGs (image | error map | saliency | prediction vs GT).

Usage (Bouchet):
  python scripts/error_saliency.py --frames ~/Dataset/MnM_cine/frames/holdout --flow_ckpt ~/adacs_mnm/motion_0160.pt \
      --out results/error_saliency --device cuda --max_slices 60 --n_vis 12
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tta_wrapper as tw  # noqa: E402
from tta_wrapper import flow as F  # noqa: E402
from tta_wrapper.metrics import compute_dice  # noqa: E402


def frame_path(a, b, T, stride):
    out, t = [a], a
    while t != b:
        t = (t + 1) % T; out.append(t)
    sub = out[::stride]
    return sub + ([out[-1]] if out[-1] != sub[-1] else [])


def adapt_along(seg, en, frames, path, args):
    """energy+pseudolabel, carry_state, lambda_p 5: returns the adapted model at the end frame."""
    state, prev_img, prev_lbl = {}, None, None
    for t in path:
        x = torch.from_numpy(frames[t][None, None]).float()
        r = tw.adapt.adapt_slice(seg, en, x, strategy="hard" if prev_lbl is not None else "none", prev_img=prev_img,
                                 prev_label=prev_lbl, registration="vxm", num_iterations=args.num_iterations, lr=args.lr,
                                 variant="eval_tta", device=args.device, lambda_e=1.0, lambda_p=args.lambda_p, state=state)
        prev_img, prev_lbl = frames[t], r.pred.astype(np.uint8)
    return state["model"]


def saliency(model, image, gt, device):
    """|d CE(model(x), gt) / dx| with the network frozen and x a parameter. Returns (pred, grad (H,W), loss)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.from_numpy(image[None, None]).float().to(device).requires_grad_(True)
    y = torch.from_numpy(gt.astype(np.int64))[None].to(device)
    logits = model(x); loss = Fn.cross_entropy(logits, y)
    loss.backward()
    g = x.grad[0, 0].abs().cpu().numpy()
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8), g, float(loss)


def where_is_error(pred, gt):
    err = (pred != gt)
    myo = gt == 2; lv = gt == 1
    band = ndi.binary_dilation(myo ^ ndi.binary_erosion(myo), iterations=2) | ndi.binary_dilation(lv ^ ndi.binary_erosion(lv), iterations=2)
    inside = (gt > 0) & ~band
    tot = err.sum() + 1e-9
    return dict(err_px=int(err.sum()), err_on_boundary=float(err[band].sum() / tot), err_inside_heart=float(err[inside].sum() / tot),
                err_background=float(err[~(band | inside)].sum() / tot), myo_px=int(myo.sum()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True); a.add_argument("--flow_ckpt", required=True)
    a.add_argument("--out", default="results/error_saliency"); a.add_argument("--device", default="cpu")
    a.add_argument("--stride", type=int, default=5); a.add_argument("--lambda_p", type=float, default=5.0)
    a.add_argument("--num_iterations", type=int, default=10); a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--max_slices", type=int, default=60); a.add_argument("--n_vis", type=int, default=12)
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seg = tw.load_seg_model(device=args.device); en = tw.load_energy_model(device=args.device)
    tw.adapt.REGISTRATION["vxm"] = F.FlowRegistrar(os.path.expanduser(args.flow_ckpt), device=args.device)
    files = sorted(glob.glob(os.path.join(args.frames, "*.npz")))[:args.max_slices]

    rows = []
    for k, f in enumerate(files):
        z = np.load(f); frames = z["frames"].astype(np.float32); ed, es = int(z["ed"]), int(z["es"]); gt = z["lbl_ES"]
        path = frame_path(ed, es, frames.shape[0], args.stride)
        src = copy.deepcopy(seg).to(args.device)
        adapted = adapt_along(seg, en, frames, path, args)
        out = {}
        for name, m in (("source", src), ("adapted", adapted)):
            pred, g, loss = saliency(m, frames[es], gt, args.device)
            with torch.no_grad():
                e = float(torch.sigmoid(en(torch.softmax(m(torch.from_numpy(frames[es][None, None]).to(args.device)), 1))).mean())
            d = compute_dice(pred, gt)[1:]
            w = where_is_error(pred, gt)
            heart = ndi.binary_dilation(gt > 0, iterations=8)
            rows.append(dict(slice=os.path.basename(f)[:-4], patient=os.path.basename(f).split("_")[0], model=name, loss=loss,
                             dice_lv=d[0], dice_myo=d[1], energy=e, sal_in_heart=float(g[heart].sum() / (g.sum() + 1e-12)),
                             sal_heart_frac_area=float(heart.mean()), **w))
            out[name] = (pred, g)
        if k < args.n_vis:
            save_fig(args.out, os.path.basename(f)[:-4], frames[es], gt, out)
        if k % 10 == 0:
            r = rows[-1]; print(f"[{k+1}/{len(files)}] {r['slice']}  adapted Myo {100*r['dice_myo']:.1f}  err on boundary {100*r['err_on_boundary']:.0f}%  saliency in heart {100*r['sal_in_heart']:.0f}%", flush=True)

    import pandas as pd
    D = pd.DataFrame(rows); D.to_csv(os.path.join(args.out, "errors.csv"), index=False)
    md = [f"# Error anatomy at the ES frame  ({len(files)} slices, {args.frames})", "",
          "| model | Myo Dice | LV Dice | error px | % of error on boundaries | % inside heart | % background | saliency mass in heart (heart = % of image) | energy |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("source", "adapted"):
        d = D[D.model == name]
        md.append(f"| {name} | {100*d.dice_myo.mean():.2f} | {100*d.dice_lv.mean():.2f} | {d.err_px.mean():.0f} | {100*d.err_on_boundary.mean():.0f} | {100*d.err_inside_heart.mean():.0f} | {100*d.err_background.mean():.0f} | {100*d.sal_in_heart.mean():.0f}% ({100*d.sal_heart_frac_area.mean():.0f}%) | {d.energy.mean():.3f} |")
    # heteroskedasticity: error vs slice size / energy
    d = D[D.model == "adapted"]
    from scipy.stats import spearmanr
    md += ["", "Adapted model, per slice (Spearman):",
           f"- Myo Dice vs myocardium size (px): {spearmanr(d.dice_myo, d.myo_px).correlation:.3f}  (apex/base slices are small)",
           f"- Myo Dice vs energy of the prediction: {spearmanr(d.dice_myo, d.energy).correlation:.3f}",
           f"- error px vs myocardium size: {spearmanr(d.err_px, d.myo_px).correlation:.3f}",
           f"- Dice spread across slices: sd {100*d.dice_myo.std():.1f}, min {100*d.dice_myo.min():.1f}, max {100*d.dice_myo.max():.1f}", "",
           "Reading: if most error sits on boundaries and saliency is concentrated in the heart, the model is uncertain about *where the edge is*, not *whether there is a heart*;",
           "errors in the background or saliency spread over the image point to domain-shift confusion (intensity/texture) rather than geometry."]
    text = "\n".join(md); print("\n" + text); open(os.path.join(args.out, "ERRORS.md"), "w").write(text + "\n")
    print(f"\nwritten to {args.out}/")


def save_fig(out, name, img, gt, outs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    for i, (mname, (pred, g)) in enumerate(outs.items()):
        ax[i, 0].imshow(img, cmap="gray"); ax[i, 0].contour(gt == 2, [0.5], colors="cyan"); ax[i, 0].contour(gt == 1, [0.5], colors="lime")
        ax[i, 0].contour(pred == 2, [0.5], colors="magenta", linestyles="dashed"); ax[i, 0].contour(pred == 1, [0.5], colors="red", linestyles="dashed")
        ax[i, 0].set_title(f"{mname}: GT (solid) vs pred (dashed), Myo {100*compute_dice(pred, gt)[2]:.1f}")
        ax[i, 1].imshow(img, cmap="gray"); ax[i, 1].imshow(np.ma.masked_where(pred == gt, np.ones_like(img)), cmap="autumn", alpha=0.8); ax[i, 1].set_title("errors (orange)")
        ax[i, 2].imshow(img, cmap="gray"); ax[i, 2].imshow(g / (g.max() + 1e-12), cmap="inferno", alpha=0.7); ax[i, 2].set_title("|d loss / d input|")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out, f"{name}.png"), dpi=70); plt.close()


if __name__ == "__main__":
    main()
