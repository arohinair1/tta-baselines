#!/usr/bin/env python
"""
Diagnose the learned (VoxelMorph / AdaCS) flow used for pseudolabel warping.

Checks, in the order Prof. Wong suggested (each prints a table; figures go to --out):

  1. IMAGE WARP   warp(prev, flow) vs curr: reconstruction MSE vs identity, and where
                  the error sits (fraction within 3 px of anatomical edges). If the flow
                  does not even reconstruct the image, nothing downstream can work.
  2. DIRECTION    the same reconstruction with the flow applied the three "wrong" ways
                  (negated flow, flow from model(curr, prev), flow applied to curr).
                  If one of those beats the forward direction, the frame of reference
                  is flipped somewhere.
  3. LABEL WARP   Dice / IoU of warp(GT_prev) vs GT_curr for every direction above, plus
                  np.unique() of every warped label (must be a subset of {0,1,2}; anything
                  else means bilinear interpolation invented a class). Also compares
                  nearest-neighbour label warping with bilinear-on-one-hot + argmax.
  4. OVERFIT      fine-tune a copy of the flow model on THESE pairs only (image MSE +
                  smoothness, the unsupervised objective), then repeat 1-3. On the
                  training pairs themselves the warp should be near perfect; if it is
                  not, the model/loss cannot express the needed motion.
  5. DEMONS       the same numbers for Nicole's Demons registration, as a reference.

Figures per pair: prev | curr | warped prev | |warped-curr| | |prev-curr| | flow arrows,
and a label overlay (GT_curr vs warped GT_prev).

Usage (Bouchet):
  python scripts/diagnose_flow.py --pairs ~/Dataset/MyoPS_pairs/holdout \
      --flow_ckpt ~/adacs_myops/motion_0180.pt --out results/flow_diag --device cuda
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tta_wrapper.flow import vxm, load_flow_model, FlowRegistrar  # noqa: E402
from tta_wrapper._bootstrap import bootstrap  # noqa: E402

bootstrap()
import SimpleITK as sitk  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402


# ───────────────────────── helpers ─────────────────────────

def dice(a, b, c):
    a, b = a == c, b == c
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


def iou(a, b, c):
    a, b = a == c, b == c
    u = (a | b).sum()
    return (a & b).sum() / u if u else float("nan")


def label_scores(warped, gt):
    return {"LV": 100 * dice(warped, gt, 1), "Myo": 100 * dice(warped, gt, 2),
            "IoU_LV": 100 * iou(warped, gt, 1), "IoU_Myo": 100 * iou(warped, gt, 2)}


def edge_band(lbl, width=3):
    """pixels within `width` px of a label boundary (any class)."""
    edges = np.zeros_like(lbl, dtype=bool)
    for c in (1, 2):
        m = lbl == c
        if m.any():
            edges |= m ^ ndi.binary_erosion(m)
    return ndi.binary_dilation(edges, iterations=width)


def recon_stats(warped_img, target_img, target_lbl):
    err = (warped_img - target_img) ** 2
    band = edge_band(target_lbl)
    heart = ndi.binary_dilation(target_lbl > 0, iterations=10)
    return {"mse": float(err.mean()),
            "mse_heart": float(err[heart].mean()) if heart.any() else float("nan"),
            "err_frac_on_edges": float(err[band].sum() / (err.sum() + 1e-12)),
            "edge_area_frac": float(band.mean())}


class Vxm:
    """thin wrapper so the diagnostic can use any VxmDense (loaded or fine-tuned)."""

    def __init__(self, model, device):
        self.m, self.dev = model, device
        self.shape = tuple(model.config["inshape"])
        L = vxm().layers.SpatialTransformer
        self.st_nn = L(self.shape, mode="nearest").to(device)
        self.st_bl = L(self.shape, mode="bilinear").to(device)

    def t(self, a):
        return torch.from_numpy(np.asarray(a, np.float32))[None, None].to(self.dev)

    def flow(self, src, tgt):
        with torch.no_grad():
            _, f, _ = self.m(self.t(src), self.t(tgt), registration=True)
        return f[0].cpu().numpy()

    def warp(self, x, flow, nearest):
        st = self.st_nn if nearest else self.st_bl
        xt = torch.from_numpy(np.asarray(x, np.float32))
        xt = xt[None, None] if xt.ndim == 2 else xt[None]
        with torch.no_grad():
            out = st(xt.to(self.dev), torch.from_numpy(flow)[None].to(self.dev))[0].cpu().numpy()
        return out[0] if np.ndim(x) == 2 else out

    def warp_label_onehot(self, lbl, flow):
        oh = np.stack([(lbl == c) for c in (0, 1, 2)]).astype(np.float32)
        return self.warp(oh, flow, nearest=False).argmax(0).astype(np.uint8)


def demons(fixed, moving, moving_lbl):
    """Nicole's registration_utils.register_and_warp_2d, but also returns the warped IMAGE."""
    f = sitk.GetImageFromArray(fixed.astype(np.float32))
    m = sitk.GetImageFromArray(moving.astype(np.float32))
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024); matcher.SetNumberOfMatchPoints(7); matcher.ThresholdAtMeanIntensityOn()
    mm = matcher.Execute(m, f)
    d = sitk.FastSymmetricForcesDemonsRegistrationFilter(); d.SetNumberOfIterations(20); d.SetStandardDeviations(1.0)
    disp = d.Execute(f, mm)
    tf = sitk.DisplacementFieldTransform(disp)
    r = sitk.ResampleImageFilter(); r.SetReferenceImage(f); r.SetTransform(tf)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    wl = sitk.GetArrayFromImage(r.Execute(sitk.GetImageFromArray(moving_lbl.astype(np.float32)))).astype(np.uint8)
    r.SetInterpolator(sitk.sitkLinear)
    wi = sitk.GetArrayFromImage(r.Execute(m))
    return wi, wl


def load_pairs(d):
    from scipy.io import loadmat
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.mat"))):
        m = loadmat(f)
        if "lbl_ED" in m:
            la, lb = m["lbl_ED"].astype(np.uint8), m["lbl_ES"].astype(np.uint8)
        else:  # AdaCS-style file with myo only
            la, lb = (m["myo_ED"] > 0).astype(np.uint8) * 2, (m["myo_ES"] > 0).astype(np.uint8) * 2
        n = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        out.append({"name": os.path.basename(f), "prev": n(m["im_ED"].astype(np.float32)),
                    "curr": n(m["im_ES"].astype(np.float32)), "lprev": la, "lcurr": lb})
    assert out, f"no .mat pairs in {d}"
    return out


# ───────────────────────── the checks ─────────────────────────

def evaluate(reg: Vxm, pairs, tag, out, n_vis, with_demons=False):
    """Runs checks 1-3 (and 5) for one flow model. Returns dict of per-variant means."""
    rows = {}        # variant -> list of dicts
    uniques = {}     # variant -> set of label values seen
    figs = []

    def add(variant, lbl_scores, rec, warped_lbl):
        rows.setdefault(variant, []).append({**lbl_scores, **(rec or {})})
        uniques.setdefault(variant, set()).update(np.unique(warped_lbl).tolist())

    for k, p in enumerate(pairs):
        prev, curr, lp, lc = p["prev"], p["curr"], p["lprev"], p["lcurr"]
        f_fwd = reg.flow(prev, curr)        # what the wrapper does: model(prev, curr)
        f_rev = reg.flow(curr, prev)        # model(curr, prev)

        # identity
        add("identity", label_scores(lp, lc), recon_stats(prev, curr, lc), lp)
        # forward (wrapper): warp prev with f_fwd -> should match curr
        wi = reg.warp(prev, f_fwd, nearest=False); wl = reg.warp(lp, f_fwd, nearest=True).astype(np.uint8)
        add("fwd  model(prev,curr), warp prev          [wrapper]", label_scores(wl, lc), recon_stats(wi, curr, lc), wl)
        # forward, bilinear one-hot label warp instead of nearest
        wl_oh = reg.warp_label_onehot(lp, f_fwd)
        add("fwd  same, label via bilinear one-hot+argmax", label_scores(wl_oh, lc), None, wl_oh)
        # forward, bilinear directly on the integer label (the "invented class" failure mode)
        wl_bad = reg.warp(lp.astype(np.float32), f_fwd, nearest=False)
        add("fwd  same, label via bilinear on int label (BAD)", label_scores(np.rint(wl_bad).astype(np.uint8), lc), None, wl_bad)
        # wrong direction 1: negate the flow
        wi = reg.warp(prev, -f_fwd, nearest=False); wl = reg.warp(lp, -f_fwd, nearest=True).astype(np.uint8)
        add("neg  model(prev,curr), warp prev with -flow", label_scores(wl, lc), recon_stats(wi, curr, lc), wl)
        # wrong direction 2: flow from the swapped call
        wi = reg.warp(prev, f_rev, nearest=False); wl = reg.warp(lp, f_rev, nearest=True).astype(np.uint8)
        add("rev  model(curr,prev), warp prev", label_scores(wl, lc), recon_stats(wi, curr, lc), wl)
        # wrong direction 3: negated swapped flow
        wi = reg.warp(prev, -f_rev, nearest=False); wl = reg.warp(lp, -f_rev, nearest=True).astype(np.uint8)
        add("bwd  model(curr,prev), warp prev with -flow  [same direction: model's backward branch]", label_scores(wl, lc), recon_stats(wi, curr, lc), wl)
        # sanity: warping curr with f_rev should reproduce prev (symmetric use)
        wi = reg.warp(curr, f_rev, nearest=False); wl = reg.warp(lc, f_rev, nearest=True).astype(np.uint8)
        add("sym  model(curr,prev), warp curr -> prev", label_scores(wl, lp), recon_stats(wi, prev, lp), wl)
        if with_demons:
            wi, wl = demons(curr, prev, lp)
            add("demons (Nicole) warp prev -> curr", label_scores(wl, lc), recon_stats(wi, curr, lc), wl)

        # flow magnitude
        mag = np.sqrt((f_fwd ** 2).sum(0))
        heart = ndi.binary_dilation(lc > 0, iterations=10)
        rows.setdefault("_flow", []).append({"mean_px": float(mag.mean()), "mean_px_heart": float(mag[heart].mean()),
                                             "max_px": float(mag.max())})
        if k < n_vis:
            figs.append(save_fig(out, f"{tag}_{k:02d}_{p['name']}", prev, curr, reg.warp(prev, f_fwd, False),
                                 reg.warp(lp, f_fwd, True).astype(np.uint8), lc, f_fwd))

    summary = {v: {kk: float(np.nanmean([r[kk] for r in rs])) for kk in rs[0]} for v, rs in rows.items()}
    summary["_unique_label_values"] = {v: sorted(u) for v, u in uniques.items()}
    return summary, figs


def crop_and_background_check(reg: "Vxm", pairs):
    """Where is the flow? Per pair: GT myocardium centroid of prev and curr (per-slice centroid
    cropping puts both at ~(128,128)), mean |flow| in background vs heart, and the median
    background displacement. Also what the labels do if the flow is decomposed into its global
    background translation and the residual."""
    lines = ["### Crop / background check", "",
             "| pair | centroid prev | centroid curr | mean flow bg (px) | mean flow heart (px) | median bg flow (dy,dx) |", "|---|---|---|---|---|---|"]
    agg = {"identity": [], "full flow": [], "translation only": [], "flow minus translation": []}
    for p in pairs:
        prev, curr, lp, lc = p["prev"], p["curr"], p["lprev"], p["lcurr"]
        ca, cb = ndi.center_of_mass(lp == 2), ndi.center_of_mass(lc == 2)
        f = reg.flow(prev, curr); mag = np.hypot(f[0], f[1])
        heart = ndi.binary_dilation(lc > 0, iterations=10); bg = ~ndi.binary_dilation(lc > 0, iterations=40)
        tr = np.array([np.median(f[0][bg]), np.median(f[1][bg])], np.float32)
        lines.append(f"| {p['name']} | ({ca[0]:.1f},{ca[1]:.1f}) | ({cb[0]:.1f},{cb[1]:.1f}) | {mag[bg].mean():.1f} | {mag[heart].mean():.1f} | ({tr[0]:+.1f},{tr[1]:+.1f}) |")
        f_tr = np.zeros_like(f); f_tr[0] += tr[0]; f_tr[1] += tr[1]
        f_res = f - tr[:, None, None]
        for k, ff in (("full flow", f), ("translation only", f_tr), ("flow minus translation", f_res)):
            wi = reg.warp(prev, ff, nearest=False); wl = reg.warp(lp, ff, nearest=True).astype(np.uint8)
            agg[k].append((((wi - curr) ** 2).mean(), ((wi - curr) ** 2)[heart].mean(), 100 * dice(wl, lc, 2)))
        agg["identity"].append((((prev - curr) ** 2).mean(), ((prev - curr) ** 2)[heart].mean(), 100 * dice(lp, lc, 2)))
    lines += ["", "| variant | img MSE | MSE in heart | Myo Dice |", "|---|---|---|---|"]
    for k, v in agg.items():
        v = np.mean(v, 0); lines.append(f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.2f} |")
    lines += ["", "If both centroids are ~(128,128) in every pair, each slice was cropped around its own GT centroid: the hearts are",
              "pre-aligned and the inter-slice shift lives in the background, which is then what the flow model learns to move."]
    return "\n".join(lines)


def save_fig(out, name, prev, curr, warped, wl, lc, flow):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    e_id, e_w = np.abs(prev - curr), np.abs(warped - curr)
    vmax = max(e_id.max(), e_w.max(), 1e-6)
    for a, im, t in zip(ax[0], [prev, curr, warped, None],
                        ["prev (source)", "curr (target)", "warp(prev) [should ≈ curr]", "flow arrows (backward: where curr samples prev)"]):
        if im is not None:
            a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t); a.axis("off")
    ax[0, 3].imshow(curr, cmap="gray", vmin=0, vmax=1)
    s = 8; H, W = curr.shape
    yy, xx = np.mgrid[0:H:s, 0:W:s]
    # flow[0] is the displacement along axis 0 (rows/y), flow[1] along axis 1 (cols/x)
    ax[0, 3].quiver(xx, yy, flow[1, ::s, ::s], flow[0, ::s, ::s], color="yellow", angles="xy",
                    scale_units="xy", scale=1, width=0.003)
    ax[1, 0].imshow(e_id, cmap="hot", vmin=0, vmax=vmax); ax[1, 0].set_title(f"|prev-curr|  mse={float((e_id**2).mean()):.4f}")
    ax[1, 1].imshow(e_w, cmap="hot", vmin=0, vmax=vmax); ax[1, 1].set_title(f"|warp(prev)-curr|  mse={float((e_w**2).mean()):.4f}")
    ax[1, 2].imshow(curr, cmap="gray", vmin=0, vmax=1)
    ax[1, 2].contour(lc == 1, [0.5], colors="lime"); ax[1, 2].contour(lc == 2, [0.5], colors="cyan")
    ax[1, 2].contour(wl == 1, [0.5], colors="red", linestyles="dashed"); ax[1, 2].contour(wl == 2, [0.5], colors="magenta", linestyles="dashed")
    ax[1, 2].set_title("GT_curr (solid) vs warp(GT_prev) (dashed)")
    mag = np.sqrt((flow ** 2).sum(0))
    im = ax[1, 3].imshow(mag, cmap="viridis"); ax[1, 3].set_title(f"|flow| px  mean={mag.mean():.2f}")
    plt.colorbar(im, ax=ax[1, 3], fraction=0.046)
    for a in ax[1]:
        a.axis("off")
    plt.tight_layout()
    path = os.path.join(out, f"{name}.png"); plt.savefig(path, dpi=80); plt.close()
    return path


def overfit(model, pairs, device, steps, lr, lam, bidir=True, log_every=50):
    """Fine-tune a COPY of the flow model on these pairs with the unsupervised objective
    (image MSE + lambda * flow-gradient L2), exactly the AdaCS loss family without the
    correspondence-scoring part. Returns the fine-tuned model."""
    m = copy.deepcopy(model).to(device).train()
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    grad_loss = vxm().losses.Grad_2d("l2", loss_mult=m.config.get("int_downsize", 2) if isinstance(m.config, dict) else 2).loss
    X = torch.from_numpy(np.stack([p["prev"] for p in pairs]))[:, None].to(device)
    Y = torch.from_numpy(np.stack([p["curr"] for p in pairs]))[:, None].to(device)
    bs = min(4, len(pairs))
    t0 = time.time()
    for it in range(steps):
        idx = torch.randperm(len(pairs))[:bs]
        src, tgt = X[idx], Y[idx]
        out = m(src, tgt, registration=False)
        if m.bidir:
            y_src, y_tgt, flow, _ = out
            l_img = 0.5 * (((y_src - tgt) ** 2).mean() + ((y_tgt - src) ** 2).mean())
        else:
            y_src, flow, _ = out
            l_img = ((y_src - tgt) ** 2).mean()
        l_reg = grad_loss(None, flow, None, None)
        loss = l_img + lam * l_reg
        opt.zero_grad(); loss.backward(); opt.step()
        if it % log_every == 0 or it == steps - 1:
            print(f"  overfit step {it:4d}  img_mse={l_img.item():.5f}  smooth={l_reg.item():.5f}  ({time.time()-t0:.0f}s)")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def fmt_table(summary, title):
    lines = [f"### {title}", "", "| variant | LV Dice | Myo Dice | IoU LV | IoU Myo | img MSE | MSE in heart | err on edges | unique labels |",
             "|---|---|---|---|---|---|---|---|---|"]
    uniq = summary.get("_unique_label_values", {})
    for v, s in summary.items():
        if v.startswith("_"):
            continue
        g = lambda k: f"{s[k]:.2f}" if k in s and not np.isnan(s[k]) else "-"
        g4 = lambda k: f"{s[k]:.4f}" if k in s and not np.isnan(s[k]) else "-"
        lines.append(f"| {v} | {g('LV')} | {g('Myo')} | {g('IoU_LV')} | {g('IoU_Myo')} | {g4('mse')} | {g4('mse_heart')} | {g('err_frac_on_edges')} | {uniq.get(v, [])} |")
    f = summary.get("_flow", {})
    if f:
        lines.append(f"\nflow magnitude: mean {f['mean_px']:.2f} px, mean inside heart {f['mean_px_heart']:.2f} px, max {f['max_px']:.2f} px")
    return "\n".join(lines)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pairs", required=True, help="dir of AdaCS-format .mat pairs (im_ED=prev, im_ES=curr, lbl_*)")
    a.add_argument("--flow_ckpt", default=None, help="motion_XXXX.pt ; omit for a fresh (untrained) VxmDense")
    a.add_argument("--out", default="results/flow_diag")
    a.add_argument("--device", default="cpu")
    a.add_argument("--n_vis", type=int, default=6)
    a.add_argument("--overfit_steps", type=int, default=500, help="0 to skip check 4")
    a.add_argument("--overfit_lr", type=float, default=1e-3)
    a.add_argument("--overfit_lambda", type=float, default=0.01, help="smoothness weight (train_vxm default 0.01)")
    a.add_argument("--max_pairs", type=int, default=0)
    a.add_argument("--no_demons", action="store_true")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    pairs = load_pairs(args.pairs)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"{len(pairs)} pairs from {args.pairs}")

    if args.flow_ckpt:
        model = load_flow_model(args.flow_ckpt, dev)
        print(f"loaded {args.flow_ckpt}  config={model.config}")
    else:
        inshape = pairs[0]["prev"].shape
        model = vxm().networks.VxmDense(inshape=inshape, nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                                         bidir=True, int_steps=7, int_downsize=2).to(dev).eval()
        print("no --flow_ckpt: using a fresh untrained VxmDense (flow ~ 0; only the overfit check is meaningful)")

    md = [f"# Flow diagnostics  ({len(pairs)} pairs, {args.pairs}, ckpt={args.flow_ckpt})", "",
          "Dice/IoU: warp(GT_prev) vs GT_curr. img MSE: warp(prev) vs curr (identity = prev vs curr).",
          "'err on edges' = fraction of squared image error within 3 px of a GT_curr boundary (edge band covers 'edge_area_frac' of the image).",
          "The 'fwd' row is exactly what tta_wrapper/flow.py does. If 'neg' or 'rev' beats it, the direction is wrong.",
          "'bwd' is -flow from model(curr, prev): for a bidirectional model this is a second estimate of the SAME direction as 'fwd', not a flip.", ""]
    results = {}

    print("\n== checks 1-3 (+5): trained model ==")
    s, figs = evaluate(Vxm(model, dev), pairs, "trained", args.out, args.n_vis, with_demons=not args.no_demons)
    results["trained"] = s
    t = fmt_table(s, "Trained flow model (as used in the TTA loop)"); print(t); md += [t, ""]
    t = crop_and_background_check(Vxm(model, dev), pairs); print("\n" + t); md += [t, ""]

    if args.overfit_steps > 0:
        print(f"\n== check 4: overfit on these {len(pairs)} pairs ({args.overfit_steps} steps, lr {args.overfit_lr}, lambda {args.overfit_lambda}) ==")
        m2 = overfit(model, pairs, dev, args.overfit_steps, args.overfit_lr, args.overfit_lambda)
        s2, figs2 = evaluate(Vxm(m2, dev), pairs, "overfit", args.out, args.n_vis, with_demons=False)
        results["overfit"] = s2
        t = fmt_table(s2, f"After overfitting on the same pairs ({args.overfit_steps} steps)"); print(t); md += [t, ""]
        torch.save(m2.state_dict(), os.path.join(args.out, "overfit_state_dict.pt"))

    # verdicts
    v = []
    tr = results["trained"]
    fwd = next(k for k in tr if k.startswith("fwd  model"))
    best_img = min((k for k in tr if not k.startswith("_") and "mse" in tr[k]), key=lambda k: tr[k]["mse"])
    best_lbl = max((k for k in tr if not k.startswith("_")), key=lambda k: np.nan_to_num(tr[k]["Myo"], nan=-1))
    v.append(f"- image reconstruction: identity MSE {tr['identity']['mse']:.4f} -> fwd {tr[fwd]['mse']:.4f}; best variant by MSE: '{best_img}'")
    v.append(f"- labels: identity Myo {tr['identity']['Myo']:.2f} -> fwd {tr[fwd]['Myo']:.2f}; best variant by Myo Dice: '{best_lbl}'")
    # only the genuinely flipped variants count; 'bwd' is the model's own estimate of the SAME direction
    learned = [k for k in tr if k.startswith("neg ") or k.startswith("rev ")]
    flipped_better = [k for k in learned if tr[k]["mse"] < tr[fwd]["mse"] - 1e-5 or np.nan_to_num(tr[k]["Myo"], nan=-1) > tr[fwd]["Myo"] + 0.5]
    v.append("- DIRECTION " + (f"LOOKS WRONG: {flipped_better} beat the wrapper's direction." if flipped_better
                               else "OK: no flipped variant beats the wrapper's direction (fwd)."))
    u = tr["_unique_label_values"][fwd]
    v.append(f"- unique values of the wrapper's warped label: {u} -> " + ("OK (nearest-neighbour)" if set(u) <= {0, 1, 2} else "NOT integer labels: interpolation problem"))
    v.append(f"- flow magnitude inside the heart: {tr['_flow']['mean_px_heart']:.2f} px (if ~0 the model learned (almost) the identity)")
    if "overfit" in results:
        o = results["overfit"]; ofwd = next(k for k in o if k.startswith("fwd  model"))
        v.append(f"- overfit on the same pairs: img MSE {o[ofwd]['mse']:.4f}, Myo Dice {o[ofwd]['Myo']:.2f}, LV Dice {o[ofwd]['LV']:.2f} "
                 + f"(before: MSE {tr[fwd]['mse']:.4f}, Myo {tr[fwd]['Myo']:.2f}, LV {tr[fwd]['LV']:.2f}). "
                 + ("Labels DO move toward GT when the model is fit on these pairs -> warp/direction are fine; the gap is training/generalisation." if o[ofwd]["Myo"] > tr[fwd]["Myo"] + 2
                    else "Labels barely move even when fit on these pairs -> image MSE between adjacent slices is not a good proxy for label alignment."))
    print("\n== verdict =="); print("\n".join(v)); md += ["## Verdict", *v, ""]

    open(os.path.join(args.out, "DIAGNOSIS.md"), "w").write("\n".join(md) + "\n")
    json.dump({"args": vars(args), "results": results}, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
    print(f"\nwritten to {args.out}/ (DIAGNOSIS.md, summary.json, overfit_state_dict.pt, PNGs)")


if __name__ == "__main__":
    main()
