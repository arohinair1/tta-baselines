#!/usr/bin/env python
"""
Flow across ALL frames of a cine slice, with consistency checks (Week 5 prep).

For each slice sequence (frames/<split>/<patient>_s<zz>.npz from prep_mnm_cine.py)
and each direction (ED -> ES forward in time, ES -> ED backward), this

  1. estimates the flow between every consecutive pair of frames with the
     trained VoxelMorph model (both ways: t->t+1 and t+1->t),
  2. composes the per-step flows into one chained ED->ES displacement field
         Phi(x) = f_last(x) + f_prev(x + f_last(x)) + ...
     so the start label is warped ONCE (nearest neighbour) with the composed field,
  3. builds a per-pixel TRUST mask from two agreement tests:
       (a) forward-backward: |f_fwd(x) + f_bwd(x + f_fwd(x))| < tau_fb at every step
           (a pixel that does not come back to itself is unreliable); the mask is
           carried along the chain,
       (b) chain vs direct: |Phi_chain(x) - Phi_direct(x)| < tau_cd, where Phi_direct
           is the model's one-shot ED->ES flow,
  4. evaluates against the ground truth at the end frame:
       Dice of   identity | direct one-shot warp | chained warp | chained warp on trusted pixels only
       and the per-pixel label accuracy of the chained pseudolabel on trusted vs untrusted pixels.

"The only thing that is persistent is the pixels that are correct": if the chained
pseudolabel is much more accurate on trusted pixels than on untrusted ones, the
mask turns the unsupervised warp into a (partial) supervised target for TTA.

Usage (Bouchet):
  python scripts/chain_flow.py --frames ~/Dataset/MnM_cine/frames/holdout --flow_ckpt ~/adacs_mnm/motion_0160.pt \
      --out results/chain_mnm --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tta_wrapper.flow import vxm, load_flow_model  # noqa: E402


def dice(a, b, c):
    a, b = a == c, b == c
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


class Flow:
    def __init__(self, model, device):
        self.m, self.dev = model, device
        self.shape = tuple(model.config["inshape"])
        L = vxm().layers.SpatialTransformer
        self.st_nn = L(self.shape, mode="nearest").to(device)
        self.st_bl = L(self.shape, mode="bilinear").to(device)

    def t(self, a):
        return torch.from_numpy(np.asarray(a, np.float32))[None, None].to(self.dev)

    def flow(self, src, tgt):
        """flow such that warp(src, flow) ~= tgt  (backward field: tgt pixel x samples src at x + flow(x))."""
        with torch.no_grad():
            _, f, _ = self.m(self.t(src), self.t(tgt), registration=True)
        return f[0]                                                        # (2,H,W) tensor on device

    def warp(self, x, flow, nearest):
        st = self.st_nn if nearest else self.st_bl
        xt = x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x, np.float32)).to(self.dev)
        xt = xt[None, None] if xt.ndim == 2 else xt[None]
        with torch.no_grad():
            out = st(xt, flow[None])[0]
        return out[0] if x.ndim == 2 else out

    def compose(self, first, second):
        """flow of warp(warp(., first), second):  Phi(x) = second(x) + first(x + second(x))."""
        return second + self.warp(first, second, nearest=False)

    def fb_error(self, f_fwd, f_bwd):
        """forward-backward inconsistency per pixel: |f_fwd(x) + f_bwd(x + f_fwd(x))|  (H,W)."""
        return (f_fwd + self.warp(f_bwd, f_fwd, nearest=False)).norm(dim=0)


def frame_path(ed, es, T, stride=1):
    """indices from ed to es going forward in time (wrapping), inclusive; every `stride`-th
    frame (the end frame is always kept, so the last step may be shorter)."""
    out, t = [ed], ed
    while t != es:
        t = (t + 1) % T
        out.append(t)
    if stride > 1:
        out = out[::stride] + ([out[-1]] if out[-1] not in out[::stride] else [])
    return out


def run_sequence(F: Flow, frames, path, lbl_start, lbl_end, tau_fb, tau_cd):
    """Chain along `path` (list of frame indices). Returns dict of metrics + arrays."""
    dev = F.dev
    steps = list(zip(path[:-1], path[1:]))
    fwd = [F.flow(frames[a], frames[b]) for a, b in steps]          # a -> b
    bwd = [F.flow(frames[b], frames[a]) for a, b in steps]          # b -> a
    # composed field (warp start label once) and the carried forward-backward trust mask
    Phi = fwd[-1]
    for f in reversed(fwd[:-1]):
        Phi = F.compose(f, Phi)
    trust = torch.ones(F.shape, device=dev, dtype=torch.bool)
    # the mask at the END frame: pixel x of the end frame is trusted if, at every step along its
    # back-trajectory, the forward-backward error there is small. Walk from the end backwards.
    tail = None   # composed flow from the current step's target frame to the end frame
    fb_steps = []
    for i in range(len(steps) - 1, -1, -1):
        e = F.fb_error(fwd[i], bwd[i])                                # defined on frame b of step i
        e_end = e if tail is None else F.warp(e, tail, nearest=False) # bring it to the end frame grid
        trust &= e_end < tau_fb
        fb_steps.append(float((e < tau_fb).float().mean()))
        tail = fwd[i] if tail is None else F.compose(fwd[i], tail)
    direct = F.flow(frames[path[0]], frames[path[-1]])
    cd = (Phi - direct).norm(dim=0)
    trust_cd = cd < tau_cd
    trust_both = trust & trust_cd

    L0 = torch.from_numpy(lbl_start.astype(np.float32)).to(dev)
    pl_chain = F.warp(L0, Phi, nearest=True).cpu().numpy().astype(np.uint8)
    pl_direct = F.warp(L0, direct, nearest=True).cpu().numpy().astype(np.uint8)
    tb, tf, tc = trust_both.cpu().numpy(), trust.cpu().numpy(), trust_cd.cpu().numpy()
    gt = lbl_end
    correct = pl_chain == gt
    heart = (gt > 0) | (pl_chain > 0)
    m = {
        "n_steps": len(steps),
        "dice_identity": [100 * dice(lbl_start, gt, c) for c in (1, 2)],
        "dice_direct": [100 * dice(pl_direct, gt, c) for c in (1, 2)],
        "dice_chain": [100 * dice(pl_chain, gt, c) for c in (1, 2)],
        "dice_chain_trusted": [100 * dice(np.where(tb, pl_chain, 0), np.where(tb, gt, 0), c) for c in (1, 2)],
        "trusted_frac_heart": float(tb[heart].mean()) if heart.any() else float("nan"),
        "trusted_frac_fb_heart": float(tf[heart].mean()) if heart.any() else float("nan"),
        "trusted_frac_cd_heart": float(tc[heart].mean()) if heart.any() else float("nan"),
        "acc_trusted_heart": float(correct[heart & tb].mean()) if (heart & tb).any() else float("nan"),
        "acc_untrusted_heart": float(correct[heart & ~tb].mean()) if (heart & ~tb).any() else float("nan"),
        "acc_all_heart": float(correct[heart].mean()) if heart.any() else float("nan"),
        "mean_disp_heart_px": float(Phi.norm(dim=0).cpu().numpy()[gt > 0].mean()) if (gt > 0).any() else float("nan"),
        "mean_disp_direct_heart_px": float(direct.norm(dim=0).cpu().numpy()[gt > 0].mean()) if (gt > 0).any() else float("nan"),
        "fb_ok_frac_per_step": fb_steps[::-1],
    }
    arrays = dict(pl_chain=pl_chain, pl_direct=pl_direct, trust=tb, fb_err_end=None, cd=cd.cpu().numpy(),
                  Phi=Phi.cpu().numpy())
    return m, arrays


def save_fig(out, name, frames, path, lbl_start, lbl_end, arr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f0, f1 = frames[path[0]], frames[path[-1]]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
    ax[0].imshow(f0, cmap="gray", vmin=0, vmax=1); ax[0].contour(lbl_start == 2, [0.5], colors="cyan"); ax[0].contour(lbl_start == 1, [0.5], colors="lime")
    ax[0].set_title(f"start frame t={path[0]} + its GT")
    ax[1].imshow(f1, cmap="gray", vmin=0, vmax=1)
    ax[1].contour(lbl_end == 2, [0.5], colors="cyan"); ax[1].contour(lbl_end == 1, [0.5], colors="lime")
    ax[1].contour(arr["pl_chain"] == 2, [0.5], colors="magenta", linestyles="dashed"); ax[1].contour(arr["pl_chain"] == 1, [0.5], colors="red", linestyles="dashed")
    ax[1].set_title(f"end frame t={path[-1]}: GT (solid) vs chained pseudolabel (dashed), {len(path)-1} steps")
    ax[2].imshow(f1, cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(np.ma.masked_where(arr["trust"], np.ones_like(f1)), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
    ax[2].set_title("UNtrusted pixels (orange)")
    im = ax[3].imshow(arr["cd"], cmap="viridis", vmax=5); ax[3].set_title("|chain - direct| flow (px)")
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); p = os.path.join(out, f"{name}.png"); plt.savefig(p, dpi=80); plt.close()
    return p


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", required=True, help="dir of <patient>_s<zz>.npz from prep_mnm_cine.py")
    a.add_argument("--flow_ckpt", default=None)
    a.add_argument("--out", default="results/chain")
    a.add_argument("--device", default="cpu")
    a.add_argument("--tau_fb", type=float, default=1.5, help="forward-backward tolerance (px) per step")
    a.add_argument("--tau_cd", type=float, default=3.0, help="chained-vs-direct tolerance (px) at the end frame")
    a.add_argument("--max_seqs", type=int, default=0)
    a.add_argument("--n_vis", type=int, default=6)
    a.add_argument("--directions", default="both", choices=["both", "ed2es", "es2ed"])
    a.add_argument("--stride", type=int, default=1, help="frames per chain step (1 = every frame)")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    files = sorted(glob.glob(os.path.join(args.frames, "*.npz")))
    if args.max_seqs:
        files = files[:args.max_seqs]
    assert files, f"no .npz in {args.frames}"
    if args.flow_ckpt:
        model = load_flow_model(args.flow_ckpt, dev)
    else:
        model = vxm().networks.VxmDense(inshape=(256, 256), nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
                                         bidir=True, int_steps=7, int_downsize=2).to(dev).eval()
        print("no --flow_ckpt: untrained model (flow ~ 0), mechanics test only")
    F = Flow(model, dev)

    rows, figs = [], []
    for k, fpath in enumerate(files):
        z = np.load(fpath)
        frames = z["frames"].astype(np.float32); ed, es = int(z["ed"]), int(z["es"])
        l_ed, l_es = z["lbl_ED"], z["lbl_ES"]
        T = frames.shape[0]
        runs = []
        if args.directions in ("both", "ed2es"):
            runs.append(("ed2es", frame_path(ed, es, T, args.stride), l_ed, l_es))
        if args.directions in ("both", "es2ed"):
            runs.append(("es2ed", frame_path(es, ed, T, args.stride), l_es, l_ed))
        for dname, path, l0, l1 in runs:
            m, arr = run_sequence(F, frames, path, l0, l1, args.tau_fb, args.tau_cd)
            m.update(seq=os.path.basename(fpath)[:-4], direction=dname)
            rows.append(m)
            if len(figs) < args.n_vis:
                figs.append(save_fig(args.out, f"{dname}_{os.path.basename(fpath)[:-4]}", frames, path, l0, l1, arr))
        if k % 10 == 0:
            print(f"[{k+1}/{len(files)}] {os.path.basename(fpath)}  steps={rows[-1]['n_steps']}  "
                  f"chain Myo {rows[-1]['dice_chain'][1]:.1f}  direct Myo {rows[-1]['dice_direct'][1]:.1f}  "
                  f"trusted {100*rows[-1]['trusted_frac_heart']:.0f}%  acc trusted/untrusted "
                  f"{100*rows[-1]['acc_trusted_heart']:.0f}/{100*rows[-1]['acc_untrusted_heart']:.0f}", flush=True)

    def mean(key, idx=None):
        v = [r[key][idx] if idx is not None else r[key] for r in rows]
        return float(np.nanmean(v))
    md = [f"# Chained flow across frames  ({len(rows)} sequences from {len(files)} slices, {args.frames}, ckpt={args.flow_ckpt})",
          f"stride={args.stride} frames per step, tau_fb={args.tau_fb} px per step, tau_cd={args.tau_cd} px at the end frame. Mean steps per sequence: {mean('n_steps'):.1f}", "",
          "| pseudolabel at the END frame | LV Dice | Myo Dice |", "|---|---|---|",
          f"| identity (start label, no warp) | {mean('dice_identity',0):.2f} | {mean('dice_identity',1):.2f} |",
          f"| direct one-shot warp start->end | {mean('dice_direct',0):.2f} | {mean('dice_direct',1):.2f} |",
          f"| chained frame-by-frame warp | {mean('dice_chain',0):.2f} | {mean('dice_chain',1):.2f} |",
          f"| chained warp, trusted pixels only | {mean('dice_chain_trusted',0):.2f} | {mean('dice_chain_trusted',1):.2f} |", "",
          "| trust mask (inside heart region) | value |", "|---|---|",
          f"| trusted fraction (both tests) | {100*mean('trusted_frac_heart'):.1f}% |",
          f"| trusted fraction (forward-backward only) | {100*mean('trusted_frac_fb_heart'):.1f}% |",
          f"| trusted fraction (chain-vs-direct only) | {100*mean('trusted_frac_cd_heart'):.1f}% |",
          f"| pixel accuracy of chained pseudolabel, ALL heart pixels | {100*mean('acc_all_heart'):.1f}% |",
          f"| pixel accuracy on TRUSTED pixels | {100*mean('acc_trusted_heart'):.1f}% |",
          f"| pixel accuracy on UNTRUSTED pixels | {100*mean('acc_untrusted_heart'):.1f}% |",
          f"| mean displacement inside heart (px): chained / direct | {mean('mean_disp_heart_px'):.2f} / {mean('mean_disp_direct_heart_px'):.2f} |", "",
          "Per direction:", "", "| direction | n | identity Myo | direct Myo | chain Myo | chain-trusted Myo | trusted % | acc trusted | acc untrusted |", "|---|---|---|---|---|---|---|---|---|"]
    for d in ("ed2es", "es2ed"):
        rs = [r for r in rows if r["direction"] == d]
        if rs:
            g = lambda k, i=None: np.nanmean([r[k][i] if i is not None else r[k] for r in rs])
            md.append(f"| {d} | {len(rs)} | {g('dice_identity',1):.2f} | {g('dice_direct',1):.2f} | {g('dice_chain',1):.2f} | {g('dice_chain_trusted',1):.2f} | {100*g('trusted_frac_heart'):.1f} | {100*g('acc_trusted_heart'):.1f} | {100*g('acc_untrusted_heart'):.1f} |")
    md += ["", "Reading: if 'acc trusted' >> 'acc untrusted', the consistency mask separates correct from wrong pseudolabel pixels,",
           "and the trusted pixels can be used as supervision. 'chain' vs 'direct' says whether stepping frame by frame beats one big jump."]
    text = "\n".join(md); print("\n" + text)
    open(os.path.join(args.out, "CHAIN.md"), "w").write(text + "\n")
    json.dump({"args": vars(args), "rows": rows}, open(os.path.join(args.out, "per_sequence.json"), "w"), indent=1, default=float)
    print(f"\nwritten to {args.out}/ (CHAIN.md, per_sequence.json, {len(figs)} PNGs)")


if __name__ == "__main__":
    main()
