#!/usr/bin/env python3
"""
Reproduce every baseline in the repo on the bundled ACDC example slices
(external/pttea_seg/example_data: 12 patients x {ED, ES} = 24 slices, 256x256)
with the bundled upstream checkpoints.

Baselines run (all through tta_wrapper):
  no_adapt              eval.py            source model, no adaptation
  pttea                 run_pttea.py       energy-only TTA (variant='pttea')
  pttea_evaltta         eval_tta.py --pseudolabel_strategy hard on the FIRST
                        slice of a case == energy-only with early stopping
                        (variant='eval_tta', strategy='none')
  pl_hard[_affine]      eval_tta.py --pseudolabel_strategy hard [--registration affine]
  pl_confidence         eval_tta.py --pseudolabel_strategy confidence
  pl_entropy            eval_tta.py --pseudolabel_strategy entropy

Usage:
  python scripts/reproduce.py                     # everything, CPU
  python scripts/reproduce.py --only no_adapt pttea pl_hard
  python scripts/reproduce.py --data /path/to/ACDC/test --device cuda
Outputs results/<timestamp>/{metrics.json, per_slice.csv, losses.json, RESULTS.md}
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tta_wrapper as tw  # noqa: E402

BASELINES = {
    #  name            strategy      variant     registration
    "no_adapt":        None,
    "pttea":           ("none",       "pttea",    "demons"),
    "pttea_evaltta":   ("none",       "eval_tta", "demons"),
    "pl_hard":         ("hard",       "eval_tta", "demons"),
    "pl_hard_affine":  ("hard",       "eval_tta", "affine"),
    "pl_confidence":   ("confidence", "eval_tta", "demons"),
    "pl_entropy":      ("entropy",    "eval_tta", "demons"),
}


def run_baseline(name, seg, en, cases, args):
    cfg = BASELINES[name]
    dice, iou, asd, rows, losses = [], [], [], [], []
    t0 = time.time()
    if name == "pttea" and args.batch_size > 1:
        # run_pttea.py style: batches of consecutive slices adapted jointly
        flat = [s for v in cases.values() for s in v]
        for b in range(0, len(flat), args.batch_size):
            chunk = flat[b:b + args.batch_size]
            torch.manual_seed(args.seed)
            res = tw.adapt.adapt_batch(seg, en, torch.cat([s.tensor() for s in chunk], 0),
                                       num_iterations=args.num_iterations, lr=args.lr, device=args.device)
            for s, r in zip(chunk, res):
                d, i, a = (tw.compute_dice(r.pred, s.label), tw.compute_iou(r.pred, s.label),
                           tw.compute_asd(r.pred, s.label))
                dice.append(d); iou.append(i); asd.append(a)
                rows.append({"baseline": name, "case": s.case, "slice": s.name,
                             "dice_endo": d[1], "dice_myo": d[2], "dice_fg": d[1:].mean(), "iters": r.iters_run})
        summary = tw.summarize(dice, iou, asd); summary["seconds"] = round(time.time() - t0, 1)
        return summary, rows, []
    for case, slices in cases.items():
        prev_img = prev_label = prev_conf = None
        for s in slices:
            x = s.tensor()
            torch.manual_seed(args.seed)
            if cfg is None:
                r = tw.predict_no_adapt(seg, x, device=args.device)
            else:
                strategy, variant, reg = cfg
                r = tw.adapt_slice(seg, en, x, strategy=strategy, variant=variant,
                                   registration=reg, prev_img=prev_img, prev_label=prev_label,
                                   prev_conf=prev_conf, num_iterations=args.num_iterations,
                                   lr=args.lr, device=args.device)
                prev_img, prev_label, prev_conf = s.image, r.pred.astype(np.uint8), r.confidence
            d, i, a = (tw.compute_dice(r.pred, s.label), tw.compute_iou(r.pred, s.label),
                       tw.compute_asd(r.pred, s.label))
            dice.append(d); iou.append(i); asd.append(a)
            rows.append({"baseline": name, "case": case, "slice": s.name,
                         "dice_endo": d[1], "dice_myo": d[2], "dice_fg": d[1:].mean(),
                         "iters": r.iters_run})
            losses.append({"slice": s.name, "losses": r.losses})
    summary = tw.summarize(dice, iou, asd)
    summary["seconds"] = round(time.time() - t0, 1)
    return summary, rows, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=tw.EXAMPLE_DATA)
    p.add_argument("--dataset", default="acdc_mat", choices=list(tw.data.LOADERS),
                   help="acdc_mat (example slices / ACDC .mat), mnm (raw M&Ms), lvquant (train .mat), myops (setup_myops.py output)")
    p.add_argument("--seg_ckpt", default=tw.SEG_CKPT)
    p.add_argument("--energy_ckpt", default=tw.ENERGY_CKPT)
    p.add_argument("--only", nargs="*", default=list(BASELINES))
    p.add_argument("--num_iterations", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="pttea only: adapt this many consecutive slices jointly (run_pttea.py style). 1 = per slice.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    seg = tw.load_seg_model(args.seg_ckpt, args.device)
    en = tw.load_energy_model(args.energy_ckpt, args.device)
    cases = tw.data.LOADERS[args.dataset](args.data)
    print(f"{len(cases)} cases, {sum(len(v) for v in cases.values())} slices from {args.data}")

    metrics, all_rows, all_losses = {}, [], {}
    for name in args.only:
        print(f"[{name}] ...", flush=True)
        m, rows, losses = run_baseline(name, seg, en, cases, args)
        metrics[name] = m; all_rows += rows; all_losses[name] = losses
        pc = m["per_class"]
        print(f"[{name}] Dice fg={m['overall_dice_fg']:.4f}  endo={pc['Endo/LV']['dice']:.4f} "
              f"myo={pc['Myocardium']['dice']:.4f}  ASD myo={pc['Myocardium']['asd']:.3f}  ({m['seconds']}s)")

    json.dump({"args": vars(args), "metrics": metrics}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    json.dump(all_losses, open(os.path.join(out, "losses.json"), "w"))
    with open(os.path.join(out, "per_slice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)

    lines = ["| baseline | Dice fg | Dice endo | Dice myo | IoU myo | ASD endo | ASD myo | sec |",
             "|---|---|---|---|---|---|---|---|"]
    for n, m in metrics.items():
        e, my = m["per_class"]["Endo/LV"], m["per_class"]["Myocardium"]
        lines.append(f"| {n} | {m['overall_dice_fg']:.4f} | {e['dice']:.4f} | {my['dice']:.4f} | "
                     f"{my['iou']:.4f} | {e['asd']:.3f} | {my['asd']:.3f} | {m['seconds']} |")
    md = "\n".join(lines)
    open(os.path.join(out, "RESULTS.md"), "w").write(
        f"# Results\n\ndata: `{args.data}`  seg: `{args.seg_ckpt}`  energy: `{args.energy_ckpt}`  "
        f"iters={args.num_iterations} lr={args.lr} batch_size={args.batch_size} device={args.device}\n\n{md}\n")
    print("\n" + md)
    if args.dataset in ("mnm", "lvquant", "myops"):
        ref = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "reference", "pttea_table1_unet_acdc.json")))[args.dataset]
        pairs = [("no_adapt", "Pretrained"), ("pttea", "Ours")]
        lines = [f"\nvs PTTEA Table 1 (UNet, ACDC->{args.dataset}); DSC in %, ASD px: ours / paper",
                 "| baseline | paper row | LV DSC | LV ASD | Myo DSC | Myo ASD |", "|---|---|---|---|---|---|"]
        for mine, theirs in pairs:
            if mine not in metrics: continue
            e, my, r = metrics[mine]["per_class"]["Endo/LV"], metrics[mine]["per_class"]["Myocardium"], ref[theirs]
            lines.append(f"| {mine} | {theirs} | {100*e['dice']:.2f} / {r['lv_dsc']} | {e['asd']:.2f} / {r['lv_asd']} | "
                         f"{100*my['dice']:.2f} / {r['myo_dsc']} | {my['asd']:.2f} / {r['myo_asd']} |")
        cmp_md = "\n".join(lines); print(cmp_md)
        open(os.path.join(out, "RESULTS.md"), "a").write("\n" + cmp_md + "\n")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
