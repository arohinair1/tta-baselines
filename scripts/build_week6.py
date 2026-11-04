#!/usr/bin/env python
"""
Week 6: assemble the econometric dataset from run_cine_tta.py outputs and run first regressions.

Unit of observation: one sequence = (patient, slice, direction). For each adaptation
setting s in the run, the outcome is

    dDice_s = Dice(end frame, setting s) - Dice(end frame, no_adapt)        (Myo and LV)

Regressors (all computable WITHOUT ground truth, so they could drive an adapt/don't-adapt rule):
    energy_start        energy score of the source model's prediction at the start frame
                        (mean sigmoid of the patch energy; 1 = every patch looks clean)
    energy_mean         mean pre-adaptation energy score over the visited frames
    energy_gain_mean    mean (energy_after - energy_before) over frames: how much the loss moved the energy
    disp_mean           mean displacement of the heart between visited frames (px)
    pl_loss_mean        mean pseudolabel loss at the last iteration (settings with a pseudolabel)
    fg_start            foreground pixels of the start prediction (heart size / apex-base proxy)
    n_steps             number of adaptation steps in the sequence
    base_dice           no_adapt Dice at the START frame (how good the source model already is)
patient-level covariates from the M&Ms metadata csv: VendorName, Centre, Pathology.

Regressions (OLS, standard errors clustered by patient):
    (1) dDice ~ energy_start + disp_mean + fg_start + n_steps + base_dice
    (2) (1) + vendor dummies
    (3) (1) + energy_gain_mean  (post-treatment variable: descriptive only, flagged as such)

Usage (Bouchet):
  pip install --user pandas statsmodels          # once
  python scripts/build_week6.py --runs results/cine_sweep/base results/cine_sweep/carry_lp5 \
      --meta ~/Dataset/MnM/OpenDataset/211230_M\&Ms_Dataset_information_diagnosis_opendataset.csv --out results/week6
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_run(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "per_frame.csv"))
    df["run"] = os.path.basename(run_dir.rstrip("/"))
    return df


def build(df):
    """per-frame rows -> one row per (run, setting, patient, slice, direction)."""
    key = ["run", "patient", "slice", "direction"]
    rows = []
    for (run, pat, sl, d), g in df.groupby(key):
        base = g[g.setting == "no_adapt"]
        if base.empty:
            continue
        b0, b1 = base[base.step == 0].iloc[0], base[base.step == base.step.max()].iloc[0]
        for s, gs in g.groupby("setting"):
            s0, s1 = gs[gs.step == 0].iloc[0], gs[gs.step == gs.step.max()].iloc[0]
            rows.append(dict(
                run=run, setting=s, patient=pat, slice=sl, direction=d, n_steps=int(s1.n_steps),
                base_dice_myo_start=b0.dice_myo_start, base_dice_myo_end=b1.dice_myo_end, base_dice_lv_end=b1.dice_lv_end,
                dice_myo_end=s1.dice_myo_end, dice_lv_end=s1.dice_lv_end,
                dDice_myo=s1.dice_myo_end - b1.dice_myo_end, dDice_lv=s1.dice_lv_end - b1.dice_lv_end,
                dDice_myo_start=s0.dice_myo_start - b0.dice_myo_start,
                energy_start=gs[gs.step == 0].energy_before.iloc[0] if s != "no_adapt" else np.nan,
                energy_mean=gs.energy_before.mean(), energy_gain_mean=(gs.energy_after - gs.energy_before).mean(),
                disp_mean=gs.disp_px.mean(), pl_loss_mean=gs.pl_loss.mean(), fg_start=int(s0.fg_px), trust_mean=gs.trust_frac.mean()))
    return pd.DataFrame(rows)


def add_meta(ds, meta_csv):
    if not meta_csv or not os.path.exists(meta_csv):
        print("no metadata csv; vendor/centre/pathology not added"); return ds
    m = pd.read_csv(meta_csv)
    code = [c for c in m.columns if "code" in c.lower()][0]
    keep = {code: "patient"}
    for c in ("VendorName", "Centre", "Pathology", "Sex", "Age"):
        if c in m.columns:
            keep[c] = c.lower()
    m = m[list(keep)].rename(columns=keep)
    return ds.merge(m, on="patient", how="left")


def regress(ds, setting, out_md, run=None):
    import statsmodels.formula.api as smf
    d = ds[(ds.setting == setting) & ((ds.run == run) if run else True)].dropna(subset=["dDice_myo", "energy_start", "disp_mean"]).copy()
    if d.empty:
        return
    # energy scores in percent so coefficients read as "Dice points per 10 points of energy": scale by 10
    d["energy_start10"] = 10 * d.energy_start; d["energy_gain10"] = 10 * d.energy_gain_mean
    d["fg_start_k"] = d.fg_start / 1000.0; d["base_dice_myo_start100"] = 100 * d.base_dice_myo_start
    d["y"] = 100 * d.dDice_myo
    specs = [("(1) image features", "y ~ energy_start10 + disp_mean + fg_start_k + n_steps + base_dice_myo_start100")]
    if "vendorname" in d and d.vendorname.notna().any():
        specs.append(("(2) + vendor", specs[0][1] + " + C(vendorname)"))
    specs.append(("(3) + energy gain (post-treatment, descriptive)", specs[0][1] + " + energy_gain10"))
    out_md.append(f"### {setting}" + (f"  [run {run}]" if run else "") + f"   n = {len(d)} sequences, {d.patient.nunique()} patients, "
                  f"mean dDice_myo = {d.y.mean():+.2f} (sd {d.y.std():.2f}), share improved = {(d.y > 0).mean():.2f}")
    out_md.append("")
    for name, f in specs:
        try:
            r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.patient})
        except Exception as e:  # noqa: BLE001
            out_md.append(f"{name}: failed ({e})"); continue
        out_md.append(f"**{name}**  R2 = {r.rsquared:.3f}")
        out_md.append("")
        out_md.append("| term | coef | cluster SE | p |"); out_md.append("|---|---|---|---|")
        for t in r.params.index:
            out_md.append(f"| {t} | {r.params[t]:+.3f} | {r.bse[t]:.3f} | {r.pvalues[t]:.3f} |")
        out_md.append("")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True, help="run dirs containing per_frame.csv (must include a no_adapt setting)")
    a.add_argument("--meta", default=None, help="M&Ms metadata csv")
    a.add_argument("--out", default="results/week6")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.concat([load_run(r) for r in args.runs], ignore_index=True)
    ds = add_meta(build(df), args.meta and os.path.expanduser(args.meta))
    ds.to_csv(os.path.join(args.out, "week6_dataset.csv"), index=False)

    md = [f"# Week 6 dataset and first regressions", f"runs: {args.runs}", "",
          "One row per (run, setting, patient, slice, direction). Outcome dDice = Dice at the END frame minus no_adapt Dice at the end frame (Myo, in Dice points).",
          "Energy scores are in [0,1] (fraction of patches the energy model calls clean); coefficients below are per 0.1 of energy.",
          "Standard errors clustered by patient.", "",
          "## Summary by run and setting (Myo, end frame)", "",
          "| run | setting | n | mean dDice | sd | share improved | mean energy_start | mean disp px |", "|---|---|---|---|---|---|---|---|"]
    for (run, s), g in ds.groupby(["run", "setting"]):
        if s == "no_adapt":
            continue
        md.append(f"| {run} | {s} | {len(g)} | {100*g.dDice_myo.mean():+.2f} | {100*g.dDice_myo.std():.2f} | {(g.dDice_myo > 0).mean():.2f} | {g.energy_start.mean():.3f} | {g.disp_mean.mean():.2f} |")
    if "vendorname" in ds:
        md += ["", "## Mean dDice (Myo) by vendor", "", "| run | setting | " + " | ".join(sorted(ds.vendorname.dropna().unique())) + " |",
               "|---|---|" + "---|" * ds.vendorname.nunique()]
        for (run, s), g in ds.groupby(["run", "setting"]):
            if s == "no_adapt":
                continue
            md.append(f"| {run} | {s} | " + " | ".join(f"{100*g[g.vendorname == v].dDice_myo.mean():+.2f} (n={int((g.vendorname == v).sum())})" for v in sorted(ds.vendorname.dropna().unique())) + " |")
    md += ["", "## Regressions", ""]
    for run in ds.run.unique():
        for s in ("energy", "energy_pl", "bn_only", "pl"):
            if s in set(ds[ds.run == run].setting):
                regress(ds, s, md, run=run)
    text = "\n".join(md); print(text)
    open(os.path.join(args.out, "REGRESSIONS.md"), "w").write(text + "\n")
    print(f"\nwritten {args.out}/week6_dataset.csv ({len(ds)} rows) and {args.out}/REGRESSIONS.md")


if __name__ == "__main__":
    main()
