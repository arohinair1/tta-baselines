#!/usr/bin/env python3
"""Tabulate results/week4/*/metrics.json (per-volume Dice, LV / Myo, plus ASD)."""
import glob, json, os
rows = []
for d in sorted(glob.glob("results/week4/*")):
    f = os.path.join(d, "metrics.json")
    if not os.path.exists(f): continue
    m = json.load(open(f))["metrics"]
    for k, v in m.items():
        e, my, pv = v["per_class"]["Endo/LV"], v["per_class"]["Myocardium"], v["per_case_dice"]
        rows.append((os.path.basename(d), k, 100*pv[1], 100*pv[2], 100*e["dice"], 100*my["dice"], e["asd"], my["asd"]))
print("| run | baseline | LV Dice (vol) | Myo Dice (vol) | LV (slice) | Myo (slice) | LV ASD | Myo ASD |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} |")
