#!/usr/bin/env python3
"""Print one table from results/sweep_*/metrics.json. Usage: python scripts/collect_sweep.py [dataset]"""
import glob, json, os, re, sys
ds = sys.argv[1] if len(sys.argv) > 1 else "myops"
ref = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "pttea_table1_unet_acdc.json")))
rows = []
for d in sorted(glob.glob(f"results/sweep_{ds}_b*_lr*")):
    f = os.path.join(d, "metrics.json")
    if not os.path.exists(f): continue
    m = json.load(open(f))["metrics"]; bs, lr = re.search(r"_b(\d+)_lr(\S+)$", d).groups()
    for k in ("no_adapt", "pttea"):
        if k in m:
            e, my = m[k]["per_class"]["Endo/LV"], m[k]["per_class"]["Myocardium"]
            rows.append((k, bs, lr, 100*e["dice"], e["asd"], 100*my["dice"], my["asd"]))
print(f"| run | batch | lr | LV DSC | LV ASD | Myo DSC | Myo ASD |\n|---|---|---|---|---|---|---|")
seen = set()
for r in rows:
    if r[0] == "no_adapt":
        if "na" in seen: continue
        seen.add("na"); print(f"| no_adapt | - | - | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} |")
    else:
        print(f"| pttea | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} |")
if ds in ref:
    for k in ("Pretrained", "Ours"):
        v = ref[ds][k]; print(f"| paper {k} | | | {v['lv_dsc']} | {v['lv_asd']} | {v['myo_dsc']} | {v['myo_asd']} |")
