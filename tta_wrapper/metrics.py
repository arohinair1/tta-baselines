"""Per-class Dice / IoU / ASD, re-exported from the repo's eval_tta so the
numbers are computed by exactly the same code."""
from ._bootstrap import bootstrap

bootstrap()
from eval_tta import compute_dice, compute_iou, compute_asd, CLASS_NAMES  # noqa: E402,F401

import numpy as np


def summarize(dice, iou, asd):
    """Same aggregation as eval_tta.print_metrics: per-class means, and
    'overall' = mean Dice over the two foreground classes."""
    dice, iou, asd = np.asarray(dice), np.asarray(iou), np.asarray(asd)
    md, mi, ma = dice.mean(0), iou.mean(0), np.nanmean(asd, 0)
    return {
        "overall_dice_fg": float(md[1:].mean()),
        "per_class": {
            CLASS_NAMES[i]: {"dice": float(md[i]), "iou": float(mi[i]), "asd": float(ma[i])}
            for i in range(len(CLASS_NAMES))
        },
        "n_slices": int(len(dice)),
    }


def per_case_dice(preds_by_case: dict, labels_by_case: dict, num_classes: int = 3):
    """Paper-style aggregation: stack a case's slices into a volume, one Dice per
    class per case, then average over cases. (Per-slice averaging penalises
    apex/base slices with a few labelled pixels.)"""
    out = []
    for c in preds_by_case:
        P = np.stack(preds_by_case[c]); L = np.stack(labels_by_case[c])
        out.append(compute_dice(P, L, num_classes))
    return np.asarray(out)
