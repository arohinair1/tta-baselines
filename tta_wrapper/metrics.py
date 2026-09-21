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
