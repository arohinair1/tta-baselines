"""tta_wrapper: a thin, tested wrapper around
   external/pttea_seg   (Zhang et al., PTTEA, ICCV 2025)
   external/pseudolabel (Nicole's fork adding pseudolabel TTA)
exposing only the loss and adaptation components."""
from ._bootstrap import bootstrap, SEG_CKPT, ENERGY_CKPT, EXAMPLE_DATA  # noqa: F401
from .losses import (energy_loss, pseudolabel_ce_loss, entropy_loss, tta_loss,  # noqa: F401
                     patch_labels_l1, patch_labels_iou, energy_train_loss, DiceCrossEntropyLoss)
from .adapt import (configure_model_for_tent, adapt_slice, adapt_slice_repo,  # noqa: F401
                    adapt_sequence, predict_no_adapt, SliceResult)
from .models import load_seg_model, load_energy_model  # noqa: F401
from .data import load_mat_dir, load_mat_slice, Slice  # noqa: F401
from .metrics import compute_dice, compute_iou, compute_asd, summarize  # noqa: F401

from .interface import PTTEA  # noqa: F401
