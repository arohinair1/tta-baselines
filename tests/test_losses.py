"""Bit-for-bit checks of tta_wrapper against the original repo code.
Run:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tta_wrapper as tw  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)


@pytest.fixture(scope="module")
def models():
    return tw.load_seg_model(), tw.load_energy_model()


@pytest.fixture(scope="module")
def sample():
    cases = tw.load_mat_dir()
    return cases["patient11"]  # [ED, ES]


# ── A. TTA losses vs eval_tta / run_pttea formulas ────────────────────

def test_energy_loss_matches_repo(models):
    _, en = models
    probs = torch.softmax(torch.randn(1, 3, 256, 256), 1)
    out = en(probs)
    repo = -nn.BCEWithLogitsLoss()(out, torch.zeros_like(out))
    assert torch.allclose(tw.energy_loss(out), repo)
    # and equals -mean(softplus(logit))
    assert torch.allclose(tw.energy_loss(out), -torch.nn.functional.softplus(out).mean())


def test_hard_ce_matches_repo():
    logits = torch.randn(1, 3, 64, 64)
    lbl = torch.randint(0, 3, (1, 64, 64))
    assert torch.allclose(tw.pseudolabel_ce_loss(logits, lbl), nn.CrossEntropyLoss()(logits, lbl))


def test_confidence_ce_matches_repo():
    logits = torch.randn(1, 3, 64, 64)
    lbl = torch.randint(0, 3, (1, 64, 64))
    w = torch.rand(1, 64, 64)
    repo = (nn.CrossEntropyLoss(reduction="none")(logits, lbl) * w).mean()
    assert torch.allclose(tw.pseudolabel_ce_loss(logits, lbl, w), repo)


def test_entropy_matches_repo():
    logits = torch.randn(1, 3, 64, 64)
    p = torch.softmax(logits, 1)
    repo = -(p * torch.log(p + 1e-8)).sum(1).mean()
    assert torch.allclose(tw.entropy_loss(logits), repo)


# ── B. Training losses vs utils.create_labels* ────────────────────────

def test_patch_labels_match_utils():
    import utils as repo_utils
    a = torch.nn.functional.one_hot(torch.randint(0, 3, (2, 256, 256)), 3).permute(0, 3, 1, 2).float()
    b = torch.softmax(torch.randn(2, 3, 256, 256), 1)
    assert torch.equal(tw.patch_labels_l1(a, b, 16, 50), repo_utils.create_labels(a, b, 16, 50))
    assert torch.equal(tw.patch_labels_iou(a, b, 16, 0.6), repo_utils.create_labels_iou(a, b, 16, 0.6))


# ── C. Full adaptation loop vs Nicole's adapt_and_predict ─────────────

@pytest.mark.parametrize("strategy", ["hard", "confidence", "entropy"])
def test_adapt_slice_equals_repo_loop(models, sample, strategy):
    seg, en = models
    ed, es = sample
    # adapt ED first to get a realistic previous prediction
    torch.manual_seed(0)
    r_prev = tw.adapt_slice(seg, en, ed.tensor(), strategy="none", num_iterations=3)
    kw = dict(strategy=strategy, prev_img=ed.image, prev_label=r_prev.pred.astype(np.uint8),
              prev_conf=r_prev.confidence, num_iterations=5, lr=0.01)
    torch.manual_seed(0)
    mine = tw.adapt_slice(seg, en, es.tensor(), variant="eval_tta", **kw)
    torch.manual_seed(0)
    repo = tw.adapt_slice_repo(seg, en, es.tensor(), **kw)
    assert mine.iters_run == repo.iters_run
    assert np.array_equal(mine.pred, repo.pred)
    assert torch.allclose(mine.probs, repo.probs, atol=1e-6)
    mine_pl = [d["pseudolabel"] for d in mine.losses]
    repo_pl = [d["pseudolabel"] for d in repo.losses]
    assert np.allclose(mine_pl, repo_pl, atol=1e-6)


def test_repo_eval_tta_has_unpack_bug():
    """Documents the bug: eval_tta's callers unpack 5 values from a 6-tuple."""
    import inspect
    import eval_tta
    src = inspect.getsource(eval_tta.eval_acdc_tta)
    assert "pred, _, conf, i_losses, _ = adapt_and_predict(" in src
    assert "return final_cls, final_probs, confidence_map, iter_losses, out_state, actual_iters" \
        in inspect.getsource(eval_tta.adapt_and_predict)


def test_matches_upstream_demo_notebook(models):
    """external/pttea_seg/demo.ipynb adapts all 24 example slices as ONE batch
    and prints the energy loss per iteration; iteration 0 is -6.542713."""
    import copy
    seg, en = models
    cases = tw.load_mat_dir()
    X = torch.cat([s.tensor() for v in cases.values() for s in v], 0)
    m = tw.configure_model_for_tent(copy.deepcopy(seg))
    parts = tw.tta_loss(m(X), en, strategy="none")
    assert abs(parts.total.item() - (-6.542713)) < 1e-3
