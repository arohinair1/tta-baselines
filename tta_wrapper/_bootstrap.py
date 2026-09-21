"""
Puts the vendored repos on sys.path and applies the minimal compatibility
patches needed to import them unchanged.

Nothing under external/ is edited. Every fix lives here so it's obvious what
had to change to make the original code run.

Patches:
  1. inspect.getargspec was removed in Python 3.11; models/modelio.py uses it.
     -> alias to a getfullargspec-based shim (same 4-tuple it expects).
  2. torch>=2.6 defaults torch.load(weights_only=True), which rejects the
     checkpoints' pickled config dicts. -> default weights_only=False.
  3. perturbation.py constructs a CUDA augmenter at import time. On a CPU box
     that raises. -> only relevant for train_energy; not imported by default.
"""
import inspect
import os
import sys
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "external")
PSEUDOLABEL_DIR = os.path.join(EXTERNAL, "pseudolabel")   # Nicole's fork (26nicolet/pseudolabel)
PTTEA_DIR = os.path.join(EXTERNAL, "pttea_seg")            # upstream (Voldemort108X/pttea_seg)

CHECKPOINT_DIR = os.path.join(PTTEA_DIR, "checkpoints")
SEG_CKPT = os.path.join(CHECKPOINT_DIR, "unet_acdc_seg.pt")
ENERGY_CKPT = os.path.join(CHECKPOINT_DIR, "unet_acdc_energy.pt")
EXAMPLE_DATA = os.path.join(PTTEA_DIR, "example_data")


def _patch_getargspec():
    if hasattr(inspect, "getargspec"):
        return
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        fa = inspect.getfullargspec(func)
        return ArgSpec(fa.args, fa.varargs, fa.varkw, fa.defaults)

    inspect.getargspec = getargspec


def _patch_torch_load():
    import torch
    if getattr(torch.load, "_tta_wrapper_patched", False):
        return
    _orig = torch.load

    def load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)

    load._tta_wrapper_patched = True
    torch.load = load


_DONE = False


def bootstrap(which="pseudolabel"):
    """Make `import models`, `import dataset`, `import eval_tta` resolve to the
    chosen vendored repo. Default is Nicole's fork, which is a superset of
    upstream for everything we use (loss + adaptation)."""
    global _DONE
    _patch_getargspec()
    _patch_torch_load()
    repo = PSEUDOLABEL_DIR if which == "pseudolabel" else PTTEA_DIR
    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"{repo} not found. Clone the repos into external/ first "
            "(see README.md)."
        )
    # Put chosen repo first; remove the other to avoid module-name clashes
    other = PTTEA_DIR if which == "pseudolabel" else PSEUDOLABEL_DIR
    sys.path = [p for p in sys.path if p not in (repo, other)]
    sys.path.insert(0, repo)
    # If a different repo's modules were imported earlier, drop them so a
    # re-import picks up the right files.
    for name in ("models", "models.segmentation", "models.energy", "models.modelio",
                 "models.template_models", "dataset", "utils", "eval_tta",
                 "registration_utils", "registration_utils_affine"):
        mod = sys.modules.get(name)
        if mod is not None and not getattr(mod, "__file__", repo).startswith(repo):
            del sys.modules[name]
    _DONE = True
    return repo
