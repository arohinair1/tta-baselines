#!/usr/bin/env python3
"""Run an upstream script unmodified with the same compat patches the wrapper
uses (inspect.getargspec on py>=3.11, torch.load weights_only=False).
Loads only tta_wrapper/_bootstrap.py by path (NOT the tta_wrapper package),
because the package imports Nicole's `models` package, which would shadow
AdaCS's `models.voxelmorph`.
Usage: python scripts/_compat_run.py <script.py> [args...]   (cwd = script's dir)"""
import importlib.util, os, runpy, sys

_bp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tta_wrapper", "_bootstrap.py")
spec = importlib.util.spec_from_file_location("_bootstrap_standalone", _bp)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod._patch_getargspec(); mod._patch_torch_load()

if os.environ.get("COMPAT_FORCE_CPU"):   # local smoke tests only: upstream hard-codes 'cuda'
    import torch
    _to = torch.nn.Module.to; _tto = torch.Tensor.to
    def _fix(a, kw):
        a = tuple("cpu" if (isinstance(x, str) and x.startswith("cuda")) or isinstance(x, torch.device) and x.type == "cuda" else x for x in a)
        if "device" in kw and str(kw["device"]).startswith("cuda"): kw["device"] = "cpu"
        return a, kw
    torch.nn.Module.to = lambda self, *a, **kw: _to(self, *_fix(a, kw)[0], **_fix(a, kw)[1])
    torch.Tensor.to = lambda self, *a, **kw: _tto(self, *_fix(a, kw)[0], **_fix(a, kw)[1])
    torch.Tensor.cuda = lambda self, *a, **kw: self; torch.nn.Module.cuda = lambda self, *a, **kw: self
    torch.cuda.is_available = lambda: False
    torch.cuda.current_device = lambda: 0
    torch.cuda.memory_allocated = lambda *a, **k: 0; torch.cuda.memory_reserved = lambda *a, **k: 0

script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path(script, run_name="__main__")
