"""Flow wrapper (AdaCS / VoxelMorph) sanity checks with an untrained model."""
import os, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tta_wrapper as tw  # noqa: E402
from tta_wrapper import flow as F  # noqa: E402


def _fresh_ckpt(tmp):
    m = F.vxm().networks.VxmDense(inshape=(256, 256), bidir=True, int_steps=7, int_downsize=2)
    p = os.path.join(tmp, "motion_test.pt"); m.save(p); return p


def test_spatial_transformer_zero_flow_is_identity():
    st = F.vxm().layers.SpatialTransformer((256, 256), mode="nearest")
    lbl = torch.zeros(1, 1, 256, 256); lbl[0, 0, 90:150, 100:160] = 2; lbl[0, 0, 110:130, 120:140] = 1
    out = st(lbl, torch.zeros(1, 2, 256, 256))
    assert torch.equal(out, lbl)


def test_registrar_matches_register_and_warp_2d_contract():
    with tempfile.TemporaryDirectory() as tmp:
        r = F.FlowRegistrar(_fresh_ckpt(tmp), cache_file=os.path.join(tmp, "c.npz"))
        a, b = tw.load_mat_dir()["patient11"]
        fl = r.estimate_flow(a.image, b.image)
        assert fl.shape == (2, 256, 256)
        w = r(b.image, a.image, a.label.astype(np.uint8))        # same signature as demons/affine
        assert w.shape == a.label.shape and w.dtype == np.uint8 and set(np.unique(w)) <= {0, 1, 2}
        probs = r(b.image, a.image, np.random.rand(3, 256, 256).astype(np.float32))  # float -> bilinear
        assert probs.shape == (3, 256, 256)
        assert r.estimate_flow(a.image, b.image) is fl              # cached
        assert os.path.exists(r.save_cache())


def test_vxm_plugs_into_adapt_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        F.register("vxm_test", F.FlowRegistrar(_fresh_ckpt(tmp)))
        seg, en = tw.load_seg_model(), tw.load_energy_model()
        a, b = tw.load_mat_dir()["patient11"]
        res = tw.adapt_sequence(seg, en, [a.tensor(), b.tensor()], strategy="hard",
                                registration="vxm_test", num_iterations=2)
        assert res[0].warped_label is None and res[1].warped_label is not None
