# tta-baselines

Reproduction of the test-time-adaptation baselines from
[26nicolet/pseudolabel](https://github.com/26nicolet/pseudolabel) (Nicole's fork) and its
upstream [Voldemort108X/pttea_seg](https://github.com/Voldemort108X/pttea_seg)
(Zhang et al., *Progressive Test Time Energy Adaptation for Medical Image Segmentation*, ICCV 2025),
wrapped so that only the **loss** and **adaptation** pieces are exposed.

Both repos are vendored **unmodified** as git submodules under `external/`. Every fix
needed to run them lives in `tta_wrapper/_bootstrap.py` so it is obvious what had to change.

```
tta-baselines/
├── external/
│   ├── pttea_seg/        upstream: code + checkpoints/ + example_data/ (24 ACDC slices)
│   └── pseudolabel/      Nicole's fork: adds pseudolabel TTA, eval_tta.py, registration
├── tta_wrapper/
│   ├── _bootstrap.py     sys.path + compat patches (no edits to external/)
│   ├── losses.py         <- all loss terms, isolated and documented
│   ├── adapt.py          <- TTA loop (reproduces run_pttea.py AND eval_tta.py)
│   ├── models.py         checkpoint loading
│   ├── data.py           .mat slice loading / preprocessing / case grouping
│   └── metrics.py        Dice / IoU / ASD re-exported from eval_tta
├── scripts/reproduce.py  runs every baseline, writes results/<run>/RESULTS.md
├── tests/test_losses.py  proves the wrapper == the original code
└── results/example_acdc/ numbers from this machine (CPU)
```

## Setup

```bash
git clone --recurse-submodules <this repo>   # or: git submodule update --init
cd tta-baselines
pip install -r requirements.txt
python -m pytest tests -q                    # 10 tests, ~15 s on CPU
python scripts/reproduce.py                  # all 7 baselines, ~5 min on CPU
```

On Bouchet, `module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0` then
`pip install --user SimpleITK monai h5py nibabel tensorboard pytest`, and pass `--device cuda`.

## What is reproduced

Everything runs on the upstream checkpoints (`unet_acdc_seg.pt`, `unet_acdc_energy.pt`,
trained on ACDC) and the 24 bundled ACDC example slices (12 patients × ED/ES). Nicole's
own trained checkpoints and the full ACDC / MnM / LVQuant / MyoPS datasets are **not** in
either repo (gitignored), so those numbers need her files; see "What you need from Nicole".

| baseline | original script | wrapper call |
|---|---|---|
| `no_adapt` | `eval.py` | `predict_no_adapt` |
| `pttea` | `run_pttea.py` (energy only, 10 iters, no early stop) | `adapt_slice(strategy='none', variant='pttea')` |
| `pttea_evaltta` | `eval_tta.py` on a first slice (energy only + early stop) | `adapt_slice(strategy='none', variant='eval_tta')` |
| `pl_hard` | `eval_tta.py --pseudolabel_strategy hard` | `adapt_slice(strategy='hard')` |
| `pl_hard_affine` | `... --registration affine` | `adapt_slice(strategy='hard', registration='affine')` |
| `pl_confidence` | `... --pseudolabel_strategy confidence` | `adapt_slice(strategy='confidence')` |
| `pl_entropy` | `... --pseudolabel_strategy entropy` | `adapt_slice(strategy='entropy')` |

Results on the example slices, CPU, `lr=0.01`, 10 iterations, `results/example_acdc/RESULTS.md`:

| baseline | Dice fg | Dice endo | Dice myo | IoU myo | ASD endo | ASD myo |
|---|---|---|---|---|---|---|
| no_adapt | 0.4242 | 0.4937 | 0.3548 | 0.2377 | 10.661 | 13.499 |
| pttea | 0.6508 | 0.7182 | 0.5834 | 0.4240 | 10.487 | 8.081 |
| pttea_evaltta | 0.6489 | 0.7145 | 0.5833 | 0.4238 | 10.638 | 8.201 |
| pl_hard | 0.6669 | 0.7331 | 0.6007 | 0.4398 | 10.288 | 7.861 |
| pl_hard_affine | 0.6675 | 0.7378 | 0.5971 | 0.4370 | 9.977 | 7.952 |
| pl_confidence | 0.6682 | 0.7338 | 0.6025 | 0.4417 | 10.315 | 7.850 |
| pl_entropy | 0.6462 | 0.7134 | 0.5791 | 0.4203 | 10.597 | 7.944 |

The example slices are the hard cases the upstream demo picked, so absolute Dice is low;
the ordering (adaptation helps a lot, pseudolabel adds ~1.5 Dice points over energy-only
on this tiny set, entropy does not) is what matters. Two independent checks that the
wrapper reproduces the originals:

* `tests/test_losses.py::test_adapt_slice_equals_repo_loop` runs Nicole's
  `eval_tta.adapt_and_predict` and `tta_wrapper.adapt_slice` on the same slice and asserts
  identical predictions, probabilities (1e-6) and per-iteration losses, for all three strategies.
* `test_matches_upstream_demo_notebook` reproduces the loss trajectory printed in
  `external/pttea_seg/demo.ipynb` (iteration 0: −6.5427, iteration 9: −8.84 there vs −8.85 here on CPU).

## The losses (the part that matters)

`tta_wrapper/losses.py`, all pure tensor functions:

```python
import tta_wrapper as tw

# TTA objective, one iteration (eval_tta.adapt_and_predict / run_pttea.py)
parts = tw.tta_loss(seg_logits, energy_model, strategy='hard',
                    warped_label=warped_prev_pred, pixel_weights=warped_prev_conf)
parts.total, parts.energy, parts.pseudolabel

tw.energy_loss(energy_logits)                   # -BCEWithLogits(E, 0) == -mean(softplus(E))
tw.pseudolabel_ce_loss(logits, warped_label)    # 'hard'
tw.pseudolabel_ce_loss(logits, warped_label, w) # 'confidence' : mean(CE_map * w)
tw.entropy_loss(logits)                         # 'entropy'    : mean pixel entropy

# energy-model training (train_energy.py)
lbl = tw.patch_labels_l1(clean_onehot, perturbed_probs, n_blocks=16, threshold=50)  # utils.create_labels
lbl = tw.patch_labels_iou(clean_onehot, perturbed_probs, n_blocks=16, iou_threshold=0.6)
tw.energy_train_loss(energy_model(perturbed_probs), lbl)                             # BCEWithLogits

# 2-channel pseudolabel seg model training (train_pseudolabel.py)
tw.DiceCrossEntropyLoss(num_classes=3)          # 0.5*Dice + 0.5*CE
```

What the TTA objective is, in one line: for each test slice, copy the pretrained UNet,
unfreeze only the BatchNorm affine parameters (running stats dropped, batch stats used),
and take 10 Adam steps (lr 0.01) on

    L = −mean softplus(E(softmax f(x)))  +  1.0 · L_pl

where `E` is the frozen patch-wise energy model (16×16 patch grid on 256×256, trained to
output logit>0 for patches that match ground truth) and `L_pl` is Nicole's pseudolabel
term: CE against the previous slice's prediction warped by Demons (or affine)
registration, optionally weighted per pixel by the previous slice's warped max-softmax
confidence, or replaced by the prediction entropy. On the first slice of each case `L_pl`
is absent, so it reduces to plain PTTEA. `eval_tta.py` also early-stops when `L_pl`
(or `L` when there is no `L_pl`) fails to improve by 1e-4 for 3 iterations.

The adaptation loop is in `tta_wrapper/adapt.py` (`adapt_slice`, `adapt_sequence`), with
the exact differences between `run_pttea.py` and `eval_tta.py` documented at the top of
the file and both reproducible via `variant=`.

## Problems found in the repo (none edited; all handled in the wrapper)

1. **`eval_tta.py` crashes as committed.** `adapt_and_predict` returns 6 values
   (`..., out_state, actual_iters`) but every caller unpacks 5 →
   `ValueError: too many values to unpack` on the first slice. The last commit
   ("add iteration flexibility") added the 6th return without updating callers, so the
   reported eval_tta results must predate it. `tests/test_losses.py::test_repo_eval_tta_has_unpack_bug`
   pins this. The wrapper calls the function and unpacks 6.
2. **Does not import on Python ≥ 3.11.** `models/modelio.py` uses `inspect.getargspec`,
   removed in 3.11. Patched in `_bootstrap.py`. (Upstream's `environment.yml` is Python 3.6 / CUDA 10.2.)
3. **`torch.load` on torch ≥ 2.6** rejects the checkpoints (`weights_only=True` default,
   pickled config dict). Patched in `_bootstrap.py`.
4. `perturbation.py` builds a CUDA augmenter at import time, so `train_energy.py` cannot
   even be imported on a CPU box; and `train_energy.py` hard-requires `wandb`. Only affects
   energy-model training, which needs her data anyway.
5. Data paths are hard-coded relative (`../../Dataset/<name>/test`, `../../Code/pttea_seg/MyoPS_Processed/test`);
   the wrapper takes explicit paths.
6. `run_pttea.py` shuffles a second `DataLoader` it never uses; harmless.

## What you need from Nicole to reproduce the *rest*

* `checkpoints/` from her machine: the seg + energy models she actually evaluated with,
  and the 2-channel pseudolabel-conditioned UNet from `train_pseudolabel.py` (used by `eval_2ch.py`).
* Processed datasets: `Dataset/ACDC/{train,val,test}` (.h5 or .mat slices), `Dataset/MnM/Testing`,
  `Dataset/LVQuant/train`, `MyoPS_Processed/test` (produced by `setup_acdc.py`, `setup_myops.py`).
* Her results table / the numbers she reports, to diff against `scripts/reproduce.py --data <path>`.

Once those are in hand: `python scripts/reproduce.py --data /path/to/ACDC/test --seg_ckpt ... --energy_ckpt ... --device cuda`
runs the same seven baselines on the full test set.
