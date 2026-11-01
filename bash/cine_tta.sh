#!/bin/bash
#SBATCH --job-name=cine_tta
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/cine_tta_%j.out
# Week 5: sequential TTA over M&Ms cine frames, 36 held-out patients, four settings each.
#   A  realistic: start from the model's own ED prediction
#   B  A + trusted-pixel mask on the pseudolabel loss
#   C  oracle: start from the ED ground truth (isolates propagation quality)
# Usage: sbatch bash/cine_tta.sh [flow_ckpt] [stride]
FLOW=${1:-$HOME/adacs_mnm/motion_0160.pt}; STRIDE=${2:-5}
cd "$SLURM_SUBMIT_DIR"
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
COMMON="--frames $HOME/Dataset/MnM_cine/frames/holdout --flow_ckpt $FLOW --device cuda --stride $STRIDE --energy_mask --directions ed2es --n_vis 2"
set -x
python scripts/run_cine_tta.py $COMMON --out results/cine_tta/A_pred
python scripts/run_cine_tta.py $COMMON --out results/cine_tta/B_pred_trust --trust
python scripts/run_cine_tta.py $COMMON --out results/cine_tta/C_gt_start --start gt
set +x
echo; echo "=== SUMMARY ==="
for d in A_pred B_pred_trust C_gt_start; do echo "## $d"; sed -n '/^| setting/,$p' results/cine_tta/$d/RESULTS.md; echo; done
