#!/bin/bash
#SBATCH --job-name=final_runs
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/final_runs_%j.out
# Final experiment: best setting (carry_state, lambda_p 5, stride 5), all five arms, BOTH directions,
# on (a) the GT-centroid crop used so far and (b) a crop centred on the model's own ED prediction (no leak).
# Then the Week 6 dataset + regressions on each.
FLOW=${1:-$HOME/adacs_mnm/motion_0160.pt}
META=$(ls $HOME/Dataset/MnM/OpenDataset/*.csv | head -1)
cd "$SLURM_SUBMIT_DIR"
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
echo "=== prep: predicted-centroid crop (holdout only)"
python scripts/prep_mnm_cine.py --data $HOME/Dataset/MnM/OpenDataset/Testing --out $HOME/Dataset/MnM_cine_pred --holdout 36 --crop_from pred --only_holdout 2>&1 | grep -v -i "warn\|meshgrid" | tail -2
SET="--settings no_adapt bn_only energy pl energy_pl --device cuda --stride 5 --energy_mask --carry_state --lambda_p 5 --directions both --n_vis 2 --flow_ckpt $FLOW"
echo "=== final, GT-centroid crop, both directions"
python scripts/run_cine_tta.py --frames $HOME/Dataset/MnM_cine/frames/holdout $SET --out results/final/gtcrop_both 2>&1 | grep -v -i "warn\|meshgrid" | sed -n '/^| setting/,$p'
echo "=== final, predicted-centroid crop, both directions"
python scripts/run_cine_tta.py --frames $HOME/Dataset/MnM_cine_pred/frames/holdout $SET --out results/final/predcrop_both 2>&1 | grep -v -i "warn\|meshgrid" | sed -n '/^| setting/,$p'
echo "=== week 6 on the final runs"
python scripts/build_week6.py --runs results/final/gtcrop_both --meta "$META" --out results/week6_final_gtcrop 2>&1 | sed -n '/## Summary/,$p'
python scripts/build_week6.py --runs results/final/predcrop_both --meta "$META" --out results/week6_final_predcrop 2>&1 | sed -n '/## Summary/,$p'
echo "=== done"
