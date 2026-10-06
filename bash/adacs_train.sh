#!/bin/bash
#SBATCH --job-name=adacs_myops
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/adacs_%j.out
# Train AdaCS (VoxelMorph motion + scoring net) on MyoPS adjacent-slice pairs,
# using upstream train_vxm.py UNCHANGED. The script hard-codes
# ../../Dataset/<name>/{train,val}/*.mat and only accepts --dataset ACDC|CAMUS|Echo,
# so we build a sandbox of symlinks that satisfies it:
#     $SANDBOX/Code/AdaCS        -> external/AdaCS
#     $SANDBOX/Dataset/ACDC/train -> $PAIRS/train
#     $SANDBOX/Dataset/ACDC/val   -> $PAIRS/val   (empty)
# Usage: sbatch bash/adacs_train.sh <pairs_dir> <model_out_dir> [epochs]
PAIRS=${1:?pairs dir (from prep_myops_pairs.py)}; OUT=${2:?model output dir}; EPOCHS=${3:-150}
REPO="$SLURM_SUBMIT_DIR"; SANDBOX="$HOME/adacs_sandbox"
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
mkdir -p "$SANDBOX/Code" "$SANDBOX/Dataset/ACDC" "$OUT"
rm -rf "$SANDBOX/Code/AdaCS"; cp -rs "$(realpath "$REPO/external/AdaCS")" "$SANDBOX/Code/AdaCS"   # real dir, symlinked files (cwd must be physical)
ln -sfn "$PAIRS/train" "$SANDBOX/Dataset/ACDC/train"
ln -sfn "$PAIRS/val"   "$SANDBOX/Dataset/ACDC/val"
cd "$SANDBOX/Code/AdaCS"
export WANDB_MODE=offline WANDB_DIR="$OUT" VXM_BACKEND=pytorch NEURITE_BACKEND=pytorch
python "$REPO/scripts/_compat_run.py" train_vxm.py --dataset ACDC --bidir --model-dir "$OUT" \
  --motion-loss-type wmse --scoring-loss-type scoringwmse --warm_start \
  --epochs "$EPOCHS" --wandb-name myops_adacs
echo "done. checkpoints:"; ls "$OUT"/*.pt
