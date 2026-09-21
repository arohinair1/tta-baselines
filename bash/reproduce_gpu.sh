#!/bin/bash
#SBATCH --job-name=tta_repro
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/tta_repro_%j.out
# Usage (from the repo root on Bouchet):
#   mkdir -p slurm_logs && sbatch bash/reproduce_gpu.sh [extra args to scripts/reproduce.py]
# e.g. sbatch bash/reproduce_gpu.sh --data /path/to/ACDC/test --seg_ckpt ... --energy_ckpt ...
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
cd "$(dirname "$0")/.."
python -m pytest tests -q -p no:cacheprovider
python scripts/reproduce.py --device cuda "$@"
