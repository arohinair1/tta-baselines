#!/bin/bash
# Sweep batch size x learning rate for PTTEA on a dataset (default MyoPS, ~1 min per job).
# Usage: bash bash/sweep_myops.sh [dataset] [data_dir]
DS=${1:-myops}; DATA=${2:-$HOME/Dataset/MyoPS_Processed/all}
for bs in 1 8 24; do for lr in 1e-2 1e-3 1e-4; do
  SKIP_TESTS=1 sbatch --job-name=sw_${DS}_b${bs}_lr${lr} bash/reproduce_gpu.sh \
    --dataset $DS --data "$DATA" --only no_adapt pttea --batch_size $bs --lr $lr \
    --out results/sweep_${DS}_b${bs}_lr${lr}
done; done
