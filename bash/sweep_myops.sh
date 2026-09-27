#!/bin/bash
# Sweep batch size x learning rate for PTTEA on a dataset (default MyoPS, ~1 min per job).
# Usage: bash bash/sweep_myops.sh [dataset] [data_dir]
DS=${1:-myops}; DATA=${2:-$HOME/Dataset/MyoPS_Processed/all}
for bs in ${BS:-1 8 24}; do for lr in ${LR:-1e-2 1e-3 1e-4}; do for em in ${EM:-0 1}; do
  flag=""; tag=""; [ "$em" = 1 ] && flag="--energy_mask" && tag="_em"
  SKIP_TESTS=1 sbatch --job-name=sw_${DS}_b${bs}_lr${lr}${tag} bash/reproduce_gpu.sh \
    --dataset $DS --data "$DATA" --only no_adapt pttea --batch_size $bs --lr $lr $flag \
    --out results/sweep_${DS}_b${bs}_lr${lr}${tag}
done; done; done
