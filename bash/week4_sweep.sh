#!/bin/bash
#SBATCH --job-name=week4
#SBATCH --account=cpsc4900
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/week4_%j.out
# Week 4 experiments on the 5 held-out MyoPS cases, all in ONE job.
#   A. the plan's four settings: no adaptation | energy only | pseudolabel only | energy + pseudolabel
#      (pseudolabel from Demons and from the learned VoxelMorph flow)
#   B. single-knob variations around energy + pseudolabel (Demons):
#      lambda_p in {0.5, 2}, energy-dependent lambda, GCE loss, carried state, lr
# Usage: sbatch bash/week4_sweep.sh   (edit DATA/FLOW/CASES below if paths differ)
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
cd "$SLURM_SUBMIT_DIR"
DATA=$HOME/Dataset/MyoPS_Processed/all
FLOW=$HOME/adacs_myops/motion_0180.pt
CACHE=$HOME/adacs_myops/flows_holdout.npz
CASES="Case121 Case122 Case123 Case124 Case125"
COMMON="--dataset myops --data $DATA --cases $CASES --device cuda --energy_mask --lr 1e-3"
run() { name=$1; shift; echo "=== $name"; python scripts/reproduce.py $COMMON --out results/week4/$name "$@" 2>&1 | grep -E "^\[|Error|error"; }

# A. four main settings
run A_no_adapt            --only no_adapt
run A_energy_only         --only pttea_evaltta
run A_pl_only_demons      --only pl_hard --lambda_e 0 --lambda_p 1
run A_pl_only_vxm         --only pl_hard --lambda_e 0 --lambda_p 1 --registration vxm --flow_ckpt $FLOW --flow_cache $CACHE
run A_energy_pl_demons    --only pl_hard --lambda_e 1 --lambda_p 1
run A_energy_pl_vxm       --only pl_hard --lambda_e 1 --lambda_p 1 --registration vxm --flow_ckpt $FLOW --flow_cache $CACHE

# B. knobs around energy + pseudolabel (Demons)
run B_lp0.5               --only pl_hard --lambda_p 0.5
run B_lp2                 --only pl_hard --lambda_p 2
run B_lambda_energy       --only pl_hard --lambda_mode energy
run B_gce                 --only pl_hard --pl_kind gce
run B_gce_lambda_energy   --only pl_hard --pl_kind gce --lambda_mode energy
run B_carry_state         --only pl_hard --carry_state
run B_carry_lr1e-4        --only pl_hard --carry_state --lr 1e-4
run B_lr1e-4              --only pl_hard --lr 1e-4
run B_iters20             --only pl_hard --num_iterations 20
python scripts/collect_week4.py
