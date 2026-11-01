#!/bin/bash
#SBATCH --job-name=cine_sweep
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/cine_sweep_%j.out
# Week 5 tuning sweep on cine frames: give the pseudolabel more leverage.
# Usage: sbatch bash/cine_sweep.sh [flow_ckpt]
FLOW=${1:-$HOME/adacs_mnm/motion_0160.pt}
cd "$SLURM_SUBMIT_DIR"
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
COMMON="--frames $HOME/Dataset/MnM_cine/frames/holdout --flow_ckpt $FLOW --device cuda --stride 5 --energy_mask --directions ed2es --n_vis 1"
run() { name=$1; shift; echo "=== $name"; python scripts/run_cine_tta.py $COMMON --out results/cine_sweep/$name "$@" 2>&1 | grep -v -i "warn\|meshgrid"; }
run base            --settings no_adapt bn_only energy pl energy_pl
run lp2             --settings energy_pl --lambda_p 2
run lp5             --settings energy_pl --lambda_p 5
run lp10            --settings energy_pl --lambda_p 10
run iters20         --settings energy energy_pl --num_iterations 20
run iters30_lp5     --settings energy_pl --num_iterations 30 --lambda_p 5
run lr3e-3          --settings energy energy_pl --lr 3e-3
run carry           --settings energy pl energy_pl --carry_state
run carry_lp5       --settings energy_pl --carry_state --lambda_p 5
run carry_lr1e-4    --settings energy energy_pl --carry_state --lr 1e-4
run lambda_energy   --settings energy_pl --lambda_mode energy --lambda_p 5
run gce_lp5         --settings energy_pl --pl_kind gce --lambda_p 5
run stride3_lp5     --settings energy energy_pl --lambda_p 5 --stride 3
run stride1_lp5     --settings energy energy_pl --lambda_p 5 --stride 1
echo; echo "=== SUMMARY (end-frame, per-volume LV / Myo)"
for d in results/cine_sweep/*/; do n=$(basename $d); python - "$d/metrics.json" "$n" <<'PY'
import json,sys; m=json.load(open(sys.argv[1]))["metrics"]
for s,v in m.items(): print(f"{sys.argv[2]:16s} {s:10s}  start {100*v['start_lv']:.2f}/{100*v['start_myo']:.2f}   end-slice {100*v['end_lv']:.2f}/{100*v['end_myo']:.2f}   end-vol {100*v['end_lv_vol']:.2f}/{100*v['end_myo_vol']:.2f}")
PY
done
