#!/bin/bash
#SBATCH --job-name=ew_sweep
#SBATCH --account=cpsc4900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/ew_sweep_%j.out
# 11/17 next steps: (1) is energy an oracle for pseudolabel quality? (2) energy-weighted pseudolabel loss, sharpness sweep.
FLOW=${1:-$HOME/adacs_mnm/motion_0160.pt}
cd "$SLURM_SUBMIT_DIR"
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 torchvision/0.24.1-foss-2024a-CUDA-12.8.0
echo "=== calibration (train pairs, 300)"
python scripts/energy_calibration.py --pairs $HOME/Dataset/MnM_cine/pairs_edes/train --flow_ckpt $FLOW --out results/energy_calib --device cuda --max_pairs 300 2>&1 | grep -v -i "warn\|meshgrid"
COMMON="--frames $HOME/Dataset/MnM_cine/frames/holdout --flow_ckpt $FLOW --device cuda --stride 5 --energy_mask --directions ed2es --n_vis 1 --carry_state --lambda_p 5"
run() { name=$1; shift; echo "=== $name"; python scripts/run_cine_tta.py $COMMON --out results/ew_sweep/$name "$@" 2>&1 | grep -v -i "warn\|meshgrid"; }
run ref_carry_lp5     --settings no_adapt bn_only energy pl energy_pl
run ew_k0             --settings energy_pl --pl_weight energy --energy_sharpness 0
run ew_k2             --settings energy_pl --pl_weight energy --energy_sharpness 2
run ew_k5             --settings energy_pl --pl_weight energy --energy_sharpness 5
run ew_k10            --settings energy_pl --pl_weight energy --energy_sharpness 10
run ew_k5_lp10        --settings energy_pl --pl_weight energy --energy_sharpness 5 --lambda_p 10
run ew_k5_trust       --settings energy_pl --pl_weight energy --energy_sharpness 5 --trust
run ew_k5_iters30     --settings energy_pl --pl_weight energy --energy_sharpness 5 --num_iterations 30
run ew_k5_pl_only     --settings pl        --pl_weight energy --energy_sharpness 5
echo; echo "=== SUMMARY (end-frame LV / Myo, slice and volume; weight% = mean pseudolabel weight inside the heart)"
for d in results/ew_sweep/*/; do n=$(basename $d); python - "$d/metrics.json" "$n" <<'PY'
import json,sys; m=json.load(open(sys.argv[1]))["metrics"]
for s,v in m.items(): print(f"{sys.argv[2]:16s} {s:10s}  start {100*v['start_lv']:.2f}/{100*v['start_myo']:.2f}   end-slice {100*v['end_lv']:.2f}/{100*v['end_myo']:.2f}   end-vol {100*v['end_lv_vol']:.2f}/{100*v['end_myo_vol']:.2f}   weight {100*v['trust']:.0f}%  plE {v['pl_energy']:.3f}  BNdelta {v['bn_delta']:.4f}")
PY
done
