#!/usr/bin/env bash
set -euo pipefail
cd /home/keti/workspace/research/so101-team
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Do not silently run this long experiment on CPU.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; run this from the GPU-enabled host environment"'

python -u scripts/train_act_cotrain.py \
  --data sim=outputs/synthetic/resip_v3_slow2 \
  --data real=outputs/real_demos/v2_remapped \
  --out outputs/act_train/cotrain_v3_sep_camera \
  --device cuda --steps 200000 --batch-size 8 --chunk-size 100 \
  --num-workers 6 --log-freq 500 --save-freq 10000 \
  --separate-camera-backbones

python -u scripts/eval_act_sim.py \
  --headless --checkpoint outputs/act_train/cotrain_v3_sep_camera/act_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --n-action-steps 10 \
  --out outputs/act_train/cotrain_v3_sep_camera/eval_nominal_n10_40.json

python -u scripts/eval_act_sim.py \
  --headless --checkpoint outputs/act_train/cotrain_v3_sep_camera/act_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --n-action-steps 10 \
  --render-randomization \
  --out outputs/act_train/cotrain_v3_sep_camera/eval_dr_n10_40.json
