#!/usr/bin/env bash
set -euo pipefail
cd /home/keti/workspace/research/so101-team
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Require the GPU instead of silently starting a much slower CPU run.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check nvidia-smi before resuming"'

python -u scripts/train_act_cotrain.py \
  --data sim=outputs/synthetic/resip_v3_slow2 \
  --data real=outputs/real_demos/v2_remapped \
  --out outputs/act_train/cotrain_v3_slow2 \
  --init-from outputs/act_train/cotrain_v3_slow2/step_0010000.pt \
  --device cuda --steps 200000 --batch-size 8 --num-workers 6 \
  --log-freq 500 --save-freq 10000

python -u scripts/eval_act_sim.py \
  --headless --checkpoint outputs/act_train/cotrain_v3_slow2/act_so101.pt \
  --num-envs 4 --num-rounds 5 --max-steps 640

python -u scripts/eval_dp_sim.py \
  --headless --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --render-randomization \
  --out outputs/dp_train/cotrain_v3_slow2/eval_dr_300.json
