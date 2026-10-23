#!/usr/bin/env bash
# Same run as s600_realv2v3_80 (run_cotrain_realv3.sh step 2) but without
# --source-fractions: every sample is drawn with equal weight, so the batch mix
# follows the data (sim 102,689 / real v2 7,865 / real v3 11,028 samples,
# about 0.845 / 0.065 / 0.091) instead of the fixed sim 0.9289 / real 0.0711.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable; check nvidia-smi (reboot after a driver update)"
PY

SEED=20261002
OUT=outputs/dp_train/s600_realv2v3_80_natural
python -u scripts/train_dp.py --data sim=outputs/synthetic/resip_v3_slow2 \
  --data real_v2=outputs/real_demos/v2_remapped --data real_v3=outputs/real_demos/v3 --out $OUT \
  --steps 100000 --batch-size 128 --num-workers 6 --amp --save-freq 10000 --seed "$SEED"
python -u scripts/eval_dp_sim.py --headless --checkpoint $OUT/dp_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --seed "$SEED" --out $OUT/eval_nominal_40.json
python -u scripts/eval_dp_sim.py --headless --checkpoint $OUT/dp_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --seed "$SEED" --out $OUT/eval_render_dr_40.json \
  --render-randomization
echo "done: $OUT"
