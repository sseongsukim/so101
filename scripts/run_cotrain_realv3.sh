#!/usr/bin/env bash
# Co-train DP on the slower real demos (real_demos/v3), then on v2 + v3 (80 real),
# under the data-count ablation's conditions: 100k steps, batch 128, AMP,
# seed 20261002, sim/real batch mix fixed at 0.9289 / 0.0711. Each run is
# followed by the ablation's sim evaluation (40 nominal + 40 rendering-DR).
# Run sequentially on the single GPU; stops on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable; check nvidia-smi (reboot after a driver update)"
PY

SIM=outputs/synthetic/resip_v3_slow2
SEED=20261002
COMMON=(--steps 100000 --batch-size 128 --num-workers 6 --amp --save-freq 10000 --seed "$SEED")

evaluate() {
  local ckpt=$1/dp_so101.pt
  python -u scripts/eval_dp_sim.py --headless --checkpoint "$ckpt" \
    --num-envs 4 --num-rounds 10 --max-steps 300 --seed "$SEED" --out "$1/eval_nominal_40.json"
  python -u scripts/eval_dp_sim.py --headless --checkpoint "$ckpt" \
    --num-envs 4 --num-rounds 10 --max-steps 300 --seed "$SEED" --out "$1/eval_render_dr_40.json" \
    --render-randomization
}

# 1) sim 600 + real v3 (40 slower demos)
OUT1=outputs/dp_train/s600_realv3_40
python -u scripts/train_dp.py --data sim=$SIM --data real=outputs/real_demos/v3 --out $OUT1 "${COMMON[@]}" \
  --source-fractions sim=0.928858295494,real=0.071141704506
evaluate $OUT1

# 2) sim 600 + real v2 (40) + real v3 (40) = 80 real; the real share stays 0.0711,
#    split between the two in proportion to their samples (7,865 / 11,028).
OUT2=outputs/dp_train/s600_realv2v3_80
python -u scripts/train_dp.py --data sim=$SIM --data real_v2=outputs/real_demos/v2_remapped \
  --data real_v3=outputs/real_demos/v3 --out $OUT2 "${COMMON[@]}" \
  --source-fractions sim=0.928858295494,real_v2=0.029615704543,real_v3=0.041525999963
evaluate $OUT2

echo "done: $OUT1 $OUT2"
