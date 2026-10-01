#!/usr/bin/env bash
# Run sequentially on the single GPU. Stop on any failed stage.
set -euo pipefail
cd /home/keti/workspace/research/so101-team
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u scripts/generate_teacher_demos.py \
  --headless --num-envs 8 --num-episodes 600 --motion-substeps 2 \
  --post-success-steps 40 --stable-success-steps 15 \
  --wrist-pos-jitter-m 0.01 --wrist-rot-jitter-deg 3 \
  --out outputs/synthetic/resip_v3_slow2

python - <<'PY'
import pickle
from pathlib import Path
import numpy as np
paths = sorted(Path('outputs/synthetic/resip_v3_slow2').glob('trajectory_*.pkl'))
assert len(paths) == 600, len(paths)
for path in paths:
    with path.open('rb') as file:
        data = pickle.load(file)
    n = len(data['actions'])
    assert n == data['first_success_step'] + 41, path
    assert np.all(data['successes'][-15:]), path
    for key in ('front_images', 'wrist_images'):
        assert data[key].shape == (n, 240, 320, 3), path
        assert data[key].dtype == np.uint8, path
    for key in ('observations', 'actions'):
        assert data[key].shape == (n, 6) and np.isfinite(data[key]).all(), path
print('Validated 600 complete, stable episodes', flush=True)
PY

python -u scripts/train_dp.py \
  --data sim=outputs/synthetic/resip_v3_slow2 \
  --data real=outputs/real_demos/v2_remapped \
  --out outputs/dp_train/cotrain_v3_slow2 \
  --steps 100000 --batch-size 128 --num-workers 6 --amp --save-freq 10000

python -u scripts/eval_dp_sim.py \
  --headless --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --num-envs 4 --num-rounds 5 --max-steps 640

python -u scripts/train_act_cotrain.py \
  --data sim=outputs/synthetic/resip_v3_slow2 \
  --data real=outputs/real_demos/v2_remapped \
  --out outputs/act_train/cotrain_v3_slow2 \
  --steps 200000 --batch-size 8 --num-workers 6 --log-freq 500 --save-freq 10000

python -u scripts/eval_act_sim.py \
  --headless --checkpoint outputs/act_train/cotrain_v3_slow2/act_so101.pt \
  --num-envs 4 --num-rounds 5 --max-steps 640

python -u scripts/eval_dp_sim.py \
  --headless --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --num-envs 4 --num-rounds 10 --max-steps 300 --render-randomization \
  --out outputs/dp_train/cotrain_v3_slow2/eval_dr_300.json
