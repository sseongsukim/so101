#!/usr/bin/env bash
# Sequential ResiP sweep -- one run at a time, since they share a GPU.
#
#   ./sweep.sh                  # every experiment below, in order
#   ./sweep.sh step gamma99     # only the named ones
#   ITERS=1500 ./sweep.sh step  # a longer run for a winner
#
# Each run writes to exp/sweep/<name>/ and exp/sweep/<name>.log, and the
# summary at the end reads evaluation success out of every run it finds there.
# Needs bash for the arrays below; `sh sweep.sh` runs dash and will not work.
cd "$(dirname "$0")"

BASE=${BASE:-exp/so101-StackCube-v0/dbc/dbc_sd4125_20260922_224312}
EPOCH=${EPOCH:-500000}
ITERS=${ITERS:-400}
SEED=${SEED:-42}
OUT=${OUT:-exp/sweep}

# The settings the 1500-iteration reference run used, minus the video rendering
# and the frequent checkpoints a comparison run does not need.
COMMON=(
  --agent=agents/resip.py
  --seed="$SEED"
  --restore_path="$BASE"
  --restore_epoch="$EPOCH"
  --online_iters="$ITERS"
  --agent.init_log_std=-3.0
  --agent.critic_warmup_iters=10
  --eval_interval=20
  --video_envs=0
  --save_interval=100
  --wandb_mode=offline
)

# name | flags on top of COMMON
#   baseline     the run already measured, at this shorter budget
#   step         approx_kl sat 9x under target_kl, so take bigger steps
#   step_critic  same, plus the critic that explained_variance says is falling behind
#   gamma99      a 300-step episode does not need a 1000-step effective horizon
#   learn_std    let PPO size its own exploration
#   dense        shaping aimed at the stalled axis: succeeding earlier
RUNS=(
  "baseline|"
  "step|--agent.lr=1e-3 --agent.num_minibatches=4"
  "step_critic|--agent.lr=1e-3 --agent.num_minibatches=4 --agent.critic_min_lr=5e-4 --agent.critic_hidden_dims=(512,512)"
  "gamma99|--agent.gamma=0.99"
  "learn_std|--agent.learn_std=true"
)

# Two runs on one GPU will not fit, so stop unless FORCE=1 says otherwise.
running=$(pgrep -af "python online.py" | grep -v "bash -c" || true)
if [ -n "$running" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "online.py가 이미 실행 중이라 sweep을 시작하지 않습니다:" >&2
  echo "$running" | sed 's/^/  /' >&2
  echo "끝나기를 기다리거나, 종료 후 다시 실행하세요 (무시하려면 FORCE=1 ./sweep.sh)." >&2
  exit 1
fi

select_runs="$*"
mkdir -p "$OUT"
started=$(date +%s)

for entry in "${RUNS[@]}"; do
  name=${entry%%|*}
  extra=${entry#*|}
  if [ -n "$select_runs" ] && [[ " $select_runs " != *" $name "* ]]; then
    continue
  fi
  log="$OUT/$name.log"
  echo "=== $name  ($(date +%H:%M))  -> $log"
  python online.py "${COMMON[@]}" \
    --save_dir="$OUT/$name" --run_group="sweep_$name" $extra > "$log" 2>&1
  status=$?
  echo "    exit $status  (누적 $(( ($(date +%s) - started) / 60 ))분)"
  if [ $status -ne 0 ]; then
    tail -n 5 "$log"
  fi
done

echo
echo "=== 결과 (evaluation/success) ==="
python - "$OUT" <<'PY'
import csv, glob, os, sys

out = sys.argv[1]
print(f"{'run':<12} {'iters':>6} {'final':>7} {'best':>7} {'@iter':>6}")
for path in sorted(glob.glob(f"{out}/*/*/resip/*/eval.csv")):
    name = path[len(out) + 1:].split("/")[0]
    rows = [r for r in csv.DictReader(open(path)) if r.get("evaluation/success")]
    if not rows:
        continue
    points = [(float(r["step"]), float(r["evaluation/success"])) for r in rows]
    best = max(points, key=lambda p: p[1])
    print(f"{name:<12} {int(points[-1][0]):>6} {points[-1][1]:>7.4f} "
          f"{best[1]:>7.4f} {int(best[0]):>6}")
PY
