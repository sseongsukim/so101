# so101

SO-101 robot assets and real-robot interface helpers for Isaac Lab experiments.

This repository intentionally does not copy the workshop task/environment. It
keeps only the parts that are useful when building a custom Isaac Lab scene:

- `so101.assets.SO101_CFG`: Isaac Lab `ArticulationCfg` using the right-mounted
  camera SO-101 USD with a black printed body.
- `so101.real.interface.LeRobotSO101Interface`: LeRobot bridge utilities for
  mapping real SO-101 joint values to Isaac Lab radians and back.
- `so101.real.control.SO101Control`: a small real-robot control wrapper with
  initial/home poses and optional Rerun logging.
- calibration helper scripts for checking and summarizing SO-101 calibration
  files.

## Install

Use the Isaac Lab virtual environment named `isaaclab`.

```bash
conda activate isaaclab
cd /home/seongsu/workspace/research/so101
pip install -e .
```

Install the real-robot dependencies in the same environment when you need the
physical robot interface.

```bash
pip install -e ".[real]"
```

## Isaac Lab Usage

```python
from so101.assets import SO101_CFG

robot_cfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

Use `SO101_CONTACT_GRASP_CFG` when contact sensors are needed for grasp logic.

### Tabletop scene viewer

The package includes a minimal scene with the robot base at `(0, 0, 0)` and a
fixed 50 cm square tabletop extending along +X. The tabletop has collision but
no rigid body, so it remains fixed without legs or gravity settings. Its top
surface is at `z=0.030081 m`, matching the actual bottom of the robot base mesh.
The robot articulation root itself remains at `(0, 0, 0)`.
The robot uses true black while the tabletop uses a slightly lighter, rough
charcoal black so their silhouettes remain distinguishable under scene lighting.

```bash
conda activate isaaclab
cd /home/seongsu/workspace/research/so101
pip install -e .
python scripts/view_tabletop_scene.py
```

Close the Isaac Sim window to stop the script. Standard `AppLauncher` options
are available; for example, use `--device cpu` when a CUDA physics device is
not desired.

### StackCube environment

Environment configuration is defined explicitly under `src/so101/configs/`.
Scripts create configurations through the public helper instead of reading
private values from `gym.spec()`:

```python
from so101.configs import make_env_cfg

env_cfg = make_env_cfg("so101-StackCube-v0", num_envs=1, device="cuda:0")
env = gym.make("so101-StackCube-v0", cfg=env_cfg)
```

`configs/base.py` contains shared simulation, control, viewer, observation, and
reward parameters. `configs/tasks.py` contains the task-specific overrides,
and `configs/registry.py` maps every Gym task ID to its configuration class.
These use Isaac Lab's dataclass-compatible `@configclass` so nested scene and
simulation configurations retain `copy`, `replace`, and validation behavior.

The registered Gymnasium tasks are `so101-StackCube-v0` (36-D simulator state)
and `so101-visual-StackCube-v0` (six encoder positions plus wrist and external
RGB observations). Both use the same right-mounted-camera robot USD and exact
Isaac Lab cuboid primitives: a movable 2.5 cm cube and a movable 4 cm target
cube. The visual task deliberately excludes simulator-only cube poses, EEF
pose, and joint velocity so its observation can also be produced on the real
robot. The state-only task does not create camera sensors or return images; the
camera is merely part of the shared robot geometry.

StackCube randomizes both cube poses on every episode reset. It samples the
small cube on a random side and always places the large cube on the opposite
side. Both centers use `x=0.20..0.30 m`; the left side is
`y=-0.15..-0.055 m` and the right side is `y=0.055..0.15 m`. Each cube receives
a random yaw while remaining flat on the tabletop.

```bash
python scripts/view_task.py --task so101-StackCube-v0
python scripts/view_task.py --task so101-visual-StackCube-v0
```

StackCube returns both cubes' absolute poses, their relative position, and the
SO-101 state: small-cube quaternion (4) and position (3), large-cube quaternion
(4) and position (3), large-minus-small position (3), end-effector position
(3) and quaternion (4), joint position (6), and joint velocity (6), for 36
dimensions.

For StackCube, the EEF position is the midpoint of two explicit distal grasp
points derived from the SO-101 colliders: one fixed to `/Robot/gripper` and one
fixed to `/Robot/jaw`. The EEF quaternion remains the `/Robot/gripper`
orientation.

StackCube uses the IsaacGym Franka cube-stack reward instead: distance `0.1`,
lift `1.5`, alignment `2.0`, and exclusive stack-success reward `16.0`. Stack
success requires cube-center XY error below 1 cm, height error below 0.5 cm,
and the end effector to be more than 2 cm from the small cube. Its distance
gain is doubled to 20 and its lift-clearance threshold is reduced to 2 cm to
match cubes half the size of the Franka example. Success sets
`terminated=True`; the 300-step (10-second) time limit sets
`truncated=True` when success has not occurred.

### Leader-arm task teleoperation

A calibrated physical SO-101 leader arm can directly command StackCube's six
absolute simulation joint targets:

```bash
lerobot-calibrate --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm_1

python scripts/teleop_task.py so101-StackCube-v0
```

The default control-rate cap is 30 Hz. Use `--rate 0` to disable wall-clock
pacing. `TELEOP_PORT` and `TELEOP_ID` can be used instead of the corresponding
command-line options.

While teleoperating, press `t` to mark the last transition successful and
terminal, save the trajectory, and pause collection without resetting the
environment. The simulated robot continues following the leader while paused,
so returning the leader to its initial pose is not recorded. Press `r` to reset
the environment and resume collection; any unsaved transitions are discarded.
Automatic success and time-limit resets are disabled during teleoperation.
Files are pickle dictionaries under `outputs/teleop`
by default (override with `--dataset-dir`) and contain NumPy arrays named
`observations`, `actions`, `rewards`, `terminals`, `successes`, and
`next_observations`. `terminals` includes both task termination and time
limits, while `successes` records task success only. Every recorded transition
prints its trajectory step, current simulated joint positions, and StackCube
reward.

Trajectory files use sequential names such as `trajectory_000000.pkl`. On
startup, `teleop_task.py` scans `--dataset-dir` and continues at the next
available index instead of overwriting existing data.

Stationary leader-arm steps are excluded: a transition is recorded whenever at
least one mapped leader joint value differs from the last stored action.

The environment action is a six-dimensional absolute SO-101 joint-position
target in radians, matching the Sim-to-Real SO-101 Workshop. The current
implementation supplies task geometry, physics, joint observations/actions,
default-pose reset behavior, and cube-stack rewards.

The workshop changes the robot color by editing the USD shader at
`Looks/material_a_3d_printed/Shader`. This package uses the same mechanism and
sets `SO101_CFG` to black by default. If you want reset-time color
randomization in your own environment:

```python
from isaaclab.managers import EventTerm
from so101.mdp.randomization import ROBOT_COLORS, randomize_robot_color, set_robot_color

reset_set_robot_visual_material = EventTerm(
    func=set_robot_color,
    mode="reset",
    params={"color": "black"},
)

reset_randomize_robot_visual_material = EventTerm(
    func=randomize_robot_color,
    mode="reset",
    params={"color_names": list(ROBOT_COLORS.keys())},
)
```

## Real Robot Usage

```bash
so101-control --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm_1
so101-manual-control --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm_1
```

Calibration helpers:

```bash
so101-calibration-stats
ROBOT_PORT=/dev/ttyACM0 ROBOT_ID=follower_arm_1 so101-check-calibration
```

## Source

The SO-101 USD, Isaac Lab robot config, and LeRobot/real-robot helper code were
adapted from `reference/Sim-to-Real-SO-101-Workshop`. Files copied from that
workshop keep their original Apache-2.0 SPDX headers.

## 모방학습 (state FBC / DBC)

`main.py`는 MjDex와 동일하게 `absl.flags`, `ml_collections.ConfigDict`,
`--agent=agents/fbc.py` 설정 파일, W&B 및 CSV 로깅을 사용합니다.
수집한 state 데이터로 offline 학습하고, `eval_interval`마다 Isaac Sim rollout으로 평가합니다.
Validation 데이터 분할은 사용하지 않습니다.

```bash
cd research/so101
python -m pip install -e '.[learning]'
python main.py --env_name=so101-StackCube-v0 --agent=agents/fbc.py \
  --offline_steps=2000000 --wandb_mode=offline
# FBC 설정 변경: --agent.batch_size=256 --agent.horizon_steps=24
```

평가를 실행하려면 Isaac Lab/Isaac Sim과 learning 의존성이 같은 Python 환경에서
사용 가능해야 합니다. JAX의 GPU 지원은 실행 환경에 맞게 설치하세요.
`--eval_interval=0`이면 Isaac Sim을 실행하지 않고 학습만 진행합니다.
`data/<env_name>.pkl`의 transition dictionary를 직접 읽습니다. 데이터 루트는
`--dataset_dir`로 지정합니다. 개별 `trajectory_*.pkl`은 미리 병합해 사용합니다.
입력 pickle은 로컬의 신뢰하는 수집 데이터여야 합니다.

모든 에피소드를 학습에 사용하며, 전체 학습 데이터로 정규화 통계를 계산합니다.
MjDex의 `FrozenDict` 기반 `Dataset`/`MultistepDataset`을 그대로 사용합니다.
정규화 후 `dataset_class[config["dataset_class"]].create(**train_dataset)`로 생성하고,
`pred_horizon`을 설정합니다. 각 sample은 현재 state와
미래 `horizon_steps`개의 action으로 구성하며, 에피소드 끝에서는 마지막 action을 반복합니다.
이미지를 포함하는 multimodal 학습은 이 runner에서 지원하지 않습니다.

### Observation 분석 및 정규화

`SO101TaskEnv._get_observations()`의 36차원 state 순서는 다음과 같습니다.
슬라이스는 Python의 끝 인덱스 제외 표기입니다.

| 슬라이스 | 값 | 정규화 |
| --- | --- | --- |
| `0:4` | 이동 큐브 quaternion | 그대로 유지 |
| `4:7` | 이동 큐브 월드 위치 | 평균/표준편차 |
| `7:11` | 고정 큐브 quaternion | 그대로 유지 |
| `11:14` | 고정 큐브 월드 위치 | 평균/표준편차 |
| `14:17` | 큐브 간 상대 위치 | 평균/표준편차 |
| `17:20` | gripper grasp 위치 | 평균/표준편차 |
| `20:24` | end-effector quaternion | 그대로 유지 |
| `24:30` | 관절 위치 (rad) | 평균/표준편차 |
| `30:36` | 관절 속도 (rad/s) | 평균/표준편차 |

현재 병합 데이터는 250개 에피소드, 31,100개 transition입니다. 위치는 대략
수십 cm 범위인데 관절 속도는 약 -86~80 rad/s까지 나타나므로 state 정규화를
기본으로 사용합니다. 표준편차가 `1e-3` 미만인 거의 고정된 성분은 scale=1로
두어 미세한 잡음을 증폭하지 않습니다. Quaternion은 단위 회전 표현이고 각
성분이 이미 [-1, 1] 범위이므로 그대로 둡니다. 이 전처리는 별도의 CLI 옵션 없이
`Normalizer.fit()`에 고정되어 있습니다.

새 사전학습에서는 정규화된 observation을 `[-5, 5]`로 clipping하고 이 설정을
`normalization.json`에 저장합니다. 평가와 DPPO fine-tuning은 저장된 observation
변환을 그대로 사용합니다. 기존 checkpoint가 clipping 없이 학습되었다면 그
설정도 유지합니다. DBC/DPPO의 DiffusionMLP와 critic은 배치 크기에 따른 수치
오차가 PPO clipping에 영향을 주지 않도록 matmul 정밀도를 `HIGHEST`로 지정합니다.

Visual 환경의 state는 관절 위치 6차원뿐이므로 이 경우 6개 모두 표준화합니다.
수집 스크립트는 `observation['state']`만 저장하므로 카메라 이미지는 pickle에
들어가지 않습니다. 6차원 state만으로는 물체 위치를 알 수 없어 임의의 물체 배치에
대응하는 정책을 학습하기 어렵습니다. 36차원 정책의 실제 로봇 배포에는 물체 및
end-effector pose 추정도 필요합니다.

현재 수집은 leader 관절 값이 바뀔 때만 transition을 저장하므로 action chunk의
한 칸은 고정 1/30초가 아니라 다음 저장 sample을 뜻합니다. 고정 주기 chunk 실행이
필요하면 수집 주기 및 timestamp 처리도 맞춰야 합니다. 또한 월드 위치를 사용하므로
여러 병렬 환경의 데이터는 환경 origin을 빼는 전처리가 별도로 필요합니다.

### Action 및 추론

Action은 6개 관절의 절대 목표 각도(rad)입니다. 관절별 학습 데이터 min/max를
사용해 `2 * (a - min) / (max - min) - 1`로 정규화합니다. 고정된 관절은 0으로
매핑합니다. 환경 설정의 `action_space=6`은 차원 선언이므로 물리적 관절 한계로
사용하지 않습니다.

각 run의 `normalization.json`에 정규화 통계를 저장합니다. 추론 시 같은 통계를
사용하고, 정책 출력을 rad로 역변환한 뒤 환경에 전달해야 합니다.

```python
import numpy as np
from utils.datasets import Normalizer

normalizer = Normalizer.load(f'{run_dir}/normalization.json')
# state: CPU numpy array, shape (36,) or (batch, 36)
observations = normalizer.normalize_observations(state)
action_chunk = agent.sample_actions(observations, rng=key)
action_chunk_rad = normalizer.unnormalize_actions(np.asarray(action_chunk))
# 첫 action만 실행한다면 action_chunk_rad[0, 0]을 사용 (단일 환경).
```

### 로그, 저장, 복원

결과는 `exp/<env_name>/<agent_name>/<exp_name>/`에 저장됩니다.
`flags.json`, `agent_config.json`, `normalization.json`,
`train.csv`, `eval.csv`, `params_<step>.pkl`을 기록합니다.
W&B group 기본값은 `{run_group}/{env_name}/{agent_name}`이며 `--wandb_group`으로
덮어쓸 수 있습니다. 주요 metric은 MjDex 형식의 `training/*`,
`time/epoch_time`, `time/total_time`, `train_step`입니다.

MjDex와 동일하게 `log_interval`마다 로그를 기록하고, `save_interval`마다
체크포인트를 저장합니다.

```bash
python main.py --agent=agents/fbc.py --offline_steps=2000000 \
  --restore_path=exp/so101-StackCube-v0/fbc/<previous_run> \
  --restore_epoch=1000000
```

복원 시 저장된 agent(optimizer 및 agent RNG 포함)와 정규화 통계를 불러옵니다.
FBC는 MjDex와 동일하게 학습 루프는 1부터 시작해 `offline_steps`번 업데이트하며,
로그와 체크포인트 번호도 새 run 기준입니다. 학습률 schedule은 현재 config로
생성하고 optimizer 상태는 복원하므로, 이어서 학습할 때는 원래 agent 설정과
총 step 수를 고려해야 합니다. 데이터 sampling RNG는 seed에서 시작합니다.



### DBC 사전학습과 DPPO fine-tuning

기본 설정은 reference DPPO의
`so101_stackcube_rm_stable_success_only_ta4_td20/2026-09-18_18-58-49_42`
실행을 기반으로 하되, StackCube 장기 학습 비교를 위해 actor LR을 `3e-6`
(최저 `3e-7`), sampling std 하한을 `0.015`로 조정했습니다. 이 값은 검증할
실험 설정이며 성능 하락 방지가 확인된 설정은 아닙니다. 기존 stable 값으로
비교하려면 `--agent.lr=1e-5 --agent.min_lr=1e-6
--agent.min_sampling_denoising_std=0.01`을 지정합니다.
Actor는 578,120개, critic은 141,313개의 파라미터를
가지며 Mish pre-activation residual MLP를 사용합니다. 기존 FBC 네트워크는 별개입니다.

```bash
python main.py --agent=agents/dbc.py --offline_steps=1000000 --seed=0 \
  --wandb_mode=offline

python online.py --agent=agents/dppo.py --seed=42 --num_envs=512 \
  --restore_path=exp/so101-StackCube-v0/dbc/<new_run> --restore_epoch=1000000 \
  --wandb_mode=offline

python inference.py --restore_path=exp/so101-StackCube-v0/dppo/<run> \
  --restore_epoch=150 --video_envs=0 --wandb_mode=offline
```

DBC는 FBC와 같은 `main.py`의 step 기반 학습 흐름을 사용합니다. 매 step마다
무작위 batch를 뽑고 같은 optimizer, checkpoint, logging 경로를 사용합니다.
기본값은 1,000,000 update, batch 256, AdamW, learning rate `3e-4`,
전체 step의 10% warmup 뒤 cosine decay입니다.

DPPO는 저장된 DBC actor 가중치로 시작합니다. 위 명령은 512개 환경을 사용하며
환경 수의 CLI 기본값은 1024입니다. Episode 300 step,
예측/실행 horizon 4, denoising 20 step 중 마지막 10 step fine-tuning입니다.
처음 10 iteration은 critic만 학습하고, iteration 11부터 actor를 학습합니다.
Actor LR은 iteration마다 갱신하며 첫 적용값 `3e-7`, 최대 `3e-6`,
warmup 10 actor iteration, cycle 1000입니다. Critic LR은 `1e-3` 고정입니다.
전체 학습은 기본 1000 iteration이고 rollout당 PPO epoch 1회, batch 7500으로
마지막 작은 minibatch까지 사용합니다.

Observation 변환은 DBC의 저장된 설정을 유지합니다. 새 DBC는 정규화 후 ±5로
clip하며 최종 action은 clip하지 않습니다. Sampling std 하한은 `0.015`,
log-prob std 하한은 `0.1`입니다.
학습 reward는 성공한 environment step 수입니다. 매 iteration 새 episode를
`seed + iteration`으로 시작하고, 10 iteration마다 기본 seed 1042/2042에
iteration에 따른 offset을 더해 평가합니다.
평가는 성공 후에도 300 step 전체를 집계합니다.
`params_initial.pkl`과 5 iteration 간격의 checkpoint를 저장합니다.

기존 GELU/LayerNorm DBC checkpoint는 새 모델과 호환되지 않습니다.
위 명령으로 새로 사전학습해야 합니다. 같은 seed라도 PyTorch와 JAX의 난수 및
연산 구현이 달라 개별 rollout이나 최종 성공률까지 같다는 의미는 아닙니다.

같은 가중치·입력·noise로 reference와 수치 비교하고 전체 학습 경로를 검증하려면:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_dppo_reference.py -v
```

이 테스트는 로컬 `reference/dppo`와 PyTorch가 필요하며 simulator 대신 작은 환경
대역을 사용합니다.

### ResiP fine-tuning

ResiP(`reference/robust-rearrangement`)는 denoising chain을 직접 학습하는 DPPO와
달리 사전학습 diffusion policy를 **얼린 채로** 두고, 그 위에 작은 Gaussian
residual policy만 PPO로 학습합니다. Base policy가 chunk(예측/실행 horizon 4개)를
제안하면 residual은 매 environment step마다 그 시점의 observation과 해당 step의
base action을 함께 보고 `action_scale * residual`만큼 보정합니다. 따라서 PPO의
transition은 chunk가 아니라 environment step이고, action 공간은 action_dim
크기의 Gaussian입니다.

```bash
python online.py --agent=agents/resip.py --seed=42 \
  --restore_path=exp/so101-StackCube-v0/dbc/<new_run> --restore_epoch=1000000 \
  --wandb_mode=offline
```

DPPO와 같은 `online.py` 흐름(rollout 수집 → GAE → PPO minibatch 갱신 → 평가/체크
포인트)을 사용하며, config와 checkpoint 형식도 동일합니다. Agent는 `modules_base`
(얼린 DBC actor), `modules_actor`(residual), `modules_critic`(residual observation에
대한 value) 세 모듈을 가집니다. Residual의 mean head는 0으로 초기화되므로 학습
시작 시점의 정책은 base policy와 정확히 같습니다. 기본값은 `action_scale=0.1`,
고정 `init_log_std=-4.0`, actor LR `3e-4`(cosine), critic LR `5e-3`(cosine, warmup
없이 0까지), clip 0.2, target KL 0.1, rollout당 PPO epoch 50회입니다.
`residual_l1`/`residual_l2`로 보정 크기에 penalty를 줄 수 있습니다.

네트워크와 최적화는 reference를 따릅니다. Actor는 ReLU MLP 256×2에 bias 없는
orthogonal(std 0) 출력, critic은 같은 형태의 ReLU MLP 256×2에 출력 layer를
orthogonal std `0.25` / bias `0.25`로 초기화합니다(DPPO가 쓰는 Mish residual MLP
`Value`와 다릅니다). Gradient는 actor와 critic을 **하나의 norm으로 묶어** 1.0으로
clip합니다. Reward는 reward 자체의 running standard deviation으로 나눈 뒤 ±5로
clip합니다 — DPPO가 쓰는 rolling discounted sum 기반 `RunningRewardScaler`와 다른
방식이라 `online.py`가 agent에 따라 갈라 씁니다.

Reference ResiP는 iteration 수를 직접 지정하지 않고 environment step 예산
(`total_timesteps=1e9`)만 주며, `num_iterations = total_timesteps // (num_envs *
num_env_steps)`로 파생시켜 cosine LR schedule 길이로 씁니다. 1024 environment
기준으로 one_leg(700 step)는 1,395, mug_rack(400 step)은 2,441 iteration입니다.
so101은 episode가 300 step이라 기본값인 1024 environment에서 iteration당 307,200
env step이고, 같은 1e9 예산이면 3,255 iteration입니다. `--online_iters`와
`--agent.lr_cycle_iters`는 같은 값으로 맞춰야 LR cycle이 학습 길이와 어긋나지
않습니다. Gradient step 수로 비교할 때는 reference가 rollout당 50 epoch(단일
minibatch, target KL로 조기 종료) 설정을 그대로 따릅니다. `lr_cycle_iters`
기본값은 0이고, 이때 cosine cycle 길이는 `--online_iters`를 그대로 따라가므로
학습 길이를 바꿔도 따로 맞출 값이 없습니다.

Rollout이 environment step 단위로 쌓이므로 메모리는 `num_envs * episode_steps *
(ob_dim + action_dim)`에 비례합니다. Minibatch 크기는 reference와 같이
`num_minibatches`(기본 1, 즉 rollout 전체가 minibatch 하나)로 지정하며
`num_envs * episode_steps`에서 자동으로 계산되므로 `--num_envs`를 바꿔도 따로
맞출 값이 없습니다. DPPO는 reference/dppo를 따라 절대 크기 `batch_size`를
그대로 씁니다.

`init_log_std`는 reference의 `-1.0` 대신 `-4.0`을 기본값으로 씁니다. so101의
action은 절대 관절 각도(radian)를 정규화한 값이라, `-1.0`(정규화 단위 std
`exp(-1) * 0.1 = 0.037`, 관절 데모 범위의 약 1.8%)을 매 control step마다 독립적으로
더하면 보정이 아니라 jitter가 됩니다. 32 environment 1 iteration 측정에서 `-1.0`은
rollout success 0.0 / eval 0.41, `-4.0`은 rollout 0.53 / eval 0.63이었습니다.
Rollout success가 0이면 `success_only` reward가 전부 0이라 학습 신호 자체가
없습니다. Reference와 같은 값으로 비교하려면 `--agent.init_log_std=-1.0`을 넘깁니다.

`inference.py`는 chunk 단위 API(`sample_actions`)를 쓰므로 residual을 chunk
시작 시점의 observation으로 한 번만 계산합니다. Step마다 다시 계산하는 학습·평가
경로와 다르므로, 정확한 성능은 `online.py`의 평가 결과를 기준으로 보십시오.

### FBC 환경 rollout 평가

학습 유틸리티는 저장소 루트의 `utils/`에 있습니다. `utils/env_utils.py`는
`teleop_task.py`처럼 AppLauncher를 먼저 실행한 뒤 task를 등록하고,
`make_env_cfg(..., num_envs=num_envs)`와 `gym.make(..., render_mode=None)`로 환경을
만듭니다. 수집 환경과 동일하게 카메라와 자동 success/timeout reset을 끕니다.
평가는 `--num_envs`개 환경에서 **에피소드 하나씩을 한 번의 배치 rollout으로**
동시에 굴립니다. 환경마다 성공 시점이 다르므로 종료된 환경은 mask로 집계를
동결하고, 전부 끝나거나 `max_episode_length`에 도달하면 rollout이 끝납니다.
길이는 환경의 `episode_length_s / step_dt`로 결정됩니다. Isaac Lab의 GPU
step 비용은 환경 수에 거의 무관해서(1개 26.5 ms vs 256개 27.4 ms), 환경을
늘려도 벽시계 시간은 사실상 그대로입니다. 관측 위치는 각 환경 원점 기준으로
보고되므로 `num_envs`를 바꿔도 정책이 보는 값의 분포는 동일합니다.

```bash
python main.py --agent=agents/fbc.py --eval_interval=250000 \
  --num_envs=50 --device=cuda:0 --headless
```

물리 시간 간격은 기존 환경 설정 그대로 `dt=1/120`, `decimation=4`입니다.
정책 action 하나마다 `env.step()`을 정확히 한 번 호출하므로 시뮬레이션에서는
30 Hz로 제어합니다. 예측 chunk의 앞 `agent.inference_steps`개 action을 한 개씩
실행한 후 현재 observation으로 다시 예측합니다. action을 여러 환경 step 동안
반복하거나 한 step에 여러 action을 실행하지 않습니다.

실제 시간도 teleop 코드처럼 매 제어 iteration의 추론·환경 실행 시간을 뺀
나머지만 sleep해서 환경의 `step_dt`에 맞춥니다. 현재 환경에서는 1/30초이며,
평가 길이와 제어 주기를 `main.py` flag로 별도 지정하지 않습니다. 첫 JIT 컴파일은 제어 시간 측정 전에
실행합니다. 추론이나 물리 계산이 1/30초보다 오래 걸리면 실제 속도는 낮아집니다.
`evaluation/control_time`은 sleep을 포함한 평균 실제 제어 시간을 기록합니다.

기존 pickle은 명령이 바뀐 시점만 저장하며 timestamp/정지 지속시간을 포함하지
않습니다. 따라서 수집 당시 생략된 정지 구간까지 정확히 재현할 수는 없습니다.
현재 평가는 연속된 저장 action 사이를 한 제어 tick으로 해석하고 수집 코드의
환경의 기본 30 Hz를 적용합니다.

평가 전 observation에 학습 통계를 적용하고, 예측 action은 rad로 역변환해서
환경에 전달합니다. W&B와 `eval.csv`에는 `evaluation/return`, `evaluation/length`,
`evaluation/success`(에피소드 성공률), `evaluation/control_time`을 기록합니다.

현재 `data/so101-StackCube-v0.pkl`의 에피소드 길이 통계는 250개 에피소드,
총 31,100 step, 평균 124.4 step, 중앙값 119 step, 최소 75 step, 최대 238 step입니다.
개별 trajectory 파일 250개의 통계도 동일합니다. 이는 저장된 transition 수이며,
정지 중 생략된 제어 tick은 포함하지 않습니다. 현재 StackCube의 환경 정의는
`src/so101/configs/tasks.py`의 `STACK_CUBE_MAX_EPISODE_STEPS = 300`입니다.

### 학습된 체크포인트 평가 (`inference.py`)

`inference.py`는 학습 없이 저장된 파라미터만 불러와 `utils/evaluation.py`의
`evaluate`를 그대로 실행하고, 결과를 W&B와 CSV로 남깁니다. MjDex의
`inference.py`와 같은 구조입니다.

```bash
python inference.py \
  --restore_path=exp/so101-StackCube-v0/fbc/fbc_sd042_20260908_224023 \
  --restore_epoch=2000000 --num_envs=50 --video_envs=4 --num_runs=3 --wandb_mode=online
```

`agent_config.json`에서 agent 설정을, `flags.json`에서 `env_name`을 읽으므로
`--agent` 설정 파일을 다시 지정하지 않습니다(`--env_name`으로 덮어쓸 수 있습니다).
정규화 통계는 데이터셋이 아니라 체크포인트의 `normalization.json`에서 불러오므로
학습 때와 동일한 입출력 scaling이 유지되고, 학습 데이터 pickle은 필요하지 않습니다.
agent 재구성에는 shape만 필요해서 example transition은 정규화 통계의 차원과
`horizon_steps`로 만듭니다.

`--num_runs`는 `--num_envs`개 에피소드 평가를 초기 상태와 sampling noise가 다른
상태로 여러 번 반복합니다(각 run seed는 `--seed`에서 재현 가능하게 파생).
run별 metric은 `evaluation/*`로 step=run index에 기록하고, 2회 이상이면
`evaluation_mean/*`, `evaluation_std/*`를 추가로 기록합니다.
`--video_envs > 0`이면 rollout 영상을 `evaluation/video`로 올립니다.
결과는 `exp/eval/<env_name>/<agent_name>/<exp_name>/`에 `flags.json`, `eval.csv`,
(2회 이상일 때) `eval_summary.json`으로 저장합니다. 집계값은 `CsvLogger`의 header가
첫 run 행에서 고정되기 때문에 CSV가 아니라 JSON으로 남깁니다.

## 모방학습 (visual ACT)

so101-ros의 standalone PyTorch ACT(`act/`)를 `src/so101/learning/act/`로 이식했다.
`config`/`model`/`policy`/`checkpoint`는 import 경로 외 동일하고, LeRobotDataset 대신
`teleop_task.py --visual`이 저장한 `trajectory_*.pkl`을 변환 없이 바로 읽는다
(`data.py`).

```bash
# 1) 수집: 30 Hz 매 tick마다 state(6) + action(6) + front/wrist RGB(240x320) 기록
python scripts/teleop_task.py --visual --dataset-dir outputs/act_sim

# 2) 학습 (Isaac Sim 불필요, torch만 있으면 됨)
python scripts/train_act.py --data-dir outputs/act_sim --out outputs/act_train/run0 \
    --steps 100000 --batch-size 8 --save-freq 10000 --min-length 30
```

샘플 t는 `observations[t]`, `front_images[t]`, `wrist_images[t]`와
`actions[t:t+chunk_size]`(에피소드 끝을 넘으면 마지막 action 반복 + `action_is_pad`)로
구성된다. key 대응: `observations`→`observation.state`, `actions`→`action`,
`front_images`→`observation.images.front`, `wrist_images`→`observation.images.wrist`.

출력 디렉토리:

- `act_so101.pt`, `step_*.pt`: `{"model", "config", "step"}` (so101-ros와 같은 포맷,
  `ACTPolicy.from_checkpoint`로 로드)
- `stats.json`: state/action/image MEAN_STD. 추론에서도 이 파일로 정규화하고, 모델
  출력 action은 `action` 통계로 역정규화한다. std가 `1e-3` 미만인 state/action 차원은
  `utils/datasets.Normalizer`와 같은 규칙으로 1.0(무스케일)로 둔다.
- `train_info.json`: 학습에 쓴 에피소드 목록, 이미지 크기, 카메라 key 대응.

추론 시 카메라 프레임(시뮬 640x480, 실물도 동일 해상도)은 반드시
`so101.learning.act.data.to_uint8_rgb`로 `train_info.json`의 크기로 줄인 뒤
`image_to_tensor`로 [0, 1] CHW로 바꾼다. `teleop_task.py`도 같은 함수로 저장하므로
학습/추론 전처리가 한 곳에서 관리된다.

시뮬 평가 (`so101-visual-StackCube-v0`, 성공은 env별로 처음 달성한 step에서 latch):

```bash
python scripts/eval_act_sim.py --checkpoint outputs/act_train/run0/act_so101.pt --headless \
    --num-envs 4 --num-rounds 5 --n-action-steps 20 --video-dir outputs/act_eval/videos
# temporal ensembling: --temporal-ensemble-coeff 0.01
```

결과는 `<checkpoint dir>/eval_sim.json`. 추론 경로(`so101.learning.act.inference.ACTRunner`)는
리사이즈 → 정규화 → `select_action` → 역정규화를 한 곳에서 하므로 실물 루프에서도 그대로 쓴다
(실물 프레임은 BGR → RGB, 관절값은 `real_to_sim_obs_processor` 변환 후 입력).

카메라/state 정렬 확인: `python -u scripts/check_camera_lag.py --headless`.
두 자세를 매 step 순간이동시키며 이미지가 state보다 몇 step 늦는지 측정하고, `env.reset()`이
돌려주는 프레임이 reset 이전 장면(stale)인지도 확인한다. 2026-09-27 측정: step 지연 0,
reset 프레임은 stale (`num_rerenders_on_reset=0`) — teleop은 reset 직후 tick을 기록하지 않고
eval은 1 step settle 후 시작하므로 영향 없음.

테스트: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q`
