# SO101 SimDist visual WM / MPPI 구현 및 계약 검토 — 2026-10-07

> 이 문서의 초기 v1 value-censoring/API는 이후 구현한
> [finite-horizon v2 검증 기록](simdist-finite-horizon-validation-20261007.md)으로
> 대체됐다. 최신 remaining-time inputs와 failure value labels는 v2 문서를 따른다.

## 현재 완료 범위와 증거

Sol high 담당 단계에서 `src/so101/learning/simdist/`에 visual world model,
episode별 데이터 windowing, loss masking, constrained MPPI, checkpoint loader,
untrimmed transition recorder를 구현했다. `scripts/train_simdist.py`,
`scripts/eval_simdist.py`, `scripts/dryrun_simdist.py`를 추가했다. 기존 ResiP 및
image DP 구현과 실행 경로는 변경하지 않았다. G, 로봇 actuation, push/merge,
장기 GPU 학습은 수행하지 않았다.

사용 가능한 기존 환경은 `conda run -n so101`이다. Python 3.11, Torch
2.7.0+cu128, NumPy 1.26.4, IsaacLab/IsaacSim이 있다. 기본 Python에 Torch가
없다는 사실은 이 conda 환경의 사용을 막지 않는다. JAX/Flax는 해당 환경에
없고, 이번 WM/MPPI 실행에 필요하지 않다. 설치나 전역 upgrade는 하지 않았다.

CPU semantic regression 9개가 통과했다. 실제 next state/image window,
episode split, RGB와 sent action의 context 영향, 미래 action에 대한 prefix
causality, censored value 및 nonexpert action masking, gamma horizon 식,
모든 candidate tick의 joint/rate/trust constraints, sent history와 startup
timing, consumed-step warm start, untrimmed recorder를 검증한다.

합성 fixture에서 CPU optimizer 2회, checkpoint 저장/strict load, offline
MPPI 3회까지 실행했다. 최종 실행의 train losses는 10.1188, 15.4140이고
episode-disjoint validation loss는 64.0408이었다. 서로 다른 batch의 2-step
loss 수치는 convergence 검증이 아니다. Artifact는
`outputs/simdist_contract_smoke/train/worldmodel.pt`, `manifest.json`,
`metrics.json`, `outputs/simdist_contract_smoke/dryrun.json`이다. Offline
planner latency는 14.81 / 12.66 / 12.81 ms였다. 이는 16px 합성 RGB, H=2,
T=3, 8 candidates, 2 iterations에서의 CPU wiring smoke다. 기본 64px,
H=3, T=8, 64 candidates, 3 iterations의 배포 latency를 증명하지 않는다.
카메라 acquisition timestamp가 없으므로 dry-run 결과는 freshness를
`false`로 기록한다. 저장된 command는 모두 `actuated=false`다.

Root 세션에서는 실제 collector로 생성한 **simulator task data**의 mixed
expert/perturbed NPZ로 CPU optimizer 2회를 실행하고 strict checkpoint reload까지
확인했다. `/tmp/so101-simdist-final-smoke/train/manifest.json`에 seed=1,
train episodes=[1,2], validation=[0], windows=406/296이 기록되어 있다.
Train loss는 10.0267/9.8907, validation loss는 2.2113이다. Held-out failure
의 censored value labels는 masked이므로 이 validation loss가 실패의 value
calibration을 검증하지 않는다.

Isaac에서 동일 seed/layout, H startup hold 후 2 action ticks만 실행한
wm_policy / mppi_reward / mppi_mc 세 arm은 각각 success=0/1이었다.
Policy 평균 latency는 약 13.55ms, reward/MC planner는 약 17.6ms였다.
`/tmp/so101-simdist-final-smoke/eval.json`에 per-step latency와 config가 있다.
이 결과는 collector→packing→2-step training→checkpoint→fresh Isaac RGB→
constrained planner→simulation command의 wiring smoke다. 충분한 학습,
teacher/DP와 동일 budget 성능 비교, 실제 로봇 robustness는 미검증이다.

## Visual 및 normalization 계약

- 배포 입력: measured 6 joint radians + front/wrist uint8 HWC RGB.
  privileged 36-D teacher state를 WM 입력으로 사용하지 않는다.
- Joint order: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
  단위는 관측과 action 모두 radians이며 action은 absolute joint target이다.
- 두 camera에 별도 CNN을 사용한다. float `/255` 후 deterministic bilinear
  resize를 적용한다. CNN은 scratch training 대상이며 image DP의 encoder나
  pretrained weights를 자동 공유하지 않는다. 기본 64px resize는 작은
  구현 baseline 선택이며 특히 wrist의 작은 물체 정보 손실을 검토해야 한다.
- joint/action/reward/value mean/std는 training episode만으로 계산한다.
  std floor는 `1e-3`, stats는 checkpoint buffer에 저장한다. Validation
  episode의 관측이나 label을 stats fitting에 넣지 않는다.
- Native transformer building blocks, latent dynamics, reward/value/policy
  heads는 `/home/keti/workspace/research/so101-ros/worldmodel/worldmodel`의
  PushT torch port를 참고/이식했다. SO101은 proprio-only encoder 대신 RGB
  encoder가 필요하다. copied code의 provenance는 각 module docstring과
  이 문서에 남긴다.
- 미래의 실제 RGB/joint를 동일 encoder에 넣어 detached latent dynamics
  target으로 사용한다. Pixel reconstruction loss는 구현하지 않았다.
  현재 latent와 target latent의 temporal/type tag를 동일하게 적용한다.
  Native detached consistency loss만으로 encoder collapse나 장기 rollout
  정확도가 방지된다는 보장은 없다.
- Reward/value heads를 causal attention으로 바꾸었다. Native PushT의
  noncausal head를 그대로 쓰면 현재 transition 예측에 미래 action 정보가
  새어 들어갈 수 있다. 현재 model은 prefix perturbation regression을
  통과했다. Policy head는 future action input과 독립적이다.

## 시간축, reward, value 계약

현재 계획 시점이 t이고 history length H, prediction length T일 때:

| 입력 / 출력 | 포함 tick |
|---|---|
| measured observation history | o[t-H] ... o[t] |
| applied / sent action history | a[t-H] ... a[t-1] |
| candidate future actions | a[t] ... a[t+T-1] |
| 실제 successor latent target | z(o[t+1]) ... z(o[t+T]) |
| reward | r[t] ... r[t+T-1] |
| value label | V(o[t+1]) ... V(o[t+T]) |

`next_observations[t]`와 `next_*_images[t]`는 action a[t] 후 실제 관측이다.
episode 내부에서는 next row의 current observation과 정확히 일치해야 한다.
Dataset은 H+T개의 실제 contiguous transitions를 가진 window만 사용하며
시작/끝 padding 자체를 만들지 않는다. Terminal successor를 duplication으로
fabricate하지 않는다. Histories가 없는 startup에서는 H번의 실제 hold ticks를
기록한 후 planner를 호출한다.

Reward는 `dense_environment_reward`가 선언된 transition에만 supervise한다.
기존 sparse successful-demo reward는 `reward_valid=false`로 제외한다.
Policy loss는 실제 applied action의 expert flag에서 시작하는 연속 expert
prefix만 사용한다. Perturbed data만으로는 policy-head proposal을 학습할 수
없으므로 fresh dense expert collection도 필요하다:

```bash
# collector의 --diverse는 all-outcome/dense recording 계약을 켠다.
# std=0이면 실제 applied actions가 clean teacher expert labels다.
conda run -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0 \
  --num-envs 2 --num-episodes 4 --max-rounds 2 --out outputs/simdist/expert
conda run -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.15 \
  --num-envs 2 --num-episodes 4 --max-rounds 2 --out outputs/simdist/perturbed
conda run -n so101 python scripts/prepare_simdist_dataset.py \
  --input outputs/simdist/expert outputs/simdist/perturbed --gamma 0.99 \
  --out outputs/simdist/aligned.npz
```

초기 baseline은 simulator tick=1/30초, gamma=0.99/tick이다. MPPI objective:

`sum(k=0..T-1) gamma^k r[t+k] + value_weight * gamma^T V(o[t+T])`.

Native adjacent port의 `gamma^(T+1)`을 사용하지 않는다. MC return은 실제
terminal boundary 이후 zero continuation이다. Timeout/collector cutoff의
unknown continuation을 zero로 맞추지 않고 episode의 `value_valid=false`로
mask한다. Accepted success-hold/post-success tail까지 끝난 true task terminal은
유효한 successor value 0을 갖는다. Collector는 이 boundary rule을 명시한다.

**장기 학습 전 미결정 항목:** 실패 episode가 모두 timeout인 현재 수집에서는
실패의 value targets가 censored되고 성공 episode만 value-head supervision에
들어간다. 이로 인한 selection bias와 OOD optimism을 해결하려면
finite-horizon task objective + remaining-time conditioning 또는 명시적인
timeout bootstrap을 먼저 결정해야 한다. 현재 stats()는 valid MC label이
없는 training split을 에러로 막는다. 이를 실행 성공을 위해 무단으로 zero
value label로 바꾸지 않는다. Reward-only planning과 MC-value planning을
분리해 보고한다.

27 Hz real loop를 30 Hz sequence인 것처럼 취급하지 않는다. Dataset은
timestamp gaps와 model tick mismatch를 거부하고 controller도 gap에서 reset을
요구한다. 실제 27 Hz 사용 전에 일정 30 Hz resampling과 action hold alignment,
또는 dt-conditioned model이 필요하다. `0.99^(30/27)`은 동일한 초당 할인율로
변환한 후보 값이며 현재 구현은 이를 적용하거나 실제 데이터를 resample하지
않았다. Servo lag와 camera acquisition lag는 별도로 측정/모델링해야 한다.

## JAX / PyTorch 경계

기존 export `outputs/teachers/resip_sd042_20260924_210715_e2000`의 base+actor
networks, schedule, matching config/normalization, fixed-noise Flax references를
사용한다. Epoch source는 handoff의 gamma99 `params_2000.pkl`이다.
`scripts/export_resip_teacher.py`는 JAX/Flax CPU environment에서 한 번
실행하는 export 도구다. Collector, WM training, planner는 Torch만 사용한다.
Export는 critic을 포함하지 않는다. ResiP critic을 state-only V API로
호출하거나 그 label을 현재 MC value에 섞지 않는다. 새 teacher 학습을 하지
않았다. Root의 기존 teacher reference parity check는 별도로 기록한다.

## MPPI / control 계약

- 모든 noisy candidate를 model rollout **전에** raw joint limits, 이전 실제
  sent command에서의 per-tick rate limit, constrained native policy trajectory의
  ±0.25 rad trust region에 projection한다. Rate limit은 rad/s * tick_seconds다.
- Policy proposals + local mean proposals를 사용하며 exact native policy
  candidate를 항상 포함한다. DP proposals hybrid는 구현하지 않았다.
- Std는 기본 0.1rad로 시작해 최소 0.01rad와 초기 최대값 사이로 제한한다.
  물리 limits/trust region은 data-coverage나 OOD value 정확도를 증명하지 않는다.
- Mean도 constraints에 projection한다. 실제 실행 후
  `update(next_obs, actual_sent_action)`을 호출해야 한다. Post-limit action과
  servo 측정 joints는 다른 field다. 제안 action을 sent history로 넣지 않는다.
- Warm-start는 마지막 plan 이후 실제 소비한 ticks만큼 앞부분을 버리고
  남은 plan을 앞으로 당긴 후 policy tail로 채운다. Native forward-roll
  버그를 복제하지 않는다.
- RGB/history context는 각 replan에 한 번만 encode하고 candidate batch에서
  공유한다. CUDA latency는 시작/끝 synchronize 후 기록한다. Eval의
  per-controller latency는 RGB가 controller에 들어온 이후의 host stack,
  resize/CNN/head/search 비용을 포함한다. Camera acquisition, GPU→CPU
  observation conversion, simulator stepping, robot I/O 전체 비용은 별도이며
  이 측정만으로 real 30 Hz controller budget을 승인하지 않는다.
- `last_report`에는 source timestamp, tick budget, latency, policy/value mode,
  history count, trust radius가 있다. Offline dry-run은 camera timestamps가
  없으면 freshness가 검증되지 않았다고 기록한다. 현재 live asynchronous
  plan age watchdog나 hardware controller를 연결하지 않았다.
- Future terminal head는 없다. True success terminal 이후로 이어지는 imagined
  reward/value extrapolation과 state coverage는 학습 후 simulation에서 검토한다.

## 재현 명령과 API

```bash
conda run -n so101 python tests/test_simdist_model_planner.py
conda run --no-capture-output -n so101 python scripts/train_simdist.py \
  --data outputs/simdist/aligned.npz --out outputs/simdist/wm_smoke \
  --steps 2 --batch-size 2 --history 2 --horizon 3 --image-size 16
conda run --no-capture-output -n so101 python scripts/eval_simdist.py \
  --checkpoint outputs/simdist/wm_smoke/worldmodel.pt --headless \
  --num-envs 1 --num-rounds 1 --max-steps 2 --samples 8 --elites 3 --iterations 1 \
  --out outputs/simdist/wm_smoke/eval_sim.json
conda run --no-capture-output -n so101 python scripts/dryrun_simdist.py \
  --checkpoint outputs/simdist_contract_smoke/train/worldmodel.pt \
  --recording outputs/simdist_contract_smoke/synthetic.npz \
  --limits-json outputs/simdist_contract_smoke/limits.json \
  --out outputs/simdist_contract_smoke/dryrun.json \
  --max-plans 3 --samples 8 --elites 3 --iterations 2
```

`train_simdist.py`는 episode split IDs/config/stats/source와 training steps를
manifest에 기록하고 strict checkpoint를 저장한다. `load_worldmodel(path,
device)`는 `(model, checkpoint_dict)`를 반환한다. MC-valid training data와
expert supervision이 없으면 명시적으로 중단한다.

`MppiController(model, lower6, upper6, max_velocity_rad_s6, config)`의 순서는
`reset(observation, prior_sent)` → H actual hold ticks에서
`update(successor_observation, actual_sent)` → `ready` →
`plan(policy_only=False)`다. Observation은 `joints/front/wrist/timestamp_seconds`
dict이고 반환은 `(T,6)` radians numpy array다. `last_report`가 latency 정보를
제공한다. `policy_only=True`, value_weight 0/1로 세 arm을 선택한다.

Evaluator는 매 arm/round 동일 reset seed와 실제 cube pose equality를 검사한다.
Rendering DR을 켜면 같은 randomizer seed를 다시 사용한다. Simulator의
`soft_joint_pos_limits`와 calibrated joint order를 사용한다. Automatic reset을
끄고 모든 arm이 같은 H startup hold ticks를 가진다. Teacher/DP 비교는 기존
entrypoint에 같은 seed/layout/budget을 주어 별도로 실행해야 한다. 기존
teacher/DP script의 startup settle/budget을 동일하게 맞추지 않은 수치는 공정한
비교 결과로 보고하지 않는다.

`TransitionRecorder`는 operator-controlled future recording을 위한 API다.
실제 current+successor snapshot, actual sent action, observation/camera acquisition
monotonic clocks를 요구한다. 모든 pause/gap을 보존하며 `trim_idle`은 없다.
Real reward/value labels는 unlabelled로 선언한다. 기존 trimmed EpisodeRecorder
파일은 dynamics fixed-step 학습에 바로 사용할 수 없다. Dry-run은 recorded
applied action으로만 history를 업데이트하고 proposal은 파일로 저장한다.

## 남은 연구 및 실행 blocker

1. 실제 task data에서 clean expert와 perturbed failure/recovery가 충분히 있는
   episode-disjoint dataset을 수집해야 한다. Unsuccessful samples를 value
   supervision에서 어떻게 다룰지 먼저 결정해야 한다.
2. 기본 visual resolution과 H/T budget의 latent stability, next-state/action
   condition sensitivity, reward/value calibration, multi-step prediction을
   task data로 검증해야 한다. Synthetic tests는 research validation이 아니다.
3. Teacher/image DP/WM policy/reward MPPI/MC MPPI의 동일 seeds/layouts,
   rendering DR, startup settle, action/rate bounds, rollout budget 비교가 남았다.
4. 기본 budget의 CPU/GPU latency와 full camera/control path latency를 측정하고
   real 27 Hz timing, plan age와 frame freshness를 연결해야 한다.
5. 실제 operator가 기록/평가하기 전에는 real robustness 향상을 주장하지 않는다.
   Wrist perception 원인 규명은 이 구현을 시작하기 위한 조건이 아니다.
