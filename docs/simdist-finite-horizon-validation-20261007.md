# SO101 SimDist v2 finite-horizon 계약과 검증

이 문서는 `simdist-sol-contract-20261007.md`의 초기 v1 censoring/성공 episode만
value를 학습하던 계약을 대체한다. 사용자 요청에 따라 finite-horizon objective와
remaining-time conditioning을 선택하고 구현했다. 실행 가능성과 연구 성능은
별도로 보고한다. 실제 로봇 actuation은 수행하지 않았다.

## 선택한 value 정의

Episode tick 0은 reset 후 camera settling hold를 끝낸 다음 첫 기록 action이다.
Declared task horizon을 N=300 simulator control ticks, gamma=0.99/tick으로 둔다.
Motion-substeps를 변경하면 collector가 선언하는 N이 달라질 수 있으며 서로 다른
N의 데이터를 하나의 고정-N model에 자동 혼합하지 않는다.

현재 o[t]에 들어가는 remaining clock은 `N-t`, 실제 successor o[t+1]의 clock은
`N-t-1`이다. **고정 task horizon의 마지막 successor 이후 continuation은 0**이다.
Stable success의 declared true terminal 이후에도 continuation은 0이다. 이 정의는
무한 시간 discounted value와 다르며 일반적인 환경 truncation에 zero bootstrap을
붙이는 관행을 정당화하지 않는다.

Recorded episode objective가 끝나는 tick을 tau라 하면 successor label은:

`V_MC(o[t+1], N-t-1) = sum(k=t+1 .. tau-1) gamma^(k-t-1) r[k]`.

Reward r[t]는 a[t]를 적용한 transition reward다. 마지막 transition의 successor
label은 0이다. MPPI는 T개의 candidate actions에 대해
`sum(k=0..T-1) gamma^k r[t+k] + gamma^T V(o[t+T], N-t-T)`를 사용한다.
Remaining objective ticks보다 긴 imagined plan의 reward suffix와 horizon value는
정확히 0으로 처리한다. `last_report.valid_plan_ticks`가 실제 objective 내
사용 가능한 command 수다. Horizon이 끝난 observation에서는 새 plan을 거부한다.

## 종료, timeout, 짧은 기록의 edge cases

| 기록의 마지막 successor | Simulator flags | `objective_terminal` | `value_valid` |
|---|---|---|---|
| Stable-success task terminal, t+1<N | terminated=true | true | dense reward면 전체 episode true |
| Declared N 도달한 실패/timeout | truncated=true일 수 있음 | true | dense reward면 전체 episode true |
| Collector cutoff, t+1<N | truncated=true 또는 boundary 미표시 | false | 전체 episode false |
| Post-success tail이 N을 초과 | 실제 flags 그대로 보존 | N successor만 true | NPZ에는 objective 내 첫 N ticks만 사용 |
| Sparse imitation reward | flags 보존 | 시간/terminal에 따라 결정 | false; dense reward loss에서도 제외 |
| Horizon metadata 없음 | 해석 불가능 | 만들지 않음 | converter가 거부 |

Full-horizon failure의 **실제 reward prefix를 전부** value supervision에 넣으므로
v1의 selective-success label 문제를 제거했다. Early collector cutoff의 unknown
continuation은 여전히 masked다. Cutoff가 발생한 데이터의 비율과 이유는 계속
보고해야 한다. Success/failure 수집 빈도 차이와 teacher 행동에 따른 value
distribution bias는 label 계약을 고치는 것으로 사라지지 않는다.

기존 파일은 실제 task horizon을 아는 경우에만
`prepare_simdist_dataset.py --task-horizon-ticks 300` fallback으로 변환한다.
Episode 길이만으로 task horizon을 추측하지 않는다. Environment
terminated/truncated를 finite-objective terminal로 덮어쓰지 않으며 별도
`objective_terminal` 필드로 표현한다. Converter가 완성한 NPZ를 atomic rename으로
저장하므로 downstream reader가 아직 압축 중인 파일을 읽지 않도록 했다.

## 코드 계약 변경

- NPZ: `remaining_ticks`, `next_remaining_ticks`, per-transition
  `task_horizon_ticks`, `objective_terminal`, scalar
  `value_objective=finite_horizon_dense_mc_v2`를 추가했다.
- Dataset: current/successor clocks가 정확히 1 tick 감소하는지, task horizon과
  objective flags가 일치하는지, valid MC labels가 reward로 계산한 successor
  return과 일치하는지 검사한다. Padding 및 episode 경계 crossing은 없다.
- RGB source preprocessing을 checkpoint의 `stored_image_shape`로 명시했다.
  기본 240x320이며 simulator/live RGB에도 collector와 같은 `to_uint8_rgb`를
  먼저 적용한 다음 WM CNN resize를 한다. 실제 기존 64px checkpoint는 default
  240x320 source shape로 strict load 가능하며 network parameter 구조는 바꾸지
  않았다. Encoder는 잘못된 source resolution을 거부한다.
- Encoder: deployable RGB/joints/action history에 episode clock fraction
  remaining/N을 추가했다. Camera/joint가 같은 observation이어도 time context가
  다르면 다른 latent를 생성한다. Privileged cube state를 추가하지 않았다.
- Dynamics: 각 candidate successor의 remaining fraction을 명시적으로 embed한다.
  Reward/value head는 causal이며 prefix 예측에 미래 action이 새지 않는다.
  Finite horizon 이후 raw reward/value가 0이 되도록 알려진 boundary를 적용한다.
- Loss: future RGB/joints **및 successor remaining time**으로 latent target을 만들고
  early-censored value와 sparse reward를 mask한다.
- Checkpoint schema는 `so101-visual-simdist-v2`다. Time-conditioning이 없는 v1
  checkpoint를 조용히 로드하지 않으며 재학습/명시적 migration이 필요하다.
- Controller observation API에는 `remaining_ticks`가 필요하다. 실제 sent action,
  timestamp와 함께 1 tick씩 clock이 감소해야 한다. Real recording의 clock과
  30 Hz alignment를 operator/controller가 명시해야 한다. `TransitionRecorder`
  constructor도 explicit `task_horizon_ticks`를 요구하고 current/successor의
  remaining clocks를 보존한다. Gaps는 지우지 않으며 fixed-step dynamics용
  alignment를 자동으로 추측하지 않는다.

## 수행한 검증과 실제 결과

Converter semantic regression 8개, RGB/model/planner regression 14개가 통과했다.
Analytic reward `[1,2,4]`, gamma=0.5, N=3에서 successor value `[4,4,0]`를
확인했다. 같은 recording을 N=300의 early cutoff로 선언하면 value labels는
invalid가 된다. N을 넘어 기록한 reward=1000이 horizon 내 label에 들어가지
않는지 검증했다. True success terminal, unknown-horizon rejection, clocks,
held-out normalization independence, causal prefix, masks, constraints, sent history,
warm start, untrimmed recorder 검증도 포함한다.

기존 collector의 실제 simulator 데이터 3 episodes / 714 transitions에서 v2
converter→CPU optimizer 2회→strict checkpoint reload→held-out calibration을
실행했다. 114-tick 성공 episode뿐 아니라 300-tick 실패 두 episode도 value-valid다.
이 예산은 learning sanity용이며 policy 성능 검증이 아니다.

`/tmp/so101-simdist-finite-horizon-smoke/train/metrics.json`의 train losses는
9.4071/6.3530, validation loss는 3.6217이다. Held-out 300-tick failure의
value MAE/RMSE는 raw reward units로 125.81/148.17이었다. Target mean/std는
0.939/0.290, prediction mean/std는 117.45/91.43이다. Finite-horizon zero target의
prediction은 정확히 0이었지만 horizon 이전 prediction은 크게 잘못됐다.
이는 label censoring을 고쳤다는 사실과 learned calibration의 품질을 구분하는
실제 증거다. 이 2-step model을 성공적인 planner로 보고하지 않는다.

Root 세션의 64px/2000-step checkpoint
`/tmp/so101-simdist-study/train_v2_64/worldmodel.pt`에서도 episode-disjoint audit는
data/time/mask 계약을 통과했다. Held-out value MAE/RMSE는 **58.99/107.68**,
target mean/std는 55.61/128.13이다. Known horizon-zero windows 3개는 prediction이
정확히 0이었다. Horizon-zero 강제 조건의 통과는 나머지 value calibration을
검증하지 않는다. 현재 MC-value arm은 **uncalibrated exploratory baseline**으로
표기해야 한다. Root에서 수행하는 paired sim 평가의 결과는 progress 문서와
machine-readable evaluation JSON에 기록한다.

추가 held-out dynamics audit의 horizon 1..8 latent MSE는
`[0.1893, 0.1918, 0.1921, 0.1986, 0.2036, 0.2084, 0.2166, 0.2259]`였다.
동일 current latent를 그대로 유지하는 persistence baseline은
`[0.00661, 0.01618, 0.02798, 0.04135, 0.05524, 0.06969, 0.08333, 0.09677]`이다.
현재 learned dynamics는 **모든 horizon에서 persistence보다 나쁘다**.
Persistence baseline은 current time tag도 유지하므로 완전한 time-aware
baseline은 아니며, 그럼에도 learned model이 이를 이기지 못했다.
Training loss가 내려간 것을 dynamics learning 성공으로 해석하지 않는다.
**현재 checkpoint는 MPPI에 적합하다는 연구 검증을 통과하지 못했다.**

`/tmp/so101-simdist-study/audit_v2_64_dynamics.json`에 이 결과와 sensitivity
진단이 있다. Global latent feature std 평균은 0.4635다. 첫 validation batch의
front/wrist blackout latent RMS 변화는 0.5254/0.8094, future action +0.05rad의
latent RMS 변화는 0.00931이었다. Visual/action branch가 완전히 상수인 현상은
관찰하지 않았지만, 이 수치는 cube state 표현이나 정확한 action dynamics를
증명하지 않는다. 단순 feature diversity를 latent-collapse 완전 검증으로
보고하지 않는다.

## 현재 checkpoint의 실패 gate와 다음 검증 protocol

현재 단계의 명확한 rejection gate는 (1) held-out dynamics가 persistence를
이기지 못하거나 (2) 미래 action suffix가 이전 tick 예측을 바꾸거나 (3) known
horizon zero가 틀리거나 (4) 후보 action constraints가 깨지는 경우다. 현재
checkpoint는 (1)에 걸렸다. (2)~(4)의 code/contract 검증 통과가 (1)을 상쇄하지
않는다. MC arm은 해석 가능한 연구 성능을 가진 최종 planner로 사용하지 않는다.

다음 학습을 시작하기 전에, 현 paired exploratory run의 결과를 동결하고 같은
episode-disjoint split에서 다음 protocol을 고정한다. Random overlapping-window
split을 사용하지 않는다. Encoder와 normalizer가 같은 상태에서 learned dynamics
및 current-latent/time-aware persistence의 **각 horizon MSE**를 비교하고 episode를
cluster로 bootstrap한 95% confidence interval을 보고한다. 다음 acceptance gate는
모든 계획 horizon의 learned/persistence MSE ratio upper CI가 1 아래인 것이다.
이 gate는 구현 baseline의 연구 검증 기준이며 native SimDist의 보편적인 이론
threshold를 주장하지 않는다. 현재 4개 validation episodes는 작은 pilot이다.

Reward/value는 train-only constant baseline과 raw MAE/RMSE, failed/successful
held-out episode별 calibration을 함께 비교한다. Known-boundary zero test에 성능
weight를 주지 않는다. Value-driven proposal ranking을 simulator에서 실제
counterfactual rollout returns와 비교하여, 예측 오차가 후보 간 실제 advantage보다
큰 상태에서는 MC-value influence를 검증됐다고 보지 않는다. Code가 실행된다는
이유로 value_weight=1을 승인하지 않는다. 처음엔 reward-only arm을 따로 비교한다.

Current representation이 완전히 collapse했는지, decoder가 current state를 복사하지
못하는지, target encoder가 학습 중 움직이며 consistency target을 어렵게 만드는지
구분한다. 다음 변경 후보는 frozen/EMA target encoder 진단, zero-initialized residual
latent dynamics와 time-aware persistence control, short-to-long horizon curriculum,
expert/perturbed state coverage 보강이다. 각각 separately named experiment로 만들고
현재 checkpoint와 architecture를 섞지 않는다. **현재 paired run 중에는 모델을
추가 학습하거나 checkpoint를 교체하지 않는다.**

## Paired five-arm evaluator

`scripts/eval_simdist.py`에 ResiP teacher, existing image DP, WM policy,
reward-only MPPI, MC-value MPPI를 한 simulation process에서 실행하는 mode를
추가했다. 모든 arm은 같은 visual task, seed/cube pose, H actual hold ticks,
declared joint limits/rate bounds, N objective ticks를 사용한다. **N=300에는 H
startup hold ticks도 포함된다.** Settling용 첫 fresh-render hold 1 tick은 objective
밖이고 모든 arm에서 동일하다. H hold ticks의 reward는 objective에 포함한다.

Torch global RNG도 seed/arm마다 명시적으로 reset한다. Teacher/MPPI는 별도
seeded generator를 사용한다. Cube reset pose와 post-settle measured joint equality를
assert한다. DR mode에서는 seed만 비교하는 데 그치지 않고 실제 USD camera poses,
light intensity/color, cube/table/backdrop material parameters를 snapshot하여
arm 간 equality를 검사하고 JSON에 저장한다.

초기 paired run의 WM inputs는 raw480x640를 바로 CNN64로 줄여 collector의
480→240 uint8→64 frontend와 달랐다. 해당 WM 결과는 preprocessing mismatch가
있는 exploratory 결과로 남기고 성능 결론에 사용하지 않는다. Fix 후 same seeds/DR
및 budget의 WM 세 arm을 새 process에서 다시 실행한다. ResiP/DP baseline은 각
기존 frontend가 맞았으므로 기존 paired baseline artifact를 사용할 수 있다.
Raw uint8와 simulator normalized float RGB를 동일 helper로 처리했을 때 recorded
frames/encoded features가 일치하는 regression을 추가하여 통과했다.

Stable success terminal은 first success + 40 post-success actions 뒤에 최근 15 ticks
연속 success일 때다. Instantaneous success는 별도 diagnostic field다. Full-horizon
failure와 early stable task terminal, dense discounted reward, per-step latency,
action/rate bounds가 machine-readable JSON에 기록된다. 기존 DP/teacher만의 평가
결과와 이 constrained/stable-hold 비교의 수치를 섞지 않는다.

```bash
conda run --no-capture-output -n so101 python scripts/eval_simdist.py \
  --checkpoint /tmp/so101-simdist-study/train_v2_64/worldmodel.pt \
  --teacher outputs/teachers/resip_sd042_20260924_210715_e2000 \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --methods resip image_dp wm_policy mppi_reward mppi_mc \
  --num-envs 1 --num-rounds 5 --max-steps 300 \
  --samples 8 --elites 3 --iterations 1 --headless \
  --out /tmp/so101-simdist-study/paired_v2.json
conda run --no-capture-output -n so101 python scripts/audit_simdist.py \
  --checkpoint /tmp/so101-simdist-study/train_v2_64/worldmodel.pt \
  --data /tmp/so101-simdist-study/aligned.npz \
  --out /tmp/so101-simdist-study/audit_v2_64.json
```

Audit의 `--data`는 checkpoint manifest의 source와 정확히 일치해야 한다. 실제
root packing 경로가 다르면 manifest source를 사용한다. Calibration target counts는
overlapping windows를 포함한다는 사실을 JSON에 명시한다. Episode split은 공유하지
않으며 normalization은 training episodes만 사용한다.

`--help`의 exit 2는 IsaacLab `add_app_launcher_args`가 help action을 잠시 지우고
`parse_known_args`를 호출하는 동안 required checkpoint/output를 먼저 검증하던
문제였다. Preliminary parse에만 두 required flags를 유예한 후 복원하여 완성된
help는 exit 0, 실제 실행의 required args 검증은 유지한다. 해당 경로를 실행 확인했다.

## 나머지 audit 상태

| 항목 | 상태와 증거 |
|---|---|
| JAX/Torch teacher boundary | 고정 gamma99 export만 사용; root fixed-noise parity 8/8 통과 |
| 실제 current/action/successor 정렬 | contiguous clocks 및 successor equality 검사 통과 |
| RGB/joint inputs, normalization | Uint8 RGB와 radians, training-only stats, input sensitivity tests 통과 |
| Reward/value semantics | Dense mask, analytic MC, full-horizon failure labels, exact zero boundary 통과 |
| Prefix causality | Dynamics/reward/value 미래 suffix action perturbation regression 통과 |
| MPPI joint/rate/trust constraints | 모든 candidate tick 및 returned mean projection regression 통과 |
| Sent history/warm start | 실제 command history와 소비 tick shift regression 통과 |
| Learned reward/value calibration | 수치 audit 실행, 품질은 부족; MC arm uncalibrated |
| Default planner latency/full control path | Sim per-step 측정 entrypoint 준비; camera I/O/servo 포함 real path 미검증 |
| Real 27 Hz vs trained 30 Hz | Gaps/tick mismatch 거부; 실제 resampling 또는 dt-conditioned model 아직 없음 |
| Multi-step dynamics accuracy/latent collapse | Actual held-out latent forecast가 전 horizon에서 persistence보다 나쁨: 현재 checkpoint fail; visual/action sensitivity만 진단 확인 |
| Success-hold hidden state | 관측 clock은 explicit; success hold counter의 완전한 Markov representation은 보장하지 않음 |
| Five-arm controlled comparison | 동일 simulation 계약 entrypoint 구현; root paired run 결과를 별도 보고 |
| Real robustness | 실제 로봇 비교 미수행; 개선 주장 불가 |

Wrist perception 원인 규명은 구현이나 simulator 검증의 착수 조건으로 사용하지
않았다. G는 후속 과제로 유지한다.

## 독립 holdout episode-bootstrap 결과 (2026-10-07; 최종 review 2026-10-08)

기존 `train_v2_64/worldmodel.pt`를 변경 없이 고정하고, 학습 seed
100/101/200/201/300과 겹치지 않는 400/401, 500/501, 600/601, 700/701 seed의
32 episode를 새로 수집했다. 그룹은 expert 8개, perturbation std 0.02/0.04/0.08
각각 8개다.
수집된 데이터는 6,736 transition, 6,416 full forecast windows다. 300-tick 목표
바깥(예: tick 337)에 발생한 성공은 converter가 objective 밖 transition과 함께
제외했으므로 성공으로 집계하지 않았다. 최종 holdout 종료 유형은 success 12,
finite-horizon 16, early truncation/censored 4다. expert 그룹은 7/8이 300 tick
이내 안정 성공했고, perturbation 성공 수는 std .02에서 4/8, .04에서 1/8,
.08에서 0/8이다. 자연 발생 비율을 유지했다.

각 episode 내부에서 overlapping window의 latent MSE를 먼저 평균하고, 그 episode
평균들을 동일한 bootstrap episode index로 paired resample했다. baseline은 측정된
현재 RGB/joint content를 유지하고 future `remaining_ticks` embedding만 바꾼다.
10,000회 percentile bootstrap의 추정량은 episode mean learned MSE / episode mean
persistence MSE이며, 사전 고정된 통과 기준은 8개 horizon 모두에서 95% CI 상한이
1 미만인 것이다. 결과는 모든 horizon에서 명확한 실패였다.

| Horizon | Error ratio | Paired episode-bootstrap 95% CI |
|---:|---:|---:|
| 1 | 28.20 | [19.08, 41.64] |
| 2 | 13.11 | [9.25, 18.83] |
| 3 | 8.96 | [6.74, 12.23] |
| 4 | 6.67 | [5.11, 8.96] |
| 5 | 5.01 | [3.88, 6.68] |
| 6 | 3.98 | [3.08, 5.30] |
| 7 | 3.49 | [2.75, 4.58] |
| 8 | 3.13 | [2.48, 4.08] |

병렬 환경 네 개가 reset round seed를 공유해 생길 수 있는 상관에 대한 민감도
분석도 수행했다. 각 round 안에서 episode macro error를 평균한 뒤 8개 round seed를
paired bootstrap한 CI는 다음과 같고, 모든 하한이 1보다 높다.

| Horizon | Round-seed cluster 95% CI |
|---:|---:|
| 1 | [20.11, 39.78] |
| 2 | [9.70, 17.84] |
| 3 | [6.99, 11.75] |
| 4 | [5.25, 8.56] |
| 5 | [3.97, 6.41] |
| 6 | [3.15, 5.09] |
| 7 | [2.80, 4.39] |
| 8 | [2.51, 3.91] |

Expert/perturbed strata와 success/horizon/truncated strata도 각각 전 horizon에서
실패했다. Noise std 0/.02/.04/.08 각 8 episode 그룹도 모두 실패했다. 이
frozen checkpoint는 learned latent multi-step dynamics가 이 time-aware
persistence baseline을 이긴다는 근거가 없으며, MPPI dynamics scoring에
적합하다는 acceptance gate를 통과하지 못했다. 이 audit는 새로운 학습 실험이나
real robustness 평가가 아니다. 신규 장기 학습 전에는 Sol high가 audit 계약과
failure analysis를 검토하고, 별도 named checkpoint에서 수정 가설을 검증해야 한다.

재현 명령:

```bash
conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p scripts/generate_teacher_demos.py --headless --diverse --num-envs 4 --num-episodes 8 --max-rounds 3 --seed 400 --out /tmp/so101-simdist-holdout/expert
conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.02 --num-envs 4 --num-episodes 8 --max-rounds 3 --seed 500 --out /tmp/so101-simdist-holdout/perturbed_002
conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.04 --num-envs 4 --num-episodes 8 --max-rounds 3 --seed 600 --out /tmp/so101-simdist-holdout/perturbed_004
conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.08 --num-envs 4 --num-episodes 8 --max-rounds 3 --seed 700 --out /tmp/so101-simdist-holdout/perturbed_008
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py --input /tmp/so101-simdist-holdout/expert /tmp/so101-simdist-holdout/perturbed_002 /tmp/so101-simdist-holdout/perturbed_004 /tmp/so101-simdist-holdout/perturbed_008 --gamma 0.99 --out /tmp/so101-simdist-holdout/aligned.npz
conda run --no-capture-output -n so101 python scripts/audit_simdist_dynamics_bootstrap.py --checkpoint /tmp/so101-simdist-study/train_v2_64/worldmodel.pt --data /tmp/so101-simdist-holdout/aligned.npz --training-episode-dirs /tmp/so101-simdist-study/expert /tmp/so101-simdist-study/perturbed_004 /tmp/so101-simdist-study/perturbed_012 --out /tmp/so101-simdist-holdout/audit_bootstrap.json --batch-size 4 --device cuda --bootstrap-repetitions 10000 --bootstrap-seed 20261007
```

검증 한계: episode는 `(round seed, env index)`로 식별했고, train/holdout seed set은
서로소임을 확인했다. 전체 gate는 최소 16 episode, subgroup gate는 최소 4 episode로
계산한다. training episode directories는 checkpoint manifest의 실제 source NPZ와
episode fingerprint 집합이 일치해야 하며, holdout fingerprint도 packed transition에서
재계산해 metadata와 대조한다. 이 provenance 검사가 통과해야 결과에
`seed_disjoint=true`가 기록된다. Sol high는 checkpoint FAIL 결과와 episode-macro
계산을 승인했고, fail-closed provenance·cluster sensitivity·invariant tests 보강
후 audit 구현과 frozen checkpoint FAIL 결론을 최종 승인했다. 코드 review blocker는
없다. 성능 gate는 여전히 실패했으며, 추가 학습 전 별도 failure analysis와 새
실험 가설 검증이 필요하다.
