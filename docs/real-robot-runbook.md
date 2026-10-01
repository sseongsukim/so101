# ResiP sim2real 런북 (SO-101 StackCube)

ResiP 논문(§II-D) 방식이다. 시뮬에서 RL로 다듬은 **state 기반 teacher**(ResiP)는 실물에
직접 올리지 않는다. teacher의 성공 rollout을 domain randomization을 넣어 렌더링하고,
그 데이터에 실물 데모를 섞어 **RGB+관절 student(Diffusion Policy)** 를 학습해서 실물에 배포한다.

```bash
conda activate so101-teleop
cd ~/workspace/research/so101-team
```

Isaac을 띄우는 스크립트는 `python -u ... --headless`로 실행한다.

| # | 단계 | 어디서 | 스크립트 |
|---|---|---|---|
| A | teacher export·검증 | 원격 | `export_resip_teacher.py`, `eval_resip_teacher.py` ✅ |
| B | 렌더링 DR 확인 | 원격 | `preview_render_randomization.py` ✅ |
| C | 합성 데모 생성 | 원격 | `generate_teacher_demos.py` ✅ (600 에피소드) |
| D | student(DP) 학습·시뮬 평가 | 원격 | `train_dp.py`, `eval_dp_sim.py` (DP 포트 ✅) |
| 0 | 하드웨어·카메라 확인 | 현장 | `view_cameras.py --snapshot` |
| 1 | 관절 매핑용 wrist 캡처 | 현장 | `web_handeye_capture.py --camera wrist` |
| 2 | follower 매핑 fit·적용 | 원격 | `fit_follower_joint_mapping.py --write` |
| 3 | actuator sys-id | 현장 + 원격 | `sysid_joint_tracking.py` |
| 4 | 실물 데모 10~40개 | 현장 | `record_real_demos.py` |
| 5 | co-training 재학습 | 원격 | `train_dp.py` (sim + real) |
| 6 | 실물 배포 | 현장 | `deploy_policy_real.py` |

---

## A. teacher export + 검증 (완료)

```bash
# JAX/Flax가 있는 환경에서 한 번 실행 (CPU로 충분). Isaac 환경에는 JAX가 없다.
python scripts/export_resip_teacher.py --run sweep/gamma99/so101-StackCube-v0/resip/resip_sd042_20260924_210715 --epoch 2000
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_resip_teacher.py   # torch 포트 == Flax (1e-4)
python -u scripts/eval_resip_teacher.py --headless --num-envs 256
```

teacher는 best policy인 `sweep/gamma99/.../resip_sd042_20260924_210715` (gamma 0.99, learn_std False)다.
2026-09-29 결과: 92.2% (256 에피소드, 첫 성공 평균 73 step). 학습 로그는 90~92%였다.
(이전 후보 learn_std run은 같은 조건에서 81.6%, 115 step.)
teacher는 `origin/main` 씬(로봇 루트가 원점)에서 학습됐으므로, 이 브랜치의 보정된 씬에서는
`so101.tasks.teacher.teacher_observation`이 위치를 로봇 기준으로 바꿔서 넣는다.
보정 없이 넣으면 성공률 0%다.

## B/C. 렌더링 DR + 합성 데모 (완료)

```bash
python -u scripts/preview_render_randomization.py --headless        # outputs/render_randomization/*.png
python -u scripts/generate_teacher_demos.py --headless --num-envs 16 --num-episodes 600 --out outputs/synthetic/resip_v2
```

- DR 항목: key/dome 조명(방향·세기·색), 큐브 색(HSV), 테이블 밝기·색, roughness,
  카메라 pose (front ±1 cm/±1.5°, wrist ±3 mm/±1°). 범위는 `RenderRandomizationCfg`에서 조정한다.
- **실물 큐브 색이 정해지면** `scenes.py`의 큐브 nominal 색을 실물에 맞추고 합성 데모를 다시 생성한다.
  DR은 nominal 색 주변만 흔든다.
- 성공한 에피소드만, 첫 성공 후 15 step까지 저장한다. 16 env 기준 1.75분에 약 14개,
  에피소드당 약 71 MB다.

## D. student 학습 + 시뮬 평가

student는 논문 코드(robust-rearrangement)의 이미지 Diffusion Policy를 이식한 것이다
(`so101.learning.dp`, ResNet18(GroupNorm) × 2 카메라 + UNet, obs 1 / pred 32 / action 8,
warm-start DDIM). 논문의 실물 설정은 R3M+transformer지만 ResNet18+UNet을 쓴다.

```bash
python -u scripts/train_dp.py --data sim=outputs/synthetic/resip_v2 --out outputs/dp_train/sim_v2 \
    --steps 100000 --batch-size 128 --amp --save-freq 10000 --num-workers 6
python -u scripts/eval_dp_sim.py --checkpoint outputs/dp_train/sim_v2/dp_so101.pt --headless \
    --num-envs 4 --num-rounds 10 --video-dir outputs/dp_eval/sim_v2
python -u scripts/eval_dp_sim.py --checkpoint outputs/dp_train/sim_v2/dp_so101.pt --headless \
    --num-envs 4 --num-rounds 10 --render-randomization      # DR 조건에서의 견고성
```

- 11 GB GPU에서는 batch 256이 OOM이 난다. batch 128 + AMP가 step당 0.36 s이므로 10만 step에 약 10시간 걸린다.
  논문은 batch 256으로 50만 step을 돌린다.
- 첫 학습 때 pickle을 memmap 캐시(`outputs/dp_cache/`)로 한 번 변환한다. 파일 목록이 바뀌면 다시 변환한다.
- 평가는 학습 데이터와 같은 조건(env 간격 30 m, 배경 있음)에서 한다.

## 0. 하드웨어 확인 — 현장

```bash
ls -l /dev/so101-leader /dev/so101-follower
python scripts/view_cameras.py --snapshot           # 640x480, 밝기 확인 (원격 확인 때 평균 약 10/255로 어두웠다)
```

- 실물 큐브: 시뮬과 같은 크기(작은 큐브 2.5 cm, 큰 큐브 4 cm)여야 한다. 테이블 보드 위치는
  front 카메라 캘리브레이션의 전제이므로 옮기지 않는다.

## 1~2. follower 관절 매핑

기존 wrist 캡처로 확인한 결과: 현재 linear 매핑은 shoulder_lift/wrist_flex/wrist_roll 스케일이
10~12% 틀렸다. physical(degree) 매핑으로 바꾸면 보드 scatter가 40 → 17.5 mm로 줄고,
교차검증에서도 같은 결과가 나온다. offset 중 shoulder_lift는 불안정해서 새 캡처가 필요하다.

```bash
# 현장: 관절(lift/elbow/flex)을 양방향으로 크게 쓰는 30~40장, 보드는 고정
python scripts/web_handeye_capture.py --camera wrist --capture-dir outputs/handeye/wrist_mapping
# 원격: UNSTABLE 경고가 없으면 --write
python -u scripts/fit_follower_joint_mapping.py --headless --capture-dir outputs/handeye/wrist_mapping --write
```

`calibration/joint_mapping/follower.yaml`이 생기면 `LeRobotSO101Interface`(kind=follower),
hand-eye solve, 실물 스크립트 전부가 자동으로 그 매핑을 쓴다 (`follower_mapping()`).
현재 카메라 외부파라미터는 wrist가 CAD nominal, front가 테이블 보드 PnP이므로
매핑이 바뀌어도 다시 풀 필요가 없다. hand-eye를 다시 푼다면 `calibrate_handeye.py --solve`가
`raw_values`에서 현재 매핑으로 관절을 다시 계산한다.

## 3. actuator sys-id — 현장 + 원격

```bash
python scripts/sysid_joint_tracking.py --real --out outputs/sysid/real.npz            # 현장, 약 60초 움직임
python -u scripts/sysid_joint_tracking.py --sim --headless --out outputs/sysid/sim.npz # 원격
python scripts/sysid_joint_tracking.py --compare outputs/sysid/real.npz outputs/sysid/sim.npz
```

관절별 지연(step), 90% 상승시간, 오버슈트, 정상상태 오차를 비교한다. 실물 지연이 시뮬보다
2~3 step 이상 크거나 오버슈트가 크게 다르면, `so101/assets/so101.py`의 stiffness/damping을
조정하고 teacher 성능을 다시 확인한 뒤 합성 데모를 재생성할지 판단한다.

## 4. 실물 데모 — 현장

```bash
python scripts/record_real_demos.py --out outputs/real_demos/v1
# r = 에피소드 시작 (큐브 배치 후), t = 성공 저장, b = 폐기, Ctrl+C = 종료 (홈 자세로 접은 뒤 토크 해제)
```

- 에피소드는 시뮬 시작 자세 근처에서 시작한다 (시작 시 로그에 표시된다).
- 논문 기준 10~40개면 충분하다. 큐브 위치는 시뮬 스폰 범위
  (로봇 기준 x 0.20~0.30 m, |y| 0.055~0.15 m, 작은 큐브와 큰 큐브는 서로 반대편)를 고르게 덮는다.

## 5. co-training — 원격

```bash
# 매핑을 확정했으면 먼저 실물 데모를 새 매핑으로 다시 변환한다
python scripts/remap_real_episodes.py --src outputs/real_demos/v1 --out outputs/real_demos/v1_remapped
# 합성 데이터로 학습한 모델을 이어서 합성 + 실물로 co-training
python -u scripts/train_dp.py --data sim=outputs/synthetic/resip_v2 --data real=outputs/real_demos/v1_remapped:5 \
    --out outputs/dp_train/cotrain_v1 --init-from outputs/dp_train/sim_v2/dp_so101.pt --reset-step \
    --steps 30000 --batch-size 128 --amp --save-freq 5000
```

- `:5`는 실물 샘플의 추출 확률 가중치다. 논문은 단순히 이어 붙였고(가중치 1), 이 경우 실물 비율은
  프레임 수 비율이 된다. 실행 시 source별 배치 비율이 출력된다.
- 처음부터 합성+실물로 학습해도 된다 (`--init-from` 없이).

## 6. 실물 배포 — 현장

```bash
python scripts/deploy_policy_real.py --policy dp --checkpoint <ckpt> --dry-run        # 먼저 추론만
python scripts/deploy_policy_real.py --policy dp --checkpoint <ckpt> --episodes 10
```

- 명령은 관절 한계로 clamp하고, tick당 0.06 rad로 rate limit한다. 각 에피소드 전에
  시작 자세로 천천히 이동하고, 종료 시 홈 자세로 접은 뒤 토크를 해제한다.
- 모든 rollout은 `outputs/real_rollouts/<run>/`에 기록된다. 에피소드마다 성공 여부를 입력한다.
- 처음에는 손을 비상정지 위치에 두고 1개 에피소드씩 확인한다.
