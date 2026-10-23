# 포트폴리오 슬라이드 6·8 촬영 순서

대상: `/home/keti/Downloads/KIA_portfolio_claude.html`. HTML은 수정하지 않는다.
캘리브레이션 파일을 그대로 읽으며, 생성물은 `outputs/portfolio_capture/`에 저장한다.
실기 동작은 아래 명령을 현장에서 실행할 때 시작된다. 이 문서 작성 과정에서는 로봇을 움직이지 않았다.

## 0. 준비

```bash
conda activate so101-teleop
cd /home/keti/workspace/research/so101-team
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

카메라를 쓰는 프로그램은 하나씩 실행한다. `view_cameras.py`, 데모 수집, 배포를
동시에 실행하면 같은 USB 장치를 중복으로 열 수 있다. Isaac도 단계마다 종료 후 다음 명령을 실행한다.
사진은 원본 PNG를 사용하면 창 크기/장식/텍스트가 섞이지 않는다.

## 1. 슬라이드 6 — 실물 9점 터치·로봇 배치

배치 안내도를 만든다. 안내도는 실물 사진을 대신하는 자료가 아니며, 인쇄 축척도 아니다.

```bash
python scripts/portfolio_capture.py touch-guide
xdg-open outputs/portfolio_capture/slide6_touch/touch_placement_guide.svg
```

노란 로봇 기준점 원점에서 +X는 테이블 안쪽, +Y는 로봇 왼쪽이다.
X=18/25/32 cm, Y=-14/0/+14 cm의 교차점 9개를 자로 표시한다.
휴대폰/별도 카메라로 로봇 전체와 9개 점이 한 화면에 나오도록 촬영한다.
그리퍼 끝이 점에 닿은 사진이 필요하면 기존 터치 수집 도구로 leader를 움직인다.

```bash
python scripts/touch_calibration.py --record
```

SPACE=현재 점 기록, u=이전 점 다시, q=종료. 점 순서는 안내도 번호와 같다.
이미 보정한 값을 사용하며 이번 사진용 촬영에서 `--fit --write`는 실행하지 않는다.
새 touch JSON은 별도 저장된다. 이전 18.2→5.5 mm 값은 과거 측정 결과이며,
이번 재촬영으로 같은 수치를 다시 측정했다는 의미로 쓰지 않는다.

## 2. 슬라이드 6 — 같은 자세 Real 카메라

테이블 보드를 보정 당시 위치에 놓는다. 보드 중심은 노란 기준점에서
X=32.75 cm, Y=+5.70 cm이고 긴 축은 Y 방향이다.
아래 명령은 실기를 5초에 걸쳐 보드 관측 자세로 이동시키고 front/wrist를 보여준다.

```bash
python scripts/portfolio_capture.py real-pose --show --move-seconds 5 --seconds 120
```

정지 후 원본 PNG와 실측 관절값이 자동으로 저장된다. 창에 포커스를 주고 s/SPACE를 누르면
다시 저장한다. q 또는 창 닫기로 종료하면 기존 `go_home()` 경로로 복귀한다.
OpenCV가 headless인 경우 기존 Matplotlib 창 경로를 사용한다.

- `outputs/portfolio_capture/slide6_alignment/real/real_front.png`
- `outputs/portfolio_capture/slide6_alignment/real/real_wrist.png`
- `outputs/portfolio_capture/slide6_alignment/real/pose.json`

원하는 자세를 직접 지정하려면 `--pose Rotation Pitch Elbow Wrist_Pitch Wrist_Roll Jaw`를
라디안 6개로 전달한다. 현재 자세만 읽고 사진을 찍으려면 `--read-only`를 사용한다.
`--read-only` 종료 시 자동 home 이동도 하지 않으므로 로봇을 지지한다.

## 3. 슬라이드 6 — 같은 자세 Sim 카메라

Real 단계가 종료된 뒤 실행한다. 직전 사진의 **실측 관절값**을 읽는다.
실측 관절값을 맞추는 비교이므로 실제 물리적 위치 오차가 완전히 사라졌다는 의미는 아니다.

```bash
python -u scripts/stream_sim_cameras.py \
  --pose-file outputs/portfolio_capture/slide6_alignment/real/pose.json \
  --clean-board --seconds 120 \
  --out outputs/portfolio_capture/slide6_alignment/sim
```

Isaac 전체 로봇 viewport와 front/wrist 창이 열린다. 깨끗한 원본 이미지는
`nominal_front.png`, `nominal_wrist.png`이다. 슬라이드의 같은 자세 Sim/Real 두 칸에는
각각 동일한 카메라(예: wrist)를 쓰거나 front/wrist를 같은 순서로 붙인 사진을 쓴다.
캡처 후 Isaac 창을 닫거나 Ctrl+C로 종료한다.

## 4. 슬라이드 6 — Rendering DR 기준·변형 사진

보드 없이 같은 장면에서 기준과 DR 이미지를 차례로 저장한다.
v3 합성 생성 설정처럼 wrist 위치 jitter ±1 cm, 회전 perturbation 최대 3°를 사용한다.
조명·큐브 색·테이블·배경도 기존 RenderRandomizer로 변경된다.
seed가 같으면 같은 설정을 재현한다. 여기서는 nominal과 DR를 같은 정지 관절 목표로 촬영한다.

```bash
python -u scripts/stream_sim_cameras.py \
  --headless --no-board --render-dr --seconds 1 --seed 20261006 \
  --wrist-pos-jitter-m 0.01 --wrist-rot-jitter-deg 3 \
  --pose-file outputs/portfolio_capture/slide6_alignment/real/pose.json \
  --out outputs/portfolio_capture/slide6_dr
xdg-open outputs/portfolio_capture/slide6_dr/nominal_wrist.png
xdg-open outputs/portfolio_capture/slide6_dr/dr_wrist.png
```

front 사진도 `nominal_front.png`/`dr_front.png`로 저장된다.
화면을 직접 캡처하려면 `--headless`를 빼고 `--seconds 120`으로 실행한다.
현재 열린 화면은 DR이며 nominal은 먼저 저장된 PNG로 비교한다.
기존 `RenderRandomizationCfg` 기본 wrist 범위(3 mm/1°)와 v3 생성기의 명시적
override(1 cm/3°)가 다르기 때문에 위 명령에 범위를 명시했다.

## 5. 슬라이드 6 — Sim 텔레옵 사진

leader를 연결하고 실행한다. 실제 follower는 이 명령으로 움직이지 않는다.

```bash
python scripts/teleop_task.py so101-visual-StackCube-v0 --visual \
  --rate 30 --dataset-dir outputs/portfolio_capture/slide6_sim_teleop
```

leader를 조작해 Sim이 큐브에 접근/파지하는 장면에서 Isaac viewport를 캡처한다.
r=새 에피소드, t=성공 저장, b=폐기. 종료는 Ctrl+C.
leader를 조작하는 본인과 Sim 화면을 함께 휴대폰으로 찍어도 텔레옵을 설명하기 좋다.
슬라이드 6의 마지막 실기 적층 사진은 아래 새 DP 실기 촬영에서 골라 사용한다.

## 6. 슬라이드 8 — 새 Sim/Real DP 촬영

기존 실기 비교에 사용했던 v3 DP를 양쪽에 동일하게 사용한다.
ablation의 `sim600_real40` 모델을 사용하려면 **두 명령의 checkpoint를 모두** 그 경로로
바꾸고 새로운 실기 평가로 표기한다. 서로 다른 모델의 영상을 섞지 않는다.

전체 로봇 영상은 Sim에서는 Isaac 기본 viewport를 화면 녹화하고,
Real에서는 휴대폰/별도 카메라로 촬영한다. front 카메라는 정책 입력 카메라여서
“로봇 전체”의 별도 외부 촬영을 대체하지 않는다.
Real front/wrist는 deployment rollout에 자동 저장된다. 배포 중 별도 viewer로 USB 카메라를 열지 않는다.

촬영 배치: 노란 기준점에서 작은 큐브 **중심** X=28 cm/Y=+8 cm,
큰 큐브 중심 X=28 cm/Y=-12 cm, 둘 다 yaw=0°. 작은 큐브 2.5 cm, 큰 큐브 4 cm.
Sim은 아래 플래그로 이 배치를 고정한다. Real은 운영자가 같은 좌표에 둔다.
동시 실행/완벽한 시간 동기화가 아니라 동일 태스크·모델·배치의 별도 실행을 비교한다.

### 6-1. Sim 실행·영상

```bash
python -u scripts/eval_dp_sim.py \
  --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --num-envs 1 --num-rounds 3 --max-steps 640 \
  --seed 20261006 --inference-steps 4 --action-horizon 8 \
  --small 28 8 0 --large 28 -12 0 --realtime \
  --video-dir outputs/portfolio_capture/slide8/sim_video \
  --out outputs/portfolio_capture/slide8/sim_showcase_eval.json
```

Isaac viewport에서 전체 로봇을 화면 녹화한다. front|wrist 합친 센서 MP4는
`sim_video/round_000.mp4`부터 자동 저장된다. 센서 MP4 시간축은 simulation control dt다.
`--realtime`은 빠르게 지나가는 GUI를 완화하지만 GPU 추론이 control dt보다 길면 화면 녹화는
늦어진다. 영상 편집 시 원본 시간축과 재생 배속을 구분한다.
고정 배치·DDIM 4-step 촬영 평가이며 기존 ablation(기본 추론 설정·40회)의 재평가로 쓰지 않는다.

### 6-2. Real 실행·영상

Sim을 종료하고 휴대폰 전체 로봇 녹화를 시작한 뒤 실행한다.

```bash
python scripts/deploy_policy_real.py --policy dp \
  --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --episodes 3 --max-seconds 25 --fps 30 \
  --inference-steps 4 --action-steps 8 \
  --record outputs/portfolio_capture/slide8/real_rollouts
```

터미널에서 Enter 없이 단일 키로 조작한다. `r`은 초기 자세로 복귀 후 대기,
배치를 맞춘 뒤 `s`를 누르면 에피소드 시작, `y/n`은 성공/실패 저장이다.
`q`는 대기·동작 중 모두 종료하며, 실행 중 `r`은 현재 시도를 중단으로 기록하고
초기 자세로 복귀해 다음 `s`를 기다린다. `--episodes`는 y/n으로 판정한 회차 수이며
중단된 시도는 별도로 보존한다. `--max-seconds`에 도달하면 정지한 채 y/n 판정을 기다린다.
기본 키 입력은 **터미널에 포커스**가 있어야 한다. X11 데스크톱 전체에서 입력을 받고 싶으면
`--keys desktop`을 추가한다. q 종료는 정책 출력을 멈춘 뒤 기존 home 경로로 복귀한다.
이 명령은 실제 로봇을 움직인다. 기본 `max-step-rad=0.25`를 유지해 촬영 중 다른 속도 제한을
추가하지 않는다. 명목 control rate는 30 Hz지만 실기는 추론/통신 overrun으로 달라질 수 있다.
현재 기본 배포처럼 시작 자세에서 카메라 brightness를 meter한다.
모든 시도·실패·중단 rollout을 보존하고, 촬영 세션의 새 성공/시도 수로 표시한다.
새 rollout에는 checkpoint 경로와 실행 인자도 함께 저장된다.
같은 저장 디렉터리에 세션별 `evaluation_*.jsonl`과 `evaluation_*_summary.json`을 기록한다.
summary에는 성공/실패/중단 수, y/n 판정 회차 기준 성공률과 전체 시도 목록을 저장한다.

### 6-3. Real front/wrist 영상 추출

```bash
ls outputs/portfolio_capture/slide8/real_rollouts/cotrain_v3_slow2/trajectory_*.pkl
python scripts/portfolio_capture.py export-rollout \
  --episode outputs/portfolio_capture/slide8/real_rollouts/cotrain_v3_slow2/trajectory_000000.pkl \
  --out outputs/portfolio_capture/slide8/real_video
```

번호가 다르면 실제 파일명을 넣는다. `front.mp4`, `wrist.mp4`, `front_wrist.mp4`와
start/end PNG, success/checkpoint/timing 메타데이터 JSON이 생성된다.
실기 영상은 저장된 `wall_times`를 사용해 실제 경과시간을 보존한다.
저장된 센서 프레임은 원래 recorder의 320×240 해상도이며 640×480 원본으로 가장하지 않는다.
성공 직후 팔이 home으로 복귀하므로 실기 적층 사진은 휴대폰 영상에서 적층 완료 시점을
고르거나 센서 영상의 해당 프레임을 캡처한다. 마지막 프레임이 반드시 최상의 적층 사진인 것은 아니다.

최종 편집: 위 행 Sim / 아래 행 Real, 열 순서 전체 로봇 / front / wrist.
단일 2×3 영상으로 구성하고, 별도 실행 영상을 맞춘 경우 “시간 정렬 비교”라고 표기한다.
Sim/Real의 파지→이동→release 시점을 보고 맞추되 실제 속도 차이를 숨기는 배속을 쓰면 배속을 표시한다.
