# Region-conditioned cube pose 구현 — 2026-10-08

## 구현 범위

`scripts/train_simdist_cube_pose.py --region-conditioned`를 추가했다.
기존 pooled/spatial/spatial-mean 기본 동작과 checkpoint head keys는 유지한다.
새 checkpoint는 `region_conditioned=True`로 생성한 head에 strict-load해야 한다.
`ConditionalCubePose`는 normalized XYZ `[B,2,3]`와 independent inside logits
`[B,2]`를 반환한다. 순서는 held/fixed다. 입력은 현재 RGB와 measured q뿐이다.
카메라 pose/K/projection, GT region, future pose, phase, success를 입력하지 않는다.

수정 파일:

- `src/so101/learning/simdist/cube_pose.py`: optional two-output head.
- `src/so101/learning/simdist/cube_pose_region.py`: versioned domain, labels,
  fit normalization/class weights, masked loss, GT-denominator metrics.
- `scripts/train_simdist_cube_pose.py`: opt-in wiring, checkpoint/report metadata.
- `tests/test_simdist_cube_pose_region_contract.py`: focused contracts and CPU fixtures.

원본 raw/archive는 변경하지 않는다. Reward, finite-horizon value,
termination/truncation 및 collector 행동에는 변경이 없다. 기존 table-support
audit는 진단용이며 학습 mask로 사용하지 않는다.

## 고정 계약

Domain version은 `so101_cube_task_region_v1_20261008`이다. Object별 inclusive
box는 robot-root translation-relative simulator-world axes(m)에서
`x,y∈[-1,1]`, `z∈[h_table−0.10,h_table+1.00]`이다. 현재 `h_table=.0300814467`.
Bounds source/version/frame/h/source를 checkpoint/report에 저장한다. 실제
archive에 h가 있으면 선언된 h와 1e-6m 이내 일치하는지 검사하며, legacy
source는 명시된 scene constant를 사용하고 cross-check coverage를 보고한다.
Cube 위치에서 h를 추정하지 않는다. Domain은 support/visibility/reachability/
success/terminal 정의가 아니며, outside를 absorbing state로 만들지 않는다.

Current와 실제 successor XYZ에서 `region_current`/`region_next`를 별도로
계산한다. Episode 내부 successor continuity와 boundary label 일치를 검사하고,
final successor도 실제 기록을 사용한다. 비유한 label은 corrupt source로 거부한다.

Q mean/std는 전체 fit frames, XYZ mean/std는 각 object의 GT-inside fit frames
에서만 구한다. XYZ std floor는 1mm다. Fit에서 object별 valid count가 0이면
거부한다. Invalid XYZ는 normalization 전에 fit mean으로 대체하고 pose loss에서
mask한다. GT 위치를 clip하지 않는다. Source arrays/files는 보존한다.

`L_xyz=.5*sum_object mean(valid XYZ normalized squared error)`이며, minibatch
object mask가 비면 그 object는 fixed weight를 유지한 differentiable zero를
기여한다. Region BCE는 모든 fit frames를 사용한다. Class order는 outside/inside,
fit-only weights는 `N/(2*N_class)`다. Class가 없으면 observed class weight=1,
absent class weight=0이며 방향을 untested로 표시한다. Oversampling은 없다.
`L=L_xyz+.1*L_region`. Epoch 선택은 calibration의 전체 masked XYZ MSE만 사용하며,
calibration에 object별 valid count가 없으면 거부한다. Decision threshold=.5다.

## 평가

- Held-Z p95: 모든 GT-held-inside frames.
- Relative XY p95: 모든 GT-both-inside frames.
- Object XYZ RMSE: 해당 object의 GT-inside frames.
- Region confusion: 모든 retained frames. FNR/recall 분모가 0이면 NA.

Predicted region으로 pose 평가 대상을 줄이지 않는다. Full raw diagnostic과
per-object/current-successor counts, phase/motion/quality/episode/reset-round
strata도 저장한다. 기존 2.5mm held-Z, 9mm relative XY 각 축 gate를 유지하며,
aggregate와 각 selection round의 pose gate를 함께 확인한다. Inside coverage
≥99%, outside recall ≥95%는 별도 engineering gate다. Outside gate는 최소
3 outside episodes와 2 reset rounds가 있어야 evaluated이며 아니면 inconclusive다.
Pose gate 통과는 region classifier나 planner 검증 통과를 뜻하지 않는다.

## CPU 검증

실행 명령:

```bash
conda run --no-capture-output -n so101 python -m pytest -q \
  tests/test_simdist_cube_pose.py \
  tests/test_simdist_cube_pose_spatial_contract.py \
  tests/test_simdist_cube_pose_region_contract.py
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py --help
```

**34/34 PASS**, CLI help exit 0. Inclusive boundaries, legitimate lifted
states outside table support, actual successor/outside→inside semantics,
nonfinite rejection, fit-only per-object norm/q-all-fit, empty mask gradients,
absent classes, GT denominators, 317-frame recall eligibility, legacy keys,
raw-pickle/NPZ CPU trainer fixtures와 weights-only strict reload를 검증했다.
Tiny one-epoch CPU fixtures는 camera stub을 쓰며 성능 증거가 아니다.

실제 데이터의 GT/q-only CPU smoke는 기존 두 raw directories
`/dev/shm/so101-object-pose-dev16`, `.../so101-object-pose-dev32-extra`와
27200–27207의 네 aligned NPZ shards에서 이미지 encoder 없이 실행했다.
Raw는 `tick<horizon`만 유지하고 archive는 episode starts/lengths로 구분한 뒤
`current_successor_regions`, `fit_region_normalization`,
`fit_region_class_weights`, `region_pose_metrics`를 호출했다. Fit seeds는
27000–27003, 27100–27103, 27200–27201이다.

결과:

- 64 episodes, 16,514 objective frames; fit 10,276 frames.
- Fit inside counts held/fixed=9,959/10,276; outside=317/0.
- Held XYZ mean(m)=`.266739,-.014756,.057752`; std=`.055752,.108706,.030728`.
- Fixed mean=`.252638,-.009103,.050286`; std=`.040335,.109788,.001358`.
- Fit region weights outside/inside: held=`16.208202/.515915`, fixed=`0/1`.
- Perfect-oracle classifier를 넣어도 held outside는 2 episodes/1 round라
  recall gate **inconclusive**다. Fixed negative class는 없으므로 역시
  **inconclusive**다. 이 probe는 eligibility/normalization 검증이며 학습 성능이 아니다.

H2 1900–1903을 읽지 않았다. GPU/Isaac 또는 real-data estimator training은
이 구현 단계에서 실행하지 않았다. Root가 별도 실행한 broader focused suite도
PASS라고 보고했다; 위 34건은 Sol의 실행 기록이다.

## 이후 실행과 남은 범위

선행 Sol high 계약 리뷰는 완료됐고, 구현/CPU 검증도 완료했다. 이번 turn에는
GPU training을 승인하거나 실행하지 않았다. 아래는 다음 실행 검토를 위한
**bounded 40-epoch 개발 명령**이며 장기 학습 명령이 아니다.

```bash
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --data-archive /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz \
    /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz \
    /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz \
    /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v5_region_mean.pt \
  --device cuda --representation spatial-mean --region-conditioned \
  --region-table-height-m .0300814467 --spatial-cache-mib 4096 \
  --feature-batch-size 8 --train-batch-size 64 --epochs 40 \
  --seed 20261008 --learning-rate 1e-3 \
  --fit-seeds 27000 27001 27002 27003 27100 27101 27102 27103 27200 27201 \
  --calibration-seeds 27202 27203 --selection-seeds 27204 27205 27206 27207
```

Spatial 후보는 동일 명령에서 representation을 `spatial`, output을 별도
`cube_pose_v5_region_spatial.pt`로 바꾸고 순차 실행한다.

계약 리뷰 후 bounded 개발 실험을 실행할 때 기존 coverage command에
`--region-conditioned --region-table-height-m .0300814467`를 추가하고,
별도 `cube_pose_v5_region_*.pt` output을 사용한다. Same 40 epochs, seed,
split과 `--spatial-cache-mib 4096`을 유지하고 spatial/mean을 순차 비교한다.
현재 16,514 map frames는 약3.31GB float32 cache다. Full-dataset prefill 여부와
reencoded_chunks를 확인한다. 변경된 flags/metadata 없이 기존 v4 checkpoint를
region-conditioned 결과로 해석하지 않는다.

기존 `evaluate_simdist_cube_pose_regions.py`는 table-support diagnostic 및
legacy single-output checkpoint용이다. 새 artifact는 trainer의 GT-domain
reports를 사용하거나 새 contract를 이해하는 별도 reader에 연결해야 한다.
현재 전용 standalone final-H2 evaluator 연결은 이 구현 범위 밖이다.

정상-state domain 보존은 root의 CPU audit로 확인됐지만, 조건부 pose 정확도와
inside classification coverage는 아직 학습 후 검증이 필요하다. Outside의
독립 episode/seed coverage 부족도 남는다. 소비된 27204–27207은 development
selection이며 H2는 candidate를 고정한 후에만 사용한다. Pose gate 실패 시
transition/planner로 진행하지 않는다. Outside detection, causal object
transitions, geometry reward/FK parity 및 controller는 별도 검증 대상으로 남는다.

## Bounded GPU development runs (2026-10-08)

After Sol high contract review and the archive CPU smoke, both predeclared
40-epoch region-conditioned candidates were trained on the same 64-episode
corpus. Outputs remain in `/dev/shm/so101-object-pose-dev16/`.

- Spatial-mean: `cube_pose_v5_region_mean.pt/.json`, best epoch 22.
  Selection GT-domain held-Z p95 `20.268 mm`; relative-XY p95
  `72.703/209.974 mm`; all four reset rounds failed.
- Learned spatial attention: `cube_pose_v5_region_spatial.pt/.json`, best epoch
  20. Selection held-Z p95 `24.286 mm`; relative-XY p95
  `75.746/125.365 mm`; all four reset rounds failed.
- Both inside classifiers covered 100% of selection in-domain frames. This is
  not outside detection validation: only the held cube has negatives (2
  episodes in one reset round), while fixed has none. Outside recall remains
  inconclusive for both objects.
- Each run prefetched all 16,514 feature frames into the declared 4 GiB CPU
  cache: 3,314,425,856 bytes, 2,094 extraction chunks, zero re-encoded chunks.
- The artifacts are development diagnostics over consumed selection seeds,
  not confirmatory/generalization evidence. Pose gate failed; no transition,
  reward/value, MPPI, H2, or real-world evaluation followed.

Commands were the exact 40-epoch commands in the preceding section, each run
once with `--representation spatial-mean --out ...cube_pose_v5_region_mean.pt`
and once with `--representation spatial --out ...cube_pose_v5_region_spatial.pt`.

This is enough evidence to reject the current frozen-DP-map + q pose front-end
as the next planner input. It does not establish that visual world models are
impossible for SO101. Sol high is reviewing a concrete next visual-supervision
step using the corrected camera-pose stream; no wrist-only causal diagnosis is
a prerequisite. Keep the present failure artifacts and do not start MPPI from
them.
