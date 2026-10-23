# SO101 cube pose v5 실패와 pixel supervision 다음 실험

2026-10-08, Sol high 검토. 이번 작업은 저장된 결과와 코드의 검토이며,
collector 실행·모델 학습·H2·robot·planner 실행은 하지 않았다.

## 1. 보존할 실패와 해석

기존 결과는 `/dev/shm/so101-object-pose-dev16/cube_pose_v5_region_{mean,spatial}.json`
및 대응 checkpoint에 있다. 실행 명령은
`simdist-cube-pose-region-implementation-20261008.md`에 보존돼 있다.

| Candidate | Fit held-Z p95 mm | Fit relative XY p95 mm | Selection held-Z p95 mm | Selection relative XY p95 mm |
|---|---:|---:|---:|---:|
| spatial-mean, epoch 22 | 13.702 | 30.293 / 48.437 | 20.268 | 72.703 / 209.974 |
| spatial, epoch 20 | 11.707 | 23.160 / 47.647 | 24.286 | 75.746 / 125.365 |

둘 다 selection 네 reset round 전체가 pose gate에 실패했다. Fit에서도
held-Z <2.5 mm, 상대 XY 각 축 <9 mm를 충족하지 못한다. 따라서 더 많은
reset 데이터만이 필요한지, 현재 representation/head가 충분한지 분리해야 한다.
Spatial은 fit을 개선했지만 selection Z/X는 mean보다 나쁘다. 일관된 승자는 없다.

- Region-conditioned normalization은 거대한 낙하 tail이 정상 pose 통계/loss를
  오염하는 경로를 제거했다. v5의 잔여 오차까지 그 원인으로 설명할 수 없다.
- Fit 10,276 frame 중 held-inside 9,959, outside 317; fixed는 전부 inside다.
  Calibration 1,903 / selection 4,335 frame은 양 object 모두 inside다.
  Selection inside coverage 100%는 outside detection 성능 증거가 아니다.
- Outside held 2 episode / 1 reset round, fixed 0 negative라 outside recall gate는
  여전히 inconclusive다. Raw/outcome/domain/value/termination 계약은 유지한다.
- Image swap 악화는 이미지가 예측에 쓰인다는 증거다. 정확한 object localization,
  가시성, monocular depth 식별성, real robustness는 증명하지 않는다.
- `FrozenDPSpatialEncoder`는 기존 camera transform과 frozen 최종 feature map을
  사용한다 (`cube_pose_spatial.py:33–52`). 기존 224 입력의 ResNet map은
  7×7/stride 32 (`dp/vision.py:101–110`). Coarse task-specific feature와
  localization 감독 부재는 검증할 가설이며 확정된 원인이 아니다.

## 2. 결정: annotation smoke GO, 이후 bounded dense-supervision baseline

다음 후보는 **현재 RGB 두 view + measured q → region-conditioned XYZ**를
유지하고, explicit pixel centre/mask 감독을 추가한다. Wrist 원인 규명은
착수 조건이 아니다. 우선 one-episode annotation/geometry smoke만 진행한다.
Smoke 통과 전 bulk collection/GPU training은 NO-GO다. Pose gate 전 planner도 NO-GO다.

### 첫 모델: pixel auxiliary + conditional 3D

- 기존 DP checkpoint camera별 convolution weights를 초기값으로 사용하되,
  첫 실험에서는 encoder를 frozen/eval로 유지한다. Source RGB 전체 240×320,
  ImageNet normalization을 사용한다. 기존 front 224 centre crop은 사용하지 않는
  **새 preprocessing version**으로 기록한다. DP 실행 경로/checkpoint는 수정하지 않는다.
- 각 camera의 layer1/2/3 map(stride 4/8/16)을 32-channel lateral projection과
  작은 FPN으로 합친다. Stride-2 heatmap + subpixel offset / object mask를 출력한다.
  Held/fixed는 별도 channel이고 front/wrist 순서를 고정한다.
- Full 240×320 layer1/2/3 maps는 두 camera float32 합계 4,300,800 bytes/frame이다.
  기존 16,514 frame이면 약 66 GiB로 전체 prefill이 불가능하다. 기존 final-map
  cache를 같은 budget 정책 없이 확장하지 않는다. 첫 후보는 frozen encoder를
  GPU minibatch에서 `no_grad`로 재계산하고 batch maps만 유지한다. 작은 bounded
  cache를 사용할 경우 hit/miss/re-extraction/peak를 명시한다. First-batch extraction
  latency/peak memory를 측정하고 40-epoch cap의 실제 실행 가능성을 확인한다.
- Predicted spatial probabilities로 FPN features를 pool한다. Camera 두 개의
  predicted features/centre summaries와 fit-normalized q를 합친 작은 head가
  기존 per-object conditional XYZ와 domain logits를 출력한다.
- Auxiliary projected-centre는 GT cube **3D centre의 projection**이다.
  Visible mask의 centroid로 대체하지 않는다. Occlusion에서 서로 달라진다.
- Optical centre-depth 보조 scalar를 추가할 수 있다. Target은 GT centre를
  camera frame으로 변환한 Z로 충분하다. 첫 smoke에는 depth-image annotator가
  필요 없다. GT depth나 mask/ROI를 forward 입력으로 사용하지 않는다.
- Camera calibration은 label 생성/oracle geometry 검증에만 쓴다. 첫 모델의
  forward signature는 RGB+q다. 향후 pixel+depth→XYZ branch는 oracle precision과
  actual calibration 확인 뒤 별도 candidate로 연결한다. Wrist pose를 사용할 때는
  inference에서 measured-q FK + 고정 extrinsic으로 재현 가능해야 한다.

기존 XYZ/region loss, fit-only per-object norm과 class weights를 그대로 유지한다.
초기 auxiliary 계수는 `L_xyz + .1 L_region + .1 L_centre + .1 L_mask`로 고정한다.
`L_centre`는 source width/height로 정규화한 centre 좌표/offset MSE,
`L_mask`는 per-camera/object BCE와 foreground/background 균형 평균이다.
한 class가 없으면 있는 class만 평균하며 empty eligible batch는 미분 가능한 0이다.
Mask task의 background는 task-region 밖이라는 뜻이 아니다. Region class weights와
mask pixel weights는 분리한다. All-zero mask도 유효 background label이다.

Centre loss는 GT-inside, positive optical Z, in-frustum centre 및 source mask
≥8 pixel인 frame에만 적용한다. Mask threshold 8은 auxiliary eligibility일 뿐,
pose gate/visibility 진실/region label로 쓰지 않는다. Mask loss는 annotator mapping이
valid인 모든 frame에 적용한다. XYZ loss와 최종 pose gate는 **모든 GT-inside**에서
유지한다. Occluded/out-of-frustum frame도 model이 유한한 pose를 출력해야 한다.
Predicted confidence/visibility로 최종 평가 분모를 줄이면 안 된다.

이 실험은 coarse-feature global regression과 explicit spatial supervision을
구분한다. Trunk finetuning·history·geometry fusion을 동시에 바꾸지 않는다.

## 3. Collector와 annotation 계약

현재 collector는 camera data_types를 RGB-only로 덮어쓴다
(`generate_teacher_demos.py:102–103`). Opt-in flag로
`instance_id_segmentation_fast`를 추가하고 colorize를 false로 설정한다.

Local IsaacLab `camera.py:665–700`은 raw int32 segmentation을 `(H,W,1)`로
돌려준다. Per-env camera `data.info[env_id][annotator]`의 mapping은 실제 smoke에서
확인한다. `idToLabels` schema와 prim path availability를 가정하지 말고, supported
schema가 아니면 fail-closed한다. ID 숫자, camera 간 동일 ID, frame 간 ID 안정성을
가정하지 않는다.

Object ID는 정확한 `/World/envs/env_i/HeldAsset` 및 `FixedAsset` root와
그 **slash 경계 descendant**의 visible prim IDs를 union한다. Env1과 env10을
prefix로 혼동하지 않는다. Exact target prim은 task scene에서 확인한다
(`tasks/scenes.py:79–91`). 다른 env/object/gripper/table pixels는 negative다.
Mapping 부재와 target pixels 0을 구분한다. 필요하면 unique semantic labels를
추가하는 별도 smoke로 해결하며 unknown schema를 임의로 해석하지 않는다.

### 저장할 최소 추가 schema

- `object_masks_current`: source sensor 좌표 `[T, camera=2, object=2, H, W]`
  bool 또는 lossless bitpack; axis/order/source H/W 명시.
- `annotation_valid_current[T,2]`, `object_mask_pixel_count[T,2,2]`;
  mapping/schema version, exact target prim paths, camera sensor frame counters.
- 현재 기존 cube XYZ + camera pose/K를 이용해 centre optical-Z, source UV,
  in-front/in-frustum flags를 **offline** 생성한다. World-forward-X quaternion을
  직접 optical projection에 쓰지 않는다. Isaac `quat_w_ros`는 +Z forward/-Y up
  (`camera_data.py:72–82`): `p_cam=R_ros.T@(p_robot_relative-t_cam_relative)`,
  `u=fx*x/z+cx`, `v=fy*y/z+cy`. Robot-root-relative/world-axis 계약 유지.
- 현재 RGB·q·GT·pose/K·mask가 같은 rendered tick이어야 한다. Annotator 추가를
  위해 `env.step`를 한 번 더 호출하면 안 된다. Parent camera refresh 수정은
  smoke에서 다시 확인한다.
- 첫 estimator에는 current labels만으로 충분하다. Transition용 archive를 만들면
  successor mask/pose/K도 실제 successor에서 기록해야 한다. `current[t+1] == next[t]`
  연속성, 마지막 successor 포함; reset frame으로 마지막 successor를 대체하지 않는다.
  기존 stable-success terminal/truncation/finite-horizon value는 변경하지 않는다.
- Full-source RGB를 보존한다. Crop/resize를 추가할 때 centres/K에 동일한 pixel
  transform, mask에 nearest transform을 적용하고 explicit version으로 기록한다.

Instance annotations의 비용은 측정해야 한다. Local IsaacLab 4.5 대응 코드는
semantic/instance_segmentation_fast에서 instancing을 끄는 workaround가 있다
(`camera.py:150–158`); instance-ID 방식에 같은 비용이 없다고 단정하지 않는다.
240×320 두 camera의 raw int32 ID buffer는 약 .586 MiB/tick이다.
Object별 bool masks는 .293 MiB/tick, bitpack이면 .0366 MiB/tick이다.
IDs를 archive에 모두 중복 저장하지 않고 mapping metadata + bitpacked object masks를
쓴다. Current/next를 T+1 unique frame으로 저장하면 중복을 피할 수 있지만 기존
archive API를 바꿀 경우 converter continuity tests가 필요하다. 원본 결과는 보존한다.
RGB 원시 bytes는 두 camera 약 .439 MiB/tick이며 compression ratio는 가정하지 않는다.
One-episode smoke에서 elapsed rendering time/peak GPU·CPU·artifact bytes를 기록한다.
메모리/디스크 여유 계산 전 bulk collection은 시작하지 않는다.

## 4. 최소 test, partition과 acceptance

### One-episode smoke: 이 항목 전부 통과해야 annotation GO

1. Foreground cube pixels가 실제 held/fixed RGB 위치와 맞는 overlay를 확인한다.
   Exact-env descendant mapping unit fixture(복수 mesh ID, env1/env10, missing
   metadata, absent object)와 per-frame remapped ID fixture가 필요하다.
2. Quaternion convention, K/pixel transform, GT centre UV/optical-depth roundtrip
   오차 <0.1 mm. 이는 algebra oracle이며 camera accuracy/visibility 증명이 아니다.
   Mask bbox와 projected centre 불일치는 occlusion/geometry 이유를 함께 기록한다.
3. 센서 frame counter/RGB-mask shape, current-successor continuity 및 실제 final
   successor 확인. Geometry source frame counter 불일치면 fail-closed한다.
4. Empty/occluded/outside object, absent class, no valid centre batch의 loss가 finite;
   invalid labels는 연산 전 neutralize하고 masked gradients 0이다.
5. Forward에서 GT masks/UV/depth/poses/region을 지워도 동일 output. Predicted pooling만
   사용. Finite q/RGB-only checkpoint strict roundtrip과 head gradients 확인.

### Bounded model experiment

- Consumed 27204–27207과 27103은 계속 development diagnostics다. H2 1900–1903은
  읽지 않는다. 새 annotation split은 reset-round 단위로 manifest에 고정한다.
- 권장 후보 seed 범위: fit 28000–28003(16 ep), epoch selection/calibration
  28004–28005(8 ep), **candidate 고정 후에만** final dev-selection
  28006–28009(16 ep). Parent가 실제 available seeds/capacity를 확인해 수집 전에 확정한다.
  같은 reset round의 env/episode를 다른 split에 넣지 않는다.
- 기존 fit corpus를 함께 사용하면 split manifest에 넣고 missing annotations는
  auxiliary mask로 명시한다. 그것만으로 새 후보의 metric 비교가 공정하다고
  간주하지 않는다. 우선 새 annotated fit/cal corpus에서 두 후보를 같은 조건으로 비교한다.
- 40 epoch cap, fixed model/training seed, 동일 preprocessing/split, calibration의
  기존 masked-XYZ criterion으로 checkpoint 하나 선택한다. 두 후보는 FPN conditional
  3D with/without pixel auxiliary supervision이며 mask/channel parameters는 유지한다.
  Training steps/cache/GPU/latency를 기록한다. Annotation smoke 후 parent가 실행한다.
- 분류·pixel-centre·depth는 GT visibility/occlusion별 진단을 추가한다. Pixel p95 오차가
  oracle camera Jacobian으로 요구되는 metric tolerance를 넘으면 geometry branch는 NO-GO다.
  고정 1-pixel gate가 metric pose gate를 대신하지 않는다.
- 최종 GT-inside held-Z p95 <2.5 mm, GT-both-inside relative XY 각 p95 <9 mm,
  aggregate + 모든 독립 selection round를 그대로 적용한다. Predicted-mask filtering 금지.
  Outside gate eligibility 부족은 inconclusive로 둔다. Pose pass만으로 transition,
  reward/value, MPPI나 real robustness pass를 선언하지 않는다.

## 5. 이번 검토의 검증 범위

저장된 두 JSON의 fit/calibration/selection 수치와 `pose_gate.passed == False`를
read-only Python으로 확인했다. 현재 collector/DP maps/Isaac camera adapter 소스를
읽었다. 이번 문서는 구현·학습·annotation 성공 결과가 아니며, 추가 테스트는
아직 작성/실행하지 않았다. 다음 구현 task는 collector annotation smoke와
label geometry utilities/tests, 그 다음 FPN model/loss integration 순서다.

## 6. First annotation smoke result (2026-10-08)

The optional collector flag `--record-cube-instance-masks` now records
uncolorized per-frame `instance_id_segmentation_fast` masks. IDs are resolved
from each camera/env's `idToLabels` into exact `HeldAsset`/`FixedAsset` path
roots and descendant prims; maps are saved per episode. Binary masks use
`packbits` (camera order front/wrist, object order held/fixed). The packer
validates contracts, mask shape, synchronized/contiguous camera frame counters,
and preserves the masks in the aligned archive. These remain privileged
training targets; the model inputs are unchanged.

Successful one-episode smoke command:

```bash
conda run --no-capture-output -n so101 \
  /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse \
  --perturbation-std 0.04 --num-envs 1 --num-episodes 1 --max-rounds 1 \
  --seed 27214 --record-cube-instance-masks \
  --out /dev/shm/so101-cube-instance-mask-counter-smoke-fixed
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-cube-instance-mask-counter-smoke-fixed \
  --out /dev/shm/so101-cube-instance-mask-counter-smoke-fixed/aligned.npz
conda run --no-capture-output -n so101 python scripts/audit_simdist_cube_instance_masks.py \
  --trajectory /dev/shm/so101-cube-instance-mask-counter-smoke-fixed/trajectory_000000.pkl \
  --out /dev/shm/so101-cube-instance-mask-counter-smoke-fixed/projection_audit.json \
  --overlay-out /dev/shm/so101-cube-instance-mask-counter-smoke-fixed/mask_rgb_overlay.png
```

The raw episode has 300 transitions, source masks `[300,2,2,480,80]` (packed
width) and 46,080,000 mask bytes; aligned NPZ is 79,227,812 bytes. Both camera
counters run synchronously/contiguously from 2 through 301. Each camera's path
map identified both cube descendant roots. Front masks contain held/fixed
pixels on 300/300 frames; wrist masks on 263/300 and 8/300 frames. The overlay
artifact confirms mask/RGB alignment. The declared camera-frame transform's
world→ROS→world numerical roundtrip is `1.85e-7 m`; this checks algebra only.

A cross-modal projection diagnostic reveals that front geometry is internally
consistent on this episode: projected GT center lies in each visible mask bbox
on all 300 frames; center pixels hit the held/fixed mask on 97.7%/100% of
visible frames. Wrist projection is not reliable yet: center-in-bbox is
72.6% for held and 12.5% for fixed, with 95th-percentile bbox distances
80.3/102.2 px. The masks themselves remain aligned to wrist RGB, so wrist mask
auxiliary supervision can be used; do not use wrist pose/K for pixel-to-XYZ
fusion until diagnosed. This does not create a wrist-only causal prerequisite.
No hard camera calibration or visibility claim follows from one DR episode.

A temporary collector regression accessed `camera.data.frame` inside the loop;
its data getter could trigger unnecessary sensor refresh and the 300-step smoke
ran beyond 10 minutes. It was stopped before writing trajectory output. Reading
`camera.frame` directly fixes this; the final smoke completed in 0.4 minutes
(the collector's rollout timer). Do not use the interrupted attempt as a
throughput measurement. GPU peak was about 5.3 GiB; raw/packed output is the
measured storage figure above. This still needs a multi-episode throughput and
storage budget before annotation bulk collection.

Audit and overlay:

- `/dev/shm/so101-cube-instance-mask-counter-smoke-fixed/projection_audit.json`
- `/dev/shm/so101-cube-instance-mask-counter-smoke-fixed/mask_rgb_overlay.png`
- `/dev/shm/so101-cube-instance-mask-counter-smoke-fixed/aligned.npz`

The first pixel-supervision model should consume **full 240×320 RGB** with
nearest-resized mask targets in that same pixel frame, use projected center
auxiliary targets for front only until wrist projection is sound, and retain
conditional 3D pose loss/gates over every GT-inside frame. Do not use a full
66-GiB early-map cache. Recompute frozen maps in minibatches and measure first
batches; keep the original v5 failure checkpoints and do not start MPPI.

Focused repository validation after collector, packer, path resolver and audit
script changes: `PYTHONPATH=. conda run --no-capture-output -n so101 pytest -q
tests/test_simdist*.py` passed **87 tests**. `py_compile` passed for the
collector, packer, mask audit, pose trainer, region module, and scene camera
configuration; `git diff --check` passed.

## Pixel FPN implementation and bounded validation (2026-10-08)

The opt-in `--representation pixel` seam is implemented in
`src/so101/learning/simdist/cube_pose_pixel.py` and
`cube_pose_pixel_training.py`. It uses full 240×320 RGB and current measured q;
the checkpoint's frozen ResNet conv1 through layer3 maps feed a small FPN.
Only conv1..layer3 are retained in the pixel model, so the unused ResNet layer4,
policy actor, and original DP wrapper stay on CPU and are not transferred to the
training GPU. Feature maps are computed per minibatch with no dataset-wide cache.
The estimator returns two conditioned 3D poses and region logits, mask logits
for both cameras, and front-camera projected-center predictions. Annotation
masks and projected geometry are loss targets only. Wrist masks provide pixel
supervision; wrist pose/K do not produce centre targets.

The raw-pickle and aligned-NPZ readers validate path-resolved instance identity,
camera/tick synchronization, image/mask dimensions, object order, coordinate
semantics, and horizon support. Masks are unpacked big-endian and resized via
nearest-exact to the RGB pixel grid. The existing all-GT-inside 3D loss and
evaluation denominators remain intact, including occluded/absent observations;
centre loss is front-only and requires positive optical depth, in-frame target,
and at least eight source-mask pixels. Selection is masked 3D XYZ loss only.
Centre/mask loss weights are configurable, including zero for a matched
no-auxiliary control.

The actual seed-27214 300-frame archive was CPU-smoked through both readers.
Raw and aligned-NPZ q/XYZ/inside/centre/source-count targets matched exactly;
RGB batches were `[3,3,240,320]`, masks `[3,2,2,240,320]`, and the full early-map
cache counter remained zero. The trunk produced expected stage maps
`64×60×80`, `128×30×40`, `256×15×20`; only conv1..layer3 keys were present and
all trunk parameters remained frozen. This validates the data/model seam, not
pose accuracy or GPU throughput.

Validation commands and outcomes:

```bash
/home/keti/miniforge3/envs/so101/bin/python -m pytest tests/test_simdist_cube_pose_pixel_contract.py -q
# 14 passed before the early-trunk memory regression additions; Sol high reports
# 49/49 focused pixel/pose tests after those additions.
PYTHONPATH=. /home/keti/miniforge3/envs/so101/bin/python -m pytest -q tests/test_simdist*.py
# 102 passed in 5.38s
/home/keti/miniforge3/envs/so101/bin/python -m py_compile scripts/train_simdist_cube_pose.py src/so101/learning/simdist/cube_pose_pixel.py src/so101/learning/simdist/cube_pose_pixel_training.py tests/test_simdist_cube_pose_pixel_contract.py
/home/keti/miniforge3/envs/so101/bin/python scripts/train_simdist_cube_pose.py --help
git diff --check
```

No GPU training, annotated bulk collection, H2 use, planner/MPPI, or real robot
evaluation was run. Current root filesystem has only about 3.2 GB free; `/dev/shm`
has about 3.0 GB free. Before GPU training, collect enough distinct development
rounds for fit/calibration/four-round selection, estimate annotation storage and
throughput, and confirm disk capacity. Then Sol high must review this pixel
architecture/trainer contract and first-batch memory/throughput before any
long run. A passing implementation smoke alone is not a pose gate: require the
declared 2.5 mm held-height p95 and 9 mm relative-XY axis p95 on every selection
round. Only after that can transition model/value/MPPI work resume. The earlier
frozen-DP spatial candidates still fail these gates; this implementation has
not yet established pose quality, planner performance, or improved real
robustness.
