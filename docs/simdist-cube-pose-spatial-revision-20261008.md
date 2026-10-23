# Bounded spatial cube-pose revision — 2026-10-08

## Decision and scope

Sol high approves a bounded **frozen spatial-feature attention** experiment.
The v2 frozen DP post-projection LayerNorm MLP fails the pose gate: selection
held-height p95 34.3 mm and relative XY p95 91/208 mm, versus 2.5/9 mm.
The same development split's q-only control failed at held-height p95 51.3 mm
and relative XY p95 76/190 mm. These comparisons do not establish image
localization or identify global pooling as the cause of the failure.
Exact DP feature parity rules out the tested transfer implementation mismatch;
it does not establish that the features preserve cube location, that the
objects are visible in the crops, or that the dataset is sufficient.

First distinguish the global projection/pooling bottleneck from inadequate
spatial features or data. Do not fine-tune the full camera backbone or start a
transition/planner sweep at this stage. H2 reset seeds 1900–1903 stay untouched.
This revision changes the estimator only; it makes no robustness claim.

## Model and contracts

`--representation spatial` retains the frozen, independent DP front/wrist
ResNet trunks and deterministic **eval** camera transforms. Camera mapping
is resolved from the loaded DP camera keys (normally front=DP index 1,
wrist=DP index 0), with output order explicitly front/wrist. uint8 RGB is cropped/resized first,
then divided by 255 and normalized by the exact DP ImageNet mean/std.
Layer4 maps before global average pooling are `[B,512,7,7]` for ResNet18 and
224px inputs. The source DP projections and LayerNorm are bypassed.
`ResnetEncoder.forward_spatial` adds no parameters/checkpoint keys; weights
remain the exact strictly loaded source DP weights. Nominal input stride is
32 pixels. The frozen wrapper shares the source camera modules and forces
eval mode, even when `.train()` is called on it; it retains no action model.

Each camera gets a trainable 1x1 convolution producing two spatial attention
maps. Spatial softmax produces weighted 512-D content and two cell-centre
coordinates in [-1,1] per map. Concatenating two maps per camera and q6 gives
`4*(512+2)+6=2062` inputs to MLP `2062→128→64→6`. Attention convolutions start
at zero, i.e. uniform spatial attention. The attention maps have no keypoint
labels and must not be described as identified cube pixels or calibrated 3D
points. The metric XYZ outputs are supervised directly.
The coordinate convention is `u=2*(column+0.5)/width-1` (right positive) and
`v=2*(row+0.5)/height-1` (down positive) after each DP transform. For 7x7 maps,
the corner centers are ±6/7 and uniform attention gives (0,0). These are map
cell coordinates, not physical camera rays or the exact ResNet receptive
field centers. The default front crop is the central 224px of 240x320 RGB;
the wrist view is resized to 224x224. A learned MLP maps camera descriptors
and measured q to robot-root-relative XYZ; there is no geometric
triangulation or claim that camera (u,v) equals robot (X,Y).

`--representation spatial-mean` is the matched control: identical frozen maps,
MLP capacity and initial state, with attention fixed uniform. It uses global
average pre-projection content and neutral soft coordinates. Its prediction
starts exactly equal to the spatial model, as tested. Therefore:

- spatial-mean vs v2 tests bypassing the learned DP compression and changing
  pose-head capacity; it does not isolate those two changes individually;
- spatial vs spatial-mean tests learning spatial selection/content pooling
  while matching the pose MLP and frozen trunk;
- `--vision-mode none` is a q-only control. No camera content enters that
  forward pass. It should be compared on the same seed split and norm stats.

Only current RGB/current measured q are inputs. Cube labels, actual successor
state, phase, success, camera/task simulator metadata and objective clock do
not enter the estimator. q and per-coordinate XYZ statistics are fit-only,
with the existing 1e-3 floors (radians for q, metres for XYZ); pose outputs
are unscaled to metres for reporting by `pose = normalized*pose_std+pose_mean`.
Labels remain robot-root-translation-relative and ordered held/fixed:
`[held_x,held_y,held_z,fixed_x,fixed_y,fixed_z]`. The collector subtracts
robot-root world translation from simulator world XYZ; it does not apply an
inverse root rotation. Thus axes retain simulator world alignment. This is
the dataset's frame convention, and would need a new contract for a rotated
robot-root pose. Stats are derived only from fit rows, never calibration or
selection. `CubePoseNormalization` exposes the same affine transforms.
The temporal label and finite-horizon value contracts are unchanged.

Frozen maps are retained as **float32 CPU chunks**, with an LRU byte limit
set by `--spatial-cache-mib` (default 2048 MiB). For 8,083 retained frames,
two 512x7x7 float32 maps take about 1.62 GB decimal; they fit the default.
The trainer probes the actual map shape and sums all fit/calibration/selection
retained lengths, then prefills sequentially only if the complete dataset fits
within that cap. In this case ResNet inference is performed once per chunk;
subsequent epochs and image-swap diagnostics reuse the cached maps. A smaller
or zero cache streams on demand and can re-encode evicted chunks; zero holds
no maps. No maps are written to disk or packed into duplicate contiguous
dataset arrays. Only the requested training batch is copied to the head's
device; q/targets and the six-coordinate predictions are small arrays.

The bound covers retained feature tensor bytes. Additional memory includes
one decoded RGB episode, one extraction chunk, one requested CPU/device
batch, ResNet activations, head gradients, tensor objects and optimizer state.
The pickle format still requires decoding one whole source episode in RAM.
Checkpoint/report metadata and stderr diagnostics expose cache limit/current/
peak bytes, expected complete-dataset bytes, hits/misses, extraction calls,
re-encoded chunks, encoded frames and episode loads. These make accidental
cache thrashing visible. No numerical precision reduction is used by the new
stream. Model checkpoints record representation, feature shape/precision,
transform/map/frame contracts, DP SHA/source, fit-only stats and seed split.
Existing pooled
default and old head/state keys remain available. To reload the new head use
`CubeSpatialPoseHead(feature_shape[0], attention=representation == 'spatial')`
and restore normalization plus the matching frozen DP map extractor; do not
load this checkpoint into the old `CubePoseHead` or WM planner.

## Fixed development comparison

Use the current 32 episodes only:

- fit: 27000–27003, 27100–27101, all 24 episodes;
- calibration/epoch selection: 27102, all 4 episodes;
- development selection: 27103, all 4 episodes.

27103 was inspected for v2 and is **development**, not an untouched test.
Do not change splits, drop failed trajectories or pick the variant/epoch from
H2. All variants select an epoch using calibration normalized MSE only. Use
40 epochs, seed 20261008, batch 64 and learning rate 1e-3 for the spatial pair.
Compare the existing v2 plus q-only as reference; if a new pooled reference is
run, use the same 40-epoch budget. There is no requirement to run more seeds
or increase collection before this diagnostic is reviewed.

Run these two commands sequentially (root/Luna owns execution):

```bash
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v3_spatial_mean.pt \
  --device cuda --feature-batch-size 8 --train-batch-size 64 --epochs 40 --spatial-cache-mib 2048 \
  --seed 20261008 --learning-rate 1e-3 --representation spatial-mean \
  --fit-seeds 27000 27001 27002 27003 27100 27101 \
  --calibration-seeds 27102 --selection-seeds 27103

conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v3_spatial.pt \
  --device cuda --feature-batch-size 8 --train-batch-size 64 --epochs 40 --spatial-cache-mib 2048 \
  --seed 20261008 --learning-rate 1e-3 --representation spatial \
  --fit-seeds 27000 27001 27002 27003 27100 27101 \
  --calibration-seeds 27102 --selection-seeds 27103
```

For the q-only reference, replace representation with `pooled`, add
`--vision-mode none` and use a separate output. Do not run the spatial models
in parallel on the occupied GPU. The root filesystem remains approximately
3.3 GB free; tmpfs had 7.1 GB free when inspected.

## Interpretation and gates

Report fit/calibration/selection height and relative XY p95, moving/static and
phase counts/errors, fit-mean reference and the image-swap diagnostic. The
script swaps jointly aligned front/wrist features across selection samples
while keeping q/targets fixed, using seed+1 and no fitting. Worsening under
swap suggests image use; a small change can reflect q shortcuts, a weak
estimator, similar scenes or redundant views. It is not causal proof of cube
localization. Current aggregate p95 metrics weight frames, with only one
selection reset seed; do not interpret them as independent-window confidence.

The existing applicability gate remains held-height p95 <2.5 mm and each
relative XY p95 <9 mm, with moving/phase strata reported. These are predeclared
engineering margins against the task thresholds, not a proven sufficient
condition for successful control. **No planner if the gate fails.**

Discriminating outcomes:

- If spatial substantially improves fit and calibration/selection over the
  matched mean control, spatial selection is useful. A pose-gate failure
  still means no transition/planner; retain this revision for further design.
- If fit is accurate but selection remains poor, prioritize reset/view/object
  coverage, crop visibility and occlusion/history diagnostics. Unfreezing
  the whole backbone could worsen the observed generalization gap.
- If both spatial variants fit poorly despite useful image-swap sensitivity,
  a bounded layer4 fine-tuning or higher-resolution layer3 spatial head is a
  subsequent hypothesis, not established by this test. Check visibility first.
- If the mean control improves most, DP global compression/head capacity is
  implicated; do not claim learned spatial keypoints were necessary.
- If q-only is competitive and image-swap barely changes errors, test a
  visual-only control and visibility before declaring pose information absent.

Do not weaken the metric gate to accommodate results. DP's front centre crop
can hide objects; RGB occlusions can make single-frame recovery ambiguous.
Neither condition is removed by this attention head. A causal history
revision or explicit projected-pixel supervision would require a separately
reviewed schema/calibration contract. Geometry reward/FK parity is still a
separate blocker for the later planner.

## Implementation validation (Sol high)

```bash
/home/keti/miniforge3/envs/so101/bin/python -m pytest tests/test_simdist_cube_pose.py tests/test_simdist_cube_pose_spatial_contract.py -q
/home/keti/miniforge3/envs/so101/bin/python -m py_compile src/so101/learning/dp/vision.py src/so101/learning/simdist/cube_pose.py src/so101/learning/simdist/cube_pose_spatial.py src/so101/learning/simdist/cube_pose_features.py scripts/train_simdist_cube_pose.py tests/test_simdist_cube_pose_spatial_contract.py
/home/keti/miniforge3/envs/so101/bin/python scripts/train_simdist_cube_pose.py --help
```

Focused CPU tests passed **18/18**, covering existing head shapes, matched
uniform initial prediction, attention gradients, spatial-permutation
discrimination, FP16-compatible head shapes, exact ResNet pre-pool-average
parity, both source camera orders with real 224px transforms/unchanged weights,
frozen-trunk backpropagation, image-coordinate and XYZ-normalization
conventions, exclusion of privileged inputs, retained-horizon/source-order
indexing, cache eviction/zero-cache behavior and full-cache reuse across
shuffled passes. Four tiny one-epoch CPU trainer fixtures exercise pooled,
pooled q-only, spatial and spatial-mean checkpoint/report wiring with a camera
stub; they validate fit-only statistics and horizon truncation. These
predictions are not accuracy evidence. Help and py_compile exit 0.

The earlier CPU-only real-checkpoint wiring probe loaded the exact
DP checkpoint and seed-27000 file
`/dev/shm/so101-object-pose-dev16/trajectory_000000.pkl`, using objective-frame
indices `[0,99,199,299]`. Before the streaming revision, calling the then-current
`_extract_features(dp, four_frame_episode, torch.device('cpu'), 2, 'spatial')`
produced finite float16 maps
`[4,2,512,7,7]`. Two `CubeSpatialPoseHead(512)` models with shared state and
attention on/off gave maximum initial prediction difference **0**; the spatial
model has **274,762** trainable parameters. This was forward-only, with no
pose accuracy interpretation. That earlier helper signature is superseded;
spatial mode now uses `FrozenDPSpatialEncoder` and `SpatialFeatureStream`.
No Isaac, GPU training, real-data estimator training or hardware actuation was
run in this revision; root/Luna owns any bounded experiment. Passing these contracts
does not establish visual pose accuracy.

## Follow-up: reset-round coverage review and collection

After the matched comparison ran, Sol high reviewed both outcomes. Learned
spatial attention greatly improves fit over the uniform spatial-mean control,
but neither estimator approaches the engineering gate on seed 27103. Spatial
fit p95 was 9.43 mm held height and 16.74/25.56 mm relative XY; selection was
36.01 mm and 66.94/318.32 mm. Spatial-mean selection was 33.55 mm and
59.99/301.99 mm. Image swap worsened both models, but that can create an OOD
image/q pairing and does not prove cube localization. The fit failure means
this cannot be attributed to data coverage alone.

Sol high approved one fixed-architecture coverage experiment before transition
or planner work:

- Fit uses all existing eight reset rounds, including the consumed development
  selection round 27103, plus new fit rounds 27200–27201 (40 episodes total).
- Calibration/epoch choice uses new rounds 27202–27203 (8 episodes).
- The new selection is rounds 27204–27207 (16 episodes), held untouched until
  both spatial and spatial-mean candidates are fixed and reported together.
- Keep the frozen spatial candidate and matched uniform mean control, 40
  epochs, seed 20261008, batch 64, learning rate 1e-3, perturbation std 0.04,
  hold 8, four environments, all diverse outcomes, and the unchanged 2.5/9 mm
  pose gate. Keep H2 1900–1903 untouched. Planner remains NO-GO if the pose gate
  fails. If this selection fails, stop collecting indiscriminately and move to
  a separately reviewed visibility, resolution, or causal-history ablation.

The tmpfs headroom cannot hold all 32 new raw pickle episodes alongside the
existing 32. New data are collected as four sequential 8-episode batches and
packed losslessly into separate `aligned.npz` shards; the redundant generated
pickles are removed only after each shard is written and inspected. These
archives preserve current RGB, joint/pose labels, actual successors, terminal
flags, phase, finite-horizon, geometry and provenance arrays. They do not carry
per-frame `successes`, so trainer reports those counts as unavailable; terminal
success remains separately available from `terminated`.

The pose trainer now accepts mixed pickle directories and `--data-archive`
shards. For archives, current front/wrist RGB arrays are decoded into host RAM;
the existing bounded map cache still stores only the extracted float32 maps.
A CPU trainer fixture covering archive input, fit-only normalization, group
split and full-cache prefill passed. This adds a host-memory consideration for
the full run; check free memory before starting and monitor the first model's
cache report (`full_dataset_fits=true`, `reencoded_chunks=0`).

Collection commands (run sequentially; each batch packs before the next):

```bash
conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.04 \
  --perturbation-hold 8 --num-envs 4 --num-episodes 8 --max-rounds 2 --seed 27200 \
  --out /dev/shm/so101-object-pose-coverage27200-27201
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-object-pose-coverage27200-27201 \
  --out /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz

conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.04 \
  --perturbation-hold 8 --num-envs 4 --num-episodes 8 --max-rounds 2 --seed 27202 \
  --out /dev/shm/so101-object-pose-coverage27202-27203
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-object-pose-coverage27202-27203 \
  --out /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz

conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.04 \
  --perturbation-hold 8 --num-envs 4 --num-episodes 8 --max-rounds 2 --seed 27204 \
  --out /dev/shm/so101-object-pose-coverage27204-27205
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-object-pose-coverage27204-27205 \
  --out /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz

conda run --no-capture-output -n so101 /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse --perturbation-std 0.04 \
  --perturbation-hold 8 --num-envs 4 --num-episodes 8 --max-rounds 2 --seed 27206 \
  --out /dev/shm/so101-object-pose-coverage27206-27207
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-object-pose-coverage27206-27207 \
  --out /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz
```

Before the two sequential GPU training commands below, remove only those four
batches' `trajectory_*.pkl` files after verifying their NPZ shards and free up
the temporary collection workspace. Never remove the earlier experiment data.
The trainer's archive-input change was syntax-checked and its focused tests
include a 1-epoch CPU fixture; no full GPU fit has run yet.

```bash
# Spatial-mean control; run first, then spatial with the same split/options.
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --data-archive /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz \
    /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz \
    /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz \
    /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v4_coverage_mean.pt \
  --device cuda --feature-batch-size 8 --train-batch-size 64 --epochs 40 \
  --spatial-cache-mib 4096 --seed 20261008 --learning-rate 1e-3 \
  --representation spatial-mean \
  --fit-seeds 27000 27001 27002 27003 27100 27101 27102 27103 27200 27201 \
  --calibration-seeds 27202 27203 --selection-seeds 27204 27205 27206 27207

conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --data-archive /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz \
    /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz \
    /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz \
    /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v4_coverage_spatial.pt \
  --device cuda --feature-batch-size 8 --train-batch-size 64 --epochs 40 \
  --spatial-cache-mib 4096 --seed 20261008 --learning-rate 1e-3 \
  --representation spatial \
  --fit-seeds 27000 27001 27002 27003 27100 27101 27102 27103 27200 27201 \
  --calibration-seeds 27202 27203 --selection-seeds 27204 27205 27206 27207
```

## Coverage-run audit and stop decision

All four collection batches completed and were packed: 32 new episodes,
8,431 finite-horizon transitions over rounds 27200–27207. The archive-only
trainer input smoke and the focused SimDist suite passed. On the fixed
spatial-mean run, the cache fully prefetched 16,514 frames into
3,314,425,856 bytes under the 4096 MiB limit with 0 re-encoded chunks. The
checkpoint/report are in tmpfs as `cube_pose_v4_coverage_mean.pt/.json`.
Calibration selected epoch 37. Its all-fit / calibration / four-round
selection held-height p95 values were 1520.6 / 1018.8 / 3161.6 mm; selection
relative-XY p95 was 208.2/445.6 mm. The unchanged gate fails. These catastrophic
metrics are a data-regime diagnostic, not evidence that the visual architecture
is intrinsically this inaccurate.

The read-only shard audit found exactly two gross-coordinate episodes among all
64 old+new episodes, both from fit seed 27200: env 2 and env 3. Env 2 first
left table XY at tick 89 and crossed below table support at tick 92; its maximum
absolute robot-relative cube coordinate is 244.24 m. Env 3 left XY at 185 and
crossed support at 187; maximum is 73.06 m. At 30 Hz the tail z fits have
estimated downward accelerations 9.80996 and 9.81006 m/s², with z RMSE 0.066
and 0.009 mm. This supports ballistic free fall after leaving the tabletop;
it argues against labeling the trajectories a simulator explosion. The table
collider spans 0.70×1.20 m. The DR backdrop floor has no collider/rigid body,
so a cube can continue falling after leaving the table.

Both failure trajectories were retained. RGB inspection around the events
shows the moving cube at the edge of the raw front image before disappearing;
the actual DP front centre crop omits it at those frames. The wrist crops are
blank or filled by a close object surface near one event. These are visual
diagnostics only: the archives have no per-frame DR camera pose/intrinsics or
segmentation, so a projection cannot certify exact visibility or occlusion.
Raw/cropped montage:
`/dev/shm/so101-object-pose-dev16/coverage27200_dp_crops.png`.

Sol high reviewed the contamination and explicitly stopped the spatial
full-corpus run. It was interrupted during cache prefill before any checkpoint
or outcome; there is no spatial v4 result. Do not compare v4 spatial-mean with
v3 spatial or treat this v4 mean result as an architecture comparison. The
remaining work is a diagnostic decomposition, then a separately reviewed
state-conditioned model contract. Do not delete failed episodes or relabel
unobservable metric targets as valid poses.

Recommended next model contract, pending implementation review: current
front/wrist RGB plus measured q predicts an object work-region/lost-state logit
and a conditional metric XYZ pose. Simulator-derived validity is a training
target only. Fit-only pose normalization and coordinate loss use valid GT pose
rows; q normalization uses all fit rows. Apply finite neutral placeholders
before loss for invalid coordinates, with explicit masks, never clip large
coordinates into invented poses. Use masked XYZ MSE over valid coordinates
plus region BCE over every transition. Preserve all raw episodes/actions,
rewards, termination and finite-horizon values. Report the existing 2.5/9 mm
gate over the GT-defined valid region together with full-corpus state
classification and coverage; never compute the gate only on model-predicted
valid frames. Workspace validity and camera visibility/occlusion require
separate labels/contracts. H2 and planner stay NO-GO pending that review, reward
parity and lost-state uncertainty handling.

Run the physical and pose-distribution diagnostics without editing the source
archives:

```bash
conda run --no-capture-output -n so101 python scripts/audit_simdist_cube_pose_coverage.py \
  --data /dev/shm/so101-object-pose-dev16/aligned.npz \
    /dev/shm/so101-object-pose-dev32-extra/aligned.npz \
    /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz \
    /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz \
    /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz \
    /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz \
  --out /dev/shm/so101-object-pose-dev16/coverage_pose_audit.json

conda run --no-capture-output -n so101 python scripts/evaluate_simdist_cube_pose_regions.py \
  --checkpoint /dev/shm/so101-object-pose-dev16/cube_pose_v4_coverage_mean.pt \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --fit-data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --data-archive /dev/shm/so101-object-pose-coverage27200-27201/aligned.npz \
    /dev/shm/so101-object-pose-coverage27202-27203/aligned.npz \
    /dev/shm/so101-object-pose-coverage27204-27205/aligned.npz \
    /dev/shm/so101-object-pose-coverage27206-27207/aligned.npz \
  --fit-seeds 27000 27001 27002 27003 27100 27101 27102 27103 27200 27201 \
  --selection-seeds 27204 27205 27206 27207 \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v4_coverage_mean_region_eval.json \
  --device cuda --batch-size 8
```

The regional checkpoint audit completed. In the 10,276 fit rows, held-cube Z
normalization had mean `-1.8793 m`, std `15.8454 m`; in the 9,950-row prefix
before the first held-cube table-support exit, mean/std were `0.0578/0.0307 m`.
The prefix's held-cube XYZ p01/p50/p99 were approximately
`(0.170,-0.204,0.043) / (0.271,-0.013,0.043) / (0.435,0.165,0.189) m`.
Thus only 326 fit rows shifted Z normalization by orders of magnitude. All
4,335 new selection frames are in this diagnostic region, and the contaminated
mean candidate still scored 3161.6 mm held-Z p95 and 208.2/445.6 mm relative XY
p95 there. This separates two contributors: the lost tails are not in selection,
but their unmasked loss/fit-only std distort the learned map and metric scaling.
This is a diagnostic partition, not yet a model training mask.

Physical, crop and regional evaluation artifacts:

- `/dev/shm/so101-object-pose-dev16/coverage_pose_audit.json`
- `/dev/shm/so101-object-pose-dev16/coverage27200_dp_crops.png`
- `/dev/shm/so101-object-pose-dev16/cube_pose_v4_coverage_mean_region_eval.json`

The main selection of the next implementation is now the Sol-reviewed
two-output state model: a GT-supervised task-region/lost-state logit for every
transition, plus conditional XYZ trained only on GT-valid work-region poses.
Use fit-only pose stats over valid rows, q stats over all fit rows, and masked
XYZ loss with a finite neutral placeholder before arithmetic. Preserve all
failed episodes and their prefix transitions and all original reward/terminal/
finite-horizon targets. Do not introduce camera-visible labels from nominal
projection: actual DR camera pose/K is absent. State validity and visibility
must stay separate. Sol high review is required before implementing this head
or any train-mask contract. H2 and planner remain NO-GO.

### Sol high region-head contract and camera-geometry smoke (2026-10-08)

Sol high finalized the bounded target domain and loss/evaluation contract. For
each object independently (held, fixed), region membership is the inclusive
robot-root-translation-relative simulator-world-axis box
`x,y ∈ [-1,1] m`, `z ∈ [h_table-0.10,h_table+1.00] m`; current version uses
`h_table=0.0300814467 m`. This is a broad model domain, not reachability,
visibility, success, termination, or an absorbing state. Outside-region
successors remain ordinary finite-horizon transitions. The source/h/version
and bounds must be serialized with each checkpoint. Nonfinite source labels
are corrupt data and fail closed; they must not be silently labeled outside.

The model emits normalized XYZ for both objects plus two independent inside-
region logits. Fit q mean/std on all fit rows; fit each object's XYZ mean/std
only on that object's in-region fit rows (std floor 1 mm; reject a fit split
with zero valid rows for either object). Replace invalid XYZ with finite neutral
placeholders before normalization/arithmetic, mask their coordinate loss, and
use per-object masked XYZ means with fixed object weights (an empty minibatch
mask contributes zero for that object). Region BCE uses every fit row and
fit-only inverse-frequency class weights; if a class is absent, its weight is
zero and that classifier direction is explicitly untested. Frozen loss is
`L_xyz + 0.1 L_region`; no oversampling. Select epochs using calibration
masked-XYZ MSE, not classification. Fix the region decision threshold at 0.5.

Evaluation denominators are GT-defined: held-Z p95 over every GT-inside held
frame; relative-XY p95 over every frame where both GT objects are inside; each
object's XYZ RMSE over its own GT-inside rows; region confusion over every
retained current frame. Report counts, per-episode/round/phase/motion strata,
inside FNR and outside recall, with zero denominators reported as NA. Never
mask pose metrics by predicted region. Preserve the existing 2.5 mm held-Z
and 9 mm per-axis relative-XY development gate; require each of four new
selection rounds to pass, and predeclare ≥99% inside coverage. Outside recall
≥95% is an engineering gate only if there are at least three outside episodes
across at least two reset rounds; otherwise that direction is inconclusive.
The consumed 27204–27207 rounds remain development data. Confirmatory evidence
still requires a frozen candidate and untouched H2 later; nothing here is a
deployment-robustness claim.

Collector camera metadata was added and packed in a one-episode Isaac smoke:
`/dev/shm/so101-camera-geometry-smoke/aligned.npz` (300 in-horizon transitions;
the raw collector had 310 including post-horizon success tail). Camera order,
array dimensions, finiteness, intrinsics, and packer round-trip passed. But
both recorded camera poses were bitwise constant across all frames while q
changed by as much as 0.141 rad. IsaacLab source inspection identified the
cause: `update_latest_camera_pose` defaults to false. The camera config now
enables it. A rerun (seed 27213) packed 120 in-horizon transitions; q changed
by up to 0.1215 rad, the front pose stayed static as expected, and wrist pose
changed by up to 0.04949 in quaternion components and 0.02005 m in position.
The corrected archive is `/dev/shm/so101-camera-geometry-smoke-pose-refresh/aligned.npz`.
The earlier seed-27212 archive remains preserved but its pose is stale and
must not be used. Intrinsics were finite and constant per sensor. Updated
camera poses enable projection diagnostics, not visibility/occlusion labels
without segmentation. They are not model inputs. No camera visibility
diagnosis or wrist-only causal claim is a prerequisite.

Focused validation after adding the camera fields: `py_compile` passed for the
collector, packer, and pose trainer; `PYTHONPATH=. conda run --no-capture-output -n so101 pytest -q tests/test_simdist_dataset_prep.py tests/test_simdist_cube_pose.py tests/test_simdist_cube_pose_spatial_contract.py` passed 34 tests. Both the original stale-pose smoke and corrected pose-refresh smoke exited and packed successfully. Region-conditioned head wiring and contract tests are in progress; no long GPU training has started.

Fixed-box support audit over the available trajectory corpus found every fixed
cube frame inside. Held cube frames are inside except for 317 frames from two
episodes in reset seed 27200, one reset round. All recorded pre/post-success
frames in other rounds are inside. The available data therefore supports both
classes for exploratory fitting, but does not meet the ≥3 outside episodes / ≥2
round requirement for an outside-recall gate; report it inconclusive. No
supported grasp/lift/stack phase was excluded by the fixed box in this corpus.
