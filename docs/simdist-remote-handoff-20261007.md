# SO101 SimDist implementation handoff — testbed2

## Authoritative user direction

### Model-specific work split (2026-10-07)

This task is staged; this note does not imply that a model switch already
happened. **Luna (medium)** owns environment/checkpoint inventory, the
specification-bounded diverse collector, its tests, and small training-command
execution where the installed environment permits it. **Sol high** (recommended
GPT-6.1 Sol, high reasoning) must review and implement the harder design work:
visual world-model wiring, JAX/PyTorch boundary, time-axis and normalization
contracts, reward/value semantics, and MPPI correctness/debugging. Do not start
long GPU training until that review resolves the listed contracts below. Passing
syntax or runtime checks alone is not evidence that the research method is
semantically correct.

Handoff items for Sol high: choose whether latent dynamics predicts
next-observation state, image features, or both; define exact sequence indices
for current observation, applied action, actual next observation, reward,
terminal/truncation, and V(s'); define gamma/tick semantics against the 30 Hz
simulator and ~27 Hz real controller; audit normalization of six radians joint
targets and RGB inputs; preserve fixed-network ResiP export boundary (JAX export
only, PyTorch execution); then verify constrained candidate action clipping and
rate limits inside every MPPI rollout, sent-action history, warm start, and plan
latency. The native PushT code is proprioception-only and is not a drop-in SO101
visual model.

Implement a SimDist-based world-model and planning baseline for SO101 StackCube on the remote Linux/GPU machine `testbed2`, using the existing `so101-team` checkout after verifying its actual path and contents.

Verified remote checkout: `/home/keti/workspace/research/so101-team`, registered in Orca as a folder project. `.git` exists inside. The earlier `/home/workspace/...` path omitted `keti`. Existing teacher run: `sweep/gamma99/so101-StackCube-v0/resip/resip_sd042_20260924_210715`, with `params_2000.pkl` and `agent_config.json` verified. Use this gamma99 StackCube teacher rather than retraining; verify matching normalization/config and epoch before export or rollout. Remote code lives at repo root (agents/, scripts/, src/, online.py), not under sim2real/.

Primary question: **Does diverse simulation distillation + planning improve real SO101 robustness compared with the existing successful-demonstration-based ResiP teacher → image DP student?**

The user suspects wrist perception contributes more than dynamics to current failures. Nevertheless, the user explicitly wants to apply the method first, measure whether performance improves, and analyze causes afterward. Do not require perception root-cause diagnosis before implementation. Diverse state/action trajectories plus rendering randomization are the intended robustness mechanism; this remains a hypothesis, not a guaranteed outcome.

Earlier feasibility-review recommendations requiring perception diagnosis before starting are superseded by this direction. Data timing, action semantics and planner runtime checks remain necessary engineering constraints. Learned-importance G is a later extension, not the first implementation milestone.

## Existing pipeline and evidence

- Local reviewed repo: `so101-imitation`, branch `sim2real`. Remote folder name may differ. Verify remote Git identity, branch, dirty state and recent commits before editing. Preserve existing work; do not reset, overwrite, or auto-pull a dirty checkout.
- Relevant code is under `sim2real/` in the local repo; an older remote checkout may instead have the same files at its root. Inspect rather than assuming layout.
- Existing ResiP state teacher has privileged 36-D cube/EEF/joint state. It generates rendered synthetic trajectories for an image + 6-D measured-joint Diffusion Policy student. The deployed real policy is this student, not directly ResiP or DPPO.
- Existing records report teacher 91.6% sim success and DP student 11/23 real success. These numbers were not rerun during local review.
- Adjacent local reference sources: `reference/simdist`, `reference/simdist_learned_importance_handoff.md`; latest PushT implementation and evidence are in `research/simdist/simdist-dppo-pusht`, not only the older `reference/Research/simdist-dppo-pusht`.
- PushT documented full DPPO data → world model → native MPPI execution. Final 100 seeded episodes: MC-value MPPI coverage 0.567 / success 65%, DPPO 0.467 / 51%. Success is coverage-based, not SO101 stacking success. Raw per-episode results were not located in local review. This is no proof of real adaptation or G efficacy.
- Native SimDist source: https://sim-dist.github.io/ and https://arxiv.org/html/2603.15759v2 . Inspect first-party source/version if local references are missing. Never assume Mac absolute paths exist on Linux.

## First action: inventory, then build

Locate applicable AGENTS.md and read repo guidance. Check remote Git branch/status, GPU availability/usage, Isaac Sim/Lab and JAX/Flax/PyTorch environments, teacher checkpoints, DP checkpoints, calibration files, synthetic/real datasets and reference repositories. Record discovered paths and missing prerequisites in a short Markdown note. Use existing working environments; do not upgrade globally or disrupt unrelated GPU jobs.

Proceed autonomously with implementation and meaningful smoke tests. If full-scale training cannot run due to missing checkpoints/data or occupied hardware, complete independent code work and report the exact blocker; do not invent experiment results. No physical robot actuation is authorized by this handoff. Prepare dry-run/evaluation entry points; real execution remains operator-controlled.

## Implementation scope

1. **Diverse sim collector:** reuse ResiP teacher, calibrated Isaac Lab scene, camera pipeline and rendering DR. Save expert, suboptimal/perturbed, failed and recovery trajectories rather than filtering for success. Apply bounded temporally coherent perturbations in simulation, and include camera-pose/light/color variation across these trajectories. Log quality/phase distributions.
2. **Data contract:** record measured joints, front/wrist RGB, actual applied 6-D absolute joint targets in radians, dense environment reward, expert flags, values, real next observations and termination/truncation separately. Keep episode boundaries and frame timing. Never treat a duplicated final observation as a real dynamics transition. Mask padded targets.
3. **Value labels:** start with simulator reward Monte-Carlo return-to-go; ensure reward, tick, gamma, success-hold and terminal/timeout semantics are explicit and consistent. The ResiP critic is conditioned on normalized state + base action; current PyTorch teacher export contains base/actor only, so do not assume a state-only critic API. MC labels do not guarantee learned value OOD safety.
4. **SO101 visual world model:** retain SimDist's latent dynamics + reward/value/policy heads and action/history conditioning. Add real-observable RGB/joint encoder; Go2 height-map and PushT proprio-only encoders are insufficient. Privileged simulator state may provide labels and diagnostics, not deployed observations. Reuse existing vision preprocessing where appropriate; do not assume the student's encoder transfers directly without training.
5. **Constrained MPPI baseline:** use native SimDist policy-head proposals first. Existing image DP stays a comparison baseline; using its proposals is an optional separately named hybrid. Respect calibrated joint order, units, limits and previous-command rate limiting inside candidate trajectories, not only after selection. Include sent actions in history. Benchmark warm-started planner latency and avoid unbounded search outside data coverage.
6. **Simulation proof:** run collector/data/model smoke checks and comparable teacher / image DP / WM policy / reward-only MPPI / MC-value MPPI evaluation on shared seeds/layouts. Document budget, configuration, actual outcomes and runtime. Start small, then scale only after contract and learning sanity checks pass.
7. **Real evaluation preparation:** support untrimmed success/failure recordings and operator-run planner dry-run/control entry points. Existing `EpisodeRecorder.trim_idle()` deletes internal pause frames; do not train fixed-step dynamics across those gaps. For old files, reject windows spanning timestamp gaps. Current real control is about 27 Hz vs sim 30 Hz, with servo lag; encode or align the actual timing. Report camera/frame freshness and plan age limitations.
8. **Later milestones:** only after baseline compare no adaptation vs uniform dynamics-only real adaptation, then G importance weighting. Do not spend the first milestone implementing G.

## Native adaptation constraints for later work

Freeze encoder/history representation, action embeddings, reward/value/policy and normalizers according to native parameter filters; optimize native `dynamics` only. Zero head loss coefficients alone do not establish frozen parameters. Verify parameter equality before/after updates.

G handoff corrections: batch-wide scalar loss scaling can largely cancel under Adam; choose weighted sampling, multiple candidate batches combined in one update, or per-sequence loss weighting. Match utility-target inner optimizer to actual adaptation or explicitly validate surrogate correlation. Use episode/session-level adapt/meta-validation/untouched-test splits; random overlapping-window splits leak information. Track utility drift after theta changes and count meta-data/target generation in data and compute budgets.

## Definition of completion for initial remote work

Deliver implementation for diverse collector, dataset conversion, visual world model, simulation planner/evaluation and real dry-run preparation; run available contract/learning/control smoke checks. Provide reproducible commands, dependency/environment selection, small-budget results and exact remaining full-scale/real operator steps. Do not declare improved real robustness until a controlled real comparison exists. Preserve original DP pipeline and default behavior. Do not push, merge, publish or actuate real hardware without further instruction.

Respond in Korean. Keep technical identifiers unchanged. Report evidence and uncertainty directly. Save this handoff in the remote repo before work so subsequent remote sessions can resume.

## 2026-10-07 staged implementation status

The requested role split was executed as a staged review: Luna completed
inventory, bounded diverse collection/data preparation and smoke runs; Sol high
implemented the visual WM/training/evaluation/dry-run modules and reviewed the
contracts. This records task ownership, not a claim that a model transition
occurred. Implementation evidence and exact commands are in
[`simdist-implementation-progress-20261007.md`](simdist-implementation-progress-20261007.md)
and [`simdist-sol-contract-20261007.md`](simdist-sol-contract-20261007.md).

A successful collector episode now has a true terminal only after stable-success
hold and post-success tail. Failure episodes are retained as truncated. The
bounded combined fixture had 714 transitions: 114 terminal-success transitions
with valid MC values and 600 failed/truncated transitions with censored MC
values. This is a serious value-training selection-bias blocker: do not start
long GPU training or interpret MC-value MPPI until a principled finite-horizon /
remaining-time target or justified timeout bootstrap is selected, implemented,
and calibrated. A two-step CPU training run and two-action-per-arm Isaac eval
were wiring smokes only; all three evaluator arms had 0/1 success, so no
performance conclusion is supported. The test suite passed 14 checks.

Existing DP checkpoints were located under `outputs/dp_train/` (including
`cotrain_v1/dp_so101.pt` and `cotrain_v3_slow2/dp_so101.pt`), but no matched
teacher/DP/world-model shared-seed comparison was run. No robot actuation,
push, or merge was performed. G remains a later milestone; wrist-perception
root-cause work is not an entry condition.

## Latest verification update (2026-10-07)

The finite-horizon v2 review selected a 300-tick, 30 Hz finite task objective
with `remaining_ticks` input. Full horizon failures now have defined MC values;
early collector cutoffs remain censored. Exact semantics, implementation/tests,
strict checkpoint schema and open assumptions are in
[`simdist-finite-horizon-validation-20261007.md`](simdist-finite-horizon-validation-20261007.md).

A 20-episode dataset (4,659 transitions; 1,196 expert actions and 3,463
perturbed) was collected and trained for 2,000 steps at 64px. Its episode-disjoint
audit found value MAE/RMSE 58.99/107.68 and latent prediction worse than
persistence at all horizons 1–8. The checkpoint failed the research-quality
gate and is not suitable evidence that WM/MPPI improves performance. The paired
five-seed DR pilot had ResiP 4/5, existing image DP 5/5, and corrected WM policy,
reward-only MPPI and MC-value MPPI each 0/5. The value and dynamics model failed
calibration/prediction gates, so those WM outcomes are diagnostic only. Details,
latencies, reproducible commands, and preprocessing correction are in the
implementation progress note and Sol v2 validation note.

The first WM paired run used an inconsistent input resize and is explicitly
excluded. A corrected rerun replayed exactly equal cube layouts and render
randomization parameters. The final offline dry-run generated five unactioned
plans and reported camera freshness unknown. Full checks passed: 30 regression
tests, script compile, `git diff --check`, and evaluator help. The SimDist task
did not launch a robot controller, push, or merge; no real robustness claim was
made. The next gate is to diagnose/improve
held-out latent dynamics against an episode-bootstrap, time-aware persistence
baseline before retraining or rerunning the policy comparison. Do not start
further model training until that acceptance protocol and failure analysis are
ready. G remains later; wrist-perception root-cause work is not prerequisite.

### Episode-bootstrap follow-up (2026-10-07; final review 2026-10-08)

The follow-up protocol is now implemented and run against 32 fresh holdout
episodes. Seed-disjoint episode-macro bootstrap results are saved at
`/tmp/so101-simdist-holdout/audit_bootstrap.json` and detailed in
[`simdist-finite-horizon-validation-20261007.md`](simdist-finite-horizon-validation-20261007.md).
The frozen checkpoint fails all eight horizons: learned/persistence latent MSE
ratios range from 28.20 (horizon 1) to 3.13 (horizon 8), with every 95% CI above
1. A second paired bootstrap clustered by the 8 shared round seeds also has
every CI lower bound above 1. Do not retrain this checkpoint for MPPI or claim
dynamics validation. The bootstrap implementation has 42 relevant passing tests
and fail-closed provenance checks matching training pickle fingerprints to the
checkpoint source NPZ. Sol high approved the checkpoint FAIL and episode-macro
calculation, then completed a final review of the strengthened implementation.
No code review blocker remains; the performance gate still fails, and failure
analysis is required before long GPU training.

### Frozen-checkpoint root-cause probes (2026-10-08)

The fixed checkpoint has a confirmed output-geometry contributor: final decoder
LayerNorm affine output, consumed without an unconstrained projection, cannot
represent target latent row mean/scale variation. A conservative geometric MSE
lower bound is above time-aware persistence on independent holdout horizons
1–5. The checkpoint also fails on train episodes, so seed shift alone does not
explain the failure. Action conditioning has measurable but weak utility;
history repetition slightly improves the probe. Online target drift/shared
encoder gradient conflict remains a plausible but unverified training cause;
one final snapshot cannot establish historical drift. Exact numbers, derivation,
commands, confidence limits, and the recommended separated ablations are in
[`simdist-dynamics-root-cause-20261008.md`](simdist-dynamics-root-cause-20261008.md).
Do not use this checkpoint for MPPI training/evidence. A post-hoc shared affine
projection fit on 64 train windows modestly reduced latent error, but remained
3.55–21.72x worse than persistence across all holdout horizons. Output freedom
alone is insufficient. A new named variant must improve the sequence model and
pass train/validation probes plus the full episode-bootstrap persistence gate
before long training or planner comparison. No robot
actuation, push, merge, or retraining was performed during this investigation.

**Real-evaluation operator note:** a separate interactive terminal ran
`deploy_policy_real.py` while this task was doing offline work. The assistant
mistakenly sent interrupt/termination signals because of the task's no-actuation
constraint. The user confirmed ownership; the process was not restarted. Whether
it sent commands or saved a complete record is unknown. Do not signal or restart
operator-owned real-evaluation processes; coordinate with the operator first.

### Joint-conditioned latent follow-up (2026-10-08)

The bounded joint-conditioned path has been wired through training, inference,
and MPPI candidate expansion. It predicts normalized future measured joints
from current history/actions, combines those with current front/wrist features,
and uses exact successor remaining-time values. Future images and joints are
labels only. The path is mutually exclusive with the separate residual latent
predictor. Sol high reviewed the definition as coherent for an 8-tick
short-horizon baseline and emphasized that scoring gradients can change the
previously calibrated joint forecaster.

The 2,000-step run and train-only joint calibration are documented in
[`simdist-applicability-progress-20261008.md`](simdist-applicability-progress-20261008.md).
On the reused v2 holdout, calibrated joint prediction passes all horizons under
both episode and round-seed bootstrap. The joint-conditioned latent branch
still fails at horizons 1–3, is inconclusive at 4–6, and passes only 7–8. Since
this holdout informed model exploration, retain exploratory status. The
current-image feature persistence is a known representational limitation for
moving cubes/contact; an action-conditioned future visual-feature head is the
next design proposal pending Sol high review. No planner or real robustness
claim follows from the joint audit.

### Model-role handoff and current applicability status (2026-10-08)

The requested role split is:

- **Luna medium:** environment/repository inventory, exact checkpoint and data
  provenance, bounded collectors with fixed contracts, focused tests and
  bounded training/evaluation commands.
- **Sol high (GPT-6.1 Sol, high reasoning recommended):** visual world-model
  architecture, JAX/PyTorch boundary, temporal/normalization/reward/value
  contracts, planner debugging, and approval of expensive experiments.

This records requested ownership and is not evidence that a model transition
occurred. The current work did not trigger a model switch. User constraints
remain: do not start from checkpoint retraining, do not require wrist-perception
root-cause analysis, keep G importance weighting as later work, and do not run
real actuation or push/merge.

The strongest existing checkpoint remains
`/tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_36ep_v1_calibrated_scores.pt`.
Its 64-episode v4–v7 pooled latent audit passed all eight horizons under both
episode and reset-seed bootstrap. It still failed score quality and planner
performance. Full 300-tick paired runs on seeds 1000–1002 had image DP 2/3,
WM policy 0/3, native reward/MC MPPI 0/3, and DP-proposal reward/MC MPPI 0/3.
Hybrid latency was about 58 ms mean and 148 ms p95. Thus latent prediction
alone did not make the model applicable.

Sol high reviewed a 52-episode development follow-up using original 36 episodes
plus the full v7 outcomes. Reset-seed split now keeps parallel environment
episodes together and reserves separate validation and calibration seeds.
Joint residual calibration precedes reward/value affine calibration; both use
calibration episodes excluded from optimizer updates. The updated scripts and
tests are in the workspace. The 2,000-step candidate passed H1's all-outcome
8-horizon latent and joint bootstrap gates, but score diagnostics remained
weak. H1 reward MSE was 2.14–3.61x a training-only time-conditioned mean on
early-truncated outcomes; value MAE/RMSE were about 76.6/115.5 and did not beat
the mean baseline with confidence.

Full 300-tick H1 paired Isaac trials (seeds 1800–1803, rendering randomization)
showed image DP 3/4 vs DP-proposal reward MPPI 0/4 at `.25 rad` trust radius;
at `.05 rad`, DP was 4/4 vs MPPI 0/4. Hybrid mean action latency was 41.7 ms
(p95 131.5 ms), above the 30 Hz budget. A one-seed, one-candidate passthrough
smoke succeeded for both DP and MPPI, showing the interface can preserve DP
behavior when no score-selected refinement is applied. The learned scorer still
assigns positive gain to candidates while the refined controller fails. See
[`simdist-applicability-progress-20261008.md`](simdist-applicability-progress-20261008.md)
and the H1 artifacts under `/tmp/so101-simdist-residual-20261008/`.

### Next Sol high review item

The current forecast predicts measured q and combines it with **fixed current
camera features**. It cannot represent future cube displacement, contact, or
occlusion. The next candidate should estimate object/gripper geometry from
current RGB, using simulator pose only as a training target, then predict
action-conditioned future object state. Future privileged pose and future RGB
remain target-only; inference receives current RGB, measured q/action history,
candidate joint actions and exact remaining ticks. Sol high should approve the
object-state coordinates, normalization, temporal alignment, reward derivation,
and held-out score/ranking gates before implementation or longer training.

Do not use v4–v7 or H1 as a new checkpoint's untouched test: they already
informed model development. H2 (16 diverse episodes, a disjoint reset-seed
range) is the reserved final test and should be collected only after the next
candidate is frozen. Before another planner sweep require, independently on H2:
all eight latent and joint bootstrap upper bounds below 1; object-state error
better than persistence; reward/value MSE ratio upper confidence bounds below
1 against a train-only remaining-time mean baseline by outcome; zero-boundary
checks; and bounded branch tests showing that predicted return ranks actual
simulator returns. Then run paired full-horizon DP/ResiP/planner comparisons
with task randomization and per-action latency. A simulator pass permits a
carefully coordinated real-evaluation proposal; it is not a real robustness
claim. No real controller, push, merge, or long training has been run in this
follow-up.

### Reproducibility commands for the 52-episode candidate

```bash
conda run --no-capture-output -n so101 python scripts/merge_simdist_datasets.py \
  --inputs /tmp/so101-simdist-study/aligned_v2_36ep.npz \
           /tmp/so101-simdist-holdout-v7/aligned.npz \
  --out /tmp/so101-simdist-study/aligned_dev_52ep_v1.npz

conda run --no-capture-output -n so101 python scripts/train_simdist.py \
  --data /tmp/so101-simdist-study/aligned_dev_52ep_v1.npz \
  --out /tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1 \
  --steps 2000 --batch-size 2 --device cuda --seed 20261008 \
  --validation-fraction 0.15 --calibration-fraction 0.15 \
  --joint-dynamics --joint-conditioned-latents \
  --init-dp-vision outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --freeze-transferred-vision

conda run --no-capture-output -n so101 python scripts/calibrate_simdist_joint_residual.py \
  --checkpoint /tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1/worldmodel.pt \
  --training-data /tmp/so101-simdist-study/aligned_dev_52ep_v1.npz \
  --out /tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1_calibrated.pt \
  --device cuda --batch-size 8

conda run --no-capture-output -n so101 python scripts/calibrate_simdist_score_heads.py \
  --checkpoint /tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1_calibrated.pt \
  --training-data /tmp/so101-simdist-study/aligned_dev_52ep_v1.npz \
  --out /tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1_calibrated_scores.pt \
  --device cuda --batch-size 8

conda run --no-capture-output -n so101 python -m pytest -q \
  tests/test_simdist_model_planner.py tests/test_simdist_dynamics_audit.py \
  tests/test_simdist_dataset_merge.py
```
### Object-state collector contract and Isaac smoke (2026-10-08)

To unblock the next object-centric experiment, `scripts/generate_teacher_demos.py`
now records two simulator-only label fields per applied transition:
`cube_positions_robot_relative_m` and
`next_cube_positions_robot_relative_m`, each `[N, 2, 3]`. The first is captured
from the current teacher observation before action; the second is captured from
the actual simulator state immediately after `env.step(action)`. The packed
archive preserves both fields. The packer rejects partial labels, undeclared
frame/semantics, invalid shapes, and mixing labeled and unlabeled trajectories.
These labels are targets for training only; inference inputs remain current
camera images and measured joints. The terminal successor is explicitly
captured, never copied from the terminal current state or from another episode.

Sol high confirmed the 36-D teacher-state indexing: held cube XYZ `[4:7]`, fixed
cube XYZ `[11:14]`, robot-relative after `teacher_observation` subtracts root
translation; measured q `[24:30]`. For legacy data, a next row may serve as a
successor only when the next measured q equals `next_observations` within that
same episode. The terminal row has no legacy object successor. The new collector
avoids this ambiguity.

Single-episode Isaac smoke command:

```bash
conda run --no-capture-output -n so101 \
  /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse \
  --perturbation-std 0.04 --num-envs 1 --num-episodes 1 --max-rounds 1 \
  --seed 26008 --out /dev/shm/so101-object-label-smoke
```

This produced 1 failed episode with 300 transitions. In the raw file,
current-label rows exactly matched teacher-state XYZ slices, every interior
successor matched the next current pose (maximum absolute discrepancy 0), and
the explicitly captured final successor was finite. The packed archive retained
all 300 current/successor labels. The final pose differed from current by only
`2.98e-8 m`, as expected for the static held object on this failed trajectory;
that is not a fabricated terminal duplicate. Python syntax checks and
`tests/test_simdist_dataset_prep.py` passed (10 tests).

This smoke establishes timestamp/frame/terminal-label wiring only. It does not
evaluate vision pose accuracy, contact-state forecasting, or task performance.
Next model design remains with Sol high review: visual RGB-to-two-cube XYZ
estimator trained on simulator targets; then causal action-conditioned object
transition model; use predicted q and exact FK for reward terms. Exact reward
reconstruction needs both cube positions, table-plane height, sizes `.025/.04 m`,
and all three grasp points (midpoint plus both fingertips). A midpoint alone is
not enough for the distance reward. First compare geometric reward recomputed
from ground-truth simulator state against recorded dense rewards; only then
train predictions.

Predeclared object-state gates from Sol high review: (1) recomputed geometric
reward from ground-truth cube XYZ and q/FK agrees with stored dense reward to
about `1e-4`; (2) on independent reset rounds, current-RGB cube height p95 error
is below `2.5 mm` and each relative XY p95 below `9 mm`, reported by visibility
and task phase; (3) for the held cube, action-conditioned horizons 1–8 beat
true-pose persistence with episode and reset-seed bootstrap upper ratio below 1
on all data and a moving-object stratum; the static fixed cube is judged by
absolute error because its zero persistence error makes a ratio meaningless;
(4) RGB-initialized rollout passes the same comparison against inferred-pose
persistence with no future labels in inputs; (5) reward ordering and latency
pass before planner evaluation. Evaluate reward-only first. Compare MPPI
argmax-candidate against elite-mean action on identical candidates, since the
existing one-candidate success does not isolate score ranking from averaging.
Keep final H2 seeds 1900–1903 untouched until the candidate and thresholds are
frozen. Root filesystem free space is about 3.3 GB; avoid large collection or
training until capacity is available.

Follow-up collector run after adding explicit cube order metadata used the same
command with `--seed 26009 --out /dev/shm/so101-object-label-smoke-v2`. It
recorded 144 transitions, nine transient success frames, and correctly marked
the episode truncated because the final stable-success hold was not satisfied.
The successor/current continuity maximum was 0, order was
`[held_cube, fixed_cube]`, all labels packed, and the terminal successor was
finite. The independent regression set now passes 46 tests across dataset
packing, model/planner, dynamics audits and dataset merge. This is still only
label plumbing evidence. The smoke also shows why “success at least once” must
not be confused with a completed stable-success episode.

Across the two 1-episode wiring smokes, held-cube successor displacement was
0–51 mm (first failed rollout) and 0–38 mm (second rollout with transient
success); fixed-cube maximum displacement was 3.9 mm and 0.84 mm. These tiny,
single-reset samples only show the labels span static and moving-object cases;
they are far too small and correlated to train or validate a pose estimator.

### First RGB cube-pose estimator: two bounded candidates (2026-10-08)

Sol high approved a diagnostic estimator: frozen image-DP front and wrist
post-projection LayerNorm features (128-D each) plus current measured q (6-D),
then MLP `262→128→64→6` predicting held/fixed robot-root-relative XYZ. RGB,
normalization, q and pose statistics are fitted only on fit reset groups;
privileged labels, future poses, clock, success and phase are not model inputs.
Calibration chooses an epoch, selection is evaluated once. The DP feature
source is step 100000 at
`outputs/dp_train/cotrain_v3_slow2/dp_so101.pt`, SHA-256
`031c58ab56382614f56e28c7b741aefb9c4cdc5431618a33b03de540bab719`; it was
trained with real and simulated images. Transferred camera feature parity was
exact on eight images per camera (max absolute error `0`).

First development collection: 16 episodes / 4 reset-round seeds
27000–27003 / 3,807 finite-horizon transitions; 8 episodes had at least one
success frame and 6 reached stable terminal success. Group split was fit
27000–27001 (8 episodes), calibration 27002 (4), selection 27003 (4). Its
checkpoint was `/dev/shm/so101-object-pose-dev16/cube_pose_v1.pt`; report was
`cube_pose_v1.json`. The original run accidentally included 11 post-horizon
tail frames in pose fitting. The script was corrected to use only ticks inside
the declared 300-tick objective and fit-mean baselines were added; no split was
changed.

The corrected v1 candidate selected epoch 7. On the single selection seed,
held-cube height absolute error p95 was `42.3 mm`; relative XY p95 was
`106/212 mm` per axis, versus predeclared limits `2.5 mm` and `9 mm`. It failed.

Second development collection: another 16 episodes / reset seeds 27100–27103;
all four outcomes per round were retained. With the earlier 16 episodes this is
32 episodes / 8,083 finite-horizon transitions. The candidate split was fit
seeds 27000–27003 and 27100–27101 (24 episodes), calibration 27102 (4), and
selection 27103 (4). It selected epoch 17. Checkpoint and report:
`/dev/shm/so101-object-pose-dev16/cube_pose_v2.pt` and `cube_pose_v2.json`.
Selection held-height p95 was `34.3 mm`; relative XY p95 was `91/208 mm`.
The fit-mean baseline had held-height p95 `44.6 mm` and relative XY p95
`72/337 mm`. Both miss the predeclared pose gate. This is a development
diagnostic with one reset seed for selection, not a population estimate.

Exact commands:

```bash
conda run --no-capture-output -n so101 \
  /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse \
  --perturbation-std 0.04 --perturbation-hold 8 --num-envs 4 \
  --num-episodes 16 --max-rounds 4 --seed 27000 \
  --out /dev/shm/so101-object-pose-dev16
conda run --no-capture-output -n so101 \
  /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse \
  --perturbation-std 0.04 --perturbation-hold 8 --num-envs 4 \
  --num-episodes 16 --max-rounds 4 --seed 27100 \
  --out /dev/shm/so101-object-pose-dev32-extra
conda run --no-capture-output -n so101 python scripts/validate_simdist_dp_vision_transfer.py \
  --checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --data /dev/shm/so101-object-pose-dev16/aligned.npz --device cuda
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_v2.pt --device cuda \
  --feature-batch-size 8 --train-batch-size 256 --epochs 120 \
  --fit-seeds 27000 27001 27002 27003 27100 27101 \
  --calibration-seeds 27102 --selection-seeds 27103
```

The data does contain moving held cubes; however, the frozen DP feature + MLP
pose head has not met millimetre-scale accuracy. It modestly improves some
fit-mean metrics while failing the gate. Do not train a transition model or
planner from these predictions. Sol high review is requested for the next
discriminating estimator change, likely a spatial/localization representation
or bounded camera fine-tuning; final H2 seeds 1900–1903 stay untouched.
Exact reward/FK planner remains NO-GO pending simulator geometry metadata and
ground-truth reward parity. Pose module/data focused tests pass (13 checks).

### Pose-head ablation and next estimator revision

A q-only control on the same 32-episode split (zeroed both DP camera feature
vectors, same MLP/training procedure) also failed selection: held-height p95
`51.3 mm`, relative XY p95 `76/190 mm`. Compared with the v2 camera+q head
(`34.3 mm`, `91/208 mm`), the images help held height but do not improve both
relative XY axes; the result does not establish whether pooling discarded
localization information or the seed coverage is too small. The fit-mean
baseline on selection was `44.6 mm`, `72/337 mm`. Keep this as a development
diagnostic only; prior selection seeds have now been consumed for model
selection.

The q-only command was:

```bash
conda run --no-capture-output -n so101 python scripts/train_simdist_cube_pose.py \
  --data-dir /dev/shm/so101-object-pose-dev16 /dev/shm/so101-object-pose-dev32-extra \
  --dp-checkpoint outputs/dp_train/cotrain_v3_slow2/dp_so101.pt \
  --out /dev/shm/so101-object-pose-dev16/cube_pose_q_only.pt --device cuda \
  --feature-batch-size 8 --train-batch-size 256 --epochs 120 \
  --fit-seeds 27000 27001 27002 27003 27100 27101 \
  --calibration-seeds 27102 --selection-seeds 27103 --vision-mode none
```

Sol high selected a frozen spatial-feature ablation next: use DP ResNet spatial
maps before global average pooling/projection/LayerNorm, then train an object
attention or soft-coordinate head for the two XYZ labels. Keep v2 global pooled
and q-only as comparisons. This cleanly tests whether global pooling removes
image localization signal while avoiding end-to-end backbone changes in the
first discrimination. New split and data must preserve whole reset-round
groups; H2 1900–1903 remains untouched. H2/planner and reward/FK remain NO-GO
until the pose gate and geometric reward parity pass. The spatial architecture
implementation is delegated to the requested Sol high role; no GPU training
has started for it.

### Ground-truth geometry reward parity (2026-10-08)

The diverse collector now additionally records post-action simulator grasp
midpoint/fixed fingertip/moving fingertip in robot-root-relative XYZ, the
robot-frame table height, held/fixed cube sizes and all StackCube reward
thresholds/weights. The packer validates label order, dimensions, finite values,
and identical task constants. A pure NumPy reward function independently
recomputes the reward from the actual successor cube and grasp geometry.

One Isaac diverse failure episode (seed 26010, 300 transitions) was packed and
audited. Recomputed reward matched every recorded dense reward: max absolute
error `1.12e-8`, MAE `2.41e-9`, 300/300 labels valid; success count was zero in
both. This validates simulator geometry/frame/reward timing using simulator
grasp points. It does **not** yet validate q-to-fingertip FK: the recorded
fingertips are privileged simulator labels. Planner remains NO-GO until an
independent FK from measured q reproduces all three points and this same reward
parity. Command and result artifact:

```bash
conda run --no-capture-output -n so101 \
  /home/keti/workspace/reference/IsaacLab/isaaclab.sh -p \
  scripts/generate_teacher_demos.py --headless --diverse \
  --perturbation-std 0.04 --num-envs 1 --num-episodes 1 --max-rounds 1 \
  --seed 26010 --out /dev/shm/so101-geometry-reward-smoke
conda run --no-capture-output -n so101 python scripts/prepare_simdist_dataset.py \
  --input /dev/shm/so101-geometry-reward-smoke --gamma 0.99 \
  --out /dev/shm/so101-geometry-reward-smoke/aligned.npz
conda run --no-capture-output -n so101 python scripts/audit_simdist_geometry_reward.py \
  --data /dev/shm/so101-geometry-reward-smoke/aligned.npz \
  --out /dev/shm/so101-geometry-reward-smoke/reward_audit.json
```

The geometry reward module, collector extension, packer checks, and pose module
focused tests pass 16 checks. Existing model/planner tests were not rerun after
the latest collector-only geometry changes.

### Spatial pose results and Sol high coverage decision (2026-10-08)

The Sol-reviewed frozen spatial-attention and matched spatial-mean runs both
completed on the development split. Both miss the fixed pose gate. Spatial fit
p95 is 9.43 mm held height and 16.74/25.56 mm relative XY; selection is
36.01 mm and 66.94/318.32 mm. Spatial-mean selection is 33.55 mm and
59.99/301.99 mm. Spatial attention helps fit but does not generalize to the
single selection reset round. Image-swap error worsens, but this is not proof
of object localization. Planner stays NO-GO; H2 1900–1903 stays untouched.

Sol high reviewed and approved one fixed-architecture coverage experiment:
all eight previous reset rounds (including consumed 27103) plus 27200–27201
for fit; 27202–27203 for calibration/epoch choice; 27204–27207 as a new 16-episode
selection. Keep both spatial and spatial-mean models, 40 epochs, seed
20261008, batch 64, learning rate 1e-3, all diverse outcomes, and the 2.5/9 mm
gate unchanged. Do not begin transition/planner evaluation unless the new pose
gate passes. The full reviewed commands and data limits are in
`docs/simdist-cube-pose-spatial-revision-20261008.md`.

New rounds are collected in 8-episode batches and losslessly packed into
`aligned.npz` shards because `/dev/shm` cannot hold the original pickles beside
all new data. The pose trainer accepts these archives alongside old pickle
directories. It decodes current RGB into host RAM; cached spatial maps remain
bounded by the declared 4096 MiB limit. Aligned archives do not retain transient
per-frame success flags, so such counts report unavailable instead of being
inferred from terminal flags.

Role boundary from the user: Luna medium owns environment/data checks, bounded
collector implementation, focused testing and command execution. Sol high
owns visual world-model structure, JAX/PyTorch interface, temporal and
normalization/reward/value contracts, and MPPI debugging. This is a model/task
assignment; it does not mean a model switch has already occurred. Long GPU
training requires the reviewed core contracts to be explicit, and execution
success alone is not research validation. No real robot actuation, push or
merge is authorized.

### Coverage anomaly diagnosis and next contract (2026-10-08)

The bounded 32-episode collection completed (8,431 finite-horizon transitions).
The fixed spatial-mean fit completed but is not a valid architecture comparison:
two seed-27200 fit episodes leave tabletop support and free-fall out of camera.
The same-corpus spatial run was interrupted during cache prefill after Sol high
reviewed the anomaly; no spatial checkpoint was produced. Across all 64 old
and new episodes, only those two cross the descriptive 1 m coordinate sentinel.

The tabletop has a 0.70×1.20 m collider; the DR backdrop floor has no collision
or rigid body. The two episodes leave table XY at ticks 89 and 185 and cross
below support at 92 and 187. Their post-event z fits estimate downward gravity
at 9.80996/9.81006 m/s² with under 0.1 mm fit RMSE. This supports ordinary
free-fall after the cube leaves the table, not a simulator explosion. The
front raw image briefly shows a cube at the edge, then the DP centre crop
omits it; wrist crops do not reliably show it. Because actual DR camera pose,
intrinsics and segmentation were not recorded, exact visibility remains
unknown.

The two episodes' 326 lost-tail fit frames are enough to shift held-Z fit
normalization from region-prefix mean/std `0.0578/0.0307 m` to full-fit
`-1.8793/15.8454 m`. Every frame in new selection is in the diagnostic table
region, but the contaminated spatial-mean candidate still has p95 errors
3161.6 mm held height and 208.2/445.6 mm relative XY. The candidate fails;
do not interpret it as architecture evidence. Per-episode physics audit and
checkpoint region evaluation commands/results are in
`docs/simdist-cube-pose-spatial-revision-20261008.md`.

Sol high's next proposed head is current front/wrist RGB + measured q to (1)
task-region/lost-state logits and (2) conditional XYZ. GT region labels are
training targets only; valid GT pose rows define pose statistics and masked XYZ
loss, while all fit rows define q stats and all transitions contribute region
BCE. Preserve failure episodes/prefixes and existing reward, terminal and
finite-horizon values. Do not clip large coordinates into fake poses, use
predicted-valid-only metrics, or conflate camera visibility with task-state
validity. This state/logit and mask contract still needs Sol high design review
before implementation. H2 and planner remain NO-GO.

### 2026-10-08 continuation: finalized region contract and camera pose refresh

Sol high finalized the object-wise region-conditioned pose-head contract in
`docs/simdist-cube-pose-spatial-revision-20261008.md`. Region is inclusive
robot-root-relative world-axis `x,y∈[-1,1] m`, `z∈[h_table−0.10,h_table+1.00] m`
with versioned `h_table=0.0300814467 m`; held/fixed labels are independent.
The head predicts `[B,2,3]` normalized XYZ and `[B,2]` independent region
logits. q statistics use every fit row; object pose statistics/loss use only
that object's GT-inside rows; region BCE uses all fit rows. Empty minibatch
masks, absent classes, fixed 0.5 threshold, GT-defined evaluation denominators,
2.5/9 mm pose gate, ≥99% in-region coverage, and inconclusive outside recall
without ≥3 episodes / ≥2 reset rounds are specified in the spatial revision.
No invalid successor becomes terminal or zero-value. H2 remains untouched;
27204–27207 are consumed development selection, never confirmatory evidence.

The collector camera geometry smoke initially showed stale constant poses.
Source inspection found IsaacLab `CameraCfg.update_latest_camera_pose=False` by
default; `src/so101/scenes/tabletop.py` now enables it. The rerun with seed
27213 and its packed archive `/dev/shm/so101-camera-geometry-smoke-pose-refresh/aligned.npz`
contains 120 in-horizon transitions: q max adjacent change 0.1215 rad, front
pose remains static as expected, and wrist pose max adjacent change is
0.04949 (position 0.02005 m). The original 27212 archive is retained but its
poses are explicitly stale and unusable. Camera intrinsics remain finite.
Camera poses are now suitable for geometry inspection, but do not imply
segmentation/visibility labels. No wrist-only causal claim is required.

Focused tests after camera metadata collection/packing passed 34 tests; the
collector/packer/trainer passed `py_compile`. Region-conditioned head wiring
and its contract tests are being completed by the Sol high review session;
no long GPU training has started. Next: finish the opt-in head/loss/evaluator,
run boundary/successor/empty-mask/absent-class tests and report train-split
class/episode support, then Sol high review before bounded GPU training. Do not
train if supported grasp/lift/stack states fall outside the fixed broad box.

A fixed-box support audit over all available trajectory archives found all fixed
cube rows inside. Held-cube outside rows are 317 frames from two episodes in
reset seed 27200 (one round). Thus the available data can fit both classes,
but cannot establish outside recall under the preregistered minimum of three
outside episodes across two rounds; that metric/gate is inconclusive. All
recorded pre/post-success phases in the other rounds remain in-region, so no
supported grasp/lift/stack sample is excluded by the fixed box.

The Sol-reviewed region-conditioned estimator was wired and CPU contract-smoked,
then run for the bounded 40-epoch spatial-mean and learned-spatial development
candidates. Both fail the GT-domain pose gate on all four consumed selection
rounds (mean: held-Z p95 20.268 mm, relative-XY 72.703/209.974 mm; spatial:
24.286 mm, 75.746/125.365 mm). Inside coverage was 100%; outside recall is
inconclusive (held has two outside episodes in one round, fixed has no outside
examples). Cache prefill succeeded for all 16,514 frames with zero re-encoding.
No planner/MPPI, H2, or real robot evaluation followed. Detailed run contract,
commands and artifacts are in
`docs/simdist-cube-pose-region-implementation-20261008.md`.

Current next blocker is visual pose accuracy. The frozen DP spatial representation
is not an acceptable pose front-end, so no transition/world-model planner
training is justified yet. Corrected per-frame wrist pose now changes with
joints; actual poses still do not provide occlusion truth without segmentation.
Sol high is reviewing an explicit image-supervision route (e.g. instance masks
or 2D geometry targets), with multi-view sensing allowed but no wrist-only
causal diagnosis requirement. Preserve H2 and hardware restrictions.

The next pixel-supervision collector reached one successful path-resolved
instance-mask smoke (seed 27214, 300 transitions). `idToLabels` maps both
cube descendant prim roots; front/wrist frame counters are synchronized and
contiguous 2–301; RGB/mask overlays align. Packed uint8 masks consume 46.08 MB
per 300 frames and the full aligned archive is 79.23 MB. Front GT projection
centres fall inside mask bboxes on all 300 frames, but wrist pose/K projection
is inconsistent (center-in-bbox 72.6% held, 12.5% fixed); use wrist masks as
pixel targets only and defer wrist geometry fusion. This does not require a
wrist-only causal diagnosis.

The collector must read `camera.frame` directly. A trial using `camera.data.frame`
re-triggered sensor updates and ran >10 minutes; interrupted output contained
only generation metadata. With the direct frame accessor, the final smoke
rollout completed in 0.4 minutes. Audit command/results and artifacts are in
`docs/simdist-cube-pose-pixel-supervision-design-20261008.md`.

### Pixel pose implementation update (2026-10-08)

Sol high implemented opt-in `--representation pixel`: full RGB + measured q,
frozen checkpoint ResNet conv1..layer3, small FPN, conditional XYZ/region
outputs, both-camera mask auxiliaries, and front-only projected-centre target.
GT annotations stay loss labels; no mask/K/pose input leakage. Source-resolution
packed masks use big-endian unpack plus nearest-exact resizing to 240×320. The
trainer retains all GT-inside 3D loss/evaluation rows (including occlusion and
absence), selects epochs on calibration masked XYZ, and streams early maps per
minibatch without a dataset-wide map cache. The unused layer4 and DP actor stay
on CPU; they are not retained in the pixel model/GPU graph. Aux loss weights can
be zeroed for the matched no-aux control.

Validation: actual seed27214 raw pickle and aligned NPZ CPU loader smoke matched
q/XYZ/inside/centre/source-mask-count labels exactly; RGB `[3,3,240,320]`, mask
`[3,2,2,240,320]`, feature cache 0. Frozen early maps matched expected stages
64×60×80, 128×30×40, 256×15×20. Sol high reports 49/49 focused tests after
memory changes; independent repository run passed **102 tests in 5.38s**.
`py_compile`, CLI `--help`, and `git diff --check` passed. No GPU training,
Isaac collection, H2, planner, or hardware actions were run in this stage.

This only establishes an implementation/data seam. There is one annotated
300-frame development episode, insufficient split support; root disk has about
3.2 GB free and `/dev/shm` about 3.0 GB free. Bulk collection and long training
remain blocked on capacity plus multi-round fit/calibration/selection data. The
new FPN still must pass the fixed per-selection-round pose gate (held height p95
≤2.5 mm and relative XY axis p95 ≤9 mm) after Sol high reviews first-batch GPU
memory/throughput. Until then no WM transition/value or MPPI training is
justified, and no real robustness claim is supported. Prior v5 spatial results
remain failures. Continue preserving H2 seeds 1900–1903 and never actuate the
physical SO101.
