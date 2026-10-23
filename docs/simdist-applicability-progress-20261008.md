# SO101 SimDist applicability progress — 2026-10-08

## Current finding

There is evidence that action-conditioned SO101 joint dynamics can generalize
under the diverse simulator perturbations. The calibrated joint residual model
passed the round-seed clustered episode bootstrap on an independent 32-episode
holdout for all eight horizons. This is a physical-state prediction result;
it does not establish object/contact prediction or MPPI task performance.

The joint-conditioned latent branch forms each successor representation from
the predicted normalized joint state, current front/wrist image features, and
the exact remaining-tick clock. It does not consume future observation labels.
Sol high reviewed this as a coherent short-horizon experimental baseline and
flagged that reward/value/latent gradients can change the joint predictor. Thus
joint residual calibration must be rerun on every final checkpoint.

For the 2,000-step joint-conditioned run, the joint residual scale was fit from
3,427 training windows only. On the previously explored 32-episode v2 holdout,
the calibrated joint MSE ratios against repeating current measured joints were
`[0.876, 0.814, 0.735, 0.626, 0.623, 0.598, 0.587, 0.528]`; all episode and
round-seed clustered intervals were below 1. The visual latent ratios against
time-aware persistence were `[1.162, 1.218, 1.189, 1.071, 1.034, 0.983, 0.864,
0.848]`. Horizons 1–3 fail; 4–6 are inconclusive; only 7–8 pass. The v2 holdout
was used to explore multiple model variants, so these numbers are exploratory,
not confirmatory.

The current visual branch holds camera features fixed. Its short-horizon latent
failure is consistent with missing object/camera feature dynamics, but does not
prove that cause. A visual-feature dynamics head is the next candidate after
Sol high review. Any such head must use current frames and candidate actions at
inference, and future images only as training targets. Its success must be
tested against time-aware persistence on a new untouched Isaac holdout before
planner evaluation.

## Exact bounded experiment commands

```bash
conda run --no-capture-output -n so101 python -m pytest -q \
  tests/test_simdist_model_planner.py tests/test_simdist_dynamics_audit.py

conda run --no-capture-output -n so101 python scripts/train_simdist.py \
  --data /tmp/so101-simdist-study/aligned.npz \
  --out /tmp/so101-simdist-residual-20261008/train_joint_conditioned_v1 \
  --steps 2000 --batch-size 4 --device cuda --seed 20261008 \
  --joint-dynamics --joint-conditioned-latents

conda run --no-capture-output -n so101 python scripts/calibrate_simdist_joint_residual.py \
  --checkpoint /tmp/so101-simdist-residual-20261008/train_joint_conditioned_v1/worldmodel.pt \
  --training-data /tmp/so101-simdist-study/aligned.npz \
  --out /tmp/so101-simdist-residual-20261008/train_joint_conditioned_v1_calibrated.pt \
  --device cuda --batch-size 4

conda run --no-capture-output -n so101 python scripts/audit_simdist_dynamics_bootstrap.py \
  --checkpoint /tmp/so101-simdist-residual-20261008/train_joint_conditioned_v1_calibrated.pt \
  --data /tmp/so101-simdist-holdout-v2/aligned.npz \
  --training-episode-dirs /tmp/so101-simdist-study/expert \
    /tmp/so101-simdist-study/perturbed_004 /tmp/so101-simdist-study/perturbed_012 \
  --out /tmp/so101-simdist-residual-20261008/joint_conditioned_holdout_v2_calibrated_exploratory.json \
  --batch-size 4 --device cuda --bootstrap-repetitions 10000 --bootstrap-seed 20261008
```

## Smoke results and blockers

- Focused semantic/audit tests: 32 passed. The first MPPI K-candidate run caught
  a singleton task-clock batch mismatch; `predict()` now explicitly broadcasts
  the current clock across candidate batches and rejects other mismatches.
- The 2k run completed with validation loss 1.5221; this scalar is only a smoke
  metric and is not a model acceptance criterion.
- `py_compile` passed for model, MPPI and training script.
- GPU was idle before the bounded run. Disk had 13 GB free (98% used); avoid
  creating extra full-size image datasets or duplicate checkpoints.
- No real robot policy, hardware actuation, push, or merge was run.
- The model still lacks a latent predictor that beats persistence at short
  horizons. No WM policy/MPPI success or real robustness result is claimed.

## Sol high review items

1. Review the proposed action-conditioned visual-feature delta head and feature
   auxiliary target contract before implementation.
2. If end-to-end scoring losses are allowed to update the joint forecaster,
   repeat joint calibration and the complete joint audit on the final model.
3. Keep v2 exploratory; collect a new seed-disjoint Isaac holdout for any
   confirmatory latent-dynamics claim.
4. Do not call this full visual world-model validation: the present forecast
   preserves current camera features and cannot represent cube motion,
   contact, or occlusion.

## DP-vision joint-conditioned checkpoint and pooled holdout (2026-10-08)

The current strongest prediction result is the 2,000-step DP-vision initialized
joint-conditioned model at
`/tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_36ep_v1_calibrated_scores.pt`.
It was trained on 36 episodes (9,091 transitions), initialized from the
existing ResNet18 DP camera encoder whose pretraining includes real and sim
data, then calibrated in the order joint residual → reward/value scores.
That initialization must be disclosed in any transfer claim.

On the pooled v4–v7 confirmation set (64 episodes, 16 reset seeds), joint
conditioned latent MSE beat time-aware persistence at all 8 horizons with
episode-bootstrap and reset-seed-clustered 95% upper bounds below 1. The
episode-bootstrap ratios were `.944, .895, .915, .876, .783, .767, .761, .750`;
the reset-seed-clustered ratios were `.944, .895, .915, .876, .783, .767, .761,
.750`. Calibrated joint prediction also beat the repeat-current-q baseline at
all horizons. Provenance and seed-disjoint checks passed. This confirms short
horizon camera-conditioned latent forecast for that checkpoint, not object
motion, contact prediction, planner benefit, or real robustness. v4–v7 have now
informed model selection and are no longer clean test data for a new checkpoint.

The same model does **not** currently supply a useful MPPI score. External
score diagnostics show reward MAE/RMSE `.748/2.06` on v4 and `1.346/3.33` on v7;
value MAE/RMSE `48.30/99.59` on v4 and `76.22/130.57` on v7. On v7 the true
value mean/std were `100.35/162.18`, while prediction mean/std were
`62.79/88.96`. The development set has few positive stable-success trajectories
relative to full-horizon failures. The value target is specifically
`V^beta`, return under the collector continuation-policy mixture; it is not a
DP- or MPPI-specific continuation value. Aggregate affine calibration cannot
establish candidate ranking.

Full-horizon paired simulator runs on seeds 1000–1002 remain negative: DP
baseline 2/3, WM policy 0/3, reward MPPI 0/3, MC-value MPPI 0/3, DP-proposal
reward MPPI 0/3 and DP-proposal MC MPPI 0/3. DP-proposal MPPI averaged about
58 ms per action (p95 about 148 ms); WM-native MPPI averaged about 41 ms
(p95 about 42.5 ms), both above the 33.3 ms 30 Hz budget. Results and artifacts
are in `/tmp/so101-simdist-residual-20261008/paired_eval_dp_proposal_mppi_3seeds.json`
and the pooled-audit JSON named `dpvision_joint_holdout_v4_v5_v6_v7_pooled_confirmatory.json`.
These results block claiming present planner applicability.

Sol high found and reviewed a hybrid-evaluator queue bug: one mutable DP runner
was shared across vector environments. `scripts/eval_simdist.py` now gives each
hybrid environment its own runner and reset state. A 2-env/20-tick Isaac smoke
completed for both environments without queue-state failure (0/2 task success
is expected because it was cut off after 20 ticks). Hybrid MPPI now skips its
unused native policy forward when a DP proposal chunk is supplied, records the
actual sent action trace, and labels DP+preprocessing+planner latency scope.
Focused planner/audit/merge tests pass (35 tests). An attempted paired DP-action
identity check across separate Isaac resets differed by up to `.040 rad`;
do not treat that run as proving exact DP identity. The unit control check does
verify that one-candidate MPPI returns the exact constrained external proposal
when the input proposal is held fixed.

## Next bounded development experiment and gates

The full v7 archive (including successes and failures) is valid development
data for a new checkpoint, but neither v7 nor pooled v4–v7 may be reused as its
untouched test. The merged 52-episode development archive was built from the
original 36 episodes plus v7, preserving all labels, episode fingerprints and
seed metadata. Raw v7 pickle copies were removed only after the packed v7
archive and audit JSON were confirmed present. A new split assigns whole reset
seed groups to optimizer, validation and reserved calibration partitions;
the joint residual is calibrated first and score heads second, with calibration
rounds excluded from optimizer updates. Training windows remain uniform over
transitions, so longer episodes still contribute more; no success-only
oversampling is allowed because it changes the target behavior-policy mixture.

Before another planner comparison, the new candidate must pass both fresh
holdout datasets independently and pooled: all eight latent and joint
episode-bootstrap and reset-seed-clustered 95% upper bounds below 1; score MAE,
RMSE, signed bias and raw MSE relative to a training-only
remaining-time-conditioned mean baseline by horizon and outcome; exact zero at
the objective boundary; and predicted-vs-actual return ordering on bounded
simulator branches. The reward-only arm is evaluated before MC MPPI. The MC arm
requires a declared continuation-policy match. Planner runs must then compare
full 300-tick paired ResiP and existing image DP baselines with DR, success hold,
candidate-return ranking and per-action latency. Passing these simulator gates
would make a controlled real evaluation *plausible* after operator coordination;
it would still not establish real robustness on its own.

Long training and robot actuation remain out of scope until this bounded
candidate and score gate is reviewed. No real robot controller, push, merge, or
long GPU training was run in this follow-up.

### New 52-episode candidate and H1 test (2026-10-08)

`aligned_dev_52ep_v1.npz` contains 12,809 transitions. The reset-seed split for
seed `20261008` is train 36 episodes/9 seeds, validation 8/2 seeds, and
calibration 8/2 seeds. The calibration split contains three stable successes;
validation contains none. A fresh 2,000-step DP-vision/joint-conditioned model
was trained from scratch, then joint and score calibration were fit on the
reserved calibration group (1,647 windows), in that order. Checkpoint:
`/tmp/so101-simdist-residual-20261008/train_dpvision_joint_conditioned_52ep_v1_calibrated_scores.pt`.
The validation loss was `0.3343`; that scalar is not an acceptance gate.

A new H1 holdout used seeds 1800–1803, 16 perturbed diverse episodes, 4,007
transitions and five stable successes. It is now a candidate-selection set,
not a final test. The frozen checkpoint passed the episode-bootstrap persistence
gate at all eight latent horizons (95% upper ratios `.983, .942, .882, .866,
.795, .782, .810, .789`) and the current-joint prediction gate at all eight
horizons. H1 success-only and early-truncation subgroup bootstraps had too few
episodes for a separate claim.

The H1 score audit used a training-window baseline conditioned on prediction
horizon and six remaining-time bins. Reward predictions were inconclusive
overall; for the two early-truncated episodes, reward MSE was 2.14–3.61 times
the baseline across horizons. Value predictions were inconclusive overall
(MAE `76.59`, RMSE `115.48`) and underpredicted successful episodes by about
`250` return units. This is not an adequate score gate.

Full 300-tick paired H1 simulator evaluations confirm that score-guided MPPI
does not improve the DP proposal. At radius `.25 rad`, image DP succeeded on
3/4 seeds and DP-proposal reward MPPI on 0/4. Restricting the trust radius to
`.05 rad` still yielded DP 4/4 and MPPI 0/4. Mean hybrid latency was `41.7 ms`
with p95 `131.5 ms`, above the 30 Hz budget. A single-seed one-candidate
passthrough control succeeded for both image DP and MPPI (1/1 each), which
validates controller wiring when no score-guided refinement is selected; paired
action traces differed slightly across simulator rerenders and are not an exact
action-identity proof. The evaluator now records sent actions and per-plan
candidate score gain. A 20-tick smoke showed positive learned score gain for
every step, despite the full-run failures; this is direct evidence that the
current scorer selects overoptimistic deviations.

Artifacts:

- Forecast audit: `/tmp/so101-simdist-residual-20261008/dpvision_joint_holdout_h1_bootstrap.json`
- Score baseline audit: `/tmp/so101-simdist-residual-20261008/score_baseline_h1.json`
- External value/reward calibration: `/tmp/so101-simdist-residual-20261008/score_audit_52ep_h1.json`
- Full paired runs: `paired_eval_h1_dp_vs_mppi_reward_4seeds.json`,
  `paired_eval_h1_dp_vs_mppi_reward_tight_trust_4seeds.json`, and
  `paired_eval_h1_dp_vs_mppi_passthrough_seed1800.json` in the same directory.

The current model predicts q dynamics and freezes current camera content across
the future rollout. It can pass latent forecast tests while failing to represent
cube motion/contact that drives StackCube reward. This is the next architecture
blocker. Recommended direction after Sol high contract review: train a visual
object-state estimator from current camera images using simulator cube/gripper
pose labels, then an action-conditioned object-state transition model; keep those
privileged poses as training targets only. Validate pose and reward prediction
on seed-disjoint diverse data before returning to MPPI. Do not spend another
large planner sweep or claim SO101 applicability until this score/ranking gap
is closed. No real robot actuation, push, or merge occurred.
