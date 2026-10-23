# SO101 SimDist dynamics failure: root-cause probes (2026-10-08)

## Scope and verdict

This is a read-only investigation of the frozen 64px checkpoint
`/tmp/so101-simdist-study/train_v2_64/worldmodel.pt`. It does not retrain,
modify the checkpoint, launch Isaac evaluation, or command the robot. The
episode-bootstrap holdout gate remains failed. The strongest confirmed
contributor is a **fixed output-geometry mismatch**: the dynamics decoder's
final affine LayerNorm restricts predictions to a set that cannot reach the
observed target latents closely enough to beat time-aware persistence on
holdout horizons 1–5. This explains a material floor, not all of the learned
model's error.

The model also underuses actions/history in these probes. Online target drift
and shared-encoder loss interference are possible from the training code, but
cannot be confirmed from this single final checkpoint. Joint/action normalization
shift exists but is modest relative to the latent prediction failure. None of
these findings establishes whether a corrected model will improve real
robustness.

## Reproduction

The bounded probe samples four complete windows per episode at local fractions
0.2, 0.4, 0.6, 0.8: 64 train, 16 original validation, and 128 independent
holdout windows. CPU inference used batch size 16, seed 42, and the checkpoint's
training normalizer. Probe arrays are summarized at
`/tmp/so101-simdist-holdout/root_cause_probes.json`.

```bash
python -m py_compile scripts/diagnose_simdist_dynamics.py
conda run --no-capture-output -n so101 python scripts/diagnose_simdist_dynamics.py \
  --checkpoint /tmp/so101-simdist-study/train_v2_64/worldmodel.pt \
  --training-data /tmp/so101-simdist-study/aligned.npz \
  --holdout-data /tmp/so101-simdist-holdout/aligned.npz \
  --out /tmp/so101-simdist-holdout/root_cause_probes.json \
  --device cpu --batch-size 16 --windows-per-episode 4 --seed 42
```

This is a bounded diagnostic sample, not a replacement for the full
episode-bootstrap acceptance audit in
[`simdist-finite-horizon-validation-20261007.md`](simdist-finite-horizon-validation-20261007.md).

## Confirmed: final decoder LayerNorm cannot represent target geometry well

`src/so101/learning/simdist/modules.py` ends each decoder layer with
`layernorm_3`; `model.py` consumes its output as the latent prediction without
an unconstrained output projection. The target from `encoder.py` is an
unrestricted visual/joint/time fusion MLP output plus tag embedding. For each
target row `z`, final LayerNorm output is `gamma * u + beta`, where
`mean(u)=0` and `std(u)<=1` (population std across latent channels; epsilon
prevents assuming exact unit variance). Let `q=(z-beta)/gamma`. With
`gmin=min(abs(gamma))`, every output row has squared error per channel at least

```text
gmin^2 * (mean(q)^2 + max(std(q) - 1, 0)^2)
```

This is a conservative lower bound for the **fixed checkpoint's final affine
LayerNorm output set**, not a bound on alternative architectures or on the
full reachable training dynamics. On the exact sampled windows:

| Split | Windows | Mean conservative lower bound | Learned MSE h1 / h8 | Persistence MSE h1 / h8 |
|---|---:|---:|---:|---:|
| Train | 64 | 0.08906 | 0.29109 / 0.30277 | 0.01120 / 0.06933 |
| Original validation | 16 | 0.01026 | 0.20429 / 0.20217 | 0.00335 / 0.07058 |
| Independent holdout | 128 | 0.04831 | 0.23208 / 0.26812 | 0.01031 / 0.06776 |

Holdout horizon lower bounds are
`[.04706, .04701, .04818, .04893, .04675, .04923, .04874, .05058]`;
corresponding persistence errors are
`[.01031, .01533, .02322, .03367, .04557, .05163, .05885, .06776]`.
Thus even an oracle within this fixed output set cannot beat persistence on
horizons 1–5 for these samples. This is direct evidence that output geometry
contributes to failure. It does not account for optimization, conditioning,
encoder quality, or target drift.

Holdout target row standard deviation averaged 0.71292 (between-row std
0.38415), versus prediction 0.81141 (between-row std 0.03858). Target row-mean
std was 0.11081, versus prediction 0.00321. This matches the structural
constraint: prediction row means are almost fixed by `beta`, while target row
means vary. The minimum absolute gamma is 0.696958.

### Post-hoc projection probe: geometry is not the whole failure

To check whether adding output freedom alone would be enough, one shared affine
map was fitted by least squares after the frozen decoder output using only the
same 64 train windows (512 horizon rows). The decoder and checkpoint were not
updated. This intentionally small probe is not a trained architecture and the
train score is in-sample. On holdout, h1/h8 MSE moved from
0.23208/0.26812 to 0.22387/0.24029, while persistence was
0.01031/0.06776. The projection-only model remained worse than persistence at
all eight horizons; its projected/persistence ratios ranged 21.72 (h1) to
3.55 (h8). Original validation also remained worse than persistence at every
horizon. Thus extra output freedom can modestly lower latent error but does not
make this checkpoint a useful predictive model. Other issues—latent features,
conditioning, optimization and/or target drift—remain consequential. Results
are included in the same `root_cause_probes.json` artifact under
`train_probe_affine_projection_*` fields.

## Other hypotheses and discriminating evidence

### Train/seed distribution shift is not the primary cause — supported

Train and original validation fail on the same sampled probe, as does the
independent holdout. The checkpoint is already poor on training episodes, so
new-seed shift alone cannot explain the gap. The full held-out audit also
reports learned/persistence ratios above one with episode-bootstrap intervals
above one at every horizon.

### Action conditioning exists, but is weak — supported, causal value unknown

On holdout, h1/h8 MSE is 0.23208/0.26812 with logged actions,
0.25993/0.30308 with batch-reversed future actions, and
0.23787/0.26605 when repeating the final sent action. The action branch changes
predictions, and correct logged actions help relative to shuffled actions, but
they do not produce accurate next-latent forecasts. These counterfactual
probes lack the actual outcomes under altered actions; they are sensitivity
checks, not causal action-effect validation.

### History/visual information is underused in this probe — suggestive

Repeating the current history for all past frames gives 0.23203/0.26431 at
h1/h8, slightly better than the original history. This suggests weak use of
history, but does not prove hidden-state aliasing or identify which camera
branch is responsible. Existing modality blackout probes only show that image
branches affect representation, not that they carry predictive information.

### Normalization/time alignment — not the leading explanation

The dataset audit already checks successor continuity, remaining-tick
decrement, tick duration, gamma, and stored RGB dimensions. Holdout
standardized joint/action standard deviations range about 0.875–1.271; the
largest per-joint fraction beyond |z|>3 is 1.8%. Means shift by up to about
0.48 standard deviations. This merits tracking but is not commensurate with
the large prediction gap. Adding 30 ticks to the current clock changes h1/h8
MSE only from 0.23208/0.26812 to 0.23136/0.27082.

### Online target drift / shared-encoder gradient conflict — unresolved

`losses.py` computes targets under `no_grad`, but this does not freeze the
target encoder. The same optimizer updates encoder parameters from dynamics,
reward/value, and policy losses; there is no frozen or EMA target encoder.
This is a credible training instability, but the one saved final checkpoint
cannot establish historical target drift or gradient interference. The saved
loss curve decreases and is insufficient evidence of stable/meaningful latent
prediction. To distinguish this hypothesis, a future short instrumentation
run must save periodic snapshots and, on a fixed probe set, record target
latent drift plus per-loss encoder gradient norms/cosines. Treat this as a
separate ablation from output geometry.

## Recommended next experiment (not run here)

Do not continue training this checkpoint for MPPI. A new, separately named
experimental variant may add an unconstrained projection after decoder
LayerNorm or specify target/prediction normalization, but the post-hoc probe
shows that output freedom alone is insufficient. The next model experiment
must include a trainable sequence model capable of using visual/history/action
signals, then rerun train/validation/holdout probes and the episode-bootstrap
gate. Keep the current architecture as a frozen comparison. In a separate
ablation, stabilize targets (EMA/frozen target encoder) and record drift/gradient
diagnostics; do not combine both changes in the first run, so their effects
remain identifiable. Require the corrected model to beat time-aware persistence
with episode-bootstrap confidence before spending a long GPU run or repeating
planner evaluations.

Even if that gate passes, compare all planner arms on matched simulation seeds
before making a real robustness claim. Real evaluation remains operator-run;
this investigation did not actuate the robot.
