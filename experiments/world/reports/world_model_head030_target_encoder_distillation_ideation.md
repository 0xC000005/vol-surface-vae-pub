# World Model HEAD030: Target Encoder Distillation Ideation

Date: 2026-05-09

## Iteration Type

`research_ideation`

## Question

After HEAD028 showed that fixed delta-PCA targets are learnable and HEAD029
separated target-space from frame-space metrics, what is the least ad hoc way
to reintroduce a learned target encoder without adding a new JEPA objective or
more tuning knobs?

## Literature Status

`supported_adjacent`, not `canonical_jepa`.

The canonical JEPA direction remains context-to-target latent prediction with
careful target construction, horizon/masking design, EMA or stop-gradient target
encoders, and simple latent prediction losses. A separate target-encoder
distillation step is adjacent because it uses a fixed PCA target as a teacher
contract before returning to JEPA prediction.

Primary-source support checked online:

- I-JEPA frames the core model as predicting target-block representations from
  a context block and treats target/context construction as central:
  https://arxiv.org/abs/2301.08243
- A-JEPA extends JEPA to audio time-frequency inputs and uses an EMA target
  encoder for target regions:
  https://arxiv.org/abs/2311.15830
- BYOL supports the broader teacher/student pattern with an online network
  predicting a slowly updated target network:
  https://arxiv.org/abs/2006.07733
- SimSiam supports the broader stop-gradient stabilization pattern without
  negatives or a momentum encoder:
  https://arxiv.org/abs/2011.10566
- DINO supports self-distillation without labels and highlights momentum
  encoders plus k-NN quality as representation evidence:
  https://arxiv.org/abs/2104.14294

This does not justify adding neighborhood, contrastive, or decoder objectives.
It justifies one bounded teacher-contract diagnostic for the target encoder.

## Local Evidence

HEAD025 showed that the canonical delta-frame EMA target encoder remained
low-rank:

```text
learned-target MRR    0.042356
pred effective rank   1.262683
target effective rank 2.077708
```

HEAD028 showed that fixed whitened PCA horizon-delta targets are useful:

```text
fixed-target MRR      0.096374
decoded delta MSE     0.015369
pred effective rank   3.863753
target effective rank 3.327211
```

The PCA oracle residual on the same fixed target contract is much smaller:

```text
val inverse-PCA residual MSE 0.001846
```

That creates a clean target-encoder question: can a learned encoder imitate the
fixed contract before we ask a context predictor to chase it?

## Modeling Choice

Run one target-encoder distillation diagnostic:

```text
future horizon delta block
-> learned target encoder
-> z_target

teacher target:
z_pca = fixed whitened PCA_8(delta_h)

loss:
MSE(z_target, stopgrad(z_pca))
```

Keep the contract fixed:

- same data builder and train/validation slice as HEAD028;
- same horizons `(1, 5, 10, 20, 30)`;
- same target dimension `8`;
- same fixed PCA fit on train deltas;
- no context predictor training in this diagnostic;
- no retrieval/neighborhood loss;
- no decoder work.

## Hypothesis

A small learned target encoder can faithfully imitate the fixed delta-PCA
contract when it is trained directly on future horizon deltas. If true, the
target encoder collapse in HEAD025 was not inevitable; it was a coupled JEPA
training failure where the target space was never anchored.

## Falsifier

The distillation diagnostic fails if the learned target encoder:

- does not reduce fixed-target MSE well below HEAD028's context-predictor
  fixed-target MSE `0.987130`;
- decodes to deltas much closer to HEAD028's context-predictor MSE `0.015369`
  than to the PCA oracle residual `0.001846`;
- collapses toward the HEAD025 rank pattern instead of matching the fixed PCA
  target health.

These are comparative gates against existing artifacts, not new selection
weights.

## Planned HEAD031 Experiment

Add a single script:

```text
experiments/world/part1_jepa_latent/target_encoder_distill.py
```

The script should reuse:

- `make_horizon_delta_matrix`
- `fit_delta_pca_target`
- `transform_delta_targets`
- `inverse_transform_delta_targets`
- `latent_prediction_metrics`
- `retrieval_metrics`
- `representation_health_metrics`

Expected outputs:

- target-to-PCA prediction metrics;
- target-space retrieval of `z_target` against `z_pca`;
- decoded delta MSE from inverse PCA;
- target representation health;
- JSON result under `results/world/`;
- checkpoint under `models/world/checkpoints/part1_jepa_latent/`.

Only after this diagnostic passes should a later HEAD train the JEPA context
predictor against a frozen learned target encoder.

## Decision

Proceed to one experiment, HEAD031, focused only on target-encoder
distillation. Do not add target-dimension sweeps, retrieval losses,
neighborhood losses, variance/covariance weights, or decoder components.
