# World Model HEAD004: Part 1 JEPA Failure Analysis

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Question

Why did HEAD003 improve prediction MSE/cosine but fail retrieval and latent
health?

## Evidence

Saved result inspected:

- `results/world/part1_jepa_smoke_head003.json` (ignored local artifact)

Training dynamics:

```text
prediction loss: 0.225814 -> 0.009857
variance loss:   0.926827 -> 0.936821
covariance loss: 0.000339 -> 0.000237
```

The prediction term improved quickly, but the variance term stayed near `0.94`.
With `gamma=1`, that means latent dimensions are nowhere near the desired
per-dimension spread. The low covariance term is misleading because low-variance
latents can have small raw covariance even when they are not useful.

Validation singular spectra:

```text
context sv top8:   [0.266439, 0.104785, 0.064589, 0.040944, 0.032047, 0.025859, 0.019570, 0.014285]
target sv top8:    [0.293004, 0.129401, 0.061867, 0.045103, 0.036680, 0.022872, 0.019783, 0.016285]
predicted sv top8: [0.033989, 0.014010, 0.008472, 0.005470, 0.004990, 0.003255, 0.001732, 0.001472]
raw target top8:   [0.230028, 0.146984, 0.074585, 0.054375, 0.041143, 0.020678, 0.016468, 0.012186]
```

The predicted latent is much more compressed than the target/context latents.
Retrieval is therefore expected to fail: the predictor learns a smooth central
future representation rather than a discriminative future state.

Retrieval evidence:

```text
JEPA top1  = 0.00390625
JEPA top5  = 0.01953125
JEPA top10 = 0.04296875
baseline top1  = 0.00390625
baseline top5  = 0.0390625
baseline top10 = 0.06640625
```

For 256 validation candidates, chance top1 is `1/256 = 0.00390625`, chance top5
is `5/256 = 0.01953125`, and chance top10 is `10/256 = 0.0390625`. HEAD003 is
therefore at chance, not merely below a high bar.

## Mechanism

The HEAD003 objective lets the model satisfy the easiest part of the task:
match the average direction of the EMA target latent. It does not require
sample identity to survive in the predicted latent.

The failed mechanism is:

```text
past -> context -> predictor -> low-variance central future latent
```

not:

```text
past -> context -> predictor -> discriminative actual future latent
```

This is a `latent_prediction` failure with a real `collapse` risk. It is not a
decoder problem and should not be hidden by moving to Part 2.

## Decision

The next experiment should directly repair the Part 1 objective:

1. Add an in-batch retrieval / InfoNCE auxiliary loss between predicted and
   target latents.
2. Increase variance pressure enough that the variance term can visibly move.
3. Track retrieval against chance in the training report.
4. Keep the flow decoder detached.

The next falsifier is:

> With retrieval loss and stronger variance pressure, validation retrieval must
> beat chance and effective rank must increase; otherwise the current
> GRU/EMA architecture is not sufficient for the Part 1 claim.
