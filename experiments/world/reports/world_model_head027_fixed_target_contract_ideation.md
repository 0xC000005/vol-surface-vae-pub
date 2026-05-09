# World Model HEAD027: Fixed Target Contract Ideation

Date: 2026-05-09

## Iteration Type

`research_ideation`

## Literature Status

`supported_adjacent`, not `canonical_jepa`.

The proposed fixed target embedding contract is not the canonical I-JEPA/V-JEPA
recipe because canonical JEPA uses a learned target encoder, often updated by
EMA or stop-gradient. Here the fixed target is a diagnostic stabilization step:
prove that a target space exists and is worth predicting before asking a learned
target encoder to discover it.

Supporting paper families:

- I-JEPA and V-JEPA support the overall context-to-target latent prediction
  framing and make target construction central:
  https://arxiv.org/abs/2301.08243 and https://arxiv.org/abs/2404.08471
- VICReg and Barlow Twins support the importance of variance, decorrelation,
  and redundancy reduction for avoiding collapsed representations:
  https://arxiv.org/abs/2105.04906 and https://arxiv.org/abs/2103.03230
- VJ-VCR supports variance/covariance regularization inside a video JEPA
  setting: https://arxiv.org/abs/2412.10925

Classification:

```text
fixed PCA/whitened delta target = supported_adjacent diagnostic
not a final JEPA objective
not a decoder path
not a new loss patch
```

## Problem

HEAD025 showed the EMA target encoder collapses even after switching from
absolute future frames to future deltas:

```text
selected target rank    about 1.96 to 2.16 by horizon
selected predicted rank about 1.23 by horizon
selected MRR            0.042356
raw persistence MRR     0.052426
```

The target encoder is now the bottleneck. Adding another predictor-side loss
would not answer whether the future-delta target space itself is usable.

## Candidate Contract

Fit a fixed linear target embedding on training horizon-delta frames:

```text
delta_h = future[:, h - 1, :] - past[:, -1, :]
z_h = whitened PCA_k(delta_h)
```

Then train:

```text
past window -> context encoder
context + horizon token -> predicted z_h
loss = MSE(predicted z_h, fixed z_h)
```

Evaluation remains unchanged in spirit:

- prediction MSE/cosine in fixed target space;
- retrieval MRR/top-k in fixed target space;
- representation health of context and predicted embeddings;
- optional linear decode from predicted z_h back to delta frames for a frame-MSE
  sanity check.

## Guardrails

- Do not add a new retrieval/neighborhood loss.
- Do not sweep target dimensions. Choose one small dimension, `k=8`, because the
  current learned target rank is around two and the supervised delta predictor
  uses 25 raw dimensions.
- Do not touch the decoder.
- Treat this as a diagnostic lower bound for target-space stability. If it
  fails, the issue is likely weak signal/data/encoder capacity rather than EMA
  target drift.

## Falsifier

The fixed target contract is not useful if:

- predicted fixed-target retrieval does not beat HEAD025's learned-target MRR
  of `0.042356`;
- predicted effective rank remains near `1`;
- a simple decode of predicted fixed targets back to delta frames cannot match
  the supervised lower-bound direction.

## Decision

Proceed with exactly one fixed-target diagnostic experiment:

```text
target_coordinate = fixed_delta_pca
target_dim = 8
loss = fixed-target MSE only, plus existing representation-health metrics
```

This is a target-space diagnostic, not a promoted replacement for JEPA. If it
works, the next learned-target attempt should imitate this stable target space.
