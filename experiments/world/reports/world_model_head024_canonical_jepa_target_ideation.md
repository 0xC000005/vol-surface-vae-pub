# World Model HEAD024: Canonical JEPA Target Ideation

Date: 2026-05-09

## Iteration Type

`research_ideation`

## Literature Status

`canonical_jepa` for the proposed next architecture change:

- I-JEPA defines the basic pattern as predicting target-block representations
  from a context block and makes target/context construction central to
  representation quality: https://arxiv.org/abs/2301.08243
- V-JEPA treats feature prediction as a stand-alone objective without negatives,
  reconstruction, or extra supervision: https://arxiv.org/abs/2404.08471

`supported_adjacent` for the existing variance/covariance health controls:

- VJ-VCR applies variance-covariance regularization in a video JEPA setting to
  avoid representation collapse: https://arxiv.org/abs/2412.10925

`not selected for immediate implementation`:

- Var-JEPA gives a more principled probabilistic formulation and argues for
  explicit latent uncertainty without ad-hoc anti-collapse regularizers, but it
  is a larger paradigm shift than the next small HEAD step:
  https://arxiv.org/abs/2603.20111

## Hypothesis

The recent failure is not that JEPA needs another ranking or neighborhood loss.
The stronger hypothesis is that the target coordinate is wrong: absolute future
frames make the target encoder chase a persistence-dominated object, while the
supervised lower bound showed future deltas are learnable and discriminative.

Falsifier for the next experiment: a canonical EMA target-encoder JEPA trained
on horizon-delta targets still produces low-rank predicted latents and fails to
beat the raw persistence baseline on retrieval/probe gates.

## Local Evidence

HEAD006 tested horizon-specific learned-target JEPA with EMA and trainable
target encoders. It remained low-rank and lost badly to raw persistence:

```text
prefix-target EMA MRR 0.034284, predicted rank 2.207024
frame-target EMA  MRR 0.033023, predicted rank 1.715300
raw persistence   MRR 0.052426, top5 0.086719, top10 0.135156
```

HEAD007 showed a supervised horizon-delta lower bound contains useful signal:

```text
best frame MSE 0.015180 vs persistence 0.022376
delta retrieval clearly above chance across horizons
```

HEAD013 remains the top-k-aware fixed-delta reference:

```text
composite score 0.580304
frame MSE 0.021323
frame MRR 0.058137
effective rank 5.284253
```

HEAD023 showed the soft-neighborhood objective is not the right main direction:

```text
composite score 0.532110
frame_mse_improvement 0.005222
ridge MRR 0.111229
```

## Decision

Do not add another retrieval, top-k, or neighborhood loss.

The next HEAD should implement one canonical target-construction experiment:

```text
past window -> context encoder
future horizon delta block -> EMA target encoder
context + horizon token -> predictor
loss = latent prediction + existing variance/covariance health terms
retrieval/probe metrics remain evaluation-only
```

The only new modeling contract should be the target coordinate:

```text
absolute: target = future block
delta:    target = future block - last observed past frame
```

For the first run, use `delta` only. Do not sweep the coordinate, do not add
neighborhood losses, and do not change the decoder.

## Next Step

Implement the minimal horizon-JEPA delta-target coordinate path in
`experiments/world/part1_jepa_latent/horizon_jepa_smoke.py`, add one focused
test, and run a single CPU smoke experiment against the same Part 1 gates.
