# World Model HEAD002: Smoke Dataset And Metric Harness

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

If HEAD001 correctly identified the local data object, then a small
manifest-aligned harness can build IV 30/30 windows and score placeholder Part 1
and Part 2 outputs without training a model.

Falsifier: the harness cannot reproduce train/validation windows on toy data, or
it fails to load real IV validation windows and emit finite metric values.

## Execution

Added:

- `experiments/world/evaluation/world_data.py`
- `experiments/world/evaluation/part1_metrics.py`
- `experiments/world/evaluation/part2_metrics.py`
- `test_code/test_world_model_evaluation.py`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- Real-data smoke using `build_iv_world_windows(split="train", max_windows=64)`
  and `build_iv_world_windows(split="val", max_windows=32)`.

## Result

Tests:

```text
3 passed in 0.07s
```

Real-data smoke shapes:

```text
train_past   (64, 30, 25)
train_future (64, 30, 25)
train_regime (64,)
val_past     (32, 30, 25)
val_future   (32, 30, 25)
val_start    [4010, 4011, 4012] ... [4039, 4040, 4041]
```

Placeholder Part 1 smoke:

```text
prediction mse       0.0128497499
prediction rmse      0.1133567375
prediction cosine    0.9844046505
retrieval top1       0.03125
retrieval top5       0.15625
retrieval mrr        0.1301671340
effective_rank       1.1998501347
offdiag_abs_mean     0.9517925019
```

These numbers use a trivial baseline: predicted latent is the last past IV
frame and target latent is the mean future IV frame. The low rank and high
off-diagonal correlation are useful sanity evidence that the representation
diagnostics can expose collapse/redundancy.

Placeholder Part 2 smoke:

```text
coverage_90             0.85
variance_ratio          0.0023870728
pairwise_distance_mean  0.3860081559
corr_frobenius          9.7649126247
effective_rank          15.3774377321
gt_effective_rank       4.1501109891
pc1_alignment           0.9376696641
sample_std_mean         0.0079548478
```

These numbers use tiny Gaussian jitter around truth. They verify metric shape
and finite-output behavior only; they are not generator quality evidence.

## Mechanism Read

The harness now gives the world-model loop a concrete execution substrate:

- Part 1 can be trained and scored without involving the decoder.
- Part 2 can be smoke-scored on direct `(B, S, 30, 5, 5)` scenario samples.
- The same manifest-aligned split used by current-panel artifacts is available
  for world-model work.

The immediate next falsifier is no longer data plumbing. It is whether a minimal
JEPA latent model can beat the trivial last-frame/mean-future placeholder on
prediction/retrieval while keeping variance/effective-rank healthy.

## Decision

Proceed to a Part 1 model smoke next:

- build a small context encoder, target encoder, and horizon predictor under
  `experiments/world/part1_jepa_latent/`;
- train on IV-only windows for a short CPU/GPU-friendly run;
- report prediction, retrieval, variance, effective-rank, and off-diagonal
  covariance/correlation metrics;
- do not attach the flow decoder until Part 1 has a non-collapsed baseline.

Primary failure class for this iteration: none. The harness passed its intended
data and metric smoke checks.
