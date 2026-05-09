# World Model HEAD013: Light Context Correlation Regularization

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

A lighter normalized context-correlation penalty should preserve most of the
HEAD012 rank/off-diagonal improvement while reducing the frozen-probe
degradation.

Falsifier: effective rank falls back near the unregularized checkpoint, or the
frozen ridge probe remains worse than HEAD012 and HEAD010.

## Execution

No code changes. Reused the HEAD012 supervised fixed-delta script with a lower
correlation penalty.

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.05 --retrieval_temperature 0.1 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.002 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_light_head013.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_light_head013.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_light_head013.pt \
  --output_json results/world/part1_context_probe_audit_head013.json
```

## Result

MRR-selected checkpoint:

```text
epoch 4
frame MSE      0.021323
frame MRR mean 0.058137
frame top1     0.011719
frame top5     0.078906
frame top10    0.140625
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Context health:

```text
variance min       0.002655
variance mean      0.011004
effective rank     5.284253
participation      3.450378
offdiag abs mean   0.343461
offdiag abs max    0.922245
```

Frozen ridge probe:

```text
MSE        0.016903
MRR mean   0.107289
top1 mean  0.045313
top5 mean  0.139844
top10 mean 0.220313
```

Comparison:

```text
HEAD010 unregularized: rank 3.688405, offdiag 0.484346, ridge MSE 0.017107, ridge MRR 0.111537
HEAD012 corr 0.005:    rank 5.732591, offdiag 0.332259, ridge MSE 0.018033, ridge MRR 0.106710
HEAD013 corr 0.002:    rank 5.284253, offdiag 0.343461, ridge MSE 0.016903, ridge MRR 0.107289
```

## Mechanism Read

The lighter correlation penalty is the best rank/probe compromise so far. It
keeps most of the HEAD012 rank improvement and greatly lowers off-diagonal
correlation relative to HEAD010, while recovering and slightly improving frozen
ridge MSE versus the unregularized checkpoint.

The remaining weakness is retrieval sharpness. Ridge MRR is still below HEAD010,
and frame top5 is below persistence even though frame MSE, MRR, top1, and top10
beat persistence. This suggests the context representation is healthier and
linearly predictive, but the selected checkpoint is not yet optimal for
discriminative nearest-neighbor ranking.

## Decision

Continue JEPA-only.

Next step:

- add a composite checkpoint/evaluation score that can trade off frame MRR,
  context effective rank, off-diagonal correlation, and frozen ridge probe MRR;
- use it to choose between the currently observed regimes instead of selecting
  only by frame-space MRR.

