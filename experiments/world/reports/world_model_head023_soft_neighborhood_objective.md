# World Model HEAD023: Soft Neighborhood Objective

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Adding a soft neighborhood contrastive term in horizon-delta space should
preserve broad top-k retrieval neighborhoods better than one-hot retrieval
alone while retaining the frame-MSE and frozen-probe gains from later training.

Falsifier: the soft-neighborhood run improves neighborhood/top-k metrics only
by selecting an early underfit checkpoint or by losing the frame-MSE/probe gate.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
- `test_code/test_world_model_evaluation.py`

Added:

- `soft_neighborhood_contrastive_loss`
- `--neighborhood_weight`
- `--neighborhood_temperature`
- `--neighborhood_target_temperature`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/supervised_horizon_frame.py test_code/test_world_model_evaluation.py`

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 192 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.05 --retrieval_temperature 0.1 \
  --neighborhood_weight 0.05 \
  --neighborhood_temperature 0.1 \
  --neighborhood_target_temperature 0.2 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.002 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_softneighborhood_head023.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_softneighborhood_head023.pt \
  --epoch_checkpoint_dir models/world/checkpoints/part1_jepa_latent/head023_softneighborhood_epochs
```

The selected checkpoint was audited with `context_probe_audit.py` and scored
with `score_context_runs.py` against HEAD013, HEAD017, and HEAD019.

## Result

Tests:

```text
18 passed in 0.72s
```

Best selected epoch:

```text
epoch 5
frame MSE 0.022259
frame MRR 0.057032
top1 0.010938
top5 0.082813
top10 0.139844
```

Raw persistence baseline:

```text
frame MSE 0.022376
MRR 0.052426
top1 0.000781
top5 0.086719
top10 0.135156
```

Context/probe audit:

```text
effective rank 5.060637
offdiag abs mean 0.361862
ridge MSE 0.017022
ridge MRR 0.111229
ridge top5 0.142188
ridge top10 0.226563
```

Top-k-aware composite ranking:

```text
HEAD013_corr_0p002       0.580304
HEAD017_mildhead         0.577865
HEAD023_softneighborhood 0.532110
HEAD019_top5             0.520371
```

HEAD023 components:

```text
frame_mse_improvement 0.005222
frame MRR             0.057032
frame_top5_delta     -0.003906
frame_top10_delta     0.004687
rank fraction         0.158145
decorrelation         0.638138
ridge MRR             0.111229
ridge MSE improvement 0.239259
```

## Mechanism Read

The soft-neighborhood auxiliary objective did not fix the main Part 1
bottleneck. It preserved slightly better broad retrieval than HEAD017 on
top5/top10 deltas and produced the best ridge MRR among the scored comparison
set, but the selected checkpoint was still early and gave up almost all of the
frame-MSE improvement that made HEAD013/HEAD017 viable.

This falsifies the tested setting, not the general idea of neighborhood-aware
objectives. The failure class remains `latent_prediction`: neighborhood ranking
and frame-space prediction are still trading off rather than co-maturing.

## Decision

Stop after this task per user request.

Do not promote HEAD023 as the current Part 1 candidate. Keep HEAD013 as the
top-k-aware reference and HEAD017 as the frame-MSE/ridge-MRR near miss. If this
workflow is resumed, the next JEPA-only move should not be another scalar
neighborhood-weight sweep; it should change the target/training schedule so
prediction fit and neighborhood retrieval mature together.
