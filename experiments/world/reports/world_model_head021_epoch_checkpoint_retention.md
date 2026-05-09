# World Model HEAD021: Epoch Checkpoint Retention

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Saving per-epoch checkpoints for the supervised fixed-delta run should enable
post-hoc composite/top-k selection without rerunning training for every scalar
selection metric.

Falsifier: the training script cannot retain epoch checkpoints cleanly or the
saved artifacts do not contain enough metadata to reload and audit individual
epochs.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/supervised_horizon_frame.py`
- `test_code/test_world_model_evaluation.py`

Added:

- `epoch_checkpoint_path`
- `save_epoch_checkpoint`
- `--epoch_checkpoint_dir`

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/supervised_horizon_frame.py`

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 192 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.075 --retrieval_temperature 0.1 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.002 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_mildhead_epochs_head021.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_mildhead_epochs_head021.pt \
  --epoch_checkpoint_dir models/world/checkpoints/part1_jepa_latent/head021_mildhead_epochs
```

## Result

Tests:

```text
17 passed in 0.72s
```

Checkpoint retention:

```text
models/world/checkpoints/part1_jepa_latent/head021_mildhead_epochs/epoch_001.pt
...
models/world/checkpoints/part1_jepa_latent/head021_mildhead_epochs/epoch_025.pt
```

Count:

```text
25
```

Result JSON includes:

```text
epoch_checkpoint_dir models/world/checkpoints/part1_jepa_latent/head021_mildhead_epochs
best epoch 7
best val MSE 0.021182
best val MRR 0.058656
best val top5 0.068750
```

## Mechanism Read

This does not change the model result; it changes the experiment surface. The
top-k/MSE/probe tradeoff can now be evaluated by loading specific epoch
checkpoints rather than rerunning the same training with different scalar
selection metrics.

The first useful post-hoc audit should compare epochs `2`, `3`, `7`, and
possibly `8`: early epochs carry top-k retrieval, while epoch `7` carries the
best MRR/MSE compromise.

## Decision

Continue JEPA-only.

Next step:

- audit saved HEAD021 epoch checkpoints post-hoc with the context probe and
  corrected top-k-aware composite score;
- choose the best epoch from actual saved states rather than from scalar
  selection proxies.

