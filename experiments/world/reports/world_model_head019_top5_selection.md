# World Model HEAD019: Top5 Checkpoint Selection

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Selecting the mild-head/light-correlation run by validation top5 should recover
broader retrieval ranking while preserving the frame-MSE persistence gate.

Falsifier: top5 selection improves broad retrieval only by losing too much frame
MSE, context health, or frozen-probe quality under the top-k-aware composite
score.

## Execution

No code changes.

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
  --selection_metric top5 \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_mildhead_top5_head019.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_mildhead_top5_head019.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_mildhead_top5_head019.pt \
  --output_json results/world/part1_context_probe_audit_head019.json
```

Composite check:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD017_mildhead_mrr results/world/part1_supervised_horizon_delta_corrreg_mildhead_head017.json results/world/part1_context_probe_audit_head017.json \
  --entry HEAD019_mildhead_top5 results/world/part1_supervised_horizon_delta_corrreg_mildhead_top5_head019.json results/world/part1_context_probe_audit_head019.json \
  --output_json results/world/part1_context_composite_scores_head019.json
```

## Result

Top5-selected checkpoint:

```text
epoch 3
frame MSE      0.022145
frame MRR mean 0.056790
frame top1     0.009375
frame top5     0.090625
frame top10    0.139844
```

Raw persistence baseline:

```text
frame MSE      0.022376
frame MRR mean 0.052426
frame top1     0.000781
frame top5     0.086719
frame top10    0.135156
```

Context audit:

```text
effective rank   4.748494
offdiag abs mean 0.368346
ridge MSE        0.017327
ridge MRR mean   0.104012
top1 mean        0.042969
top5 mean        0.141406
top10 mean       0.221875
```

Top-k-aware composite score:

```text
HEAD013_corr_0p002      0.580304
HEAD017_mildhead_mrr    0.577865
HEAD019_mildhead_top5   0.520371
```

## Mechanism Read

Top5 selection works narrowly: it beats persistence on frame top5 and top10
while still barely beating persistence on frame MSE. But it is too early and
undertrained relative to the balanced candidates. Frame-MSE improvement,
context rank, and frozen-probe metrics all fall enough that the composite score
drops well below HEAD013 and HEAD017.

This means top-k selection alone is not the right repair. The model needs a
training or selection objective that preserves the later-epoch MSE/probe gains
while avoiding top5/top10 collapse.

## Decision

Continue JEPA-only.

Next step:

- do not use top5-only checkpoint selection as the main gate;
- consider a composite checkpoint selector only if per-epoch checkpoints are
  saved, or test a softer top-k/ranking objective that does not force such an
  early checkpoint.

