# World Model HEAD020: Softer Head Retrieval Pressure

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Reducing the mild-head retrieval weight from `0.075` to `0.06` should preserve
the later-epoch MSE/probe gains while reducing broad top-k retrieval loss.

Falsifier: the run again selects an early checkpoint whose top-k gains come at
the cost of frame MSE, context health, and frozen-probe quality.

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
  --retrieval_weight 0.06 --retrieval_temperature 0.1 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.002 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_softhead_head020.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_softhead_head020.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_softhead_head020.pt \
  --output_json results/world/part1_context_probe_audit_head020.json
```

Composite check:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD019_top5 results/world/part1_supervised_horizon_delta_corrreg_mildhead_top5_head019.json results/world/part1_context_probe_audit_head019.json \
  --entry HEAD020_softhead results/world/part1_supervised_horizon_delta_corrreg_softhead_head020.json results/world/part1_context_probe_audit_head020.json \
  --output_json results/world/part1_context_composite_scores_head020.json
```

## Result

MRR-selected checkpoint:

```text
epoch 2
frame MSE      0.022327
frame MRR mean 0.058461
frame top1     0.012500
frame top5     0.087500
frame top10    0.138281
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
effective rank   4.495739
offdiag abs mean 0.373337
ridge MSE        0.017281
ridge MRR mean   0.098365
top1 mean        0.039063
top5 mean        0.128906
top10 mean       0.202344
```

Top-k-aware composite score:

```text
HEAD013_corr_0p002 0.580304
HEAD019_top5       0.520371
HEAD020_softhead   0.501753
```

## Mechanism Read

This is another negative top-k repair. Softer retrieval pressure produces a
checkpoint with positive top5/top10 deltas and a tiny frame-MSE edge, but the
selected checkpoint is even earlier than HEAD019 and weaker on context rank,
ridge MRR, and overall composite score.

The evidence now suggests that top-k gains appear early before the context
representation and frozen probes mature, then decay as the supervised head
improves MSE/probe fit. The next useful step is to save per-epoch checkpoints or
add post-hoc epoch-level composite selection, not continue scalar retrieval
weight sweeps.

## Decision

Continue JEPA-only.

Next step:

- add per-epoch checkpoint retention or top-k/MSE/probe epoch diagnostics so the
  tradeoff can be selected post-hoc rather than through a single scalar
  validation metric.

