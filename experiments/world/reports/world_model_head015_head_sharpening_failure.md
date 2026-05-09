# World Model HEAD015: Head Sharpening Failure

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

Keeping the HEAD013 context objective but increasing predictor capacity and
contrastive pressure should recover retrieval sharpness without sacrificing the
fixed-delta forecast edge.

Falsifier: the run improves probe retrieval only by losing frame forecast
quality or context health.

## Execution

No code changes.

Run:

```text
python experiments/world/part1_jepa_latent/supervised_horizon_frame.py \
  --device cpu --epochs 25 --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --hidden_dim 64 --context_dim 32 \
  --predictor_hidden_dim 256 \
  --target_mode delta --frame_weight 0.25 \
  --retrieval_weight 0.1 --retrieval_temperature 0.07 \
  --context_variance_weight 0.05 \
  --context_covariance_weight 0.0 \
  --context_correlation_weight 0.002 \
  --context_variance_gamma 0.1 \
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_headsharp_head015.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_headsharp_head015.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_headsharp_head015.pt \
  --output_json results/world/part1_context_probe_audit_head015.json
```

Composite check:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD015_headsharp results/world/part1_supervised_horizon_delta_corrreg_headsharp_head015.json results/world/part1_context_probe_audit_head015.json \
  --output_json results/world/part1_context_composite_scores_head015.json
```

## Result

MRR-selected checkpoint:

```text
epoch 4
frame MSE      0.023928
frame MRR mean 0.057297
frame top1     0.012500
frame top5     0.081250
frame top10    0.134375
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
effective rank   4.832314
offdiag abs mean 0.385486
ridge MSE        0.016111
ridge MRR mean   0.111276
top1 mean        0.048438
top5 mean        0.146875
top10 mean       0.222656
```

Composite score:

```text
HEAD015_headsharp 0.537697
HEAD013_corr_0p002 0.534420
```

## Mechanism Read

This is a useful failure. The stronger head-side contrastive setting improves
the frozen ridge probe and nearly recovers HEAD010 ridge MRR, but it loses the
actual frame forecast edge: frame MSE is worse than raw persistence, and top5
and top10 also trail persistence.

The composite scorer from HEAD014 incorrectly ranks HEAD015 above HEAD013
because it does not include a frame-MSE improvement term or a hard persistence
gate. That makes HEAD015 a diagnostic counterexample: probe strength alone is
not enough if the saved checkpoint no longer beats a raw persistence forecast.

## Decision

Continue JEPA-only.

Next step:

- patch the composite score to include frame-MSE improvement over persistence
  and/or a hard penalty when frame MSE is worse than persistence;
- re-score HEAD010 through HEAD015 before running more model variants.

