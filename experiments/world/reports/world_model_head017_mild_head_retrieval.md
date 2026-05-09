# World Model HEAD017: Mild Head Retrieval Adjustment

Date: 2026-05-09

## Iteration Type

`experiment`

## Hypothesis

A milder head-side retrieval adjustment should recover some retrieval sharpness
from the HEAD013-style healthier context while preserving the corrected
frame-MSE persistence gate.

Falsifier: the run either fails to beat HEAD013 under the corrected composite
score or repeats HEAD015's frame-MSE persistence failure.

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
  --selection_metric mrr \
  --output_json results/world/part1_supervised_horizon_delta_corrreg_mildhead_head017.json \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_mildhead_head017.pt
```

Audit:

```text
python experiments/world/part1_jepa_latent/context_probe_audit.py \
  --device cpu --batch_size 128 \
  --max_train_windows 2048 --max_val_windows 256 \
  --ridge_alpha 0.001 \
  --checkpoint models/world/checkpoints/part1_jepa_latent/supervised_horizon_delta_corrreg_mildhead_head017.pt \
  --output_json results/world/part1_context_probe_audit_head017.json
```

Composite check:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD017_mildhead results/world/part1_supervised_horizon_delta_corrreg_mildhead_head017.json results/world/part1_context_probe_audit_head017.json \
  --output_json results/world/part1_context_composite_scores_head017.json
```

## Result

MRR-selected checkpoint:

```text
epoch 7
frame MSE      0.021182
frame MRR mean 0.058656
frame top1     0.018750
frame top5     0.068750
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
effective rank   5.323299
offdiag abs mean 0.352284
ridge MSE        0.017022
ridge MRR mean   0.110486
top1 mean        0.051563
top5 mean        0.137500
top10 mean       0.233594
```

Corrected composite score:

```text
HEAD017_mildhead 0.587240
HEAD013_corr_0p002 0.581476
```

## Mechanism Read

HEAD017 is the new best corrected-composite candidate. It keeps the frame-MSE
edge over persistence, improves frame MRR and top1 over HEAD013, keeps context
rank in the improved range, and recovers most of the frozen ridge MRR without
the HEAD015 forecast failure.

The weakness is still broader retrieval ranking. Frame top5 remains below
persistence, and frame top10 is slightly below persistence. The model is
becoming sharper for the correct nearest neighbor/top1 case, but not yet for
the broader top-k neighborhood.

## Decision

Continue JEPA-only with HEAD017 as the current fixed-delta context candidate.

Next step:

- add or tune a top-k-aware selection/evaluation criterion so the candidate
  cannot improve top1/MRR while losing top5/top10 against persistence;
- avoid increasing contrastive pressure as aggressively as HEAD015.

