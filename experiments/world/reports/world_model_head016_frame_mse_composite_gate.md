# World Model HEAD016: Frame-MSE Composite Gate

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Hypothesis

Adding frame-MSE improvement over raw persistence to the composite Part 1 score
should prevent persistence-failing checkpoints from outranking balanced
rank/probe candidates.

Falsifier: HEAD015 still ranks above HEAD013 after the frame-MSE correction.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/score_context_runs.py`
- `test_code/test_world_model_evaluation.py`

Added composite component:

```text
frame_mse_improvement = (persistence_mse - frame_mse) / persistence_mse
```

The term is allowed to be negative, so checkpoints worse than persistence are
penalized directly.

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/score_context_runs.py`

Run:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD010_unregularized results/world/part1_supervised_horizon_delta_contrastive_mrr_head009.json results/world/part1_context_probe_audit_head010.json \
  --entry HEAD011_covreg results/world/part1_supervised_horizon_delta_contextreg_head011.json results/world/part1_context_probe_audit_head011.json \
  --entry HEAD012_corr_0p005 results/world/part1_supervised_horizon_delta_corrreg_head012.json results/world/part1_context_probe_audit_head012.json \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD015_headsharp results/world/part1_supervised_horizon_delta_corrreg_headsharp_head015.json results/world/part1_context_probe_audit_head015.json \
  --output_json results/world/part1_context_composite_scores_head016.json
```

## Result

Tests:

```text
15 passed in 0.72s
```

Corrected ranking:

```text
HEAD013_corr_0p002    0.581476
HEAD012_corr_0p005    0.558285
HEAD010_unregularized 0.526537
HEAD011_covreg        0.495492
HEAD015_headsharp     0.468361
```

HEAD015 now has a negative frame-MSE improvement:

```text
frame_mse_improvement -0.069337
```

HEAD013 remains the best current candidate:

```text
frame_mse_improvement 0.047056
frame MRR             0.058137
rank fraction         0.165133
decorrelation         0.656539
ridge MRR             0.107289
ridge MSE improvement 0.244584
```

## Mechanism Read

The corrected score fixes the HEAD015 counterexample. Strong frozen-probe
metrics no longer hide failure to beat the raw persistence forecast. The ranking
now matches the intended Part 1 contract: a candidate must preserve basic
future-frame predictive value while also improving context health and frozen
probe quality.

HEAD013 remains the current best fixed-delta context candidate. The next model
change should target retrieval sharpness without using contrastive settings that
damage frame MSE.

## Decision

Continue JEPA-only with HEAD013 as the current candidate.

Next step:

- test a milder head-side retrieval adjustment that keeps the corrected
  composite gate, or add a composite-aware checkpoint selection mode before
  running more variants.

