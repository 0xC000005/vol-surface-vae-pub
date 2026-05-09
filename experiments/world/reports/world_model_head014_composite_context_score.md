# World Model HEAD014: Composite Context Score

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Hypothesis

A simple composite Part 1 score can make the rank/probe/retrieval tradeoff
explicit and identify the best current context checkpoint candidate among
HEAD010 through HEAD013.

Falsifier: the score is unable to distinguish the observed regimes or selects a
checkpoint that clearly violates the Part 1 gates.

## Execution

Added:

- `experiments/world/part1_jepa_latent/score_context_runs.py`

Updated:

- `test_code/test_world_model_evaluation.py`

Composite components:

- frame-space MRR from the supervised training result;
- context effective-rank fraction;
- context decorrelation term `1 - offdiag_abs_mean`;
- frozen ridge-probe MRR;
- frozen ridge-probe MSE improvement over the zero-delta baseline.

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
  --output_json results/world/part1_context_composite_scores_head014.json
```

## Result

Tests:

```text
14 passed in 0.70s
```

Composite weights:

```text
frame_mrr             1.0
rank_fraction         0.5
decorrelation         0.25
ridge_mrr             1.0
ridge_mse_improvement 0.5
```

Ranking:

```text
HEAD013_corr_0p002    0.534420
HEAD012_corr_0p005    0.519010
HEAD010_unregularized 0.476795
HEAD011_covreg        0.451840
```

Best current candidate:

```text
HEAD013_corr_0p002
frame MRR             0.058137
rank fraction         0.165133
decorrelation         0.656539
ridge MRR             0.107289
ridge MSE improvement 0.244584
```

## Mechanism Read

The composite score agrees with the qualitative read: HEAD013 is the best
current Part 1 compromise. It does not have the sharpest ridge MRR, but it has
substantially healthier context structure than HEAD010 and better ridge MSE
than both HEAD010 and HEAD012.

HEAD011 remains a useful negative control: covariance-magnitude regularization
is dominated by the normalized correlation trials. HEAD012 is healthier than
HEAD013 on rank/decorrelation, but it pays too much probe-MSE cost.

## Decision

Continue JEPA-only with HEAD013 as the current fixed-delta context candidate.

Next step:

- use the composite scorer as a gate while testing whether the predictor/head
  can recover retrieval sharpness from the healthier context;
- try a head-side retrieval/temperature or predictor-capacity adjustment before
  changing the encoder objective again.

