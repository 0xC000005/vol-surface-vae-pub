# World Model HEAD018: Top-K Composite Gate

Date: 2026-05-09

## Iteration Type

`post_experiment_analysis`

## Hypothesis

Adding frame top5/top10 deltas against raw persistence to the composite Part 1
score should prevent a candidate from improving top1/MRR while quietly losing
broader retrieval ranking.

Falsifier: HEAD017 still ranks above HEAD013 despite worse top5/top10 deltas.

## Execution

Updated:

- `experiments/world/part1_jepa_latent/score_context_runs.py`
- `test_code/test_world_model_evaluation.py`

Added composite components:

```text
frame_top5_delta = frame_top5 - persistence_top5
frame_top10_delta = frame_top10 - persistence_top10
```

Both terms use weight `0.5` and may be negative.

Validation commands:

- `pytest test_code/test_world_model_evaluation.py -q`
- `python -m py_compile experiments/world/part1_jepa_latent/score_context_runs.py`

Run:

```text
python experiments/world/part1_jepa_latent/score_context_runs.py \
  --entry HEAD010_unregularized results/world/part1_supervised_horizon_delta_contrastive_mrr_head009.json results/world/part1_context_probe_audit_head010.json \
  --entry HEAD012_corr_0p005 results/world/part1_supervised_horizon_delta_corrreg_head012.json results/world/part1_context_probe_audit_head012.json \
  --entry HEAD013_corr_0p002 results/world/part1_supervised_horizon_delta_corrreg_light_head013.json results/world/part1_context_probe_audit_head013.json \
  --entry HEAD015_headsharp results/world/part1_supervised_horizon_delta_corrreg_headsharp_head015.json results/world/part1_context_probe_audit_head015.json \
  --entry HEAD017_mildhead results/world/part1_supervised_horizon_delta_corrreg_mildhead_head017.json results/world/part1_context_probe_audit_head017.json \
  --output_json results/world/part1_context_composite_scores_head018.json
```

## Result

Tests:

```text
16 passed in 0.71s
```

Top-k-aware ranking:

```text
HEAD013_corr_0p002    0.580304
HEAD017_mildhead      0.577865
HEAD012_corr_0p005    0.559066
HEAD010_unregularized 0.527709
HEAD015_headsharp     0.465236
```

HEAD013 top-k deltas:

```text
frame_top5_delta  -0.007813
frame_top10_delta  0.005469
```

HEAD017 top-k deltas:

```text
frame_top5_delta  -0.017969
frame_top10_delta -0.000781
```

HEAD012 is the only listed correlation run with positive top5 and top10 deltas:

```text
frame_top5_delta  0.000781
frame_top10_delta 0.000781
```

## Mechanism Read

The top-k gate changes the decision. HEAD017 remains a useful model variant
because it improves frame MSE, frame MRR/top1, rank, and ridge MRR, but it loses
too much broad retrieval ranking. HEAD013 is again the current best candidate
under the stricter Part 1 score.

HEAD012 is informative: it is top-k healthier but lower on ridge MSE
improvement. That suggests the next useful move is not simply more contrastive
pressure; it is either a top-k-aware checkpoint selector or a softer retrieval
objective that improves neighborhood ranking without sacrificing probe fit.

## Decision

Continue JEPA-only with HEAD013 as the current top-k-aware fixed-delta context
candidate.

Next step:

- test checkpoint selection by `top5` on the light-correlation/mild-head family,
  or add an evaluation-only composite selector over saved epoch histories if
  per-epoch checkpoints become available.

