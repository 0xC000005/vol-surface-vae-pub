# World Model HEAD072: Frozen Probe Audit

Date: 2026-05-09

Iteration type: `experiment`

## Purpose

Evaluate whether the HEAD070 masked-multiview embeddings carry useful market
state information in a downstream probe. Prediction remains a downstream probe,
not a pretraining objective.

## Implementation

Added `experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py`.

The audit:

- loads `masked_multiview_barlow_head070.pt`;
- encodes clean histories with synthetic masks set to visible;
- uses the last clean latent and mean clean latent as frozen features;
- builds future-summary targets from IV surfaces:
  `future_mean_delta` and `future_range`;
- compares ridge probes against raw surface and full-geometry baselines;
- reports MSE, MAE, RMSE, R2, and representation health.

The default ridge alpha is `10.0`, chosen after a small sensitivity check showed
that `1e-2` overfit raw high-dimensional baselines and made the audit
uninterpretable.

## Validation

Focused test:

```bash
pytest test_code/test_world_model_evaluation.py::test_masked_multiview_probe_targets_and_regression_metrics -q
```

Result: `1 passed in 0.74s`.

Audit command:

```bash
python experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py --device cpu
```

Saved result: `results/world/masked_multiview_barlow_probe_head072.json`.

## Probe Results

Validation metrics with ridge alpha `10.0`:

| feature | future_mean_delta MSE | future_mean_delta R2 | future_range MSE | future_range R2 |
| --- | ---: | ---: | ---: | ---: |
| mean target baseline | 0.013939 | -0.003432 | 0.063851 | -3.640187 |
| HEAD070 clean last latent | 0.011635 | 0.162420 | 0.047185 | -2.428997 |
| HEAD070 clean mean latent | 0.017193 | -0.237643 | 0.055849 | -3.058684 |
| raw surface last | 0.006484 | 0.533258 | 0.054625 | -2.969679 |
| raw surface flat | 0.009746 | 0.298396 | 0.050972 | -2.704215 |
| raw geometry last | 0.013460 | 0.031052 | 0.064093 | -3.657776 |
| raw geometry flat | 0.022308 | -0.605903 | 0.275230 | -19.001497 |

## Interpretation

HEAD070 embeddings are not empty mask-invariance features:

- the clean last latent beats the mean target baseline on both targets;
- it is the best tested feature for `future_range`;
- it has positive R2 for `future_mean_delta`.

But the probe also shows a real limitation:

- raw surface last-day features are much better for `future_mean_delta`;
- raw surface features remain competitive for `future_range`;
- full-geometry flat baselines overfit unless strongly regularized.

The current representation is therefore useful, but not yet a dominant
forecasting state representation. That is acceptable at this stage because
forecasting was not the pretraining objective, but it gives a concrete Part 1
probe bar for future work.

## Decision / Next Step

Do not move to the flow decoder yet. Next iteration should be
`post_experiment_analysis`: decide whether the best next Part 1 step is a
geometry-aware encoder improvement, a representation/projection split, or a
more stable downstream probe protocol.

## Artifacts

- `experiments/world/part1_jepa_latent/masked_multiview_barlow_probe_audit.py`
- `test_code/test_world_model_evaluation.py`
- `experiments/world/evaluation/README.md`
- `experiments/world/reports/world_model_head072_frozen_probe_audit.md`
- `results/world/masked_multiview_barlow_probe_head072.json`
