# World Model HEAD073: Probe Interpretation

Date: 2026-05-09

Iteration type: `post_experiment_analysis`

## Question

Does HEAD072 imply the HEAD070 representation is weak, or does it show
complementary information relative to raw surface features?

## Evidence

The frozen probe split by target:

| feature | mean-delta MSE | mean-delta R2 | range MSE | range R2 |
| --- | ---: | ---: | ---: | ---: |
| mean target baseline | 0.013939 | -0.003432 | 0.063851 | -3.640187 |
| HEAD070 clean last latent | 0.011635 | 0.162420 | 0.047185 | -2.428997 |
| raw surface last | 0.006484 | 0.533258 | 0.054625 | -2.969679 |
| raw surface flat | 0.009746 | 0.298396 | 0.050972 | -2.704215 |
| raw geometry last | 0.013460 | 0.031052 | 0.064093 | -3.657776 |

Interpretation:

- raw surface last-day features dominate `future_mean_delta`;
- HEAD070 clean last latents dominate `future_range`;
- full geometry features are not automatically better and can overfit;
- HEAD070 latents carry market-state information, but not the same information
  as the raw surface level.

This is compatible with the pretraining objective: same-state masked-view
invariance should not be expected to preserve every low-level feature needed
for a simple autoregressive delta probe.

## Decision

Do not change pretraining yet.

Before adding a projection head, geometry-aware encoder, or any extra
regularizer, test feature complementarity in the frozen probe:

- `raw_surface_last + HEAD070 clean last latent`;
- `raw_surface_flat + HEAD070 clean last latent`;
- optionally `raw_geometry_last + HEAD070 clean last latent`.

If the combined feature improves both mean-delta and range probes, HEAD070 is a
useful state summary even if raw surface features remain necessary. If it does
not improve raw probes, then the representation is mostly redundant for
downstream forecasting despite strong same-state retrieval.

## Next Step

Update the probe audit feature set with a small number of combined features.
Keep the same checkpoint, ridge alpha, targets, and train/validation windows.

Falsifier: combined features fail to improve over the best raw or best latent
feature on either target.

## Artifacts

- `results/world/masked_multiview_barlow_probe_head072.json`
- `experiments/world/reports/world_model_head072_frozen_probe_audit.md`
- `experiments/world/reports/world_model_head073_probe_interpretation.md`
