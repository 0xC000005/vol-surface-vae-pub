# Evaluation Split

Keep the two claims separate.

## Modules

- `world_data.py`: manifest-aligned IV 30/30 window builder.
- `masked_multiview_data.py`: geometry-aware masked multiview window builder
  with separate observed and synthetic mask channels.
- `masked_multiview_metrics.py`: same-state alignment, retrieval,
  Barlow-style cross-correlation, representation health, and mask-visibility
  summaries for masked multiview batches.
- `part1_metrics.py`: latent prediction, retrieval, and representation-health
  diagnostics.
- `part2_metrics.py`: compact path-sample diagnostics for decoder smokes.

Part 1 smoke scripts live under `experiments/world/part1_jepa_latent/`.
`masked_multiview_jepa_smoke.py` is the first geometry-aware masked-multiview
encoder smoke. It uses value plus observed/synthetic mask channels and reports
the diagnostics from this folder; it does not use future prediction as a
pretraining target.
`masked_multiview_barlow_smoke.py` is the cleaner direct two-view Barlow
baseline: one shared encoder, two structured masked views, and redundancy
reduction applied directly to the evaluated embeddings. Its default loss uses
canonical mean-scaled Barlow off-diagonal weighting.

## Part 1 Metrics

These test whether Part 1 learned a robust market-state representation before
scenario generation:

- same-state masked-view alignment,
- same-state retrieval top-k and MRR,
- Barlow/cross-correlation diagonal and off-diagonal checks,
- embedding variance,
- effective rank,
- off-diagonal covariance/correlation norm,
- geometry-stratified mask diagnostics,
- frozen probes for future summaries and downstream tasks.

## Part 2 Metrics

These test whether the flow decoder produces calibrated, diverse, realistic
conditional scenarios:

- CRPS / pinball / coverage,
- Energy Score / Variogram Score,
- sample diversity and variance ratio,
- correlation matrix error,
- PCA/eigenvalue spectrum error,
- tail and drawdown metrics,
- conditionality by context bucket.

The reporting rule is: Part 1 success does not imply Part 2 success, and Part 2
success does not prove the world-model representation is non-collapsed.
