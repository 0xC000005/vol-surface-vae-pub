# Evaluation Split

Keep the two claims separate.

## Part 1 Metrics

These test whether the latent world model learned useful future-predictive
state:

- future-latent prediction error,
- future-latent retrieval top-k and MRR,
- embedding variance,
- effective rank,
- off-diagonal covariance/correlation norm,
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
