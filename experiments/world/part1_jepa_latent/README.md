# Part 1: JEPA Latent World Model

Purpose: learn a predictive latent state from past multivariate time-series
windows without hand-defining slow state variables.

Expected training form:

```text
past window -> context encoder -> latent state z_t
z_t + horizon tokens -> predictor -> predicted future latent
actual future window -> target/EMA encoder -> target future latent
```

Expected losses:

```text
L = L_future_latent_prediction + lambda_var L_variance + lambda_cov L_covariance
```

Quality gates for this part should be representation-focused:

- latent prediction error by horizon,
- retrieval top-k against candidate future windows,
- embedding variance and effective rank,
- off-diagonal covariance/correlation norm,
- frozen probes for future state summaries.

Do not align arbitrary time windows. The positive pair is the past context and
the actual future window from the same sample.
