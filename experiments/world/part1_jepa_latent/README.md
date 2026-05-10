# Part 1: JEPA Latent World Model

Purpose: learn a robust market-state representation from multivariate
time-series windows without hand-defining slow state variables.

Current default training form:

```text
same market window / same relative time position
-> structured masked view A + observed/synthetic mask channels
-> shared encoder -> z_a
same market window / same relative time position
-> structured masked view B + observed/synthetic mask channels
-> shared encoder -> z_b
```

Expected losses:

```text
L =
    L_masked_multiview_invariance(z_a, z_b)
  + lambda_redundancy L_barlow_or_vicreg
  + lambda_var L_variance
  + lambda_cov L_covariance
```

Quality gates for this part should be representation-focused:

- same-state masked-view alignment,
- same-state retrieval top-k or MRR,
- Barlow/cross-correlation diagonal and off-diagonal terms,
- embedding variance and effective rank,
- off-diagonal covariance/correlation norm,
- mask-artifact diagnostics,
- frozen probes for future/range/state summaries after pretraining.

Do not align arbitrary time windows. The positive pair for the current branch is
two structured masked views of the same window at the same relative index. EMA
targets and predictor heads belong to a separately gated context-to-target JEPA
experiment, not the default masked-multiview objective.

## Current Smoke

- `jepa_smoke.py`: minimal IV-only GRU context encoder, EMA target encoder, and
  predictor. Use it as a Part 1 falsifier, not as the final architecture.
- `masked_multiview_barlow_smoke.py`: current direct two-view reference path:
  one shared encoder, two structured masked views, Barlow-style redundancy
  reduction applied directly to the evaluated embeddings.
