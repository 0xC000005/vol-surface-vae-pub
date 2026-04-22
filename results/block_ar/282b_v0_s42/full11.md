# 220h Full 11-Suite Multi-Horizon Validation

- model: `282b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/282b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `5/11`
- failed suites: `coverage, conditionality, time_series, regime_coverage, distributional_fidelity, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.791`
- h30 cov90: `0.662`
- turb/calm ratio: `1.071`
- MR ratio h1: `1.052`
- MR ratio h30: `0.785`

**Additional v2 Suites**
- time-series ACF corr: `0.941`
- block boundary ratio: `0.998`
- cointegration gen/GT ratio: `0.913`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `5/25`
- level KS pass cells: `0/25`
- corr ratio: `1.039`
- rank ratio: `1.092`
- max-jump KS: `0.378`
