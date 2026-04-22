# 220h Full 11-Suite Multi-Horizon Validation

- model: `270b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/270b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.974`
- h30 cov90: `0.859`
- turb/calm ratio: `0.996`
- MR ratio h1: `1.005`
- MR ratio h30: `-0.464`

**Additional v2 Suites**
- time-series ACF corr: `0.923`
- block boundary ratio: `1.056`
- cointegration gen/GT ratio: `0.892`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `1/25`
- level KS pass cells: `0/25`
- corr ratio: `1.175`
- rank ratio: `0.497`
- max-jump KS: `1.000`
