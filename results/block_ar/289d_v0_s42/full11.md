# 220h Full 11-Suite Multi-Horizon Validation

- model: `289d`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/289d_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.781`
- MR ratio h30: `0.700`

**Additional v2 Suites**
- time-series ACF corr: `0.544`
- block boundary ratio: `0.621`
- cointegration gen/GT ratio: `1.192`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `0.212`
- rank ratio: `2.168`
- max-jump KS: `1.000`
