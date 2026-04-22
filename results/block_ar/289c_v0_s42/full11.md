# 220h Full 11-Suite Multi-Horizon Validation

- model: `289c`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/289c_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `3/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.912`
- MR ratio h30: `0.807`

**Additional v2 Suites**
- time-series ACF corr: `0.634`
- block boundary ratio: `0.643`
- cointegration gen/GT ratio: `0.292`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `5/25`
- corr ratio: `0.596`
- rank ratio: `1.356`
- max-jump KS: `1.000`
