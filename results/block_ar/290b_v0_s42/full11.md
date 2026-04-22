# 220h Full 11-Suite Multi-Horizon Validation

- model: `290b`
- checkpoint: `/home/max/Documents/vol-surface-vae-pub/models/backfill/290a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.000`
- h30 cov90: `0.000`
- turb/calm ratio: `1.000`
- MR ratio h1: `0.418`
- MR ratio h30: `0.679`

**Additional v2 Suites**
- time-series ACF corr: `0.402`
- block boundary ratio: `0.789`
- cointegration gen/GT ratio: `0.202`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `12/25`
- corr ratio: `1.293`
- rank ratio: `0.498`
- max-jump KS: `1.000`
