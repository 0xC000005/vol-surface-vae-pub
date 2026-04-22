# 220h Full 11-Suite Multi-Horizon Validation

- model: `266d`
- checkpoint: `models/backfill/266d_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `1/11`
- failed suites: `surface, coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.001`
- h30 cov90: `0.001`
- turb/calm ratio: `0.925`
- MR ratio h1: `1.959`
- MR ratio h30: `0.812`

**Additional v2 Suites**
- time-series ACF corr: `0.803`
- block boundary ratio: `0.687`
- cointegration gen/GT ratio: `0.705`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `1.557`
- rank ratio: `0.477`
- max-jump KS: `1.000`
