# 220h Full 11-Suite Multi-Horizon Validation

- model: `266a`
- checkpoint: `models/backfill/266a_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.824`
- h30 cov90: `0.888`
- turb/calm ratio: `0.911`
- MR ratio h1: `2.111`
- MR ratio h30: `0.872`

**Additional v2 Suites**
- time-series ACF corr: `0.786`
- block boundary ratio: `0.493`
- cointegration gen/GT ratio: `0.174`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `1.871`
- rank ratio: `0.346`
- max-jump KS: `1.000`
