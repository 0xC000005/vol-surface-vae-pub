# 220h Full 11-Suite Multi-Horizon Validation

- model: `267b`
- checkpoint: `models/backfill/267b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.016`
- h30 cov90: `0.032`
- turb/calm ratio: `1.799`
- MR ratio h1: `2.055`
- MR ratio h30: `0.920`

**Additional v2 Suites**
- time-series ACF corr: `0.830`
- block boundary ratio: `0.746`
- cointegration gen/GT ratio: `0.197`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `2/25`
- corr ratio: `0.441`
- rank ratio: `1.050`
- max-jump KS: `1.000`
