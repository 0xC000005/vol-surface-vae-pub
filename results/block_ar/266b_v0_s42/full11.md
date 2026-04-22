# 220h Full 11-Suite Multi-Horizon Validation

- model: `266b`
- checkpoint: `models/backfill/266b_v0_s42/best_model.pt`
- windows: `192`
- samples per window: `48`
- suite score: `2/11`
- failed suites: `coverage, conditionality, time_series, cointegration, regime_coverage, distributional_fidelity, cross_cell_correlation, mean_reversion, pathwise_jump_realism`

**Horizon Summary**
- h1 cov90: `0.202`
- h30 cov90: `0.382`
- turb/calm ratio: `1.056`
- MR ratio h1: `1.889`
- MR ratio h30: `1.796`

**Additional v2 Suites**
- time-series ACF corr: `0.637`
- block boundary ratio: `0.973`
- cointegration gen/GT ratio: `0.371`
- regime coverage overall: `False`

**Fidelity / Structure**
- daily-change KS pass cells: `0/25`
- level KS pass cells: `0/25`
- corr ratio: `2.165`
- rank ratio: `0.229`
- max-jump KS: `0.979`
